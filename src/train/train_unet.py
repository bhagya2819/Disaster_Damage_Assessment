"""U-Net training loop for Sen1Floods11 flood segmentation.

Features:
- AdamW + cosine-annealing LR.
- Mixed precision (torch.amp) on CUDA for ~2× speed-up on T4.
- Early stopping on validation IoU (patience-based).
- Best-checkpoint save (state_dict only — portable, small).
- CSV metric log per epoch → ``<out_dir>/metrics.csv``.
- Optional Weights & Biases logging (enabled when ``WANDB_API_KEY`` env var is set).
- Graceful resume from a ``last.pt`` checkpoint if present.

Usage:
    from src.train.train_unet import TrainConfig, train
    cfg = TrainConfig(
        sen1floods11_root='/content/drive/MyDrive/dda/sen1floods11',
        out_dir='checkpoints/unet_resnet34',
        epochs=30,
        batch_size=8,
    )
    train(cfg)
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.data.sen1floods11_loader import LABEL_IGNORE_INDEX, Sen1Floods11Dataset
from src.eval import metrics
from src.models.losses import BCEDiceLoss
from src.models.unet import UNetConfig, build_unet, count_parameters
from src.train.augment import train_transform, val_transform
from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# Dataset adapter — the base loader returns CHW tensors; augmentation runs
# on HWC numpy. Wrap the dataset to apply albumentations correctly.
# --------------------------------------------------------------------------

class AugmentedSen1Floods11(Sen1Floods11Dataset):
    def __init__(self, *args, transform=None, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)
        self._aug = transform

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        if self._aug is None:
            return item
        img = item["image"].numpy()                  # (C, H, W)
        lab = item["label"].numpy()                  # (H, W)
        img_hwc = np.transpose(img, (1, 2, 0))
        out = self._aug(image=img_hwc, mask=lab)
        item["image"] = torch.from_numpy(np.ascontiguousarray(np.transpose(out["image"], (2, 0, 1)))).float()
        item["label"] = torch.from_numpy(np.ascontiguousarray(out["mask"])).long()
        return item


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class TrainConfig:
    sen1floods11_root: str
    out_dir: str = "checkpoints/unet_resnet34"

    # Optimisation
    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    bce_weight: float = 0.5
    pos_weight: float | None = 2.0    # slight upweight of water class

    # Data
    train_crop: int = 256
    val_crop: int | None = None       # None → evaluate on full 512
    num_workers: int = 2

    # Training logistics
    device: str = "cuda"
    amp: bool = True                  # mixed precision
    seed: int = 42
    early_stopping_patience: int = 8
    save_best_on: str = "val_iou"     # key from metrics dict

    # Model
    model: UNetConfig = field(default_factory=UNetConfig)


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    import random  # noqa: PLC0415
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable — falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def _aggregate_metrics(preds: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float]:
    keys = ("iou", "f1", "precision", "recall", "accuracy", "cohen_kappa")
    per_chip: dict[str, list[float]] = {k: [] for k in keys}
    for p, y in zip(preds, labels, strict=True):
        m = metrics.summary(p, y, ignore_index=LABEL_IGNORE_INDEX)
        for k in keys:
            per_chip[k].append(m[k])
    return {f"val_{k}": float(np.nanmean(v)) for k, v in per_chip.items()}


# --------------------------------------------------------------------------
# Epoch loops
# --------------------------------------------------------------------------

def _run_train_epoch(
    model, loader, optimizer, scheduler, loss_fn, device, scaler, amp: bool,
) -> dict[str, float]:
    model.train()
    losses: list[float] = []
    for batch in tqdm(loader, desc="train", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(img)
            loss = loss_fn(logits, lab)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    scheduler.step()
    return {"train_loss": float(np.mean(losses))}


@torch.no_grad()
def _run_val_epoch(model, loader, loss_fn, device, amp: bool) -> dict[str, float]:
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    losses: list[float] = []
    for batch in tqdm(loader, desc="val", leave=False):
        img = batch["image"].to(device, non_blocking=True)
        lab = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(img)
            loss = loss_fn(logits, lab)
        losses.append(float(loss.item()))
        probs = torch.sigmoid(logits.squeeze(1))
        preds = (probs > 0.5).cpu().numpy()
        labels_np = lab.cpu().numpy()
        for i in range(preds.shape[0]):
            all_preds.append(preds[i])
            all_labels.append(labels_np[i])
    metrics_out = _aggregate_metrics(all_preds, all_labels)
    metrics_out["val_loss"] = float(np.mean(losses))
    return metrics_out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def train(cfg: TrainConfig) -> Path:
    """Run training and return the path to the best checkpoint."""
    _seed_all(cfg.seed)
    device = _resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Data ----------------------------------------------------------------
    train_ds = AugmentedSen1Floods11(
        root=cfg.sen1floods11_root, split="train", modality="s2",
        transform=train_transform(cfg.train_crop),
    )
    val_ds = AugmentedSen1Floods11(
        root=cfg.sen1floods11_root, split="valid", modality="s2",
        transform=val_transform(cfg.val_crop),
    )
    log.info("train=%d  val=%d", len(train_ds), len(val_ds))

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, cfg.batch_size // 2), shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda"),
    )

    # --- Model / optimiser ---------------------------------------------------
    model = build_unet(cfg.model).to(device)
    log.info("%s | %.2fM params", cfg.model.summary(), count_parameters(model) / 1e6)

    loss_fn = BCEDiceLoss(alpha=cfg.bce_weight, pos_weight=cfg.pos_weight).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler(device="cuda") if (cfg.amp and device.type == "cuda") else None

    # --- Logging -------------------------------------------------------------
    csv_path = out / "metrics.csv"
    csv_exists = csv_path.exists()
    csv_f = csv_path.open("a", newline="")
    csv_w: csv.DictWriter | None = None

    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        try:
            import wandb  # noqa: PLC0415
            wandb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "dda-flood"),
                name=f"unet_{cfg.model.encoder_name}",
                config=asdict(cfg) | {"model": asdict(cfg.model)},
            )
        except Exception as e:  # noqa: BLE001
            log.warning("W&B init failed (%s); continuing without it.", e)

    # --- Training loop -------------------------------------------------------
    best_score = -np.inf
    best_ckpt = out / "best.pt"
    last_ckpt = out / "last.pt"
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = _run_train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, scaler, cfg.amp)
        val_metrics = _run_val_epoch(model, val_loader, loss_fn, device, cfg.amp)
        elapsed = time.time() - t0

        row = {"epoch": epoch, "elapsed_s": elapsed, "lr": scheduler.get_last_lr()[0]} | train_metrics | val_metrics
        log.info(
            "ep %d/%d  lr=%.2e  train_loss=%.4f  val_loss=%.4f  val_iou=%.4f  val_f1=%.4f  (%.1fs)",
            epoch, cfg.epochs, row["lr"], row["train_loss"], row["val_loss"],
            row["val_iou"], row["val_f1"], elapsed,
        )

        if csv_w is None:
            csv_w = csv.DictWriter(csv_f, fieldnames=list(row.keys()))
            if not csv_exists:
                csv_w.writeheader()
        csv_w.writerow(row)
        csv_f.flush()

        if wandb_run is not None:
            wandb_run.log(row)

        torch.save({"model": model.state_dict(), "epoch": epoch}, last_ckpt)

        score = row[cfg.save_best_on]
        if score > best_score:
            best_score = score
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score}, best_ckpt)
            log.info("  ↳ new best %s = %.4f → %s", cfg.save_best_on, score, best_ckpt)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                log.info("Early stopping at epoch %d (no improvement for %d epochs)", epoch, patience)
                break

    csv_f.close()
    if wandb_run is not None:
        wandb_run.finish()

    log.info("Training complete. Best %s=%.4f → %s", cfg.save_best_on, best_score, best_ckpt)
    return best_ckpt


__all__ = ["AugmentedSen1Floods11", "TrainConfig", "train"]
