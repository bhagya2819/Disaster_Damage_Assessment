"""Final comparison table — Classical vs U-Net vs Hybrid on Sen1Floods11 test.

Emits the ``{classical, u-net, hybrid} x {IoU, F1, kappa, OA, runtime_ms}`` table
required by the course rubric's "Analysis & Results" criterion, with per-chip
IoU arrays saved for downstream significance testing.

Usage:
    python scripts/run_final_comparison.py \
        --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
        --checkpoint /content/drive/MyDrive/dda/checkpoints/unet_resnet34/best.pt \
        --out-dir results/final_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.sen1floods11_loader import LABEL_IGNORE_INDEX, Sen1Floods11Dataset  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.ablation import AblationConfig, predict as classical_predict  # noqa: E402
from src.eval.fusion import fuse_weighted  # noqa: E402
from src.eval.significance import mcnemar_test, paired_bootstrap_iou  # noqa: E402
from src.inference.predict import predict_chip  # noqa: E402
from src.models.unet import load_checkpoint  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

METRIC_KEYS = ("iou", "f1", "precision", "recall", "accuracy", "cohen_kappa")


def _aggregate(preds: list[np.ndarray], labels: list[np.ndarray]) -> dict[str, float]:
    per_chip: dict[str, list[float]] = {k: [] for k in METRIC_KEYS}
    for p, y in zip(preds, labels, strict=True):
        m = metrics.summary(p, y, ignore_index=LABEL_IGNORE_INDEX)
        for k in METRIC_KEYS:
            per_chip[k].append(m[k])
    return {k: float(np.nanmean(v)) for k, v in per_chip.items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sen1floods11-root", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to best U-Net checkpoint.")
    p.add_argument("--split", default="test", choices=["valid", "test", "bolivia"])
    p.add_argument("--out-dir", default="results/final_comparison")
    p.add_argument("--classical-index", default="ndwi")
    p.add_argument("--classical-threshold", default="yen")
    p.add_argument("--classical-morph", action="store_true", help="Apply morphology to classical mask.")
    p.add_argument("--hybrid-weight-unet", type=float, default=0.7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    per_chip_dir = out_dir / "per_chip"
    per_chip_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading U-Net: %s  (device=%s)", args.checkpoint, device)
    model = load_checkpoint(args.checkpoint, device=device)

    ds = Sen1Floods11Dataset(args.sen1floods11_root, split=args.split, modality="s2")
    n = len(ds)
    log.info("Sen1Floods11 split=%s n=%d", args.split, n)

    classical_cfg = AblationConfig(
        index=args.classical_index,
        threshold=args.classical_threshold,
        morphology=args.classical_morph,
    )
    log.info("Classical config: %s", classical_cfg.name)

    classical_preds: list[np.ndarray] = []
    unet_preds: list[np.ndarray] = []
    hybrid_preds: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    per_chip_iou = {"classical": np.zeros(n), "unet": np.zeros(n), "hybrid": np.zeros(n)}

    t_classical_ms = 0.0
    t_unet_ms = 0.0
    t_hybrid_ms = 0.0

    for i in tqdm(range(n), desc="eval"):
        s = ds[i]
        img = s["image"].numpy()
        y = s["label"].numpy()

        t0 = time.perf_counter()
        c_mask = classical_predict(img, classical_cfg)
        t_classical_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        u_probs = predict_chip(model, img, device=device)
        u_mask = u_probs >= 0.5
        t_unet_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        h_mask = fuse_weighted(
            u_probs,
            c_mask.astype(np.float32),
            weight_a=args.hybrid_weight_unet,
            threshold=0.5,
        )
        t_hybrid_ms += (time.perf_counter() - t0) * 1000

        classical_preds.append(c_mask)
        unet_preds.append(u_mask)
        hybrid_preds.append(h_mask)
        labels_all.append(y)
        per_chip_iou["classical"][i] = metrics.iou(c_mask, y, ignore_index=LABEL_IGNORE_INDEX)
        per_chip_iou["unet"][i] = metrics.iou(u_mask, y, ignore_index=LABEL_IGNORE_INDEX)
        per_chip_iou["hybrid"][i] = metrics.iou(h_mask, y, ignore_index=LABEL_IGNORE_INDEX)

    # Aggregate metrics.
    rows = {
        "classical": _aggregate(classical_preds, labels_all),
        "unet": _aggregate(unet_preds, labels_all),
        "hybrid": _aggregate(hybrid_preds, labels_all),
    }
    runtimes = {
        "classical": t_classical_ms / n,
        "unet": t_unet_ms / n,
        "hybrid": t_hybrid_ms / n,
    }

    log.info("\n=== Final comparison on split=%s (n=%d) ===", args.split, n)
    log.info("%-10s  %-6s  %-6s  %-6s  %-6s  %-6s  %-7s  %s",
             "method", "IoU", "F1", "prec", "rec", "acc", "κ", "runtime ms/chip")
    for name, m in rows.items():
        log.info(
            "%-10s  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %.2f",
            name, m["iou"], m["f1"], m["precision"], m["recall"], m["accuracy"], m["cohen_kappa"],
            runtimes[name],
        )

    # Pairwise bootstrap CIs relative to classical.
    boot_unet = paired_bootstrap_iou(per_chip_iou["unet"], per_chip_iou["classical"])
    boot_hyb = paired_bootstrap_iou(per_chip_iou["hybrid"], per_chip_iou["classical"])
    boot_uh = paired_bootstrap_iou(per_chip_iou["unet"], per_chip_iou["hybrid"])
    log.info("\nPaired bootstrap 95%% CIs (deltaIoU):")
    log.info("  unet     - classical : %+0.4f  [%+0.4f, %+0.4f]", boot_unet.mean_delta, boot_unet.ci_lower, boot_unet.ci_upper)
    log.info("  hybrid   - classical : %+0.4f  [%+0.4f, %+0.4f]", boot_hyb.mean_delta, boot_hyb.ci_lower, boot_hyb.ci_upper)
    log.info("  unet     - hybrid    : %+0.4f  [%+0.4f, %+0.4f]", boot_uh.mean_delta, boot_uh.ci_lower, boot_uh.ci_upper)

    # McNemar (pooled pixel-level) vs classical.
    pa = {name: np.concatenate([x.ravel() for x in preds]) for name, preds in
          (("classical", classical_preds), ("unet", unet_preds), ("hybrid", hybrid_preds))}
    yy = np.concatenate([y.ravel() for y in labels_all])
    mc_unet = mcnemar_test(pa["unet"], pa["classical"], yy)
    mc_hyb = mcnemar_test(pa["hybrid"], pa["classical"], yy)
    log.info("McNemar unet vs classical: chi2=%.1f  p=%.2e  %s", mc_unet.statistic, mc_unet.p_value,
             "significant" if mc_unet.significant() else "ns")
    log.info("McNemar hybrid vs classical: chi2=%.1f  p=%.2e  %s", mc_hyb.statistic, mc_hyb.p_value,
             "significant" if mc_hyb.significant() else "ns")

    # Persist outputs.
    summary = {
        "split": args.split,
        "n_chips": n,
        "classical_config": classical_cfg.name,
        "hybrid_weight_unet": args.hybrid_weight_unet,
        "metrics": rows,
        "runtime_ms_per_chip": runtimes,
        "bootstrap_ci": {
            "unet_minus_classical": boot_unet.__dict__,
            "hybrid_minus_classical": boot_hyb.__dict__,
            "unet_minus_hybrid": boot_uh.__dict__,
        },
        "mcnemar": {
            "unet_vs_classical": mc_unet.__dict__,
            "hybrid_vs_classical": mc_hyb.__dict__,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    for name, arr in per_chip_iou.items():
        np.savez(per_chip_dir / f"{name}.npz", iou=arr)
    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
