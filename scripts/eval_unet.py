"""Evaluate a trained U-Net checkpoint on Sen1Floods11 test split and optionally
compare against the classical winner.

Writes:
- ``<out_dir>/unet_metrics.json``  — aggregated metrics.
- ``<out_dir>/per_chip/unet.npz``  — per-chip IoU array (parallel to Phase-3 ablation npz files).

Example:
    python scripts/eval_unet.py \
        --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
        --checkpoint /content/drive/MyDrive/dda/checkpoints/unet_resnet34/best.pt \
        --classical-per-chip results/per_chip/ndwi_yen_raw.npz \
        --out-dir results/unet/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.sen1floods11_loader import LABEL_IGNORE_INDEX, Sen1Floods11Dataset  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.significance import mcnemar_test, paired_bootstrap_iou  # noqa: E402
from src.inference.predict import predict_chip  # noqa: E402
from src.models.unet import load_checkpoint  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sen1floods11-root", required=True)
    p.add_argument("--checkpoint", required=True, help="Path to best.pt produced by train_unet.py")
    p.add_argument("--split", default="test", choices=["valid", "test", "bolivia"])
    p.add_argument("--classical-per-chip", default=None,
                   help="Optional .npz from Phase-3 ablation for paired comparison.")
    p.add_argument("--out-dir", default="results/unet")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    log.info("Loading checkpoint → %s (device=%s)", args.checkpoint, device)
    model = load_checkpoint(args.checkpoint, device=device)

    ds = Sen1Floods11Dataset(args.sen1floods11_root, split=args.split, modality="s2")
    log.info("Evaluating on Sen1Floods11 split=%s n=%d", args.split, len(ds))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_chip_dir = out_dir / "per_chip"
    per_chip_dir.mkdir(exist_ok=True)

    per_chip_iou = np.empty(len(ds), dtype=np.float64)
    agg: dict[str, list[float]] = {
        k: [] for k in ("iou", "f1", "precision", "recall", "accuracy", "cohen_kappa")
    }

    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for i in tqdm(range(len(ds)), desc="eval"):
        sample = ds[i]
        img = sample["image"].numpy()
        y = sample["label"].numpy()
        probs = predict_chip(model, img, device=device)
        pred = probs >= args.threshold
        m = metrics.summary(pred, y, ignore_index=LABEL_IGNORE_INDEX)
        per_chip_iou[i] = m["iou"]
        for k in agg:
            agg[k].append(m[k])
        all_preds.append(pred)
        all_labels.append(y)

    agg_mean = {k: float(np.nanmean(v)) for k, v in agg.items()}
    log.info("\nU-Net mean metrics on split=%s:", args.split)
    for k, v in agg_mean.items():
        log.info("  %-13s %.4f", k, v)

    (out_dir / "unet_metrics.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "n_chips": len(ds),
        "threshold": args.threshold,
        "means": agg_mean,
    }, indent=2))
    np.savez(per_chip_dir / "unet.npz", iou=per_chip_iou)

    # Optional paired comparison vs classical.
    if args.classical_per_chip:
        classical_iou = np.load(args.classical_per_chip)["iou"]
        if classical_iou.shape != per_chip_iou.shape:
            log.warning("Classical IoU length %d ≠ U-Net %d — skipping paired test.",
                        len(classical_iou), len(per_chip_iou))
        else:
            boot = paired_bootstrap_iou(per_chip_iou, classical_iou, n_bootstrap=10_000)
            log.info("\nPaired IoU: U-Net − classical = %+0.4f  [95%% CI %+0.4f, %+0.4f]",
                     boot.mean_delta, boot.ci_lower, boot.ci_upper)

            # McNemar over pooled pixels.
            pa = np.concatenate([p.ravel() for p in all_preds])
            yy = np.concatenate([y.ravel() for y in all_labels])
            log.info("(McNemar needs paired pixel-level classical masks — run from notebook)")
            _ = (pa, yy)  # placeholder; the notebook does the pooled McNemar

    log.info("Done → %s", out_dir)


if __name__ == "__main__":
    main()
