"""CLI wrapper for U-Net training.

Example:
    python scripts/train_unet.py \
        --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
        --out-dir /content/drive/MyDrive/dda/checkpoints/unet_resnet34 \
        --epochs 30 --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.train.train_unet import TrainConfig, train  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sen1floods11-root", required=True)
    p.add_argument("--out-dir", default="checkpoints/unet_resnet34")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--bce-weight", type=float, default=0.5)
    p.add_argument("--pos-weight", type=float, default=2.0)
    p.add_argument("--train-crop", type=int, default=256)
    p.add_argument("--val-crop", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    args = p.parse_args()

    cfg = TrainConfig(
        sen1floods11_root=args.sen1floods11_root,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        bce_weight=args.bce_weight,
        pos_weight=args.pos_weight,
        train_crop=args.train_crop,
        val_crop=args.val_crop,
        num_workers=args.num_workers,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
        early_stopping_patience=args.patience,
    )
    best = train(cfg)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
