"""CLI wrapper for the classical-DIP ablation study on Sen1Floods11.

Usage:
    python scripts/run_ablation.py \
        --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
        --split test \
        --out-dir results/

    # Smoke test on 5 chips only:
    python scripts/run_ablation.py ... --limit 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.sen1floods11_loader import Sen1Floods11Dataset
from src.eval.ablation import run_ablation
from src.utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sen1floods11-root", required=True, help="Sen1Floods11 HandLabeled root.")
    p.add_argument("--split", default="test", choices=["train", "valid", "test", "bolivia"])
    p.add_argument("--out-dir", default="results")
    p.add_argument("--limit", type=int, default=None, help="Max chips (smoke test).")
    args = p.parse_args()

    ds = Sen1Floods11Dataset(root=args.sen1floods11_root, split=args.split, modality="s2")
    log.info("Loaded Sen1Floods11 split=%s n=%d", args.split, len(ds))

    df = run_ablation(ds, limit=args.limit, results_dir=args.out_dir)

    # Print the sorted top 5 by mean IoU for quick inspection.
    top = df.sort_values("mean_iou", ascending=False).head(5)
    log.info("\nTop 5 by mean IoU:\n%s", top[["config", "mean_iou", "mean_f1", "mean_cohen_kappa"]].to_string(index=False))

    out_path = Path(args.out_dir) / "ablation.csv"
    log.info("Full results written to %s", out_path)


if __name__ == "__main__":
    main()
