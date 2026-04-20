"""Download the Sen1Floods11 dataset to Google Drive from Google Colab.

Sen1Floods11 (Bonafilia et al., 2020) lives in the PUBLIC Google Cloud Storage
bucket `gs://sen1floods11/`. No GCP auth is needed to read it; `gsutil` is
already installed in Colab.

Dataset splits available:
  - HandLabeled           (~3 GB)  446 hand-annotated chips  ← we use this
  - WeaklyLabeled         (~40 GB) 4,385 auto-labeled chips  ← optional
  - PermanentWaterJRC     (~2 GB)  JRC Global Surface Water priors
  - S1Hand / S2Hand / LabelHand    split folders under HandLabeled

For the course project the HandLabeled subset is sufficient to train a U-Net
with IoU ≥ 0.75 on the test split. Fire WeaklyLabeled only if you want to
push results in Phase 7.

Usage in Google Colab — paste each block into its OWN cell, with NO leading
indentation (Colab magics like %cd must sit at column 0):

Cell 1:
!git clone https://github.com/bhagya2819/Disaster_Damage_Assessment.git
%cd Disaster_Damage_Assessment

Cell 2:
!pip install -q rich

Cell 3:
from google.colab import drive
drive.mount('/content/drive')

Cell 4:
from google.colab import auth
auth.authenticate_user()

Cell 5:
!python scripts/download_sen1floods11.py --dest /content/drive/MyDrive/dda/sen1floods11 --subset hand

After the first download, subsequent Colab sessions just mount the same Drive
folder — no re-download needed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

GCS_ROOT = "gs://sen1floods11/v1.1"

SUBSETS: dict[str, list[str]] = {
    "hand": [
        f"{GCS_ROOT}/data/flood_events/HandLabeled",
        f"{GCS_ROOT}/splits/flood_handlabeled",
    ],
    "weak": [
        f"{GCS_ROOT}/data/flood_events/WeaklyLabeled",
        f"{GCS_ROOT}/splits/flood_weaklylabeled",
    ],
    "jrc": [
        f"{GCS_ROOT}/data/perm_water",
    ],
    "all": [
        f"{GCS_ROOT}/data",
        f"{GCS_ROOT}/splits",
    ],
}


def check_gsutil() -> None:
    if shutil.which("gsutil") is None:
        sys.exit(
            "gsutil not found. In Colab it is preinstalled; elsewhere run "
            "`curl https://sdk.cloud.google.com | bash`."
        )


def rsync(src: str, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["gsutil", "-m", "rsync", "-r", src, str(dst)]
    print(f"\n→ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dest",
        type=Path,
        required=True,
        help="Destination root (e.g. /content/drive/MyDrive/dda/sen1floods11).",
    )
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()),
        default="hand",
        help="Which slice to download. 'hand' is enough for the course project.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gsutil commands without running them.",
    )
    args = parser.parse_args()

    check_gsutil()
    args.dest.mkdir(parents=True, exist_ok=True)

    targets = SUBSETS[args.subset]
    for src in targets:
        rel = src.replace(f"{GCS_ROOT}/", "")
        dst = args.dest / rel
        if args.dry_run:
            print(f"DRY: gsutil -m rsync -r {src} {dst}")
        else:
            rsync(src, dst)

    print(f"\n✓ Done. Sen1Floods11 '{args.subset}' → {args.dest}")
    print("  Add this path to your .env as SEN1FLOODS11_DIR for the loader.")


if __name__ == "__main__":
    main()
