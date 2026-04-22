"""Build tiny bundled sample chips from Sen1Floods11 for the Streamlit demo.

Extracts 3 diverse test-split chips (low, medium, high flood fraction),
downsamples them from 512×512 to 128×128 to keep the bundle small, and saves
them as compressed ``.npz`` files in ``app/sample_chips/``. The total
bundle weighs ~500 kB.

Rationale: Streamlit Community Cloud can't host the 3 GB Sen1Floods11 dump.
Bundled samples let the public app run end-to-end without any external data.

Run once from a machine that has the full Sen1Floods11 dataset on disk:

    python scripts/build_sample_chips.py \
        --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
        --out-dir app/sample_chips

Commit the resulting ``app/sample_chips/*.npz`` files to the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.sen1floods11_loader import Sen1Floods11Dataset  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)


def _downsample_chip(image: np.ndarray, label: np.ndarray, factor: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Block-average the image, nearest-downsample the label."""
    c, h, w = image.shape
    h_new, w_new = h // factor, w // factor

    img_out = image[:, : h_new * factor, : w_new * factor].reshape(
        c, h_new, factor, w_new, factor
    ).mean(axis=(2, 4)).astype(np.float32)

    lab_out = label[: h_new * factor : factor, : w_new * factor : factor].astype(np.int16)
    return img_out, lab_out


def _choose_diverse_chips(ds: Sen1Floods11Dataset, n: int = 3) -> list[int]:
    """Pick `n` chips spanning low / medium / high flood fractions."""
    fractions: list[tuple[int, float]] = []
    for i in range(len(ds)):
        label = ds[i]["label"].numpy()
        valid = label != -1
        if not valid.any():
            frac = 0.0
        else:
            frac = float((label[valid] == 1).mean())
        fractions.append((i, frac))

    # Sort by flood fraction, then pick quantiles.
    fractions.sort(key=lambda kv: kv[1])
    if n == 1:
        return [fractions[len(fractions) // 2][0]]

    quantile_indices = [int(round(q * (len(fractions) - 1))) for q in np.linspace(0.1, 0.9, n)]
    return [fractions[q][0] for q in quantile_indices]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sen1floods11-root", required=True)
    p.add_argument("--split", default="test", choices=["train", "valid", "test", "bolivia"])
    p.add_argument("--out-dir", default="app/sample_chips")
    p.add_argument("--n-chips", type=int, default=3)
    p.add_argument("--downsample-factor", type=int, default=4)
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ds = Sen1Floods11Dataset(args.sen1floods11_root, split=args.split, modality="s2")
    log.info("Loaded %s split: n=%d", args.split, len(ds))

    picks = _choose_diverse_chips(ds, n=args.n_chips)
    log.info("Chosen chip indices: %s", picks)

    manifest = []
    for i, idx in enumerate(picks):
        sample = ds[idx]
        chip_id = sample["chip_id"]
        img, lab = _downsample_chip(
            sample["image"].numpy(),
            sample["label"].numpy(),
            factor=args.downsample_factor,
        )
        flood_frac = float((lab[lab != -1] == 1).mean()) if (lab != -1).any() else 0.0
        out_path = out / f"chip_{i:02d}_{chip_id}.npz"
        np.savez_compressed(out_path, image=img, label=lab, chip_id=chip_id)
        size_kb = out_path.stat().st_size / 1024
        log.info("Saved %s  shape=%s  flood=%.1f%%  %.1f kB",
                 out_path.name, img.shape, flood_frac * 100, size_kb)
        manifest.append({
            "path": out_path.name,
            "chip_id": chip_id,
            "shape": list(img.shape),
            "flood_fraction": flood_frac,
            "pixel_size_m": 40,  # original 10 m × downsample factor 4
        })

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %d chips + manifest.json to %s", len(picks), out)


if __name__ == "__main__":
    main()
