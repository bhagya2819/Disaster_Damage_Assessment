"""Download real satellite chips from Sen1Floods11 (public GCS) for the demo.

Downloads only ~50 MB total (not the full 3 GB dataset), processes them into
bundled sample chips, and saves to app/sample_chips/. After running this,
the Streamlit app will show real satellite flood imagery instead of synthetic data.

Usage:
    python scripts/download_sample_chips.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Sen1Floods11 public GCS bucket (v1.1)
GCS_BASE = "https://storage.googleapis.com/sen1floods11/v1.1"

# Curated chips for presentation demo — diverse countries, range of flood levels
CHIPS = [
    # --- HIGH FLOOD (dramatic for demo) ---
    {
        "chip_id": "Sri-Lanka_534068",
        "desc": "🇱🇰 Sri Lanka — massive flooding (98% flooded)",
    },
    {
        "chip_id": "Pakistan_849790",
        "desc": "🇵🇰 Pakistan — severe flooding (67% flooded)",
    },
    # --- MEDIUM FLOOD (clear flood vs land contrast) ---
    {
        "chip_id": "India_900498",
        "desc": "🇮🇳 India — moderate flooding (48% flooded)",
    },
    {
        "chip_id": "India_591317",
        "desc": "🇮🇳 India — river overflow (46% flooded)",
    },
    {
        "chip_id": "India_747992",
        "desc": "🇮🇳 India — widespread inundation (45% flooded)",
    },
    # --- LOW FLOOD (shows precision — detecting small flood areas) ---
    {
        "chip_id": "Paraguay_790830",
        "desc": "🇵🇾 Paraguay — partial flooding (12% flooded)",
    },
    {
        "chip_id": "USA_430764",
        "desc": "🇺🇸 USA — coastal flood (6% flooded)",
    },
    {
        "chip_id": "Ghana_313799",
        "desc": "🇬🇭 Ghana — minor flooding (6% flooded)",
    },
]

# Sen1Floods11 S2 band order: 13 bands [B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B11,B12,QA60]
# DDA subset: B2,B3,B4,B8,B11,B12 = 0-based indices (1,2,3,7,10,11)
S2_BAND_0BASED = [1, 2, 3, 7, 10, 11]
REFLECTANCE_SCALE = 10_000.0

OUT_DIR = _REPO_ROOT / "app" / "sample_chips"


def download_file(url: str, dest: Path) -> None:
    """Download a file with progress."""
    print(f"  ↓ {dest.name}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"✅ ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ {e}")
        raise


def process_chip(chip_id: str, s2_path: Path, label_path: Path, factor: int = 4) -> dict:
    """Read an S2+Label chip, extract DDA bands, downsample, return metadata."""
    import rasterio  # noqa: PLC0415

    with rasterio.open(s2_path) as src:
        all_bands = src.read().astype(np.float32)

    if all_bands.shape[0] >= 13:
        image = all_bands[S2_BAND_0BASED]
    else:
        image = all_bands[:6]

    image = np.clip(image / REFLECTANCE_SCALE, 0.0, 1.0)

    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.int64)

    # Downsample to 128x128
    c, h, w = image.shape
    h_new, w_new = h // factor, w // factor

    img_ds = image[:, :h_new * factor, :w_new * factor].reshape(
        c, h_new, factor, w_new, factor
    ).mean(axis=(2, 4)).astype(np.float32)

    lab_ds = label[:h_new * factor:factor, :w_new * factor:factor].astype(np.int16)

    valid = lab_ds != -1
    flood_frac = float((lab_ds[valid] == 1).mean()) if valid.any() else 0.0

    return {
        "image": img_ds,
        "label": lab_ds,
        "chip_id": chip_id,
        "flood_fraction": flood_frac,
    }


def main() -> None:
    print("=" * 60)
    print("  Downloading real satellite chips for the demo")
    print(f"  {len(CHIPS)} chips from Sen1Floods11 (public dataset)")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUT_DIR / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Clean old chips
    for old in OUT_DIR.glob("chip_*.npz"):
        old.unlink()

    manifest = []

    for i, chip_info in enumerate(CHIPS):
        chip_id = chip_info["chip_id"]
        print(f"\n[{i+1}/{len(CHIPS)}] {chip_info['desc']}")

        s2_url = f"{GCS_BASE}/data/flood_events/HandLabeled/S2Hand/{chip_id}_S2Hand.tif"
        label_url = f"{GCS_BASE}/data/flood_events/HandLabeled/LabelHand/{chip_id}_LabelHand.tif"

        s2_path = tmp_dir / f"{chip_id}_S2Hand.tif"
        label_path = tmp_dir / f"{chip_id}_LabelHand.tif"

        download_file(s2_url, s2_path)
        download_file(label_url, label_path)

        print(f"  Processing...", end=" ", flush=True)
        data = process_chip(chip_id, s2_path, label_path)
        print(f"✅ flood={data['flood_fraction']*100:.1f}%")

        out_path = OUT_DIR / f"chip_{i:02d}_{chip_id}.npz"
        np.savez_compressed(
            out_path,
            image=data["image"],
            label=data["label"],
            chip_id=chip_id,
        )
        size_kb = out_path.stat().st_size / 1024
        print(f"  → {out_path.name} ({size_kb:.0f} kB)")

        manifest.append({
            "path": out_path.name,
            "chip_id": chip_id,
            "shape": list(data["image"].shape),
            "flood_fraction": data["flood_fraction"],
            "pixel_size_m": 40,
        })

        s2_path.unlink(missing_ok=True)
        label_path.unlink(missing_ok=True)

    try:
        tmp_dir.rmdir()
    except OSError:
        pass

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print(f"  ✅ {len(manifest)} real satellite chips ready!")
    print(f"  Saved to: app/sample_chips/")
    print(f"")
    for entry in manifest:
        print(f"    {entry['chip_id']:25s}  flood={entry['flood_fraction']*100:5.1f}%")
    print(f"\n  Restart the Streamlit app to use them.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
