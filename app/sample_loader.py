"""Loader for the bundled sample chips that ship with the Streamlit app.

If the real Sen1Floods11 dataset is not mounted, the app falls back to
``app/sample_chips/*.npz`` — a handful of tiny pre-extracted chips kept in
the repo so the public demo is always functional.

If no bundled chips exist either, :func:`synthetic_chip` returns a
hand-crafted 128×128 reflectance chip with a clear water/land split — so the
app is *never* blocked by missing data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_DIR = Path(__file__).resolve().parent / "sample_chips"
MANIFEST = SAMPLE_DIR / "manifest.json"


@dataclass(frozen=True)
class SampleChip:
    chip_id: str
    image: np.ndarray     # (C, H, W) float32 reflectance
    label: np.ndarray     # (H, W) int with {-1, 0, 1}
    pixel_size_m: float
    flood_fraction: float

    def shape(self) -> tuple[int, int]:
        return self.image.shape[1:]  # (H, W)


def bundled_manifest() -> list[dict]:
    """Return the contents of ``app/sample_chips/manifest.json`` (empty list if missing)."""
    if not MANIFEST.exists():
        return []
    try:
        return json.loads(MANIFEST.read_text())
    except json.JSONDecodeError:
        return []


def load_bundled(index: int) -> SampleChip | None:
    """Load the idx-th bundled sample. Returns None if bundle is absent."""
    manifest = bundled_manifest()
    if not manifest or index < 0 or index >= len(manifest):
        return None
    entry = manifest[index]
    path = SAMPLE_DIR / entry["path"]
    if not path.exists():
        return None
    npz = np.load(path)
    return SampleChip(
        chip_id=str(entry.get("chip_id", path.stem)),
        image=npz["image"].astype(np.float32),
        label=npz["label"].astype(np.int64),
        pixel_size_m=float(entry.get("pixel_size_m", 40.0)),
        flood_fraction=float(entry.get("flood_fraction", 0.0)),
    )


def synthetic_chip(seed: int = 0, h: int = 128, w: int = 128) -> SampleChip:
    """Always-available synthetic fallback: left half water, right half vegetation.

    Used when neither the real dataset nor the bundled samples are available —
    the UI can still demonstrate the full pipeline.
    """
    rng = np.random.default_rng(seed)
    water = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01], dtype=np.float32)
    veg = np.array([0.05, 0.08, 0.04, 0.45, 0.25, 0.15], dtype=np.float32)

    img = np.empty((6, h, w), dtype=np.float32)
    img[:, :, : w // 2] = water[:, None, None] + rng.normal(0, 0.005, (6, h, w // 2)).astype(np.float32)
    img[:, :, w // 2 :] = veg[:, None, None] + rng.normal(0, 0.005, (6, h, w - w // 2)).astype(np.float32)
    img = np.clip(img, 0.0, 1.0)

    label = np.zeros((h, w), dtype=np.int64)
    label[:, : w // 2] = 1
    return SampleChip(
        chip_id="synthetic_bimodal",
        image=img,
        label=label,
        pixel_size_m=10.0,
        flood_fraction=0.5,
    )


def load_geotiff_as_chip(path: str | Path, pixel_size_m: float = 10.0) -> SampleChip:
    """Read a user-uploaded 6-band reflectance GeoTIFF into a SampleChip.

    The app's "upload your own" tab calls this. The file must have at least
    6 bands in the DDA order (B2, B3, B4, B8, B11, B12), already scaled to
    reflectance in [0, 1].
    """
    import rasterio  # noqa: PLC0415

    path = Path(path)
    with rasterio.open(path) as src:
        if src.count < 6:
            raise ValueError(
                f"Expected >= 6 bands (B2, B3, B4, B8, B11, B12); got {src.count}."
            )
        arr = src.read([1, 2, 3, 4, 5, 6]).astype(np.float32)
        px_size = abs(src.transform.a) if src.transform.a else pixel_size_m

    # If the data looks like raw DN (max > 20), scale to reflectance.
    if np.nanmax(arr) > 20:
        arr = np.clip(arr / 10_000.0, 0.0, 1.0)
    else:
        arr = np.clip(arr, 0.0, 1.0)

    h, w = arr.shape[1:]
    return SampleChip(
        chip_id=path.stem,
        image=arr,
        label=np.full((h, w), -1, dtype=np.int64),  # no GT for uploaded chips
        pixel_size_m=float(px_size),
        flood_fraction=float("nan"),
    )


__all__ = [
    "MANIFEST",
    "SAMPLE_DIR",
    "SampleChip",
    "bundled_manifest",
    "load_bundled",
    "load_geotiff_as_chip",
    "synthetic_chip",
]
