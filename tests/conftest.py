"""Shared pytest fixtures — mostly synthetic GeoTIFFs so the loader tests
never depend on network, GEE, or the ~3 GB Sen1Floods11 download."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def _write_geotiff(
    path: Path,
    arr: np.ndarray,
    crs: str = "EPSG:32643",
    pixel_size: float = 10.0,
    nodata: float | None = None,
) -> None:
    """Write a (C, H, W) or (H, W) numpy array as a GeoTIFF."""
    if arr.ndim == 2:
        arr = arr[None, ...]
    count, height, width = arr.shape
    transform = from_origin(west=500_000, north=1_100_000, xsize=pixel_size, ysize=pixel_size)
    profile = {
        "driver": "GTiff",
        "dtype": arr.dtype,
        "count": count,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": "lzw",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


@pytest.fixture
def sen1floods11_root(tmp_path: Path) -> Path:
    """Build a minimal Sen1Floods11 HandLabeled tree with 4 synthetic chips.

    Layout mirrors the real dataset so the loader cannot tell it's a fake.
    """
    root = tmp_path / "sen1floods11"
    s1 = root / "data" / "flood_events" / "HandLabeled" / "S1Hand"
    s2 = root / "data" / "flood_events" / "HandLabeled" / "S2Hand"
    lab = root / "data" / "flood_events" / "HandLabeled" / "LabelHand"
    splits = root / "splits" / "flood_handlabeled"
    for d in (s1, s2, lab, splits):
        d.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    chip_ids = ["Bolivia_103757", "Paraguay_009876", "India_012345", "Ghana_054321"]

    for cid in chip_ids:
        # S2: 13 bands, uint16 in [0, 10000]
        s2_arr = rng.integers(0, 10_000, size=(13, 64, 64), dtype=np.uint16)
        _write_geotiff(s2 / f"{cid}_S2Hand.tif", s2_arr)
        # S1: 2 bands (VV, VH), linear sigma-naught as float32
        s1_arr = rng.uniform(0.001, 0.5, size=(2, 64, 64)).astype(np.float32)
        _write_geotiff(s1 / f"{cid}_S1Hand.tif", s1_arr)
        # Label: int16 in {-1, 0, 1}
        lbl = rng.choice([-1, 0, 1], size=(64, 64), p=[0.1, 0.6, 0.3]).astype(np.int16)
        _write_geotiff(lab / f"{cid}_LabelHand.tif", lbl)

    # Split CSVs: two per split to give every split ≥ 2 items.
    for split, members in [
        ("train", chip_ids[:2]),
        ("valid", chip_ids[2:3]),
        ("test", chip_ids[3:]),
        ("bolivia", [chip_ids[0]]),
    ]:
        csv_path = splits / f"flood_{split}_data.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            for cid in members:
                w.writerow([f"{cid}_S1Hand.tif", f"{cid}_LabelHand.tif"])

    return root


@pytest.fixture
def reference_raster(tmp_path: Path) -> Path:
    """A small synthetic Sentinel-2-like 6-band GeoTIFF to test rasterization against."""
    path = tmp_path / "ref.tif"
    arr = np.random.default_rng(1).uniform(0, 1, size=(6, 128, 128)).astype(np.float32)
    _write_geotiff(path, arr, crs="EPSG:32643", pixel_size=10.0)
    return path
