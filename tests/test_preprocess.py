"""Tests for src/preprocess/*."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from src.preprocess.cloud_mask import fraction_cloudy, scl_cloud_mask
from src.preprocess.coregister import assert_aligned, coregister
from src.preprocess.histogram_match import match_histograms_perband
from src.preprocess.reflectance import dn_to_reflectance, reflectance_to_dn


# ------------- reflectance -------------

def test_dn_to_reflectance_range() -> None:
    dn = np.array([[0, 1000, 5000, 10000, 11000]], dtype=np.uint16)
    out = dn_to_reflectance(dn, apply_offset=True, clip=True)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    # DN=1000 with -1000 offset → reflectance 0
    assert np.isclose(out[0, 1], 0.0, atol=1e-6)


def test_dn_reflectance_roundtrip() -> None:
    rng = np.random.default_rng(0)
    refl = rng.uniform(0.05, 0.9, size=(20, 20)).astype(np.float32)
    dn = reflectance_to_dn(refl)
    refl2 = dn_to_reflectance(dn)
    assert np.allclose(refl, refl2, atol=2e-4)


# ------------- SCL cloud mask -------------

def test_scl_cloud_mask_keeps_clear() -> None:
    # SCL class 4 = vegetation (clear), 9 = high-prob cloud (reject)
    scl = np.array([[4, 9, 4], [5, 8, 4]], dtype=np.uint8)
    mask = scl_cloud_mask(scl)
    expected = np.array([[1, 0, 1], [1, 0, 1]], dtype=bool)
    assert np.array_equal(mask, expected)
    assert np.isclose(fraction_cloudy(mask), 2 / 6)


# ------------- histogram matching -------------

def test_histogram_match_preserves_shape_and_band_count() -> None:
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1, (3, 32, 32)).astype(np.float32)
    ref = rng.uniform(0.3, 0.7, (3, 32, 32)).astype(np.float32)
    out = match_histograms_perband(src, ref)
    assert out.shape == src.shape
    assert out.dtype == src.dtype


def test_histogram_match_shifts_distribution_toward_reference() -> None:
    # Construct a clearly offset source; matched output should be closer to
    # the reference mean than the source mean was.
    src = (np.random.default_rng(2).uniform(0.0, 0.3, (1, 64, 64))).astype(np.float32)
    ref = (np.random.default_rng(3).uniform(0.6, 0.9, (1, 64, 64))).astype(np.float32)
    out = match_histograms_perband(src, ref)
    assert abs(out.mean() - ref.mean()) < abs(src.mean() - ref.mean())


def test_histogram_match_band_count_mismatch() -> None:
    with pytest.raises(ValueError, match="Band count mismatch"):
        match_histograms_perband(np.zeros((3, 4, 4)), np.zeros((4, 4, 4)))


# ------------- coregister -------------

def _write_raster(path: Path, arr: np.ndarray, crs: str, transform, nodata=None) -> None:
    if arr.ndim == 2:
        arr = arr[None, ...]
    profile = {
        "driver": "GTiff",
        "dtype": arr.dtype,
        "count": arr.shape[0],
        "height": arr.shape[1],
        "width": arr.shape[2],
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)


def test_coregister_aligns_to_reference(tmp_path: Path) -> None:
    # Source in one CRS / transform, reference in another.
    src_arr = np.ones((1, 32, 32), dtype=np.float32) * 0.5
    src_transform = from_origin(500_000, 1_100_000, 30.0, 30.0)
    src_path = tmp_path / "src.tif"
    _write_raster(src_path, src_arr, crs="EPSG:32643", transform=src_transform)

    ref_arr = np.zeros((1, 96, 96), dtype=np.float32)
    ref_transform = from_origin(500_000, 1_100_000, 10.0, 10.0)
    ref_path = tmp_path / "ref.tif"
    _write_raster(ref_path, ref_arr, crs="EPSG:32643", transform=ref_transform)

    out_path = coregister(src_path, ref_path, tmp_path / "out.tif")
    assert_aligned(out_path, ref_path)

    with rasterio.open(out_path) as dst:
        assert dst.count == 1
        assert dst.shape == (96, 96)
