"""Tests for app/sample_loader.py — bundled chips + synthetic fallback + upload."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from app import sample_loader


def test_synthetic_chip_shape_and_range() -> None:
    chip = sample_loader.synthetic_chip(seed=0, h=64, w=64)
    assert chip.image.shape == (6, 64, 64)
    assert chip.image.dtype == np.float32
    assert 0.0 <= chip.image.min() <= chip.image.max() <= 1.0
    # Synthetic chip is 50 % water on the left.
    assert chip.label.shape == (64, 64)
    assert chip.label[:, :32].mean() == 1.0
    assert chip.label[:, 32:].mean() == 0.0
    assert chip.flood_fraction == pytest.approx(0.5, abs=0.01)


def test_bundled_manifest_empty_when_missing(monkeypatch, tmp_path: Path) -> None:
    """With no manifest, bundled_manifest() returns []."""
    monkeypatch.setattr(sample_loader, "SAMPLE_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(sample_loader, "MANIFEST", tmp_path / "nonexistent" / "manifest.json")
    assert sample_loader.bundled_manifest() == []


def test_load_bundled_returns_none_when_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sample_loader, "SAMPLE_DIR", tmp_path / "nonexistent")
    monkeypatch.setattr(sample_loader, "MANIFEST", tmp_path / "nonexistent" / "manifest.json")
    assert sample_loader.load_bundled(0) is None


def test_load_bundled_roundtrip(tmp_path: Path, monkeypatch) -> None:
    """Write a fake bundle, then load it back."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    img = np.random.default_rng(0).uniform(0, 1, (6, 32, 32)).astype(np.float32)
    lab = np.zeros((32, 32), dtype=np.int64)
    lab[:, :16] = 1
    np.savez_compressed(bundle / "chip_00_fake.npz", image=img, label=lab, chip_id="fake_00")
    manifest_path = bundle / "manifest.json"
    manifest_path.write_text(
        '[{"path": "chip_00_fake.npz", "chip_id": "fake_00", '
        '"flood_fraction": 0.5, "pixel_size_m": 40}]'
    )

    monkeypatch.setattr(sample_loader, "SAMPLE_DIR", bundle)
    monkeypatch.setattr(sample_loader, "MANIFEST", manifest_path)

    chip = sample_loader.load_bundled(0)
    assert chip is not None
    assert chip.chip_id == "fake_00"
    assert chip.image.shape == (6, 32, 32)
    assert chip.pixel_size_m == 40.0


def test_load_geotiff_as_chip_scales_dn(tmp_path: Path) -> None:
    """DN-scale input (0-10000) must be rescaled to [0, 1]."""
    path = tmp_path / "dn.tif"
    arr = np.random.default_rng(1).integers(0, 10_000, (6, 64, 64), dtype=np.uint16)
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="uint16", count=6, height=64, width=64,
        crs="EPSG:32643", transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
    ) as dst:
        dst.write(arr)

    chip = sample_loader.load_geotiff_as_chip(path)
    assert chip.image.shape == (6, 64, 64)
    assert 0.0 <= chip.image.min() <= chip.image.max() <= 1.0
    assert chip.pixel_size_m == pytest.approx(10.0)


def test_load_geotiff_rejects_too_few_bands(tmp_path: Path) -> None:
    """<6 bands must raise."""
    path = tmp_path / "rgb.tif"
    arr = np.zeros((3, 16, 16), dtype=np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="float32", count=3, height=16, width=16,
        crs="EPSG:32643", transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
    ) as dst:
        dst.write(arr)

    with pytest.raises(ValueError, match=">= 6 bands"):
        sample_loader.load_geotiff_as_chip(path)


def test_load_geotiff_already_reflectance(tmp_path: Path) -> None:
    """Reflectance-scale input (0-1) must be preserved."""
    path = tmp_path / "refl.tif"
    arr = np.random.default_rng(2).uniform(0, 1, (6, 16, 16)).astype(np.float32)
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="float32", count=6, height=16, width=16,
        crs="EPSG:32643", transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
    ) as dst:
        dst.write(arr)

    chip = sample_loader.load_geotiff_as_chip(path)
    # Should not have been divided by 10_000.
    assert chip.image.max() > 0.1
    assert chip.image.max() <= 1.0


def test_synthetic_chip_seed_reproducible() -> None:
    a = sample_loader.synthetic_chip(seed=42)
    b = sample_loader.synthetic_chip(seed=42)
    np.testing.assert_allclose(a.image, b.image)
