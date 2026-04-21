"""Tests for src/analysis/*."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin

from src.analysis.error_analysis import CATEGORY_NAMES, categorise, tabulate_errors
from src.analysis.quantify import AreaSummary, area_summary, landcover_breakdown
from src.analysis.severity import (
    Severity,
    SeverityConfig,
    cell_counts,
    classify,
    classify_raster,
)


# ---------- severity ----------

def test_severity_thresholds() -> None:
    # 100x100 mask; top-left 10x10 flooded (=1%), next strip 20x20 flooded (=4%), etc.
    mask = np.zeros((100, 100), dtype=bool)
    mask[:10, :10] = True  # 100 pixels -> 1% of 10000
    frac, cls = classify(mask, cfg=SeverityConfig(cell_px=100))
    assert frac.shape == (1, 1)
    assert cls.shape == (1, 1)
    # 1% < 5% default threshold → Severity.NONE
    assert Severity(int(cls[0, 0])) == Severity.NONE


def test_severity_severe_class() -> None:
    mask = np.ones((100, 100), dtype=bool)
    _, cls = classify(mask, cfg=SeverityConfig(cell_px=100))
    assert Severity(int(cls[0, 0])) == Severity.SEVERE


def test_severity_grid_partitioning() -> None:
    # 200x200 with 2x2 grid of 100x100 cells, each cell flooded at 0, 10, 20, 50 %.
    mask = np.zeros((200, 200), dtype=bool)
    mask[:100, 100:100 + 32] = True   # top-right cell ~10.2% flooded → Low
    mask[100:, :100].flat[:2000] = True  # bottom-left cell 20% → Moderate
    mask[100:, 100:].flat[:5000] = True  # bottom-right cell 50% → Severe
    _, cls = classify(mask, cfg=SeverityConfig(cell_px=100))
    assert cls.shape == (2, 2)
    assert Severity(int(cls[0, 0])) == Severity.NONE
    assert Severity(int(cls[0, 1])) == Severity.LOW
    assert Severity(int(cls[1, 0])) == Severity.MODERATE
    assert Severity(int(cls[1, 1])) == Severity.SEVERE


def test_severity_depth_weight() -> None:
    mask = np.zeros((100, 100), dtype=bool)
    mask[:10, :10] = True  # 1% flooded alone → Severity.NONE
    depth = np.ones((100, 100), dtype=np.float32)  # depth=1 everywhere
    cfg = SeverityConfig(cell_px=100, depth_weight=0.9)
    _, cls = classify(mask, depth=depth, cfg=cfg)
    # Effective fraction: 0.1*0.01 + 0.9*1.0 = 0.901 → Severe
    assert Severity(int(cls[0, 0])) == Severity.SEVERE


def test_cell_counts_sums_to_total() -> None:
    cls = np.array([[0, 1, 2], [3, 0, 1]], dtype=np.int8)
    counts = cell_counts(cls)
    assert sum(counts.values()) == cls.size


def test_classify_raster_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "mask.tif"
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:50, :50] = 1
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="uint8", count=1, height=100, width=100,
        crs="EPSG:32643",
        transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
    ) as dst:
        dst.write(arr, 1)

    out = classify_raster(path, tmp_path / "sev.tif", cfg=SeverityConfig(cell_px=50))
    with rasterio.open(out) as src:
        frac = src.read(1)
        cls = src.read(2)
    assert frac.shape == (2, 2)
    # Top-left cell is fully flooded, others not.
    assert frac[0, 0] == 1.0
    assert Severity(int(cls[0, 0])) == Severity.SEVERE


# ---------- quantify ----------

def test_area_summary(tmp_path: Path) -> None:
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:50, :] = 1  # 5000 flooded px
    path = tmp_path / "mask.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="uint8", count=1, height=100, width=100,
        crs="EPSG:32643",
        transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
    ) as dst:
        dst.write(arr, 1)

    s = area_summary(path)
    assert isinstance(s, AreaSummary)
    # 5000 px × 100 m² = 0.5 km²; total 1 km²; fraction = 0.5
    assert s.flooded_km2 == pytest.approx(0.5)
    assert s.total_km2 == pytest.approx(1.0)
    assert s.flooded_fraction == pytest.approx(0.5)


def test_landcover_breakdown(tmp_path: Path) -> None:
    # Build 10x10 mask with the right half flooded, and a 10x10 WorldCover
    # with the top half class 40 (cropland), bottom half class 50 (built-up).
    profile = {
        "driver": "GTiff", "dtype": "uint8", "count": 1,
        "height": 10, "width": 10,
        "crs": "EPSG:32643",
        "transform": from_origin(500_000, 1_100_000, 10.0, 10.0),
    }
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:, 5:] = 1
    with rasterio.open(tmp_path / "mask.tif", "w", **profile) as dst:
        dst.write(mask, 1)

    lc = np.zeros((10, 10), dtype=np.uint8)
    lc[:5, :] = 40
    lc[5:, :] = 50
    with rasterio.open(tmp_path / "lc.tif", "w", **profile) as dst:
        dst.write(lc, 1)

    df = landcover_breakdown(tmp_path / "mask.tif", tmp_path / "lc.tif")
    assert isinstance(df, pd.DataFrame)
    assert set(df["class_code"].tolist()) == {40, 50}
    # Each class has 50 total px and 25 flooded px → class_fraction_flooded = 0.5, share = 0.5.
    for row in df.itertuples():
        assert row.class_fraction_flooded == pytest.approx(0.5)
        assert row.share_of_flood == pytest.approx(0.5)


def test_landcover_shape_mismatch_raises(tmp_path: Path) -> None:
    # Build two differently-sized rasters.
    for name, size in (("a.tif", 10), ("b.tif", 20)):
        arr = np.zeros((size, size), dtype=np.uint8)
        with rasterio.open(
            tmp_path / name, "w",
            driver="GTiff", dtype="uint8", count=1, height=size, width=size,
            crs="EPSG:32643",
            transform=from_origin(500_000, 1_100_000, 10.0, 10.0),
        ) as dst:
            dst.write(arr, 1)
    with pytest.raises(ValueError, match="share shape"):
        landcover_breakdown(tmp_path / "a.tif", tmp_path / "b.tif")


# ---------- error analysis ----------

def test_categorise_classes_known_surfaces() -> None:
    # Build a (6, 4, 4) chip whose four pixels cover four categories.
    chip = np.zeros((6, 4, 4), dtype=np.float32)
    # pixel (0, 0): turbid water — B3 high, B11 low, B4 low, B8 low
    chip[:, 0, 0] = [0.05, 0.20, 0.03, 0.02, 0.02, 0.01]
    # pixel (0, 1): vegetation — B4 low, B8 high
    chip[:, 0, 1] = [0.05, 0.08, 0.04, 0.45, 0.25, 0.15]
    # pixel (0, 2): dark land — MNDWI<0, NIR<0.1
    chip[:, 0, 2] = [0.03, 0.03, 0.04, 0.05, 0.10, 0.06]
    # pixel (0, 3): bare / sparse — NDVI 0.1, MNDWI<0
    chip[:, 0, 3] = [0.15, 0.12, 0.10, 0.12, 0.18, 0.15]
    cat = categorise(chip)
    assert cat[0, 0] == 0   # turbid_water
    assert cat[0, 1] == 2   # vegetation
    assert cat[0, 2] == 1   # dark_land
    assert cat[0, 3] == 3   # bare_sparse


def test_tabulate_errors_dataframe_shape() -> None:
    rng = np.random.default_rng(0)
    imgs = [rng.uniform(0, 1, (6, 8, 8)).astype(np.float32) for _ in range(3)]
    preds = [rng.integers(0, 2, (8, 8)).astype(bool) for _ in range(3)]
    labels = [rng.integers(0, 2, (8, 8), dtype=np.int64) for _ in range(3)]
    df = tabulate_errors(imgs, preds, labels)
    assert list(df["category"]) == list(CATEGORY_NAMES)
    assert {"fp_count", "fn_count", "fp_pct", "fn_pct"}.issubset(df.columns)
    # Percentages should sum to ~100 (or 0 if no errors).
    assert pytest.approx(df["fp_pct"].sum(), abs=1e-6) in (0.0, 100.0)
    assert pytest.approx(df["fn_pct"].sum(), abs=1e-6) in (0.0, 100.0)
