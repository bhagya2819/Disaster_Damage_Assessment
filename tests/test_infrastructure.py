"""Tests for src/analysis/infrastructure.py — no network, synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import LineString, Polygon

from src.analysis.infrastructure import (
    InfrastructureImpact,
    buildings_flooded,
    compute,
    roads_flooded,
)


CRS = "EPSG:32643"
PX = 10.0  # metres per pixel


@pytest.fixture
def flood_mask_tif(tmp_path: Path) -> Path:
    """Build a 100×100 flood mask GeoTIFF at 10 m pixels, flooded on the left half."""
    arr = np.zeros((100, 100), dtype=np.uint8)
    arr[:, :50] = 1
    path = tmp_path / "mask.tif"
    with rasterio.open(
        path, "w",
        driver="GTiff", dtype="uint8", count=1, height=100, width=100,
        crs=CRS, transform=from_origin(500_000, 1_100_000, PX, PX),
    ) as dst:
        dst.write(arr, 1)
    return path


@pytest.fixture
def roads_gdf() -> gpd.GeoDataFrame:
    # Three 1000 m horizontal roads: one fully on left (flooded), one fully on
    # right (dry), one crossing both halves.
    lines = [
        LineString([(500_010, 1_099_990), (500_490, 1_099_990)]),  # fully on left
        LineString([(500_510, 1_099_970), (500_990, 1_099_970)]),  # fully on right
        LineString([(500_010, 1_099_950), (500_990, 1_099_950)]),  # crossing
    ]
    classes = ["residential", "residential", "primary"]
    return gpd.GeoDataFrame({"highway": classes, "geometry": lines}, crs=CRS)


@pytest.fixture
def buildings_gdf() -> gpd.GeoDataFrame:
    # Four 30×30 m square buildings; two on the flooded left half, two on the right.
    polys = [
        Polygon.from_bounds(500_100, 1_099_100, 500_130, 1_099_130),  # left
        Polygon.from_bounds(500_200, 1_099_200, 500_230, 1_099_230),  # left
        Polygon.from_bounds(500_700, 1_099_700, 500_730, 1_099_730),  # right
        Polygon.from_bounds(500_800, 1_099_800, 500_830, 1_099_830),  # right
    ]
    return gpd.GeoDataFrame({"building": ["yes"] * 4, "geometry": polys}, crs=CRS)


# -------- roads --------

def test_roads_flooded_counts_left_half(flood_mask_tif, roads_gdf) -> None:
    km_fl, km_tot, df = roads_flooded(flood_mask_tif, roads_gdf=roads_gdf)
    # Left road (480 m) is fully flooded; right road (480 m) is not; crossing
    # road (980 m) is half flooded (~490 m). Total expected ≈ 970 m ≈ 0.97 km.
    assert 0.7 < km_fl < 1.2, f"flooded km {km_fl}"
    assert 1.8 < km_tot < 2.1, f"total km {km_tot}"
    assert "km_flooded" in df.columns
    assert set(df["highway"]) == {"residential", "primary"}


def test_roads_empty_gdf(flood_mask_tif) -> None:
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS)
    km_fl, km_tot, df = roads_flooded(flood_mask_tif, roads_gdf=empty)
    assert km_fl == 0.0 and km_tot == 0.0
    assert df.empty


# -------- buildings --------

def test_buildings_flooded_count(flood_mask_tif, buildings_gdf) -> None:
    flooded, total, fl_area, tot_area = buildings_flooded(
        flood_mask_tif, buildings_gdf=buildings_gdf
    )
    assert total == 4
    assert flooded == 2  # the two on the left half
    # Total footprint: 4 × 30 × 30 = 3600 m²
    assert tot_area == pytest.approx(3600.0, abs=1.0)
    # Flooded footprint: 2 × 30 × 30 = 1800 m²
    assert fl_area == pytest.approx(1800.0, abs=1.0)


def test_buildings_empty_gdf(flood_mask_tif) -> None:
    empty = gpd.GeoDataFrame(geometry=[], crs=CRS)
    flooded, total, fa, ta = buildings_flooded(flood_mask_tif, buildings_gdf=empty)
    assert flooded == 0 and total == 0 and fa == 0.0 and ta == 0.0


def test_buildings_filters_non_polygons(flood_mask_tif) -> None:
    mixed = gpd.GeoDataFrame(
        {"building": ["yes", "yes"]},
        geometry=[
            LineString([(500_100, 1_099_100), (500_110, 1_099_110)]),  # should be dropped
            Polygon.from_bounds(500_100, 1_099_100, 500_130, 1_099_130),  # kept
        ],
        crs=CRS,
    )
    flooded, total, _, _ = buildings_flooded(flood_mask_tif, buildings_gdf=mixed)
    assert total == 1
    assert flooded == 1


# -------- compute() --------

def test_compute_returns_impact(flood_mask_tif, roads_gdf, buildings_gdf) -> None:
    impact = compute(flood_mask_tif, roads_gdf=roads_gdf, buildings_gdf=buildings_gdf)
    assert isinstance(impact, InfrastructureImpact)
    assert impact.aoi_area_km2 == pytest.approx(1.0, abs=0.01)
    assert impact.buildings_flooded_count == 2
    assert impact.buildings_total_count == 4
    summary = impact.as_summary()
    assert summary["buildings_fraction_flooded"] == pytest.approx(0.5, abs=0.01)
    assert summary["roads_fraction_flooded"] > 0
