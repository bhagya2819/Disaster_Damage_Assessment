"""Unit tests for src/data/*.

All tests use synthetic fixtures (see conftest.py) so they run in < 2 s and
require no network access.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch
from shapely.geometry import Polygon

from src.data.aoi import load_aoi
from src.data.ground_truth import flood_pixel_fraction, rasterize_flood_polygons
from src.data.sen1floods11_loader import (
    LABEL_IGNORE_INDEX,
    S2_BAND_INDICES,
    Sen1Floods11Dataset,
)


# ----------------- AOI config -----------------

def test_load_kerala_aoi() -> None:
    aoi = load_aoi("configs/kerala_2018.yaml")
    assert aoi.name == "kerala_2018"
    assert aoi.pixel_size_m == 10
    assert len(aoi.bands) == 6
    w, s, e, n = aoi.bbox
    assert w < e and s < n, "bbox must be well-formed"
    # Kerala sanity check
    assert 75 < w < 77 and 8 < s < 12


def test_aoi_bbox_geojson_roundtrip() -> None:
    aoi = load_aoi("configs/kerala_2018.yaml")
    geom = aoi.bbox_geojson()
    assert geom["type"] == "Polygon"
    # 5 points: 4 corners + closing vertex
    assert len(geom["coordinates"][0]) == 5


# ----------------- Sen1Floods11 loader -----------------

def test_s2_dataset_len_and_item(sen1floods11_root: Path) -> None:
    ds = Sen1Floods11Dataset(sen1floods11_root, split="train", modality="s2")
    assert len(ds) == 2

    item = ds[0]
    assert set(item.keys()) >= {"image", "label", "chip_id"}
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].dtype == torch.float32
    assert item["image"].shape == (len(S2_BAND_INDICES), 64, 64)
    # reflectance is clipped to [0, 1]
    assert 0.0 <= item["image"].min().item() <= item["image"].max().item() <= 1.0

    assert item["label"].dtype == torch.long
    assert item["label"].shape == (64, 64)
    assert item["label"].min().item() >= LABEL_IGNORE_INDEX
    assert item["label"].max().item() <= 1


def test_s1_dataset_item_is_db(sen1floods11_root: Path) -> None:
    ds = Sen1Floods11Dataset(sen1floods11_root, split="train", modality="s1")
    item = ds[0]
    assert item["image"].shape == (2, 64, 64)
    # S1 converted to dB → typical range roughly [-40, 5]
    assert item["image"].min().item() >= -50
    assert item["image"].max().item() <= 10


def test_all_splits_nonempty(sen1floods11_root: Path) -> None:
    for split in ("train", "valid", "test", "bolivia"):
        ds = Sen1Floods11Dataset(sen1floods11_root, split=split, modality="s2")  # type: ignore[arg-type]
        assert len(ds) >= 1, f"split '{split}' unexpectedly empty"


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        Sen1Floods11Dataset(tmp_path / "does_not_exist", split="train", modality="s2")


# ----------------- Ground-truth rasterizer -----------------

def test_rasterize_flood_polygons(reference_raster: Path, tmp_path: Path) -> None:
    # Build a single polygon covering the left half of the reference raster.
    with rasterio.open(reference_raster) as src:
        crs = src.crs
        b = src.bounds
        half_x = (b.left + b.right) / 2.0
        poly = Polygon(
            [(b.left, b.bottom), (half_x, b.bottom), (half_x, b.top), (b.left, b.top)]
        )
    vec_path = tmp_path / "flood.geojson"
    gpd.GeoDataFrame(geometry=[poly], crs=crs).to_file(vec_path, driver="GeoJSON")

    out = rasterize_flood_polygons(
        vector_path=vec_path,
        reference_raster=reference_raster,
        out_path=tmp_path / "mask.tif",
    )

    with rasterio.open(out) as src:
        assert src.count == 1
        assert src.crs == crs
        mask = src.read(1)

    assert mask.shape == (128, 128)
    # Left half flooded, right half not → fraction ≈ 0.5.
    frac = flood_pixel_fraction(out)
    assert 0.45 < frac < 0.55


def test_rasterize_reprojects_crs(reference_raster: Path, tmp_path: Path) -> None:
    # Polygon in WGS84, reference in UTM — rasterizer must reproject.
    poly = Polygon([(76.0, 9.5), (77.0, 9.5), (77.0, 10.5), (76.0, 10.5)])
    vec_path = tmp_path / "wgs84.geojson"
    gpd.GeoDataFrame(geometry=[poly], crs="EPSG:4326").to_file(vec_path, driver="GeoJSON")

    out = rasterize_flood_polygons(
        vector_path=vec_path,
        reference_raster=reference_raster,
        out_path=tmp_path / "mask.tif",
    )
    with rasterio.open(out) as src:
        mask = src.read(1)
    # Polygon is far from the synthetic raster extent (UTM false origin) → 0% flood.
    # Just assert no crash and binary output.
    assert set(np.unique(mask).tolist()).issubset({0, 1})
