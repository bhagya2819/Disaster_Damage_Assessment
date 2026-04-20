"""Build the Kerala 2018 ground-truth flood mask.

UNOSAT publishes flood-extent polygons for major events via ReliefWeb / their
Humanitarian Data Exchange mirror. The canonical Kerala 2018 product is
"UNOSAT Kerala Flood Waters 20180822" (vector).

This module rasterises a flood polygon shapefile / GeoPackage to a binary 10 m
mask aligned with a reference Sentinel-2 GeoTIFF, so the resulting mask is
pixel-for-pixel comparable with model predictions.

Typical use::

    from src.data.aoi import load_aoi
    from src.data.ground_truth import rasterize_flood_polygons

    aoi = load_aoi("configs/kerala_2018.yaml")
    rasterize_flood_polygons(
        vector_path="data/gt/unosat_kerala_2018.shp",
        reference_raster="data/raw/kerala_2018/kerala_2018_post.tif",
        out_path="data/gt/kerala_gt.tif",
    )

If you do not have the UNOSAT shapefile, see the docstring of
``rasterize_flood_polygons`` for retrieval instructions.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from src.utils.logging import get_logger

log = get_logger(__name__)


def rasterize_flood_polygons(
    vector_path: str | Path,
    reference_raster: str | Path,
    out_path: str | Path,
    burn_value: int = 1,
    background_value: int = 0,
    dtype: str = "uint8",
) -> Path:
    """Rasterize flood polygons onto the grid of a reference GeoTIFF.

    Parameters
    ----------
    vector_path
        Shapefile / GeoPackage / GeoJSON of flood polygons. Any CRS — it will
        be reprojected to the reference raster's CRS automatically.
    reference_raster
        A Sentinel-2 composite that defines the target grid (transform, CRS,
        shape). Typically the post-event raster from ``gee_download``.
    out_path
        Where to write the binary mask GeoTIFF.
    burn_value
        Pixel value for "flood" (default 1).
    background_value
        Pixel value for "non-flood" (default 0).
    dtype
        Output raster dtype. 'uint8' is fine for binary masks.

    Returns
    -------
    Path to the written mask.

    Notes
    -----
    If you lack the UNOSAT polygon file, retrieve it from:
      * UNOSAT portal: https://unosat.org/products/2728
      * HDX mirror: https://data.humdata.org/ (search "Kerala flood 2018 UNOSAT")
    and save it as ``data/gt/unosat_kerala_2018.shp``.
    """
    vector_path = Path(vector_path)
    reference_raster = Path(reference_raster)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reading vector: %s", vector_path)
    gdf = gpd.read_file(vector_path)
    if gdf.empty:
        raise ValueError(f"{vector_path} contains 0 features.")

    with rasterio.open(reference_raster) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)
        ref_profile = ref.profile.copy()

    if gdf.crs != ref_crs:
        log.info("Reprojecting polygons %s → %s", gdf.crs, ref_crs)
        gdf = gdf.to_crs(ref_crs)

    shapes = ((geom, burn_value) for geom in gdf.geometry if geom is not None and not geom.is_empty)
    mask = rasterize(
        shapes=shapes,
        out_shape=ref_shape,
        transform=ref_transform,
        fill=background_value,
        dtype=dtype,
        all_touched=False,
    )

    ref_profile.update(
        count=1,
        dtype=dtype,
        nodata=None,
        compress="lzw",
    )

    with rasterio.open(out_path, "w", **ref_profile) as dst:
        dst.write(mask, 1)

    pct_flood = float((mask == burn_value).mean() * 100)
    log.info("Wrote mask %s  flood=%.2f%% of AOI", out_path, pct_flood)
    return out_path


def flood_pixel_fraction(mask_path: str | Path, flood_value: int = 1) -> float:
    """Return the fraction of valid pixels labelled as flood."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        nodata = src.nodata
    if nodata is not None:
        valid = arr[arr != nodata]
    else:
        valid = arr
    if valid.size == 0:
        return float("nan")
    return float(np.mean(valid == flood_value))
