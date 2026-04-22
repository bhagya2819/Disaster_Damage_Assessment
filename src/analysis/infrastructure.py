"""Infrastructure impact under a flood mask.

Given a flood-mask GeoTIFF and an AOI, fetch OpenStreetMap roads and
buildings via ``osmnx``, clip them to the AOI, rasterise onto the mask's
grid, and report:

- total km of roads affected (by road class).
- total building footprint area and count under flooded pixels.

Both outputs are returned as tidy pandas DataFrames suitable for inclusion
in the PDF damage report.

Note
----
``osmnx`` queries OSM's Overpass API at call time. Runs on Colab; may hit
rate limits on repeated invocation from the same IP. Cache the GeoDataFrames
to disk with ``.to_file(path, driver='GeoJSON')`` if you need repeated runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import box

from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class InfrastructureImpact:
    roads_km_flooded: float
    roads_km_total: float
    buildings_flooded_count: int
    buildings_total_count: int
    buildings_flooded_area_m2: float
    buildings_total_area_m2: float
    roads_by_class: pd.DataFrame       # per-highway-class km flooded
    aoi_area_km2: float

    def as_summary(self) -> dict[str, float]:
        return {
            "roads_km_flooded": self.roads_km_flooded,
            "roads_km_total": self.roads_km_total,
            "roads_fraction_flooded": (
                self.roads_km_flooded / self.roads_km_total if self.roads_km_total else 0.0
            ),
            "buildings_flooded_count": self.buildings_flooded_count,
            "buildings_total_count": self.buildings_total_count,
            "buildings_fraction_flooded": (
                self.buildings_flooded_count / self.buildings_total_count
                if self.buildings_total_count else 0.0
            ),
            "buildings_flooded_area_m2": self.buildings_flooded_area_m2,
            "buildings_total_area_m2": self.buildings_total_area_m2,
        }


def _read_mask(mask_path: str | Path, flood_value: int = 1) -> tuple[np.ndarray, object, object]:
    """Read a flood-mask GeoTIFF, return (bool mask, transform, CRS)."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs
    return (arr == flood_value), transform, crs


def _aoi_from_mask(mask_path: str | Path) -> tuple[tuple[float, float, float, float], object]:
    """Return (bbox_wgs84, native_crs) for an input raster."""
    with rasterio.open(mask_path) as src:
        bounds = src.bounds
        crs = src.crs
    # Reproject to WGS84 for osmnx queries.
    gdf = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=crs)
    gdf_wgs = gdf.to_crs("EPSG:4326")
    b = gdf_wgs.total_bounds  # [minx, miny, maxx, maxy]
    return (float(b[0]), float(b[1]), float(b[2]), float(b[3])), crs


# ---------------------------------------------------------------------------
# OSM fetch (thin wrappers so tests can monkey-patch)
# ---------------------------------------------------------------------------

def fetch_roads(bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Fetch drivable road network for the AOI as a GeoDataFrame."""
    try:
        import osmnx as ox  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError("osmnx required; `pip install osmnx`.") from e

    # osmnx 2.x signature: bbox=(minx, miny, maxx, maxy)
    minx, miny, maxx, maxy = bbox_wgs84
    graph = ox.graph_from_bbox(bbox=(minx, miny, maxx, maxy), network_type="drive", simplify=True)
    gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    return gdf.reset_index()


def fetch_buildings(bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Fetch OSM building polygons for the AOI as a GeoDataFrame."""
    try:
        import osmnx as ox  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError("osmnx required; `pip install osmnx`.") from e

    minx, miny, maxx, maxy = bbox_wgs84
    tags = {"building": True}
    gdf = ox.features_from_bbox(bbox=(minx, miny, maxx, maxy), tags=tags)
    return gdf.reset_index()


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def roads_flooded(
    mask_path: str | Path,
    roads_gdf: gpd.GeoDataFrame | None = None,
    flood_value: int = 1,
) -> tuple[float, float, pd.DataFrame]:
    """Compute km of roads under flood.

    Returns (roads_km_flooded, roads_km_total, by_class_df).
    """
    mask, transform, crs = _read_mask(mask_path, flood_value=flood_value)
    h, w = mask.shape

    if roads_gdf is None:
        bbox, _ = _aoi_from_mask(mask_path)
        roads_gdf = fetch_roads(bbox)

    if roads_gdf.empty:
        return 0.0, 0.0, pd.DataFrame(columns=["highway", "km_total", "km_flooded"])

    roads_proj = roads_gdf.to_crs(crs)

    # Total km by class.
    roads_proj = roads_proj.copy()
    roads_proj["length_m"] = roads_proj.geometry.length
    total_km = float(roads_proj["length_m"].sum() / 1000.0)

    # Compute flooded length by intersecting each road with the flooded-pixel
    # polygon set. For efficiency we rasterise a road presence mask and
    # then do pixel-wise AND with the flood mask.
    road_mask = rasterize(
        [(geom, 1) for geom in roads_proj.geometry if geom is not None and not geom.is_empty],
        out_shape=(h, w),
        transform=transform,
        dtype="uint8",
        all_touched=True,
    ).astype(bool)

    flooded_road_pixels = int((road_mask & mask).sum())
    px_size_m = abs(transform.a)
    # Approximate flooded km via pixel count × pixel size.
    flooded_km = float(flooded_road_pixels * px_size_m / 1000.0)

    # Per-class breakdown: approximate by proportional split — rasterise one
    # class at a time and count.
    rows = []
    by_class = (
        roads_proj.groupby(roads_proj.get("highway", "unknown").astype(str))
        if "highway" in roads_proj.columns
        else [("all", roads_proj)]
    )
    for cls, sub in by_class:
        sub_gdf = sub if hasattr(sub, "geometry") else roads_proj
        km_tot = float(sub_gdf["length_m"].sum() / 1000.0) if len(sub_gdf) else 0.0
        if sub_gdf.empty:
            rows.append({"highway": cls, "km_total": km_tot, "km_flooded": 0.0})
            continue
        rm = rasterize(
            [(g, 1) for g in sub_gdf.geometry if g is not None and not g.is_empty],
            out_shape=(h, w), transform=transform, dtype="uint8", all_touched=True,
        ).astype(bool)
        km_fl = float((rm & mask).sum() * px_size_m / 1000.0)
        rows.append({"highway": str(cls), "km_total": km_tot, "km_flooded": km_fl})

    df = pd.DataFrame(rows).sort_values("km_flooded", ascending=False).reset_index(drop=True)
    return flooded_km, total_km, df


def buildings_flooded(
    mask_path: str | Path,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    flood_value: int = 1,
) -> tuple[int, int, float, float]:
    """Return (flooded_count, total_count, flooded_area_m2, total_area_m2)."""
    mask, transform, crs = _read_mask(mask_path, flood_value=flood_value)

    if buildings_gdf is None:
        bbox, _ = _aoi_from_mask(mask_path)
        buildings_gdf = fetch_buildings(bbox)

    if buildings_gdf.empty:
        return 0, 0, 0.0, 0.0

    b = buildings_gdf.to_crs(crs).copy()
    # Keep only polygon / multipolygon — lines/points can appear in OSM dumps.
    b = b[b.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)
    if b.empty:
        return 0, 0, 0.0, 0.0

    b["area_m2"] = b.geometry.area
    total_count = int(len(b))
    total_area = float(b["area_m2"].sum())

    # A building is "flooded" if ANY of its pixels overlap flooded pixels.
    # We rasterise per-building using the feature id baked in as raster value.
    b = b.reset_index(drop=True)
    b["_idx"] = np.arange(len(b), dtype=np.int32) + 1
    building_id_raster = rasterize(
        [(geom, int(i)) for geom, i in zip(b.geometry, b["_idx"], strict=True)
         if geom is not None and not geom.is_empty],
        out_shape=mask.shape, transform=transform, dtype="int32", all_touched=False,
    )
    flooded_ids = set(np.unique(building_id_raster[mask]).tolist()) - {0}
    flooded_count = len(flooded_ids)

    # Sum flooded-building areas (full footprint, not just the flooded portion).
    flooded_area = float(b.loc[b["_idx"].isin(flooded_ids), "area_m2"].sum())
    return flooded_count, total_count, flooded_area, total_area


def compute(
    mask_path: str | Path,
    roads_gdf: gpd.GeoDataFrame | None = None,
    buildings_gdf: gpd.GeoDataFrame | None = None,
    flood_value: int = 1,
) -> InfrastructureImpact:
    """Top-level convenience — runs both road and building analyses."""
    mask_path = Path(mask_path)

    with rasterio.open(mask_path) as src:
        h, w = src.height, src.width
        px_m2 = abs(src.transform.a) * abs(src.transform.e)
    aoi_km2 = float(h * w * px_m2 / 1e6)

    log.info("Infrastructure analysis on %s (AOI %.2f km²)", mask_path, aoi_km2)
    rk_fl, rk_tot, rby = roads_flooded(mask_path, roads_gdf=roads_gdf, flood_value=flood_value)
    bc_fl, bc_tot, ba_fl, ba_tot = buildings_flooded(
        mask_path, buildings_gdf=buildings_gdf, flood_value=flood_value
    )

    return InfrastructureImpact(
        roads_km_flooded=rk_fl,
        roads_km_total=rk_tot,
        buildings_flooded_count=bc_fl,
        buildings_total_count=bc_tot,
        buildings_flooded_area_m2=ba_fl,
        buildings_total_area_m2=ba_tot,
        roads_by_class=rby,
        aoi_area_km2=aoi_km2,
    )


__all__ = [
    "InfrastructureImpact",
    "buildings_flooded",
    "compute",
    "fetch_buildings",
    "fetch_roads",
    "roads_flooded",
]
