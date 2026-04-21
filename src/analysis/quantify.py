"""Quantify flood extent and break it down by land cover.

Two public entry points:

- :func:`area_summary`            — top-line area statistics for a flood mask.
- :func:`landcover_breakdown`     — tabular per-land-cover-class area affected,
  using an ESA WorldCover raster (or any integer-coded land-cover raster
  aligned to the same grid).

All outputs are plain Python / pandas, so the Streamlit app and the PDF
report generator can both consume them without reaching into raster code.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# ESA WorldCover v200 class codes (10 m, 2021). Names per the ESA product spec.
WORLDCOVER_CLASSES: dict[int, str] = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}


@dataclass(frozen=True)
class AreaSummary:
    flooded_km2: float
    total_km2: float
    flooded_fraction: float
    pixel_area_m2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "flooded_km2": self.flooded_km2,
            "total_km2": self.total_km2,
            "flooded_fraction": self.flooded_fraction,
            "pixel_area_m2": self.pixel_area_m2,
        }


def _pixel_area_m2(transform) -> float:  # type: ignore[no-untyped-def]
    """Pixel ground area in square metres, assuming a projected CRS."""
    return abs(transform.a) * abs(transform.e)


def area_summary(mask_path: str | Path, flood_value: int = 1) -> AreaSummary:
    """Compute flooded area (km²), total AOI area (km²) and fraction."""
    mask_path = Path(mask_path)
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        transform = src.transform

    px_area = _pixel_area_m2(transform)
    total_px = arr.size
    flooded_px = int(np.sum(arr == flood_value))

    flooded_km2 = flooded_px * px_area / 1e6
    total_km2 = total_px * px_area / 1e6
    frac = float(flooded_km2 / total_km2) if total_km2 > 0 else 0.0

    return AreaSummary(
        flooded_km2=float(flooded_km2),
        total_km2=float(total_km2),
        flooded_fraction=frac,
        pixel_area_m2=float(px_area),
    )


def landcover_breakdown(
    mask_path: str | Path,
    worldcover_path: str | Path,
    flood_value: int = 1,
) -> pd.DataFrame:
    """Per-land-cover-class flooded area breakdown.

    The two rasters **must already be pixel-aligned** (same CRS, transform,
    and shape). Coregister upstream with ``src.preprocess.coregister`` if
    they aren't.

    Returns a ``DataFrame`` with columns:
        class_code, class_name, total_px, flooded_px, flooded_km2,
        class_fraction_flooded, share_of_flood.
    """
    mask_path = Path(mask_path)
    worldcover_path = Path(worldcover_path)

    with rasterio.open(mask_path) as m, rasterio.open(worldcover_path) as lc:
        if (m.height, m.width) != (lc.height, lc.width):
            raise ValueError(
                "mask and WorldCover must share shape; "
                f"got mask={m.shape} vs worldcover={lc.shape}"
            )
        if m.crs != lc.crs:
            raise ValueError(f"CRS mismatch: {m.crs} vs {lc.crs}")
        mask = m.read(1)
        lc_arr = lc.read(1)
        px_area = _pixel_area_m2(m.transform)

    flooded_mask = mask == flood_value
    total_flood_px = int(flooded_mask.sum())

    rows = []
    classes = np.unique(lc_arr)
    for c in classes:
        class_px = int((lc_arr == c).sum())
        class_flood_px = int(((lc_arr == c) & flooded_mask).sum())
        if class_px == 0:
            continue
        rows.append({
            "class_code": int(c),
            "class_name": WORLDCOVER_CLASSES.get(int(c), f"unknown_{int(c)}"),
            "total_px": class_px,
            "flooded_px": class_flood_px,
            "flooded_km2": class_flood_px * px_area / 1e6,
            "class_fraction_flooded": class_flood_px / class_px if class_px > 0 else 0.0,
            "share_of_flood": class_flood_px / total_flood_px if total_flood_px > 0 else 0.0,
        })

    df = pd.DataFrame(rows).sort_values("flooded_km2", ascending=False).reset_index(drop=True)
    return df


def population_exposed(
    mask_path: str | Path,
    population_path: str | Path,
    flood_value: int = 1,
) -> float:
    """Sum the population raster over flooded pixels.

    WorldPop is distributed as people-per-pixel at 100 m. If the mask is at
    10 m, the population raster must be pre-resampled to 10 m (typically via
    ``preprocess.coregister.coregister`` with ``Resampling.sum`` or
    ``Resampling.average`` / mean + area ratio).
    """
    with rasterio.open(mask_path) as m, rasterio.open(population_path) as pop:
        if (m.height, m.width) != (pop.height, pop.width):
            raise ValueError(
                "mask and population raster must share shape; coregister first."
            )
        mask = m.read(1)
        p = pop.read(1).astype(np.float64)

    # Treat negative / nodata pixels as zero.
    p = np.where(np.isfinite(p) & (p >= 0), p, 0.0)
    return float(p[mask == flood_value].sum())


__all__ = [
    "AreaSummary",
    "WORLDCOVER_CLASSES",
    "area_summary",
    "landcover_breakdown",
    "population_exposed",
]
