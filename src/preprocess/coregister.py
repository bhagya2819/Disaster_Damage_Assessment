"""Raster coregistration utilities.

GEE exports pre- and post-event composites on the same grid if the request is
issued with the same ``region`` and ``crs`` parameters — which our downloader
does. However, when users combine ad-hoc `.SAFE` tiles or products from
different sources, alignment is not guaranteed.

This module provides a one-call helper that reprojects a source raster onto
the grid (CRS + transform + shape) of a reference raster.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

from src.utils.logging import get_logger

log = get_logger(__name__)


def coregister(
    source_path: str | Path,
    reference_path: str | Path,
    out_path: str | Path,
    resampling: Resampling = Resampling.bilinear,
) -> Path:
    """Reproject ``source_path`` onto the grid of ``reference_path``.

    Writes a new GeoTIFF at ``out_path`` with the reference CRS, transform,
    width and height. Band count and dtype are inherited from the source.
    """
    source_path, reference_path, out_path = Path(source_path), Path(reference_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(reference_path) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_shape = (ref.height, ref.width)

    with rasterio.open(source_path) as src:
        dst_profile = src.profile.copy()
        dst_profile.update(
            crs=ref_crs,
            transform=ref_transform,
            width=ref_shape[1],
            height=ref_shape[0],
            compress="lzw",
        )

        with rasterio.open(out_path, "w", **dst_profile) as dst:
            for b in range(1, src.count + 1):
                src_band = src.read(b)
                dst_band = np.zeros(ref_shape, dtype=src_band.dtype)
                reproject(
                    source=src_band,
                    destination=dst_band,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=resampling,
                )
                dst.write(dst_band, b)

    log.info("Coregistered %s → %s (grid of %s)", source_path, out_path, reference_path)
    return out_path


def assert_aligned(path_a: str | Path, path_b: str | Path) -> None:
    """Raise if two rasters do not share CRS, transform and shape."""
    with rasterio.open(path_a) as a, rasterio.open(path_b) as b:
        assert a.crs == b.crs, f"CRS mismatch: {a.crs} vs {b.crs}"
        assert a.transform == b.transform, f"Transform mismatch:\n{a.transform}\n{b.transform}"
        assert (a.height, a.width) == (b.height, b.width), (
            f"Shape mismatch: ({a.height},{a.width}) vs ({b.height},{b.width})"
        )
