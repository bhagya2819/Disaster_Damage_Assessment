"""Sentinel-1 GRD downloader — companion to src/data/gee_download.py.

Downloads the VV and VH bands of a Sentinel-1 IW GRD stack over the AOI
and date window, reducing via median. The Sentinel-1 bands in GEE are
already calibrated γ⁰ in linear units, so :func:`src.dip.sar.to_db` must be
applied downstream for dB-scale analysis.

Usage:
    from src.data.aoi import load_aoi
    from src.data.sar_download import ee_initialize, download_s1_composite

    aoi = load_aoi("configs/kerala_2018.yaml")
    ee_initialize(project="dda-flood")
    download_s1_composite(aoi, window="post", out_path="data/raw/kerala_2018_s1_post.tif")
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

# Make `src` importable when invoked as `python scripts/foo.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.aoi import AOIConfig, load_aoi  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

log = get_logger(__name__)

Window = Literal["pre", "post"]


def ee_initialize(project: str | None = None) -> None:
    """Lazy wrapper over :func:`src.data.gee_download.ee_initialize`."""
    from src.data.gee_download import ee_initialize as _init  # noqa: PLC0415

    _init(project=project)


def build_s1_composite(aoi: AOIConfig, window: Window):  # type: ignore[no-untyped-def]
    """Build a median composite of Sentinel-1 IW GRD (VV + VH) over the window."""
    import ee  # noqa: PLC0415

    start, end = (aoi.pre_event if window == "pre" else aoi.post_event).as_tuple()
    geom = aoi.to_ee_geometry()

    collection = (
        ee.ImageCollection(aoi.collections.get("s1_grd", "COPERNICUS/S1_GRD"))
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .select(["VV", "VH"])
    )

    count = collection.size().getInfo()
    log.info("S1 window '%s' [%s → %s]: %d scenes", window, start, end, count)
    if count == 0:
        raise RuntimeError(
            f"No Sentinel-1 GRD scenes for {window} window {start}→{end}. "
            f"Try widening the date range."
        )

    composite = collection.reduce(ee.Reducer.median())
    # Rename bands from VV_median / VH_median back to VV / VH.
    composite = composite.select(["VV_median", "VH_median"], ["VV", "VH"])
    return composite.clip(geom).toFloat()


def download_s1_composite(
    aoi: AOIConfig,
    window: Window,
    out_path: str | Path,
) -> Path:
    """Download a SAR composite GeoTIFF to ``out_path``."""
    import geemap  # noqa: PLC0415

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = build_s1_composite(aoi, window)
    log.info("Exporting SAR composite → %s", out_path)
    geemap.ee_export_image(
        img,
        filename=str(out_path),
        scale=aoi.pixel_size_m,
        region=aoi.to_ee_geometry(),
        crs=aoi.crs,
        file_per_band=False,
    )
    if not out_path.exists():
        raise RuntimeError(f"Export finished but {out_path} not found on disk")
    log.info("✓ Wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


def _cli() -> None:
    p = argparse.ArgumentParser(description="Download Sentinel-1 GRD composites via GEE.")
    p.add_argument("--config", required=True)
    p.add_argument("--window", choices=["pre", "post", "both"], default="both")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--project", default=None)
    args = p.parse_args()

    aoi = load_aoi(args.config)
    ee_initialize(project=args.project)
    out_dir = Path(args.out_dir)
    windows: list[Window] = ["pre", "post"] if args.window == "both" else [args.window]
    for w in windows:
        out = out_dir / f"{aoi.name}_s1_{w}.tif"
        download_s1_composite(aoi, w, out)


if __name__ == "__main__":
    _cli()
