"""Google Earth Engine downloader for Sentinel-2 L2A scenes.

Produces a cloud-masked, reducer-aggregated composite for a given AOI and date
window, exported as a multi-band GeoTIFF. SCL-band masking is applied before
the composite reducer so residual cloud/shadow pixels do not bias reflectance.

Typical use:

    from src.data.aoi import load_aoi
    from src.data.gee_download import ee_initialize, download_s2_composite

    aoi = load_aoi("configs/kerala_2018.yaml")
    ee_initialize(project="dda-flood")
    download_s2_composite(aoi, window="pre",  out_path="data/raw/kerala_pre.tif")
    download_s2_composite(aoi, window="post", out_path="data/raw/kerala_post.tif")

CLI:

    python -m src.data.gee_download \
        --config configs/kerala_2018.yaml \
        --window both \
        --out-dir data/raw/kerala_2018 \
        --project dda-flood
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from src.data.aoi import AOIConfig, load_aoi
from src.utils.logging import get_logger

log = get_logger(__name__)

Window = Literal["pre", "post"]

# SCL (Scene Classification) class values to MASK OUT in Sentinel-2 L2A.
#   3  = cloud shadows
#   8  = cloud medium-probability
#   9  = cloud high-probability
#   10 = cirrus
#   11 = snow / ice
SCL_MASK_CLASSES: tuple[int, ...] = (3, 8, 9, 10, 11)


def ee_initialize(project: str | None = None) -> None:
    """Authenticate (if needed) and initialize the Earth Engine client."""
    import ee  # noqa: PLC0415

    try:
        ee.Initialize(project=project)
    except Exception:  # noqa: BLE001
        log.info("Earth Engine not initialized — running ee.Authenticate()")
        ee.Authenticate()
        ee.Initialize(project=project)
    log.info("Earth Engine initialized (project=%s)", project)


def _mask_s2_scl(image):  # type: ignore[no-untyped-def]
    """Apply SCL-based cloud/shadow mask to a Sentinel-2 L2A image."""
    import ee  # noqa: PLC0415

    scl = image.select("SCL")
    mask = ee.Image.constant(1)
    for cls in SCL_MASK_CLASSES:
        mask = mask.And(scl.neq(cls))
    return image.updateMask(mask)


def _apply_reducer(collection, reducer_name: str):  # type: ignore[no-untyped-def]
    import ee  # noqa: PLC0415

    reducer = {
        "median": ee.Reducer.median(),
        "mean": ee.Reducer.mean(),
        "min": ee.Reducer.min(),
        "max": ee.Reducer.max(),
    }.get(reducer_name)
    if reducer is None:
        raise ValueError(f"Unknown reducer '{reducer_name}'")
    return collection.reduce(reducer)


def build_s2_composite(aoi: AOIConfig, window: Window):  # type: ignore[no-untyped-def]
    """Build a cloud-masked Sentinel-2 L2A composite as an ``ee.Image``."""
    import ee  # noqa: PLC0415

    start, end = (aoi.pre_event if window == "pre" else aoi.post_event).as_tuple()
    geom = aoi.to_ee_geometry()

    collection = (
        ee.ImageCollection(aoi.collections["s2_l2a"])
        .filterBounds(geom)
        .filterDate(start, end)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", aoi.max_cloud_cover_pct))
        .map(_mask_s2_scl)
        .select(list(aoi.bands))
    )

    count = collection.size().getInfo()
    log.info("Window '%s' [%s → %s]: %d scenes after filtering", window, start, end, count)
    if count == 0:
        raise RuntimeError(
            f"No Sentinel-2 scenes returned for {window} window {start}→{end}. "
            f"Relax max_cloud_cover_pct or widen the date range."
        )

    composite = _apply_reducer(collection, aoi.composite_reducer)

    # `.reduce(median())` suffixes band names with "_median"; rename back so
    # downstream code can expect ["B2", "B3", ...].
    new_names = [f"{b}_{aoi.composite_reducer}" for b in aoi.bands]
    composite = composite.select(new_names, list(aoi.bands))

    # Clip to exact AOI rectangle for export.
    return composite.clip(geom).toFloat()


def download_s2_composite(
    aoi: AOIConfig,
    window: Window,
    out_path: str | Path,
    max_pixels: int = int(1e9),
) -> Path:
    """Download a Sentinel-2 composite to a local GeoTIFF.

    Uses ``geemap.ee_export_image`` which hits the SYNCHRONOUS GEE download
    endpoint. For AOIs up to ~1° x 1° at 10 m this completes in a minute or two;
    larger AOIs should instead use ``ee.batch.Export.image.toDrive``.
    """
    import geemap  # noqa: PLC0415

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = build_s2_composite(aoi, window)
    log.info("Exporting composite → %s", out_path)
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
    p = argparse.ArgumentParser(description="Download Sentinel-2 L2A composites via GEE.")
    p.add_argument("--config", required=True, help="Path to AOI YAML config.")
    p.add_argument(
        "--window",
        choices=["pre", "post", "both"],
        default="both",
        help="Which date window to download.",
    )
    p.add_argument("--out-dir", default="data/raw", help="Output directory.")
    p.add_argument("--project", default=None, help="GCP project id for ee.Initialize.")
    args = p.parse_args()

    aoi = load_aoi(args.config)
    ee_initialize(project=args.project)

    out_dir = Path(args.out_dir)
    windows: list[Window] = ["pre", "post"] if args.window == "both" else [args.window]
    for w in windows:
        out = out_dir / f"{aoi.name}_{w}.tif"
        download_s2_composite(aoi, w, out)


if __name__ == "__main__":
    _cli()
