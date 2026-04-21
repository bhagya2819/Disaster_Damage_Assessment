"""End-to-end classical DIP baseline: pre + post Sentinel-2 → flood mask.

Pipeline:

    pre, post  ──▶  MNDWI_pre, MNDWI_post
                        │
                        ├──▶  MNDWI_post thresholded (Otsu)     = raw water mask
                        │
                        └──▶  ΔMNDWI = post − pre thresholded   = change mask
                                 │
                                 └──▶  intersect → flood-only mask
                                         │
                                         └──▶ morphology.clean  = output

Output is written as a binary (uint8) GeoTIFF on the post-event grid, plus a
sidecar JSON with summary statistics (flooded area km², threshold values,
method names) used later by the report generator.

Usage:
    python -m src.pipelines.classical_baseline \
        --pre  data/raw/kerala_2018/kerala_2018_pre.tif \
        --post data/raw/kerala_2018/kerala_2018_post.tif \
        --out  data/processed/kerala_2018/flood_mask_classical.tif
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio

from src.dip import change_detection, indices, morphology, thresholding
from src.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ClassicalResult:
    flood_fraction: float        # [0, 1]
    flood_area_km2: float
    mndwi_threshold: float
    delta_mndwi_threshold: float
    method: str = "mndwi_otsu_intersect"


def _read_stack(path: str | Path) -> tuple[np.ndarray, rasterio.profiles.Profile, float]:
    """Return (stack (C,H,W), profile, pixel_area_m2)."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        profile = src.profile.copy()
        # Sentinel-2 DDA composites export in metric CRS → resolution in metres.
        px_w, px_h = abs(src.transform.a), abs(src.transform.e)
        return arr, profile, float(px_w * px_h)


def run_classical_baseline(
    pre_path: str | Path,
    post_path: str | Path,
    out_path: str | Path,
    opening_radius: int = 1,
    closing_radius: int = 1,
    min_object_area: int = 25,
    min_hole_area: int = 25,
) -> ClassicalResult:
    """Run the full classical pipeline and persist artefacts."""
    pre_path, post_path, out_path = Path(pre_path), Path(post_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Reading pre/post stacks")
    pre, _, _ = _read_stack(pre_path)
    post, profile, px_area = _read_stack(post_path)

    if pre.shape != post.shape:
        raise ValueError(
            f"pre/post shape mismatch {pre.shape} vs {post.shape}. "
            f"Run src.preprocess.coregister first."
        )

    log.info("Computing MNDWI for pre and post")
    mndwi_pre = indices.mndwi(pre)
    mndwi_post = indices.mndwi(post)

    log.info("Otsu-thresholding MNDWI_post")
    water_res = thresholding.otsu(mndwi_post)
    log.info("  water threshold = %.4f", water_res.value)

    log.info("Otsu-thresholding ΔMNDWI to isolate NEW water")
    delta = change_detection.image_difference(mndwi_pre, mndwi_post)
    change_res = thresholding.otsu(delta)
    log.info("  change threshold = %.4f", change_res.value)

    log.info("Intersecting water ∩ change and cleaning")
    raw_mask = water_res.mask & change_res.mask
    flood_mask = morphology.clean(
        raw_mask,
        opening_radius=opening_radius,
        closing_radius=closing_radius,
        min_object_area=min_object_area,
        min_hole_area=min_hole_area,
    )

    flood_fraction = float(flood_mask.mean())
    flooded_km2 = flood_fraction * flood_mask.size * px_area / 1e6

    profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(flood_mask.astype(np.uint8), 1)
    log.info("Wrote flood mask → %s", out_path)

    result = ClassicalResult(
        flood_fraction=flood_fraction,
        flood_area_km2=flooded_km2,
        mndwi_threshold=water_res.value,
        delta_mndwi_threshold=change_res.value,
    )
    sidecar = out_path.with_suffix(".json")
    sidecar.write_text(json.dumps(asdict(result), indent=2))
    log.info("Stats → %s (flood=%.2f%%, area=%.2f km²)", sidecar, flood_fraction * 100, flooded_km2)
    return result


def _cli() -> None:
    p = argparse.ArgumentParser(description="Classical DIP flood-mask baseline.")
    p.add_argument("--pre", required=True, help="Pre-event Sentinel-2 6-band GeoTIFF.")
    p.add_argument("--post", required=True, help="Post-event Sentinel-2 6-band GeoTIFF.")
    p.add_argument("--out", required=True, help="Output binary flood mask GeoTIFF.")
    p.add_argument("--opening-radius", type=int, default=1)
    p.add_argument("--closing-radius", type=int, default=1)
    p.add_argument("--min-object-area", type=int, default=25)
    p.add_argument("--min-hole-area", type=int, default=25)
    args = p.parse_args()

    run_classical_baseline(
        pre_path=args.pre,
        post_path=args.post,
        out_path=args.out,
        opening_radius=args.opening_radius,
        closing_radius=args.closing_radius,
        min_object_area=args.min_object_area,
        min_hole_area=args.min_hole_area,
    )


if __name__ == "__main__":
    _cli()
