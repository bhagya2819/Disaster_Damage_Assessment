"""One-off CLI to rasterize the UNOSAT Kerala 2018 flood shapefile.

Reads the config at ``configs/kerala_2018.yaml`` to get the reference raster
and output paths, then delegates to
``src.data.ground_truth.rasterize_flood_polygons``.

Usage:
    python scripts/build_kerala_ground_truth.py \
        --shapefile data/gt/unosat_kerala_2018.shp \
        --reference data/raw/kerala_2018/kerala_2018_post.tif

If no ``--output`` is given, writes to the path in the YAML config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src` importable when invoked as `python scripts/build_kerala_ground_truth.py`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.aoi import load_aoi  # noqa: E402
from src.data.ground_truth import rasterize_flood_polygons  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/kerala_2018.yaml")
    p.add_argument("--shapefile", required=True, help="UNOSAT polygon file.")
    p.add_argument("--reference", required=True, help="Post-event Sentinel-2 GeoTIFF.")
    p.add_argument("--output", default=None, help="Output GeoTIFF path (defaults to config).")
    args = p.parse_args()

    aoi = load_aoi(args.config)
    out = Path(args.output or aoi.ground_truth.get("rasterized_mask", "data/gt/kerala_gt.tif"))

    rasterize_flood_polygons(
        vector_path=args.shapefile,
        reference_raster=args.reference,
        out_path=out,
    )


if __name__ == "__main__":
    main()
