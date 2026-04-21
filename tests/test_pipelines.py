"""Integration test for src.pipelines.classical_baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from src.pipelines.classical_baseline import run_classical_baseline


def _write_stack(path: Path, stack: np.ndarray) -> None:
    profile = {
        "driver": "GTiff",
        "dtype": stack.dtype,
        "count": stack.shape[0],
        "height": stack.shape[1],
        "width": stack.shape[2],
        "crs": "EPSG:32643",
        "transform": from_origin(500_000, 1_100_000, 10.0, 10.0),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(stack)


def test_classical_baseline_endtoend(tmp_path: Path) -> None:
    """Build synthetic pre (dry) and post (flooded) scenes and verify the
    pipeline detects the flood pixels."""
    rng = np.random.default_rng(42)
    h, w = 48, 48

    # Pre: everything is vegetation.
    pre = np.broadcast_to(
        np.array([0.05, 0.08, 0.04, 0.45, 0.25, 0.15], dtype=np.float32)[:, None, None],
        (6, h, w),
    ).copy()
    pre += rng.normal(0, 0.005, pre.shape).astype(np.float32)

    # Post: left half flooded (water spectrum), right half unchanged.
    post = pre.copy()
    water = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01], dtype=np.float32)
    post[:, :, : w // 2] = water[:, None, None] + rng.normal(
        0, 0.005, (6, h, w // 2)
    ).astype(np.float32)
    post = np.clip(post, 0, 1)

    pre_path = tmp_path / "pre.tif"
    post_path = tmp_path / "post.tif"
    out_path = tmp_path / "flood.tif"
    _write_stack(pre_path, pre)
    _write_stack(post_path, post)

    result = run_classical_baseline(pre_path, post_path, out_path)

    with rasterio.open(out_path) as dst:
        mask = dst.read(1).astype(bool)

    # Left half ~flooded; expect detection > 30% and < 70% of AOI.
    assert 0.30 < result.flood_fraction < 0.70

    # Spatial check: pipeline should detect substantially more water on the
    # flooded side than the dry side.
    left_mean = mask[:, : w // 2].mean()
    right_mean = mask[:, w // 2 :].mean()
    assert left_mean > right_mean + 0.5

    # JSON sidecar exists and parses.
    sidecar = out_path.with_suffix(".json")
    assert sidecar.exists()
    stats = json.loads(sidecar.read_text())
    assert stats["method"] == "mndwi_otsu_intersect"
    assert 0.30 < stats["flood_fraction"] < 0.70
