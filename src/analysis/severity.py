"""Damage severity classification from a binary flood mask.

Given a high-resolution flood mask (typically at 10 m Sentinel-2 pixel size),
we partition the AOI into a regular grid of cells and assign each cell a
severity class in {None, Low, Moderate, Severe} based on the fraction of
flooded pixels inside it. A second, optional depth-proxy signal can be
supplied (e.g. normalised NDWI intensity or water-occurrence persistence
from JRC); when present, its cell-mean is used as a secondary tie-breaker.

Defaults are tuned for a 1 km × 1 km cell at 10 m pixels (100 × 100 = 10 000
pixels per cell):

| Class     | Flooded fraction |
|-----------|------------------|
| None      | < 5 %            |
| Low       | 5 – 15 %         |
| Moderate  | 15 – 40 %        |
| Severe    | ≥ 40 %           |

These thresholds follow typical humanitarian rapid-damage-assessment tiers
(UNOSAT, Copernicus EMS); the caller may override via :class:`SeverityConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import numpy as np
import rasterio


class Severity(IntEnum):
    NONE = 0
    LOW = 1
    MODERATE = 2
    SEVERE = 3


@dataclass(frozen=True)
class SeverityConfig:
    """Tunable thresholds and cell-size settings."""

    # Cell side in pixels. 100 px @ 10 m = 1 km.
    cell_px: int = 100

    # Upper bounds of flooded fraction for each non-severe class.
    none_max: float = 0.05
    low_max: float = 0.15
    mod_max: float = 0.40

    # Optional depth-proxy weighting. If > 0 and a depth array is supplied
    # to :func:`classify`, the cell's effective flooded fraction is blended
    # as:  frac_eff = (1 - w) * frac + w * depth_norm
    depth_weight: float = 0.0

    class_names: tuple[str, ...] = field(default_factory=lambda: ("none", "low", "moderate", "severe"))

    def thresholds(self) -> tuple[float, float, float]:
        return self.none_max, self.low_max, self.mod_max


def _block_reduce_mean(arr: np.ndarray, block: int) -> np.ndarray:
    """Mean-pool an (H, W) array with a square block. Trailing rows/cols dropped."""
    h, w = arr.shape
    hh = (h // block) * block
    ww = (w // block) * block
    a = arr[:hh, :ww]
    return a.reshape(hh // block, block, ww // block, block).mean(axis=(1, 3))


def classify(
    mask: np.ndarray,
    depth: np.ndarray | None = None,
    cfg: SeverityConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Classify severity per cell.

    Parameters
    ----------
    mask
        (H, W) bool / uint8 flood mask (1 = flooded).
    depth
        (H, W) optional depth-proxy in [0, 1]. Aligned with ``mask``. Passing
        this with ``cfg.depth_weight > 0`` blends it into the score.
    cfg
        :class:`SeverityConfig`; default if None.

    Returns
    -------
    frac : (H', W') float32 — flooded fraction per cell.
    cls  : (H', W') int8    — Severity enum values per cell.
    """
    cfg = cfg or SeverityConfig()
    m = np.asarray(mask).astype(np.float32)
    if m.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {m.shape}")
    frac = _block_reduce_mean(m, cfg.cell_px)

    effective = frac
    if depth is not None and cfg.depth_weight > 0:
        d = np.clip(np.asarray(depth, dtype=np.float32), 0.0, 1.0)
        if d.shape != m.shape:
            raise ValueError(f"depth shape {d.shape} must match mask {m.shape}")
        d_cell = _block_reduce_mean(d, cfg.cell_px)
        effective = (1.0 - cfg.depth_weight) * frac + cfg.depth_weight * d_cell

    none_max, low_max, mod_max = cfg.thresholds()
    cls = np.full(effective.shape, Severity.NONE, dtype=np.int8)
    cls[effective >= none_max] = Severity.LOW
    cls[effective >= low_max] = Severity.MODERATE
    cls[effective >= mod_max] = Severity.SEVERE
    return frac.astype(np.float32), cls


def classify_raster(
    mask_path: str | Path,
    out_path: str | Path,
    depth_path: str | Path | None = None,
    cfg: SeverityConfig | None = None,
) -> Path:
    """Read a mask GeoTIFF, write a severity GeoTIFF at the coarsened grid."""
    mask_path = Path(mask_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = cfg or SeverityConfig()

    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        src_profile = src.profile.copy()
        src_transform = src.transform

    depth = None
    if depth_path is not None:
        with rasterio.open(depth_path) as src:
            depth = src.read(1).astype(np.float32)

    frac, cls = classify(mask, depth=depth, cfg=cfg)

    # Coarsen the transform: cell pixel size scales by cfg.cell_px.
    new_transform = src_transform * src_transform.scale(cfg.cell_px, cfg.cell_px)
    out_profile = src_profile.copy()
    out_profile.update(
        height=cls.shape[0],
        width=cls.shape[1],
        transform=new_transform,
        count=2,
        dtype="float32",
        nodata=None,
        compress="lzw",
    )
    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(frac.astype(np.float32), 1)
        dst.set_band_description(1, "flooded_fraction")
        dst.write(cls.astype(np.float32), 2)
        dst.set_band_description(2, "severity_class")
    return out_path


def cell_counts(cls: np.ndarray) -> dict[str, int]:
    """Return number of cells per severity class."""
    return {
        "none": int((cls == Severity.NONE).sum()),
        "low": int((cls == Severity.LOW).sum()),
        "moderate": int((cls == Severity.MODERATE).sum()),
        "severe": int((cls == Severity.SEVERE).sum()),
    }


__all__ = [
    "Severity",
    "SeverityConfig",
    "cell_counts",
    "classify",
    "classify_raster",
]
