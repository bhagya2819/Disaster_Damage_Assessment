"""Thresholding algorithms for converting a continuous index map to a binary mask.

Every function returns a ``ThresholdResult`` with the chosen threshold value
and a boolean mask (``True`` = foreground / water). Having the threshold
exposed makes the ablation table reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.filters import (
    threshold_li,
    threshold_local,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
)


@dataclass(frozen=True)
class ThresholdResult:
    mask: np.ndarray  # bool (H, W) — True = foreground / water
    value: float      # chosen threshold on the input index
    method: str


def _finite(x: np.ndarray) -> np.ndarray:
    """Strip NaN/inf from an index map before histogram-based thresholding."""
    return x[np.isfinite(x)]


def otsu(index: np.ndarray) -> ThresholdResult:
    """Otsu's method (1979): minimises within-class variance. Bimodal baseline."""
    t = float(threshold_otsu(_finite(index)))
    return ThresholdResult(mask=index > t, value=t, method="otsu")


def triangle(index: np.ndarray) -> ThresholdResult:
    """Zack-Rosenfeld-Hummel (1977) triangle method. Robust when classes are unbalanced."""
    t = float(threshold_triangle(_finite(index)))
    return ThresholdResult(mask=index > t, value=t, method="triangle")


def yen(index: np.ndarray) -> ThresholdResult:
    """Yen (1995): maximises entropy criterion. Tends to be more permissive than Otsu."""
    t = float(threshold_yen(_finite(index)))
    return ThresholdResult(mask=index > t, value=t, method="yen")


def li(index: np.ndarray) -> ThresholdResult:
    """Li & Lee (1993): minimum cross-entropy."""
    t = float(threshold_li(_finite(index)))
    return ThresholdResult(mask=index > t, value=t, method="li")


def fixed(index: np.ndarray, value: float) -> ThresholdResult:
    """Hard-coded threshold (use only with a literature-referenced value)."""
    return ThresholdResult(mask=index > value, value=value, method=f"fixed_{value}")


def adaptive(
    index: np.ndarray,
    block_size: int = 51,
    offset: float = 0.0,
    method: str = "gaussian",
) -> ThresholdResult:
    """Local/adaptive threshold — threshold varies across the scene.

    Useful when illumination / atmosphere gradients make a single global
    threshold fail. ``block_size`` must be odd; 51 px @ 10 m ≈ 500 m window.
    """
    if block_size % 2 == 0:
        raise ValueError("block_size must be odd.")
    local_t = threshold_local(index, block_size=block_size, method=method, offset=offset)
    return ThresholdResult(
        mask=index > local_t,
        value=float(local_t.mean()),
        method=f"adaptive_{method}_b{block_size}",
    )


def auto(index: np.ndarray, method: str = "otsu") -> ThresholdResult:
    """Dispatch helper used by the ablation harness."""
    return {
        "otsu": otsu,
        "triangle": triangle,
        "yen": yen,
        "li": li,
    }[method](index)


__all__ = [
    "ThresholdResult",
    "adaptive",
    "auto",
    "fixed",
    "li",
    "otsu",
    "triangle",
    "yen",
]
