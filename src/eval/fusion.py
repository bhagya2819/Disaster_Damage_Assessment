"""Hybrid classical + deep-learning fusion for flood masks.

Two simple, interpretable strategies:

- :func:`fuse_weighted` — convex combination of two probability/score maps,
  then threshold. Works for {U-Net prob, classical continuous index} or
  {U-Net prob, classical binary mask cast to float}.
- :func:`fuse_agreement` — pixel-wise AND (water iff both methods say water).
  High precision, lower recall — useful for conservative "confident-only" maps.

Design intent: keep the logic simple enough to describe in one figure of the
final report, and trivially implementable in the Streamlit UI.
"""

from __future__ import annotations

import numpy as np


def fuse_weighted(
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    weight_a: float = 0.5,
    threshold: float = 0.5,
) -> np.ndarray:
    """Return a binary mask from ``weight_a * a + (1 - weight_a) * b``.

    Inputs should be in [0, 1]; binary masks work as-is (they already are
    in {0, 1}). ``weight_a`` controls the trust given to method A.
    """
    if prob_a.shape != prob_b.shape:
        raise ValueError(f"Shape mismatch: {prob_a.shape} vs {prob_b.shape}")
    if not (0.0 <= weight_a <= 1.0):
        raise ValueError(f"weight_a must be in [0, 1]; got {weight_a}")
    combined = weight_a * prob_a.astype(np.float32) + (1.0 - weight_a) * prob_b.astype(np.float32)
    return (combined >= threshold).astype(bool)


def fuse_agreement(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Pixel-wise AND of two binary masks — high-precision intersection."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    return mask_a.astype(bool) & mask_b.astype(bool)


def fuse_union(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """Pixel-wise OR of two binary masks — high-recall union."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(f"Shape mismatch: {mask_a.shape} vs {mask_b.shape}")
    return mask_a.astype(bool) | mask_b.astype(bool)


__all__ = ["fuse_agreement", "fuse_union", "fuse_weighted"]
