"""Binary morphological post-processing.

After thresholding we typically have salt-and-pepper noise: isolated pixels
classified as water over dry land, and small holes inside real water bodies.
Morphological opening (erosion → dilation) removes the former; closing
(dilation → erosion) plus hole-filling fixes the latter.
"""

from __future__ import annotations

import numpy as np
from skimage.morphology import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_opening,
    disk,
    remove_small_holes,
    remove_small_objects,
)


def clean(
    mask: np.ndarray,
    opening_radius: int = 1,
    closing_radius: int = 1,
    min_object_area: int = 25,
    min_hole_area: int = 25,
) -> np.ndarray:
    """Default "clean-up" pipeline for flood masks.

    1. Opening with a disk(radius=opening_radius) — removes speckle.
    2. Closing with a disk(radius=closing_radius) — bridges small gaps.
    3. Remove connected water components smaller than min_object_area px.
    4. Fill holes smaller than min_hole_area px.

    Defaults are tuned for 10 m Sentinel-2 imagery: 25 px ≈ 2,500 m² ≈ 0.25 ha.
    """
    m = mask.astype(bool, copy=False)
    if opening_radius > 0:
        m = binary_opening(m, footprint=disk(opening_radius))
    if closing_radius > 0:
        m = binary_closing(m, footprint=disk(closing_radius))
    if min_object_area > 0:
        m = remove_small_objects(m, min_size=min_object_area, connectivity=2)
    if min_hole_area > 0:
        m = remove_small_holes(m, area_threshold=min_hole_area, connectivity=2)
    return m


def boundary(mask: np.ndarray) -> np.ndarray:
    """Return the 1-pixel-wide outline of a binary mask.

    Useful for overlaying flood edges on RGB composites without obscuring
    the underlying imagery.
    """
    m = mask.astype(bool, copy=False)
    return m ^ binary_erosion(m, footprint=disk(1))


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Dilate a binary mask by a disk of ``radius`` px."""
    return binary_dilation(mask.astype(bool, copy=False), footprint=disk(radius))


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Erode a binary mask by a disk of ``radius`` px."""
    return binary_erosion(mask.astype(bool, copy=False), footprint=disk(radius))


__all__ = ["boundary", "clean", "dilate", "erode"]
