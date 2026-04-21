"""Classical spatial-domain filters for preprocessing and coastline refinement.

Intentionally a thin typed wrapper over ``scikit-image`` and ``scipy.ndimage``
so the Phase-2 walkthrough notebook can demonstrate each filter with a
single import from one module.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.feature import canny
from skimage.filters import sobel


def gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gaussian low-pass filter. Works on 2-D or per-band for 3-D arrays."""
    if image.ndim == 2:
        return gaussian_filter(image.astype(np.float32), sigma=sigma)
    if image.ndim == 3:
        return np.stack([gaussian_filter(image[i].astype(np.float32), sigma=sigma) for i in range(image.shape[0])])
    raise ValueError(f"Expected 2-D or 3-D array, got {image.ndim}-D")


def median(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Median filter — robust to salt-and-pepper noise, preserves edges."""
    if image.ndim == 2:
        return median_filter(image, size=size)
    if image.ndim == 3:
        return np.stack([median_filter(image[i], size=size) for i in range(image.shape[0])])
    raise ValueError(f"Expected 2-D or 3-D array, got {image.ndim}-D")


def bilateral(image: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 3.0) -> np.ndarray:
    """Edge-preserving bilateral smoothing (scikit-image)."""
    from skimage.restoration import denoise_bilateral  # noqa: PLC0415

    if image.ndim == 2:
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial).astype(np.float32)
    if image.ndim == 3:
        return np.stack([
            denoise_bilateral(image[i], sigma_color=sigma_color, sigma_spatial=sigma_spatial).astype(np.float32)
            for i in range(image.shape[0])
        ])
    raise ValueError(f"Expected 2-D or 3-D array, got {image.ndim}-D")


def sobel_edges(image: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude for a 2-D image."""
    if image.ndim != 2:
        raise ValueError("Sobel expects a 2-D array; apply per band upstream if needed.")
    return sobel(image).astype(np.float32)


def canny_edges(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: float | None = None,
    high_threshold: float | None = None,
) -> np.ndarray:
    """Canny edge detector → boolean array."""
    if image.ndim != 2:
        raise ValueError("Canny expects a 2-D array.")
    return canny(image, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)


__all__ = ["bilateral", "canny_edges", "gaussian", "median", "sobel_edges"]
