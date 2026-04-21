"""Classical change-detection algorithms for pre/post raster pairs.

All functions take matched ``(C, H, W)`` arrays ``pre`` and ``post`` (same
shape, same CRS, same transform — enforce with ``src.preprocess.coregister``)
and return a single-band change magnitude or mask.

Four techniques are provided:

1. :func:`image_difference` — per-pixel ``post − pre`` on a chosen band or
   index. Signed. Classic baseline.
2. :func:`image_ratio` — ``post / pre``. Multiplicative; robust to additive
   illumination offsets.
3. :func:`pca_change` — stack pre and post, run PCA across the channel
   dimension; the last component concentrates change. See Deng (2008).
4. :func:`change_vector_analysis` — magnitude of the multi-band difference
   vector. Captures change direction via angle; magnitude gives intensity.

Each returns a continuous score; pair with ``src.dip.thresholding`` to
binarise.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def image_difference(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Signed post − pre. Works on any shape; dtype promoted to float32."""
    if pre.shape != post.shape:
        raise ValueError(f"Shape mismatch: {pre.shape} vs {post.shape}")
    return (post.astype(np.float32) - pre.astype(np.float32))


def image_ratio(pre: np.ndarray, post: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Pixel-wise post / (pre + eps)."""
    if pre.shape != post.shape:
        raise ValueError(f"Shape mismatch: {pre.shape} vs {post.shape}")
    return (post.astype(np.float32) / (pre.astype(np.float32) + eps))


def change_vector_analysis(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """Change magnitude ||post − pre||_2 along the channel axis.

    Expects ``(C, H, W)``; returns ``(H, W)``.
    """
    if pre.ndim != 3 or post.ndim != 3:
        raise ValueError("CVA expects (C, H, W) arrays.")
    diff = post.astype(np.float32) - pre.astype(np.float32)
    return np.sqrt(np.sum(diff**2, axis=0))


def pca_change(
    pre: np.ndarray,
    post: np.ndarray,
    n_components: int | None = None,
) -> np.ndarray:
    """PCA-based change map.

    Stacks pre and post along the channel axis → ``(2C, H·W)`` samples, fits
    PCA, and returns the **last** principal component reshaped back to
    ``(H, W)``. In classical remote-sensing work the last component
    concentrates the change signal while the first components capture the
    scene-common variance. See Deng et al. (2008) for derivation.

    Parameters
    ----------
    n_components
        If None, defaults to ``2 * C`` (full decomposition). Use a smaller
        value (e.g. 3) for faster inference on large rasters.
    """
    if pre.ndim != 3 or pre.shape != post.shape:
        raise ValueError("pca_change expects matching (C, H, W) arrays.")

    c, h, w = pre.shape
    stacked = np.concatenate([pre, post], axis=0).reshape(2 * c, h * w).T  # (H·W, 2C)

    k = n_components or 2 * c
    pca = PCA(n_components=k, svd_solver="full")
    transformed = pca.fit_transform(stacked)  # (H·W, k)

    change_component = transformed[:, -1]
    return np.abs(change_component).reshape(h, w).astype(np.float32)


def mndwi_difference(pre_stack: np.ndarray, post_stack: np.ndarray) -> np.ndarray:
    """Shortcut: MNDWI_post − MNDWI_pre on 6-band DDA stacks.

    The single index commonly used as the canonical flood change-detection
    signal.
    """
    from src.dip.indices import mndwi  # noqa: PLC0415  # avoid circular import at module load

    return mndwi(post_stack) - mndwi(pre_stack)


__all__ = [
    "change_vector_analysis",
    "image_difference",
    "image_ratio",
    "mndwi_difference",
    "pca_change",
]
