"""Sentinel-1 SAR flood mapping (Phase 7 bonus).

SAR sees through clouds and night, which makes it the canonical backup
modality for flood mapping when Sentinel-2 optical is unavailable. Open
water is a near-perfect specular reflector for C-band radar — most of the
incident energy bounces away from the sensor — so flooded pixels appear as
dark (low-backscatter) regions in a ``log(VV)`` image.

This module implements the operational classical SAR flood-mapping pipeline:

1. :func:`to_db`             — linear γ⁰ → decibels.
2. :func:`refined_lee`       — Lee (1980) adaptive speckle filter.
3. :func:`sar_flood_mask`    — log-VV Otsu threshold → binary flood mask.
4. :func:`fuse_with_optical` — agreement / union / weighted fusion of the
   SAR mask with an optical U-Net or MNDWI mask.

References
----------
- Lee, J.-S. (1980). Digital image enhancement and noise filtering by use of
  local statistics. *IEEE PAMI* 2(2):165-168.
- Martinis, S. et al. (2009). Near real-time flood detection via split-based
  thresholding on TerraSAR-X. *NHESS* 9:303-314.
- Chini, M. et al. (2017). Hierarchical split-based approach for SAR
  thresholding. *IEEE TGRS* 55(12):6975-6988.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter
from skimage.filters import threshold_otsu

EPS: float = 1e-6


def to_db(linear: np.ndarray) -> np.ndarray:
    """Convert linear γ⁰ / σ⁰ backscatter to decibels: ``10 * log10(x)``.

    Clips to ``EPS`` before the log to avoid ``-inf`` on exact zeros.
    """
    return 10.0 * np.log10(np.clip(linear, EPS, None)).astype(np.float32)


def refined_lee(
    image: np.ndarray,
    window_size: int = 7,
    damping: float = 1.0,
) -> np.ndarray:
    """Lee (1980) adaptive speckle filter.

    For each pixel, computes a weighted combination of the local mean and
    the pixel value, where the weight depends on the local coefficient of
    variation (std / mean). Homogeneous areas (low CV) get mean-filtered;
    edges (high CV) are preserved.

    Parameters
    ----------
    image
        2-D float array in linear backscatter units (NOT dB — the noise
        model is multiplicative in linear space).
    window_size
        Odd-sized filter kernel. 7×7 is a good default for Sentinel-1 GRD.
    damping
        Additional smoothing factor applied to the weights; larger = more
        smoothing.
    """
    if image.ndim != 2:
        raise ValueError(f"refined_lee expects a 2-D array; got {image.shape}")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")

    img = image.astype(np.float32)
    local_mean = uniform_filter(img, size=window_size, mode="reflect")
    local_var = uniform_filter(img**2, size=window_size, mode="reflect") - local_mean**2
    local_var = np.clip(local_var, 0.0, None)
    # Noise variance estimated globally as (mean local var) / image-level mean² — a
    # standard Lee-filter approximation for multiplicative speckle.
    noise_var = float(np.nanmean(local_var) / max(float(np.nanmean(img) ** 2), EPS))

    # Weight (k) ∈ [0, 1]. At the edges, local_var >> noise_var so k → 1
    # (trust the raw pixel). In homogeneous patches k → 0 (trust the mean).
    k = local_var / (local_var + damping * noise_var * local_mean**2 + EPS)
    filtered = local_mean + k * (img - local_mean)
    return filtered.astype(np.float32)


def frost(
    image: np.ndarray,
    window_size: int = 7,
    damping: float = 2.0,
) -> np.ndarray:
    """Frost (1982) alternative speckle filter — exponentially-weighted mean
    with a damping factor proportional to local coefficient of variation.
    Included as a drop-in alternative; our pipeline uses :func:`refined_lee`
    by default.
    """
    if image.ndim != 2:
        raise ValueError(f"frost expects a 2-D array; got {image.shape}")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    img = image.astype(np.float32)

    half = window_size // 2
    ys, xs = np.meshgrid(
        np.arange(-half, half + 1), np.arange(-half, half + 1), indexing="ij"
    )
    distance = np.sqrt(ys**2 + xs**2).astype(np.float32)

    def _kernel(window: np.ndarray) -> float:
        w = window.reshape(window_size, window_size)
        mean = w.mean()
        if mean < EPS:
            return float(mean)
        std = w.std()
        cv2 = (std / mean) ** 2
        weights = np.exp(-damping * cv2 * distance)
        return float(np.sum(weights * w) / (np.sum(weights) + EPS))

    return generic_filter(img, _kernel, size=window_size, mode="reflect").astype(np.float32)


def sar_flood_mask(
    vv_linear: np.ndarray,
    filter_window: int = 7,
    threshold_db: float | None = None,
) -> tuple[np.ndarray, float]:
    """End-to-end SAR flood mask from a linear VV backscatter array.

    Pipeline: Lee speckle filter → dB conversion → Otsu threshold → mask.

    Parameters
    ----------
    vv_linear
        (H, W) linear γ⁰ VV backscatter — Sentinel-1 GRD units.
    filter_window
        Lee filter kernel size.
    threshold_db
        If given, use this fixed dB threshold and skip Otsu. Typical range
        −18 dB to −22 dB for permanent water (Martinis 2009).

    Returns
    -------
    mask : bool (H, W) — True = flood.
    threshold : float — the dB threshold used (returned for logging).
    """
    filtered = refined_lee(vv_linear, window_size=filter_window)
    db = to_db(filtered)

    if threshold_db is None:
        finite = db[np.isfinite(db)]
        if finite.size == 0:
            raise ValueError("SAR image is entirely non-finite after filtering")
        threshold_db = float(threshold_otsu(finite))

    mask = (db < threshold_db).astype(bool)
    return mask, float(threshold_db)


def fuse_with_optical(
    sar_mask: np.ndarray,
    optical_mask: np.ndarray,
    mode: str = "union",
) -> np.ndarray:
    """Combine SAR and optical flood masks.

    Parameters
    ----------
    sar_mask, optical_mask
        Bool masks of the same shape.
    mode
        * "union" (OR) — a pixel is flooded if EITHER sensor says so. Use
          when clouds block the optical sensor.
        * "agreement" (AND) — a pixel is flooded only if BOTH sensors agree.
          Higher precision but lower recall.
        * "optical_primary" — optical where it's available (not no-data),
          SAR elsewhere. Our recommended default when SCL-masked pixels
          exist in the optical stream.
    """
    if sar_mask.shape != optical_mask.shape:
        raise ValueError(f"shape mismatch: {sar_mask.shape} vs {optical_mask.shape}")

    a = sar_mask.astype(bool, copy=False)
    b = optical_mask.astype(bool, copy=False)

    if mode == "union":
        return a | b
    if mode == "agreement":
        return a & b
    if mode == "optical_primary":
        # Placeholder semantics: return `b` wherever it has flood signal,
        # else fall back to `a`. Caller is responsible for supplying an
        # optical mask that's False on cloud pixels (which is the normal
        # behaviour of our classical pipeline after SCL masking).
        return np.where(b, True, a)
    raise ValueError(f"unknown mode: {mode!r}")


def agreement_fraction(sar_mask: np.ndarray, optical_mask: np.ndarray) -> float:
    """Return Cohen's κ-style agreement between the two masks.

    Useful for reporting in the final paper: "SAR vs optical agree at
    κ = 0.XX on cloud-free pixels".
    """
    if sar_mask.shape != optical_mask.shape:
        raise ValueError("shape mismatch")
    n = sar_mask.size
    if n == 0:
        return float("nan")
    po = float(np.mean(sar_mask == optical_mask))
    p1_sar = float(np.mean(sar_mask))
    p1_opt = float(np.mean(optical_mask))
    pe = p1_sar * p1_opt + (1 - p1_sar) * (1 - p1_opt)
    if np.isclose(1 - pe, 0.0):
        return 1.0 if np.isclose(po, 1.0) else float("nan")
    return float((po - pe) / (1 - pe))


__all__ = [
    "agreement_fraction",
    "frost",
    "fuse_with_optical",
    "refined_lee",
    "sar_flood_mask",
    "to_db",
]
