"""Sentinel-2 cloud and cloud-shadow masking.

Two paths are supported:

1. **SCL-based mask** (default) — fast, no extra dependency, works offline.
   The Scene Classification Layer is a 20 m classification raster included in
   every L2A product. Classes 3, 8, 9, 10 and 11 correspond to shadow, medium-
   and high-probability cloud, thin cirrus and snow respectively.

2. **s2cloudless probability mask** — optional, more accurate. Uses the
   pretrained gradient-boosted-tree classifier from Sentinel Hub. Requires
   the B1, B2, B4, B5, B8, B8A, B9, B10, B11, B12 bands.

Both functions return a boolean array where True = CLEAR (not cloud/shadow),
so the standard usage is ``image * mask[..., None]``.
"""

from __future__ import annotations

import numpy as np

# SCL class values considered "not clear" for our purposes.
# 0=no data, 1=saturated/defective, 2=dark areas, 3=cloud shadows, 4=veg,
# 5=not-vegetated, 6=water, 7=unclassified, 8=cloud med, 9=cloud high,
# 10=thin cirrus, 11=snow/ice.
_BAD_SCL_CLASSES: tuple[int, ...] = (0, 1, 3, 8, 9, 10, 11)


def scl_cloud_mask(
    scl: np.ndarray,
    bad_classes: tuple[int, ...] = _BAD_SCL_CLASSES,
) -> np.ndarray:
    """Boolean mask where True means the pixel is CLEAR (usable).

    Parameters
    ----------
    scl
        2-D int array of SCL class values (same H,W as the reflectance raster).
    bad_classes
        Class values to mask OUT. Defaults to the five cloud/shadow/snow
        classes plus no-data and saturated. Pass an explicit subset to be
        more permissive.
    """
    mask = np.ones_like(scl, dtype=bool)
    for cls in bad_classes:
        mask &= scl != cls
    return mask


def s2cloudless_probability(
    bands: np.ndarray,
    threshold: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the s2cloudless classifier and return (clear_mask, probability).

    Parameters
    ----------
    bands
        Float32 reflectance array shaped (10, H, W) with the band order
        **[B01, B02, B04, B05, B08, B8A, B09, B10, B11, B12]**. If you only
        have the 6-band DDA subset, use :func:`scl_cloud_mask` instead.
    threshold
        Probability ≥ threshold marks a pixel as cloud.

    Returns
    -------
    clear_mask : bool array (H, W), True = clear
    probability : float32 array (H, W), cloud probability in [0, 1]
    """
    try:
        from s2cloudless import S2PixelCloudDetector  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "s2cloudless is required for this function. Install with "
            "`pip install s2cloudless`."
        ) from e

    if bands.shape[0] != 10:
        raise ValueError(f"s2cloudless needs 10 bands; got {bands.shape[0]}.")

    # s2cloudless expects (n_images, H, W, n_bands) reflectance in [0, 1].
    batch = np.transpose(bands, (1, 2, 0))[None, ...].astype(np.float32)
    detector = S2PixelCloudDetector(
        threshold=threshold, average_over=4, dilation_size=2, all_bands=False
    )
    prob = detector.get_cloud_probability_maps(batch)[0]
    clear = prob < threshold
    return clear, prob


def fraction_cloudy(clear_mask: np.ndarray) -> float:
    """Return the fraction of pixels marked as cloudy (not clear)."""
    return float(1.0 - clear_mask.mean())
