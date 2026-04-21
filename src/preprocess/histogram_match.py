"""Histogram matching between two multi-band rasters.

Used to normalise illumination/phenology differences between the pre- and
post-event composites so that difference-based change detection is driven
by actual surface change rather than by seasonal reflectance drift.

Uses scikit-image's ``match_histograms``; we apply it per band independently.
"""

from __future__ import annotations

import numpy as np
from skimage.exposure import match_histograms


def match_histograms_perband(
    source: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Match each band of ``source`` to the corresponding band in ``reference``.

    Parameters
    ----------
    source, reference
        Arrays shaped (C, H, W). Must have the same band count and dtype.

    Returns
    -------
    Array shaped (C, H, W), same dtype as ``source``, histogram-matched.
    """
    if source.shape[0] != reference.shape[0]:
        raise ValueError(
            f"Band count mismatch: source={source.shape[0]}, reference={reference.shape[0]}"
        )

    out = np.empty_like(source)
    for i in range(source.shape[0]):
        out[i] = match_histograms(source[i], reference[i]).astype(source.dtype)
    return out
