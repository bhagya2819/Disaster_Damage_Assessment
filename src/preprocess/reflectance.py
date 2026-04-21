"""Sentinel-2 reflectance conversion utilities.

Sentinel-2 L2A products are distributed as 16-bit DN (digital numbers) where
surface reflectance ρ ∈ [0, 1] ≈ DN / 10000. Post-harmonisation (2022-01-25
onward), ESA introduced a radiometric offset of -1000 that must also be
subtracted before scaling.

Our GEE downloader (``src/data/gee_download.py``) already exports data in
float reflectance space, so for on-disk rasters this module is a no-op;
however, when users bring their own DN rasters (e.g. raw `.SAFE` tiles) they
should call :func:`dn_to_reflectance` first.
"""

from __future__ import annotations

import numpy as np

# ESA quantification value and processing-baseline offset.
# See https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi
QUANTIFICATION_VALUE: float = 10_000.0
RADIOMETRIC_OFFSET: float = -1_000.0  # applied for processing-baseline ≥ 04.00


def dn_to_reflectance(
    dn: np.ndarray,
    apply_offset: bool = True,
    clip: bool = True,
) -> np.ndarray:
    """Convert Sentinel-2 L2A digital numbers to surface reflectance.

    Parameters
    ----------
    dn
        Array of DN values (any shape). Typically uint16 in [0, 10000+offset].
    apply_offset
        If True, subtracts the processing-baseline-04.00 radiometric offset of
        -1000 before dividing by 10000. Safe to leave on even for older
        products — the offset encoded in the metadata will be 0 and the
        subtraction is a no-op then. Set False only if you have already
        removed the offset upstream.
    clip
        Clip to [0, 1] after scaling.

    Returns
    -------
    float32 reflectance array with the same shape as ``dn``.
    """
    x = dn.astype(np.float32, copy=True)
    if apply_offset:
        x = x + RADIOMETRIC_OFFSET
    x = x / QUANTIFICATION_VALUE
    if clip:
        np.clip(x, 0.0, 1.0, out=x)
    return x


def reflectance_to_dn(reflectance: np.ndarray, dtype: np.dtype = np.uint16) -> np.ndarray:
    """Inverse of :func:`dn_to_reflectance` for round-tripping in tests."""
    x = reflectance * QUANTIFICATION_VALUE - RADIOMETRIC_OFFSET
    x = np.clip(x, 0, np.iinfo(dtype).max)
    return x.astype(dtype)
