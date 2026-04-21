"""Spectral water / vegetation indices for Sentinel-2.

All indices operate on the DDA 6-band stack, ordered as:

    index  band  wavelength   purpose
    -----  ----  ----------   ------------------
      0     B2   492 nm       blue
      1     B3   560 nm       green
      2     B4   665 nm       red
      3     B8   842 nm       near-infrared
      4     B11  1610 nm      shortwave-infrared 1
      5     B12  2190 nm      shortwave-infrared 2

Each function takes an ``(6, H, W)`` float32 reflectance array and returns
an ``(H, W)`` float32 index array, typically in [-1, 1]. Division by zero
is handled via a small epsilon added to the denominator.

References
----------
- McFeeters (1996). NDWI. IJRS 17(7):1425–1432.
- Xu (2006). MNDWI. IJRS 27(14):3025–3033.
- Rouse et al. (1974). NDVI. NASA Goddard.
- Feyisa et al. (2014). AWEI (shadow / non-shadow). RSE 140:23–35.
"""

from __future__ import annotations

import numpy as np

EPS: float = 1e-9

# DDA band-stack positions (0-based).
_B2, _B3, _B4, _B8, _B11, _B12 = 0, 1, 2, 3, 4, 5


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Return num / (den + EPS) as float32, preserving NaN in inputs."""
    return (num / (den + EPS)).astype(np.float32)


def ndwi(stack: np.ndarray) -> np.ndarray:
    """McFeeters NDWI = (Green − NIR) / (Green + NIR).

    Positive over water, negative over vegetation. Fails over urban/built-up
    pixels (they can give high positive NDWI) — use :func:`mndwi` there.
    """
    g, nir = stack[_B3], stack[_B8]
    return _safe_ratio(g - nir, g + nir)


def mndwi(stack: np.ndarray) -> np.ndarray:
    """Modified NDWI (Xu 2006) = (Green − SWIR1) / (Green + SWIR1).

    Primary flood index for this project: handles urban built-up better than
    NDWI because built-up surfaces have high SWIR1 reflectance, driving the
    numerator negative.
    """
    g, swir1 = stack[_B3], stack[_B11]
    return _safe_ratio(g - swir1, g + swir1)


def ndvi(stack: np.ndarray) -> np.ndarray:
    """NDVI = (NIR − Red) / (NIR + Red). Complement to water indices."""
    nir, r = stack[_B8], stack[_B4]
    return _safe_ratio(nir - r, nir + r)


def awei_nsh(stack: np.ndarray) -> np.ndarray:
    """Automated Water Extraction Index, **non-shadow** variant (Feyisa 2014).

    AWEInsh = 4·(Green − SWIR1) − (0.25·NIR + 2.75·SWIR2)

    Designed for scenes where dark shadows are rare. Higher is wetter.
    """
    return (
        4.0 * (stack[_B3] - stack[_B11]) - (0.25 * stack[_B8] + 2.75 * stack[_B12])
    ).astype(np.float32)


def awei_sh(stack: np.ndarray) -> np.ndarray:
    """AWEI **shadow-suppressing** variant (Feyisa 2014).

    AWEIsh = Blue + 2.5·Green − 1.5·(NIR + SWIR1) − 0.25·SWIR2

    Preferred when topographic or building shadows may be confused with water.
    """
    return (
        stack[_B2] + 2.5 * stack[_B3] - 1.5 * (stack[_B8] + stack[_B11]) - 0.25 * stack[_B12]
    ).astype(np.float32)


def compute_all(stack: np.ndarray) -> dict[str, np.ndarray]:
    """Return every index in this module as a ``{name: array}`` dict."""
    return {
        "ndwi": ndwi(stack),
        "mndwi": mndwi(stack),
        "ndvi": ndvi(stack),
        "awei_nsh": awei_nsh(stack),
        "awei_sh": awei_sh(stack),
    }


__all__ = ["awei_nsh", "awei_sh", "compute_all", "mndwi", "ndvi", "ndwi"]
