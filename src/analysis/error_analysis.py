"""Error-category analysis on Sen1Floods11.

Given predicted masks, ground-truth masks and the source reflectance chips,
classify every prediction error into one of a small, interpretable set of
spectral categories. The output is a single aggregated DataFrame that can be
plotted directly as a confusion-by-category bar chart.

Categories use two easy-to-compute indices:

    NDVI = (B8 − B4) / (B8 + B4)     — vegetation
    MNDWI = (B3 − B11) / (B3 + B11)  — water (sensitive to built-up)

| Spectral category     | Rule                             | Typical surfaces            |
|-----------------------|----------------------------------|-----------------------------|
| turbid_water          | MNDWI > 0.1  AND  NDVI < 0       | sediment-rich flood water   |
| dark_land             | MNDWI < 0    AND  NIR < 0.1      | asphalt, shadow, burnt      |
| vegetation            | NDVI > 0.3                       | crops, forest               |
| bare_sparse           | NDVI in [0, 0.3] AND MNDWI < 0 AND NIR >= 0.10 | bare soil / built-up |
| other                 | otherwise                         | — catch-all                 |

For every chip we tabulate how many false positives and false negatives fall
into each category — useful both for the qualitative narrative in the report
and for proposing targeted future work (e.g., "50 % of FPs are dark_land →
augment with shadow examples").
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# 6-band DDA stack positions.
_B2, _B3, _B4, _B8, _B11, _B12 = 0, 1, 2, 3, 4, 5
EPS = 1e-9


def _ndvi(stack: np.ndarray) -> np.ndarray:
    return (stack[_B8] - stack[_B4]) / (stack[_B8] + stack[_B4] + EPS)


def _mndwi(stack: np.ndarray) -> np.ndarray:
    return (stack[_B3] - stack[_B11]) / (stack[_B3] + stack[_B11] + EPS)


def categorise(stack: np.ndarray) -> np.ndarray:
    """Return a per-pixel category index (int8) for a (C, H, W) reflectance chip.

    0 turbid_water · 1 dark_land · 2 vegetation · 3 bare_sparse · 4 other
    """
    ndvi = _ndvi(stack)
    mndwi = _mndwi(stack)
    nir = stack[_B8]

    cat = np.full(ndvi.shape, 4, dtype=np.int8)  # default: other
    cat[(mndwi > 0.1) & (ndvi < 0.0)] = 0                    # turbid water
    cat[(mndwi < 0.0) & (nir < 0.10)] = 1                    # dark land
    cat[ndvi > 0.3] = 2                                      # vegetation
    # bare_sparse excludes dark pixels so the categories partition uniquely.
    cat[(ndvi >= 0.0) & (ndvi <= 0.3) & (mndwi < 0.0) & (nir >= 0.10)] = 3
    return cat


CATEGORY_NAMES: tuple[str, ...] = (
    "turbid_water", "dark_land", "vegetation", "bare_sparse", "other",
)


def tabulate_errors(
    images: list[np.ndarray],
    preds: list[np.ndarray],
    labels: list[np.ndarray],
    ignore_index: int = -1,
) -> pd.DataFrame:
    """Aggregate FP / FN counts by spectral category across many chips.

    Parameters
    ----------
    images
        List of (C, H, W) reflectance chips. Must be aligned with ``preds``
        and ``labels``.
    preds
        List of (H, W) bool predictions.
    labels
        List of (H, W) int labels in {0, 1, ignore_index}.

    Returns
    -------
    DataFrame with columns:
        category, fp_count, fn_count, fp_pct, fn_pct.
    """
    fp = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)
    fn = np.zeros(len(CATEGORY_NAMES), dtype=np.int64)

    for img, p, y in zip(images, preds, labels, strict=True):
        cat = categorise(img)
        valid = y != ignore_index
        yy = y[valid].astype(bool)
        pp = p[valid].astype(bool)
        cc = cat[valid]

        # FP: predicted 1, label 0.
        fp_mask = pp & ~yy
        # FN: predicted 0, label 1.
        fn_mask = ~pp & yy

        for k in range(len(CATEGORY_NAMES)):
            fp[k] += int(np.sum(fp_mask & (cc == k)))
            fn[k] += int(np.sum(fn_mask & (cc == k)))

    fp_total = int(fp.sum())
    fn_total = int(fn.sum())
    df = pd.DataFrame({
        "category": CATEGORY_NAMES,
        "fp_count": fp,
        "fn_count": fn,
        "fp_pct": (fp / fp_total * 100) if fp_total else 0.0,
        "fn_pct": (fn / fn_total * 100) if fn_total else 0.0,
    })
    return df


__all__ = ["CATEGORY_NAMES", "categorise", "tabulate_errors"]
