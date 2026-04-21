"""Statistical significance tests for comparing two segmentation methods.

Two methods are provided:

- :func:`mcnemar_test` — Non-parametric χ² on a 2×2 contingency table of
  per-pixel disagreements. Appropriate when we want to know: *"are the
  pixel-level errors of method A and method B the same?"*. Uses the
  continuity-corrected form (Edwards 1948) because chi-squared overestimates
  significance on small discordant counts.

- :func:`paired_bootstrap_iou` — Paired bootstrap confidence interval on the
  per-chip IoU difference. Appropriate when we want a **confidence interval**
  on Δ-IoU rather than a yes/no significance answer. More interpretable for
  the final report.

References
----------
- McNemar, Q. (1947). Note on the sampling error of the difference between
  correlated proportions or percentages. Psychometrika 12:153-157.
- Edwards, A.L. (1948). Note on the 'correction for continuity'.
  Psychometrika 13:185–187.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from src.eval.metrics import IGNORE_INDEX, iou


@dataclass(frozen=True)
class McNemarResult:
    b: int                # pixels where A correct, B wrong
    c: int                # pixels where A wrong, B correct
    statistic: float
    p_value: float

    def significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha


def mcnemar_test(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    label: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> McNemarResult:
    """McNemar's test with continuity correction on (A correct) vs (B correct).

    χ² = (|b − c| − 1)² / (b + c)

    where b = pixels only A got right, c = pixels only B got right. Under H₀
    (the two methods are equivalent), χ² ~ χ²(1).
    """
    if pred_a.shape != pred_b.shape or pred_a.shape != label.shape:
        raise ValueError("pred_a, pred_b and label must all share shape.")

    y = label.ravel()
    keep = y != ignore_index
    y = y[keep].astype(bool)
    a = pred_a.ravel()[keep].astype(bool)
    b = pred_b.ravel()[keep].astype(bool)

    a_correct = a == y
    b_correct = b == y
    n_b = int(np.sum(a_correct & ~b_correct))
    n_c = int(np.sum(~a_correct & b_correct))

    if n_b + n_c == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0)

    chi2 = (abs(n_b - n_c) - 1) ** 2 / (n_b + n_c)
    p = float(1 - stats.chi2.cdf(chi2, df=1))
    return McNemarResult(b=n_b, c=n_c, statistic=float(chi2), p_value=p)


@dataclass(frozen=True)
class BootstrapResult:
    mean_delta: float
    ci_lower: float
    ci_upper: float
    confidence: float


def paired_bootstrap_iou(
    per_chip_iou_a: np.ndarray,
    per_chip_iou_b: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> BootstrapResult:
    """Paired bootstrap CI on ΔIoU = IoU_A − IoU_B.

    Parameters
    ----------
    per_chip_iou_a, per_chip_iou_b
        1-D arrays of IoU values, one per test chip; must be paired (same
        chip index = same chip).
    n_bootstrap
        Number of resampling iterations.
    confidence
        CI level, e.g. 0.95 for a 95% interval.
    """
    a = np.asarray(per_chip_iou_a, dtype=np.float64)
    b = np.asarray(per_chip_iou_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: {a.shape} vs {b.shape}")
    if a.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays of per-chip IoU.")
    # Drop any NaNs (can happen for chips with no positive class).
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if a.size == 0:
        return BootstrapResult(float("nan"), float("nan"), float("nan"), confidence)

    rng = np.random.default_rng(seed)
    deltas = a - b
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.integers(0, deltas.size, size=deltas.size)
        boot_means[i] = deltas[idx].mean()

    alpha = 1.0 - confidence
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        mean_delta=float(deltas.mean()),
        ci_lower=float(lo),
        ci_upper=float(hi),
        confidence=confidence,
    )


def per_chip_iou(
    preds: list[np.ndarray],
    labels: list[np.ndarray],
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Compute IoU per (pred, label) pair. Length of inputs must match."""
    if len(preds) != len(labels):
        raise ValueError(f"length mismatch: {len(preds)} vs {len(labels)}")
    return np.array(
        [iou(p, y, ignore_index) for p, y in zip(preds, labels, strict=True)],
        dtype=np.float64,
    )


__all__ = [
    "BootstrapResult",
    "McNemarResult",
    "mcnemar_test",
    "paired_bootstrap_iou",
    "per_chip_iou",
]
