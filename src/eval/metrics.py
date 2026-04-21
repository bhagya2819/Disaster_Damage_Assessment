"""Binary segmentation metrics for flood-mask evaluation.

All functions operate on ``numpy`` arrays to stay framework-agnostic and are
designed to be aggregated across many chips by the ablation harness.

Conventions
-----------
- Predictions and labels are 2-D (H, W) or flat arrays of equal shape.
- Values are either bool or int in {0, 1}. Anything else is coerced to bool
  with ``!= 0`` except the label ignore sentinel (``-1``) which is masked out
  before every metric.
- Positive class (flood / water) = 1.

Implemented
-----------
- :func:`confusion_matrix_2x2`  — (tn, fp, fn, tp) counts.
- :func:`iou`                    — intersection-over-union = TP / (TP+FP+FN).
- :func:`dice`, :func:`f1`       — 2·TP / (2·TP+FP+FN)  (Dice ≡ F1 for binary).
- :func:`precision`, :func:`recall`, :func:`accuracy`.
- :func:`per_class_accuracy`     — accuracy on class 0 and class 1 separately.
- :func:`cohen_kappa`            — Cohen's κ.
- :func:`summary`                — returns every metric as a dict.

The ``ignore_index`` kwarg (default ``-1``, matching Sen1Floods11) drops pixels
with that label before any computation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IGNORE_INDEX: int = -1


@dataclass(frozen=True)
class Confusion:
    tn: int
    fp: int
    fn: int
    tp: int

    @property
    def n(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def as_dict(self) -> dict[str, int]:
        return {"tn": self.tn, "fp": self.fp, "fn": self.fn, "tp": self.tp}


def _prepare(pred: np.ndarray, label: np.ndarray, ignore_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten, drop ignore-labelled pixels, and coerce to bool."""
    if pred.shape != label.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape} label={label.shape}")
    p = pred.ravel()
    y = label.ravel()
    if ignore_index is not None:
        keep = y != ignore_index
        p = p[keep]
        y = y[keep]
    return p.astype(bool), y.astype(bool)


def confusion_matrix_2x2(
    pred: np.ndarray,
    label: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> Confusion:
    p, y = _prepare(pred, label, ignore_index)
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    tn = int(np.sum(~p & ~y))
    return Confusion(tn=tn, fp=fp, fn=fn, tp=tp)


def iou(pred: np.ndarray, label: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    denom = c.tp + c.fp + c.fn
    return float(c.tp / denom) if denom > 0 else float("nan")


def dice(pred: np.ndarray, label: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    denom = 2 * c.tp + c.fp + c.fn
    return float(2 * c.tp / denom) if denom > 0 else float("nan")


# Dice and F1 are identical for binary segmentation; alias for clarity.
f1 = dice


def precision(pred: np.ndarray, label: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    denom = c.tp + c.fp
    return float(c.tp / denom) if denom > 0 else float("nan")


def recall(pred: np.ndarray, label: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    denom = c.tp + c.fn
    return float(c.tp / denom) if denom > 0 else float("nan")


def accuracy(pred: np.ndarray, label: np.ndarray, ignore_index: int = IGNORE_INDEX) -> float:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    return float((c.tp + c.tn) / c.n) if c.n > 0 else float("nan")


def per_class_accuracy(
    pred: np.ndarray,
    label: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, float]:
    c = confusion_matrix_2x2(pred, label, ignore_index)
    cls0 = c.tn / (c.tn + c.fp) if (c.tn + c.fp) > 0 else float("nan")
    cls1 = c.tp / (c.tp + c.fn) if (c.tp + c.fn) > 0 else float("nan")
    return {"class0_acc": float(cls0), "class1_acc": float(cls1)}


def cohen_kappa(
    pred: np.ndarray,
    label: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> float:
    """Cohen's κ = (p_o - p_e) / (1 - p_e), chance-corrected agreement."""
    c = confusion_matrix_2x2(pred, label, ignore_index)
    n = c.n
    if n == 0:
        return float("nan")
    p_o = (c.tp + c.tn) / n
    p_pos = (c.tp + c.fp) / n  # marginal for predicted positive
    y_pos = (c.tp + c.fn) / n  # marginal for actual positive
    p_e = p_pos * y_pos + (1 - p_pos) * (1 - y_pos)
    if np.isclose(1 - p_e, 0.0):
        return 1.0 if np.isclose(p_o, 1.0) else float("nan")
    return float((p_o - p_e) / (1 - p_e))


def summary(
    pred: np.ndarray,
    label: np.ndarray,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, float]:
    """Return every metric as a single flat dict — used by the ablation CSV."""
    c = confusion_matrix_2x2(pred, label, ignore_index)
    out: dict[str, float] = {
        "iou": iou(pred, label, ignore_index),
        "f1": f1(pred, label, ignore_index),
        "precision": precision(pred, label, ignore_index),
        "recall": recall(pred, label, ignore_index),
        "accuracy": accuracy(pred, label, ignore_index),
        "cohen_kappa": cohen_kappa(pred, label, ignore_index),
    }
    out.update(per_class_accuracy(pred, label, ignore_index))
    out.update({k: float(v) for k, v in c.as_dict().items()})
    return out


__all__ = [
    "IGNORE_INDEX",
    "Confusion",
    "accuracy",
    "cohen_kappa",
    "confusion_matrix_2x2",
    "dice",
    "f1",
    "iou",
    "per_class_accuracy",
    "precision",
    "recall",
    "summary",
]
