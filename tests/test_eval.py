"""Tests for src/eval/*."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.ablation import (
    AblationConfig,
    all_configs,
    load_per_chip_iou,
    predict,
    run_ablation,
)
from src.eval.metrics import (
    IGNORE_INDEX,
    accuracy,
    cohen_kappa,
    confusion_matrix_2x2,
    dice,
    iou,
    per_class_accuracy,
    precision,
    recall,
    summary,
)
from src.eval.significance import (
    mcnemar_test,
    paired_bootstrap_iou,
    per_chip_iou,
)


# ---------- metrics: corner cases ----------

def test_all_correct_perfect_scores() -> None:
    y = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64)
    p = y.astype(bool)
    assert iou(p, y) == 1.0
    assert dice(p, y) == 1.0
    assert precision(p, y) == 1.0
    assert recall(p, y) == 1.0
    assert accuracy(p, y) == 1.0
    assert cohen_kappa(p, y) == pytest.approx(1.0)


def test_all_wrong_zero_iou() -> None:
    y = np.array([[0, 1, 0], [1, 1, 0]], dtype=np.int64)
    p = (~y.astype(bool))
    assert iou(p, y) == 0.0
    assert dice(p, y) == 0.0


def test_half_right_mixed() -> None:
    # 4 TP, 1 FP, 0 FN out of 5 positives; accuracy = 0.9
    y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    c = confusion_matrix_2x2(p, y)
    assert (c.tp, c.fp, c.fn, c.tn) == (4, 1, 1, 4)
    assert iou(p, y) == pytest.approx(4 / 6)
    assert dice(p, y) == pytest.approx(2 * 4 / (2 * 4 + 1 + 1))


def test_ignore_index_is_dropped() -> None:
    y = np.array([[0, 1, -1], [1, 1, -1]], dtype=np.int64)
    p = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
    # Ignoring the -1 pixels: 2 TP, 0 FP, 0 FN, 1 TN → IoU=1, accuracy=1.
    assert iou(p, y, ignore_index=IGNORE_INDEX) == 1.0
    assert accuracy(p, y, ignore_index=IGNORE_INDEX) == 1.0


def test_per_class_accuracy() -> None:
    y = np.array([[0, 0, 1, 1]])
    p = np.array([[0, 1, 1, 0]], dtype=bool)
    pca = per_class_accuracy(p, y)
    assert pca["class0_acc"] == 0.5
    assert pca["class1_acc"] == 0.5


def test_cohen_kappa_zero_when_random_agreement() -> None:
    # 50% accuracy but expected 50% by chance → κ ≈ 0
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = rng.integers(0, 2, 500).astype(bool)
    k = cohen_kappa(p, y)
    assert abs(k) < 0.2


def test_summary_returns_all_keys() -> None:
    y = np.array([[0, 1], [1, 0]])
    p = np.array([[0, 1], [1, 0]], dtype=bool)
    out = summary(p, y)
    for key in ("iou", "f1", "precision", "recall", "accuracy",
                "cohen_kappa", "class0_acc", "class1_acc", "tp", "fp", "fn", "tn"):
        assert key in out


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        iou(np.zeros((3, 3), dtype=bool), np.zeros((4, 4), dtype=np.int64))


# ---------- significance tests ----------

def test_mcnemar_identical_methods() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    p = (y.copy() ^ rng.integers(0, 2, 200)).astype(bool)  # noisy preds
    res = mcnemar_test(p, p, y)
    assert res.statistic == 0.0
    assert res.p_value == pytest.approx(1.0)


def test_mcnemar_one_always_better() -> None:
    # Method A always correct; method B always wrong → extremely significant.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1] * 20, dtype=np.int64)
    a = y.astype(bool)
    b = (~y.astype(bool))
    res = mcnemar_test(a, b, y)
    assert res.p_value < 0.001


def test_bootstrap_ci_covers_mean() -> None:
    a = np.array([0.80, 0.75, 0.90, 0.85, 0.70])
    b = np.array([0.70, 0.60, 0.85, 0.80, 0.65])
    boot = paired_bootstrap_iou(a, b, n_bootstrap=2000, seed=42)
    assert boot.ci_lower <= boot.mean_delta <= boot.ci_upper
    assert boot.mean_delta == pytest.approx((a - b).mean())


def test_per_chip_iou_length_matches() -> None:
    preds = [np.ones((4, 4), dtype=bool) for _ in range(3)]
    labels = [np.ones((4, 4), dtype=np.int64) for _ in range(3)]
    v = per_chip_iou(preds, labels)
    assert v.shape == (3,)
    assert np.allclose(v, 1.0)


# ---------- ablation harness ----------

def test_all_configs_count() -> None:
    cfgs = all_configs()
    # 4 indices × 4 thresholds × 2 morphology = 32
    assert len(cfgs) == 32
    names = {c.name for c in cfgs}
    assert len(names) == 32  # all unique


def test_predict_returns_boolean_mask() -> None:
    # Synthetic 6-band reflectance: water pixels at top-half
    img = np.zeros((6, 16, 16), dtype=np.float32)
    img[:, :8, :] = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01])[:, None, None]
    img[:, 8:, :] = np.array([0.05, 0.08, 0.04, 0.45, 0.25, 0.15])[:, None, None]

    cfg = AblationConfig(index="mndwi", threshold="otsu", morphology=True)
    mask = predict(img, cfg)
    assert mask.dtype == bool
    assert mask.shape == (16, 16)
    # Top half should be predominantly classified as water.
    assert mask[:8].mean() > mask[8:].mean() + 0.5


def test_run_ablation_smoke(sen1floods11_root: Path, tmp_path: Path) -> None:
    """Tiny end-to-end ablation on the synthetic Sen1Floods11 fixture."""
    from src.data.sen1floods11_loader import Sen1Floods11Dataset

    ds = Sen1Floods11Dataset(sen1floods11_root, split="train", modality="s2")

    cfgs = [
        AblationConfig("mndwi", "otsu", True),
        AblationConfig("ndwi", "triangle", False),
    ]
    df = run_ablation(ds, configs=cfgs, results_dir=tmp_path)

    assert len(df) == 2
    assert set(df["config"]) == {c.name for c in cfgs}
    assert (tmp_path / "ablation.csv").exists()
    # Per-chip IoU persisted
    for c in cfgs:
        arr = load_per_chip_iou(c.name, results_dir=tmp_path)
        assert arr.shape == (len(ds),)
