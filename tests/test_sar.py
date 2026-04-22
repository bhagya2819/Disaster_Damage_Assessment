"""Tests for src/dip/sar.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.dip import sar


def _synthetic_sar(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    """Build a synthetic VV linear backscatter image with a water/land split.

    Real Sentinel-1 γ⁰ values: land ≈ 0.05–0.3 (−13 to −5 dB),
    water ≈ 0.001–0.01 (−30 to −20 dB). We simulate with multiplicative
    gamma-distributed speckle.
    """
    rng = np.random.default_rng(seed)
    arr = np.empty((h, w), dtype=np.float32)
    # Left half: water, low backscatter.
    arr[:, : w // 2] = rng.gamma(shape=4.0, scale=0.001, size=(h, w // 2))
    # Right half: land, higher backscatter.
    arr[:, w // 2:] = rng.gamma(shape=4.0, scale=0.05, size=(h, w - w // 2))
    return arr


# ---------- to_db ----------

def test_to_db_basic() -> None:
    x = np.array([[1.0, 0.1, 0.01]], dtype=np.float32)
    out = sar.to_db(x)
    # 10 * log10([1, 0.1, 0.01]) = [0, -10, -20]
    np.testing.assert_allclose(out, np.array([[0.0, -10.0, -20.0]]), atol=1e-4)


def test_to_db_handles_zero() -> None:
    x = np.zeros((4, 4), dtype=np.float32)
    out = sar.to_db(x)
    assert np.all(np.isfinite(out))
    assert out.min() < 0  # clipped to EPS → large negative dB


# ---------- speckle filters ----------

def test_refined_lee_preserves_shape_and_dtype() -> None:
    img = _synthetic_sar()
    out = sar.refined_lee(img, window_size=7)
    assert out.shape == img.shape
    assert out.dtype == np.float32


def test_refined_lee_smooths_homogeneous_region() -> None:
    rng = np.random.default_rng(1)
    base = np.ones((64, 64), dtype=np.float32) * 0.05
    noisy = base + rng.normal(0, 0.01, base.shape).astype(np.float32)
    filtered = sar.refined_lee(noisy, window_size=7)
    # Standard deviation should drop meaningfully after filtering.
    assert filtered.std() < noisy.std() * 0.8


def test_refined_lee_odd_window_required() -> None:
    with pytest.raises(ValueError, match="odd"):
        sar.refined_lee(_synthetic_sar(), window_size=6)


def test_frost_filter_runs() -> None:
    out = sar.frost(_synthetic_sar(h=24, w=24), window_size=5)
    assert out.shape == (24, 24)


# ---------- sar_flood_mask ----------

def test_sar_flood_mask_detects_water_side() -> None:
    img = _synthetic_sar(seed=42)
    mask, thr = sar.sar_flood_mask(img)
    assert mask.shape == img.shape
    assert mask.dtype == bool
    # Typical Otsu dB threshold for flood water: between −25 and −10.
    assert -30.0 <= thr <= -5.0, f"Threshold {thr} dB outside plausible range"
    # Left half (water) should be mostly flagged; right half (land) should not.
    left = mask[:, : img.shape[1] // 2].mean()
    right = mask[:, img.shape[1] // 2 :].mean()
    assert left > right + 0.5, f"left={left:.3f} right={right:.3f}"


def test_sar_flood_mask_fixed_threshold() -> None:
    img = _synthetic_sar(seed=0)
    mask, thr = sar.sar_flood_mask(img, threshold_db=-20.0)
    assert thr == -20.0
    assert mask.dtype == bool


# ---------- fusion ----------

@pytest.fixture
def pair_masks():
    a = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)  # SAR
    b = np.array([[1, 1, 0], [0, 0, 1]], dtype=bool)  # optical
    return a, b


def test_fuse_union(pair_masks):
    a, b = pair_masks
    out = sar.fuse_with_optical(a, b, mode="union")
    np.testing.assert_array_equal(out, np.array([[1, 1, 1], [0, 0, 1]], dtype=bool))


def test_fuse_agreement(pair_masks):
    a, b = pair_masks
    out = sar.fuse_with_optical(a, b, mode="agreement")
    np.testing.assert_array_equal(out, np.array([[1, 0, 0], [0, 0, 0]], dtype=bool))


def test_fuse_optical_primary(pair_masks):
    a, b = pair_masks
    out = sar.fuse_with_optical(a, b, mode="optical_primary")
    # Where `b` has True, keep True; otherwise fall back to `a`.
    expected = np.array([[1, 1, 1], [0, 0, 1]], dtype=bool)
    np.testing.assert_array_equal(out, expected)


def test_fuse_invalid_mode(pair_masks):
    a, b = pair_masks
    with pytest.raises(ValueError, match="unknown mode"):
        sar.fuse_with_optical(a, b, mode="bananas")


def test_fuse_shape_mismatch():
    a = np.zeros((4, 4), dtype=bool)
    b = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="shape"):
        sar.fuse_with_optical(a, b)


# ---------- agreement fraction ----------

def test_agreement_fraction_identical() -> None:
    a = np.random.default_rng(0).integers(0, 2, (16, 16)).astype(bool)
    assert sar.agreement_fraction(a, a) == pytest.approx(1.0)


def test_agreement_fraction_opposite() -> None:
    a = np.array([[True, False], [True, False]])
    b = ~a
    # po = 0, pe = 0.5 (both are balanced) → κ = (0 - 0.5) / (1 - 0.5) = -1.
    k = sar.agreement_fraction(a, b)
    assert k < 0
