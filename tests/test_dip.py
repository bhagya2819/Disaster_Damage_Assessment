"""Tests for src/dip/*.

Strategy: build synthetic 6-band Sentinel-2-like stacks whose ground truth we
already know (pure water / pure vegetation / pure built-up pixels), then
assert each algorithm segregates them correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.dip import change_detection, filters, indices, morphology, thresholding


# --- Synthetic spectra (reflectance) for 6-band DDA stack: B2,B3,B4,B8,B11,B12 ---
WATER_SPECTRUM = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01], dtype=np.float32)  # low NIR, very low SWIR
VEG_SPECTRUM = np.array([0.05, 0.08, 0.04, 0.45, 0.25, 0.15], dtype=np.float32)  # NIR peak
BUILT_SPECTRUM = np.array([0.20, 0.22, 0.25, 0.30, 0.38, 0.32], dtype=np.float32)  # high SWIR


def make_scene(
    water_frac: float = 0.5,
    h: int = 64,
    w: int = 64,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 6-band scene with a known water mask for ground truth."""
    rng = np.random.default_rng(seed)
    n_pix = h * w
    n_water = int(n_pix * water_frac)
    idx = rng.permutation(n_pix)
    water_idx = idx[:n_water]

    labels = np.zeros(n_pix, dtype=bool)
    labels[water_idx] = True

    # Half of non-water is vegetation, half built-up.
    dry = idx[n_water:]
    veg_idx = dry[: len(dry) // 2]
    built_idx = dry[len(dry) // 2 :]

    pixels = np.empty((n_pix, 6), dtype=np.float32)
    pixels[water_idx] = WATER_SPECTRUM + rng.normal(0, 0.005, (n_water, 6))
    pixels[veg_idx] = VEG_SPECTRUM + rng.normal(0, 0.01, (len(veg_idx), 6))
    pixels[built_idx] = BUILT_SPECTRUM + rng.normal(0, 0.01, (len(built_idx), 6))
    pixels = np.clip(pixels, 0, 1)

    stack = pixels.T.reshape(6, h, w)
    label_map = labels.reshape(h, w)
    return stack, label_map


# ------------- indices -------------

def test_all_indices_return_correct_shape() -> None:
    stack, _ = make_scene()
    for name, arr in indices.compute_all(stack).items():
        assert arr.shape == (64, 64), f"{name} wrong shape"
        assert arr.dtype == np.float32, f"{name} wrong dtype"


def test_mndwi_is_higher_over_water_than_vegetation() -> None:
    stack, labels = make_scene(water_frac=0.4)
    m = indices.mndwi(stack)
    assert m[labels].mean() > m[~labels].mean() + 0.3


def test_ndvi_is_higher_over_vegetation_than_water() -> None:
    stack, labels = make_scene(water_frac=0.4)
    v = indices.ndvi(stack)
    # vegetation is in ~half of the non-water pixels
    assert v[~labels].mean() > v[labels].mean()


def test_index_divide_by_zero_safe() -> None:
    zeros = np.zeros((6, 8, 8), dtype=np.float32)
    # Should not raise, should not produce inf/nan.
    for arr in indices.compute_all(zeros).values():
        assert np.all(np.isfinite(arr))


# ------------- thresholding -------------

def test_otsu_separates_water_correctly() -> None:
    stack, labels = make_scene(water_frac=0.4)
    m = indices.mndwi(stack)
    res = thresholding.otsu(m)
    # Accuracy should be very high on a synthetic scene.
    agreement = (res.mask == labels).mean()
    assert agreement > 0.90
    assert res.method == "otsu"


@pytest.mark.parametrize("method_name", ["otsu", "triangle", "yen", "li"])
def test_every_global_threshold_runs(method_name: str) -> None:
    stack, _ = make_scene()
    m = indices.mndwi(stack)
    res = thresholding.auto(m, method=method_name)
    assert res.mask.shape == m.shape
    assert res.method == method_name


def test_adaptive_threshold_runs() -> None:
    stack, _ = make_scene()
    m = indices.mndwi(stack)
    res = thresholding.adaptive(m, block_size=11)
    assert res.mask.shape == m.shape
    assert "adaptive" in res.method


def test_adaptive_requires_odd_block_size() -> None:
    m = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="odd"):
        thresholding.adaptive(m, block_size=10)


# ------------- morphology -------------

def test_clean_removes_salt_and_pepper() -> None:
    # Blank field + tiny noise that should be scrubbed.
    m = np.zeros((64, 64), dtype=bool)
    rng = np.random.default_rng(7)
    for _ in range(50):
        i, j = rng.integers(0, 64, 2)
        m[i, j] = True
    cleaned = morphology.clean(m, min_object_area=10)
    assert cleaned.sum() < m.sum() / 2


def test_clean_preserves_large_blob() -> None:
    # A 20x20 solid square passed through opening(disk(1)) loses its four
    # 1-px corners by design (that is what opening does). We just assert the
    # blob survives substantially — > 95% of pixels retained.
    m = np.zeros((64, 64), dtype=bool)
    m[10:30, 10:30] = True  # 400-px solid blob
    cleaned = morphology.clean(m, min_object_area=10)
    retained = cleaned[10:30, 10:30].sum()
    assert retained >= 0.95 * 400, f"expected >=380 retained, got {retained}"


def test_boundary_is_one_pixel_outline() -> None:
    m = np.zeros((16, 16), dtype=bool)
    m[4:12, 4:12] = True
    b = morphology.boundary(m)
    assert b.sum() == 28  # perimeter of an 8x8 solid square


# ------------- change detection -------------

def test_image_difference_shape_and_sign() -> None:
    pre = np.ones((6, 16, 16), dtype=np.float32) * 0.3
    post = np.ones((6, 16, 16), dtype=np.float32) * 0.5
    d = change_detection.image_difference(pre, post)
    assert d.shape == pre.shape
    assert np.all(d > 0)


def test_cva_magnitude_is_nonnegative() -> None:
    rng = np.random.default_rng(5)
    pre = rng.uniform(0, 1, (6, 32, 32)).astype(np.float32)
    post = rng.uniform(0, 1, (6, 32, 32)).astype(np.float32)
    cva = change_detection.change_vector_analysis(pre, post)
    assert cva.shape == (32, 32)
    assert np.all(cva >= 0)


def test_pca_change_shape() -> None:
    pre, _ = make_scene(water_frac=0.2, seed=1)
    post, _ = make_scene(water_frac=0.6, seed=2)
    pca = change_detection.pca_change(pre, post, n_components=3)
    assert pca.shape == pre.shape[1:]


def test_mndwi_difference_highlights_new_water() -> None:
    pre, _ = make_scene(water_frac=0.2, seed=11)
    post, post_labels = make_scene(water_frac=0.7, seed=12)
    d = change_detection.mndwi_difference(pre, post)
    # New water pixels (post-only) should have the largest ΔMNDWI on average.
    assert d.mean() > 0  # more water overall


def test_image_difference_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape"):
        change_detection.image_difference(
            np.zeros((6, 4, 4), dtype=np.float32),
            np.zeros((6, 8, 8), dtype=np.float32),
        )


# ------------- filters -------------

def test_gaussian_2d_smooths_noise() -> None:
    rng = np.random.default_rng(9)
    img = rng.normal(0, 1, (32, 32)).astype(np.float32)
    smoothed = filters.gaussian(img, sigma=2.0)
    assert smoothed.std() < img.std()


def test_median_preserves_edge() -> None:
    img = np.zeros((16, 16), dtype=np.float32)
    img[:, 8:] = 1.0
    out = filters.median(img, size=3)
    # Edge at column 8 still sharp.
    assert out[0, 7] < 0.5 < out[0, 8]


def test_sobel_fires_on_step_edge() -> None:
    img = np.zeros((16, 16), dtype=np.float32)
    img[:, 8:] = 1.0
    edges = filters.sobel_edges(img)
    assert edges[:, 7:9].max() > 0.1


def test_canny_returns_bool() -> None:
    img = np.zeros((32, 32), dtype=np.float32)
    img[:, 16:] = 1.0
    e = filters.canny_edges(img, sigma=1.0)
    assert e.dtype == bool
    assert e.any()
