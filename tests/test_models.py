"""Tests for src/models/*, src/train/augment.py, src/inference/predict.py,
src/eval/fusion.py. All CPU-only, all quick."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import torch
from rasterio.transform import from_origin

from src.eval.fusion import fuse_agreement, fuse_union, fuse_weighted
from src.inference.predict import _cosine_window, predict_chip, predict_raster
from src.models.losses import BCEDiceLoss, DiceLoss
from src.models.unet import UNetConfig, build_unet, count_parameters, load_checkpoint
from src.train.augment import sanity_check_roundtrip, train_transform, val_transform


# ---------------- model ----------------

def test_unet_forward_shape_and_dtype() -> None:
    model = build_unet(UNetConfig(encoder_weights=None))  # skip ImageNet download
    x = torch.randn(2, 6, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)
    assert out.dtype == torch.float32


def test_unet_param_count_reasonable() -> None:
    model = build_unet(UNetConfig(encoder_weights=None))
    n = count_parameters(model)
    # ResNet-34 U-Net is typically ~24M params.
    assert 15e6 < n < 40e6, f"unexpected param count {n}"


def test_load_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = build_unet(UNetConfig(encoder_weights=None))
    ckpt = tmp_path / "ckpt.pt"
    torch.save({"model": model.state_dict(), "epoch": 3}, ckpt)
    loaded = load_checkpoint(str(ckpt), UNetConfig(encoder_weights=None))
    for a, b in zip(model.state_dict().values(), loaded.state_dict().values(), strict=True):
        assert torch.equal(a, b)


# ---------------- losses ----------------

def test_dice_loss_perfect_prediction() -> None:
    # Logits strongly predicting target → dice loss near 0.
    target = torch.tensor([[[0, 1, 0], [1, 1, 0]]], dtype=torch.int64)
    logits = (target.float() * 2 - 1) * 10  # +10 for 1s, -10 for 0s
    loss = DiceLoss()(logits, target)
    assert loss.item() < 0.05


def test_dice_loss_all_wrong() -> None:
    target = torch.zeros(1, 4, 4, dtype=torch.int64)
    target[..., :2] = 1
    logits = torch.full((1, 4, 4), -5.0)  # always predict 0
    loss = DiceLoss()(logits, target)
    assert loss.item() > 0.9


def test_bce_dice_respects_ignore_index() -> None:
    target = torch.full((1, 4, 4), -1, dtype=torch.int64)  # all ignore
    logits = torch.randn(1, 4, 4)
    loss = BCEDiceLoss()(logits, target)
    # With no valid pixels, the loss must be exactly zero.
    assert loss.item() == 0.0


def test_bce_dice_backward() -> None:
    # Smoke: gradient flows through the combined loss.
    target = torch.tensor([[[0, 1, 0], [1, 1, 0]]], dtype=torch.int64)
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    loss = BCEDiceLoss(alpha=0.5)(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


# ---------------- augmentation ----------------

def test_train_transform_preserves_band_count() -> None:
    rng = np.random.default_rng(0)
    chip = rng.uniform(0, 1, (64, 64, 6)).astype(np.float32)
    mask = rng.integers(0, 2, (64, 64), dtype=np.int64)
    out_chip, out_mask = sanity_check_roundtrip(train_transform(crop_size=32), chip, mask)
    assert out_chip.shape == (32, 32, 6)
    assert out_mask.shape == (32, 32)
    assert out_chip.dtype == np.float32


def test_val_transform_noop() -> None:
    rng = np.random.default_rng(1)
    chip = rng.uniform(0, 1, (32, 32, 6)).astype(np.float32)
    mask = rng.integers(0, 2, (32, 32), dtype=np.int64)
    out_chip, out_mask = sanity_check_roundtrip(val_transform(crop_size=None), chip, mask)
    assert out_chip.shape == chip.shape
    assert out_mask.shape == mask.shape


# ---------------- fusion ----------------

def test_fuse_weighted_monotone_weight() -> None:
    a = np.array([[0.9, 0.9], [0.9, 0.9]])
    b = np.array([[0.1, 0.1], [0.1, 0.1]])
    mostly_a = fuse_weighted(a, b, weight_a=0.9)
    mostly_b = fuse_weighted(a, b, weight_a=0.1)
    assert mostly_a.all()
    assert not mostly_b.any()


def test_fuse_agreement_and_union() -> None:
    a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 0], [0, 1, 0]], dtype=bool)
    assert np.array_equal(fuse_agreement(a, b), np.array([[1, 0, 0], [0, 0, 0]], dtype=bool))
    assert np.array_equal(fuse_union(a, b), np.array([[1, 1, 0], [0, 1, 0]], dtype=bool))


def test_fuse_weighted_rejects_bad_weight() -> None:
    with pytest.raises(ValueError):
        fuse_weighted(np.zeros((2, 2)), np.zeros((2, 2)), weight_a=1.5)


# ---------------- inference ----------------

def test_cosine_window_shape() -> None:
    w = _cosine_window(64)
    assert w.shape == (64, 64)
    assert 0.0 <= w.min() <= w.max() <= 1.0


def test_predict_chip_probability_range() -> None:
    model = build_unet(UNetConfig(encoder_weights=None)).eval()
    chip = np.random.default_rng(0).uniform(0, 1, (6, 64, 64)).astype(np.float32)
    probs = predict_chip(model, chip, device="cpu")
    assert probs.shape == (64, 64)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0


def test_predict_raster_writes_valid_output(tmp_path: Path) -> None:
    # Build a small 6-band synthetic raster.
    arr = np.random.default_rng(1).uniform(0, 1, (6, 96, 96)).astype(np.float32)
    src_path = tmp_path / "in.tif"
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": 6,
        "height": 96,
        "width": 96,
        "crs": "EPSG:32643",
        "transform": from_origin(500_000, 1_100_000, 10.0, 10.0),
    }
    with rasterio.open(src_path, "w", **profile) as dst:
        dst.write(arr)

    model = build_unet(UNetConfig(encoder_weights=None)).eval()
    out = predict_raster(model, src_path, tmp_path / "out.tif", tile=64, overlap=16, device="cpu")

    with rasterio.open(out) as dst:
        assert dst.count == 2  # probability + mask
        probs = dst.read(1)
        mask = dst.read(2)
    assert probs.shape == (96, 96)
    assert 0.0 <= probs.min() <= probs.max() <= 1.0
    assert set(np.unique(mask).astype(int).tolist()).issubset({0, 1})
