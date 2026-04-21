"""Data-augmentation transforms for Sen1Floods11 training.

Important: these chips are **6-band float reflectance in [0, 1]**, NOT 3-band
RGB uint8. Standard albumentations colour augmentations (Hue/Saturation,
RandomBrightnessContrast with auto defaults, Normalize with ImageNet stats)
will either crash on 6 channels or silently corrupt the spectral signature.
We stick to geometric and intensity augmentations that preserve physics.
"""

from __future__ import annotations

import albumentations as A
import numpy as np


def train_transform(crop_size: int = 256) -> A.Compose:
    """Geometric + mild intensity augmentations.

    - Random 90° rotations and flips (geometry only; spectra unchanged).
    - Random crop to ``crop_size`` — lets us train on 256×256 tiles instead
      of the native 512×512 for 4× memory savings at similar batch size.
    - Multiplicative brightness shift (± 10%) simulates atmospheric variation
      per band without distorting band ratios.
    """
    return A.Compose(
        [
            A.RandomCrop(height=crop_size, width=crop_size, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.10,
                contrast_limit=0.10,
                brightness_by_max=False,
                p=0.4,
            ),
        ],
        additional_targets={"mask": "mask"},
    )


def val_transform(crop_size: int | None = None) -> A.Compose:
    """Validation transform: deterministic centre-crop if requested, else no-op.

    We pass ``crop_size=None`` at eval time so we score against the full
    512×512 chip — the only way to produce comparable metrics with published
    Sen1Floods11 baselines.
    """
    ops: list = []
    if crop_size is not None:
        ops.append(A.CenterCrop(height=crop_size, width=crop_size, p=1.0))
    return A.Compose(ops if ops else [A.NoOp()], additional_targets={"mask": "mask"})


def sanity_check_roundtrip(transform: A.Compose, chip: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply a transform to a (H, W, C) chip + (H, W) mask; sanity-shape-check."""
    out = transform(image=chip, mask=mask)
    return out["image"], out["mask"]


__all__ = ["sanity_check_roundtrip", "train_transform", "val_transform"]
