"""PyTorch Dataset for the Sen1Floods11 HandLabeled split.

Dataset layout (after running ``scripts/download_sen1floods11.py --subset hand``)::

    <root>/
        data/flood_events/HandLabeled/
            S1Hand/       *.tif  (2 bands: VV, VH)
            S2Hand/       *.tif  (13 bands: B1..B12 + B8A)
            LabelHand/    *.tif  (1 band: 0=non-water, 1=water, -1=ignore)
        splits/flood_handlabeled/
            flood_train_data.csv
            flood_valid_data.csv
            flood_test_data.csv
            flood_bolivia_data.csv   (held-out region)

The CSVs list chip ids (e.g. ``Bolivia_103757``) which map to the basenames in
each of S1Hand/S2Hand/LabelHand.

This loader returns Sentinel-2 imagery by default (we use S2 for the course
project); set ``modality="s1"`` to use SAR.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

Split = Literal["train", "valid", "test", "bolivia"]
Modality = Literal["s1", "s2"]

# Sen1Floods11 stores S2Hand chips with 13 bands in the order
#   [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, QA60]
# (see https://github.com/cloudtostreet/Sen1Floods11).
#
# The DDA subset (defined in configs/kerala_2018.yaml) is
#   [B2, B3, B4, B8, B11, B12]
# which corresponds to 1-based rasterio band indices (2, 3, 4, 8, 11, 12).
#
# Prior versions of this loader used (1, 2, 3, 7, 10, 11) — an off-by-one that
# loaded the adjacent bands (B1/B2/B3/B7/B9/B11) and silently degraded every
# downstream spectral index.
S2_BAND_INDICES: tuple[int, ...] = (2, 3, 4, 8, 11, 12)  # 1-based TIFF band indices
S2_REFLECTANCE_SCALE: float = 10_000.0

# S1 is stored as linear-scale sigma-naught (already calibrated). We convert to
# decibels for a distribution closer to Gaussian.
S1_EPS: float = 1e-6

# Label convention in Sen1Floods11:
#   0 = non-water, 1 = water, -1 = no-label / ignore
LABEL_IGNORE_INDEX: int = -1


@dataclass(frozen=True)
class Sen1Floods11Paths:
    """Resolved on-disk paths for the HandLabeled subset."""

    root: Path

    @property
    def s1_dir(self) -> Path:
        return self.root / "data" / "flood_events" / "HandLabeled" / "S1Hand"

    @property
    def s2_dir(self) -> Path:
        return self.root / "data" / "flood_events" / "HandLabeled" / "S2Hand"

    @property
    def label_dir(self) -> Path:
        return self.root / "data" / "flood_events" / "HandLabeled" / "LabelHand"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits" / "flood_handlabeled"

    def split_csv(self, split: Split) -> Path:
        return self.splits_dir / f"flood_{split}_data.csv"

    def validate(self) -> None:
        for p in (self.s1_dir, self.s2_dir, self.label_dir, self.splits_dir):
            if not p.exists():
                raise FileNotFoundError(
                    f"Sen1Floods11 directory missing: {p}. Did you run "
                    f"scripts/download_sen1floods11.py --subset hand ?"
                )


def _read_split_csv(csv_path: Path) -> list[str]:
    """Read chip ids from a Sen1Floods11 split CSV.

    The CSVs in Sen1Floods11 have two columns (S1 path, label path); the chip
    id is the basename without extension. We accept either the full row or a
    single-column list of ids.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    ids: list[str] = []
    with csv_path.open() as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            basename = Path(row[0]).stem  # tolerates full path or bare id
            basename = basename.replace("_S1Hand", "").replace("_LabelHand", "")
            ids.append(basename)
    return ids


def _read_s2(path: Path) -> np.ndarray:
    """Read the 6-band DDA subset from a Sen1Floods11 S2 chip, as float32 reflectance."""
    with rasterio.open(path) as src:
        arr = src.read(list(S2_BAND_INDICES)).astype(np.float32)
    arr = arr / S2_REFLECTANCE_SCALE
    return np.clip(arr, 0.0, 1.0)


def _read_s1(path: Path) -> np.ndarray:
    """Read VV/VH as float32 dB."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)  # (2, H, W) linear sigma-naught
    arr = 10.0 * np.log10(np.clip(arr, S1_EPS, None))
    return arr


def _read_label(path: Path) -> np.ndarray:
    """Read the single-band label as int64 with values in {-1, 0, 1}."""
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.int64)
    return arr


class Sen1Floods11Dataset(Dataset):
    """PyTorch Dataset over the HandLabeled split.

    Parameters
    ----------
    root : str | Path
        Top-level directory where ``scripts/download_sen1floods11.py`` wrote data.
    split : {"train", "valid", "test", "bolivia"}
        Which CSV split to load.
    modality : {"s2", "s1"}
        Which sensor stack to return as features.
    transform : callable, optional
        An ``albumentations.Compose`` taking ``image=..., mask=...``. If given,
        augmentations run on numpy (H, W, C) and the result is converted to
        tensors. If None, numpy → tensor conversion still happens.
    band_stats : dict[str, tuple[float, float]], optional
        Per-modality (mean, std) for z-score normalization. If None, images are
        left in their natural [0, 1] reflectance (S2) or dB (S1) range.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split = "train",
        modality: Modality = "s2",
        transform=None,  # type: ignore[no-untyped-def]
        band_stats: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.paths = Sen1Floods11Paths(root=Path(root))
        self.paths.validate()

        self.split = split
        self.modality = modality
        self.transform = transform
        self.band_stats = band_stats

        self.ids = _read_split_csv(self.paths.split_csv(split))
        if not self.ids:
            raise RuntimeError(f"Split '{split}' has 0 chips — CSV is empty?")

    def __len__(self) -> int:
        return len(self.ids)

    def _img_path(self, chip_id: str) -> Path:
        if self.modality == "s2":
            return self.paths.s2_dir / f"{chip_id}_S2Hand.tif"
        return self.paths.s1_dir / f"{chip_id}_S1Hand.tif"

    def _label_path(self, chip_id: str) -> Path:
        return self.paths.label_dir / f"{chip_id}_LabelHand.tif"

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if self.band_stats is None:
            return img
        mean, std = self.band_stats[self.modality]
        return (img - mean) / (std + 1e-8)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chip_id = self.ids[idx]
        img = _read_s2(self._img_path(chip_id)) if self.modality == "s2" else _read_s1(self._img_path(chip_id))
        label = _read_label(self._label_path(chip_id))

        img = self._normalize(img)

        if self.transform is not None:
            # albumentations expects HWC; our arrays are CHW.
            img_hwc = np.transpose(img, (1, 2, 0))
            out = self.transform(image=img_hwc, mask=label)
            img = np.transpose(out["image"], (2, 0, 1))
            label = out["mask"]

        return {
            "image": torch.from_numpy(np.ascontiguousarray(img)).float(),
            "label": torch.from_numpy(np.ascontiguousarray(label)).long(),
            "chip_id": chip_id,
        }


__all__ = [
    "LABEL_IGNORE_INDEX",
    "S2_BAND_INDICES",
    "Sen1Floods11Dataset",
    "Sen1Floods11Paths",
]
