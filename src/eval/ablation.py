"""Ablation harness — run every classical DIP combination on the
Sen1Floods11 test split and aggregate metrics.

The ablation combinatorially varies:

- **Index**      — which water-index family to threshold  (NDWI | MNDWI | AWEInsh | AWEIsh)
- **Threshold**  — threshold selection method             (Otsu | Triangle | Yen | Li)
- **Morphology** — whether to apply :func:`dip.morphology.clean`  (on | off)

Total: 4 × 4 × 2 = **32 combinations** per chip. Over the standard 89-chip
test split that's 2,848 evaluations — a few minutes on Colab CPU.

Output per ``run_ablation`` call:

1. ``results/ablation.csv`` — long-format CSV with one row per **combination**
   aggregating mean metrics across all chips.
2. ``results/per_chip/<config>.npz`` — per-chip IoU arrays for each config,
   enabling the downstream McNemar / bootstrap comparisons.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.data.sen1floods11_loader import LABEL_IGNORE_INDEX, Sen1Floods11Dataset
from src.dip import indices, morphology, thresholding
from src.eval.metrics import summary
from src.utils.logging import get_logger

log = get_logger(__name__)

# ---- Search space ----------------------------------------------------------

INDEX_FNS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "ndwi": indices.ndwi,
    "mndwi": indices.mndwi,
    "awei_nsh": indices.awei_nsh,
    "awei_sh": indices.awei_sh,
}

THRESHOLD_FNS: dict[str, Callable[[np.ndarray], thresholding.ThresholdResult]] = {
    "otsu": thresholding.otsu,
    "triangle": thresholding.triangle,
    "yen": thresholding.yen,
    "li": thresholding.li,
}


@dataclass(frozen=True)
class AblationConfig:
    index: str
    threshold: str
    morphology: bool

    @property
    def name(self) -> str:
        return f"{self.index}_{self.threshold}_{'morph' if self.morphology else 'raw'}"


def all_configs() -> list[AblationConfig]:
    return [
        AblationConfig(index=i, threshold=t, morphology=m)
        for i, t, m in itertools.product(INDEX_FNS, THRESHOLD_FNS, (True, False))
    ]


def predict(image: np.ndarray, cfg: AblationConfig) -> np.ndarray:
    """Apply one classical-DIP configuration to a (C,H,W) reflectance chip.

    Returns a bool mask of the same (H, W).
    """
    idx_map = INDEX_FNS[cfg.index](image)
    # Replace non-finite pixels with the minimum finite value so thresholding
    # doesn't crash on NaN — these pixels will be filtered out by the label's
    # ignore_index anyway.
    finite_mask = np.isfinite(idx_map)
    if not finite_mask.all():
        idx_map = np.where(finite_mask, idx_map, idx_map[finite_mask].min())

    result = THRESHOLD_FNS[cfg.threshold](idx_map)
    mask = result.mask
    if cfg.morphology:
        mask = morphology.clean(mask)
    return mask


# ---- Aggregation -----------------------------------------------------------

def _fresh_accumulator() -> dict[str, list[float]]:
    return {
        "iou": [], "f1": [], "precision": [], "recall": [],
        "accuracy": [], "cohen_kappa": [],
        "class0_acc": [], "class1_acc": [],
        "tp": [], "fp": [], "fn": [], "tn": [],
    }


@dataclass
class AblationRow:
    config: AblationConfig
    n_chips: int
    per_chip_iou: np.ndarray
    mean: dict[str, float]
    std: dict[str, float]


def run_ablation(
    dataset: Sen1Floods11Dataset,
    configs: list[AblationConfig] | None = None,
    limit: int | None = None,
    results_dir: str | Path = "results",
) -> pd.DataFrame:
    """Execute the ablation and persist results.

    Parameters
    ----------
    dataset
        A :class:`Sen1Floods11Dataset` yielding ``{'image', 'label', ...}``
        dicts. Usually the ``test`` split.
    configs
        Subset of configurations to evaluate. Defaults to :func:`all_configs`.
    limit
        If given, evaluate on only the first ``limit`` chips (smoke test).
    results_dir
        Where to write ``ablation.csv`` and ``per_chip/*.npz``.

    Returns
    -------
    DataFrame with one row per configuration and columns for every metric.
    """
    configs = configs or all_configs()
    results_dir = Path(results_dir)
    per_chip_dir = results_dir / "per_chip"
    per_chip_dir.mkdir(parents=True, exist_ok=True)

    n_chips = len(dataset) if limit is None else min(limit, len(dataset))
    log.info("Ablation: %d configs × %d chips = %d runs", len(configs), n_chips, len(configs) * n_chips)

    # Preload images and labels to avoid disk thrash — Sen1Floods11 test split
    # is only ~89 chips at 64×64 float32 = < 20 MB.
    log.info("Preloading %d chips", n_chips)
    images: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for i in tqdm(range(n_chips), desc="preload", leave=False):
        sample = dataset[i]
        images.append(sample["image"].numpy())
        labels.append(sample["label"].numpy())

    rows: list[AblationRow] = []
    t0 = time.time()

    for cfg in tqdm(configs, desc="configs"):
        acc = _fresh_accumulator()
        chip_iou = np.empty(n_chips, dtype=np.float64)
        for k, (img, y) in enumerate(zip(images, labels, strict=True)):
            mask = predict(img, cfg)
            m = summary(mask, y, ignore_index=LABEL_IGNORE_INDEX)
            for key, arr in acc.items():
                arr.append(m[key])
            chip_iou[k] = m["iou"]

        mean = {k: float(np.nanmean(v)) for k, v in acc.items()}
        std = {k: float(np.nanstd(v)) for k, v in acc.items()}

        # Persist per-chip IoU for downstream significance tests.
        np.savez(per_chip_dir / f"{cfg.name}.npz", iou=chip_iou)

        rows.append(AblationRow(cfg, n_chips, chip_iou, mean, std))

    df = pd.DataFrame([
        {
            "config": r.config.name,
            "index": r.config.index,
            "threshold": r.config.threshold,
            "morphology": r.config.morphology,
            "n_chips": r.n_chips,
            **{f"mean_{k}": v for k, v in r.mean.items()},
            **{f"std_{k}": v for k, v in r.std.items()},
        }
        for r in rows
    ])

    csv_path = results_dir / "ablation.csv"
    df.to_csv(csv_path, index=False)
    log.info("Wrote %s in %.1fs", csv_path, time.time() - t0)
    return df


def load_per_chip_iou(config_name: str, results_dir: str | Path = "results") -> np.ndarray:
    """Load per-chip IoU written by :func:`run_ablation`."""
    path = Path(results_dir) / "per_chip" / f"{config_name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"No per-chip record for '{config_name}' at {path}")
    return np.load(path)["iou"]


__all__ = [
    "AblationConfig",
    "AblationRow",
    "INDEX_FNS",
    "THRESHOLD_FNS",
    "all_configs",
    "load_per_chip_iou",
    "predict",
    "run_ablation",
]
