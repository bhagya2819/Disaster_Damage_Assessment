"""Unified flood-segmentation pipeline.

A single entrypoint, :func:`run_pipeline`, that takes a (C, H, W) reflectance
chip and a method name in {"classical", "unet", "hybrid"} and returns:

- ``mask`` — bool (H, W) binary flood mask.
- ``probs`` — float32 (H, W) probability map (1.0 for classical, sigmoid(logits) for U-Net).
- ``stats`` — reporting-ready dict (flooded_fraction, flooded_km2, …).
- ``method`` — the method actually used (may differ from requested if a
  fallback occurred, e.g. hybrid without a checkpoint).

The Streamlit app and the PDF report generator both call this — they never
reach into model / index / morphology code directly. This keeps the UI layer
thin and means any future method can be added in one place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from src.eval.ablation import AblationConfig, predict as classical_predict
from src.eval.fusion import fuse_weighted
from src.inference.predict import predict_chip
from src.models.unet import UNetConfig, load_checkpoint
from src.utils.logging import get_logger

log = get_logger(__name__)

Method = Literal["classical", "unet", "hybrid"]


@dataclass(frozen=True)
class PipelineResult:
    method: Method
    mask: np.ndarray       # bool (H, W)
    probs: np.ndarray      # float32 (H, W), in [0, 1]
    stats: dict[str, float]
    runtime_ms: float

    def as_dict(self) -> dict:
        return {
            "method": self.method,
            "mask_shape": list(self.mask.shape),
            "stats": self.stats,
            "runtime_ms": self.runtime_ms,
        }


def _stats_for_mask(
    mask: np.ndarray,
    pixel_size_m: float = 10.0,
) -> dict[str, float]:
    """Compute reporting stats from a binary flood mask.

    Assumes all pixels are ``pixel_size_m`` m on a side (Sen1Floods11 and
    Sentinel-2 default).
    """
    flooded_px = int(mask.sum())
    total_px = int(mask.size)
    px_area_m2 = pixel_size_m * pixel_size_m
    return {
        "flooded_px": flooded_px,
        "total_px": total_px,
        "flooded_fraction": float(flooded_px / total_px) if total_px else 0.0,
        "flooded_km2": flooded_px * px_area_m2 / 1e6,
        "total_km2": total_px * px_area_m2 / 1e6,
        "pixel_size_m": pixel_size_m,
    }


def run_pipeline(
    chip: np.ndarray,
    method: Method = "unet",
    checkpoint_path: str | None = None,
    device: str | torch.device = "cpu",
    classical_index: str = "ndwi",
    classical_threshold: str = "yen",
    classical_morphology: bool = False,
    hybrid_weight_unet: float = 0.7,
    pixel_size_m: float = 10.0,
    unet_cfg: UNetConfig | None = None,
) -> PipelineResult:
    """Run the chosen method on a single reflectance chip.

    Parameters
    ----------
    chip : np.ndarray
        (C, H, W) float32 reflectance in [0, 1], 6 DDA bands.
    method
        "classical" · "unet" · "hybrid".
    checkpoint_path
        Required for "unet" and "hybrid". If absent for "hybrid", falls back
        to "classical".
    device
        "cuda" · "cpu" · ``torch.device``. Used by "unet" / "hybrid".

    Returns
    -------
    :class:`PipelineResult`.
    """
    if chip.ndim != 3:
        raise ValueError(f"chip must be (C, H, W); got {chip.shape}")

    t0 = time.perf_counter()

    if method == "classical":
        classical_cfg = AblationConfig(
            index=classical_index, threshold=classical_threshold, morphology=classical_morphology
        )
        mask = classical_predict(chip, classical_cfg)
        probs = mask.astype(np.float32)

    elif method == "unet":
        if checkpoint_path is None:
            raise ValueError("method='unet' requires checkpoint_path")
        model = load_checkpoint(checkpoint_path, cfg=unet_cfg, device=device)
        probs = predict_chip(model, chip, device=device)
        mask = probs >= 0.5

    elif method == "hybrid":
        if checkpoint_path is None:
            log.warning("Hybrid requested but no checkpoint_path — falling back to classical.")
            return run_pipeline(
                chip,
                method="classical",
                classical_index=classical_index,
                classical_threshold=classical_threshold,
                classical_morphology=classical_morphology,
                pixel_size_m=pixel_size_m,
            )
        classical_cfg = AblationConfig(
            index=classical_index, threshold=classical_threshold, morphology=classical_morphology
        )
        classical_mask = classical_predict(chip, classical_cfg)
        model = load_checkpoint(checkpoint_path, cfg=unet_cfg, device=device)
        unet_probs = predict_chip(model, chip, device=device)
        probs = (
            hybrid_weight_unet * unet_probs
            + (1.0 - hybrid_weight_unet) * classical_mask.astype(np.float32)
        ).astype(np.float32)
        mask = fuse_weighted(
            unet_probs,
            classical_mask.astype(np.float32),
            weight_a=hybrid_weight_unet,
            threshold=0.5,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    runtime_ms = (time.perf_counter() - t0) * 1000
    stats = _stats_for_mask(mask, pixel_size_m=pixel_size_m)
    return PipelineResult(
        method=method, mask=mask, probs=probs, stats=stats, runtime_ms=runtime_ms
    )


__all__ = ["Method", "PipelineResult", "run_pipeline"]
