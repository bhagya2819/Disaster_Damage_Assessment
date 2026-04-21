"""U-Net for binary flood segmentation on Sentinel-2 chips.

Thin, typed wrapper around ``segmentation_models_pytorch.Unet`` so the rest
of the codebase can ``build_unet(...)`` without knowing SMP details.

Key choices (documented in the Phase-4 report):
- **Encoder**: ResNet-34 — good accuracy/compute trade-off for a free-tier
  Colab T4. Larger encoders (ResNet-50, EfficientNet-B0) blow past 4 GB VRAM
  at batch-size 8 on 512×512 input.
- **Weights**: ImageNet pretraining, re-adapted via ``in_channels=6`` (SMP
  handles the first-conv-layer expansion automatically).
- **Output**: single-channel logits (apply sigmoid downstream). The training
  loss uses ``BCEWithLogitsLoss`` which is numerically stabler than
  sigmoid+BCE.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


@dataclass(frozen=True)
class UNetConfig:
    in_channels: int = 6         # DDA subset: B2, B3, B4, B8, B11, B12
    out_channels: int = 1         # binary water / non-water
    encoder_name: str = "resnet34"
    encoder_weights: str | None = "imagenet"
    activation: str | None = None  # raw logits — sigmoid applied later
    decoder_channels: tuple[int, ...] = field(default_factory=lambda: (256, 128, 64, 32, 16))

    def summary(self) -> str:
        return (
            f"UNet(encoder={self.encoder_name}, weights={self.encoder_weights}, "
            f"in={self.in_channels}, out={self.out_channels})"
        )


def build_unet(cfg: UNetConfig | None = None) -> nn.Module:
    """Construct a fresh U-Net from the given config (defaults are fine for most runs)."""
    cfg = cfg or UNetConfig()
    model = smp.Unet(
        encoder_name=cfg.encoder_name,
        encoder_weights=cfg.encoder_weights,
        in_channels=cfg.in_channels,
        classes=cfg.out_channels,
        activation=cfg.activation,
        decoder_channels=list(cfg.decoder_channels),
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, cfg: UNetConfig | None = None, device: str | torch.device = "cpu") -> nn.Module:
    """Load trained weights into a fresh U-Net."""
    model = build_unet(cfg)
    state = torch.load(path, map_location=device, weights_only=False)
    # Support both plain state_dicts and {"model": ..., "optimizer": ...} wrappers.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.to(device).eval()
    return model


__all__ = ["UNetConfig", "build_unet", "count_parameters", "load_checkpoint"]
