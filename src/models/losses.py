"""Loss functions for binary flood segmentation.

Two losses and a combined ``BCEDiceLoss``:

- **BCE** — pixel-wise binary cross-entropy, gives well-calibrated
  probabilities but is class-frequency sensitive.
- **Soft Dice** — directly optimises Dice/IoU, robust to class imbalance.
- **BCEDiceLoss** — α·BCE + (1−α)·Dice. α default 0.5 in the literature.

All three honour the Sen1Floods11 ignore index (``-1``): pixels with that
label are removed from the loss computation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE_INDEX: int = -1


def _mask_ignore(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten and drop ignore-labelled pixels.

    Returns (logits_flat, target_flat_float, valid_mask_flat).
    """
    if logits.ndim == 4 and logits.size(1) == 1:
        logits = logits.squeeze(1)
    assert logits.shape == target.shape, f"{logits.shape} vs {target.shape}"

    valid = target != IGNORE_INDEX
    return logits[valid], target[valid].float(), valid


class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation. Robust to class imbalance."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f, target_f, _ = _mask_ignore(logits, target)
        if logits_f.numel() == 0:
            return logits.new_zeros(())
        probs = torch.sigmoid(logits_f)
        intersection = (probs * target_f).sum()
        denom = probs.sum() + target_f.sum() + self.smooth
        return 1.0 - (2.0 * intersection + self.smooth) / denom


class BCEDiceLoss(nn.Module):
    """Combined BCE + Dice.

    ``alpha`` is the weight on BCE; Dice gets ``(1 - alpha)``.
    """

    def __init__(self, alpha: float = 0.5, pos_weight: float | None = None, smooth: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.smooth = smooth
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None,
            persistent=False,
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits_f, target_f, _ = _mask_ignore(logits, target)
        if logits_f.numel() == 0:
            return logits.new_zeros(())

        bce = F.binary_cross_entropy_with_logits(
            logits_f, target_f, pos_weight=self.pos_weight
        )

        probs = torch.sigmoid(logits_f)
        intersection = (probs * target_f).sum()
        denom = probs.sum() + target_f.sum() + self.smooth
        dice = 1.0 - (2.0 * intersection + self.smooth) / denom

        return self.alpha * bce + (1.0 - self.alpha) * dice


__all__ = ["BCEDiceLoss", "DiceLoss", "IGNORE_INDEX"]
