"""Tiled U-Net inference for arbitrary-size Sentinel-2 rasters.

Two public entry points:

- :func:`predict_chip`       — single in-memory (C, H, W) float32 reflectance array.
  Fine for Sen1Floods11 512×512 chips. Returns an (H, W) probability map.

- :func:`predict_raster`     — GeoTIFF-in, GeoTIFF-out. Tiles the input raster with
  configurable tile size and overlap, blends overlapping predictions with a
  cosine window, and preserves the source CRS/transform.

Tile blending uses a 2-D raised-cosine window so that predictions at tile
edges (where the receptive field is truncated) contribute less than predictions
from tile centres — this removes visible seams.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import torch
from torch import nn
from tqdm.auto import tqdm

from src.utils.logging import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------
# In-memory chip inference
# --------------------------------------------------------------------------

@torch.no_grad()
def predict_chip(
    model: nn.Module,
    chip: np.ndarray,
    device: str | torch.device = "cpu",
    amp: bool = False,
) -> np.ndarray:
    """Run the model on a (C, H, W) reflectance chip.

    Returns an (H, W) float32 probability map in [0, 1].
    """
    model.eval()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    t = torch.from_numpy(np.ascontiguousarray(chip)).float().unsqueeze(0).to(device)
    with torch.amp.autocast(device_type=device.type, enabled=amp):
        logits = model(t)
    probs = torch.sigmoid(logits.squeeze(1).squeeze(0)).float().cpu().numpy()
    return probs


# --------------------------------------------------------------------------
# Tile blending
# --------------------------------------------------------------------------

def _cosine_window(size: int) -> np.ndarray:
    """2-D raised-cosine window in [0, 1] with soft edges."""
    w1 = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(size) / max(size - 1, 1))
    return (w1[:, None] * w1[None, :]).astype(np.float32)


@torch.no_grad()
def predict_raster(
    model: nn.Module,
    src_path: str | Path,
    dst_path: str | Path,
    bands: tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    tile: int = 512,
    overlap: int = 64,
    threshold: float | None = 0.5,
    device: str | torch.device = "cpu",
    amp: bool = False,
) -> Path:
    """Run the model over an arbitrary-size GeoTIFF and write a mask + prob GeoTIFF.

    Parameters
    ----------
    model
        Trained U-Net (already on ``device``).
    src_path
        Input GeoTIFF with at least ``max(bands)`` bands of float reflectance.
        Our GEE downloader produces this directly.
    dst_path
        Output path. Two bands: (1) flood probability, (2) binary mask.
    bands
        1-based rasterio band indices to feed the model. Default matches the
        DDA 6-band export (B2, B3, B4, B8, B11, B12).
    tile
        Tile side length (px). Should be ≥ 128; U-Net needs ≥ 32 for the
        bottleneck, and IoU-stable results need ≥ 256.
    overlap
        Pixels of overlap between adjacent tiles. Larger = smoother seams,
        slower inference.
    threshold
        If not None, binarise the probability map at this value; else only
        the probability band is written.
    """
    src_path, dst_path = Path(src_path), Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if overlap >= tile:
        raise ValueError(f"overlap ({overlap}) must be < tile ({tile})")
    if tile <= 0:
        raise ValueError("tile must be positive")

    model.eval()
    device = torch.device(device) if not isinstance(device, torch.device) else device
    model.to(device)

    with rasterio.open(src_path) as src:
        H, W = src.height, src.width
        profile = src.profile.copy()
        stack = src.read(list(bands)).astype(np.float32)

    log.info("Inference on %s  shape=(%d, %d)  tiles=%d overlap=%d", src_path, H, W, tile, overlap)

    step = tile - overlap
    probs = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)
    win = _cosine_window(tile)

    ys = list(range(0, max(H - tile, 0) + 1, step))
    xs = list(range(0, max(W - tile, 0) + 1, step))
    if ys[-1] != max(H - tile, 0):
        ys.append(max(H - tile, 0))
    if xs[-1] != max(W - tile, 0):
        xs.append(max(W - tile, 0))

    total = len(ys) * len(xs)
    pbar = tqdm(total=total, desc="tiles", leave=False)
    for y in ys:
        for x in xs:
            patch = stack[:, y:y + tile, x:x + tile]
            if patch.shape[1:] != (tile, tile):
                # edge tile smaller than nominal — pad with zeros.
                padded = np.zeros((patch.shape[0], tile, tile), dtype=np.float32)
                padded[:, : patch.shape[1], : patch.shape[2]] = patch
                patch = padded

            p = predict_chip(model, patch, device=device, amp=amp)
            probs[y:y + tile, x:x + tile] += p * win
            weight[y:y + tile, x:x + tile] += win
            pbar.update(1)
    pbar.close()

    # Normalize by accumulated weights.
    with np.errstate(invalid="ignore"):
        probs = np.where(weight > 0, probs / np.maximum(weight, 1e-8), 0.0).astype(np.float32)

    out_profile = profile.copy()
    out_profile.update(
        count=2 if threshold is not None else 1,
        dtype="float32",
        nodata=None,
        compress="lzw",
    )
    with rasterio.open(dst_path, "w", **out_profile) as dst:
        dst.write(probs, 1)
        dst.set_band_description(1, "flood_probability")
        if threshold is not None:
            mask = (probs >= threshold).astype(np.float32)
            dst.write(mask, 2)
            dst.set_band_description(2, f"flood_mask@{threshold}")

    log.info("Wrote %s", dst_path)
    return dst_path


__all__ = ["predict_chip", "predict_raster"]
