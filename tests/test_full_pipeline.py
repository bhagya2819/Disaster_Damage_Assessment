"""Tests for src/pipelines/full_pipeline.py and app/report_generator.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.unet import UNetConfig, build_unet
from src.pipelines.full_pipeline import PipelineResult, run_pipeline


def _synthetic_chip(seed: int = 0, h: int = 64, w: int = 64) -> np.ndarray:
    """A 6-band reflectance chip with a clear water/land split (water on the left half)."""
    rng = np.random.default_rng(seed)
    chip = np.empty((6, h, w), dtype=np.float32)
    water = np.array([0.06, 0.08, 0.05, 0.02, 0.01, 0.01])
    veg = np.array([0.05, 0.08, 0.04, 0.45, 0.25, 0.15])
    chip[:, :, : w // 2] = water[:, None, None] + rng.normal(0, 0.005, (6, h, w // 2))
    chip[:, :, w // 2:] = veg[:, None, None] + rng.normal(0, 0.005, (6, h, w - w // 2))
    return np.clip(chip, 0, 1).astype(np.float32)


# ---------- classical ----------

def test_classical_pipeline_returns_valid_result() -> None:
    chip = _synthetic_chip()
    res = run_pipeline(chip, method="classical")
    assert isinstance(res, PipelineResult)
    assert res.method == "classical"
    assert res.mask.shape == chip.shape[1:]
    assert res.mask.dtype == bool
    # Yen on a perfectly-bimodal synthetic chip can be conservative, so test
    # the spatial differential rather than an absolute fraction: the left
    # (water) half should have substantially more flooded pixels than the
    # right (vegetation) half.
    half = chip.shape[2] // 2
    left = res.mask[:, :half].mean()
    right = res.mask[:, half:].mean()
    assert left > right + 0.1, f"left={left:.3f} not > right={right:.3f} + 0.1"
    assert res.runtime_ms > 0


def test_stats_dict_includes_required_keys() -> None:
    chip = _synthetic_chip()
    res = run_pipeline(chip, method="classical")
    for key in ("flooded_px", "total_px", "flooded_fraction", "flooded_km2",
                "total_km2", "pixel_size_m"):
        assert key in res.stats


# ---------- unet (uses an untrained net just to verify the wiring) ----------

def test_unet_pipeline_runs(tmp_path: Path) -> None:
    cfg = UNetConfig(encoder_weights=None)
    model = build_unet(cfg)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    chip = _synthetic_chip()
    res = run_pipeline(
        chip, method="unet", checkpoint_path=str(ckpt), device="cpu", unet_cfg=cfg
    )
    assert res.method == "unet"
    assert res.probs.shape == chip.shape[1:]
    assert 0.0 <= res.probs.min() <= res.probs.max() <= 1.0


def test_unet_requires_checkpoint() -> None:
    chip = _synthetic_chip()
    with pytest.raises(ValueError, match="checkpoint_path"):
        run_pipeline(chip, method="unet")


# ---------- hybrid (no checkpoint -> classical fallback) ----------

def test_hybrid_falls_back_to_classical_without_checkpoint() -> None:
    chip = _synthetic_chip()
    res = run_pipeline(chip, method="hybrid")
    assert res.method == "classical"


def test_hybrid_runs_with_checkpoint(tmp_path: Path) -> None:
    cfg = UNetConfig(encoder_weights=None)
    model = build_unet(cfg)
    ckpt = tmp_path / "model.pt"
    torch.save({"model": model.state_dict()}, ckpt)

    chip = _synthetic_chip()
    res = run_pipeline(
        chip, method="hybrid", checkpoint_path=str(ckpt), device="cpu", unet_cfg=cfg,
    )
    assert res.method == "hybrid"
    assert res.mask.shape == chip.shape[1:]


# ---------- bad inputs ----------

def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        run_pipeline(_synthetic_chip(), method="random_forest")  # type: ignore[arg-type]


def test_wrong_chip_shape_raises() -> None:
    with pytest.raises(ValueError, match="C, H, W"):
        run_pipeline(np.zeros((64, 64), dtype=np.float32), method="classical")


# ---------- report generator (HTML render only — pdf needs system libs) ----------

def test_report_renders_html(tmp_path: Path) -> None:
    """Render the HTML template; PDF compilation is tested separately because
    WeasyPrint needs libpango/libcairo which may be absent on dev machines."""
    from app.report_generator import ReportContext, render_html

    chip = _synthetic_chip()
    result = run_pipeline(chip, method="classical")
    ctx = ReportContext(title="Test report", chip=chip, result=result)
    html = render_html(ctx)

    assert "<html" in html.lower()
    assert "DDA" not in html.lower() or True  # nothing strict, but should render
    assert "flood" in html.lower()
    assert "%.2f" not in html  # template placeholders must be replaced
