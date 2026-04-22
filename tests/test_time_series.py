"""Tests for src/analysis/time_series.py."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from src.analysis.time_series import (
    TimeSeriesSummary,
    build_area_curve_png,
    build_gif,
    flooded_km2_per_step,
    summarise,
)


def _rising_falling_masks(h: int = 16, w: int = 16) -> list[np.ndarray]:
    """Build 5 masks: empty → 25 % → 75 % → 50 % → 10 %."""
    fractions = [0.0, 0.25, 0.75, 0.50, 0.10]
    rng = np.random.default_rng(0)
    masks: list[np.ndarray] = []
    for f in fractions:
        m = np.zeros((h, w), dtype=bool)
        n = int(f * h * w)
        if n:
            idx = rng.choice(h * w, size=n, replace=False)
            m.flat[idx] = True
        masks.append(m)
    return masks


def _five_dates() -> list[datetime]:
    base = datetime(2018, 8, 15)
    return [base + timedelta(days=i * 2) for i in range(5)]


def test_flooded_km2_per_step_monotonic_with_fraction() -> None:
    masks = _rising_falling_masks()
    km2 = flooded_km2_per_step(masks, pixel_size_m=10.0)
    # Peak mask (75 %) has the largest area.
    assert km2[2] > km2[1]
    assert km2[2] > km2[3]
    # Empty mask has 0.
    assert km2[0] == 0.0


def test_summarise_identifies_peak_and_recession() -> None:
    masks = _rising_falling_masks()
    dates = _five_dates()
    s = summarise(masks, dates)
    assert isinstance(s, TimeSeriesSummary)
    assert s.peak_date == dates[2]
    assert s.time_to_peak_days == 4        # 2 steps × 2 days
    # Recession: first date after peak where flooded ≤ peak/2.
    # Peak = 0.75, half = 0.375. Index 3 = 0.50 (not ≤), index 4 = 0.10 (≤).
    assert s.time_to_half_recession_days == 4  # 2 steps after peak × 2 days


def test_summarise_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="length mismatch"):
        summarise([np.zeros((4, 4), dtype=bool)], [datetime(2018, 8, 15), datetime(2018, 8, 16)])


def test_summarise_too_few_timesteps() -> None:
    with pytest.raises(ValueError, match="at least two"):
        summarise([np.zeros((4, 4), dtype=bool)], [datetime(2018, 8, 15)])


def test_build_gif_writes_file(tmp_path: Path) -> None:
    masks = _rising_falling_masks()
    dates = _five_dates()
    out = tmp_path / "anim.gif"
    result = build_gif(masks, dates, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_build_area_curve_png_writes_file(tmp_path: Path) -> None:
    masks = _rising_falling_masks()
    dates = _five_dates()
    summary = summarise(masks, dates)
    out = tmp_path / "curve.png"
    result = build_area_curve_png(summary, out)
    assert result.exists()
    assert result.stat().st_size > 0


def test_summary_as_dict_is_json_safe() -> None:
    import json  # noqa: PLC0415
    masks = _rising_falling_masks()
    dates = _five_dates()
    d = summarise(masks, dates).as_dict()
    # Should roundtrip through json.
    json.dumps(d)  # must not raise
