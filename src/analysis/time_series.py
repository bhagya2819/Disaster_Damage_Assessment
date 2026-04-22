"""Flood progression over time — animation + area curve.

Given a sequence of aligned flood masks (one per date), this module:

1. Computes the flooded-area curve in km² per timestep.
2. Builds an animated GIF or MP4 that steps through the masks with a
   timestamp caption.
3. Quantifies temporal metrics: peak flooded day, time-to-peak, 50 %-recession
   day.

Inputs are plain numpy arrays + a list of ``datetime`` objects — no raster
I/O, so the module is easy to test and to call from the Streamlit app.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402


@dataclass(frozen=True)
class TimeSeriesSummary:
    dates: list[datetime]
    flooded_km2: list[float]
    peak_date: datetime
    peak_km2: float
    time_to_peak_days: int
    time_to_half_recession_days: int | None

    def as_dict(self) -> dict:
        return {
            "dates": [d.isoformat() for d in self.dates],
            "flooded_km2": self.flooded_km2,
            "peak_date": self.peak_date.isoformat(),
            "peak_km2": self.peak_km2,
            "time_to_peak_days": self.time_to_peak_days,
            "time_to_half_recession_days": self.time_to_half_recession_days,
        }


def flooded_km2_per_step(
    masks: Sequence[np.ndarray],
    pixel_size_m: float = 10.0,
) -> list[float]:
    """Return flooded km² for each mask in the sequence."""
    px_area = pixel_size_m * pixel_size_m
    return [float(np.asarray(m, dtype=bool).sum() * px_area / 1e6) for m in masks]


def summarise(
    masks: Sequence[np.ndarray],
    dates: Sequence[datetime],
    pixel_size_m: float = 10.0,
) -> TimeSeriesSummary:
    """Compute flooded-area curve and key temporal metrics."""
    if len(masks) != len(dates):
        raise ValueError(f"length mismatch: {len(masks)} masks vs {len(dates)} dates")
    if len(masks) < 2:
        raise ValueError("need at least two timesteps to build a time-series")

    km2 = flooded_km2_per_step(masks, pixel_size_m=pixel_size_m)
    peak_idx = int(np.argmax(km2))
    peak_km2 = float(km2[peak_idx])
    t0 = dates[0]
    tpeak = dates[peak_idx]
    days_to_peak = int((tpeak - t0).days)

    # 50%-recession: first date AFTER the peak where flooded <= peak/2.
    half_recession_days: int | None = None
    for k in range(peak_idx + 1, len(km2)):
        if km2[k] <= peak_km2 / 2:
            half_recession_days = int((dates[k] - tpeak).days)
            break

    return TimeSeriesSummary(
        dates=list(dates),
        flooded_km2=list(km2),
        peak_date=tpeak,
        peak_km2=peak_km2,
        time_to_peak_days=days_to_peak,
        time_to_half_recession_days=half_recession_days,
    )


def build_gif(
    masks: Sequence[np.ndarray],
    dates: Sequence[datetime],
    out_path: str | Path,
    background: np.ndarray | None = None,
    fps: int = 2,
    cmap: str = "Blues",
) -> Path:
    """Build an animated GIF of flood progression.

    Parameters
    ----------
    masks
        Sequence of (H, W) bool arrays.
    dates
        One timestamp per mask, same length.
    out_path
        Output .gif path.
    background
        Optional (H, W, 3) RGB image drawn beneath every frame for context.
    fps
        Frames per second.
    """
    if len(masks) != len(dates):
        raise ValueError(f"length mismatch: {len(masks)} masks vs {len(dates)} dates")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    if background is not None:
        ax.imshow(background)
    im = ax.imshow(masks[0], cmap=cmap, alpha=0.55, vmin=0, vmax=1)
    title = ax.set_title(dates[0].strftime("%Y-%m-%d"))
    ax.axis("off")

    def _update(i: int):
        im.set_data(masks[i].astype(np.float32))
        title.set_text(dates[i].strftime("%Y-%m-%d"))
        return im, title

    anim = FuncAnimation(fig, _update, frames=len(masks), blit=False)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out_path


def build_area_curve_png(
    summary: TimeSeriesSummary,
    out_path: str | Path,
    dpi: int = 150,
) -> Path:
    """Plot the flooded-area curve with the peak annotated."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(summary.dates, summary.flooded_km2, marker="o", linewidth=2, color="#2b5797")
    ax.fill_between(summary.dates, 0, summary.flooded_km2, alpha=0.15, color="#2b5797")
    ax.axvline(summary.peak_date, linestyle="--", color="#c44e52", alpha=0.6)
    ax.annotate(
        f"peak {summary.peak_km2:.1f} km²\n{summary.peak_date:%Y-%m-%d}",
        xy=(summary.peak_date, summary.peak_km2),
        xytext=(5, 5), textcoords="offset points",
        color="#c44e52", fontsize=9,
    )
    ax.set_xlabel("date"); ax.set_ylabel("flooded area (km²)")
    ax.set_title("Flood extent over time")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


__all__ = [
    "TimeSeriesSummary",
    "build_area_curve_png",
    "build_gif",
    "flooded_km2_per_step",
    "summarise",
]
