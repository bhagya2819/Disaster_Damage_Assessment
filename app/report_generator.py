"""PDF damage-report generator.

Fills the Jinja template at ``app/templates/report.html`` with pipeline
outputs and compiles to PDF via WeasyPrint. Figures are embedded as base64
PNGs so the PDF is a single self-contained file.

Usage:

    from src.pipelines.full_pipeline import run_pipeline
    from app.report_generator import build_report, ReportContext

    result = run_pipeline(chip, method="unet", checkpoint_path=...)
    ctx = ReportContext(title="Kerala flood assessment", chip=chip, result=result)
    build_report(ctx, out_path="kerala_report.pdf")
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from src.analysis.severity import SeverityConfig, Severity, classify as severity_classify  # noqa: E402
from src.pipelines.full_pipeline import PipelineResult  # noqa: E402

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Phase-3/4/5 published benchmark table — locked numbers.
DEFAULT_BENCHMARK_ROWS: list[dict[str, float | str]] = [
    {"method": "Classical (ndwi_yen_raw)", "iou": 0.4401, "f1": 0.5475, "kappa": 0.4968,
     "accuracy": 0.8869, "recall": 0.7601, "precision": 0.5902},
    {"method": "U-Net (ResNet-34)", "iou": 0.5475, "f1": 0.6604, "kappa": 0.6522,
     "accuracy": 0.9709, "recall": 0.7337, "precision": 0.6719},
    {"method": "Hybrid (w_unet=0.7)", "iou": 0.5313, "f1": 0.6364, "kappa": 0.6136,
     "accuracy": 0.9698, "recall": 0.7356, "precision": 0.6401},
]

DEFAULT_METHODOLOGY = (
    "Pre/post Sentinel-2 L2A surface-reflectance chips (6 DDA bands: B2, B3, B4, B8, B11, B12) "
    "are processed through the selected pipeline. The classical baseline applies McFeeters NDWI "
    "with Yen thresholding to produce a binary water mask. The U-Net is a ResNet-34-encoder "
    "segmentation network trained on the Sen1Floods11 HandLabeled split with BCE+Dice loss. "
    "The hybrid variant combines the U-Net probability map with the classical mask via a "
    "weighted sum (default w_unet=0.7). Metrics are evaluated against the Sen1Floods11 test "
    "split (n=90 chips)."
)


@dataclass
class ReportContext:
    """Inputs required to fill the template."""

    title: str
    chip: np.ndarray                # (C, H, W) reflectance
    result: PipelineResult
    severity_cfg: SeverityConfig = field(default_factory=lambda: SeverityConfig(cell_px=16))
    benchmark_rows: list[dict[str, Any]] = field(default_factory=lambda: list(DEFAULT_BENCHMARK_ROWS))
    methodology_text: str = DEFAULT_METHODOLOGY


# ---------- rendering helpers ----------

def _stretch(x: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    a, b = np.percentile(finite, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def _rgb_from_chip(chip: np.ndarray) -> np.ndarray:
    """Post-event true-colour composite: B4, B3, B2 (DDA indices 2, 1, 0)."""
    bands = np.stack([_stretch(chip[i]) for i in (2, 1, 0)])
    return np.transpose(bands, (1, 2, 0))


def _fig_to_b64(fig: plt.Figure, dpi: int = 140) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _make_rgb_overlay_b64(chip: np.ndarray, mask: np.ndarray) -> str:
    rgb = _rgb_from_chip(chip)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Post-event · RGB")
    axes[0].axis("off")
    axes[1].imshow(rgb)
    axes[1].imshow(mask, cmap="Blues", alpha=0.55)
    axes[1].set_title("Flood mask overlay")
    axes[1].axis("off")
    return _fig_to_b64(fig)


def _make_severity_b64(
    mask: np.ndarray, cfg: SeverityConfig
) -> tuple[str, list[dict[str, Any]], int]:
    frac, cls = severity_classify(mask, cfg=cfg)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im = axes[0].imshow(frac, cmap="YlGnBu", vmin=0, vmax=1)
    axes[0].set_title("Flooded fraction / cell")
    axes[0].axis("off")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    im2 = axes[1].imshow(cls, cmap="RdYlBu_r", vmin=0, vmax=3)
    axes[1].set_title("Severity class")
    axes[1].axis("off")
    cb = plt.colorbar(im2, ax=axes[1], shrink=0.8, ticks=[0, 1, 2, 3])
    cb.ax.set_yticklabels(["None", "Low", "Mod", "Severe"])
    b64 = _fig_to_b64(fig)

    # Each cell covers (cell_px * pixel_size_m)² m².
    # We don't know pixel_size_m here — default to 10 m (Sentinel-2).
    cell_area_km2 = (cfg.cell_px * 10.0) ** 2 / 1e6
    total_cells = int(cls.size)
    rows = []
    for enum_val, label in zip(
        (Severity.NONE, Severity.LOW, Severity.MODERATE, Severity.SEVERE),
        ("None", "Low", "Moderate", "Severe"),
        strict=True,
    ):
        count = int((cls == enum_val).sum())
        rows.append({
            "name": label,
            "count": count,
            "area_km2": count * cell_area_km2,
            "share_pct": (count / total_cells * 100) if total_cells else 0.0,
        })
    return b64, rows, int(cfg.cell_px * 10.0 // 1000)


def render_html(ctx: ReportContext) -> str:
    """Render the HTML report (used standalone or as input to PDF engine)."""
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html")

    rgb_b64 = _make_rgb_overlay_b64(ctx.chip, ctx.result.mask)
    sev_b64, sev_rows, sev_cell_km = _make_severity_b64(ctx.result.mask, ctx.severity_cfg)

    return template.render(
        title=ctx.title,
        method=ctx.result.method,
        timestamp=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
        stats=ctx.result.stats,
        rgb_img_b64=rgb_b64,
        severity_img_b64=sev_b64,
        severity_rows=sev_rows,
        severity_cell_km=sev_cell_km,
        benchmark_rows=ctx.benchmark_rows,
        methodology_text=ctx.methodology_text,
        runtime_ms=ctx.result.runtime_ms,
    )


def build_report(ctx: ReportContext, out_path: str | Path) -> Path:
    """Render HTML and compile to PDF at ``out_path``."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(ctx)

    try:
        from weasyprint import HTML  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "weasyprint is required to build PDFs. "
            "Install via `pip install weasyprint` (needs libcairo/pango on Linux)."
        ) from e

    HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf(str(out_path))
    return out_path


def build_report_bytes(ctx: ReportContext) -> bytes:
    """Return the PDF as in-memory bytes — used by Streamlit's download button."""
    from weasyprint import HTML  # noqa: PLC0415
    html = render_html(ctx)
    return HTML(string=html, base_url=str(TEMPLATE_DIR)).write_pdf()


__all__ = [
    "DEFAULT_BENCHMARK_ROWS",
    "DEFAULT_METHODOLOGY",
    "ReportContext",
    "build_report",
    "build_report_bytes",
    "render_html",
]
