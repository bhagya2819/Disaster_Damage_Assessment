"""Disaster Damage Assessment — Streamlit demo app.

Layout:
- Sidebar: data source, method, run button.
- Main (tabs): Map · Metrics · Downloads · About.

Run locally:
    streamlit run app/streamlit_app.py

Run in Colab (free-tier) with an ngrok tunnel:
    python scripts/colab_streamlit.py

The app caches the U-Net checkpoint load via ``@st.cache_resource`` so the
~100 MB model is loaded once per session, not per prediction.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

# Make `src` importable whether the script is run from repo root or elsewhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.sen1floods11_loader import (  # noqa: E402
    LABEL_IGNORE_INDEX,
    Sen1Floods11Dataset,
)
from src.eval import metrics  # noqa: E402
from src.pipelines.full_pipeline import run_pipeline  # noqa: E402

# Optional imports — handled at call site so the app can still load if missing.
try:
    import folium  # noqa: PLC0415
    from streamlit_folium import st_folium  # noqa: PLC0415
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False


# =========================================================================
# Config
# =========================================================================

DEFAULT_SEN1F_ROOT = os.environ.get(
    "SEN1FLOODS11_DIR", "/content/drive/MyDrive/dda/sen1floods11"
)
DEFAULT_CKPT = os.environ.get(
    "DDA_CHECKPOINT",
    "/content/drive/MyDrive/dda/checkpoints/unet_resnet34/best.pt",
)


# =========================================================================
# Cached resources
# =========================================================================

@st.cache_resource(show_spinner="Loading Sen1Floods11 test index…")
def load_dataset(root: str) -> Sen1Floods11Dataset | None:
    try:
        return Sen1Floods11Dataset(root=root, split="test", modality="s2")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not load Sen1Floods11 at `{root}`: {e}")
        return None


@st.cache_data(show_spinner="Reading chip…")
def load_chip(root: str, index: int) -> dict | None:
    ds = load_dataset(root)
    if ds is None or index >= len(ds):
        return None
    sample = ds[index]
    return {
        "image": sample["image"].numpy(),
        "label": sample["label"].numpy(),
        "chip_id": sample["chip_id"],
    }


@st.cache_resource(show_spinner="Loading U-Net checkpoint…")
def checkpoint_available(ckpt_path: str) -> bool:
    return Path(ckpt_path).exists()


# =========================================================================
# Helpers
# =========================================================================

def stretch(x: np.ndarray, lo: int = 2, hi: int = 98) -> np.ndarray:
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return x
    a, b = np.percentile(finite, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-9), 0, 1)


def rgb_from_chip(chip: np.ndarray) -> np.ndarray:
    bands = np.stack([stretch(chip[i]) for i in (2, 1, 0)])
    return np.transpose(bands, (1, 2, 0))


def mask_png_bytes(mask: np.ndarray) -> bytes:
    """Return a small PNG-encoded byte stream of a bool mask."""
    from PIL import Image  # noqa: PLC0415
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mask_geotiff_bytes(mask: np.ndarray) -> bytes:
    """Return a tiny GeoTIFF of the binary mask (no geo-reference)."""
    import rasterio  # noqa: PLC0415
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": mask.shape[0],
        "width": mask.shape[1],
    }
    buf = io.BytesIO()
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        buf.write(memfile.read())
    buf.seek(0)
    return buf.read()


def stats_csv_bytes(stats: dict, method: str, chip_id: str, iou: float | None) -> bytes:
    import csv  # noqa: PLC0415
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["field", "value"])
    w.writerow(["chip_id", chip_id])
    w.writerow(["method", method])
    w.writerow(["iou_vs_gt", "" if iou is None else f"{iou:.4f}"])
    for k, v in stats.items():
        w.writerow([k, v])
    return buf.getvalue().encode("utf-8")


# =========================================================================
# UI
# =========================================================================

st.set_page_config(page_title="DDA · Flood mapping", page_icon="🌊", layout="wide")

st.title("🌊 Disaster Damage Assessment — flood mapping")
st.caption(
    "Sentinel-2 optical imagery · Sen1Floods11 benchmark · classical DIP + U-Net hybrid"
)

# --- Sidebar ---

with st.sidebar:
    st.header("Configuration")

    root = st.text_input("Sen1Floods11 root", value=DEFAULT_SEN1F_ROOT)
    ckpt_path = st.text_input("U-Net checkpoint (best.pt)", value=DEFAULT_CKPT)

    ds = load_dataset(root)
    if ds is None:
        st.stop()

    ckpt_ok = checkpoint_available(ckpt_path)
    if not ckpt_ok:
        st.warning(
            "U-Net checkpoint not found — classical is the only available method."
        )

    chip_index = st.slider("Chip index", 0, max(len(ds) - 1, 0), value=0)
    st.caption(f"Total chips in test split: **{len(ds)}**")

    method_options = ["classical"] + (["unet", "hybrid"] if ckpt_ok else [])
    method = st.selectbox("Method", method_options, index=(1 if ckpt_ok else 0))

    device = st.selectbox(
        "Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"], index=0
    )

    st.divider()
    run = st.button("Run prediction", type="primary", use_container_width=True)


# --- Load the selected chip ---

chip_bundle = load_chip(root, chip_index)
if chip_bundle is None:
    st.error(f"Could not read chip {chip_index}")
    st.stop()

chip = chip_bundle["image"]
label = chip_bundle["label"]
chip_id = chip_bundle["chip_id"]


# --- Run pipeline on demand ---

if "result" not in st.session_state or st.session_state.get("last_chip_id") != (chip_id, method):
    if not run:
        st.info("Choose settings in the sidebar and click **Run prediction**.")
        preview = rgb_from_chip(chip)
        st.image(preview, caption=f"Chip {chip_id} · RGB preview", use_column_width=True)
        st.stop()

if run:
    with st.spinner(f"Running {method} pipeline on {chip_id}…"):
        result = run_pipeline(
            chip,
            method=method,  # type: ignore[arg-type]
            checkpoint_path=ckpt_path if method != "classical" else None,
            device=device,
        )
        iou_vs_gt = float(metrics.iou(result.mask, label, ignore_index=LABEL_IGNORE_INDEX))
    st.session_state["result"] = result
    st.session_state["last_chip_id"] = (chip_id, method)
    st.session_state["iou_vs_gt"] = iou_vs_gt

result = st.session_state.get("result")
iou_vs_gt = st.session_state.get("iou_vs_gt")
if result is None:
    st.stop()


# --- Tabs ---

tab_map, tab_metrics, tab_downloads, tab_about = st.tabs(
    ["🗺️ Map", "📊 Metrics", "⬇️ Downloads", "ℹ️ About"]
)

with tab_map:
    cols = st.columns(3)
    cols[0].image(rgb_from_chip(chip), caption="Post-event · RGB", use_column_width=True)
    cols[1].image(result.mask.astype(np.uint8) * 255, clamp=True,
                  caption=f"{method.upper()} flood mask", use_column_width=True)
    cols[2].image(np.where(label == -1, 0, label).astype(np.uint8) * 255, clamp=True,
                  caption="Ground truth", use_column_width=True)

    # --- Overlay image ---
    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb_from_chip(chip))
    ax.imshow(result.mask, cmap="Blues", alpha=0.45)
    ax.set_title(f"{chip_id} · {method}")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

with tab_metrics:
    s = result.stats
    cols = st.columns(4)
    cols[0].metric("Flooded", f"{s['flooded_km2']:.3f} km²")
    cols[1].metric("Of AOI", f"{s['flooded_fraction']*100:.1f} %")
    cols[2].metric("Method runtime", f"{result.runtime_ms:.1f} ms")
    if iou_vs_gt is not None:
        cols[3].metric("IoU vs ground truth", f"{iou_vs_gt:.3f}")

    st.subheader("Method benchmark (Sen1Floods11 test, n = 90)")
    import pandas as pd  # noqa: PLC0415
    st.dataframe(pd.DataFrame([
        {"method": "Classical (ndwi_yen_raw)", "IoU": 0.440, "F1": 0.547, "κ": 0.497,
         "Accuracy": 0.887},
        {"method": "U-Net (ResNet-34)", "IoU": 0.548, "F1": 0.660, "κ": 0.652,
         "Accuracy": 0.971},
        {"method": "Hybrid (w_unet=0.7)", "IoU": 0.531, "F1": 0.636, "κ": 0.614,
         "Accuracy": 0.970},
    ]))

with tab_downloads:
    col1, col2, col3 = st.columns(3)

    col1.download_button(
        "Mask (PNG)",
        data=mask_png_bytes(result.mask),
        file_name=f"{chip_id}_{method}_mask.png",
        mime="image/png",
    )

    col2.download_button(
        "Mask (GeoTIFF)",
        data=mask_geotiff_bytes(result.mask),
        file_name=f"{chip_id}_{method}_mask.tif",
        mime="image/tiff",
    )

    col3.download_button(
        "Stats (CSV)",
        data=stats_csv_bytes(result.stats, method, chip_id, iou_vs_gt),
        file_name=f"{chip_id}_{method}_stats.csv",
        mime="text/csv",
    )

    st.divider()
    if st.button("Generate PDF damage report"):
        try:
            from app.report_generator import ReportContext, build_report_bytes  # noqa: PLC0415
            ctx = ReportContext(
                title=f"DDA report · {chip_id}",
                chip=chip,
                result=result,
            )
            pdf_bytes = build_report_bytes(ctx)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{chip_id}_{method}_report.pdf",
                mime="application/pdf",
            )
            st.success("Report ready.")
        except Exception as e:  # noqa: BLE001
            st.error(f"PDF generation failed: {e}")
            st.caption("Hint: WeasyPrint needs system libs (libpango, libcairo); "
                       "works out of the box on Colab and Streamlit Cloud.")

with tab_about:
    st.markdown(
        """
### About

This demo implements the full DDA pipeline described in the course project PRD.

- **Classical baseline** — NDWI (McFeeters 1996) + Yen thresholding, no morphology.
- **U-Net** — ResNet-34 encoder, 6-channel Sentinel-2 input, trained on Sen1Floods11 HandLabeled.
- **Hybrid** — weighted fusion of U-Net probabilities and classical mask.

**Repo:** https://github.com/bhagya2819/Disaster_Damage_Assessment

**Key design decisions** (see `PRD.md`):
- PyTorch + segmentation-models-pytorch for the model layer.
- BCE + Dice loss with `pos_weight = 2.0` for class imbalance.
- Cosine LR schedule, AdamW, mixed precision on CUDA.
- Evaluation on Sen1Floods11 test split; per-chip IoU stored for significance tests.
"""
    )
