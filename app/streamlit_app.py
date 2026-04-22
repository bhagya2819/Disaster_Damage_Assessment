"""Disaster Damage Assessment — Streamlit demo app.

Three data sources:
- **Bundled samples** — 3 pre-extracted chips in `app/sample_chips/`. Always works.
- **Sen1Floods11 test split** — needs the 3 GB dataset mounted; best for
  in-depth inspection.
- **Upload your own GeoTIFF** — 6-band reflectance raster for any AOI.

Three methods (+ SAR when an S1 GRD raster is uploaded):
- Classical · U-Net · Hybrid · SAR-optical fusion.

Run locally:
    streamlit run app/streamlit_app.py
Run in Colab via tunnel:
    python scripts/colab_streamlit.py
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from app.sample_loader import (  # noqa: E402
    SampleChip,
    bundled_manifest,
    load_bundled,
    load_geotiff_as_chip,
    synthetic_chip,
)
from src.data.sen1floods11_loader import (  # noqa: E402
    LABEL_IGNORE_INDEX,
    Sen1Floods11Dataset,
)
from src.eval import metrics  # noqa: E402
from src.pipelines.full_pipeline import run_pipeline  # noqa: E402

# ===========================================================================
# Config
# ===========================================================================

DEFAULT_SEN1F_ROOT = os.environ.get(
    "SEN1FLOODS11_DIR", "/content/drive/MyDrive/dda/sen1floods11"
)
DEFAULT_CKPT = os.environ.get(
    "DDA_CHECKPOINT",
    "/content/drive/MyDrive/dda/checkpoints/unet_resnet34/best.pt",
)


# ===========================================================================
# Cached resources
# ===========================================================================

@st.cache_resource(show_spinner="Loading Sen1Floods11 test index…")
def try_load_sen1floods11(root: str) -> Sen1Floods11Dataset | None:
    try:
        return Sen1Floods11Dataset(root=root, split="test", modality="s2")
    except Exception:  # noqa: BLE001
        return None


@st.cache_resource
def checkpoint_available(ckpt_path: str) -> bool:
    return Path(ckpt_path).exists()


# ===========================================================================
# Helpers
# ===========================================================================

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
    from PIL import Image  # noqa: PLC0415
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def mask_geotiff_bytes(mask: np.ndarray) -> bytes:
    import rasterio  # noqa: PLC0415
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "height": mask.shape[0],
        "width": mask.shape[1],
    }
    with rasterio.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
        return memfile.read()


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


def _save_uploaded_to_temp(upload) -> str:  # type: ignore[no-untyped-def]
    suffix = Path(upload.name).suffix or ".tif"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.close()
    return tmp.name


def _sar_optical_fusion(chip: np.ndarray, sar_path: str, method_label: str) -> tuple[np.ndarray, dict]:
    """Run SAR flood detection and fuse with the optical U-Net prediction.

    Returns (binary mask, stats dict). Uses sr.dip.sar internally.
    """
    import rasterio  # noqa: PLC0415

    from src.dip import sar as sarmod  # noqa: PLC0415

    with rasterio.open(sar_path) as src:
        vv = src.read(1).astype(np.float32)

    sar_mask, thr = sarmod.sar_flood_mask(vv)
    return sar_mask, {"sar_threshold_db": float(thr), "sar_method_label": method_label}


# ===========================================================================
# UI
# ===========================================================================

st.set_page_config(page_title="DDA · Flood mapping", page_icon="🌊", layout="wide")
st.title("🌊 Disaster Damage Assessment — flood mapping")
st.caption(
    "Sentinel-2 optical + Sentinel-1 SAR · Sen1Floods11 benchmark · classical DIP + U-Net + hybrid"
)

# ---- Sidebar ----

with st.sidebar:
    st.header("Configuration")

    # Data source
    source = st.radio(
        "Data source",
        ["Bundled samples", "Sen1Floods11 test", "Upload your own"],
        help=(
            "Bundled: 3 tiny pre-extracted chips, always available.\n"
            "Sen1Floods11: needs the 3 GB dataset mounted.\n"
            "Upload: any 6-band reflectance GeoTIFF."
        ),
    )

    # Method
    ckpt_path = st.text_input("U-Net checkpoint", value=DEFAULT_CKPT)
    ckpt_ok = checkpoint_available(ckpt_path)
    if not ckpt_ok:
        st.info("No U-Net checkpoint at this path — classical is the only available method.")

    methods = ["classical"] + (["unet", "hybrid"] if ckpt_ok else [])
    method = st.selectbox("Method", methods, index=(1 if ckpt_ok else 0))

    device = st.selectbox(
        "Device", ["cuda" if torch.cuda.is_available() else "cpu", "cpu"], index=0
    )

    # Optional SAR upload
    st.divider()
    st.caption("**Optional · Sentinel-1 SAR overlay**")
    sar_upload = st.file_uploader(
        "VV-band GeoTIFF (linear γ⁰)", type=["tif", "tiff"], key="sar",
        help="Upload a Sentinel-1 GRD VV raster aligned with the chip to enable SAR-optical fusion.",
    )
    sar_fusion_mode = st.selectbox(
        "SAR fusion mode",
        ["union (cloud-robust)", "agreement (high-precision)", "optical_primary"],
        disabled=sar_upload is None,
    )

    # Data-source-specific inputs
    st.divider()
    chip: SampleChip | None = None

    if source == "Bundled samples":
        manifest = bundled_manifest()
        if not manifest:
            st.warning("No bundled chips found. Using a synthetic fallback.")
            chip = synthetic_chip()
        else:
            labels = [f"{i}: {m.get('chip_id', 'chip')}  ({m.get('flood_fraction', 0)*100:.1f} % flooded)"
                      for i, m in enumerate(manifest)]
            pick = st.selectbox("Sample chip", range(len(manifest)), format_func=lambda i: labels[i])
            chip = load_bundled(int(pick))
            if chip is None:
                st.warning("Manifest lists the chip but file missing — synthetic fallback.")
                chip = synthetic_chip()

    elif source == "Sen1Floods11 test":
        root = st.text_input("Sen1Floods11 root", value=DEFAULT_SEN1F_ROOT)
        ds = try_load_sen1floods11(root)
        if ds is None:
            st.error(f"Could not load Sen1Floods11 at `{root}`. Falling back to a bundled sample.")
            chip = load_bundled(0) or synthetic_chip()
        else:
            chip_index = st.slider("Chip index", 0, max(len(ds) - 1, 0), 0)
            item = ds[int(chip_index)]
            chip = SampleChip(
                chip_id=item["chip_id"],
                image=item["image"].numpy(),
                label=item["label"].numpy(),
                pixel_size_m=10.0,
                flood_fraction=float("nan"),
            )

    else:  # Upload your own
        upload = st.file_uploader(
            "Upload a 6-band reflectance GeoTIFF", type=["tif", "tiff"], key="rgb"
        )
        if upload is None:
            st.info("No file yet. Using a bundled sample in the meantime.")
            chip = load_bundled(0) or synthetic_chip()
        else:
            try:
                tmp = _save_uploaded_to_temp(upload)
                chip = load_geotiff_as_chip(tmp)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not parse upload: {e}")
                chip = synthetic_chip()

    st.divider()
    run = st.button("Run prediction", type="primary", use_container_width=True)


# ---- Run pipeline ----

if chip is None:
    st.stop()

if run:
    with st.spinner(f"Running {method} on {chip.chip_id}…"):
        result = run_pipeline(
            chip.image,
            method=method,  # type: ignore[arg-type]
            checkpoint_path=ckpt_path if method != "classical" else None,
            device=device,
            pixel_size_m=chip.pixel_size_m,
        )

        sar_meta: dict = {}
        if sar_upload is not None:
            try:
                sar_tmp = _save_uploaded_to_temp(sar_upload)
                sar_mask, sar_meta = _sar_optical_fusion(chip.image, sar_tmp, sar_fusion_mode)
                from src.dip.sar import agreement_fraction, fuse_with_optical  # noqa: PLC0415

                mode_token = sar_fusion_mode.split(" ")[0]
                fused = fuse_with_optical(sar_mask, result.mask, mode=mode_token)
                sar_meta["sar_vs_optical_kappa"] = float(agreement_fraction(sar_mask, result.mask))
                # Re-package into the result for downstream tabs.
                result = result.__class__(
                    method="sar_fusion",
                    mask=fused,
                    probs=fused.astype(np.float32),
                    stats={
                        **result.stats,
                        "flooded_px": int(fused.sum()),
                        "flooded_fraction": float(fused.mean()),
                        "flooded_km2": float(
                            fused.sum() * chip.pixel_size_m ** 2 / 1e6
                        ),
                    },
                    runtime_ms=result.runtime_ms,
                )
            except Exception as e:  # noqa: BLE001
                st.warning(f"SAR fusion failed ({e}); showing optical-only result.")

        iou_vs_gt = None
        valid_gt = (chip.label != LABEL_IGNORE_INDEX)
        if valid_gt.any():
            iou_vs_gt = float(metrics.iou(result.mask, chip.label, ignore_index=LABEL_IGNORE_INDEX))

    st.session_state["result"] = result
    st.session_state["chip"] = chip
    st.session_state["iou_vs_gt"] = iou_vs_gt
    st.session_state["sar_meta"] = sar_meta

result = st.session_state.get("result")
chip = st.session_state.get("chip", chip)
iou_vs_gt = st.session_state.get("iou_vs_gt")
sar_meta = st.session_state.get("sar_meta", {})

if result is None:
    st.info("Choose inputs in the sidebar and click **Run prediction**.")
    st.image(rgb_from_chip(chip.image), caption=f"{chip.chip_id} · RGB preview",
             use_column_width=True)
    st.stop()


# ---- Tabs ----

tab_map, tab_metrics, tab_downloads, tab_about = st.tabs(
    ["🗺️ Map", "📊 Metrics", "⬇️ Downloads", "ℹ️ About"]
)

with tab_map:
    cols = st.columns(3)
    cols[0].image(rgb_from_chip(chip.image), caption="Post-event · RGB", use_column_width=True)
    cols[1].image(result.mask.astype(np.uint8) * 255, clamp=True,
                  caption=f"{result.method.upper()} flood mask",
                  use_column_width=True)
    gt_show = (
        np.where(chip.label == LABEL_IGNORE_INDEX, 0, chip.label).astype(np.uint8) * 255
        if (chip.label != LABEL_IGNORE_INDEX).any()
        else np.zeros_like(result.mask, dtype=np.uint8)
    )
    cols[2].image(
        gt_show,
        clamp=True,
        caption=("Ground truth" if (chip.label != LABEL_IGNORE_INDEX).any()
                 else "No ground truth available"),
        use_column_width=True,
    )

    import matplotlib.pyplot as plt  # noqa: PLC0415
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(rgb_from_chip(chip.image))
    ax.imshow(result.mask, cmap="Blues", alpha=0.45)
    ax.set_title(f"{chip.chip_id} · {result.method}")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)

with tab_metrics:
    s = result.stats
    cols = st.columns(4)
    cols[0].metric("Flooded", f"{s['flooded_km2']:.3f} km²")
    cols[1].metric("Of AOI", f"{s['flooded_fraction']*100:.1f} %")
    cols[2].metric("Method runtime", f"{result.runtime_ms:.1f} ms")
    if iou_vs_gt is not None:
        cols[3].metric("IoU vs GT", f"{iou_vs_gt:.3f}")

    if sar_meta:
        with st.expander("SAR-optical fusion details", expanded=True):
            st.json(sar_meta)

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
        file_name=f"{chip.chip_id}_{result.method}_mask.png",
        mime="image/png",
    )

    col2.download_button(
        "Mask (GeoTIFF)",
        data=mask_geotiff_bytes(result.mask),
        file_name=f"{chip.chip_id}_{result.method}_mask.tif",
        mime="image/tiff",
    )

    col3.download_button(
        "Stats (CSV)",
        data=stats_csv_bytes(result.stats, result.method, chip.chip_id, iou_vs_gt),
        file_name=f"{chip.chip_id}_{result.method}_stats.csv",
        mime="text/csv",
    )

    st.divider()
    if st.button("Generate PDF damage report"):
        try:
            from app.report_generator import ReportContext, build_report_bytes  # noqa: PLC0415
            ctx = ReportContext(
                title=f"DDA report · {chip.chip_id}",
                chip=chip.image,
                result=result,
            )
            pdf_bytes = build_report_bytes(ctx)
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=f"{chip.chip_id}_{result.method}_report.pdf",
                mime="application/pdf",
            )
            st.success("Report ready.")
        except Exception as e:  # noqa: BLE001
            st.error(f"PDF generation failed: {e}")
            st.caption("Hint: WeasyPrint needs system libs (libpango, libcairo); "
                       "see `packages.txt` if deploying to Streamlit Cloud.")

with tab_about:
    st.markdown(
        """
### About

This demo implements the full DDA pipeline described in `PRD.md`.

- **Classical** — NDWI (McFeeters 1996) + Yen thresholding, morphology off.
- **U-Net** — ResNet-34 encoder, 6-band Sentinel-2 input, trained on Sen1Floods11 HandLabeled.
- **Hybrid** — 70/30 weighted fusion of U-Net probabilities and the classical mask.
- **SAR fusion** *(optional when a VV raster is uploaded)* — Refined-Lee speckle filter → log-VV Otsu → fused with the optical mask.

**Repo:** https://github.com/bhagya2819/Disaster_Damage_Assessment

### Data sources
- **Bundled samples** (default) — 3 tiny pre-extracted Sen1Floods11 chips shipped with the repo.
- **Sen1Floods11 test** — the full 90-chip evaluation split (requires the 3 GB dataset mounted).
- **Upload** — any 6-band reflectance GeoTIFF in the DDA band order.

### Key results (Sen1Floods11 test, n = 90)
U-Net IoU **0.548**, +0.108 over classical (paired bootstrap 95 % CI [+0.037, +0.114], McNemar p ≈ 0).
"""
    )
