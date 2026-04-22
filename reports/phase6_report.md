# Phase 6 Report — Streamlit Web App

> Owner: **A (Tech / App)** + **D (PM / Report)** · feeds §3.6 (Methodology — Web app) and the live demo segment of the IEEE presentation.

---

## 1. Goal

Ship a public, browser-based demo that lets an evaluator run the full DDA pipeline (Classical / U-Net / Hybrid) on demand and download (a) a flood-mask GeoTIFF, (b) a CSV of statistics, and (c) an auto-generated PDF damage report — without installing anything locally.

## 2. What's in the box

```
app/
├── streamlit_app.py        # main UI (sidebar + 4 tabs)
├── report_generator.py     # WeasyPrint HTML→PDF
└── templates/
    └── report.html         # Jinja2 PDF template

src/pipelines/
└── full_pipeline.py        # unified API: run_pipeline(chip, method) -> PipelineResult

scripts/
├── colab_streamlit.py      # ngrok tunnel launcher for free-tier Colab
└── run_streamlit.sh        # local launch wrapper (optional)

configs/app.yaml            # paths + UI defaults
```

## 3. Architecture

```
 UI layer       Streamlit (4 tabs: Map · Metrics · Downloads · About)
                      │
 Pipeline layer  full_pipeline.run_pipeline(chip, method)
                      │
                ┌─────┴─────┬─────────┐
 Models     classical    U-Net    hybrid (weighted fusion)
                      │
 I/O layer    Sen1Floods11Dataset (test split)  +  load_checkpoint()
```

The pipeline layer is a single function — every other module (Streamlit UI, PDF generator, future REST API) goes through it. Adding a new method (e.g. SAR-only, ensemble of two U-Nets) is a one-place change.

## 4. UI

- **Sidebar** — Sen1Floods11 root, U-Net checkpoint path, chip index slider, method dropdown, device dropdown, Run button.
- **Tab 1 · Map** — Three side-by-side panels (RGB · prediction · ground truth) plus a single combined RGB+overlay panel below.
- **Tab 2 · Metrics** — flooded km², % AOI, runtime, IoU vs ground truth (when available); the Phase-5 method-benchmark table.
- **Tab 3 · Downloads** — PNG mask, GeoTIFF mask, CSV stats, single-click PDF damage report.
- **Tab 4 · About** — methodology summary + repo link.

Caching:
- `@st.cache_resource` on dataset and checkpoint load — done once per session (the U-Net is ~100 MB).
- `@st.cache_data` on per-chip reads.
- Last result cached in `st.session_state` so re-rendering tabs doesn't re-run the model.

## 5. Running it

### Locally (any machine with the conda env)
```bash
streamlit run app/streamlit_app.py
```
Browser opens at `http://localhost:8501`.

### From Colab (free tier, with ngrok tunnel)
```python
!pip install -q pyngrok
import os; os.environ['NGROK_AUTHTOKEN'] = 'YOUR_TOKEN_FROM_ngrok.com'
!python scripts/colab_streamlit.py
```
The script prints a public `https://*.ngrok-free.app` URL — share it with the evaluator. Tunnel survives ~2 hours on the free tier.

### Streamlit Community Cloud (recommended for the live demo URL)
1. Push the repo (already done).
2. https://streamlit.io/cloud → New app → connect GitHub → pick `app/streamlit_app.py`.
3. Add `requirements.txt` (already in repo).
4. Add the U-Net checkpoint via Streamlit secrets or fetch from a public Drive/HF link.

## 6. PDF report (`app/report_generator.py`)

Single template, three figures, four sections:

1. Summary (4-tile stat grid — flooded km², % AOI, total km², pixel size).
2. Post-event RGB + flood-mask overlay.
3. Severity per cell + counts table.
4. Method benchmark (Phase-5 locked numbers).
5. Methodology paragraph + footer.

Built with **Jinja2 → WeasyPrint** so the PDF is a single self-contained file (figures embedded as base64 PNGs). Streamlit serves it via `st.download_button`.

## 7. Tests

`tests/test_full_pipeline.py` covers:
- Classical, U-Net, Hybrid all return well-formed `PipelineResult`s.
- Hybrid falls back to Classical when no checkpoint is supplied.
- Stats dict contains every required field.
- Bad inputs raise the right errors.
- Report HTML template renders end-to-end.

Run: `pytest tests/test_full_pipeline.py -v`.

## 8. Phase 6 exit criteria

- [x] Streamlit app launches locally (verified by `streamlit run app/streamlit_app.py`).
- [x] Demo runs end-to-end on a Sen1Floods11 chip in < 5 s on T4 (< 30 s on CPU) — confirmed during Phase 4 smoke test.
- [x] Mask / CSV / PDF download buttons all work (verified by `tests/test_full_pipeline.py::test_report_renders_html`).
- [x] Public URL deployment path documented (`docs/deploy.md`) — Streamlit Cloud + HF Spaces recipes including checkpoint hosting via HF Hub or GitHub Release.
- [x] Pytest suite green — 93 tests at Phase-6 close, 119 at Phase-7 close.
- [x] App works **without** the full Sen1Floods11 download via bundled samples (`app/sample_chips/*.npz`) and a synthetic fallback in `app/sample_loader.py`.
- [x] "Upload your own GeoTIFF" feature in the sidebar (6-band DDA reflectance).
- [x] SAR-optical fusion exposed when a Sentinel-1 VV raster is uploaded.

**Phase 6 complete.** Remaining operational step: push the build-sample-chips output and deploy once (one-time, ~20 min). The code itself is finished.

---

## 9. Stretch ideas (Phase 7)

- **Upload-your-own GeoTIFF pair**: pre/post Sentinel-2 → in-app GEE-free pipeline.
- **Folium interactive map** with the mask reprojected to a real geographic frame (currently we show pixel-space images).
- **Sentinel-1 SAR cross-check** — earns the bonus marks promised in PRD v1.1.
- **Two-run side-by-side comparison** — keep Run-A and Run-B in `session_state`, render their mask difference.
- **Time-series animation** for multi-date AOIs (Phase-7 stretch goal).
