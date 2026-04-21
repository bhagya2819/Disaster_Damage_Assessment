# Product Requirements Document (PRD)
## Disaster Damage Assessment — Flood Mapping from Sentinel-2 Imagery
### Course: Digital Image Processing · Domain: Remote Sensing & Satellite Image Processing

---

## 0. Document Control

| Field | Value |
|---|---|
| Project Name | Disaster Damage Assessment (DDA) |
| Sub-domain | Remote Sensing & Satellite Image Processing |
| Focus Disaster | Floods (Kerala, India — August 2018) |
| Primary Data | Sentinel-2 L2A optical imagery (10 m) |
| Auxiliary Dataset | Sen1Floods11 (for supervised training/validation) |
| Approach | Hybrid — Classical DIP + Deep Learning |
| Deliverable | Web application with interactive map + PDF report generator |
| Team Size | 3–4 members |
| Timeline | 6–8 weeks |
| Compute Budget | Google Colab (free tier T4 GPU) |
| Document Version | v1.2 |
| Last Updated | 2026-04-21 |
| Change Log | v1.1 — locked §13 open questions: AOI = team's choice (Kerala 2018 confirmed); PyTorch allowed; presentation = 20 min; Sentinel-1 SAR earns bonus marks → promoted from stretch to should-have. · v1.2 — UNOSAT / Copernicus-EMS ground-truth retrieval failed (dead URLs). Pivoted: **Sen1Floods11 is now the sole source of quantitative metrics**; Kerala 2018 becomes a qualitative case study + self-annotated spot checks. Removed "IoU on Kerala GT" as a hard success criterion. Hand-annotated patches (Phase 1 🟡) promoted to 🟢 must-have. |

---

## 1. Executive Summary

This project builds a **production-ready, web-based disaster-damage-assessment system** that ingests pre- and post-event Sentinel-2 satellite imagery for a user-selected area of interest (AOI) and produces:

1. A pixel-accurate **flood-water extent mask** (post-event).
2. A **pre/post change-detection map** highlighting newly inundated pixels.
3. A **damage severity classification** (No / Low / Moderate / Severe) at AOI and pixel level.
4. An **automated quantitative damage report** (area flooded in km², % of AOI affected, land-cover-wise breakdown, population exposure).

The system combines **classical Digital Image Processing** techniques (atmospheric-corrected reflectance handling, spectral indices such as NDWI/MNDWI/NDVI, histogram-based and Otsu thresholding, morphological post-processing, image differencing, PCA-based change detection) with a **deep-learning segmentation model** (U-Net trained on Sen1Floods11) to deliver both explainability and state-of-the-art accuracy.

The Kerala 2018 floods serve as the end-to-end case study. The pipeline is region-agnostic — any user-supplied AOI and date pair will work.

---

## 2. Problem Definition & Motivation

### 2.1 Problem Statement
Floods are the most frequent and damaging natural disaster globally, and in India they account for the largest share of disaster-related economic loss. After a flood event, disaster-response agencies need **rapid, objective, and spatially explicit** information on the extent and severity of inundation to:

- Prioritise search-and-rescue deployment.
- Estimate affected population, cropland, and infrastructure.
- Allocate relief funds and insurance payouts.
- Plan long-term mitigation.

Ground surveys are slow, expensive, and often infeasible during active flooding. **Satellite remote sensing with digital image processing** provides the only scalable alternative.

### 2.2 Motivation
- **Societal:** Kerala's August 2018 floods killed 483 people and displaced ~1.4 million; rapid damage mapping would have materially improved response.
- **Technical:** Flood mapping is a canonical DIP problem — it exercises radiometric correction, spectral index computation, thresholding, morphological operations, change detection, and supervised segmentation.
- **Course-fit:** The problem naturally demands every technique family covered in a DIP syllabus (point, neighborhood, frequency, segmentation, morphology, classification) plus modern CNN-based segmentation — a perfect fit for the hybrid rubric criterion.

### 2.3 Background Research (to be expanded in Phase 1)
- MNDWI (Xu, 2006) — modified water index, superior to NDWI for urban flood pixels.
- Otsu thresholding (Otsu, 1979) — automatic bimodal segmentation baseline.
- U-Net (Ronneberger, 2015) — benchmark architecture for biomedical & remote-sensing segmentation.
- Sen1Floods11 (Bonafilia et al., 2020) — labeled flood mask dataset.
- UNOSAT rapid-mapping reports for Kerala 2018.

---

## 3. Goals & Non-Goals

### 3.1 Goals
- G1. Produce a flood mask with **IoU ≥ 0.75** against Sen1Floods11 validation split. *(v1.2: this is now the **sole** quantitative accuracy gate; Kerala GT is not required.)*
- G2. Deliver a **single-click web demo** where a user selects AOI + dates → gets maps + report.
- G3. Demonstrate **side-by-side** performance of ≥ 3 classical DIP methods and ≥ 1 deep-learning method (ablation table).
- G4. Auto-generate a **PDF damage report** with quantitative statistics and overlay maps.
- G5. Make the full pipeline **reproducible** on Google Colab free tier (no paid assets required).

### 3.2 Non-Goals (Out of Scope for v1)
- Real-time ingestion of newly-captured imagery (the pipeline accepts already-downloaded scenes or Earth Engine assets).
- Mobile-native application.
- Disaster types other than floods.
- ~~SAR (Sentinel-1) processing — optional stretch only if time permits.~~ **Moved in-scope (v1.1):** instructor confirmed Sentinel-1 SAR earns bonus marks; now a should-have in Phase 7.
- Fine-grained building-level damage classification (requires sub-meter imagery; out of Sentinel-2 resolution).

---

## 4. Success Criteria (Mapped to Course Rubric)

| Rubric Criterion (Weight) | PRD Success Signal |
|---|---|
| Problem Definition (10%) | §2 of this PRD + Phase 1 literature review doc with ≥ 10 citations. |
| Implementation of DIP Techniques (20%) | Phase 2 classical DIP module implements and compares ≥ 5 techniques (spectral indices, Otsu, adaptive threshold, morphology, PCA change detection) with written justification for each. |
| Technical Accuracy & Code Quality (20%) | Modular `src/` package, type-hinted, `pytest` unit tests ≥ 70% coverage, pre-commit hooks, README + API docs. |
| Analysis & Results (20%) | Phase 5 ablation table, confusion matrices, IoU/F1/κ/OA, qualitative map grid, statistical significance test. |
| Report Quality & Documentation (10%) | IEEE-style final report, 10–15 pages, with citations, figures, and methodology reproducible from code. |
| Presentation & Discussion (20%) | Phase 8 dry-run + live web demo + slide deck with architecture diagram and results. |

---

## 5. System Architecture Overview

```
 ┌──────────────────┐   ┌───────────────────┐   ┌─────────────────────┐
 │ Data Ingestion   │→→ │ Preprocessing &   │→→ │ Analysis Core       │
 │ (Sen2 / GEE /    │   │ Classical DIP     │   │ (Classical + U-Net) │
 │  Sen1Floods11)   │   │ (clouds, indices) │   │                     │
 └──────────────────┘   └───────────────────┘   └─────────┬───────────┘
                                                          │
                           ┌──────────────────────────────┴─┐
                           ↓                                ↓
                ┌────────────────────┐            ┌──────────────────┐
                │ Evaluation Suite   │            │ Web App (Streamlit│
                │ (IoU/F1/κ/OA,      │            │  + Folium map)   │
                │  ablation, plots)  │            │                  │
                └──────────┬─────────┘            └─────────┬────────┘
                           ↓                                ↓
                    ┌──────────────┐                 ┌─────────────┐
                    │ PDF Report   │←←←←←←←←←←←←←←←←←│ User AOI +  │
                    │ Generator    │                 │ date picker │
                    └──────────────┘                 └─────────────┘
```

---

## 6. Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| Language | Python 3.11 | Standard for geospatial + DL. |
| Geospatial I/O | `rasterio`, `geopandas`, `shapely`, `pyproj` | Industry standard. |
| Data access | Google Earth Engine Python API, `sentinelsat` fallback | Free Sentinel-2 access without manual download. |
| Classical DIP | `numpy`, `scipy.ndimage`, `scikit-image`, `opencv-python` | Covers filters, morphology, thresholds, histograms. |
| Deep Learning | PyTorch 2.x + `segmentation-models-pytorch` | Lightweight, Colab-friendly U-Net. |
| Visualization | `matplotlib`, `folium`, `rasterio.plot`, `plotly` | Static + interactive maps. |
| Web App | Streamlit 1.x + `streamlit-folium` | Fastest path to polished demo. |
| Report gen | `reportlab` or `WeasyPrint` (HTML→PDF) | Auto PDF with figures. |
| Testing | `pytest`, `pytest-cov` | Quality gate. |
| Dev tooling | `ruff`, `black`, `pre-commit`, `mypy` | Code quality for rubric. |
| Env mgmt | `conda-lock` + `environment.yml` | Reproducibility. |
| Version control | Git + GitHub (private repo) | Collaboration + submission artifact. |

---

## 7. Dataset & Study Area

### 7.1 Primary Case Study — Kerala Floods, August 2018
- **Pre-event Sentinel-2 tile:** 2018-07-06 to 2018-07-20 (baseline dry-period composite).
- **Post-event Sentinel-2 tile:** 2018-08-19 to 2018-08-25.
- **AOI bounding box (approx.):** 9.5°N, 76.0°E → 10.5°N, 77.0°E (Alappuzha, Ernakulam, Thrissur).
- **Bands used:** B2, B3, B4, B8, B11, B12 (10–20 m).
- **Ground reference (v1.2):** ~~UNOSAT flood extent shapefile~~ — external polygon mirrors unavailable at project time. **Manual GT on 5 stratified 512×512 patches is the sole Kerala reference** (annotated in QGIS, owned by B). Kerala evaluation is therefore **qualitative + spot-check** rather than pixel-wise.

### 7.2 Training/Validation Benchmark — Sen1Floods11
- 4,831 labeled 512×512 chips across 11 global flood events.
- Splits: train / val / test (standard Bonafilia split).
- Used for U-Net training and reported model metrics.

### 7.3 Auxiliary Layers (for exposure analysis — Phase 7)
- **WorldPop** 100 m population raster (2018).
- **OpenStreetMap** roads & buildings (downloaded via `osmnx`).
- **ESA WorldCover** 10 m land-cover map.

---

## 8. Phase-by-Phase Implementation Plan

> **Legend:** 🟢 = must-have · 🟡 = should-have · 🔵 = stretch/optional
> Each to-do is atomic and assignable. Owners are referenced by role letters (A/B/C/D — see §9).

---

### PHASE 0 — Project Setup & Kickoff (Days 1–3)
**Goal:** Unblock everyone with environment, repo, data access, and shared understanding.

#### To-Do
- [ ] 🟢 Create GitHub repo `dda-flood-kerala` with MIT license and `.gitignore` (Python). — Owner: A
- [ ] 🟢 Add `environment.yml` pinning Python 3.11 + all libs in §6. — Owner: A
- [ ] 🟢 Configure `pre-commit` with `ruff`, `black`, `end-of-file-fixer`. — Owner: A
- [ ] 🟢 Create folder skeleton: `src/`, `notebooks/`, `data/raw/`, `data/processed/`, `reports/`, `tests/`, `app/`. — Owner: A
- [ ] 🟢 Write initial `README.md` stub (quick-start, architecture diagram placeholder). — Owner: D
- [ ] 🟢 Register Google Earth Engine accounts for all members; verify Colab GPU access. — Owner: B
- [ ] 🟢 Download Sen1Floods11 to Google Drive (shared folder). — Owner: B
- [ ] 🟢 Create shared Trello/GitHub Projects board mirroring these phases. — Owner: D
- [ ] 🟢 Schedule 30-minute weekly sync + 15-minute mid-week stand-up. — Owner: D
- [ ] 🟡 Set up Weights & Biases (free tier) for experiment tracking. — Owner: C

**Exit criteria:** Every member can clone, `conda env create`, and run `python -c "import rasterio, torch; print('ok')"` inside Colab.

---

### PHASE 1 — Problem Definition, Literature Review & Data Acquisition (Week 1)
**Goal:** Lock the problem statement, assemble the dataset, and deliver a written background doc for the final report.

#### To-Do
- [ ] 🟢 Draft formal problem statement (1 page) with motivation, research questions, hypotheses. — Owner: D
- [ ] 🟢 Conduct literature review (≥ 10 peer-reviewed sources) — save as `reports/lit_review.md`. — Owner: D
- [ ] 🟢 Document the **exact rubric mapping** (§4) in `reports/rubric_mapping.md`. — Owner: D
- [ ] 🟢 Implement `src/data/gee_download.py` — GEE script to pull pre/post Sentinel-2 L2A scenes for a given AOI + date range; returns 6-band GeoTIFF. — Owner: B
- [ ] 🟢 Run download for Kerala AOI (pre: Jul 2018, post: Aug 2018). Store in `data/raw/kerala_2018/`. — Owner: B
- [ ] 🟢 Implement `src/data/sen1floods11_loader.py` — PyTorch `Dataset` class with band normalization and train/val/test splits. — Owner: C
- [ ] ~~🟢 Fetch UNOSAT Kerala 2018 flood shapefile → rasterize to 10 m ground-truth mask.~~ **Dropped in v1.2** — UNOSAT/Copernicus portal URLs dead; external GT unobtainable within the project window.
- [ ] 🟢 **(promoted from 🟡 in v1.2)** Manually annotate 5 × 512×512 validation patches in QGIS. Save as `data/gt/kerala_patches/patch_{01..05}.geojson`. These become the Kerala spot-check set. — Owner: B
- [ ] 🟢 Write `notebooks/01_data_eda.ipynb` — visualize pre/post RGB composites, band histograms, cloud-cover stats. — Owner: C
- [ ] 🟢 Unit tests for data loaders (`tests/test_data.py`) — assert shapes, dtype, CRS, NoData handling. — Owner: A

**Deliverable:** `reports/phase1_report.md` + raw data on shared Drive. Literature review merged to main branch.

**Exit criteria:** `pytest tests/test_data.py` passes; EDA notebook renders without errors.

---

### PHASE 2 — Preprocessing & Classical DIP Techniques (Week 2)
**Goal:** Implement the full classical DIP toolkit. This phase is where the bulk of "Implementation of DIP Techniques (20%)" rubric marks are earned.

#### To-Do — Preprocessing
- [ ] 🟢 `src/preprocess/reflectance.py` — convert DN → TOA/BOA reflectance, scale to [0,1]. — Owner: B
- [ ] 🟢 `src/preprocess/cloud_mask.py` — implement cloud/shadow mask using the Sentinel-2 SCL band + optional s2cloudless. — Owner: B
- [ ] 🟢 `src/preprocess/gap_fill.py` — fill cloud-masked pixels via temporal median composite across ±10-day window. — Owner: B
- [ ] 🟢 `src/preprocess/coregister.py` — ensure pre/post rasters are pixel-aligned (reproject + clip to common grid). — Owner: B
- [ ] 🟡 Histogram matching between pre/post images to normalize illumination. — Owner: C

#### To-Do — Spectral Indices & DIP Techniques
- [ ] 🟢 `src/dip/indices.py` — implement **NDWI**, **MNDWI**, **NDVI**, **AWEInsh**, **AWEIsh**. Each as pure NumPy function with docstring + formula reference. — Owner: C
- [ ] 🟢 `src/dip/thresholding.py` — implement **Otsu**, **adaptive (Gaussian/mean)**, **global fixed**, **triangle** thresholds; return binary water mask. — Owner: C
- [ ] 🟢 `src/dip/morphology.py` — opening, closing, hole-filling, small-object removal (`scikit-image.morphology`). — Owner: C
- [ ] 🟢 `src/dip/change_detection.py` — implement (a) image differencing on MNDWI, (b) image ratioing, (c) **PCA-based change detection**, (d) CVA (change vector analysis). — Owner: C
- [ ] 🟡 `src/dip/filters.py` — Gaussian, median, bilateral smoothing; Sobel/Canny edges for coastline refinement. — Owner: A
- [ ] 🟡 Frequency-domain analysis notebook — FFT of water vs land patches to justify spatial-domain choice. — Owner: A
- [ ] 🟢 `notebooks/02_classical_dip_walkthrough.ipynb` — visual walkthrough of every technique on Kerala data with captioned figures. — Owner: C
- [ ] 🟢 Unit tests for every `src/dip/*.py` module with synthetic fixtures. — Owner: A

**Deliverable:** `reports/phase2_report.md` explaining each technique, its DIP-theoretic justification, and chosen hyperparameters.

**Exit criteria:** Given a pre+post Kerala image pair, running `python -m src.pipelines.classical_baseline` writes a binary flood mask GeoTIFF in ≤ 60 s on CPU.

---

### PHASE 3 — Baseline Evaluation & Hybrid Integration (Week 3)
**Goal:** Quantitatively evaluate every classical method against ground truth; pick the best classical baseline for integration with the DL model.

#### To-Do
- [ ] 🟢 `src/eval/metrics.py` — implement IoU, Dice/F1, precision, recall, Cohen's κ, overall accuracy, per-class accuracy, confusion matrix. — Owner: A
- [ ] 🟢 `src/eval/ablation.py` — run every combination of {index} × {threshold} × {morphology on/off} against GT; export `results/ablation.csv`. — Owner: C
- [ ] 🟢 Generate qualitative comparison grid (PNG) — 4×4 figure of method vs GT overlays. — Owner: C
- [ ] 🟢 Statistical significance — McNemar's test between top-2 classical methods. — Owner: A
- [ ] 🟢 Pick best classical method as "classical baseline" and document choice in `reports/phase3_report.md`. — Owner: D

**Exit criteria:** Ablation table with ≥ 12 rows and a clearly justified winner.

---

### PHASE 4 — Deep Learning Segmentation (Weeks 3–4, overlaps with Phase 3)
**Goal:** Train a U-Net on Sen1Floods11, evaluate on Kerala, and compare against the classical baseline.

#### To-Do
- [ ] 🟢 `src/models/unet.py` — U-Net via `segmentation-models-pytorch` with ResNet-34 encoder, 6-channel input. — Owner: C
- [ ] 🟢 `src/train/train_unet.py` — training loop: BCE + Dice loss, Adam, cosine LR, early stopping, W&B logging. — Owner: C
- [ ] 🟢 Train on Sen1Floods11 train split for ≤ 50 epochs on Colab T4 (~4–6 hrs). — Owner: C
- [ ] 🟢 Save best checkpoint to Drive; export ONNX for inference portability. — Owner: C
- [ ] 🟢 `src/inference/predict.py` — tile-based inference on arbitrary-size GeoTIFF with overlap blending. — Owner: A
- [ ] 🟢 Evaluate U-Net on Sen1Floods11 test split + Kerala GT; log metrics. — Owner: C
- [ ] 🟡 Implement **hybrid fusion**: weighted combination of classical MNDWI+Otsu mask and U-Net probability map (ensemble). — Owner: C
- [ ] 🟡 Lightweight distillation or quantization for CPU inference in the web app. — Owner: A
- [ ] 🟢 `notebooks/03_dl_training_and_eval.ipynb` — training curves, test predictions, confusion matrix. — Owner: C

**Exit criteria:** U-Net achieves IoU ≥ 0.75 on Sen1Floods11 test split; inference on full Kerala scene completes in ≤ 2 min on Colab.

---

### PHASE 5 — Full Analysis, Damage Severity & Results (Week 5)
**Goal:** Produce all the analysis artifacts the final report will cite.

#### To-Do
- [ ] 🟢 `src/analysis/severity.py` — classify AOI grid cells into 4 severity classes using {flooded %, depth proxy from NDWI intensity, duration if multi-date}. — Owner: C
- [ ] 🟢 `src/analysis/quantify.py` — compute flooded area (km²), % of AOI, per-landcover breakdown using ESA WorldCover. — Owner: B
- [ ] 🟢 Produce final ablation table: {Classical best, U-Net, Hybrid} × {IoU, F1, κ, OA, runtime}. — Owner: A
- [ ] 🟢 Generate all figures for report (maps, histograms, confusion matrices, training curves) as 300 DPI PNG in `reports/figures/`. — Owner: D
- [ ] 🟢 Write `reports/phase5_analysis.md` — narrative analysis with statistical interpretation. — Owner: D
- [ ] 🟡 Error-analysis notebook — categorise false positives (shadows, dark soil, asphalt) and false negatives (turbid water, vegetation-covered). — Owner: C

**Exit criteria:** Single command `make results` regenerates every figure + table from scratch.

---

### PHASE 6 — Web Application (Week 6)
**Goal:** Ship the interactive demo that anchors the "Presentation & Discussion" rubric.

#### To-Do
- [ ] 🟢 `app/streamlit_app.py` — layout: sidebar (AOI picker, date pickers, method dropdown, run button) + main (tabs: Map, Report, About). — Owner: A
- [ ] 🟢 Folium + `streamlit-folium` interactive map with layers: pre-RGB, post-RGB, flood mask, change map, severity choropleth. Toggle + opacity sliders. — Owner: A
- [ ] 🟢 Wire backend: on "Run", call `src/pipelines/full_pipeline.py` (classical / U-Net / hybrid switch) and render results. — Owner: A
- [ ] 🟢 Caching with `@st.cache_data` so repeat runs on same AOI are instant. — Owner: A
- [ ] 🟢 Download buttons: GeoTIFF mask, CSV statistics, PDF report. — Owner: A
- [ ] 🟢 `app/report_generator.py` — HTML template → WeasyPrint PDF with maps, stats, methodology section. — Owner: D
- [ ] 🟢 Deploy to **Streamlit Community Cloud** (free) or Hugging Face Spaces. — Owner: A
- [ ] 🟢 Smoke tests: end-to-end run on Kerala AOI in ≤ 3 min from click to PDF. — Owner: A
- [ ] 🟡 Add session-state "history" so user can compare two runs side-by-side. — Owner: A

**Exit criteria:** Public URL that anyone (including evaluator) can load and run without local setup.

---

### PHASE 7 — Stretch / "Wow" Features (Week 7)
**Goal:** Push into the Excellent band of rubric criteria 3–5.

#### To-Do
- [ ] 🔵 **Time-series animation** — loop over 5 dates in Aug 2018, produce flood-progression GIF + Streamlit slider. — Owner: B
- [ ] 🔵 **Population exposure overlay** — intersect flood mask with WorldPop → estimated people affected. — Owner: B
- [ ] 🔵 **Infrastructure impact** — intersect flood mask with OSM roads & buildings → km of roads and count of buildings in flooded cells. — Owner: B
- [ ] 🔵 **Advanced cloud-gap filling** — harmonic-regression temporal filling (HANTS) across 1-year Sentinel-2 stack. — Owner: C
- [ ] 🟡 **Sentinel-1 SAR flood mapping (bonus-marked by instructor)** — download VV/VH GRD scenes via GEE for same Kerala pre/post dates; apply speckle filter (Refined Lee or Gamma MAP), compute VV backscatter change, threshold (Otsu on log-VV) to generate independent SAR flood mask; fuse with optical mask and report agreement (cloud-robust comparison). Document as its own subsection in final report. — Owner: C
- [ ] 🔵 Polish PDF report template with institute logo, captions, TOC. — Owner: D

**Exit criteria:** Each enabled stretch feature is toggleable in the Streamlit UI and documented in the final report.

---

### PHASE 8 — Report, Presentation & Submission (Week 8)
**Goal:** Convert the engineered artefacts into submission-quality report and presentation.

#### To-Do — Report (10% rubric)
- [ ] 🟢 Final report (10–15 pages, IEEE template, LaTeX on Overleaf) — sections: Abstract, Intro, Related Work, Dataset & Study Area, Methodology (Classical + DL), Experiments, Results, Discussion, Conclusion, References. — Owner: D
- [ ] 🟢 Insert all Phase-5 figures with captions and in-text references. — Owner: D
- [ ] 🟢 ≥ 15 references in IEEE style. — Owner: D
- [ ] 🟢 Appendix: architecture diagram, hyperparameter table, hardware spec. — Owner: D
- [ ] 🟢 Proof-read pass by all members; run Grammarly + LaTeX spell-check. — Owner: all

#### To-Do — Presentation (20% rubric)
- [ ] 🟢 **20-minute** presentation: 18-slide deck (Google Slides) targeting ~15 min talk + 5 min Q&A. Structure (with suggested time): Title/Team (0:30) → Problem & Motivation (2:00) → Related Work (1:30) → Data & Study Area (1:30) → Methodology: Classical DIP (3:00) → Methodology: U-Net (2:00) → **SAR cross-check** (1:30) → Results & Ablation (2:30) → **Live web-app demo** (3:00) → Exposure/Impact (1:00) → Conclusion & Future Work (1:00) → Q&A (5:00). — Owner: D
- [ ] 🟢 Embed short demo screen-recording (60 s) as fallback if live demo fails. — Owner: A
- [ ] 🟢 Dry-run the full **20-minute** presentation **twice** as a team; time each speaker with a stopwatch; trim any section running long. — Owner: all
- [ ] 🟢 Prepare anticipated Q&A — list of 20 likely questions and crisp 1–2-sentence answers. — Owner: all
- [ ] 🟢 Assign speaking sections by member expertise (A: architecture+app, B: data+exposure, C: DIP+DL methods, D: problem+results+report). — Owner: D

#### To-Do — Submission
- [ ] 🟢 Tag release `v1.0` on GitHub; attach report PDF, slide PDF, README. — Owner: A
- [ ] 🟢 Verify Streamlit deployment URL is live and demo flow works. — Owner: A
- [ ] 🟢 Final rubric self-audit against §4 — confirm every criterion has its evidence artefact. — Owner: D
- [ ] 🟢 Submit via course LMS before deadline with ≥ 24 hr buffer. — Owner: D

**Exit criteria:** Report submitted, demo URL stable, tag pushed, dry-run confidence ≥ 8/10 from every team member.

---

## 9. Roles & Responsibilities (3–4 members)

| Role | Primary Scope | Typical To-Dos |
|---|---|---|
| **A — Tech Lead / App & Infra** | Repo hygiene, web app, inference pipeline, evaluation utilities | CI/CD, Streamlit, metrics, unit tests |
| **B — Data & Geospatial Lead** | GEE downloads, coregistration, preprocessing, exposure overlays | Sentinel-2 ingestion, WorldPop/OSM, cloud masking |
| **C — DIP & ML Lead** | Spectral indices, thresholding, U-Net training, ablations | `src/dip/*`, model training, hybrid fusion |
| **D — PM / Report & Presentation Lead** | Literature, report, deck, rubric mapping, demos | Writing, figures, stakeholder interface |

> Adjust for team size: if 3 members, merge A+B; if 4, keep split as above.

---

## 10. Risk Register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Cloud cover in post-event Sentinel-2 over Kerala | High | Med | Phase-7 cloud-gap fill; fallback to Sentinel-1 SAR. |
| R2 | GEE quota/auth issues | Med | Med | Pre-download data to Drive in Phase 1; cache locally. |
| R3 | U-Net training slow on free Colab T4 | Med | Med | Use ResNet-34 encoder (not -50); limit to 50 epochs; checkpoint frequently. |
| R4 | Streamlit Cloud memory limit exceeded when loading full rasters | Med | High | Pre-tile Kerala raster; load only AOI sub-tile on demand. |
| R5 | Team member unavailable (illness, exams) | Med | High | Cross-train on at least one adjacent module; keep to-dos atomic. |
| R6 | Scope creep into earthquakes/fires | Low | High | This PRD locks scope to floods; changes require written amendment. |
| R7 | Ground-truth shapefile licensing ambiguity | Low | Med | Use UNOSAT (CC-BY) + self-annotated patches; cite clearly. |

---

## 11. Deliverables Checklist (final submission bundle)

- [ ] GitHub repo (public or course-access) with README, LICENSE, `environment.yml`.
- [ ] Trained U-Net checkpoint + ONNX export on Drive (link in README).
- [ ] Live web-app URL.
- [ ] Final PDF report (≥ 10 pages, IEEE).
- [ ] Slide deck PDF + 30-s demo screen-recording.
- [ ] Ablation results CSV + figures directory.
- [ ] Auto-generated Kerala damage PDF report (sample output).
- [ ] Rubric self-audit document (`reports/rubric_mapping.md`).

---

## 12. Timeline at a Glance (6–8 Weeks)

| Week | Phase | Milestone |
|---|---|---|
| 0 (days 1–3) | Phase 0 | Repo + env ready |
| 1 | Phase 1 | Data in hand, lit review done |
| 2 | Phase 2 | Classical DIP toolkit complete |
| 3 | Phase 3 + start 4 | Classical baseline + U-Net training started |
| 4 | Phase 4 | U-Net trained, hybrid fusion done |
| 5 | Phase 5 | Analysis & results finalized |
| 6 | Phase 6 | Web app deployed |
| 7 | Phase 7 | Stretch features + polish |
| 8 | Phase 8 | Report, deck, dry-runs, submit |

---

## 13. Resolved Decisions (locked in v1.1, 2026-04-20)

| # | Question | Resolution | Impact on Plan |
|---|---|---|---|
| Q1 | Does the instructor supply an AOI, or is Kerala 2018 accepted? | **Team's own choice** — Kerala 2018 confirmed. | AOI locked; no change to Phase 1 downloads. |
| Q2 | Is PyTorch allowed? | **Yes.** | No change — stack stays as §6. |
| Q3 | Presentation length? | **20 minutes** (≈ 15 min talk + 5 min Q&A). | Phase 8 deck expanded to 18 slides with per-section timing. |
| Q4 | Do Sentinel-1 SAR results earn extra marks? | **Yes — bonus marks.** | SAR promoted from 🔵 stretch to 🟡 should-have in Phase 7; new to-do expands scope (speckle filter, VV thresholding, fusion with optical, dedicated report subsection). |

No further open questions at this time. Future amendments will version-bump this document.

---

*End of PRD v1.1.*
