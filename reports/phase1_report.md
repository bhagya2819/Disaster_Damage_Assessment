# Phase 1 Report — Problem Definition & Data Acquisition

> Owner: **D (PM/Report)** — this document goes into §2 and §4 of the final IEEE-style report.

---

## 1. Problem Statement (~1 page)

### 1.1 Context
Floods are India's most frequent and economically damaging natural disaster. The August 2018 Kerala floods killed 483 people, displaced ~1.4 million, and caused estimated losses of ₹40,000 crore. Post-disaster response hinges on **fast, spatially explicit** maps of inundation extent — ground surveys are too slow and too dangerous during active flooding. Satellite remote sensing is the only scalable alternative; free, global, 10 m Sentinel-2 imagery makes the problem tractable with student-level compute.

### 1.2 Formal problem
Given:
- a pre-event Sentinel-2 L2A surface-reflectance composite, `I_pre ∈ ℝ^{6×H×W}`, over an AOI,
- a post-event composite `I_post ∈ ℝ^{6×H×W}` within 1–2 weeks of the flood peak,

produce:
- a binary flood mask `M ∈ {0,1}^{H×W}` (water = 1),
- a 4-class severity map `S ∈ {0,1,2,3}^{H×W}` (none/low/moderate/severe),
- a quantitative damage report: flooded area in km², % of AOI, per-landcover breakdown, population & infrastructure exposure.

### 1.3 Research questions
- **RQ1.** How does a *classical* DIP pipeline (MNDWI + Otsu + morphology) compare against a *deep-learning* U-Net on flood-mask IoU, when both are evaluated on the Sen1Floods11 benchmark and on independent Kerala ground truth?
- **RQ2.** Does a *hybrid* ensemble (classical index + CNN probability) outperform either component?
- **RQ3.** Which spectral indices (NDWI / MNDWI / AWEI / NDVI) are most discriminative for *turbid* floodwaters over vegetated terrain, as occur in Kerala?
- **RQ4.** How much does Sentinel-1 SAR (VV/VH log-backscatter) compensate for cloud cover in the post-event optical composite?

### 1.4 Hypotheses
- **H1.** MNDWI + Otsu + morphological opening achieves IoU ≥ 0.60 on Sen1Floods11 test.
- **H2.** A ResNet-34 U-Net trained on Sen1Floods11 HandLabeled reaches IoU ≥ 0.75.
- **H3.** Hybrid fusion (mean-of-probabilities) beats the better singleton by ≥ 2 IoU points.
- **H4.** SAR-only flood mask agrees with optical mask at κ ≥ 0.55 on non-cloudy pixels.

### 1.5 Scope
In-scope: optical (Sentinel-2) + SAR (Sentinel-1) flood mapping for Kerala AOI.
Out-of-scope: sub-meter building-level damage (needs Maxar), other disaster types, real-time ingestion.

### 1.6 Deliverables (this phase)
- `data/raw/kerala_2018/kerala_2018_{pre,post}.tif`
- `data/gt/kerala_gt.tif`
- Sen1Floods11 HandLabeled tree on shared Drive
- `reports/lit_review.md`
- `reports/rubric_mapping.md`
- Passing `pytest tests/test_data.py`

---

## 2. Data Summary

| Asset | Source | Size | Licence |
|---|---|---|---|
| Sentinel-2 L2A composites (Kerala pre/post) | GEE `COPERNICUS/S2_SR_HARMONIZED` | ~120 MB | Copernicus CC-BY |
| Sen1Floods11 HandLabeled | `gs://sen1floods11/` | ~3 GB | CC-BY 4.0 |
| UNOSAT Kerala 2018 flood polygons | UNOSAT product #2728 | ~5 MB | CC-BY-IGO |
| ESA WorldCover v200 (2021) | GEE `ESA/WorldCover/v200` | on-demand | CC-BY 4.0 |
| WorldPop 100 m (2018) | `WorldPop/GP/100m/pop` | ~50 MB | CC-BY 4.0 |
| OSM buildings + roads | `osmnx` | on-demand | ODbL |

---

## 3. Phase 1 Exit Criteria — Audit

_Check each box after its artefact is verified on disk / in CI:_

- [ ] Sentinel-2 pre and post composites downloaded and loadable via `rasterio`.
- [ ] Sen1Floods11 HandLabeled tree readable; train/valid/test/bolivia splits all non-empty.
- [ ] UNOSAT polygons rasterised to a 10 m mask aligned with the post raster.
- [ ] `pytest tests/test_data.py` passes locally and in Colab.
- [ ] EDA notebook runs end-to-end without errors.
- [ ] Literature review ≥ 10 references committed.

---

*Update this file as Phase 1 progresses; final version feeds §2 of the IEEE report.*
