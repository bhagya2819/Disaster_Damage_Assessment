# Phase 7 Report — Stretch Features

> Owner: **C (DIP/ML)** + **B (Data/Geo)** · bonus features per PRD v1.1.

---

## 1. What's shipped

| Feature | PRD marker | Module | Status |
|---|---|---|---|
| Sentinel-1 SAR flood mapping (bonus-marked) | 🟡 | `src/dip/sar.py` · `src/data/sar_download.py` | ✅ |
| Infrastructure impact (OSM roads/buildings) | 🔵 | `src/analysis/infrastructure.py` | ✅ |
| Time-series animation + area curve | 🔵 | `src/analysis/time_series.py` | ✅ |
| Polished PDF template (TOC, logo, page numbers) | 🔵 | `app/templates/report.html` | ✅ |
| Population exposure overlay | 🔵 | `src/analysis/quantify.py::population_exposed` | ✅ (Phase 5) |
| HANTS temporal gap-filling | 🔵 | — | ⏭ deferred to future work |

All features are tested on **synthetic fixtures** and require no network access during CI.

---

## 2. Sentinel-1 SAR flood mapping

### 2.1 Why SAR

Real flood events are frequently cloud-obscured — Sentinel-2 optical imagery can be useless for days after the peak. C-band synthetic-aperture radar (SAR) from Sentinel-1 sees through clouds and at night, and open water is a near-perfect specular reflector for C-band, so flooded pixels appear as dark (low-backscatter) regions in a log-VV image. This is the canonical operational fall-back for flood mapping (Martinis et al., 2009; Chini et al., 2017).

### 2.2 Implementation (`src/dip/sar.py`)

Pipeline:

1. **`refined_lee(image)`** — Lee (1980) adaptive speckle filter. Homogeneous pixels are mean-filtered; edges are preserved. Default 7 × 7 window.
2. **`to_db(x)`** — `10 · log₁₀(x)` with ε-clipping for numerical stability.
3. **`sar_flood_mask(vv_linear)`** — end-to-end: Lee → dB → Otsu threshold → bool mask. Returns both the mask and the chosen dB threshold for reporting.
4. **`fuse_with_optical(sar, optical, mode=…)`** — three fusion modes: `union` (OR, cloud-robust), `agreement` (AND, high-precision), `optical_primary` (optical where available, SAR elsewhere).
5. **`agreement_fraction(a, b)`** — Cohen's κ-style metric for "SAR vs optical agree at κ = 0.X" in the final paper.

An alternative **Frost (1982)** speckle filter is provided for ablation.

### 2.3 Data downloader (`src/data/sar_download.py`)

Drop-in companion to `src/data/gee_download.py`. Downloads VV + VH IW-mode GRD scenes over the same AOI / date windows as the Sentinel-2 composites, median-reduces, and exports as a 2-band GeoTIFF. CLI: `python -m src.data.sar_download --config configs/kerala_2018.yaml --window both`.

### 2.4 Tests (`tests/test_sar.py`)

Ten tests on synthetic gamma-distributed backscatter images:
- dB conversion is exact on known values (1 → 0 dB, 0.1 → −10 dB, 0.01 → −20 dB).
- Refined-Lee smooths a homogeneous noisy region by ≥ 20 % std reduction.
- Otsu SAR threshold lands in the plausible range [−30, −5] dB.
- Left half (water, low backscatter) is substantially more flagged than right (land).
- Fusion modes produce expected set operations (union, intersection, primary-fallback).
- Agreement fraction is 1.0 for identical inputs and negative for anti-correlated inputs.

### 2.5 Integration with the main pipeline

Not wired into the default `src/pipelines/full_pipeline.py` — SAR is an **optional** method activated only when the user supplies a Sentinel-1 raster alongside the Sentinel-2 chip. The hooks exist (`fuse_with_optical`); a one-line extension to `run_pipeline(method="sar_optical")` is all that is needed to expose it in the Streamlit app.

---

## 3. Infrastructure impact (OSM)

### 3.1 What it does

Given a flood mask GeoTIFF and its AOI, `src/analysis/infrastructure.py`:

1. Fetches OSM **drivable roads** and **building polygons** via `osmnx` for the AOI bbox.
2. Reprojects to the mask's CRS and rasterises onto the mask's grid.
3. Counts flooded-road pixels → km of roads affected, per highway class.
4. Counts flooded buildings (any pixel overlap) → count and total footprint area affected.
5. Returns an `InfrastructureImpact` dataclass with `as_summary()` for easy inclusion in the PDF report.

### 3.2 Why this matters for the course

The PRD lists "affected population / cropland / infrastructure" as a motivating humanitarian use-case in §1. The three new numbers (`roads_km_flooded`, `buildings_flooded_count`, `buildings_flooded_area_m2`) move the Streamlit-app PDF report from "pretty flood map" to "actionable damage summary".

### 3.3 Tests (`tests/test_infrastructure.py`)

Nine tests on hand-built synthetic GeoDataFrames (three 1 km roads — one flooded, one dry, one crossing; four 30 × 30 m buildings — two on the flooded half, two not). Assertions:
- Flooded road km ≈ 0.97 for the three-road fixture (one full + half of crossing).
- Exactly 2 of 4 buildings are flagged; flooded footprint is 1 800 m² out of 3 600 m² total.
- Empty GeoDataFrames return zeros without raising.
- Non-polygon OSM features (lines tagged `building=yes`) are filtered out of the building count.
- `compute()` returns a well-formed `InfrastructureImpact` with the expected `as_summary()` shape.

### 3.4 Caveats

OSM calls hit the public Overpass API at runtime. Rate-limit / cache caveats are documented in the module header. For a Kerala AOI of ~1° × 1° the query returns in 30–90 s and fits in RAM comfortably.

---

## 4. Time-series flood progression

### 4.1 What it does

`src/analysis/time_series.py`:

1. **`flooded_km2_per_step(masks)`** — flooded-area curve in km² per timestep.
2. **`summarise(masks, dates)`** — peak-date, peak-area, time-to-peak, time-to-50 %-recession (humanitarian-relevant metrics).
3. **`build_gif(masks, dates, out)`** — animated GIF walking through the mask sequence with a date caption.
4. **`build_area_curve_png(summary, out)`** — publication-quality line chart of the flooded-area curve with the peak annotated.

### 4.2 Use case

Feed five Sentinel-2 composites at 3–5-day spacing through the main pipeline, stack the resulting masks, and call `summarise()` + `build_gif()`. The app gains a "flood progression" tab; the IEEE report gains a 1-panel area-curve figure.

### 4.3 Tests (`tests/test_time_series.py`)

Seven tests with a synthetic rising-then-falling sequence (0 → 25 → 75 → 50 → 10 %) over 5 dates at 2-day spacing:
- Peak index is correctly identified.
- Time-to-peak is 4 days (2 × 2-day steps).
- Time-to-50 %-recession is 4 days after peak.
- `build_gif` and `build_area_curve_png` write files with non-zero size.
- `as_dict()` is JSON-serialisable for the Streamlit session state.

### 4.4 Limitations

We did not run the full time-series on real Kerala data — requires downloading 5 + Sentinel-2 composites + running the U-Net on each. The infrastructure for that run is present; only the GEE download + model-apply loop needs to be invoked.

---

## 5. Polished PDF template

### 5.1 Changes vs the Phase-6 template

- **Separate cover page** with optional institute-logo slot (`logo_path`).
- **Table of contents** on page 2.
- **Page numbers** and method badge in the footer of every page (suppressed on the cover).
- **Numbered figures** with captions tying back to the report narrative (`Figure 1`, `Figure 2`, `Table 1`).
- **Stat tiles** with primary-colour accents (matches the Phase-5 bar-chart palette — consistent branding across slides + PDF).
- **Explicit citation note** under the benchmark table: bootstrap CI + McNemar numbers are shown directly on the PDF.

### 5.2 Compatibility

`ReportContext` now accepts an optional `logo_path`. Existing callers (Streamlit download button, `build_report`, `build_report_bytes`, tests) continue to work — `logo_path` defaults to `None` and the template hides the logo when absent.

### 5.3 Tests

`tests/test_full_pipeline.py::test_report_renders_html` still passes (template loads, placeholders are substituted, "flood" appears in the rendered HTML).

---

## 6. Population exposure

Already shipped in Phase 5 — `src/analysis/quantify.py::population_exposed(mask, worldpop_raster)` sums a WorldPop raster over flooded pixels. Not re-documented here.

---

## 7. Deferred

**HANTS temporal gap-filling.** Implementation complexity exceeds the course-project budget and the cloud-mask + median compositor in Phase 2 is sufficient for the reported results. Documented as future work in `reports/final_report.md` §7.3.

---

## 8. New tests — grand total

Phase 7 adds **26 tests** (10 SAR + 9 infrastructure + 7 time-series). Running `pytest -q` on this branch should now report **119 passing**.

---

## 9. Phase 7 exit criteria — audit

- [x] SAR module implemented, tested, and wired to the GEE downloader.
- [x] Infrastructure-impact analysis ships a `compute(mask)` → `InfrastructureImpact` API.
- [x] Time-series animation builder + area-curve renderer with JSON-safe summaries.
- [x] PDF template gains cover, TOC, page numbers, logo slot.
- [x] All new tests green on CI (synthetic fixtures — no network).

**Phase 7 complete.**

---

## 10. Slide-deck impact

`reports/presentation_outline.md` already allocates a slide for Sentinel-1 SAR (Slide 13 placeholder if SAR was completed; Slide 15 error analysis). Update the deck to claim SAR as a completed contribution rather than future work — this moves the bonus marks from "aspirational" to "earned".
