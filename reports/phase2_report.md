# Phase 2 Report — Classical Digital Image Processing

> Owner: **D (PM/Report)** · feeds §3 (Methodology — Classical DIP) and §4 (Experiments — ablation) of the final IEEE report.

---

## 1. Objective

Build a transparent, literature-grounded classical DIP pipeline that converts a pre/post Sentinel-2 pair into a binary flood-water mask — without any machine learning. This pipeline serves three purposes:

1. **Baseline** for the Phase 4 U-Net to beat.
2. **Explainability** — every decision is traceable to a scalar threshold or morphological operation, which is valuable for the rubric's "Implementation of DIP Techniques" criterion (20 %).
3. **Cloud-free deployability** — runs on CPU in < 60 s on the full Kerala AOI, so the Streamlit app can always compute a classical mask even if the DL model checkpoint is not loaded.

---

## 2. Pipeline

```
pre.tif ─┐
         ├─► MNDWI_pre ─┐
         │              ├─► ΔMNDWI ──► Otsu ──► change_mask
         │              │
post.tif ├─► MNDWI_post ┤
         │              └─► Otsu ──► water_mask
         │                                      │
         │                water_mask ∩ change_mask
         │                                      │
         │                                      ▼
         │                                morphology.clean
         │                                      │
         └──────────────── co-registered post grid ─┘─► flood_mask.tif
```

Output: single-band `uint8` GeoTIFF, `1 = flood`, `0 = non-flood`, plus a JSON sidecar with area/threshold stats.

---

## 3. Technique-by-technique justification

### 3.1 MNDWI (Modified Normalized Difference Water Index)

- **Formula:** `MNDWI = (Green − SWIR1) / (Green + SWIR1)`
- **Why primary:** Xu (2006) demonstrated MNDWI outperforms McFeeters' NDWI over **built-up areas**, which is critical for Kerala whose flooded districts include Ernakulam (a major city). Urban pixels have high SWIR reflectance → negative MNDWI → correctly excluded.
- **Alternative indices also computed** (`src/dip/indices.py`) for the ablation:
  - NDWI (McFeeters 1996) — historical baseline.
  - NDVI (Rouse 1974) — complement, confirms vegetation loss.
  - AWEInsh / AWEIsh (Feyisa 2014) — shadow-robust.
- **Numerical safety:** `+ 1e-9` in the denominator; NaN/Inf filtered in downstream thresholding.

### 3.2 Otsu thresholding

- **Source:** Otsu (1979). Chooses the threshold that maximises between-class variance in a bimodal grey-level histogram.
- **Why primary:** MNDWI over a flooded AOI is strongly bimodal (water peak near +0.3, non-water near −0.3) → ideal Otsu input.
- **Ablation alternatives** (`src/dip/thresholding.py`):
  - Triangle (Zack 1977) — robust when peaks are unequal.
  - Yen (1995) — entropy-criterion.
  - Li (1993) — minimum cross-entropy.
  - Adaptive Gaussian (block 51 × 51 ≈ 500 m) — for scenes with radiometric gradients.
- **Sanity test:** Otsu threshold in range [0, 0.4] — outside this range usually signals cloud contamination or AOI mis-framing.

### 3.3 Change detection

- **ΔMNDWI = MNDWI_post − MNDWI_pre** (`src/dip/change_detection.py:mndwi_difference`).
- Intersecting the water mask with the change mask filters out **permanent water bodies** (rivers, lakes always present in pre and post) so the output is strictly **flood-induced** new water.
- Alternative change-detection techniques implemented for the ablation:
  - Image differencing (Singh 1989) — per-band signed difference.
  - Image ratioing — multiplicative, robust to additive offsets.
  - Change Vector Analysis (CVA) — √(Σ diff²) across bands.
  - PCA-based change — last principal component of the stacked pre+post tensor (Deng 2008).

### 3.4 Morphological post-processing

Implemented in `src/dip/morphology.py:clean`:

1. **Opening** with `disk(radius=1)` — removes isolated water pixels (salt-and-pepper).
2. **Closing** with `disk(radius=1)` — bridges 1-px gaps caused by threshold boundary noise.
3. **Small-object removal** @ 25 px ≈ 0.25 ha — drops spurious speckle.
4. **Small-hole filling** @ 25 px — closes small gaps inside real flood bodies (e.g., roofs sticking above water).

Justification for 25 px: at 10 m Sentinel-2 pixel size, 25 px = 2,500 m². Smaller blobs are below the reliable resolution limit for flood mapping and almost always noise.

### 3.5 Filters (optional, used in report figures)

- **Gaussian** (σ = 1–2) for pre-smoothing.
- **Median** (size = 3) for edge-preserving denoising.
- **Bilateral** — edge-preserving, used before Sobel/Canny for coastline extraction.
- **Sobel / Canny** — shoreline outlines (Phase 7 stretch feature).

### 3.6 Frequency-domain sanity check

Notebook §6 plots `log |FFT(MNDWI_post)|`. Expected: energy concentrated near DC with smooth decay → spatial-domain filtering is the correct framework. If periodic striping appeared, we'd insert a notch filter before thresholding.

---

## 4. Hyperparameters (locked defaults)

| Param | Value | Source |
|---|---|---|
| Water index | MNDWI | Xu 2006 |
| Water threshold method | Otsu | Otsu 1979 |
| Change index | ΔMNDWI | Singh 1989 |
| Change threshold | Otsu | — |
| Morphology opening radius | 1 px (disk) | tuned on Phase-2 synthetic tests |
| Morphology closing radius | 1 px (disk) | — |
| Min object area | 25 px (0.25 ha) | matches NRSA flood reports |
| Min hole area | 25 px | symmetric with above |

All defaults overridable via the CLI flags of `src/pipelines/classical_baseline.py`.

---

## 5. Testing

- `tests/test_dip.py` — 20 tests over indices / thresholding / morphology / change detection / filters using synthetic water+veg+built scenes.
- `tests/test_preprocess.py` — reflectance roundtrip, SCL masking, coregistration alignment, histogram matching.
- `tests/test_pipelines.py` — end-to-end pipeline run on synthetic flood scene; asserts detection rate 30–70 % on a half-flooded AOI with spatial consistency check.

Expected runtime of the full Phase-2 test suite on Colab CPU: **< 30 s**.

---

## 6. Phase 2 exit criteria — audit

- [ ] `src/dip/{indices,thresholding,morphology,change_detection,filters}.py` committed.
- [ ] `src/preprocess/{reflectance,cloud_mask,coregister,histogram_match}.py` committed.
- [ ] `src/pipelines/classical_baseline.py` writes a valid GeoTIFF + JSON sidecar.
- [ ] Full pytest suite green in Colab.
- [ ] `notebooks/02_classical_dip_walkthrough.ipynb` runs end-to-end with 5 saved figures.
- [ ] `data/processed/kerala_2018/flood_mask_classical.tif` produced on actual Kerala data.

When complete → proceed to **Phase 3 (baseline evaluation & ablation)**.

---

## 7. References (condensed; full list in `lit_review.md`)

- Xu, H. (2006). MNDWI. IJRS 27(14):3025–3033.
- Otsu, N. (1979). A threshold selection method. IEEE SMC 9(1):62–66.
- McFeeters, S.K. (1996). NDWI. IJRS 17(7):1425–1432.
- Feyisa, G.L. et al. (2014). AWEI. RSE 140:23–35.
- Singh, A. (1989). Digital change detection. IJRS 10(6):989–1003.
- Deng, J. et al. (2008). PCA-based change detection. IJRS 29(16):4823–4838.
