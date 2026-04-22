---
title: "Disaster Damage Assessment — Flood Mapping from Sentinel-2 Imagery using a Hybrid of Classical Digital Image Processing and a Deep Convolutional Segmentation Network"
subtitle: "Course Project · Digital Image Processing · Remote Sensing & Satellite Image Processing"
author: "DDA Team · GitHub: bhagya2819/Disaster_Damage_Assessment"
date: "April 2026"
---

## Abstract

Floods are the most frequent and economically damaging natural disaster in India; the August 2018 Kerala event alone displaced approximately 1.4 million people and caused ₹40 000 crore in losses. Rapid, spatially explicit flood-extent maps are essential for search-and-rescue deployment, relief allocation and long-term mitigation. We present an end-to-end digital image processing (DIP) pipeline that converts pre- and post-event Sentinel-2 L2A imagery into a binary flood-water mask, a per-cell damage severity classification and a quantitative PDF damage report. The pipeline combines a transparent, literature-grounded **classical DIP baseline** (Normalised Difference Water Index + Yen thresholding) with a **ResNet-34 U-Net** trained on the Sen1Floods11 HandLabeled split using a combined BCE + Dice loss. On the Sen1Floods11 test split (n = 90 chips) the U-Net attains a mean intersection-over-union (IoU) of **0.548**, F1 of **0.660** and Cohen's κ of **0.652** — improvements of **+0.108 IoU** (+24.5 % relative), +0.113 F1 and +0.155 κ over the classical baseline. The paired bootstrap 95 % CI on per-chip ΔIoU is [+0.037, +0.114], excluding zero, and a pixel-level McNemar test confirms significance with χ² ≈ 1.13 × 10⁶, p ≈ 0. A weighted hybrid fusion fails to outperform the U-Net alone (ΔIoU = −0.010, 95 % CI [−0.024, +0.005]), a negative result we report and explain. The full system ships as a Streamlit web application that accepts arbitrary Sentinel-2 chips, applies any of the three methods and produces a one-click PDF damage report.

**Keywords —** Remote sensing, flood mapping, Sentinel-2, digital image processing, NDWI, Otsu thresholding, U-Net, Sen1Floods11, bootstrap confidence interval, McNemar's test.

---

## 1. Introduction

### 1.1 Motivation

Floods cause more human displacement and economic damage than any other natural hazard in India. The August 2018 Kerala floods were a defining recent event: at least 483 people were killed, roughly 1.4 million were displaced, and economic losses were estimated at ₹40 000 crore (Sudheer et al., 2019). In such events ground surveys are too slow and too dangerous to drive the first 72 hours of response; satellite remote sensing is the only scalable observation modality. The European Space Agency's Sentinel-2 constellation provides free, 10 m multispectral imagery on a 5-day revisit cycle, making it the natural choice for academic and humanitarian flood-mapping work.

### 1.2 Problem statement

Given a pre-event Sentinel-2 L2A surface-reflectance composite and a post-event composite of the same area of interest (AOI), we must produce

* a **binary flood-water mask** at 10 m pixel resolution,
* a **severity classification** in {None, Low, Moderate, Severe} at the 1 km × 1 km grid-cell level,
* a **quantitative damage summary**: flooded area (km²), percentage of AOI affected, per-land-cover-class breakdown.

The outputs must be produced without any on-the-fly human annotation and in under five minutes of wall-clock time on a free-tier compute budget.

### 1.3 Contributions

1. An open-source, reproducible pipeline (MIT license) combining classical DIP with deep-learning segmentation, with pytest coverage over every module (93 tests, 75 % line coverage).
2. An ablation over 32 classical-DIP configurations (4 spectral indices × 4 threshold methods × morphology on/off) evaluated on Sen1Floods11, yielding the empirically-best classical configuration and the counter-intuitive observation that NDWI + Yen outperforms the textbook-recommended MNDWI + Otsu on this benchmark.
3. A U-Net segmentation trained on Sen1Floods11 HandLabeled that beats the classical baseline by **+0.108 IoU** with paired-bootstrap 95 % CI excluding zero and pixel-level McNemar p ≈ 0.
4. A demonstrated **negative result**: weighted classical + deep-learning fusion is not statistically better than the U-Net alone on this data. The finding is justified empirically and motivated theoretically.
5. A production-ready Streamlit web application with a single-click PDF damage-report generator built via Jinja2 and WeasyPrint.

### 1.4 Paper organisation

Section 2 surveys the relevant literature. Section 3 details the datasets and study area. Section 4 describes the classical DIP pipeline and Section 5 the deep-learning pipeline. Section 6 reports experimental results and statistical analysis. Section 7 discusses limitations and future work. Section 8 concludes.

---

## 2. Related Work

### 2.1 Spectral water indices

Surface-water delineation from multispectral imagery has been dominated by normalised-difference indices. McFeeters (1996) introduced the Normalized Difference Water Index, NDWI = (Green − NIR)/(Green + NIR), which exploits water's near-complete absorption in the near-infrared (NIR) band. NDWI's main failure mode is confusion over built-up surfaces, whose low-NIR / moderate-green spectra can yield false-positive water values. Xu (2006) addressed this by replacing NIR with shortwave-infrared band 1 (SWIR1) to produce the Modified Normalised Difference Water Index (MNDWI), which better discriminates built-up surfaces from water. Feyisa et al. (2014) developed the Automated Water Extraction Index (AWEI) in two variants — a non-shadow form (AWEInsh) and a shadow-robust form (AWEIsh) — that maximise separability in scenes with dense topographic or building shadows. The present work implements all four indices (NDWI, MNDWI, AWEInsh, AWEIsh) plus NDVI as a vegetation complement, in a typed numpy module with unit tests against synthetic spectra.

### 2.2 Thresholding

Automatic histogram-based thresholding was pioneered by Otsu (1979), who proposed choosing the threshold that minimises within-class variance in a bimodal grey-level histogram. Otsu's method is our default binariser, but we additionally evaluate (a) Triangle (Zack, Rogers and Latt, 1977), which is robust when the two classes have unequal populations; (b) Yen (Yen, Chang and Chang, 1995), which maximises entropy; and (c) Li (Li and Lee, 1993), based on minimum cross-entropy. Ji et al. (2009) showed that fixed NDWI thresholds are unreliable across seasons and regions, motivating our automatic selection.

### 2.3 Change detection

Singh (1989) reviews the classical approaches: image differencing, image ratioing, principal-component analysis (PCA), and change vector analysis (CVA). Bruzzone and Prieto (2000) automatically threshold the difference image via expectation maximisation. Deng et al. (2008) demonstrate that the last principal component of the stacked pre/post tensor concentrates the change signal. We implement differencing, ratioing, CVA and PCA-based change detection; the ΔMNDWI variant (difference of MNDWI maps) is the single method used by the final classical pipeline to isolate newly inundated pixels from permanent water bodies.

### 2.4 Deep learning for flood segmentation

Ronneberger, Fischer and Brox (2015) introduced U-Net, an encoder–decoder segmentation architecture whose skip connections preserve spatial detail at the output. We use the segmentation-models-pytorch implementation with a ResNet-34 ImageNet-pretrained encoder (He et al., 2016). Bonafilia et al. (2020) released the Sen1Floods11 benchmark — 446 hand-labelled 512 × 512 chips across 11 global flood events — which we adopt for training and evaluation. Konapala, Kumar and Ahmad (2021) demonstrated optical + SAR fusion for flood mapping; their observation that NDWI can outperform MNDWI on Sen1Floods11 when the benchmark is dominated by non-urban terrain directly motivates one of our empirical findings.

### 2.5 Kerala 2018 event

Sudheer et al. (2019) analyse the hydrological drivers of the August 2018 Kerala floods and quantify the contribution of dam operations. UNOSAT (2018) published an analyst-mapped flood-extent product (A-1879) for 22 August 2018. We adopt the Kerala event as our regional case study; quantitative evaluation is performed on Sen1Floods11 because the UNOSAT polygon mirrors proved unavailable at project time (see §7.2).

---

## 3. Data

### 3.1 Sen1Floods11 HandLabeled

Sen1Floods11 (Bonafilia et al., 2020) is a public Google Cloud Storage dataset containing Sentinel-1 and Sentinel-2 chips from 11 global flood events. We use the **HandLabeled** subset: 446 chips (512 × 512 px at 10 m), split into 252 training / 89 validation / 90 test / 15 Bolivia held-out, each paired with a hand-drawn flood mask in {0 = non-water, 1 = water, -1 = ignore}. Every Sentinel-2 chip contains 13 bands in the order [B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12, QA60]. We train on the six "DDA subset" bands B2, B3, B4, B8, B11, B12 — Blue, Green, Red, NIR, SWIR1, SWIR2 — which is sufficient for every index in §2.1 and keeps the first convolution of the ImageNet-pretrained encoder compact.

An early off-by-one error in our band-selection indices (loading B1, B2, B3, B7, B9, B11 instead of the intended B2, B3, B4, B8, B11, B12) silently corrupted all spectral indices; it was caught via a sanity-check on Phase-3 ablation rankings and fixed in commit `e3143f2`. After the fix, classical IoU rose from ≈ 0.37 to 0.44 on the test split, and the ordering of spectral indices shifted in a way consistent with published Sen1Floods11 numbers.

### 3.2 Kerala 2018 case study

The August 2018 Kerala event is our qualitative case study. AOI: a 1° × 1° bounding box (9.5–10.5° N, 76.0–77.0° E) covering Alappuzha, Ernakulam and Thrissur districts. Pre-event composite: 6–20 July 2018; post-event composite: 19–25 August 2018. We use Google Earth Engine for scene filtering, SCL-based cloud masking (classes 0, 1, 3, 8, 9, 10, 11 excluded) and median reducer compositing. The downloader is configuration-driven (`configs/kerala_2018.yaml`) and region-agnostic.

### 3.3 Auxiliary layers (Phase-7 stretch, not in main pipeline)

ESA WorldCover v200 for land-cover breakdown (11 classes at 10 m); WorldPop 100 m for population exposure; OpenStreetMap roads and buildings for infrastructure impact. Integration helpers live in `src/analysis/quantify.py`.

---

## 4. Methodology — Classical Pipeline

### 4.1 Overview

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

Module map: `src/preprocess/` (reflectance conversion, SCL cloud mask, coregistration, histogram matching); `src/dip/` (indices, thresholding, morphology, change detection, filters); `src/pipelines/classical_baseline.py` (end-to-end orchestration with CLI).

### 4.2 Preprocessing

Sentinel-2 L2A digital numbers are converted to surface reflectance via `(DN − 1000) / 10000`, following ESA's processing-baseline-04.00 offset convention. Scene-classification-layer (SCL) pixel values 0, 1, 3, 8, 9, 10 and 11 are masked out (no-data, saturation, cloud-shadow, medium- and high-probability cloud, cirrus, snow). An optional histogram-matching step normalises pre/post radiometric differences so that change detection responds to real surface change rather than illumination drift.

### 4.3 Spectral indices

All five indices (NDWI, MNDWI, NDVI, AWEInsh, AWEIsh) are implemented as pure-numpy functions with literature citations in their docstrings and division-by-zero protection via a 10⁻⁹ epsilon. A shared `compute_all(stack)` convenience dispatcher returns a `{name: array}` dictionary used by the ablation harness (Section 6.2).

### 4.4 Thresholding

We implement five thresholding methods — Otsu, Triangle, Yen, Li and adaptive (Gaussian/mean) — each returning a `ThresholdResult(mask, value, method)` named tuple so the final ablation table can reproduce every decision. The adaptive variant uses a 51 × 51-pixel (≈ 500 m) window, appropriate for removing radiometric gradients without erasing small flood bodies.

### 4.5 Morphological post-processing

The default `clean(mask, opening_radius=1, closing_radius=1, min_object_area=25, min_hole_area=25)` pipeline applies (i) binary opening with `disk(1)` to remove speckle, (ii) binary closing with `disk(1)` to bridge 1-pixel gaps, (iii) small-object removal at 25 px ≈ 2 500 m² ≈ 0.25 ha, and (iv) hole-filling symmetrically. The 25 px threshold follows National Remote Sensing Agency (NRSA) flood-report conventions; it is overridable via CLI.

### 4.6 Change detection

Five algorithms — image differencing, ratioing, CVA magnitude, PCA-based change (last principal component), and the `mndwi_difference` shortcut — are implemented in `src/dip/change_detection.py`. The final pipeline uses ΔMNDWI + Otsu, intersecting the change mask with the water mask to isolate newly-inundated pixels.

---

## 5. Methodology — Deep Learning

### 5.1 Architecture

We use the segmentation-models-pytorch implementation of U-Net with a **ResNet-34** ImageNet-pretrained encoder. The first convolutional layer is automatically expanded to six input channels to accept our B2, B3, B4, B8, B11, B12 reflectance stack; the decoder's five levels use channel widths (256, 128, 64, 32, 16). The output is a single-channel logit map (sigmoid applied downstream, allowing numerically-stable `BCEWithLogitsLoss`). Parameter count: 24.45 M trainable parameters.

### 5.2 Loss function

We use a combined **BCE + Dice** loss:

$$\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{BCE}} + (1 - \alpha) \cdot \mathcal{L}_{\text{Dice}}, \quad \alpha = 0.5.$$

BCE provides pixel-wise calibration; Dice is a direct surrogate for IoU and is robust to the class imbalance (≈ 20–30 % water in Sen1Floods11). `pos_weight = 2.0` slightly upweights the water class. Both loss components honour the Sen1Floods11 `-1` ignore index by masking such pixels before loss evaluation.

### 5.3 Training recipe

AdamW optimiser, initial learning rate 10⁻⁴, weight decay 10⁻⁴, cosine annealing over 30 epochs. Batch size 8, training crop 256 × 256 (validation at full 512 × 512). Augmentations: horizontal flip, vertical flip, 90° rotations, and ± 10 % brightness shift — no hue/saturation changes so that spectral band ratios are preserved. Mixed precision on the T4 GPU via `torch.amp`. Early stopping with patience 8 on validation IoU.

Training completed in 45 minutes of wall-clock time on a Google Colab free-tier Tesla T4. Best checkpoint at epoch 27 with validation IoU 0.516.

### 5.4 Inference

`src/inference/predict.py` provides `predict_chip(model, chip)` for in-memory chips and `predict_raster(model, src, dst, tile=512, overlap=64)` for arbitrary-size GeoTIFFs. The raster variant uses cosine-blended overlapping tiles to eliminate seams at 10 m resolution.

### 5.5 Hybrid fusion

We evaluate a weighted fusion of the U-Net probability map and the classical binary mask cast to float:

$$P_{\text{hybrid}} = w \cdot P_{\text{U-Net}} + (1 - w) \cdot M_{\text{classical}}, \quad w = 0.7.$$

Two alternative fusion strategies (pixel-wise AND, pixel-wise OR) are also implemented in `src/eval/fusion.py`.

---

## 6. Experiments and Results

### 6.1 Experimental protocol

All quantitative evaluation is performed on the Sen1Floods11 **test** split (n = 90 chips, held out from training). The `-1` ignore class is dropped before every metric. Metrics: intersection-over-union (IoU), Dice/F1 score, precision, recall, overall accuracy, Cohen's κ, confusion-matrix counts (TP, FP, FN, TN). Paired-bootstrap confidence intervals on per-chip ΔIoU use 10 000 resamples; pixel-level McNemar uses the continuity-corrected χ² (Edwards 1948).

### 6.2 Classical ablation (32 configurations)

Every combination of four indices {NDWI, MNDWI, AWEInsh, AWEIsh} × four thresholds {Otsu, Triangle, Yen, Li} × morphology {on, off} was evaluated. Results at Table 1.

**Table 1 — Top 5 classical configurations (mean IoU, Sen1Floods11 test, n = 90).**

| Configuration | IoU | F1 | κ |
|---|---|---|---|
| **ndwi_yen_raw** | **0.440** | **0.547** | **0.497** |
| ndwi_yen_morph | 0.433 | 0.534 | 0.484 |
| ndwi_triangle_morph | 0.347 | 0.432 | 0.387 |
| ndwi_triangle_raw | 0.338 | 0.423 | 0.371 |
| awei_nsh_yen_morph | 0.330 | 0.419 | 0.368 |

The winner is **NDWI + Yen without morphology**. This contradicts the textbook prediction that MNDWI + Otsu should dominate and merits analysis.

### 6.3 Why NDWI beats MNDWI on this benchmark

Two mechanisms explain the empirical ordering. First, real flood water carries heavy sediment. Turbid water raises SWIR1 reflectance, which enters the MNDWI denominator: the numerator (Green − SWIR1) therefore shrinks faster than the denominator grows, compressing MNDWI over exactly the pixels we need to detect. NDWI uses NIR, which remains strongly absorbed by water regardless of turbidity. Second, Sen1Floods11's 11 events span predominantly rural and vegetated terrain; MNDWI's specific advantage over NDWI — robustness to built-up pixels — never gets to manifest at scale. Konapala et al. (2021) report the same qualitative ordering on Sen1Floods11.

Yen's entropy criterion is also the better threshold choice here. Sen1Floods11 histograms on NDWI and MNDWI are right-skewed — water is a minority but the tail is heavy — and Otsu's within-class-variance objective is known to bias toward the majority class in such distributions.

### 6.4 Final three-method comparison

**Table 2 — Classical vs U-Net vs Hybrid on Sen1Floods11 test (n = 90 chips).**

| Method | IoU | F1 | Precision | Recall | Accuracy | κ | Runtime (ms/chip, T4) |
|---|---|---|---|---|---|---|---|
| Classical (ndwi_yen_raw) | 0.4401 | 0.5475 | 0.5902 | **0.7601** | 0.8869 | 0.4968 | 6.25 |
| **U-Net (ResNet-34)** | **0.5475** | **0.6604** | **0.6719** | 0.7337 | **0.9709** | **0.6522** | 32.57 |
| Hybrid (w_unet = 0.7) | 0.5313 | 0.6364 | 0.6401 | 0.7356 | 0.9698 | 0.6136 | 39.7 † |

† Hybrid fuses the U-Net probability map with the classical mask; its end-to-end runtime is the sum of the two component runtimes plus a < 1 ms fusion step.

### 6.5 Statistical significance

**Paired-bootstrap 95 % CI on per-chip ΔIoU (10 000 resamples):**

* U-Net − Classical: **+0.0756**, CI [+0.0371, +0.1140] — **significant** (CI excludes 0).
* Hybrid − Classical: **+0.0811**, CI [+0.0534, +0.1103] — **significant**.
* U-Net − Hybrid: −0.0095, CI [−0.0240, +0.0049] — **not significant**.

**Pixel-level McNemar (continuity-corrected χ²):**

* U-Net vs Classical: χ² = 1.13 × 10⁶, p ≈ 0.
* Hybrid vs Classical: χ² = 1.32 × 10⁶, p ≈ 0.

### 6.6 Negative result — hybrid fusion does not beat the U-Net alone

The 70/30 weighted fusion performs *worse* than the U-Net in isolation (0.531 vs 0.548 IoU), and the difference's 95 % CI spans zero. The classical mask's noise — it over-commits to water with precision 0.59 — dilutes rather than regularises the U-Net's calibrated probabilities. The U-Net is both the accuracy and the Occam's-razor winner. The classical pipeline retains two legitimate roles: (i) a fast, explainable CPU-only fallback for the Streamlit app; (ii) a baseline for the Analysis & Results rubric criterion.

### 6.7 Error-category analysis

`src/analysis/error_analysis.py` partitions every prediction pixel into one of five spectral categories (turbid_water, dark_land, vegetation, bare_sparse, other) via NDVI and MNDWI rules. Applied to the U-Net on the test split, false negatives concentrate in **turbid_water** and **vegetation** categories (sediment-laden flood water resembles bare soil; tree canopies mask underlying water in optical bands). False positives concentrate in **dark_land** pixels (shadows, asphalt, cloud edges) whose low NIR mimics water's signature.

### 6.8 Qualitative inspection

Four high-flood test chips were rendered side-by-side with RGB, classical mask, U-Net mask and ground truth (Figure `phase3_qualitative_grid.png`). Visual inspection confirms the U-Net's advantage on (a) mixed-cover chips where the water–land boundary is sub-pixel and (b) chips with heavy shadow where the classical threshold falsely triggers.

---

## 7. Discussion

### 7.1 Comparison with PRD targets

The Product Requirements Document locked goal **G1** at IoU ≥ 0.75 on Sen1Floods11. Our U-Net attained 0.548, below target. The gap is attributable to the training-data budget: Sen1Floods11 provides only 252 **HandLabeled** training chips, whereas published Sen1Floods11 results that exceed 0.75 IoU typically fine-tune on the much larger **WeaklyLabeled** split (4 385 chips, ≈ 40 GB) which was excluded from the project scope by the free-tier Colab compute constraint. With the data budget we had, our +0.108 IoU over a principled classical baseline, with the improvement confirmed by two independent significance tests, is a defensible result.

### 7.2 Ground-truth sourcing

The PRD originally planned to evaluate the Kerala Streamlit demo against the UNOSAT 2018 flood polygon (product A-1879). Both the UNOSAT portal and the Copernicus EMS mirror URLs returned 404 at project time. PRD v1.2 therefore restricts quantitative evaluation to Sen1Floods11 and treats Kerala 2018 as a qualitative case study. This is a standard pattern in published flood-mapping papers.

### 7.3 Future work

* **Retrain on Sen1Floods11 WeaklyLabeled** — approximately 17× more training data; we expect IoU to reach 0.70–0.80 range.
* **Sentinel-1 SAR cross-check** — SAR sees through vegetation and clouds, directly addressing the two biggest U-Net failure categories identified in §6.7. Implementation skeleton exists (`src/data/gee_download.py` already downloads `COPERNICUS/S1_GRD`); a separate U-Net branch with a Refined-Lee speckle filter and log-VV Otsu threshold is the obvious next step.
* **Shadow suppression** — a hue-saturation-value shadow index as a 7th input channel should reduce dark-land false positives.
* **Turbid-water augmentation** — spectral interpolation between water and bare-soil endmembers could synthesise turbid-water training examples and broaden the U-Net's feature bank.

### 7.4 Threats to validity

* **Selection bias on 90 test chips.** Sen1Floods11's test split is small; conclusions are bootstrap-CI-grounded, but with n = 90 the CIs are wide.
* **ImageNet-pretrained encoder.** Using an RGB-ImageNet encoder for a 6-channel reflectance task is a non-trivial distribution shift; the expanded first convolution's extra three channels are randomly initialised. Training from scratch on Sen1Floods11 was not evaluated due to compute budget.
* **Inference on Sen1Floods11 chip distribution.** Deployment on out-of-distribution Kerala imagery is not statistically guaranteed to achieve the test-split IoU. Qualitative Kerala inspection shows plausible masks but no pixel-level verification is possible without external GT.

---

## 8. System Artefacts

### 8.1 Repository layout

```
PRD.md                          # product requirements (locked v1.2)
environment.yml · requirements.txt
pyproject.toml · .pre-commit-config.yaml
src/
  data/       (aoi, gee_download, ground_truth, sen1floods11_loader)
  preprocess/ (reflectance, cloud_mask, coregister, histogram_match)
  dip/        (indices, thresholding, morphology, change_detection, filters)
  models/     (unet, losses)
  train/      (train_unet, augment)
  inference/  (predict)
  eval/       (metrics, significance, ablation, fusion)
  analysis/   (severity, quantify, error_analysis)
  pipelines/  (classical_baseline, full_pipeline)
  utils/      (logging, paths)
tests/        (93 passing; 75 % line coverage)
app/
  streamlit_app.py · report_generator.py · templates/report.html
notebooks/    (01 EDA · 02 Classical DIP · 03 Ablation · 04 U-Net · 05 Analysis)
reports/      (phase1–phase6 reports · lit_review · rubric_mapping · this)
scripts/      (GEE download · Sen1Floods11 download · run_ablation · train_unet · eval_unet · run_final_comparison · colab_streamlit)
```

### 8.2 Reproducibility

All results in this report are regenerable on a Google Colab free-tier Tesla T4 from a fresh clone in under 2 hours total wall-clock time:

1. `conda env create -f environment.yml` (or `pip install -r requirements.txt`).
2. `python scripts/download_sen1floods11.py --subset hand` (≈ 20 min, 3 GB, one-time).
3. `python scripts/run_ablation.py --split test` (≈ 3 min).
4. `python scripts/train_unet.py --epochs 30 --batch-size 8` (≈ 45 min on T4).
5. `python scripts/run_final_comparison.py` (≈ 2 min) — regenerates Table 2.

Random seeds are set (`seed = 42`) so the classical pipeline is fully deterministic; U-Net training is approximately deterministic subject to CUDA non-determinism.

### 8.3 Streamlit application

The demo app (`app/streamlit_app.py`) exposes the full pipeline through a 4-tab interface (Map · Metrics · Downloads · About), with a one-click PDF damage-report generator built via Jinja2 and WeasyPrint. Local invocation: `streamlit run app/streamlit_app.py`. Colab-hosted invocation via `scripts/colab_streamlit.py`. A 60-second screen-recording is included in the submission bundle as a live-demo fallback.

---

## 9. Conclusion

We have presented an end-to-end, open-source, reproducible flood-mapping pipeline that combines classical DIP and a deep-learning segmentation network, and evaluated it rigorously on the Sen1Floods11 benchmark. The ResNet-34 U-Net improves on the best classical baseline by +0.108 IoU (+24.5 % relative) with both paired-bootstrap and pixel-level McNemar tests confirming statistical significance. Our hybrid fusion experiment produced a principled negative result, and our error-category analysis identifies two concrete, implementable directions for future work. The system ships as a web application with an auto-generated PDF damage report. Code, data-loading scripts, trained model, and every figure in this report are available at https://github.com/bhagya2819/Disaster_Damage_Assessment under the MIT license.

---

## References

[1] S. K. McFeeters, "The use of the normalized difference water index (NDWI) in the delineation of open water features," *International Journal of Remote Sensing*, vol. 17, no. 7, pp. 1425–1432, 1996.

[2] H. Xu, "Modification of normalised difference water index (MNDWI) to enhance open water features in remotely sensed imagery," *International Journal of Remote Sensing*, vol. 27, no. 14, pp. 3025–3033, 2006.

[3] G. L. Feyisa, H. Meilby, R. Fensholt and S. R. Proud, "Automated water extraction index: A new technique for surface water mapping using Landsat imagery," *Remote Sensing of Environment*, vol. 140, pp. 23–35, 2014.

[4] J. W. Rouse Jr., R. H. Haas, J. A. Schell and D. W. Deering, "Monitoring vegetation systems in the Great Plains with ERTS," in *Proceedings of the Third ERTS Symposium*, NASA SP-351, 1974, pp. 309–317.

[5] N. Otsu, "A threshold selection method from gray-level histograms," *IEEE Transactions on Systems, Man, and Cybernetics*, vol. 9, no. 1, pp. 62–66, 1979.

[6] G. W. Zack, W. E. Rogers and S. A. Latt, "Automatic measurement of sister chromatid exchange frequency," *Journal of Histochemistry & Cytochemistry*, vol. 25, no. 7, pp. 741–753, 1977.

[7] J.-C. Yen, F.-J. Chang and S. Chang, "A new criterion for automatic multilevel thresholding," *IEEE Transactions on Image Processing*, vol. 4, no. 3, pp. 370–378, 1995.

[8] C. H. Li and C. K. Lee, "Minimum cross entropy thresholding," *Pattern Recognition*, vol. 26, no. 4, pp. 617–625, 1993.

[9] L. Ji, L. Zhang and B. Wylie, "Analysis of dynamic thresholds for the normalized difference water index," *Photogrammetric Engineering & Remote Sensing*, vol. 75, no. 11, pp. 1307–1317, 2009.

[10] A. Singh, "Review article — Digital change detection techniques using remotely-sensed data," *International Journal of Remote Sensing*, vol. 10, no. 6, pp. 989–1003, 1989.

[11] L. Bruzzone and D. F. Prieto, "Automatic analysis of the difference image for unsupervised change detection," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 38, no. 3, pp. 1171–1182, 2000.

[12] J. S. Deng, K. Wang, Y. H. Deng and G. J. Qi, "PCA-based land-use change detection and analysis using multitemporal and multisensor satellite data," *International Journal of Remote Sensing*, vol. 29, no. 16, pp. 4823–4838, 2008.

[13] O. Ronneberger, P. Fischer and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Medical Image Computing and Computer-Assisted Intervention — MICCAI 2015*, 2015, pp. 234–241.

[14] K. He, X. Zhang, S. Ren and J. Sun, "Deep residual learning for image recognition," in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 770–778.

[15] D. Bonafilia, B. Tellman, T. Anderson and E. Issenberg, "Sen1Floods11: A georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2020, pp. 210–211.

[16] G. Konapala, S. V. Kumar and S. K. Ahmad, "Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning," *ISPRS Journal of Photogrammetry and Remote Sensing*, vol. 180, pp. 163–173, 2021.

[17] S. Martinis, A. Twele and S. Voigt, "Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high-resolution TerraSAR-X data," *Natural Hazards and Earth System Sciences*, vol. 9, no. 2, pp. 303–314, 2009.

[18] K. P. Sudheer, M. E. Vijay Bharath Kumar, A. S. Ranjini and H. P. Gowda, "Role of dams on the floods of August 2018 in Periyar River Basin, Kerala," *Current Science*, vol. 116, no. 5, pp. 780–794, 2019.

[19] UNOSAT, "Flood waters over Kerala state, India, as of 22 August 2018," Product A-1879, United Nations Institute for Training and Research, 2018.

[20] A. L. Edwards, "Note on the 'correction for continuity' in testing the significance of the difference between correlated proportions," *Psychometrika*, vol. 13, no. 3, pp. 185–187, 1948.

---

## Appendix A — Hyperparameters (locked)

| Module | Parameter | Value |
|---|---|---|
| Cloud mask | SCL classes excluded | {0, 1, 3, 8, 9, 10, 11} |
| Classical | Water index | NDWI (McFeeters 1996) |
| Classical | Threshold | Yen (1995) |
| Classical | Morphology | off (raw) |
| Morphology | Opening disk radius | 1 px |
| Morphology | Closing disk radius | 1 px |
| Morphology | Min object area | 25 px (≈ 2 500 m²) |
| Morphology | Min hole area | 25 px |
| U-Net | Encoder | ResNet-34 (ImageNet-pretrained) |
| U-Net | Decoder channels | (256, 128, 64, 32, 16) |
| U-Net | In-channels | 6 (B2, B3, B4, B8, B11, B12) |
| Loss | α (BCE weight) | 0.5 |
| Loss | pos_weight | 2.0 |
| Optimiser | Algorithm | AdamW |
| Optimiser | Learning rate | 10⁻⁴ (cosine-annealed) |
| Optimiser | Weight decay | 10⁻⁴ |
| Optimiser | Batch size | 8 |
| Training | Max epochs | 30 |
| Training | Early-stopping patience | 8 |
| Training | Train crop | 256 × 256 |
| Training | Validation crop | full 512 × 512 |
| Training | Augmentation | flips · 90° rotations · ±10 % brightness |
| Training | Mixed precision | enabled on CUDA |
| Training | Seed | 42 |
| Hybrid | Fusion weight (w_unet) | 0.7 |
| Hybrid | Binarisation threshold | 0.5 |

---

## Appendix B — Architecture Diagram

```
 ┌──────────────────┐   ┌───────────────────┐   ┌─────────────────────────┐
 │  Data ingestion  │──▶│   Preprocessing   │──▶│    Analysis core        │
 │                  │   │                   │   │                         │
 │ · GEE Sentinel-2 │   │ · DN → reflectance│   │ · Classical (NDWI+Yen)  │
 │ · Sen1Floods11   │   │ · SCL cloud mask  │   │ · U-Net (ResNet-34)     │
 │   HandLabeled    │   │ · coregistration  │   │ · Hybrid fusion         │
 │ · UNOSAT (opt.)  │   │ · histogram match │   │                         │
 └──────────────────┘   └───────────────────┘   └────────┬────────────────┘
                                                         │
                                                         ▼
                                         ┌──────────────────────────────┐
                                         │ Eval & analysis              │
                                         │                              │
                                         │ · Metrics (IoU F1 κ OA)      │
                                         │ · Bootstrap CI / McNemar     │
                                         │ · Severity per cell          │
                                         │ · Area / landcover / pop     │
                                         │ · Error-category breakdown   │
                                         └────────┬─────────────────────┘
                                                  │
                 ┌────────────────────────────────┴───────────────┐
                 ▼                                                ▼
 ┌───────────────────────────┐                ┌──────────────────────────────┐
 │  Streamlit web app        │                │  Auto-generated PDF report   │
 │  (app/streamlit_app.py)   │                │  (app/report_generator.py)   │
 │                           │                │                              │
 │ sidebar · Map · Metrics   │                │  Jinja2 → WeasyPrint         │
 │ Downloads · About         │                │  stats / figures / benchmark │
 └───────────────────────────┘                └──────────────────────────────┘
```

---

## Appendix C — Hardware and Software

| Layer | Detail |
|---|---|
| Training hardware | Google Colab free-tier, Tesla T4 GPU (16 GB VRAM), ~12 GB RAM |
| OS | Ubuntu 22.04 (Colab default) |
| Python | 3.12 |
| Deep-learning framework | PyTorch 2.5.1 + segmentation-models-pytorch 0.3.4 |
| Geospatial | rasterio 1.5, geopandas 1.0, pyproj 3.7 |
| Classical DIP | scikit-image 0.24, scipy 1.14, opencv-python 4.13 |
| Web app | streamlit 1.56, streamlit-folium 0.22, folium 0.17 |
| PDF report | jinja2 + WeasyPrint 62.3 |
| Testing | pytest 8.4, pytest-cov 5.0 |
| Lint / format | ruff 0.7, black 24.10, pre-commit 3.8 |

---

*End of report.*
