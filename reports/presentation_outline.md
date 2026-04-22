# Presentation Outline — 20-minute talk (15 min + 5 min Q&A)

> **Deck target**: 18 slides · Google Slides or PowerPoint.
> **Speakers (3–4 members)**: assign per PRD §9 — A (Tech/App), B (Data/Geo), C (DIP/ML), D (PM/Report).
> **Dry-runs**: at least two full timed run-throughs before the evaluation.

---

## Timing budget

| Section | Slides | Target |
|---|---|---|
| Title + team | 1 | 0:30 |
| Problem & motivation | 2–3 | 2:00 |
| Related work | 4 | 1:30 |
| Data & study area | 5 | 1:30 |
| Classical DIP pipeline | 6–8 | 3:00 |
| U-Net pipeline | 9–10 | 2:00 |
| Hybrid fusion | 11 | 1:00 |
| Results & ablation | 12–13 | 2:30 |
| **Live demo / screen-recording** | 14 | 3:00 |
| Error analysis & limitations | 15–16 | 1:30 |
| Conclusion & future work | 17 | 1:00 |
| Q&A | 18 (title) | 5:00 |
| **Total** | 18 | **20:00** |

---

## Slide 1 · Title + team

> Speaker: D (PM) · 30 s

Content:
- **Title**: "Disaster Damage Assessment — Flood Mapping from Sentinel-2 using Hybrid Classical DIP + U-Net"
- Subtitle: "Course Project · Digital Image Processing · Remote Sensing"
- Team member names and roles (A, B, C, D)
- GitHub: `bhagya2819/Disaster_Damage_Assessment`
- Single hero image: side-by-side pre/post Kerala RGB composite with overlaid flood mask.

Speaker notes:
> "We're presenting an end-to-end flood-mapping pipeline that combines classical digital image processing with a U-Net segmentation model. Our team of four split the work as …"

---

## Slide 2 · Problem

> Speaker: D · 60 s

Content:
- Kerala 2018: 483 dead, 1.4 M displaced, ₹40 000 crore loss.
- Ground surveys are too slow during active flooding — satellite + DIP is the only scalable alternative.
- Formal input/output: pre/post Sentinel-2 → binary flood mask + severity + damage report.

Key visual: photo of Kerala 2018 flood.

---

## Slide 3 · Why this is a DIP problem

> Speaker: D · 60 s

Content:
- Point operations: reflectance conversion, spectral-index arithmetic.
- Neighborhood operations: thresholding, morphology.
- Frequency domain: FFT sanity check.
- Segmentation: U-Net encoder–decoder CNN.
- Ground truth: Sen1Floods11 (hand-annotated flood masks, 11 global events).

Visual: annotated pipeline diagram — each stage labelled with its DIP category.

---

## Slide 4 · Related work

> Speaker: D · 90 s

Content (bullet list with IEEE refs):
- **Indices** — NDWI (McFeeters 1996), MNDWI (Xu 2006), AWEI (Feyisa 2014).
- **Thresholding** — Otsu (1979), Triangle (Zack 1977), Yen (1995), Li (1993).
- **Change detection** — Singh (1989), Deng (2008).
- **Deep learning** — U-Net (Ronneberger 2015), ResNet (He 2016), Sen1Floods11 (Bonafilia 2020), SAR fusion (Konapala 2021).
- **Kerala-specific** — Sudheer (2019), UNOSAT A-1879 (2018).

---

## Slide 5 · Data & study area

> Speaker: B · 90 s

Content:
- **Sen1Floods11** HandLabeled: 446 chips (512 × 512 @ 10 m), 11 global events, 252 / 89 / 90 / 15 splits.
- **DDA band subset**: B2, B3, B4, B8, B11, B12 (Blue, Green, Red, NIR, SWIR1, SWIR2).
- **Kerala 2018** AOI: 9.5–10.5° N, 76.0–77.0° E; pre 6–20 Jul, post 19–25 Aug 2018.
- Data flow: Google Earth Engine → SCL cloud mask → median composite → GeoTIFF.

Visuals: world map with 11 flood events dotted, zoom-in to Kerala AOI.

---

## Slide 6 · Classical DIP pipeline — overview

> Speaker: C · 60 s

Content: the pipeline diagram from §4.1 of the report.

Visual:
```
pre/post  →  MNDWI  →  Otsu       →  water mask
                  └─── ΔMNDWI → Otsu →  change mask
                                            ↓
                                    water ∩ change
                                            ↓
                                    morphology.clean
                                            ↓
                                   flood mask GeoTIFF
```

---

## Slide 7 · Classical DIP — spectral indices

> Speaker: C · 60 s

Content:
- NDWI = (Green − NIR) / (Green + NIR) — McFeeters 1996.
- MNDWI = (Green − SWIR1) / (Green + SWIR1) — Xu 2006 (urban-robust).
- NDVI, AWEInsh, AWEIsh.
- All five implemented in `src/dip/indices.py`, 100 % test coverage.

Visual: the **phase2_indices_post.png** 2×3 grid figure.

---

## Slide 8 · Classical DIP — thresholding + morphology

> Speaker: C · 60 s

Content:
- Five thresholds implemented (Otsu, Triangle, Yen, Li, adaptive Gaussian).
- Morphology `clean()`: opening (remove speckle) → closing (bridge gaps) → small-object removal (25 px ≈ 0.25 ha) → hole filling.
- `ThresholdResult(mask, value, method)` named tuple — every decision is logged for the ablation.

Visual: **phase2_thresholding.png** 2×3 panel comparing all 5 threshold methods + the histogram with overlaid threshold lines.

---

## Slide 9 · U-Net architecture

> Speaker: C · 60 s

Content:
- Encoder: ResNet-34, ImageNet-pretrained.
- First conv layer expanded from 3 → 6 input channels (SMP default).
- Decoder: 5 levels with channels (256, 128, 64, 32, 16).
- 24.45 M trainable parameters.
- Sigmoid applied downstream → numerically-stable `BCEWithLogitsLoss`.

Visual: canonical U-Net architecture diagram with ResNet-34 blocks labelled.

---

## Slide 10 · U-Net — training recipe

> Speaker: C · 60 s

Content:
- Loss: **0.5 · BCE + 0.5 · Dice**, `pos_weight=2.0`, honours `-1` ignore index.
- Optimiser: AdamW, lr 10⁻⁴ cosine-annealed, weight decay 10⁻⁴.
- Batch 8, train crop 256 × 256, val full 512 × 512.
- Augmentations: flips + 90° rotations + ± 10 % brightness (no hue/sat — spectra must be preserved).
- Mixed precision on T4 GPU. 30 epochs, early-stopping patience 8.
- Wall-clock training time: **~45 min** on Colab free-tier T4.

Visual: training curves from `reports/figures/phase4_training_curves.png`.

---

## Slide 11 · Hybrid fusion

> Speaker: C · 60 s

Content:
- Formula: `P_hybrid = 0.7 · P_U-Net + 0.3 · M_classical`, threshold at 0.5.
- Two alternative fusion strategies also implemented (AND, OR).
- Intuition: classical as a weak prior regularising the CNN.
- Spoiler: didn't work — see next-to-next slide.

Visual: simple block diagram of the weighted fusion.

---

## Slide 12 · Results — three-method comparison

> Speaker: D · 90 s

Content: the Phase-5 comparison table.

| Method | IoU | F1 | κ | Accuracy | Precision | Recall | Runtime |
|---|---|---|---|---|---|---|---|
| Classical | 0.4401 | 0.5475 | 0.4968 | 0.8869 | 0.5902 | **0.7601** | 6.25 ms |
| **U-Net** | **0.5475** | **0.6604** | **0.6522** | **0.9709** | **0.6719** | 0.7337 | 32.6 ms |
| Hybrid | 0.5313 | 0.6364 | 0.6136 | 0.9698 | 0.6401 | 0.7356 | 39.7 ms |

- U-Net beats classical by **+24.5 % IoU** relative.
- Accuracy jump 0.887 → 0.971 (+8.4 pp).
- Classical has best recall — it over-predicts water.

Visual: **phase5_metrics_bars.png**.

---

## Slide 13 · Statistical significance

> Speaker: D · 60 s

Content:
- **Paired bootstrap** (10 000 resamples) on per-chip ΔIoU:
  - U-Net − Classical: **+0.076**, 95 % CI [+0.037, +0.114] — excludes 0 ✓
  - Hybrid − Classical: +0.081, CI [+0.053, +0.110] — excludes 0 ✓
  - U-Net − Hybrid: −0.010, CI [−0.024, +0.005] — **spans 0** ✗
- **McNemar** (pixel-level): both U-Net and Hybrid vs classical, p ≈ 0.
- **Negative result**: hybrid fusion does not improve over U-Net alone. We keep classical only as a CPU fallback.

Visual: forest plot of the three bootstrap CIs.

---

## Slide 14 · LIVE DEMO

> Speaker: A (app lead) · 3:00

Content:
- Open the Streamlit app (or show the **60-second screen-recording** if the live tunnel is unavailable).
- Pick a Sen1Floods11 chip → click Run → show: RGB, U-Net mask, GT side-by-side; metrics; then generate PDF.
- Close by showing the downloaded PDF report.

Fallback: MP4 screen-recording embedded in slide.

---

## Slide 15 · Error-category breakdown

> Speaker: C · 60 s

Content:
- False negatives concentrate in **turbid_water** and **vegetation** — the U-Net under-detects sediment-laden water and tree-canopy-occluded water.
- False positives concentrate in **dark_land** — shadows, asphalt, cloud edges.
- Two concrete future-work directions derive directly from this.

Visual: **phase5_error_categories.png** bar chart.

---

## Slide 16 · Limitations & threats to validity

> Speaker: D · 30 s

Content (bullets):
- Only 252 HandLabeled training chips → IoU ceiling ≈ 0.55 on HandLabeled-only.
- n = 90 test set → bootstrap CIs are wide.
- ImageNet encoder × 6-channel reflectance = distribution shift.
- No pixel-level ground truth for Kerala 2018 (UNOSAT mirror dead) → case study is qualitative.

---

## Slide 17 · Conclusion & future work

> Speaker: D · 60 s

Content:
- **Contribution 1** — reproducible open-source pipeline with 93-test pytest suite, 75 % coverage.
- **Contribution 2** — empirical evidence that NDWI > MNDWI on Sen1Floods11 (counter-intuitive).
- **Contribution 3** — U-Net beats classical by +0.108 IoU with two independent significance tests.
- **Contribution 4** — honest negative result on hybrid fusion.
- **Contribution 5** — Streamlit demo + auto-generated PDF report.
- **Future work**: WeaklyLabeled training, SAR cross-check, shadow suppression, turbid-water augmentation.

---

## Slide 18 · Questions & discussion

> 5:00

Content:
- "Thank you!" + team names again.
- Repo URL prominently displayed: `github.com/bhagya2819/Disaster_Damage_Assessment`.
- "We're happy to take questions."

Have `reports/qna.md` open on the laptop during Q&A.

---

## Speaker-assignment suggestion (3–4 members)

| Slides | Speaker | Phase-owner |
|---|---|---|
| 1, 2, 3, 4 | D (PM/Report) | Problem + related work |
| 5 | B (Data/Geo) | Datasets |
| 6, 7, 8 | C (DIP/ML) | Classical pipeline |
| 9, 10, 11 | C (DIP/ML) | U-Net + fusion |
| 12, 13 | D (PM/Report) | Results + statistics |
| 14 | A (Tech/App) | Live demo |
| 15 | C (DIP/ML) | Error analysis |
| 16, 17 | D (PM/Report) | Limitations + conclusion |
| 18 (Q&A) | All | Split questions by domain |

If only 3 members, merge A into C (same person demos + explains the U-Net).

---

## Deck build notes

- Use a **clean, minimal template** (e.g., Google Slides "Modern Writer" or "Coral"). Avoid heavy graphics.
- Consistent colour scheme: classical = red, U-Net = blue, hybrid = purple. Apply to every bar chart.
- Font: Sans-serif (Arial / Roboto) at minimum 18 pt body, 24 pt headers.
- Every numeric claim on a slide must be traceable to `reports/phase*_report.md` or `results/final_comparison/summary.json`.
- Back up the deck as a PDF and commit to the repo (`reports/slides.pdf`) before the presentation.

---

## Dry-run checklist

- [ ] Deck drafted and reviewed by all members.
- [ ] Figures at 300 DPI, not pixelated when projected.
- [ ] 60-second screen-recording of the app embedded as a fallback.
- [ ] Two full timed rehearsals; every speaker stays within their slide budget.
- [ ] Q&A sheet reviewed by each speaker.
- [ ] Backup laptop + USB stick with deck PDF.
- [ ] Rehearse the 5-second "thank you" ending to avoid awkward silence.
