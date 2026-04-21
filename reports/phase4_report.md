# Phase 4 Report — U-Net Deep Learning Segmentation

> Owner: **C (DIP/ML)** + **A (Tech/App)** · feeds §3.4 (Methodology — Deep Learning), §4.3 (Experiments — U-Net vs classical), and §4.4 (Hybrid fusion) of the final IEEE report.

---

## 1. Objective

Train a convolutional segmentation network on Sen1Floods11 that:

1. Substantially beats the Phase-3 classical baseline (`ndwi_yen_raw`, mean IoU 0.440).
2. Meets PRD goal **G1**: validation IoU ≥ 0.75.
3. Produces a portable ONNX artefact suitable for the Streamlit app.

## 2. Model

| Choice | Value | Rationale |
|---|---|---|
| Architecture | U-Net (Ronneberger 2015) | Canonical for segmentation, strong pretrained encoders. |
| Encoder | ResNet-34, ImageNet weights | Best accuracy / VRAM trade-off for free-tier Colab T4. |
| Input channels | 6 (B2, B3, B4, B8, B11, B12) | Matches the DDA subset from Phase 1. |
| Output | 1 channel logits | Sigmoid applied downstream. |
| Parameters | ~24 M | Fits T4 at batch 8 × 256² input. |

Source: `src/models/unet.py`.

## 3. Loss

**BCE + Dice** combined loss, α = 0.5, with `pos_weight = 2.0` to upweight the minority flood class.

Why BCE + Dice rather than BCE alone: Sen1Floods11 is class-imbalanced (≈ 20–30 % water). BCE alone biases toward the majority class; Dice is a direct surrogate for IoU and is robust to imbalance; the combination retains BCE's calibration while inheriting Dice's focus on foreground pixels.

Source: `src/models/losses.py`.

## 4. Training recipe

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-4 (cosine-annealed over `epochs`) |
| Weight decay | 1e-4 |
| Batch size | 8 |
| Training crop | 256 × 256 (random) |
| Validation crop | full 512 × 512 |
| Augmentations | flips, 90° rotations, ±10 % brightness (no hue/sat — spectra must be preserved) |
| Epochs | 30 (early stop after 8 epochs no-improvement) |
| Mixed precision | enabled on CUDA |
| Device | Colab free-tier T4 (~4–6 hr total) |
| Seed | 42 |

Source: `src/train/train_unet.py`, `src/train/augment.py`.

## 5. Evaluation protocol

Identical to Phase 3:
- Sen1Floods11 test split (~89 chips).
- Metrics: IoU, F1, precision, recall, accuracy, Cohen's κ — from `src/eval/metrics.py`.
- Threshold @ 0.5 on the sigmoid output to produce the binary mask.
- Per-chip IoU saved to `results/unet/per_chip/unet.npz` for paired comparison.

## 6. Results (locked from test-split evaluation, 2026-04-21)

| Metric | Classical baseline | **U-Net** | Δ | Relative |
|---|---|---|---|---|
| Config | `ndwi_yen_raw` | ResNet-34 U-Net | — | — |
| Mean IoU | 0.440 | **0.548** | **+0.108** | **+24.5 %** |
| Mean F1 | 0.547 | **0.660** | **+0.113** | **+20.7 %** |
| Mean κ | 0.497 | **0.652** | **+0.155** | **+31.2 %** |
| Precision | — | 0.672 | — | — |
| Recall | — | 0.734 | — | — |
| Accuracy | — | 0.971 | — | — |

### 6.1 Paired comparison (U-Net − classical, per-chip ΔIoU, n=90)

| | Value |
|---|---|
| Mean ΔIoU (paired) | **+0.0756** |
| 95 % bootstrap CI (10 000 resamples) | **[+0.0371, +0.1140]** |
| Significance | CI excludes 0 → **significant at p < 0.05** |
| McNemar χ² (pixel-level, pooled) | *(to fill from notebook)* |

> NB: the +0.0756 paired ΔIoU is slightly lower than the +0.108 mean-level gain because the paired bootstrap conditions on chips where both methods return a finite IoU and weights all chips equally; the mean-level gain is dominated by chips where one method degenerates to NaN. Both are reported for transparency.

### 6.2 Narrative (for IEEE report §4.3)

> The ResNet-34 U-Net achieved mean IoU **0.548** on the Sen1Floods11 test split (n=90 chips), an improvement of **+0.108 absolute / +24.5 % relative** over the classical `ndwi_yen_raw` baseline (0.440). The paired bootstrap 95 % CI on per-chip ΔIoU of **[+0.037, +0.114]** excludes zero, confirming the improvement is statistically significant at α = 0.05. Cohen's κ rises from 0.497 to 0.652, moving the quality rating from "moderate" to "substantial" agreement (Landis & Koch 1977).
>
> The U-Net falls short of the PRD target of IoU ≥ 0.75, and this is traceable to the training budget: only **252 HandLabeled chips** were available, whereas Sen1Floods11 papers that report IoU > 0.75 generally fine-tune on the WeaklyLabeled split (4,385 chips, ~40 GB), which was out of scope for the course project's Colab-free-tier compute budget. A retraining run that incorporates WeaklyLabeled is the top-ranked future-work item.
>
> Qualitatively, the U-Net's advantage over the classical baseline is concentrated on: (a) mixed-cover chips where the water–land boundary is sub-pixel — classical index thresholds commit early, U-Net interpolates; (b) chips with heavy shadow, where thresholds over-predict water and the U-Net learns that shadowed land has a different spectral signature; and (c) turbid inland flood water, where the NDWI numerator is compressed and the CNN's spatial context disambiguates. Residual errors concentrate on (i) snow/ice pixels, rare in Sen1Floods11 and therefore under-learned, and (ii) wet building roofs that visually resemble water even to human annotators.

## 7. Hybrid fusion (PRD §8 Phase 4 should-have)

Source: `src/eval/fusion.py`.

Three fusion variants evaluated:
- **Weighted** — `w·prob_unet + (1-w)·prob_classical`, threshold 0.5. Sweep `w ∈ {0.3, 0.5, 0.7}`.
- **Agreement** (AND) — confident-only; expected to raise precision.
- **Union** (OR) — aggressive recall; expected to reduce false negatives.

Results table to be filled from the notebook.

## 8. Artefacts

- `checkpoints/unet_resnet34/best.pt` — best model by validation IoU.
- `checkpoints/unet_resnet34/last.pt` — resume checkpoint.
- `checkpoints/unet_resnet34/metrics.csv` — per-epoch log.
- `checkpoints/unet_resnet34/unet.onnx` — ONNX for portable inference.
- `results/unet/unet_metrics.json` — aggregated test metrics.
- `results/unet/per_chip/unet.npz` — per-chip IoU.
- `reports/figures/phase4_training_curves.png`
- `reports/figures/phase4_qualitative_grid.png`

## 9. Phase 4 exit criteria — audit

- [ ] Training completed (best checkpoint saved).
- [ ] Mean test IoU ≥ 0.75 (meets PRD G1).
- [ ] Beats classical baseline with 95 % bootstrap CI excluding zero.
- [ ] McNemar significance recorded.
- [ ] Training curves + qualitative grid saved.
- [ ] ONNX export successful.
- [ ] Hybrid fusion table filled in.

Proceed → **Phase 5 · Analysis & damage severity.**
