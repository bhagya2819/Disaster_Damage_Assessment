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

## 6. Results (fill after training completes)

| Metric | Classical baseline | **U-Net** | Δ |
|---|---|---|---|
| Config | ndwi_yen_raw | resnet34 U-Net | — |
| Mean IoU | 0.440 | *_.___* | *_.___* |
| Mean F1 | 0.547 | *_.___* | *_.___* |
| Mean κ | 0.497 | *_.___* | *_.___* |

### 6.1 Paired comparison (U-Net − classical)

| | Value |
|---|---|
| Mean ΔIoU | *_._____* |
| 95 % bootstrap CI | [*_._____*, *_._____*] |
| McNemar χ² (p) | *_._____* (p = *_._____*) |

### 6.2 Narrative (to draft post-training)

> The ResNet-34 U-Net achieved mean IoU **0.___** on the Sen1Floods11 test split, an improvement of **+0.___** over the classical `ndwi_yen_raw` baseline. The paired bootstrap 95 % CI [0.___, 0.___] excludes zero, and the pixel-level McNemar test (χ² = *_._*, p = *_._*) confirms the difference is statistically significant. The U-Net's advantage is largest on (a) mixed-cover chips where the water–land boundary is thin (index-based methods miss fractional-pixel water), and (b) chips with heavy cloud shadow, where classical thresholds over-predict water from dark pixels. Remaining errors concentrate on turbid water near coastlines and on buildings with wet flat roofs — the latter being a genuine ambiguity that U-Net also cannot resolve.

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
