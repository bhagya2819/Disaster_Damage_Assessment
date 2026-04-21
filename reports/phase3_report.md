# Phase 3 Report — Evaluation & Ablation Study

> Owner: **D (PM/Report)** · feeds §4 (Experiments) of the final IEEE report.

---

## 1. Objective

Quantitatively compare every classical DIP configuration on the same held-out data, choose a winner as the "classical baseline" for Phase 4 comparison, and produce the qualitative & statistical artefacts the rubric's 20 % Analysis & Results criterion demands.

## 2. Evaluation protocol

| Item | Choice |
|---|---|
| Benchmark | **Sen1Floods11** HandLabeled, `test` split (~89 chips, 512×512). |
| Modality | Sentinel-2 6-band reflectance (B2, B3, B4, B8, B11, B12), normalised to [0, 1]. |
| Ground truth | Per-chip binary flood mask in {0, 1}; pixels labelled `-1` are ignored. |
| Metrics | IoU, F1 (=Dice), precision, recall, accuracy, Cohen's κ, per-class accuracy, TP/FP/FN/TN. Source: `src/eval/metrics.py`. |
| Per-chip records | IoU saved to `results/per_chip/<config>.npz` for downstream significance tests. |

*(v1.2 of the PRD dropped UNOSAT/EMS as a second evaluation source — Sen1Floods11 is now the sole quantitative GT for all Phase-3 numbers.)*

## 3. Ablation search space

`src/eval/ablation.py` computes every point in the Cartesian product:

- **Index** (4): NDWI, MNDWI, AWEInsh, AWEIsh
- **Threshold** (4): Otsu, Triangle, Yen, Li
- **Morphology** (2): `morphology.clean(...)` on / off

→ **4 × 4 × 2 = 32 configurations.** On Sen1Floods11 test (~89 chips) this is 2,848 mask evaluations, about **3–5 min on Colab CPU**.

Output: `results/ablation.csv`, long-format with mean ± std per metric per configuration.

## 4. Statistical significance

Two tools live in `src/eval/significance.py`:

1. **McNemar's test (continuity-corrected χ²)** on per-pixel disagreements between the top-two configurations. H₀: the two methods are pixel-level equivalent.
2. **Paired bootstrap CI** on per-chip ΔIoU with 10 000 resamples and a 95 % CI. Gives an effect-size interpretation rather than a yes/no verdict.

## 5. Winning configuration — to be filled in after the notebook run

| Metric | Value |
|---|---|
| Config name | *(filled by notebook)* |
| Index | |
| Threshold method | |
| Morphology | |
| Mean IoU | |
| Mean F1 | |
| Mean κ | |
| ΔIoU vs runner-up (95% CI) | |
| McNemar χ² (p-value) | |

**Narrative** (draft to revise after data is in):

> On the Sen1Floods11 test split, the **MNDWI + Otsu + morphology** configuration achieved the highest mean IoU (`_.___`), confirming the theoretical expectation that (a) SWIR-based water indices outperform NIR-based ones over urban areas (Xu 2006), (b) Otsu's assumption of a bimodal histogram is well-matched to MNDWI's observed distribution, and (c) removing salt-and-pepper speckle produces a non-trivial IoU gain (+0.XX over raw thresholding). The runner-up configuration (`<runner>`) trailed by ΔIoU = `_.___` (95 % CI [`_.___`, `_.___`]); a McNemar test on pooled pixels gave χ² = `_.___` (p = `_.___`), `(not)` significant at α = 0.05. **This winning configuration is adopted as the classical baseline for Phase 4 comparison against the U-Net.**

## 6. Artefacts produced

- `results/ablation.csv` — 32-row table
- `results/per_chip/*.npz` — 32 × per-chip IoU arrays
- `reports/figures/phase3_ablation_bars.png` — horizontal bar chart
- `reports/figures/phase3_heatmap.png` — index × threshold IoU heatmap
- `reports/figures/phase3_qualitative_grid.png` — 4×4 grid (4 chips × [RGB, Winner, Runner, GT])

## 7. Phase 3 exit criteria — audit

- [ ] Full pytest suite green (incl. `tests/test_eval.py`).
- [ ] `run_ablation.py` completed on test split; CSV present and has 32 rows.
- [ ] Winner + runner-up identified and recorded in §5.
- [ ] Bootstrap CI computed; values filled in §5.
- [ ] McNemar χ² + p-value recorded in §5.
- [ ] Ablation bar chart, heatmap and qualitative grid saved as PNGs.
- [ ] Narrative paragraph in §5 finalised.

When complete → begin **Phase 4 · U-Net training**.

---

## 8. Next-phase dependencies

Phase 4 will reuse:
- `src/eval/metrics.py` — same metric suite evaluates the U-Net on the same test split.
- `src/eval/significance.py` — the bootstrap / McNemar comparison now runs U-Net vs the classical winner documented above.
- `results/per_chip/<winner>.npz` — paired inputs for the U-Net-vs-classical CI.
