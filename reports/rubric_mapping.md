# Rubric Mapping — Self-Audit

This document maps every course rubric criterion to the **concrete artefacts** that demonstrate it, and points to the files / phases where the evidence lives. Tick the boxes as work lands.

---

## Criterion 1 · Problem Definition (10%) — 5 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| Clear problem statement with motivation | §1 of `phase1_report.md` | `reports/phase1_report.md` | [ ] |
| Background research ≥ 10 citations | `lit_review.md` | `reports/lit_review.md` | [x] scaffolded |
| Research questions + hypotheses | §1.3–1.4 of `phase1_report.md` | `reports/phase1_report.md` | [x] scaffolded |
| Explicit scope / non-goals | §3 of PRD; §1.5 of phase1 | `PRD.md`, `reports/phase1_report.md` | [x] |

## Criterion 2 · Implementation of DIP Techniques (20%) — 10 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| ≥ 5 DIP techniques implemented | `src/dip/` modules (Phase 2) | `src/dip/{indices,thresholding,morphology,change_detection,filters}.py` | [ ] Phase 2 |
| Written justification per technique | `reports/phase2_report.md` | `reports/` | [ ] Phase 2 |
| DIP walkthrough notebook | `notebooks/02_classical_dip_walkthrough.ipynb` | `notebooks/` | [ ] Phase 2 |
| Frequency-domain analysis (Fourier) | `notebooks/02b_frequency_analysis.ipynb` | `notebooks/` | [ ] Phase 2 |

## Criterion 3 · Technical Accuracy & Code Quality (20%) — 10 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| Modular src/ layout | Phase 0 skeleton | `src/` | [x] |
| Type hints + docstrings | All modules | `src/`, `tests/` | [x] ongoing |
| `pytest` with ≥ 70% coverage | `tests/` + `pyproject.toml` | `tests/`, `pyproject.toml` | [ ] Phase 2–5 |
| Pre-commit (ruff + black + nbstripout) | `.pre-commit-config.yaml` | root | [x] |
| CI pipeline (optional) | `.github/workflows/ci.yml` | root | [ ] Phase 6 polish |
| Reproducible env | `environment.yml` + `requirements.txt` | root | [x] |
| README + architecture diagram | `README.md` + report appendix | root, report | [x] README / [ ] diagram |

## Criterion 4 · Analysis & Results (20%) — 10 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| IoU / F1 / precision / recall | `src/eval/metrics.py` (Phase 3) | `src/eval/` | [ ] Phase 3 |
| Confusion matrix + per-class accuracy | `src/eval/metrics.py` | `src/eval/` | [ ] Phase 3 |
| Cohen's κ + overall accuracy | `src/eval/metrics.py` | `src/eval/` | [ ] Phase 3 |
| Ablation table (≥ 12 rows) | `results/ablation.csv` (Phase 3) | `results/` | [ ] Phase 3 |
| Qualitative map grid | `reports/figures/` PNGs | `reports/figures/` | [ ] Phase 3–5 |
| Statistical significance (McNemar) | `src/eval/ablation.py` | `src/eval/` | [ ] Phase 3 |

## Criterion 5 · Report Quality & Documentation (10%) — 5 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| IEEE-style report 10–15 pages | Overleaf → `report/final_report.pdf` | `reports/` | [ ] Phase 8 |
| All figures captioned + referenced | Report body | `reports/` | [ ] Phase 8 |
| ≥ 15 references in IEEE format | Report bibliography | `reports/` | [x] 14 scaffolded → +1 |
| Appendix: arch diagram, hyperparams | Report appendix | `reports/` | [ ] Phase 8 |

## Criterion 6 · Presentation & Discussion (20%) — 10 marks

| Evidence needed | Artefact | Location | Status |
|---|---|---|---|
| 18-slide deck (20 min) | `reports/slides.pdf` | `reports/` | [ ] Phase 8 |
| Live Streamlit web-app demo | deployed URL | Streamlit Cloud | [ ] Phase 6 |
| 60 s fallback screen-recording | `reports/demo.mp4` | `reports/` | [ ] Phase 8 |
| 20 anticipated Q&A drafted | `reports/qna.md` | `reports/` | [ ] Phase 8 |
| Two full dry-runs timed | Rehearsal notes | N/A | [ ] Phase 8 |

---

## Self-grading target

| Criterion | Marks | Our target |
|---|---|---|
| Problem Definition | 5 | **5** |
| DIP Implementation | 10 | **9–10** |
| Code Quality | 10 | **9–10** |
| Analysis & Results | 10 | **9–10** |
| Report Quality | 5 | **5** |
| Presentation | 10 | **9–10** |
| **Total** | **50** | **46–50** |

Phase 8 final-audit check: every row above must be ticked or explicitly justified.
