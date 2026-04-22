# Rubric Self-Audit — Final

> Live as of Phase 8 completion · 2026-04-22.
> Target: 46–50 / 50.

---

## Criterion 1 · Problem Definition (10 %) — target **5 / 5**

| Evidence needed | Artefact | Status |
|---|---|---|
| Clear problem statement with motivation | `reports/final_report.md` §1 | ✅ |
| Background research ≥ 10 citations | `reports/final_report.md` §2 (20 IEEE refs) | ✅ |
| Research questions + hypotheses | `reports/phase1_report.md` §1.3–1.4 | ✅ |
| Explicit scope / non-goals | `PRD.md` §3 | ✅ |

**Expected grade: 5 / 5 (Excellent)** — problem is precisely scoped, 20 citations, hypotheses explicitly stated and audited in §6.3 of the report.

---

## Criterion 2 · Implementation of DIP Techniques (20 %) — target **9.5 / 10**

| Evidence needed | Artefact | Status |
|---|---|---|
| ≥ 5 DIP techniques implemented | `src/dip/indices.py` (5 indices) + `thresholding.py` (5 methods) + `morphology.py` + `change_detection.py` (4 methods) + `filters.py` (5 filters) | ✅ |
| Written justification per technique | `reports/final_report.md` §2 + §4 | ✅ |
| DIP walkthrough notebook | `notebooks/02_classical_dip_walkthrough.ipynb` | ✅ |
| Frequency-domain analysis (FFT) | `notebooks/02_classical_dip_walkthrough.ipynb` §6 | ✅ |

**Expected grade: 9.5 / 10** — all 5 technique families are implemented (point, neighborhood, morphology, frequency, change detection) and justified with primary references. Half-point deducted because we did not implement frequency-domain filtering *in the pipeline* — only as a sanity check.

---

## Criterion 3 · Technical Accuracy & Code Quality (20 %) — target **9.5 / 10**

| Evidence needed | Artefact | Status |
|---|---|---|
| Modular src/ layout | `src/{data,preprocess,dip,models,train,inference,eval,analysis,pipelines,utils}/` | ✅ |
| Type hints + docstrings | Every module | ✅ |
| `pytest` with ≥ 70 % coverage | `tests/` + `pyproject.toml` · **93 tests · 75 % line coverage** | ✅ |
| Pre-commit (ruff + black + nbstripout) | `.pre-commit-config.yaml` | ✅ |
| CI pipeline | — | ⏭ deferred |
| Reproducible env | `environment.yml` · `requirements.txt` | ✅ |
| README + architecture diagram | `README.md` · `reports/final_report.md` Appendix B | ✅ |

**Expected grade: 9.5 / 10** — half-point deducted for not wiring up GitHub Actions CI. Everything else is professional-grade: typed, tested, formatted, reproducible.

---

## Criterion 4 · Analysis & Results (20 %) — target **10 / 10**

| Evidence needed | Artefact | Status |
|---|---|---|
| IoU / F1 / precision / recall | `src/eval/metrics.py` + `reports/final_report.md` §6.4 Table 2 | ✅ |
| Confusion matrix + per-class accuracy | `reports/figures/phase5_confusion.png` | ✅ |
| Cohen's κ + overall accuracy | Reported for every method in Table 2 | ✅ |
| Ablation table (≥ 12 rows) | `results/ablation.csv` · 32 rows | ✅ |
| Qualitative map grid | `reports/figures/phase3_qualitative_grid.png` | ✅ |
| Statistical significance | Paired bootstrap + McNemar in §6.5 | ✅ |
| Error-category analysis | `src/analysis/error_analysis.py` + §6.7 | ✅ |
| Negative result reporting | Hybrid fusion, §6.6 | ✅ |

**Expected grade: 10 / 10** — we exceed every sub-criterion: 32 ablation rows (> 12 required), two independent significance tests, a principled negative result, and error-category analysis tying results to future work.

---

## Criterion 5 · Report Quality & Documentation (10 %) — target **5 / 5**

| Evidence needed | Artefact | Status |
|---|---|---|
| IEEE-style report 10–15 pages | `reports/final_report.md` + `reports/final_report.pdf` (built via `scripts/build_report_pdf.py`) | ✅ |
| All figures captioned + referenced | §6 of the report | ✅ |
| ≥ 15 references in IEEE format | §References — **20 refs** | ✅ |
| Appendix: arch diagram, hyperparams, hardware | Appendix A, B, C of the report | ✅ |

**Expected grade: 5 / 5** — the report is complete, has 20 citations (>1 above "Excellent" threshold), reproducibility section explains how to regenerate every number.

---

## Criterion 6 · Presentation & Discussion (20 %) — target **9 / 10**

| Evidence needed | Artefact | Status |
|---|---|---|
| 18-slide deck (20 min) | `reports/presentation_outline.md` (slide-by-slide content + timing + speaker assignments) | 🟡 to-build |
| Live Streamlit web-app demo | `app/streamlit_app.py` (code ready; live URL blocked by network) | 🟡 |
| 60-s fallback screen-recording | — | 🟡 to-record |
| 20 anticipated Q&A drafted | `reports/qna.md` (20 + 4 bonus) | ✅ |
| Two full dry-runs timed | — | 🟡 to-do |

**Expected grade: 9 / 10** (pending dry-runs). The deck outline is complete and all numeric claims are traceable; Q&A sheet is reviewed. The main risk is the live demo — a 60-second pre-recorded screen capture is the fallback per PRD §8.

---

## Final grade forecast

| Criterion | Weight | Target | Weighted |
|---|---|---|---|
| Problem Definition | 10 % · 5 marks | 5 / 5 | 5.0 |
| DIP Implementation | 20 % · 10 marks | 9.5 / 10 | 9.5 |
| Code Quality | 20 % · 10 marks | 9.5 / 10 | 9.5 |
| Analysis & Results | 20 % · 10 marks | 10 / 10 | 10.0 |
| Report Quality | 10 % · 5 marks | 5 / 5 | 5.0 |
| Presentation | 20 % · 10 marks | 9 / 10 | 9.0 |
| **Total** | 100 % · 50 marks | — | **48.0 / 50** |

Realistic bounds: **46 – 49 / 50**, depending on the quality of the live presentation delivery.

---

## Submission-day checklist

- [ ] `git tag v1.0 && git push --tags`
- [ ] Build final PDF: `python scripts/build_report_pdf.py --output reports/final_report.pdf`
- [ ] Export slide deck to PDF and commit as `reports/slides.pdf`
- [ ] Record 60-s demo screen-capture, commit as `reports/demo.mp4`
- [ ] Run full test suite: `pytest -v` (expect 93 passed)
- [ ] Final rubric self-audit (this file) — every row ticked
- [ ] Submit via course LMS with ≥ 24 hour buffer before deadline
