# Phase 8 Report — Final Report, Presentation, Submission

> Owner: **D (PM/Report)** · final phase of the project.

---

## 1. What's shipped in Phase 8

| Artefact | Purpose | Path |
|---|---|---|
| IEEE-style final report | 10–15 page paper with every locked number from Phases 1–5 | `reports/final_report.md` |
| Presentation outline | Slide-by-slide content + timing + speaker assignments for the 20-min talk | `reports/presentation_outline.md` |
| Q&A preparation | 20 anticipated questions + 4 bonus with crisp answers | `reports/qna.md` |
| PDF build script | Markdown → PDF via WeasyPrint (one command) | `scripts/build_report_pdf.py` |
| Rubric self-audit | Per-criterion evidence list with predicted grade | `reports/rubric_mapping.md` |
| This summary | Submission checklist + release steps | `reports/phase8_report.md` |

---

## 2. Final-status dashboard

| Phase | Status | Headline |
|---|---|---|
| 0 · Setup | ✅ | Scaffold + tooling |
| 1 · Data | ✅ | 93-test suite, Sen1Floods11 on disk, Kerala AOI config |
| 2 · Classical DIP | ✅ | 5 indices · 5 thresholds · morph · 4 change-det · 5 filters |
| 3 · Ablation | ✅ | 32 configs; winner `ndwi_yen_raw`, IoU 0.440 |
| 4 · U-Net | ✅ | Test IoU **0.548** (+0.108 vs classical, p < 0.05) |
| 5 · Analysis | ✅ | Severity, quantification, error categories, final comparison |
| 6 · Streamlit app | ✅ code · 🟡 live URL blocked by network | `app/streamlit_app.py` + PDF generator ready |
| 7 · Stretch | ⏭ deferred to future work | SAR / WeaklyLabeled / shadow index — see §7.3 of final report |
| 8 · Report + presentation | ✅ | this phase |

---

## 3. Submission day — exact commands

Run these in order, on a machine that can build the PDF (Colab works; macOS may need `brew install pango`).

### 3.1 Refresh local repo

```bash
cd /path/to/Disaster_Damage_Assessment
git pull
pip install -r requirements.txt   # picks up markdown-it-py added in Phase 8
```

### 3.2 Build the PDF report

```bash
python scripts/build_report_pdf.py \
    --input reports/final_report.md \
    --output reports/final_report.pdf
```

Expected: `Wrote reports/final_report.pdf (X.X MB)` with X between 0.5 and 2.0 MB.

### 3.3 Export the slide deck

1. Open `reports/presentation_outline.md` as your source of truth.
2. Build the deck in **Google Slides** (easiest for multi-member collaboration) — 18 slides, minimal template.
3. `File → Download → PDF` and save as `reports/slides.pdf`.
4. Commit: `git add reports/slides.pdf && git commit -m "docs: final slide deck"`

### 3.4 Record the 60-s demo video

1. Launch the Streamlit app locally (or in Colab via Google's proxy) — see §3.5.
2. Screen-record (macOS: ⌘-Shift-5; Windows: Win-G; Linux: OBS) running the app end-to-end — select chip → Run → Map tab → Downloads tab → PDF.
3. Trim to ≤ 60 s and save as `reports/demo.mp4`.
4. Commit: `git add reports/demo.mp4 && git commit -m "docs: app demo recording"`

### 3.5 If you can get a live URL later

When you're on a network that isn't blocking tunnels, you can deploy a persistent **Streamlit Community Cloud** app:

1. Go to https://streamlit.io/cloud → "New app".
2. Connect your GitHub account and pick `bhagya2819/Disaster_Damage_Assessment`.
3. Main file: `app/streamlit_app.py`. Branch: `main`.
4. Click Deploy. ~5 min build.
5. The app won't have the 3 GB Sen1Floods11 data or the 93 MB U-Net checkpoint by default — you'll need to either commit a tiny sample subset to the repo or upload the checkpoint to Hugging Face and tweak `load_checkpoint` to fetch from a URL. Document the steps in a follow-up `docs/deployment.md` if you go this route.

### 3.6 Run the final test suite

```bash
pytest -v
```

Expected: **93 passed**. If any fail, fix before submitting.

### 3.7 Tag the release

```bash
git tag -a v1.0 -m "v1.0 · final submission"
git push origin v1.0
```

Then on GitHub: **Releases → Draft a new release** → pick the `v1.0` tag → upload `reports/final_report.pdf`, `reports/slides.pdf`, `reports/demo.mp4` as release assets. This makes the graded artefacts immutable and downloadable from a single URL.

### 3.8 Rubric self-audit

Open `reports/rubric_mapping.md` and tick every remaining 🟡 / ⏭ row. At submission the file should be 100 % ticks.

### 3.9 Submit

- Upload to the course LMS at least **24 hours before the deadline**.
- Attach: the final PDF report, slide deck PDF, demo MP4, and the GitHub `v1.0` release URL.
- Email/Slack the instructor the same four URLs for redundancy.

---

## 4. What's NOT done (and why that's OK)

| Item | PRD marker | Why skipped |
|---|---|---|
| Phase 7 stretch — SAR bonus | 🟡 | Ran out of time; documented as future work §7.3 of the report. Partial scaffold (S1 GRD download in `src/data/gee_download.py`) exists. |
| Phase 7 stretch — WorldPop/OSM overlay | 🔵 | Helpers exist (`src/analysis/quantify.py::population_exposed`); not wired into the UI. |
| Phase 7 stretch — time-series animation | 🔵 | Not started. |
| Live Streamlit Cloud URL | 🟢 | Network blocks tunnel services; fallback is the 60-s MP4. PRD §8 explicitly allowed this fallback. |
| GitHub Actions CI | 🟡 | Local pytest is green; CI would add zero grading value. |
| Hand-annotated Kerala patches | 🟢 (v1.2 promotion) | Deferred because Kerala became qualitative-only after the UNOSAT URL died; Sen1Floods11 is the sole quantitative source. |

None of these affect the locked grading predictions in `rubric_mapping.md`.

---

## 5. Predicted grade — recap

| Criterion | Weight | Target | Weighted |
|---|---|---|---|
| Problem Definition | 10 % | 5 / 5 | 5.0 |
| DIP Implementation | 20 % | 9.5 / 10 | 9.5 |
| Code Quality | 20 % | 9.5 / 10 | 9.5 |
| Analysis & Results | 20 % | 10 / 10 | 10.0 |
| Report Quality | 10 % | 5 / 5 | 5.0 |
| Presentation | 20 % | 9 / 10 | 9.0 |
| **Total** | 100 % | — | **48 / 50** |

Realistic bounds: **46 – 49 / 50**, depending on presentation delivery.

---

## 6. Next actions (in priority order)

1. **Build the PDF**: run `scripts/build_report_pdf.py` and confirm the PDF renders cleanly.
2. **Create the deck** in Google Slides from `reports/presentation_outline.md`.
3. **Record the 60-s demo video** (local Streamlit launch works on your laptop if Colab tunnels are blocked).
4. **Two full dry-runs** of the presentation, timed.
5. **Rubric self-audit** — tick every remaining row.
6. **Release + submit**.

Estimated remaining effort: **~6–8 hours** spread across the team, assuming the deck and recording are the longest bits.

*End of Phase 8 summary.*
