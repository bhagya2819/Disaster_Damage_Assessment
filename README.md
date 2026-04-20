# DDA — Disaster Damage Assessment (Flood Mapping)

**Course:** Digital Image Processing
**Domain:** Remote Sensing & Satellite Image Processing
**Case study:** Kerala Floods, August 2018 (Sentinel-2 L2A + Sen1Floods11 benchmark)

End-to-end pipeline that takes **pre/post-event Sentinel-2 imagery** for an area of interest and produces:

1. Flood-water extent mask (pixel-accurate binary segmentation)
2. Pre/post change-detection map
3. Damage severity classification (No / Low / Moderate / Severe)
4. Automated quantitative PDF damage report (area flooded, population exposure, road/building impact)

The project combines **classical DIP** (spectral indices, Otsu/adaptive thresholding, morphology, PCA change detection) with a **U-Net** trained on Sen1Floods11, and ships as a **Streamlit web app**.

See [`PRD.md`](./PRD.md) for the full phase-by-phase plan.

---

## Quick start

### Local (conda)
```bash
conda env create -f environment.yml
conda activate dda-flood
pre-commit install
pytest
```

### Google Colab
```python
!git clone <repo-url> dda-flood && cd dda-flood
!pip install -r requirements.txt
```

### Run the web app locally
```bash
streamlit run app/streamlit_app.py
```

---

## Project layout

```
.
├── PRD.md                     # Product requirements (phase plan)
├── README.md                  # This file
├── environment.yml            # Conda env (local)
├── requirements.txt           # Pip deps (Colab)
├── pyproject.toml             # Package + ruff/black/mypy/pytest config
├── .pre-commit-config.yaml    # Lint/format hooks
├── Makefile                   # Common commands
├── src/                       # Python package
│   ├── data/                  # Ingestion (GEE, Sen1Floods11 loader)
│   ├── preprocess/            # Reflectance, cloud mask, coregistration
│   ├── dip/                   # Classical DIP (indices, thresholds, morphology, change detection)
│   ├── models/                # U-Net architecture
│   ├── train/                 # Training loops
│   ├── inference/             # Tiled prediction
│   ├── eval/                  # Metrics, ablation harness
│   ├── analysis/              # Severity classification, quantification
│   ├── pipelines/             # End-to-end orchestration
│   └── utils/                 # Shared helpers (geo, logging, io)
├── notebooks/                 # EDA, walkthroughs, ablation studies
├── data/
│   ├── raw/                   # Downloaded Sentinel-2 scenes (gitignored)
│   ├── processed/             # Coregistered, cloud-masked tiles
│   ├── gt/                    # Ground-truth (UNOSAT, manual)
│   └── external/              # WorldPop, OSM, ESA WorldCover
├── reports/
│   ├── figures/               # 300 DPI PNG/SVG for final report
│   └── *.md                   # Phase reports + rubric mapping
├── tests/                     # pytest suite
├── app/                       # Streamlit app + report generator
├── configs/                   # YAML configs for experiments
└── scripts/                   # One-off CLI utilities
```

---

## Roles (3–4 members)

| Role | Scope |
|---|---|
| **A — Tech / App** | Repo, CI, Streamlit, inference, evaluation utilities |
| **B — Data / Geo** | GEE, preprocessing, coregistration, exposure overlays |
| **C — DIP / ML** | Spectral indices, thresholding, U-Net, hybrid fusion |
| **D — PM / Report** | Literature, report, deck, rubric mapping |

---

## Environment variables

Copy `.env.example` → `.env` and fill in:

- `GEE_SERVICE_ACCOUNT_JSON` — path to Earth Engine service-account key (optional; user-auth works too)
- `WANDB_API_KEY` — Weights & Biases logging (optional)

---

## Licence

MIT. See [`LICENSE`](./LICENSE).
