# Deployment guide — Streamlit Community Cloud & Hugging Face Spaces

Two free paths to get a public URL for the DDA Streamlit app that evaluators
can open in any browser, without network-level tunnel blocks (ngrok,
trycloudflare, etc.).

---

## Option A · Streamlit Community Cloud (recommended)

Free, backed by Streamlit directly, auto-deploys from GitHub on every push.

### A.1 Prerequisites in the repo (already shipped)

| File | Purpose |
|---|---|
| `app/streamlit_app.py` | App entry point |
| `requirements.txt` | Pip dependencies |
| `packages.txt` | Apt packages (libpango/libcairo for WeasyPrint) |
| `.streamlit/config.toml` | Theme + server settings |
| `app/sample_chips/*.npz` | Bundled demo chips (no external dataset needed) |

### A.2 One-time build of bundled chips

Before pushing, on any machine that has the full Sen1Floods11 dataset:

```bash
python scripts/build_sample_chips.py \
    --sen1floods11-root /content/drive/MyDrive/dda/sen1floods11 \
    --out-dir app/sample_chips \
    --n-chips 3

git add app/sample_chips/*.npz app/sample_chips/manifest.json
git commit -m "chore(app): bundle 3 sample chips for public demo"
git push
```

Resulting bundle: ~500 kB total. Commit them — the `.gitignore` has been
loosened to allow `app/sample_chips/**/*.npz`.

### A.3 Hosting the 93 MB U-Net checkpoint

Streamlit Cloud's 1 GB disk is enough for the checkpoint, but Git and
Git LFS aren't. Two clean options:

**A.3.a Hugging Face Hub (free, recommended)**
1. Go to https://huggingface.co/new → create a free account if needed.
2. Click **New model** → name it `dda-flood-unet-resnet34` → Public.
3. Upload `best.pt` via the browser (drag & drop).
4. Public URL is `https://huggingface.co/<your-username>/dda-flood-unet-resnet34/resolve/main/best.pt`.
5. Copy that URL — we'll wire it into a Streamlit secret.

**A.3.b GitHub Release asset**
1. `git tag v1.0 && git push origin v1.0`.
2. GitHub → Releases → Draft new release → pick `v1.0` → drag `best.pt` as an asset.
3. The public download URL is `https://github.com/bhagya2819/Disaster_Damage_Assessment/releases/download/v1.0/best.pt`.

### A.4 Modify the app to fetch the checkpoint on first run

Add a small loader that caches the checkpoint locally and downloads if
missing (lightly patch `app/streamlit_app.py`):

```python
@st.cache_resource(show_spinner="Fetching U-Net checkpoint…")
def fetch_checkpoint(url: str) -> str:
    local = Path("/tmp/dda_best.pt")
    if not local.exists():
        import urllib.request
        urllib.request.urlretrieve(url, local)
    return str(local)
```

Then in the sidebar, default `ckpt_path` to the result of `fetch_checkpoint(st.secrets["CHECKPOINT_URL"])` when running on Streamlit Cloud.

Add the secret:
1. Go to the app's dashboard on Streamlit Cloud.
2. Settings → Secrets → paste:
   ```toml
   CHECKPOINT_URL = "https://huggingface.co/<you>/dda-flood-unet-resnet34/resolve/main/best.pt"
   ```

### A.5 Deploy

1. Push everything to GitHub main.
2. Open https://streamlit.io/cloud.
3. Sign in with GitHub → authorize access to `bhagya2819/Disaster_Damage_Assessment`.
4. **New app** → pick the repo, branch `main`, main file `app/streamlit_app.py`.
5. Advanced settings → Python version `3.12`.
6. **Deploy.** ~5 minutes for first build (pip install), subsequent pushes ~1 min.
7. You get a permanent URL like `https://dda-flood-bhagya2819.streamlit.app`.

### A.6 RAM notes

Streamlit Community Cloud gives you **1 GB RAM** on the free tier. Our
footprint at idle:

| Component | Approx. RAM |
|---|---|
| Streamlit runtime | 180 MB |
| PyTorch (CPU) | 220 MB |
| segmentation-models-pytorch + encoder load | 120 MB |
| U-Net checkpoint loaded | 100 MB |
| Bundled sample chip (128 × 128 × 6) | < 1 MB |
| **Total when running a prediction** | **~600 MB** |

That leaves ~400 MB headroom — comfortably under the limit. Avoid uploading
large user GeoTIFFs (> 2 000 × 2 000 px) or it'll OOM.

---

## Option B · Hugging Face Spaces (if Streamlit Cloud is blocked)

Hugging Face Spaces gives you **16 GB RAM** on the free tier — much more
generous than Streamlit Cloud — and also supports Streamlit natively.

### B.1 Steps

1. Go to https://huggingface.co/new-space.
2. **Owner**: your HF username. **Space name**: `dda-flood`. **SDK**: `Streamlit`. **Visibility**: Public.
3. Hugging Face creates a new git repo at `https://huggingface.co/spaces/<you>/dda-flood`.
4. `git clone` it locally and copy the DDA repo contents in (or push straight from GitHub via the Space's `git remote` setting).
5. Ensure `app.py` (or `app/streamlit_app.py`) is at the root or configured via `app_file: app/streamlit_app.py` in a `README.md` frontmatter block.
6. Push. Space auto-builds.

### B.2 Space config (`README.md` frontmatter at repo root for HF)

```yaml
---
title: DDA Flood Mapping
emoji: 🌊
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.31.0
app_file: app/streamlit_app.py
pinned: false
---
```

Add this block at the top of the root `README.md` **only if deploying to HF**
— Streamlit Cloud ignores it but HF parses it.

### B.3 Advantage

Since the 93 MB U-Net checkpoint can live **inside the Space repo itself**
(HF-hosted git LFS is free), you don't need the HuggingFace-Hub-URL
indirection from §A.4.

---

## Option C · Fallback for presentation: recorded screencast

If neither A nor B works from your network (unlikely — both are on Google-
and Hugging-Face-hosted domains rarely blocked), record a 60-second demo:

1. Run the app locally: `streamlit run app/streamlit_app.py`.
2. Screen-record (macOS: ⌘-Shift-5; Windows: Win-G; Linux: OBS): open URL → click "Run prediction" → show Map tab → Metrics tab → click "Generate PDF report" → show downloaded PDF.
3. Trim to ≤ 60 seconds. Save as `reports/demo.mp4`.
4. Embed in the presentation deck as the final "Live Demo" slide.

Per PRD §8, a recorded video is an acceptable fallback when the live URL is unavailable.

---

## Troubleshooting

### "WeasyPrint: cannot find libpango"
Make sure `packages.txt` is committed at the repo root. Streamlit Cloud
reads this file and installs the listed apt packages before pip.

### "Out of memory" during U-Net inference
You are probably on a large uploaded GeoTIFF. Streamlit Cloud free tier
is 1 GB; downsample the input, or upgrade to Community Plus ($20/mo) for
more RAM.

### "Checkpoint not found" on first app load
Verify the `CHECKPOINT_URL` secret. Curl it from your laptop to confirm the
URL returns a binary file, not an HTML error page (happens with HF private
repos — make sure the model repo is Public).

### Bundled chips missing in deployed app
Confirm `app/sample_chips/*.npz` was actually committed (Streamlit Cloud
shows the file tree on the app's "Manage" page). The `.gitignore` has an
allow-list for these paths but new .npz files still need explicit `git add`.
