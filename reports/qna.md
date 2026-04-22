# Q&A Preparation — 20 Anticipated Evaluator Questions

> Review before each dry-run. Every speaker should know the answers to the questions touching their area (problem · data · classical DIP · U-Net · results · app · limitations).

---

### Problem & motivation (D)

**Q1 · Why floods, not earthquakes or fires?**
Floods are the most economically damaging natural disaster in India — Kerala 2018 alone caused ₹40 000 crore of losses. Optical remote sensing works well for flood extent (water has a clear spectral signature); buildings collapsed in an earthquake need sub-meter imagery that Sentinel-2 doesn't provide. Our compute budget (free Colab) ruled out high-res sensors, so flood became the best-fit problem.

**Q2 · Why Sentinel-2 and not Landsat or commercial imagery?**
Sentinel-2 is free, 10 m, global, with a 5-day revisit and a published Python API. Landsat is coarser (30 m) with slower revisit. Commercial imagery (Maxar, Planet) is paid and was out of scope.

---

### Data (B)

**Q3 · Why Sen1Floods11 and not your own labels?**
Sen1Floods11 provides 446 hand-labelled flood chips across 11 global events — far more ground truth than we could produce in a semester. It's also the de-facto benchmark in recent flood-mapping papers, so our numbers are directly comparable to published work.

**Q4 · Your U-Net IoU is 0.548, but some papers report 0.75+. Why?**
They train on Sen1Floods11's larger **WeaklyLabeled** split (4 385 chips, ≈ 40 GB) rather than the **HandLabeled** split (252 chips) we used. WeaklyLabeled was excluded due to our free-tier Colab disk/compute budget. On the HandLabeled-only regime, 0.55 IoU is in line with published baselines.

**Q5 · Where is the Kerala ground truth?**
The UNOSAT polygon (product A-1879) and the Copernicus EMS mirror URLs both returned 404 during the project window. PRD v1.2 was amended to use Sen1Floods11 as the sole quantitative evaluation source and treat Kerala 2018 as a qualitative case study. We documented the decision in `reports/phase5_analysis.md`.

---

### Classical DIP (C)

**Q6 · Why does NDWI beat MNDWI on Sen1Floods11, when the textbook says MNDWI is better?**
Two reasons. First, flood water is highly turbid — sediments raise SWIR1 reflectance, which enters the MNDWI denominator and shrinks the index over flooded pixels. NDWI uses NIR which is absorbed by water regardless of turbidity. Second, Sen1Floods11 is dominated by rural/vegetated terrain; MNDWI's specific advantage over NDWI (urban-robustness) doesn't manifest. Konapala et al. (2021) report the same ordering.

**Q7 · Why Yen and not Otsu?**
Sen1Floods11 histograms on water indices are right-skewed — water is a minority class with a heavy tail — and Otsu's within-class-variance objective biases toward the majority class in such distributions. Yen's maximum-entropy criterion handles skewed distributions better. Empirically, Yen scored +0.13 IoU over Otsu on NDWI in our Phase-3 ablation.

**Q8 · Why don't you apply morphology in the winning config?**
Our ablation found `ndwi_yen_raw` (no morphology) beats `ndwi_yen_morph` by 0.007 IoU. The default `min_object_area = 25 px` removes legitimate small flood pockets that Yen was correctly catching. We chose not to tune morphology parameters to avoid selection-on-validation.

**Q9 · Why combine water and change masks?**
Water mask alone includes permanent rivers/lakes (not flood damage). The ΔMNDWI change mask alone includes spurious changes unrelated to water. Their intersection isolates **newly inundated** pixels, which is the actual disaster extent.

---

### U-Net (C)

**Q10 · Why ResNet-34 and not a larger encoder?**
ResNet-34 fits in the T4's 16 GB VRAM at batch size 8 with 256 × 256 crops. ResNet-50 blew past VRAM; EfficientNet-B0 had marginal accuracy gains for ~2× the compute. We chose the accuracy/compute sweet spot.

**Q11 · Why BCE + Dice and not just BCE?**
Sen1Floods11 is class-imbalanced (~25 % water). BCE alone biases toward the majority (non-water) class. Dice is a direct surrogate for IoU and is robust to imbalance. Combining them (α = 0.5) keeps BCE's probability calibration and Dice's foreground focus. `pos_weight = 2.0` further upweights water.

**Q12 · You kept ImageNet weights on a 6-channel reflectance input. Doesn't that hurt?**
Yes, it's a distribution shift — the extra three input channels are randomly initialised and the network has never seen reflectance-scale inputs during ImageNet pretraining. But the encoder's later layers learn spectrally-independent shape features, and fine-tuning from ImageNet typically outperforms training-from-scratch on data-poor benchmarks. We validated this empirically via the training curves.

**Q13 · Why no Sentinel-1 (SAR)?**
Time budget. SAR integration was promoted to should-have in PRD v1.1 (instructor confirmed bonus marks) but descoped to future work when tunnel-setup for the Streamlit demo took longer than expected. The downloader and loader for S1 GRD already exist in `src/data/gee_download.py` — adding a parallel SAR U-Net is 1–2 days of work.

---

### Results & statistics (D)

**Q14 · Your ΔIoU CI is [+0.037, +0.114]. How do you interpret that?**
We resampled the 90 per-chip IoU differences with replacement 10 000 times and took the 2.5th and 97.5th percentiles. Because the entire interval sits above zero, we reject the null hypothesis that U-Net and classical perform equivalently on this benchmark at the 95 % confidence level. The bootstrap is distribution-free — no normality assumption.

**Q15 · Why didn't the hybrid fusion help?**
The classical mask is noisy — it has 0.76 recall but 0.59 precision, meaning many false positives. Fusing it with the U-Net's calibrated probabilities adds noise rather than information. With w = 0.7 we're down-weighting the classical component but still paying its FP cost. A different fusion formulation (e.g., using classical only to seed U-Net's attention) could help but was beyond our scope.

**Q16 · Is McNemar the right test here?**
It's the standard test for comparing two classifiers on the same pixels. Both U-Net and Hybrid vs Classical gave χ² values on the order of 10⁶ (p ≈ 0) because we pool ≈ 12 M valid pixels across 90 chips. With that many observations, even small accuracy differences become significant — so we also report the effect-size-sensitive bootstrap CI.

**Q17 · Cohen's κ jumped from 0.50 to 0.65. What's that on the Landis-Koch scale?**
0.50 is in the "moderate agreement" band (0.41–0.60). 0.65 is in the "substantial agreement" band (0.61–0.80). The jump confirms the U-Net doesn't just marginally improve accuracy — it meaningfully changes the quality category of the flood map.

---

### App & system (A)

**Q18 · Can the app run without a GPU?**
Yes. The classical path runs in ~6 ms/chip on CPU. The U-Net path runs in ~200 ms/chip on CPU (vs 33 ms on T4). The app auto-detects and falls back to CPU if no GPU is present. Hybrid also supports CPU.

**Q19 · How would you deploy this for real-time operational use?**
Three changes: (i) ingest Sentinel-2 via the Copernicus Open Access Hub push notifications instead of polling; (ii) replace the Streamlit UI with a FastAPI REST service for programmatic clients; (iii) containerise and deploy behind a load balancer — the U-Net does ~30 predictions/s/GPU, so a small cluster handles national-scale coverage. For the course project we chose Streamlit for the demo + PDF report rather than a production API.

---

### Generic (anyone)

**Q20 · What's the most important thing you learned?**
The empirical pipeline must be trusted more than the textbook prediction. We expected MNDWI + Otsu + morphology to win the classical ablation based on the literature; NDWI + Yen without morphology actually won. Our ablation harness caught it, and the error analysis explained it in terms of water turbidity and Sen1Floods11's geographic composition. A well-instrumented experiment surfaces surprises that motivate stronger science.

---

## Bonus "gotcha" questions

**Q-bonus 1 · What if an evaluator uploads a non-Sentinel-2 image?**
The app's `Sen1Floods11Dataset` reads only the HandLabeled test split; a "bring-your-own" upload path is possible via `src/pipelines/full_pipeline.run_pipeline(chip, method)` but not wired into the UI. We would need a file uploader + band-ordering validator.

**Q-bonus 2 · Why 0.5 as the U-Net binarisation threshold?**
Default — we didn't optimise it. Sweeping the threshold on the validation set could improve F1 by a few points; deliberately not done to avoid overfitting to the 89-chip validation split.

**Q-bonus 3 · How big is your U-Net checkpoint?**
~93 MB on disk (state_dict only). ~380 MB in RAM during inference. ONNX export succeeds at ~93 MB; 8-bit quantisation would cut it to ~25 MB.

**Q-bonus 4 · What happens if both pre and post images are cloudy?**
The SCL-based cloud mask zeros out cloud pixels before the reducer. If cloud cover > 40 % in a window (our threshold), the GEE downloader raises an error requesting a wider date window. The classical path would under-detect water; the U-Net is robust only to the extent cloud-free pixels are present.
