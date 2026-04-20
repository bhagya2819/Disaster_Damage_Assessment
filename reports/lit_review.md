# Literature Review — Flood Mapping from Satellite Imagery

> Owner: **D (PM/Report)** · Target: ≥ 10 peer-reviewed references, organised by theme.
> This file feeds §2 (Related Work) of the final IEEE-style report.

---

## Theme 1 · Spectral water indices

1. **McFeeters, S.K. (1996).** *The use of the Normalized Difference Water Index (NDWI) in the delineation of open water features.* **IJRS 17(7):1425–1432.**
   NDWI = (Green − NIR) / (Green + NIR). Baseline; fails over urban/built-up pixels.
2. **Xu, H. (2006).** *Modification of normalised difference water index (MNDWI) to enhance open water features in remotely sensed imagery.* **IJRS 27(14):3025–3033.**
   MNDWI = (Green − SWIR) / (Green + SWIR). Fixes NDWI's urban-confusion issue; our primary index.
3. **Feyisa, G.L., Meilby, H., Fensholt, R., Proud, S.R. (2014).** *Automated Water Extraction Index: A new technique for surface water mapping using Landsat imagery.* **RSE 140:23–35.**
   AWEInsh / AWEIsh — outperforms NDWI/MNDWI in shadow-heavy terrain.

## Theme 2 · Thresholding & classical segmentation

4. **Otsu, N. (1979).** *A threshold selection method from gray-level histograms.* **IEEE Trans. SMC 9(1):62–66.**
   The classic bimodal-histogram automatic threshold; our default binariser for MNDWI.
5. **Ji, L., Zhang, L., Wylie, B. (2009).** *Analysis of dynamic thresholds for the normalized difference water index.* **PE&RS 75(11):1307–1317.**
   Shows fixed NDWI thresholds are unreliable across seasons — motivates Otsu/adaptive choice.

## Theme 3 · Change detection

6. **Singh, A. (1989).** *Digital change detection techniques using remotely-sensed data.* **IJRS 10(6):989–1003.**
   Foundational review; covers image differencing, ratioing, PCA, CVA — we use all four.
7. **Bruzzone, L., Prieto, D.F. (2000).** *Automatic analysis of the difference image for unsupervised change detection.* **IEEE TGRS 38(3):1171–1182.**
   Automatic thresholding of a difference image via Expectation-Maximisation.

## Theme 4 · Deep learning for flood segmentation

8. **Ronneberger, O., Fischer, P., Brox, T. (2015).** *U-Net: Convolutional Networks for Biomedical Image Segmentation.* **MICCAI.**
   The benchmark encoder-decoder architecture; we use an SMP ResNet-34 U-Net.
9. **Bonafilia, D., Tellman, B., Anderson, T., Issenberg, E. (2020).** *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1.* **CVPRW.**
   Introduces our training benchmark and publishes baseline CNN numbers.
10. **Konapala, G., Kumar, S.V., Ahmad, S.K. (2021).** *Exploring Sentinel-1 and Sentinel-2 diversity for flood inundation mapping using deep learning.* **ISPRS J. 180:163–173.**
    Optical + SAR fusion with CNNs; motivates our Phase 7 Sentinel-1 cross-check.

## Theme 5 · Kerala 2018 event-specific

11. **Sudheer, K.P. et al. (2019).** *Role of dams on the floods of August 2018 in Periyar River Basin, Kerala.* **Current Science 116(5):780–794.**
    Provides event hydrology context — useful for discussing why the post-window (Aug 19–25) captures the flood peak.
12. **UNOSAT (2018).** *Flood Waters over Kerala State, India, as of 22 August 2018.* Product #2728.
    Our ground-truth reference.

## Theme 6 · SAR-based flood mapping (Phase 7 / bonus)

13. **Martinis, S., Twele, A., Voigt, S. (2009).** *Towards operational near real-time flood detection using a split-based automatic thresholding procedure on high-resolution TerraSAR-X data.* **NHESS 9:303–314.**
    Automatic SAR threshold; template for our VV-log Otsu approach.
14. **Chini, M., Hostache, R., Giustarini, L., Matgen, P. (2017).** *A hierarchical split-based approach for parametric thresholding of SAR images: flood inundation as a test case.* **IEEE TGRS 55(12):6975–6988.**
    Robust alternative when Otsu fails on unimodal SAR histograms.

---

## Gap analysis

- Most Kerala 2018 studies use either **optical-only** or **SAR-only** methods. We contribute an end-to-end pipeline with classical + DL + SAR cross-check.
- Sen1Floods11 papers rarely benchmark against a **classical baseline** — our ablation fills that gap.
- Few publicly shipped web apps exist for this workflow — our Streamlit deliverable fills that operational gap.

---

*To be expanded / revised by D during Phase 1; final version cited in IEEE format in the report.*
