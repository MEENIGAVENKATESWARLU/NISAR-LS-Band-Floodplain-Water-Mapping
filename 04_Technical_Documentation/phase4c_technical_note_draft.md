# PHASE-WATER | Technical Note Draft
## Physics-aware dual-frequency NISAR GCOV comparison for dynamic surface-water classification over the Bihar Ganga floodplain

**Scientific status:** Exploratory, not independently accuracy-validated.

### 1. Objective and study design

We investigated how NISAR L-band and S-band SAR observations differ in their provisional mapping of surface-water classes and temporal transitions across a heterogeneous alluvial floodplain. The test was designed around two co-acquired L/S pairs (19 August and 24 September 2026). The goal was to establish a reproducible common-footprint baseline, quantify disagreement and identify vulnerabilities to changes in sensor processing and radiometry. Improved accuracy was a hypothesis for future testing, not an observed result.

### 2. Data preparation and common-support protocol

All science inputs were NISAR Level-2 GCOV HDF5 products. L-band diagonal covariance terms HHHH and HVHV were taken from frequencyA at 10 m; the corresponding S-band terms were also read from frequencyA at 10 m. S-band additionally supplied the complex HHHV covariance, used only for physical-consistency diagnostics and not as an unfair extra predictor in the baseline comparison. The L-band frequencyB group is a distinct bandwidth/gridding group of the L-band product, not a separate S-band sensor.

Products were verified on the EPSG:32645 grid. Both S-band scenes use the same 10 m pixels as L-band but begin 288 rows and 72 columns into the L raster. Pixel-centre coordinates support exact integer-offset alignment without interpolation. All comparisons were restricted to 576,601,215 pixels (57,660.12 km²) valid in all four source images. The study processed the complete available S-band GCOV rectangle and did not use the smaller search bounding box to clip scientific products.

### 3. Exploratory classification and change detection

A first-pass unsupervised HH/HV classifier used August-derived frequency-specific thresholds fixed across both dates, with logistic transition scores. Common-polarization L-only and S-only baselines were contrasted with an equal-weight fusion of their scores. The fusion values are ordinal classification scores, not calibrated probabilities. Each dated binary map yielded a four-class transition map: stable non-water, stable water, newly classified water and classified water loss. Native-resolution area calculations used 0.0001 km² for each valid 10 m pixel. The same four-scene mask was used for every method.

### 4. Results

The L-only model classified 2,981.16 km² as gain and 11,377.30 km² as loss (net -8,396.14 km²). The S-only model classified 7,113.72 km² as gain and 9,958.02 km² as loss (net -2,844.30 km²). Equal-weight fusion classified 3,614.85 km² as gain and 10,637.58 km² as loss (net -7,022.73 km²). These figures represent changes in model-assigned class, not independently established changes in inundation extent.

The two single-frequency methods disagreed on 232,738,230 four-class labels (40.36% of four-scene valid pixels). Date-specific binary water masks disagreed on 26.30% (August) and 22.13% (September) of common support. Engineering checks confirmed the contingency-table totals and water-area accounting.

### 5. Independent optical evidence and limitations

The original 400-point evaluation design was stratified by provisional change class. Cloud-screened optical image chips were available for 115 of 116 August candidates. AI-assisted, predictor-blinded visual screening yielded 83 interpretable optical-date classifications, including 49 within 24 hours of NISAR. Unresolved cases were left without labels. Only 5 interpretations were water-like, so this subset is strongly class-imbalanced. **No independently verified NISAR-time reference labels were established**. Optical screening was used descriptively only, not as an accuracy assessment; cloud selection, time offsets, and 20–30 m optical measurement support may bias comparison with 10 m SAR pixels.

### 6. Processing uncertainty and sensitivity

August S-band was a PR product, whereas September S-band was UR. The S-band HV lower-tail anomaly therefore matters for temporal interpretation. Matched-grid numerical tests revealed no reported complex covariance positive-semidefinite violations in the preflight sample, but that is not a full product calibration test.
In spatial diagnostics (628,336 jointly valid samples), 39.84% of September S-HV values were below −40 dB, versus 0.00% in August. A fixed dB adjustment is not a defensible correction without independent noise/signal and product-processing evidence. These statistics are diagnostic only, not full-scene prevalence estimates.

### 7. Conclusions and next scientific steps

The study establishes a reproducible multi-date NISAR L/S GCOV processing and comparison pipeline. Its strongest observation is the large discrepancy in provisional temporal water-class assignments across two co-acquired frequencies, compounded by a pronounced September S-HV lower tail. The method is suitable as a transparent baseline for testing physics-guided fusion, but it does not demonstrate improved accuracy or verified hydrological change. Future work requires temporally matched, independently adjudicated reference water labels; validation of mixed land/water boundaries; a source-level UR/PR radiometric comparability assessment; and ablations of cross-polarized/covariance features under controlled equal-support evaluation.

### Reproducibility and source attribution

NISAR L2 GCOV sources: NASA Earthdata/ASF for L-band and ISRO NRSC Bhoonidhi for S-band. Computed results are traceable to the saved Phase 3G JSON and GeoTIFF outputs and Phase 3H/3J quality diagnostics. The original raw HDF5 files remain unmodified. Final figure captions must retain the exploratory/uncertain status even if warnings are omitted from the graphic layout.
