# PHASE-WATER | Audited Scientific Results (Phase 4C)

**Status:** Engineering-verified exploratory classification. This document does not report independently validated flood extent, classification accuracy, precision, recall, IoU, F1, confidence intervals, or improvement from fusion.

## Data and common analysis support

Four NISAR L2 GCOV products, including co-acquired L/S pairs from 19 August and 24 September 2026, were compared over a common native 10 m EPSG:32645 grid. The S-band grid is shifted 72 columns east and 288 rows south relative to the L-band grid; exact integer alignment avoids spatial resampling. All three methods share the identical four-scene valid support.

**Jointly valid support:** 576,601,215 pixels = 57,660.12 km². The full acquisition footprint was processed, not the smaller catalogue-discovery box.

## Four-class temporal comparisons (provisional, km²)

| Method | Stable non-water | Stable water | Classified gain | Classified loss | Net classified change |
|---|---:|---:|---:|---:|---:|
| L-only | 32,343.73 | 10,957.94 | 2,981.16 | 11,377.30 | -8,396.14 |
| S-only | 33,980.61 | 6,607.77 | 7,113.72 | 9,958.02 | -2,844.30 |
| Equal-weight L+S | 35,082.11 | 8,325.59 | 3,614.85 | 10,637.58 | -7,022.73 |

**L-versus-S four-class disagreement:** 232,738,230 pixels (40.36%) on shared valid support.
Per-date L/S provisional binary disagreement was 26.30% in August and 22.13% in September.

## Optical screening—not independent accuracy validation

The original sample contains 400 sites. 116 had candidate optical observations reviewed by AI; 83 yielded interpretable optical-date screening labels (of which 49 were within 24 hours and 58 within 72 hours). 5 are provisionally water-like and 78 non-water-like.
**Confirmed SAR-time reference labels: 0.** Cloud-screened observations are availability-selected and temporally mismatched, often with 20–30 m optical spatial support. They must not be substituted into the official reference field or reported as scene-wide accuracy.

## Radiometric / processing uncertainty

September S-band was processed as Urgent Response (UR); August S-band is PR. Radiometric equivalence is not independently established.
A spatial diagnostic of 628,336 common valid samples found S-HV values below −40 dB at 0.00% (August) and 39.84% (September); corresponding p10 values were -19.91 and -52.25 dB. This is diagnostic sampling, not an unbiased full-scene tail-prevalence estimate or a sensor calibration.

## What the study can and cannot conclude

- Supported: reproducible same-pass dual-frequency data alignment, full-footprint comparative class maps, explicit frequency disagreements and transparent sensitivity diagnostics.
- Not supported: L/S fusion superiority, true flood/river-water area or change, physical attribution of S-HV anomalies, or independent classification performance metrics.
- Interpret the L/S disagreements as **method sensitivity** rather than proof of accuracy differences. The 50:50 fusion score is not a calibrated posterior probability.

## Suggested figure captions

**Figure 1.** Spatial distributions of provisional surface-water classifications from NISAR L-band, S-band and equal-weight fusion for August and September 2026, on common four-scene valid support (10 m, EPSG:32645).
**Figure 2.** Four-class temporal classification transitions from August to September 2026: stable non-water, stable water, newly classified water and classified water loss. September S-band UR processing limits direct radiometric interpretability.
**Figure 3.** Locations of L/S disagreement in the four-class change maps on the identical jointly valid footprint. Disagreement is not an independent classification error.

## Provenance

Results calculated from outputs/phase3g/phase3g_report.json; August optical diagnostic from outputs/phase4b/phase4b_AI_screened_400.csv and phase4b_AI_screening_diagnostic_report.json. Optional S-HV diagnostic from outputs/phase3h/phase3h_report.json.
