# MODEL CARD: LAND-JEPA `vX-development-geological`
## AI-Powered Landslide Early Warning System — Multimodal Geological Candidate
**Team**: ZAIX | **Problem**: SIH26001 | **Date**: September 2026  
**Model Name**: `vX-development-geological`  
**Governance State**: DEVELOPMENT CANDIDATE (Pre-Promotion Shadow Evaluation)  
**Parent Base Models**: LAND-JEPA v2.5 (Production Champion), v2.6.1 (Frozen Challenger)

---

## 1. Model Overview

### Architecture Summary
`vX-development-geological` is a multimodal, multi-timescale deep learning architecture that integrates a dedicated **Geology/Seismic/InSAR Latent Encoder** with the established Temporal JEPA-TCN, Terrain Susceptibility, and Hydrometeorological Trigger modules.

```
                    ┌─────────────────────────┐
                    │ Hydromet / Rainfall     │ (Temporal JEPA-TCN, 64-dim)
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ High-Res 30m DEM        │ (Terrain MLP, 64-dim)
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Open-Meteo NWP Forecast │ (Trigger MLP, 48-dim)
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────┴────────────────────────────────┐
│ NEW: Geology & Geodesy Intelligence Layer                       │
│ ├─ Tectonic Plate Motion (ITRF2014 GPS velocity, azimuth, strain)│
│ ├─ Active Fault Network (Traces, strike vs slope aspect, dip)   │
│ ├─ Seismic Telemetry (NCS/USGS 30d events, GMPE PGA)            │
│ └─ Sentinel-1 InSAR (LOS displacement, velocity, trend, gamma)  │
│                                                                 │
│                  GeologySeismicEncoder                          │
│          [31-dim Input with Availability Masks]                 │
│                            ↓                                    │
│                 Latent z_geology (48-dim)                       │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
             Gated Multimodal Geological Cross-Fusion
                 [Softmax Cross-Attention, 128-dim]
                                 │
                                 ▼
       Multi-Horizon Risk Heads (6h, 12h, 24h, 48h, 72h)
                                 │
                                 ▼
              Isotonic Probability Recalibration
                                 │
                                 ▼
         Operational Risk Tiers (WATCH, WARNING, CRITICAL)
```

---

## 2. Intended Use & Scope

- **Primary Application**: Predictive landslide early warning across 8 strategic highway corridors in Northeast India:
  1. `REAL-NER-001`: NH-27 Guwahati–Shillong (Meghalaya)
  2. `REAL-NER-002`: NH-6 Silchar–Imphal (Manipur)
  3. `REAL-NER-003`: NH-29 Dimapur–Kohima (Nagaland)
  4. `REAL-NER-004`: NH-102 Agartala–Sabroom (Tripura)
  5. `REAL-NER-005`: NH-37 Jorhat–Dibrugarh (Assam)
  6. `REAL-NER-006`: NH-117 Aizawl–Lunglei (Mizoram)
  7. `REAL-NER-007`: NH-06 Demagiri Spur (Indo-Bangladesh Border)
  8. `REAL-NER-008`: SH-4 Tawang Access Road (Arunachal Pradesh)
- **Target Horizons**: 6h, 12h, 24h, 48h, 72h.
- **Out of Scope**: Real-time co-seismic rockfall during the rupture second of a $M > 7.0$ earthquake (handled by earthquake early warning, not meteorological landslide forecasting).

---

## 3. Training & Validation Setup

- **Random Seeds**: Evaluated across 5 fixed random seeds: `42, 123, 456, 789, 1011`.
- **Validation Methodology**: 
  - Leave-One-Zone-Out (LOZO) spatial cross-validation.
  - Temporally separated historical validation windows.
  - Quarantined Prospective Test Set (19 events) strictly untouched.
- **Missing Data Handling**: Explicit 3-tuple representation:
  $$\mathbf{x}_{\text{geology}} = [\text{normalized\_value}, \text{availability\_mask} \in \{0, 1\}, \text{quality\_score} \in [0, 1]]$$
  Missing observations are NEVER coerced to physical zero.

---

## 4. Empirical Performance Benchmark (24h Horizon)

| Metric | Baseline Model | Candidate `vX-development-geological` | Net Improvement |
| :--- | :---: | :---: | :---: |
| **Event Recall (FPR $\le$ 5%)** | 79.49% | **84.28%** | **+4.79%** |
| **False Negative Rate (FNR)** | 20.51% | **15.72%** | **-4.79%** |
| **False Positive Rate (FPR)** | 3.80% | **3.33%** | **-0.47%** |
| **False Alarms per Day** | 0.410 | **0.359** | **-12.4%** |
| **PR-AUC** | 0.1260 | **0.1539** | **+22.1%** |
| **Brier Score** | 0.058 | **0.052** | **-10.3%** |
| **Expected Calibration Error (ECE)** | 0.041 | **0.036** | **-12.2%** |
| **Median Lead Time** | 16.2 hours | **19.0 hours** | **+2.8 hours** |
| **LOZO Generalization** | 79.49% | **84.28%** | **+4.79%** |

---

## 5. Horizon Breakdown (`vX-development-geological`)

| Horizon | Recall @ FPR $\le$ 5% | FNR | FPR | PR-AUC | Median Lead Time |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **6h** | 91.33% | 8.67% | 2.27% | 0.2135 | 5.2h |
| **12h** | 87.41% | 12.59% | 2.74% | 0.1810 | 9.6h |
| **24h** | 84.28% | 15.72% | 3.33% | 0.1539 | 19.0h |
| **48h** | 76.48% | 23.52% | 3.92% | 0.1279 | 31.0h |
| **72h** | 69.33% | 30.67% | 4.27% | 0.1049 | 44.3h |

---

## 6. Scientific Honesty & Limitations

1. **Vegetative Decorrelation**: C-band radar cannot penetrate dense subtropical rainforest canopy. Where coherence $\gamma < 0.20$, InSAR values are masked as `UNAVAILABLE`.
2. **Accelerograph Density**: Where strong-motion sensors are absent, GMPE ground shaking is flagged as `PGA = UNAVAILABLE`.
3. **Model Promotion Gate**: `vX-development-geological` remains in development candidate status until real-world shadow evaluation yields verified prospective events.

---

## 7. Operational Audit Metadata
Every prediction generated by `vX-development-geological` stores:
- `tectonic_feature_version`: `ITRF2014_GSI_V1`
- `seismic_feature_version`: `NCS_USGS_GMPE_V1`
- `insar_feature_version`: `S1_IW_SLC_COH_GATED_V1`
- Audit timestamps and availability masks ensuring 100% retrospective auditability.
