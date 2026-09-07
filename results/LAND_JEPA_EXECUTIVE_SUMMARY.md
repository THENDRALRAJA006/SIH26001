# LAND-JEPA: Executive Summary
## AI-Powered Landslide Prediction, Early Warning, and Prospective Validation Report

**Project**: LAND-JEPA (Smart India Hackathon SIH26001)  
**Team**: ZAIX  
**Geographic Domain**: Northeast India (NER) — 8 Monitored Highway Corridors  
**Document Classification**: Scientific & Operational Executive Summary  
**Date**: September 2026  
**Current Production Benchmark**: `v2.5-TRIGGER-AWARE-CHAMPION`  
**Current Prospective Challenger**: `v2.6.1-CHALLENGER`  

---

### 1. Problem Statement & Regional Context
Northeast India (NER) accounts for over 70% of India's annual landslide casualties and infrastructure disruption. The region is characterized by steep Himalayan geomorphology, high seismic vulnerability (Zone V, PGA $\ge 0.36g$), and severe seasonal monsoons with extreme convective cloudbursts. Conventional early warning methods rely on empirical rainfall intensity-duration thresholds (e.g., published IMD/GSI guidelines), which trigger high false-alarm rates ($> 0.10$ false alarms/day) and deliver minimal operational lead time (often $< 1.0$ to $6.0$ hours), leading to alarm fatigue, closed logistics corridors, and unmitigated disaster impacts.

---

### 2. The LAND-JEPA Solution
**LAND-JEPA** is an AI-powered geotechnical early-warning system developed by Team ZAIX. Unlike classical black-box models or single-variable precipitation thresholds, LAND-JEPA integrates self-supervised representation learning through a **Joint Embedding Predictive Architecture (JEPA)** with physics-conditioned multimodal fusion:
- **Self-Supervised Temporal Encoder**: Dilated causal Temporal Convolutional Networks (TCN) pre-trained on 406,080 hourly environmental sequences to learn continuous slope hydration and pore-pressure dynamics without label supervision.
- **Static Geomorphic Representation**: High-resolution 30m digital surface model derivatives (slope, aspect, curvature, Topographic Wetness Index, Topographic Position Index).
- **Multi-Trigger Physics Integration**: 86 environmental features spanning 8 trigger families including road-cut excavation, culvert choking, freeze-thaw cycles, and co-seismic shaking priors.
- **Forecast-Aware Multi-Horizon Heads**: 5 dedicated hazard prediction heads generating calibrated failure probabilities across 6h, 12h, 24h, 48h, and 72h lead times conditioned on Numerical Weather Prediction (NWP) ensemble uncertainty.

---

### 3. Scientific Data Provenance & Anti-Fabrication Guarantee
All model parameters and operational pipelines operate exclusively on authenticated, auditable scientific data sources:
1. **NASA Global Landslide Catalog (GLC v1.1) & ISRO Bhuvan**: 170 confirmed physical landslide events with verified spatial coordinates and timestamps (2011–2016).
2. **Copernicus DEM GLO-30**: 30m spatial geomorphology.
3. **ECMWF ERA5-Land Reanalysis**: 406,080 hourly atmospheric, soil moisture, and thermal records.
4. **Open-Meteo High-Resolution NWP & Ensemble Spread**: Real-time deterministic precipitation and 30-member forecast spread.
5. **Geological Survey of India (GSI) & USGS**: Seismotectonic fault maps and regional Peak Ground Acceleration records.
6. **OpenStreetMap & Border Roads Organisation (BRO)**: Highway alignments, culverts, and slope-cut buffers.

> [!IMPORTANT]
> **Scientific Integrity Guarantee**: Where instrumented ground sensor arrays (e.g., continuous in-situ GNSS prisms or high-frequency InSAR deformation) do not exist along historical corridors, they are explicitly tagged as `UNAVAILABLE`. No synthetic shortcuts, fake sensors, or fabricated data are used.

---

### 4. Operational Performance Summary

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PERFORMANCE SUMMARY MATRIX                                       │
├────────────────────────────────┬───────────────────────────────┬─────────────────────────────────┤
│ Metric                         │ v2.5 Production Champion      │ v2.6.1 Challenger               │
│                                │ (Trigger-Aware Benchmark)     │ (Multi-Season Minimax)          │
├────────────────────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ Evaluation Paradigm            │ HISTORICAL MULTI-SEASON       │ HISTORICAL MULTI-SEASON         │
│ Data Splits Evaluated          │ 2013, 2014, 2015 Monsoons     │ 2013, 2014, 2015 Monsoons       │
│ Event Recall @ WARNING         │ 78.9% (30/38 events)          │ 81.6% (31/38 events)            │
│ False Negative Rate (FNR)      │ 21.1%                         │ 18.4%                           │
│ False Positive Rate (FPR)      │ 3.69%                         │ 3.45%                           │
│ Operational False Alarms / Day │ 0.0532 fa/day                 │ 0.0425 fa/day (-20.1%)          │
│ PR-AUC (Sliding Window)        │ 0.1135                        │ 0.1285 (+13.2%)                 │
│ Brier Reliability Score        │ 0.0119                        │ 0.0098 (Superior)               │
│ Expected Calibration Error     │ 0.0049                        │ 0.0028 (Well-Calibrated)        │
│ Median Advance Warning Lead    │ 24.0 hours                    │ 25.2 hours (+1.2h)              │
├────────────────────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ Real Prospective Surveillance  │ 720h / 120 cycles (Sep 2026)  │ 720h / 120 cycles (Sep 2026)    │
│ New Verified Ground Events     │ 0 events                      │ 0 events                        │
│ Prospective Event Recall       │ UNDEFINED (Zero ground events)│ UNDEFINED (Zero ground events)  │
│ Operational Alert Episodes     │ 127 episodes                  │ 164 episodes (24h persisted)    │
│ Prospective Status             │ ACTIVE PRODUCTION BENCHMARK   │ FROZEN CHALLENGER (SHADOW MODE) │
└────────────────────────────────┴───────────────────────────────┴─────────────────────────────────┘
```

*Note: Historical multi-season metrics reflect offline backtesting on confirmed 2013–2015 monsoon disasters. In accordance with scientific reporting standards, historical validation is never conflated with prospective real-world accuracy.*

---

### 5. Prospective Shadow Surveillance Status
Between September 2026 and October 2026, LAND-JEPA operated in automated, immutable prospective shadow surveillance across all 8 Northeast India corridors:
- **Total Prospective Predictions Logged**: 11,520 corridor-hour prediction records across 120 forecast cycles.
- **Observed Disasters in Active Period**: **0 new independently verified landslide failures** occurred across the monitored corridor bounds during this observation window.
- **Scientific Verdict**: Per Section 8 of the Master Prospective Protocol, **Event Recall and Lead Time cannot be estimated from zero events**. The statistical verdict is officially recorded as **`INSUFFICIENT_EVIDENCE`**.
- **Operational Stability**: `v2.6.1` demonstrated complete resolution of the single-season threshold saturation that affected `v2.6`, maintaining disciplined alert persistence across peak monsoon sequences.

---

### 6. Operational Warning Tiers & Decision Governance
LAND-JEPA separates pure mathematical probability from human operational action through three calibrated warning tiers:

| Tier | Operational Threshold | FPR Ceiling | Action Protocols Triggered |
| :--- | :---: | :---: | :--- |
| **WATCH** | $P \ge 0.6531$ | $\le 10\%$ | Highway patrol frequency elevated to 15 min; culvert clearing crews alerted; SMS advisories to village councils. |
| **WARNING** | $P \ge 0.7724$ | $\le 5\%$ | Night travel bans for heavy transport; school closures in high-slope zones; NDRF staged at regional hubs. |
| **CRITICAL** | $P \ge 0.9550$ | $\le 1\%$ | Immediate highway closure (NH-29, NH-10, NH-27); evacuation of toe-adjacent settlements; incident command activation. |

---

### 7. Known Physical Failure Modes & Current Limitations
1. **Micro-Cloudbursts ($< 12\text{ km}$)**: Convective rain cells localized beneath the spatial resolution of global numerical weather models remain the primary cause of un-forecasted flash failures.
2. **Anthropogenic Road-Cut Excavation**: Sudden mechanical toe excavation by heavy construction machinery without antecedent rainfall represents an unresolved non-meteorological failure mode.
3. **Absence of Real-Time Slope Deformation Telemetry**: Due to dense sub-tropical vegetation, radar decorrelation prevents continuous C-band InSAR tracking, and physical ground GNSS arrays are absent along remote hill routes.
4. **Small Independent Event Sample Size**: While the historical catalog contains 170 confirmed incidents, the low frequency of catastrophic failures per individual corridor requires extended multi-season prospective surveillance before definitive promotion.

---

### 8. Final Institutional Recommendation
> **"LAND-JEPA demonstrates strong progress in forecast-aware landslide risk prediction, including improved historical multi-season validation (81.6% Event Recall @ FPR <= 3.45%, 25.2h Lead Time) and an operational real-time prospective surveillance architecture. However, generalized 90–95% prospective event recall has not yet been established due to sample size constraints during active surveillance. The system remains in active shadow evaluation, with v2.5 retained as the production benchmark and v2.6.1 maintained as the frozen prospective challenger."**
