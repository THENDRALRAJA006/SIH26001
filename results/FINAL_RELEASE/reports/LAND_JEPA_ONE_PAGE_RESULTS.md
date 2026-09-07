# LAND-JEPA: Executive One-Page Results Summary
## AI-Powered Landslide Risk Intelligence & Operational Early Warning

**Smart India Hackathon 2026** — Problem ID: `SIH26001`  
**Team**: ZAIX | **Domain**: Northeast India (NER) — 8 Strategic Transport Corridors  
**Document Classification**: Institutional Executive Briefing  
**Release Date**: September 2026  
**Operational Status**: `v2.5-TRIGGER-AWARE-CHAMPION` (Active Production) | `v2.6.1-CHALLENGER` (Frozen Shadow Mode)

---

### 1. The Regional Disaster Problem
Northeast India's strategic highway corridors (NH-27, NH-6, SH-4) suffer hundreds of catastrophic monsoonal landslides annually, causing fatalities, isolated border districts, and logistics paralysis. Conventional early-warning systems rely on empirical rainfall intensity-duration ($I$-$D$) thresholds, which suffer from two fatal operational flaws:
1. **Severe False Alarm Fatigue**: Firing false alarms on normal monsoon days ($\text{FPR} > 5.0\%$, $>0.10$ false alarms/day).
2. **Inadequate Action Lead Times**: Detecting failure only hours or minutes before collapse ($<1.0\text{h}$ to $6.0\text{h}$), preventing proactive highway management.

### 2. The LAND-JEPA Innovation
Unlike black-box regressors or single-parameter rainfall lines, **LAND-JEPA** integrates self-supervised representation learning with geotechnical physics:
- **Latent Embedding Prediction**: Pre-trained on 406,080 hourly environmental sequences using a Joint Embedding Predictive Architecture (JEPA) and causal Temporal Convolutional Networks (TCN) to model continuous soil hydration memory without future data leakage.
- **Multimodal Physics Fusion**: Gated fusion of 86 geotechnical indicators across 8 physical trigger families including convective rainfall rate, multi-layer soil saturation, 30m digital elevation derivatives, road-cut excavation geometries, culvert scour, and seismic shaking priors.
- **Multi-Season Minimax Thresholding**: Resolves single-season overfitting by optimizing operational alert boundaries across multiple monsoon regimes with 24-hour advisory persistence grouping.

### 3. Model Architecture & Prediction Pipeline
```
[Observations + 72h NWP + DEM + Hydrology + Road-Cut]
                          │
                          ▼
        [Physics Engine: 86 Variables in 8 Trigger Families]
                          │
                          ▼
            [Temporal JEPA-TCN Sequence Encoder (64d)]
                          │
                          ▼
                [Gated Multimodal Fusion (128d)]
                          │
                          ▼
       [Multi-Horizon Direct Heads: 6h, 12h, 24h, 48h, 72h]
                          │
                          ▼
               [Isotonic Probability Calibration]
                          │
                          ▼
       [Minimax Operational Warning Tiers: WATCH / WARNING / CRITICAL]
```

### 4. Authoritative Scientific Data Provenance
All inputs are authenticated open scientific data feeds. Synthetic shortcuts and fabricated sensors are prohibited:
- **Disaster Catalog**: NASA Global Landslide Catalog (GLC v1.1) & ISRO Bhuvan (170 confirmed physical events).
- **Atmospheric & Hydrologic Reanalysis**: ECMWF ERA5-Land (406,080 hourly records, 2011–2016).
- **Numerical Weather Prediction (NWP)**: Open-Meteo 72h deterministic QPF (11km) + 30-member ECMWF-EPS ensemble spread.
- **Geomorphology & Infrastructure**: Copernicus DEM GLO-30 (30m) + OSM/BRO vector highway alignments and cut buffers.
- *Explicit Data Gap*: Continuous in-situ GNSS displacement sensors are documented as `UNAVAILABLE` along NER corridors.

### 5. Multi-Horizon Prediction Capacities
Five dedicated prediction heads output calibrated disaster probabilities across distinct operational planning windows:
- **6 Hours**: Tactical response, traffic diversion, emergency crew alert (8.8% FNR).
- **12 Hours**: Freight routing restrictions, staging equipment at corridor bottlenecks (13.2% FNR).
- **24 Hours**: Primary civil defense evacuation and transit advisory tier (18.4% FNR).
- **48 Hours**: Regional logistics pre-positioning, multi-agency preparedness (31.6% FNR).
- **72 Hours**: Strategic transport corridor staging and heavy equipment standby (47.4% FNR).

---

### 6. Core Performance Summary: Historical vs. Prospective

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PERFORMANCE SUMMARY MATRIX                                       │
├────────────────────────────────┬───────────────────────────────┬─────────────────────────────────┤
│ Metric / Property              │ v2.5 Production Champion      │ v2.6.1 Challenger               │
│                                │ (Trigger-Aware Benchmark)     │ (Multi-Season Minimax)          │
├────────────────────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ Evaluation Paradigm            │ HISTORICAL VALIDATION         │ HISTORICAL VALIDATION           │
│ Historical Validation Period   │ 2013, 2014, 2015 Monsoons     │ 2013, 2014, 2015 Monsoons       │
│ Event Recall @ WARNING         │ 78.9% (30 / 38 events)        │ 81.6% (31 / 38 events)          │
│ False Negative Rate (FNR)      │ 21.1%                         │ 18.4%                           │
│ False Positive Rate (FPR)      │ 3.69%                         │ 3.45%                           │
│ Daily False Alarms / Corridor  │ 0.0532 fa/day                 │ 0.0425 fa/day (-20.1%)          │
│ Precision-Recall AUC (PR-AUC)  │ 0.1135                        │ 0.1285 (+13.2%)                 │
│ Brier Reliability Score        │ 0.0119                        │ 0.0098 (Sharply Calibrated)     │
│ Expected Calibration Error     │ 0.0049                        │ 0.0028 (Well-Calibrated)        │
│ Median Advance Warning Lead    │ 24.0 hours                    │ 25.2 hours (+1.2h)              │
│ Verified Frozen Thresholds     │ WATCH=0.120, WARN=0.198, CRIT=0.450 │ WATCH=0.6531, WARN=0.7724, CRIT=0.9550│
├────────────────────────────────┼───────────────────────────────┼─────────────────────────────────┤
│ Prospective Surveillance Rec.  │ 11,520 records (720h / 120 cyc)│ 11,520 records (720h / 120 cyc)│
│ New Confirmed Ground Events    │ 0 events                      │ 0 events                        │
│ Prospective Event Recall       │ UNDEFINED (0 ground events)   │ UNDEFINED (0 ground events)     │
│ Prospective Lead Time          │ UNDEFINED (0 ground events)   │ UNDEFINED (0 ground events)     │
│ Prospective Status             │ INSUFFICIENT EVIDENCE         │ INSUFFICIENT EVIDENCE           │
│ Institutional Deployment Role  │ ACTIVE PRODUCTION BENCHMARK   │ FROZEN CHALLENGER (SHADOW MODE) │
└────────────────────────────────┴───────────────────────────────┴─────────────────────────────────┘
```

---

### 7. Primary Limitations & Governance Rules
1. **Sample Size Constraint**: Verified historical slope disaster catalog comprises 170 events. Prospective performance remains unproven on new physical events.
2. **Forecast Spatial Resolution**: Global NWP at $\approx 11\text{km}$ cannot anticipate micro-scale convective cloudburst cells ($<3\text{km}$).
3. **Non-Meteorological Failures**: Mechanical slope toe removal by road excavation contractors cannot be detected by meteorological algorithms.
4. **Promotion Gate**: `v2.6.1-CHALLENGER` is held in shadow quarantine until $\ge 15$ independently verified physical disaster events are captured in prospective monitoring.

### 8. Institutional Scientific Verdict
> **"LAND-JEPA demonstrates strong progress in forecast-aware landslide risk prediction, including improved historical multi-season validation (81.6% Event Recall @ FPR ≤ 3.45%, 25.2h Lead Time) and a functioning real-time prospective surveillance architecture. However, generalized 90–95% prospective event recall has not yet been established. The system remains under controlled shadow evaluation."**
