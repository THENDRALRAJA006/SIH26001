# Model Card: LAND-JEPA Multi-Horizon Landslide Risk Intelligence System
## Version: v2.5-TRIGGER-AWARE-CHAMPION (Active Production) & v2.6.1-CHALLENGER (Frozen Prospective)

**Smart India Hackathon 2026** — Problem ID: `SIH26001`  
**Developer**: Team ZAIX  
**Primary Application**: Regional Geotechnical Early Warning & Disaster Preparedness  
**Target Domain**: Northeast India (NER) — 8 Monitored Transport Corridors  
**Model License**: Apache 2.0 (Open-Access Research & Humanitarian Use)  
**Card Status**: Production & Operational Release — September 2026  

---

## 1. Model Details

### 1.1 Overview
The **LAND-JEPA** (Joint Embedding Predictive Architecture for Landslide Risk) model is a self-supervised and physics-fused deep learning system engineered to predict the onset and probability of slope instability across multiple advance forecast horizons ($6\text{h}, 12\text{h}, 24\text{h}, 48\text{h}, 72\text{h}$). It couples a causal Temporal Convolutional Network (TCN) pre-trained with self-supervised latent prediction with high-resolution geomorphology, geotechnical hydro-mechanics, and Numerical Weather Prediction (NWP) forecast uncertainty.

### 1.2 Model Lineage & Variants
- **`v2.5-TRIGGER-AWARE-CHAMPION`** (*Production Benchmark*): 68-feature trigger-aware architecture utilizing temperature scaling, operating with an operational WARNING threshold of $P_{\text{warn}} = 0.1980$.
- **`v2.6-RAW-SINGLE-SEASON`** (*Superseded*): 86-feature multi-trigger model that failed operational threshold robustness due to single-season calibration overfitting on the 2015 monsoon ($P_{\text{warn}} = 0.0929$, causing 100% warning saturation during peak 2026 monsoon).
- **`v2.6.1-CHALLENGER`** (*Current Frozen Challenger*): Multi-season minimax robust model utilizing isotonic regression calibration ($P_{\text{warn}} = 0.7724$) coupled with a 24-hour storm advisory persistence grouping algorithm.

### 1.3 Architecture Specifications
```
Input Sequences:
- 168h Hourly Weather/Hydrology [T=168, D_temp=18]
- Static Geomorphology [D_terr=8]
- Multi-Trigger Indicators [D_trig=60]
       │
       ├──► [Temporal JEPA-TCN Encoder] ──► Temporal Embedding z_t ∈ R^64
       ├──► [Terrain MLP Encoder]       ──► Spatial Embedding z_s ∈ R^32
       └──► [Trigger Physics Encoder]   ──► Trigger Embedding z_p ∈ R^32
                     │
                     ▼
           [Gated Multimodal Fusion]
                     │
         Latent State h ∈ R^128
                     │
    ┌────────────────┼────────────────┬────────────────┬────────────────┐
    ▼                ▼                ▼                ▼                ▼
 [Head 6h]      [Head 12h]       [Head 24h]       [Head 48h]       [Head 72h]
    │                │                │                │                │
 P(Y|6h)          P(Y|12h)         P(Y|24h)         P(Y|48h)         P(Y|72h)
    │                │                │                │                │
    └────────────────┴────────────────┼────────────────┴────────────────┘
                                      ▼
                      [Isotonic Probability Calibration]
                                      ▼
                      Calibrated Risk Probabilities P_cal
                                      ▼
                     [Minimax Operational Thresholding]
                                      ▼
                         WATCH / WARNING / CRITICAL
```

---

## 2. Intended Use & Scope

### 2.1 Primary Intended Applications
- **Highway Early Warning**: Providing state disaster management authorities (SDMA), Border Roads Organisation (BRO), and National Highways and Infrastructure Development Corporation (NHIDCL) with 24 to 48 hours of advance operational lead time before slope failure.
- **Logistics & Emergency Prepositioning**: Staging National Disaster Response Force (NDRF) personnel, heavy excavation equipment, and emergency medical supplies prior to monsoon highway severed events.
- **Public Preparedness**: Powering the zero-friction anonymous Citizen Portal to deliver advisory notifications and crowd-sourced photographic telemetry to hill communities.

### 2.2 Out-of-Scope & Prohibited Uses
- **Individual Site Geotechnical Engineering**: The model is a regional early-warning tool ($30\text{m}$ to $1\text{km}$ cell resolution) and MUST NOT replace site-specific boreholes, inclinometer surveys, or structural retaining wall engineering.
- **Automated Physical Interventions**: Model predictions MUST NOT automatically actuate traffic blockades without confirmation by an authorized Field Officer.
- **Instantaneous Co-Seismic Liquefaction**: While the model incorporates seismic zone factors and fault proximity, it is not an earthquake early-warning system and cannot predict sudden dynamic rupture within seconds of an epicenter event.

---

## 3. Data Lineage & Environmental Census

### 3.1 Scientific Data Sources
The model operates exclusively on auditable, open-science datasets:
1. **NASA Global Landslide Catalog (GLC v1.1) & ISRO Bhuvan**: 170 confirmed, deduplicated landslide events from 2011 to 2016 with complete provenance.
2. **Copernicus DEM GLO-30**: Global 30m Digital Elevation Model providing elevation, slope, aspect, curvature, Topographic Wetness Index (TWI), and Topographic Position Index (TPI).
3. **ECMWF ERA5-Land**: 406,080 hourly records covering surface runoff, temperature, relative humidity, volumetric soil moisture (layers 1–4), and surface pressure.
4. **Open-Meteo Weather API**: Operational real-time meteorological observations and 72-hour forecast precipitation (QPF) with 30-member ensemble spread.
5. **Geological Survey of India (Bhukosh) & USGS**: Seismotectonic Atlas of India, Main Boundary Thrust (MBT) and Main Central Thrust (MCT) spatial vectors, and historical earthquake PGA records.
6. **OpenStreetMap & BRO Records**: Vector alignments of NH-27, NH-6, NH-29, NH-102, NH-37, NH-117, NH-06, and SH-4.

### 3.2 Feature Space Architecture (86 Variables across 8 Families)
- **`CONVECTIVE_PRECIPITATION` (15 features)**: Sub-hourly peak intensity proxy, spatial rain gradients ($1\text{km}, 5\text{km}, 10\text{km}, 25\text{km}$), temporal accumulations ($1\text{h}, 3\text{h}, 6\text{h}, 12\text{h}$), QPF burst ratio, spatial divergence Laplacian.
- **`HYDROLOGY_SOIL_WETNESS` (9 features)**: Volumetric soil moisture, saturation ratio, 5-day recursive Soil Water Index (SWI), Antecedent Precipitation Index (API $\alpha=0.92$), infiltration/runoff proxies, pore-pressure ratio.
- **`TERRAIN_GEOMORPHOLOGY` (8 features)**: Slope angle, aspect, plan/profile curvature, terrain relief, TWI, TPI, slope variability, infinite slope safety factor reciprocal.
- **`ROAD_CUT_EXCAVATION` (9 features)**: Distance to road centerline, cut proximity score, slope differential ($\theta_{\text{cut}} - \theta_{\text{natural}}$), toe excavation risk index, vertical face height.
- **`DRAINAGE_CULVERT_SCOUR` (7 features)**: Culvert proximity, choke risk index (upstream area $\times$ runoff), stream scour susceptibility, drainage network density.
- **`FREEZE_THAW_THERMAL` (8 features)**: Surface temperature, zero-crossing counter, hours below $0^\circ\text{C}$, thermal cycle duration, freeze-thaw transition rate.
- **`SEISMIC_COSEISMIC_PRIOR` (8 features)**: Regional seismic zone coefficient (Zone V = 0.36), estimated Peak Ground Acceleration (PGA), fault distance, Newmark critical acceleration proxy ($a_c / \text{PGA}$), $\text{PGA} \times \text{SWI}$ interaction.
- **`FORECAST_UNCERTAINTY` (4 features)**: Ensemble forecast mean precipitation, ensemble spread, normalized uncertainty index, lead time.

> [!NOTE]
> **Sensor Availability Declaration**: Continuous in-situ highway GNSS displacement arrays and pre-2014 SAR interferometry do not exist across the corridors and are explicitly registered as `UNAVAILABLE`.

---

## 4. Performance & Validation Evidence

### 4.1 Historical Multi-Season Validation (2013, 2014, 2015 Monsoons)

| Metric | v2.5-TRIGGER-AWARE-CHAMPION | v2.6.1-CHALLENGER | Baseline GSI/IMD Threshold |
| :--- | :---: | :---: | :---: |
| **Event Recall @ WARNING** | **78.9%** (30/38) | **81.6%** (31/38) | 22.2% (8/36) |
| **False Negative Rate (FNR)** | 21.1% | **18.4%** | 77.8% |
| **False Positive Rate (FPR)** | 3.69% | **3.45%** | 5.00% |
| **False Alarms / Day** | 0.0532 fa/day | **0.0425 fa/day** | 0.1020 fa/day |
| **PR-AUC (Sliding Window)** | 0.1135 | **0.1285** | 0.0797 |
| **Brier Score** | 0.0119 | **0.0098** | 0.0126 |
| **Expected Calibration Error (ECE)** | 0.0049 | **0.0028** | 0.0480 |
| **Median Advance Warning Lead Time** | 24.0 hours | **25.2 hours** | 20.5 hours |

*Labeling Requirement: All metrics in Section 4.1 represent OFFLINE HISTORICAL MULTI-SEASON BACKTESTING and must not be cited as prospective real-world accuracy.*

### 4.2 Prospective Real-World Surveillance (September – October 2026)
- **Surveillance Protocol**: Real-time automated ingestion of live Open-Meteo forecasts and hourly risk inference across all 8 NER corridors with immutable append-only ledger logging.
- **Duration**: 30 days (720 hours, 120 cycles, 240 corridor-days).
- **Predictions Logged**: 11,520 corridor-hour prediction records.
- **Independently Verified Disaster Events in Window**: **0 events**.
- **Prospective Metric Verdict**:
  - **Prospective Event Recall**: **`UNDEFINED`** (Division by zero ground events).
  - **Prospective Lead Time**: **`UNDEFINED`**.
  - **Overall Status**: **`INSUFFICIENT_EVIDENCE`**.
- **Operational Findings**: `v2.6.1` generated 164 consolidated 24h warning episodes across 240 corridor-days, eliminating the 100% warning saturation exhibited by `v2.6`.

---

## 5. Decision Governance & Operational Tiers

Model-generated probabilities are mapped into three actionable operational tiers using minimax thresholds fitted across multi-season validation folds:

```
Probability Scale:  0.0 ─────── [0.6531] ─────── [0.7724] ─────── [0.9550] ─────── 1.0
Operational Tier:       NOMINAL         WATCH           WARNING         CRITICAL
False Positive Cap:      N/A            FPR ≤ 10%       FPR ≤ 5%        FPR ≤ 1%
```

- **WATCH ($P \ge 0.6531$, FPR $\le 10\%$)**: Elevate sensor polling; notify municipal culvert maintenance teams; broadcast weather advisory to village heads.
- **WARNING ($P \ge 0.7724$, FPR $\le 5\%$)**: Preposition heavy machinery at vulnerable bends; enforce night-travel restrictions; stage NDRF quick-response teams.
- **CRITICAL ($P \ge 0.9550$, FPR $\le 1\%$)**: Implement immediate corridor closure; divert highway transit; evacuate vulnerable settlements along toe-slope margins.

---

## 6. Known Failure Modes & Scientific Limitations

1. **Sub-Grid Cloudbursts**: Convective cloudbursts delivering $>80\text{ mm/h}$ with horizontal diameters $<10\text{ km}$ cannot be resolved by regional NWP grids ($12\text{ km}$ resolution), leading to unavoidable missed events without local radar or micro-barometer coverage.
2. **Non-Meteorological Toe Excavation**: Mechanized road cutting and quarrying that over-steepens hillside slopes beyond the internal angle of friction ($\phi$) triggers structural failure during dry periods with zero atmospheric signature.
3. **Culvert Blockage Runaway**: Micro-topographic debris jams that choke road drainage pipes cause localized hydraulic scouring that regional DEM models cannot resolve.
4. **Sample Size Constraints**: The total number of confirmed historical landslide events across 8 corridors is 170. While sufficient for statistical convergence across multi-season cross-validation, ongoing multi-year prospective surveillance is mandatory to establish empirical prospective recall.

---

## 7. Model Status & Promotion Criteria

- **`v2.5-TRIGGER-AWARE-CHAMPION`**: **ACTIVE PRODUCTION BENCHMARK**. Serves all operational API endpoints and citizen/officer dashboards.
- **`v2.6.1-CHALLENGER`**: **FROZEN PROSPECTIVE CHALLENGER**. Operating concurrently in shadow surveillance.
- **Promotion Threshold**: `v2.6.1` will be eligible for promotion to production champion ONLY after accumulating a minimum of **15 independently verified ground disaster events** during prospective surveillance that confirm equal or superior sensitivity ($\ge 78.9\%$) without exceeding the false-positive ceiling ($\text{FPR} \le 5\%$).
