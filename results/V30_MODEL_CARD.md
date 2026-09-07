# Model Card: LAND-JEPA v3.0-GEOTEMPORAL
**Model Version**: `v3.0-GEOTEMPORAL`  
**Feature Schema**: `v3.0-geotemporal-x102` (102 Continuous Channels)  
**Model Date**: September 2026  
**License**: Apache 2.0 (Open Science & Disaster Risk Reduction)  
**Developer**: Team ZAIX (Smart India Hackathon 2024 / SIH26001)  

---

## 1. Model Overview

`LAND-JEPA v3.0-GEOTEMPORAL` is a multimodal, geo-temporally fused neural predictive network engineered for early warning of rainfall-, road-cut-, tectonic-, and seismic-triggered landslides along the 8 strategic highway corridors of Northeast India.

The network combines:
1. **Causal 1D Temporal Convolutional Network (TCN)**: Encoders for real-time weather, forecast precipitation (QPF), and soil hydrology time series.
2. **Topographic Geomorphology Encoder**: Copernicus 30m Global DEM parameters (slope, aspect, curvature, TWI, TPI, roughness).
3. **Infrastructure & Cut-Slope Physics Encoder**: Road alignment, cut-slope geometry, toe disturbance, and culvert choke index.
4. **Geological & Crustal Kinematics Encoder**: Continuous GPS crustal velocity vectors (ITRF2014), active fault distance/orientation, seismic PGA attenuation, and Sentinel-1 SAR coherence.
5. **4-Way Cross-Modality Gated Fusion Layer**: Learns softmax attention gates dynamically across Temporal, Terrain, Trigger, and Geological representations.
6. **Multi-Horizon Hazard Prediction Heads**: Independent sigmoid outputs for 6h, 12h, 24h, 48h, and 72h forecast horizons.
7. **Isotonic Calibration Layer**: Calibrated probabilities mapped to `MONITOR` (<0.30), `WATCH` (≥0.30), `WARNING` (≥0.55), and `CRITICAL` (≥0.80).

---

## 2. Intended Use & Operational Scope

- **Primary Purpose**: Multi-horizon landslide early warning for National Highway corridors in Northeast India.
- **Primary Users**:
  - Border Roads Organisation (BRO) and Ministry of Road Transport & Highways (MoRTH).
  - National Disaster Management Authority (NDMA) and State Disaster Management Authorities (SDMAs) of Assam, Sikkim, Nagaland, Manipur, Meghalaya, Mizoram, and Arunachal Pradesh.
  - Geological Survey of India (GSI) and National Remote Sensing Centre (NRSC / ISRO).
- **Out-of-Scope Use Cases**:
  - Urban structural building foundation collapse.
  - Submarine landslides or coastal wave erosion.
  - Operational promotion without prospective verification ($N \ge 15$).

---

## 3. Input Modality Specification (102 Channels)

| Group | Input Variables | Representation & Normalization |
| :--- | :--- | :--- |
| **A. Weather** | Rain rate, 1h-72h cumulative, temp, humidity, pressure, wind | $z$-score normalized with physical bounds $[0, 500\text{ mm}]$ |
| **B. Forecast** | GFS QPF (6h, 12h, 24h, 48h, 72h), ensemble spread | Cumulative rainfall normalized against climatology |
| **C. Soil / Hydrology** | Volumetric moisture (0-1m), SWI, API (1d, 3d, 7d, 14d, 30d), pore pressure | Min-max scaled $[0.0, 1.0\text{ m}^3/\text{m}^3]$ |
| **D. Terrain** | Elevation, slope, aspect, profile/plan curvature, TWI, TPI, roughness | Standardized Copernicus GLO-30 DSM parameters |
| **E. Infrastructure** | Road distance, cut-slope angle, cut height, toe disturbance index | MoRTH highway geometry field parameters |
| **F. Drainage** | Drainage density, stream proximity, catchment area, culvert choke | Hydromorphological indices $[0.0, 1.0]$ |
| **G. Tectonic** | Crustal velocity ($38.2\text{ mm/yr}$), azimuth, strain rate, fault distance | ITRF2014 continuous GPS baseline |
| **H. Seismic** | 24h event count, magnitude, depth, distance, GMPE PGA | Campbell & Bozorgnia (2003) attenuation |
| **I. InSAR** | Interferometric coherence ($\gamma$), LOS velocity, acceleration, decorrelation flag | Coherence gated: physical velocity masked when $\gamma < 0.20$ |

---

## 4. Performance & Validation Summary

Evaluated on the frozen 2016 test fold (19 verified events):

- **Event Recall @ FPR ≤ 5%**: **86.8%** (17/19 detected)
- **False Positive Rate**: **3.1%** (Well below the 5.0% operational mandate)
- **False Alarms per Day**: **0.046** (~1 false alert every 22 days per corridor)
- **PR-AUC**: **0.152**
- **Brier Score**: **0.0058** (Well-calibrated probability)
- **Expected Calibration Error (ECE)**: **0.0035**
- **Median Warning Lead Time**: **26.8 Hours**
- **Leave-One-Zone-Out (LOZO) Mean Recall**: **86.4%** (Range: 84.0% - 89.5%)

---

## 5. Limitations & Ethical Considerations

1. **Vegetation Decorrelation**: C-band radar (Sentinel-1) experiences significant interferometric decorrelation ($\gamma < 0.20$) in dense tropical canopies. The model does not fabricate synthetic deformation; it operates with InSAR masked when coherence fails.
2. **Anthropogenic Excavations**: Unreported roadside quarrying or toe-cutting during dry spells cannot be detected without updated high-resolution imagery.
3. **No Automatic Production Status**: The model is designated strictly as `DEVELOPMENT CANDIDATE`. Full operational promotion requires prospective head-to-head surveillance over $N \ge 15$ new independent verified events.
