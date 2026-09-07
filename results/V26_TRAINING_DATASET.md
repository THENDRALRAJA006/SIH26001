# LAND-JEPA v2.6: Training Dataset Specification & Lineage

**Project**: LAND-JEPA — AI-Based Landslide Early Warning and Risk Monitoring  
**Problem**: SIH26001 | **Team**: ZAIX | **Region**: Northeast India (8 Monitored Corridors)  
**Dataset Version**: `v2.6-AUTHENTIC-MULTI-TRIGGER`  
**Creation Date**: 2026-09-06  
**Status**: Pre-Execution Data Coverage Gate Passed  

---

## 1. Executive Summary & Core Objective

The **LAND-JEPA v2.6 Training Dataset** directly incorporates physical trigger features designed to resolve the four residual failure mechanisms discovered during the 90-day prospective shadow surveillance test:
1. **Road-Cut Toe Excavation & Slope Over-Steepening**: Anthropogenic slope excavation removing toe support without meteorological precursor.
2. **Co-Seismic Fault Slip & Seismic Ground Motion**: Earthquake PGA ground shaking and regional fault proximity triggering joint slip under dry or moist antecedent soil.
3. **Culvert Choking & Road Drainage Scour**: Hydraulic conveyance blockage causing localized scouring and highway embankment breaches.
4. **Localized Convective Cloudburst & Microbursts**: Extreme sub-hourly downpours (<15–30 minutes) unresolvable on coarse synoptic forecast grids.

---

## 2. Strict Scientific Protocol & Data Quarantine

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     DATA INTEGRITY & QUARANTINE VERIFICATION                           │
├───────────────────────────────────────┬────────────────────────────────────────────────┤
│ Requirement                           │ Implementation Status                          │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ 1. Zero Retraining on v2.5 Test Set   │ [PASS] 2026 prospective 19 events quarantined │
│ 2. Pre-Execution Data Coverage Gate   │ [PASS] Audited 2010–2025 across all 8 sources  │
│ 3. Zero Sensor Fabrication            │ [PASS] Missing GNSS/InSAR marked UNAVAILABLE   │
│ 4. Verified Physical Lineage          │ [PASS] Copernicus DEM 30m, ERA5-Land, USGS, GSI│
│ 5. Temporal Causality Enforcement     │ [PASS] Strict pre-event historical windows     │
└───────────────────────────────────────┴────────────────────────────────────────────────┘
```

> [!CAUTION]
> **Strict Non-Leakage Quarantine**: The 19 independently verified real disaster events from the 2026 90-day prospective test are **strictly quarantined**. Neither the 19 events nor any prediction outputs from the prospective shadow test are included in the training or validation partitions of v2.6.

---

## 3. Dataset Splitting & Scientific Period

Based on the empirical Pre-Execution Data Coverage Gate audit (`results/DATA_COVERAGE_GATE.json`), the continuous multi-variate environmental record exists across 2011–2016:

| Partition | Time Period | Monsoon Seasons | Events Count | Role in v2.6 Pipeline |
| :--- | :---: | :---: | :---: | :--- |
| **Training Split** | 2011-01-01 to 2014-12-31 | 4 Annual Monsoons | 96 Verified Events | Primary model parameter optimization across 5 random seeds |
| **Validation Split** | 2015-01-01 to 2015-12-31 | 1 Full Monsoon | 58 Verified Events | Validation-only feature selection, thresholding, calibration, and candidate selection |
| **Held-Out Baseline Split** | 2016-01-01 to 2016-10-15 | 1 Full Monsoon | 19 Verified Events | Frozen historical benchmark for v2.5 baseline comparison |
| **Quarantined Prospective** | 2026-06-01 to 2026-09-04 | 1 Prospective Monsoon | 19 Real Events | **STRICTLY QUARANTINED — FORBIDDEN FROM RETRAINING** |

---

## 4. Multi-Trigger Feature Architecture (86 Variables)

The v2.6 feature space is partitioned into 8 cohesive physical families:

### 1. Convective Precipitation & Cloudburst (15 features)
- `subhourly_intensity_proxy`: High-rate peak intensity proxy from 15-minute / 30-minute convective instability.
- `spatial_rain_grad_1km`, `spatial_rain_grad_5km`, `spatial_rain_grad_10km`, `spatial_rain_grad_25km`: Multi-scale horizontal rain gradient capturing convective cells.
- `nowcast_qpf_burst_ratio`: Ratio of immediate 1h precipitation to total 24h accumulation.
- `convective_divergence_index`: Spatial Laplacian $\nabla^2 P$ indicating hyper-localized downpour centers.
- `temporal_rain_acceleration`: Second temporal derivative $\frac{\partial^2 P}{\partial t^2}$.

### 2. Hydrology & Soil Wetness (9 features)
- `sm_volumetric`, `sm_sat_ratio`, `SWI` (Soil Water Index 5-day recursive filter).
- `api_92`: Antecedent Precipitation Index ($k = 0.92$).
- `inf_proxy`, `runoff_proxy`, `pore_press_proxy`, `antecedent_saturation_index`.

### 3. Terrain Geomorphology (8 features)
- `slope_deg`, `aspect_deg`, `curvature`, `relief_m`, `twi` (Topographic Wetness Index), `tpi` (Topographic Position Index).
- `slope_variability`, `stability_proxy` (Infinite slope factor of safety reciprocal).

### 4. Road-Cut Toe Excavation (9 features)
- `dist_to_road_km`, `road_cut_proximity` ($1 / (1 + d)$).
- `road_cut_indicator`: Flags slopes $> 20^\circ$ within 900m of the highway corridor.
- `road_cut_slope_diff`: Differential cut-slope steepening $\Delta \theta = \theta_{\text{cut}} - \theta_{\text{natural}}$.
- `toe_excavation_risk_index`: Geotechnical toe stress ratio $\frac{\tan(\theta)}{d + 5.0}$.
- `human_slope_disturbance_index`: Corridor buffer indicator for highway widening.
- `cut_face_height_m`: Estimated vertical face height.

### 5. Drainage Network & Culvert Scour (7 features)
- `culvert_proximity`: Proximity to highway cross-drain culverts ($1 / (1 + d / 120)$).
- `culvert_choke_risk`: Upstream contributing drainage area $\times$ surface runoff proxy.
- `scour_susceptibility_index`: Stream gradient $\times$ flow accumulation at road edge.
- `drainage_density_km_km2`: Channel length per unit area.
- `culvert_blockage_potential`: Interaction of high culvert proximity and saturated soil.

### 6. Freeze/Thaw Thermal Dynamics (8 features)
- `temperature_c`, `temp_cross_0c`: Diurnal zero-crossing detection.
- `hours_below_0c`, `hours_above_0c_after_freeze`: Thermal cycle duration.
- `freeze_thaw_cycle_count`, `rapid_temp_transition`: Captures winter/spring rock wedge failure in high altitude corridors (Tawang, Gangtok-Mangan).

### 7. Seismic & Co-Seismic Fault Prior (8 features)
- `seismic_zone_factor`: Static regional seismotectonic coefficient (GSI Zone V = 0.36).
- `seismic_pga_g`: Peak Ground Acceleration ($g$) from regional earthquakes and fault distance.
- `fault_distance_km`: Proximity to Main Central Thrust (MCT), Main Boundary Thrust (MBF), and Naga Thrust.
- `coseismic_newmark_proxy`: Critical acceleration ratio $a_c / \text{PGA}$.
- `coseismic_soil_interaction`: Interaction term $\text{PGA} \times \text{SWI}$.
- `gnss_deformation_status`, `insar_deformation_status`: Explicit availability flags (`UNAVAILABLE` in training partition; zero fabrication).

### 8. Forecast Uncertainty (4 features)
- `forecast_rain_mean_mm`, `forecast_spread`, `forecast_uncertainty`, `forecast_lead_time`.

---

## 5. Negative Declarations & Authentic Provenance

1. **GNSS Arrays**: No continuous highway deformation GNSS network exists along the 8 corridors; marked `UNAVAILABLE`.
2. **Pre-2014 InSAR**: Sentinel-1 SAR was launched in 2014; InSAR for 2011–2014 does not exist and is marked `UNAVAILABLE`.
3. **No Synthetic Shortcuts**: All terrain derivatives originate directly from Copernicus GLO-30 DSM; all weather and soil moisture parameters derive from ECMWF ERA5-Land.
