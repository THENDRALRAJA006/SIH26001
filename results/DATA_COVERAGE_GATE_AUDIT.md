# PRE-EXECUTION DATA COVERAGE GATE AUDIT REPORT

**Project**: LAND-JEPA — AI-Based Landslide Early Warning and Risk Monitoring  
**Problem**: SIH26001 | **Team**: ZAIX | **Region**: Northeast India (8 Monitored Highway Corridors)  
**Audit Timestamp**: 2026-09-05T18:47:40.090425+00:00  
**Gate Status**: **GATE PASSED — SCIENTIFIC PERIOD SELECTED**

---

## 1. Executive Summary & Hard Stop Verification

Before creating features or training models for **LAND-JEPA v2.6**, this audit examined the actual, empirical data coverage of all required trigger sources across Northeast India for the target years **2010–2025**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        DATA COVERAGE GATE INTEGRITY CERTIFICATE                        │
├───────────────────────────────────────┬────────────────────────────────────────────────┤
│ Requirement                           │ Verification Status                            │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ 1. No Assumed Coverage                │ [PASS] Audited actual raw and processed bytes  │
│ 2. Genuine InSAR / GNSS Accounting    │ [PASS] Marked UNAVAILABLE for missing periods  │
│ 3. Zero Value Fabrication             │ [PASS] Strict missing flags enforced           │
│ 4. Automatic Valid Period Selection   │ [PASS] Selected based on real multi-year data  │
│ 5. Quarantine of 2026 Prospective Set │ [PASS] 19 prospective events strictly quarantined│
└───────────────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 2. Comprehensive Source-by-Source Coverage Table

| Source | Earliest Timestamp | Latest Timestamp | Spatial Coverage | Temporal Res | Record Count | Missing % | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rainfall (ECMWF ERA5-Land & IMD)** | `2011-01-01T00:00:00` | `2016-10-15T23:00:00` | All 8 NER Corridors (100%) | 1-Hour (Hourly) | 406,080 | 0.00% | `AVAILABLE` |
| **Atmospheric Weather (ERA5-Land 2m Temp, Pressure, Wind)** | `2011-01-01T00:00:00` | `2016-10-15T23:00:00` | All 8 NER Corridors (100%) | 1-Hour (Hourly) | 406,080 | 0.00% | `AVAILABLE` |
| **Soil Moisture (ERA5-Land 0-7cm & 7-28cm Volumetric)** | `2011-01-01T00:00:00` | `2016-10-15T23:00:00` | All 8 NER Corridors (100%) | 1-Hour (Hourly) | 406,080 | 0.00% | `AVAILABLE` |
| **DEM / Terrain Geomorphology (Copernicus GLO-30 30m DSM)** | `Static Invariant (2010-Present)` | `Static Invariant (2010-Present)` | All 8 NER Corridors (30m Grid) | Static Topography | 8 | 0.00% | `AVAILABLE` |
| **Road & Cut-Slope Geometry (OSM / BRO Highway Network)** | `Static Baseline (2010-Present)` | `Static Baseline (2010-Present)` | All 8 NER Corridors (Buffer < 50m) | Static Corridor Geometry | 8 | 0.00% | `AVAILABLE` |
| **Drainage Network & Culverts (HydroSHEDS 3-arcsec & Culvert Assets)** | `Static Hydrology Baseline` | `Static Hydrology Baseline` | All 8 NER Corridors (Stream Orders 1-5) | Static Catchment Flow | 8 | 0.00% | `AVAILABLE` |
| **Seismic Activity & Fault Priors (USGS/IMD Catalog & GSI Atlas)** | `2010-01-01T00:00:00` | `2025-12-31T23:59:59` | All 8 NER Corridors (Within 150 km) | Event-Based & Static Zone V Prior | 182 | 0.00% | `AVAILABLE` |
| **Freeze/Thaw Thermal Dynamics (Derived from ERA5 0C crossings)** | `2011-01-01T00:00:00` | `2016-10-15T23:00:00` | High-Altitude Corridors (REAL-NER-006, 008) | 1-Hour (Diurnal zero-crossings) | 406,080 | 0.00% | `AVAILABLE` |
| **InSAR Surface Deformation (ESA Sentinel-1 SAR)** | `2014-10-01T00:00:00` | `2016-10-15T00:00:00` | Partial NER (452 acquisitions, C-Band decorrelated) | 12-Day Repeat Pass | 452 | 58.40% (2010-2014 UNAVAILABLE; post-2014 decorrelated in dense canopy) | `UNAVAILABLE_HISTORICAL (2010-2014) / DEGRADED_OPTIONAL (2014-2016)` |
| **Continuous High-Rate GNSS Deformation Arrays** | `N/A` | `N/A` | 0 of 8 Corridors (No public continuous GNSS array) | N/A | 0 | 100.00% | `UNAVAILABLE (Never Fabricate)` |

---

## 3. Availability by Year Matrix (2010 – 2025)

| Year | Rainfall | Weather | Soil Moisture | DEM/Terrain | Roads | Drainage | Seismic | Freeze/Thaw | InSAR (S1) | GNSS Arrays |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **2010** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2011** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2012** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2013** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2014** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE |
| **2015** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE |
| **2016** | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE |
| **2017** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2018** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2019** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2020** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2021** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2022** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2023** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2024** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |
| **2025** | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | AVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE |

---

## 4. Automatic Selection of Maximum Scientifically Valid Period

Based strictly on real, un-fabricated data coverage across all 8 NER corridors:

- **Earliest Continuous Multi-Variate Year**: `2011`
- **Latest Continuous Multi-Variate Year**: `2016`
- **Selected Training Partition**: `2011-01-01 to 2014-12-31 (4 Seasons)`
- **Selected Validation Partition**: `2015-01-01 to 2015-12-31 (1 Season)`
- **Selected Held-Out Baseline Test**: `2016-01-01 to 2016-10-15 (Frozen Baseline)`

### Strict Negative Declarations (Zero Fabrication Policy):
1. **Continuous GNSS Arrays**: No public continuous highway deformation network exists along the 8 NER highway corridors during 2010–2025. Marked `UNAVAILABLE`. No GNSS values will be fabricated.
2. **InSAR Pre-2014**: The Copernicus Sentinel-1 constellation was launched in April 2014; therefore, InSAR data for 2010 to mid-2014 does not physically exist. Marked `UNAVAILABLE`.
3. **Quarantine of 2026 Prospective Evaluation**: The 19 prospective events and 4,320 predictions from the completed 90-day prospective test are **strictly quarantined** and forbidden from any v2.6 training, feature selection, or validation tuning.
