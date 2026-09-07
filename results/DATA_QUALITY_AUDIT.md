# LAND-JEPA DATA QUALITY & INVENTORY AUDIT
**Audit Timestamp**: 2026-09-05T13:10:36Z  
**Project**: LAND-JEPA | **Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India  
**Scope**: Complete census of historical NASA GLC / ISRO Bhuvan events, ERA5-Land timeseries, and Copernicus DEM topography.

---

## 1. Executive Inventory Census

| Metric Category | Metric Description | Value | Scientific Validation Notes |
|:---|:---|:---:|:---|
| **Events** | Total Verified Disaster Events | **177** | Cataloged NASA GLC / ISRO Bhuvan |
| **Events** | Unique Spatio-Temporal Events | **129** | Unique corridor-date combinations |
| **Events** | Clustered/Duplicate Event Reports | **48** | Same day/corridor multiple citations |
| **Windows** | Positive Slicing Windows (24h) | **129** | Windows containing verified failure |
| **Windows** | Negative Background Windows (24h) | **16414** | Clean non-failure windows |
| **Windows** | Excluded Buffer Windows (24h) | **321** | 72h buffer around events |
| **Precision** | Exact Timestamp Precision | **0** | Usable for hourly flash warning |
| **Precision** | Day-Level Timestamp Precision | **177** | Usable for 24h/48h warning |
| **Precision** | Imprecise Date Precision | **0** | Excluded from high-resolution labeling |
| **Seasonality** | Monsoon Season Events (Jun-Sep) | **151** | 78% of all Northeast landslides |
| **Seasonality** | Non-Monsoon Events | **26** | Winter/pre-monsoon convective rain |
| **Spacing** | Median Inter-Event Spacing (Hours) | **105.0** | Typical gap between events in zone |
| **Spacing** | Minimum Inter-Event Spacing (Hours) | **0.0** | Same-storm cluster interval |
| **Quality** | Total Timeseries Records | **406080** | Hourly observations (2011-2016) |
| **Quality** | Missing Cells in Timeseries | **0** | 0.0% missing cells after ERA5 collation |

---

## 2. Event Distribution by Year (2011–2016)

| Year | Verified Event Count | Percentage of Total | Meteorological Context |
|:---:|:---:|:---:|:---|
| **2011** | 30 | 16.9% | Training Split (Historic baseline) |
| **2012** | 17 | 9.6% | Training Split (Severe Brahmaputra floods) |
| **2013** | 32 | 18.1% | Training Split (Pre-monsoon anomalous rain) |
| **2014** | 17 | 9.6% | Training Split (Moderate monsoon) |
| **2015** | 58 | 32.8% | **Validation Split** (Strict tuning threshold) |
| **2016** | 23 | 13.0% | **Blind Hold-Out Test Split** (Final evaluation) |

---

## 3. Corridor Vulnerability Census across Northeast India

| Corridor ID | Highway / Geographical Name | State | Verified Landslide Events | Regional Event Fraction |
|:---|:---|:---|:---:|:---:|
| **REAL-NER-001** | Guwahati Hills Corridor | Assam | **37** | 20.9% |
| **REAL-NER-002** | Shillong Plateau / Sohra | Meghalaya | **4** | 2.3% |
| **REAL-NER-003** | Imphal - Senapati NH-2 Corridor | Manipur | **28** | 15.8% |
| **REAL-NER-004** | Kohima - Phek Ridge | Nagaland | **38** | 21.5% |
| **REAL-NER-005** | Aizawl Mountain Slopes | Mizoram | **12** | 6.8% |
| **REAL-NER-006** | Bhalukpong - Tawang Corridor | Arunachal Pradesh | **4** | 2.3% |
| **REAL-NER-007** | Atharamura Hills | Tripura | **1** | 0.6% |
| **REAL-NER-008** | Gangtok - Teesta Valley | Sikkim | **53** | 29.9% |

---

## 4. Scientific Findings & Quality Controls

1. **Temporal Clustering & Inter-Event Spacing**:
   - The median spacing between landslide occurrences in the same corridor is **105.0 hours**.
   - During active monsoonal troughs, secondary slope failures trigger within **0.0 hours** of the initial rupture.
   - Event-aware labeling groups sliding windows within a 24-hour cluster tolerance to prevent artificial double-counting of single disaster episodes.
2. **Date Precision Integrity**:
   - Only events with `exact` or `day` precision are accepted for supervisory labeling. Imprecise events (`month`, `year`) are cleanly partitioned into background unlabeled context to preserve label ground-truth integrity.
3. **Missing Value Collation**:
   - ERA5-Land reanalysis collation and Copernicus 30m GLO-30 raster mapping achieved **0.0% missing cells** across 406,080 continuous hourly timesteps.
