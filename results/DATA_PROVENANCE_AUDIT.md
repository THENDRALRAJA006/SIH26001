# LAND-JEPA: Dataset Provenance, Geographic Integrity, and Data Leakage Audit

**Audit Date**: September 2026  
**Audited Artifact**: LAND-JEPA Machine Learning Pipeline & Benchmark Suite  
**Target Domain**: High-Hazard Mountain Corridors of Northeast India (NER)  
**Classification**: Research & Software Integration Integrity Report  
**Compliance**: Smart India Hackathon 2026 (Problem Statement SIH26001)

---

## Executive Summary

This document establishes the empirical data provenance, lineage, geographic scope, temporal partitions, and zero-leakage guarantees for the input streams utilized by the **LAND-JEPA** modeling suite (`XGBoost`, `Supervised TCN`, `JEPA-TCN`, and `Fused LAND-JEPA`).

> [!CAUTION]
> **PRIMARY INTEGRITY FINDING: ALL CURRENT DATA IS DEMO / SYNTHETIC**  
> All environmental observations (rainfall, weather, soil hydrology proxies), terrain parameters, ground deformation vectors, and landslide occurrence catalogs currently ingested by the model training, pre-training, and evaluation pipelines are **100% SYNTHETIC / DEMO DATA**. No real-world physical measurements or ground-truth event reports have been ingested into the active ML pipelines. All records carry explicit provenance markers (`is_demo=True`, `source="DEMO_*"`). This synthetic data was created to rigorously validate software integration, architecture stability, causal temporal windowing, and label-efficiency mechanics. It must NOT be claimed as real-world geotechnical proof.

---

## 1. Input Datasets Inventory & Lineage

The LAND-JEPA pipeline ingests six primary data streams. Below is the detailed audit for each stream:

### 1.1 Precipitation / Rainfall Time-Series

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | `DEMO_RAINFALL` (Procedural Synthetic Hourly Precipitation) |
| **Operational Reality** | **DEMO / SYNTHETIC** (`is_demo=True`, `source="DEMO_RAINFALL"`) |
| **Executing Generator** | `ml/ingestion/demo/rainfall_demo.py` (`DemoRainfallProvider`) |
| **Intended Real-World Source** | IMD Gridded Daily/Hourly Rainfall (0.25° grid) / CHIRPS / ERA5-Land (ECMWF CDS) |
| **Original Reference URLs** | - IMD: [https://www.imdpune.gov.in/](https://www.imdpune.gov.in/)<br>- CHIRPS: [https://www.chc.ucsb.edu/data/chirps](https://www.chc.ucsb.edu/data/chirps)<br>- ERA5-Land: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: India Meteorological Department, UC Santa Barbara / USGS, ECMWF) |
| **Geographic Coverage** | 8 synthetic zone centroids across Northeast India (Assam, Meghalaya, Manipur, Nagaland, Mizoram, Arunachal Pradesh, Tripura) |
| **Temporal Coverage** | `2020-09-30 00:00:00+00:00` to `2023-09-30 00:00:00+00:00` (UTC-aware, exactly 3 calendar years = 26,281 hourly steps) |
| **Number of Raw Records** | **210,248 hourly records** ($8\text{ zones} \times 26,281\text{ hours}$) |
| **Data Generation Method** | Sinusoidal monsoon baseline peaking in July–August ($M=7.5$), diurnal afternoon cycle, zone-specific orographic scaling ($0.5\text{--}3.5\text{ mm/h}$ base), and stochastic lognormal extreme burst triggers. |

---

### 1.2 Surface Meteorology & Weather Time-Series

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | `DEMO_WEATHER` (Procedural Synthetic Hourly Surface Meteorology) |
| **Operational Reality** | **DEMO / SYNTHETIC** (`is_demo=True`, `source="DEMO_WEATHER"`) |
| **Executing Generator** | `ml/ingestion/demo/weather_demo.py` (`DemoWeatherProvider`) |
| **Intended Real-World Source** | Open-Meteo Historical / Forecast Weather API / ERA5-Land Surface |
| **Original Reference URLs** | - Open-Meteo: [https://api.open-meteo.com/v1/forecast](https://api.open-meteo.com/v1/forecast)<br>- ERA5-Land: [https://cds.climate.copernicus.eu/](https://cds.climate.copernicus.eu/) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: Open-Meteo GmbH, ECMWF) |
| **Geographic Coverage** | 8 synthetic zone centroids across Northeast India |
| **Temporal Coverage** | `2020-09-30 00:00:00+00:00` to `2023-09-30 00:00:00+00:00` (26,281 hourly steps) |
| **Number of Raw Records** | **210,248 hourly records** ($8\text{ zones} \times 26,281\text{ hours}$) |
| **Variables Included** | `temperature_c`, `humidity_pct`, `wind_speed_ms`, `wind_dir_deg`, `pressure_hpa` |
| **Data Generation Method** | Climatologically bounded seasonal sinusoidal cycle ($10\text{--}35^\circ\text{C}$), monsoon-correlated relative humidity ($65\text{--}98\%$), lognormal wind velocity, and barometric elevation proxies. |

---

### 1.3 Hydrologic & Unsaturated Soil Mechanics Proxy Features

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | Deterministic Hydrologic State Vector (`PhysicsStateEstimator`) |
| **Operational Reality** | **SYNTHETIC / COMPUTED PROXY** (Deterministically derived from synthetic rainfall) |
| **Executing Generator** | `ml/features/physics_state.py` (`PhysicsStateEstimator`) |
| **Intended Real-World Source** | ESA CCI Soil Moisture Combined / In-situ piezometers / Satellite SWI |
| **Original Reference URLs** | - ESA CCI SM: [https://www.esa-soilmoisture-cci.org/](https://www.esa-soilmoisture-cci.org/)<br>- Copernicus Land: [https://land.copernicus.eu/](https://land.copernicus.eu/) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: European Space Agency, CWC) |
| **Geographic Coverage** | 8 synthetic zone centroids across Northeast India |
| **Temporal Coverage** | `2020-09-30 00:00:00+00:00` to `2023-09-30 00:00:00+00:00` (26,281 hourly steps) |
| **Number of Raw Records** | **210,248 hourly state records** |
| **Variables Included** | Soil Water Index (`swi`, $T=24\text{h}$ and $T=72\text{h}$ recursive filters), pore-pressure proxy ratio ($r_u$), infinite slope stability proxy index |
| **Mathematical Formulation** | Exponential recursive infiltration filter: $SWI_t = SWI_{t-1} + K(P_t - SWI_{t-1})$, normalized to field capacity; $r_u = \text{clip}(SWI \times 0.65, 0, 1)$. |

---

### 1.4 Static Geomorphometric Terrain Attributes

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | `DEMO_TERRAIN_SRTM_SYNTHETIC` (Synthetic Digital Elevation Attributes) |
| **Operational Reality** | **DEMO / SYNTHETIC** (`is_demo=True`, `source="DEMO_TERRAIN_SRTM_SYNTHETIC"`) |
| **Executing Generator** | `ml/ingestion/demo/terrain_landslide_demo.py` (`DemoTerrainProvider`) |
| **Intended Real-World Source** | NASA SRTM 30m / ALOS World 3D (AW3D30 12.5m) / Copernicus DEM GLO-30 |
| **Original Reference URLs** | - NASA EarthExplorer: [https://earthexplorer.usgs.gov/](https://earthexplorer.usgs.gov/)<br>- ALOS AW3D30: [https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/index.htm](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/index.htm)<br>- Copernicus DEM: [https://spacedata.copernicus.eu/](https://spacedata.copernicus.eu/) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: NASA / USGS, JAXA, ESA) |
| **Geographic Coverage** | 8 synthetic zone boundaries in Northeast India |
| **Temporal Coverage** | **Static** (Invariant across all temporal windows) |
| **Number of Raw Records** | **8 records** (One 8-dimensional attribute profile per zone) |
| **Variables Included** | `elevation_m`, `slope_deg`, `aspect_deg`, `curvature`, `tpi`, `twi`, `lithology_class`, `land_cover`, `terrain_class` |
| **Terrain Classes Simulated** | `steep_hill` (slope 25–55°), `moderate_hill` (slope 15–35°), `foothill` (5–20°), `valley` (2–10°), `floodplain` (0–5°). |

---

### 1.5 Historical Landslide Inventory Catalog

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | `DEMO_LANDSLIDE_INVENTORY` (Synthetic Landslide Event Catalog) |
| **Operational Reality** | **DEMO / SYNTHETIC** (`is_demo=True`, `source="DEMO_LANDSLIDE_INVENTORY"`) |
| **Executing Generator** | `ml/ingestion/demo/terrain_landslide_demo.py` (`DemoLandslideInventoryProvider`) |
| **Intended Real-World Source** | Bhuvan Landslide Atlas (ISRO/NRSC), GSI Bhukosh, NASA Global Landslide Catalog |
| **Original Reference URLs** | - Bhuvan ISRO: [https://bhuvan-app3.nrsc.gov.in/disaster/disaster.php?id=landslide](https://bhuvan-app3.nrsc.gov.in/disaster/disaster.php?id=landslide)<br>- GSI Bhukosh: [https://bhukosh.gsi.gov.in/](https://bhukosh.gsi.gov.in/)<br>- NASA GLC: [https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/h9d8-neg4](https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/h9d8-neg4) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: ISRO / NRSC, Geological Survey of India, NASA) |
| **Geographic Coverage** | Bounding box 88.0°E to 97.5°E, 21.5°N to 28.5°N |
| **Temporal Coverage** | `2020-09-30 00:00:00+00:00` to `2023-09-30 00:00:00+00:00` |
| **Raw Candidate Events** | **200 generated candidates** |
| **Events Surviving Seasonal Filter** | **117 events** (simulating 60% non-monsoon rejection sampling) |
| **Imprecise Excluded Events** | **31 events** (`date_precision` in `month`, `year`, or `unknown` filtered out by `LabelBuilder`) |
| **Total Usable Ground-Truth Events** | **86 unique events** (`date_precision` in `exact` [10%] or `day` [60%]) |
| **Distribution Across Splits** | - Train: 63 usable events<br>- Validation: 15 usable events<br>- Holdout Test: **8 usable events** |

---

### 1.6 Interferometric SAR (InSAR) Ground Velocity & Coherence

| Property | Audit Record |
| :--- | :--- |
| **Dataset / Product Name** | Synthetic Line-of-Sight Ground Displacement & Coherence Stream |
| **Operational Reality** | **DEMO / SYNTHETIC** (`is_demo=True`, presence mask active) |
| **Executing Generator** | `ml/models/fusion.py` (`InSARDeformationEncoder`) / `gis/insar_adapter.py` |
| **Intended Real-World Source** | Sentinel-1 SAR (ESA Copernicus) processed via SBAS/PS-InSAR (MintPy/ISCE2) / NISAR |
| **Original Reference URLs** | - Copernicus Data Space: [https://dataspace.copernicus.eu/](https://dataspace.copernicus.eu/)<br>- NASA-ISRO NISAR: [https://nisar.jpl.nasa.gov/](https://nisar.jpl.nasa.gov/) |
| **Publishing Organization** | Generated in-house by ZAIX Team. (Target orgs: ESA, ISRO, NASA JPL) |
| **Geographic Coverage** | 8 synthetic zone boundaries in Northeast India |
| **Temporal Coverage** | Simulated multi-temporal interferometric velocity ($V_{\text{LOS}}$ in $\text{mm/yr}$, coherence $\gamma \in [0, 1]$) |
| **Number of Raw Records** | 8 zone vectors with boolean availability mask |

---

## 2. Sliding Window Generation & Split Breakdown

Sliding windows are constructed using `WindowGenerator` (`ml/features/window_generator.py`) with a context duration $T = 168\text{ hours}$ (7 days), target forecast horizon $\Delta = 24\text{ hours}$, and stride $S = 24\text{ hours}$ across all 8 zones:

$$\text{Total Raw Windows Generated} = 8,704 \text{ windows}$$

### 2.1 Window Categorization by LabelBuilder

| Category | Definition | Count | Percentage |
| :--- | :--- | :---: | :---: |
| **Positive Windows ($y = 1$)** | Usable event occurred strictly within $(context\_end, context\_end + 24\text{h}]$ | **86** | 1.0% |
| **Negative Windows ($y = 0$)** | No event in $(context\_end, context\_end + 24\text{h}]$ AND no event in $[context\_end - 72\text{h}, context\_end]$ | **8,365** | 96.1% |
| **Ambiguous Excluded ($y = -1$)** | Event occurred within 72h pre-event buffer $[context\_end - 72\text{h}, context\_end]$ | **253** | 2.9% |
| **Total Usable Labeled Windows** | Sum of Clean Positives and Clean Negatives | **8,451** | **100.0%** |

### 2.2 Strict Temporal Partitions

The dataset is partitioned using strict temporal forward-chaining. Windows are assigned based on `context_end`:

```
2020-10-07 ────────────────────────► 2022-12-31 ────────► 2023-06-30 ────────► 2023-09-30
           TRAIN SET (6,352)                     VAL (1,406)              TEST (693)
           (63 Positives, 0.99%)                 (15 Pos, 1.07%)          (8 Pos, 1.15%)
```

| Split Partition | `context_end` Range (UTC) | Duration | Windows ($N$) | Positives ($N_{\text{pos}}$) | Positive Rate |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Train** | `2020-10-07 00:00:00` to `2022-12-31 23:00:00` | 27 Months | **6,352** | 63 | 0.99% |
| **Validation** | `2023-01-01 00:00:00` to `2023-06-30 23:00:00` | 6 Months | **1,406** | 15 | 1.07% |
| **Holdout Test** | `2023-07-01 00:00:00` to `2023-09-30 00:00:00` | 3 Months | **693** | **8** | **1.15%** |
| **Total Labeled** | `2020-10-07` to `2023-09-30` | 36 Months | **8,451** | **86** | **1.02%** |

---

## 3. Geographic Partitioning Audit

| Split Partition | Geographic Regions & Zones Included | Spatial Holdout Applied? |
| :--- | :--- | :---: |
| **Train** | All 8 zones (`DEMO-NER-001` through `DEMO-NER-008`) | **NO** |
| **Validation** | All 8 zones (`DEMO-NER-001` through `DEMO-NER-008`) | **NO** |
| **Holdout Test** | All 8 zones (`DEMO-NER-001` through `DEMO-NER-008`) | **NO** |

### Geographic Identification of the 8 Synthetic Zones
1. `DEMO-NER-001`: Kamrup Hills, Kamrup District, Assam (Centroid: 91.8°E, 26.1°N)
2. `DEMO-NER-002`: Meghalaya East, East Khasi Hills, Meghalaya (Centroid: 91.9°E, 25.4°N)
3. `DEMO-NER-003`: Manipur Valley, Imphal West, Manipur (Centroid: 93.9°E, 24.8°N)
4. `DEMO-NER-004`: Nagaland South, Phek District, Nagaland (Centroid: 94.5°E, 25.5°N)
5. `DEMO-NER-005`: Mizoram North, Aizawl District, Mizoram (Centroid: 92.7°E, 23.7°N)
6. `DEMO-NER-006`: Arunachal West, West Kameng, Arunachal Pradesh (Centroid: 92.2°E, 27.1°N)
7. `DEMO-NER-007`: Tripura Hills, Dhalai District, Tripura (Centroid: 91.7°E, 23.9°N)
8. `DEMO-NER-008`: Assam Foothills, Sonitpur District, Assam (Centroid: 92.8°E, 26.6°N)

> [!NOTE]
> **Spatial Cross-Validation Caveat**:  
> The current benchmark strictly measures **temporal generalizability** into future monsoon seasons across known spatial zones. It does NOT evaluate spatial zero-shot transfer to unseen mountain valleys (leave-one-zone-out spatial CV). For regional deployment, spatial cross-validation across distinct geomorphic sub-provinces is required.

---

## 4. Exact Label Construction Method

The labeling mechanism is implemented in `LabelBuilder` (`ml/features/label_builder.py`). Every sliding window is assigned a label $y \in \{0, 1, -1\}$ based on the following deterministic rules:

### Step 1: Temporal Window Alignment
For window index $i$, the context sequence spans $[t - 168\text{h}, t]$, where $t = context\_end$.  
The target prediction window spans $(t, t + 24\text{h}]$, where $t + 24\text{h} = target\_end$.

### Step 2: Date Precision Filtering
Historical landslide records contain variable temporal precision. In accordance with geotechnical integrity standards:
- **Usable Events**: Records with `date_precision == "exact"` (confidence = 1.0) or `date_precision == "day"` (confidence = 0.8) are retained.
- **Excluded Events**: Records with `date_precision` in `("month", "year", "unknown")` are permanently excluded ($31\text{ events}$ in current dataset). They are neither positive labels nor clean negatives.

### Step 3: Positive Label Assignment ($y = 1$)
If at least one usable event occurs within the target window:
$$\exists e \in \mathcal{E}_{zone} \quad \text{s.t.} \quad t < \text{occurred\_at}(e) \le t + 24\text{h} \implies y = 1$$

### Step 4: Ambiguity Buffer Filtering ($y = -1$)
In progressive slope failures, ground deformation, pore pressure buildup, and micro-shearing precede macroscopic collapse. Labeling the pre-failure window as a "clean negative" misleads the model. A 72-hour pre-event negative buffer is applied:
$$\exists e \in \mathcal{E}_{zone} \quad \text{s.t.} \quad t - 72\text{h} \le \text{occurred\_at}(e) \le t \implies y = -1 \quad (\text{Excluded})$$
All windows marked $y = -1$ are purged from training, validation, and testing.

### Step 5: Clean Negative Assignment ($y = 0$)
Windows with zero events in $(t, t + 24\text{h}]$ and zero events in $[t - 72\text{h}, t]$ are assigned $y = 0$.

### Step 6: Subsampling in Label-Efficiency Regimes
Under reduced label regimes ($f \in \{0.01, 0.05, 0.10, 0.25, 0.50\}$):
- Negative windows ($y = 0$) are never subsampled (background negative terrain is assumed accessible).
- A deterministic random subset of positive training windows of size $\max(1, \lfloor N_{\text{pos, train}} \times f \rfloor)$ is retained ($y = 1$).
- The remaining positive training windows are converted to $y = -1$ (hidden from the supervised loss function).

---

## 5. Comprehensive Leakage Audit: Did Any Future Information Enter the Input?

### 5.1 Verification Matrix

| Leakage Vector | Audit Check | Empirical Code Assertion | Status |
| :--- | :--- | :--- | :---: |
| **Temporal Context Overlap** | Does context window $x_{1:t}$ contain observations from $t' > t$? | `WindowPair._assert_no_leakage`: raises `ValueError` if `context_end > target_start` | **ZERO LEAKAGE (PASSED)** |
| **Target Horizon Overlap** | Does target window overlap with past context? | Checked in `WindowGenerator.generate_arrays`: strictly non-overlapping slice indices | **ZERO LEAKAGE (PASSED)** |
| **Normalizer Leakage** | Were normalizers fit on validation or test sets? | `FeatureNormalizer` and `TemporalNormalizer` fit strictly on `train` partition ($N=6,352$) | **ZERO LEAKAGE (PASSED)** |
| **Physics State Filter Leakage** | Did recursive SWI / pore pressure filters look ahead? | `PhysicsStateEstimator`: strictly causal forward recursive filter; no backward pass | **ZERO LEAKAGE (PASSED)** |
| **Label Leakage into JEPA** | Did landslide labels enter self-supervised pre-training? | `JEPATrainer` loss is computed exclusively between context predicted latent and target latent; no label tensor exists | **ZERO LEAKAGE (PASSED)** |
| **Threshold Selection Leakage** | Was the decision threshold chosen on the holdout test set? | Threshold $\theta^*$ selected exclusively by maximizing F1 on validation set ($N=1,406$); test set evaluated blind | **ZERO LEAKAGE (PASSED)** |
| **Temporal Partition Shuffling** | Were samples randomly shuffled across time boundaries? | Split masks generated purely on `context_end < val_cutoff` and `context_end >= test_cutoff` | **ZERO LEAKAGE (PASSED)** |

### 5.2 Stop-Gradient and Target Encoder Isolation
In self-supervised pretraining (`scripts/pretrain_jepa.py` and `ml/models/jepa_model.py`):
- Target representations $y_{t+1:t+\Delta}$ are passed through `y_target.detach()`.
- The target TCN and target projection head are updated exclusively via Exponential Moving Average ($\tau = 0.999$); no gradients are backpropagated into the future target sequence.
- This guarantees that representation learning does not leak future gradient information into the context encoder.

---

## 6. Recommendations for Operational Real-World Transition

Before deploying this system in an operational command center (SDMA/NDMA):
1. **Replace Demo Ingestion Adapters**: Connect `IMDRainfallProvider` and `BhuvanLandslideProvider` using authenticated API keys.
2. **Historical Georeferencing Ingest**: Ingest verified historical records from the GSI National Landslide Susceptibility Mapping (NLSM) database.
3. **Spatial Cross-Validation**: Re-run the benchmark using leave-one-district-out spatial cross-validation to assess transferability across geomorphic domains.
4. **Calibrate Alert Thresholds**: Transition from unconstrained validation F1 argmax to cost-sensitive threshold selection with a fixed False Positive Rate cap ($\le 5\%$).
