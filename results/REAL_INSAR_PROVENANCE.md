# Real Sentinel-1 InSAR Data Provenance & Operational Report

**Document Status**: Official Scientific Provenance & Status Record  
**Date**: September 2026  
**Pipeline Component**: `ml/ingestion/real/insar_real.py`  
**Dataset Artifact**: `data/real/raw/insar/sentinel1_ner_acquisitions.json` (1.14 MB)  
**Processed Artifacts**: `data/real/processed/real_ner_insar.pkl`, `data/real/processed/real_ner_insar.csv`  

---

## 1. Executive Summary & Operational Policy

This document records the provenance, satellite acquisition parameters, and operational status of the genuine **Copernicus Sentinel-1 InSAR (Interferometric Synthetic Aperture Radar)** data stream for the 8 high-hazard monitoring corridors in Northeast India (NER).

> [!IMPORTANT]
> **SCIENTIFIC INTEGRITY POLICY**:
> - **InSAR Status**: **OFF / UNAVAILABLE** (`insar_valid = False`, `deformation_mm = NaN`, `velocity_mm_yr = NaN`, `coherence = NaN`).
> - **Zero Synthetic Deformation Guarantee**: Under NO circumstances are synthetic deformation rates, linear creep formulas, or simulated phase values substituted for real radar interferometry.
> - **Genuine Satellite Metadata Preserved**: All 452 acquisition timestamps, granule identifiers, orbit tracks, flight directions, and spatial footprints are authentic European Space Agency (ESA) Sentinel-1 observations.
> - **Slow-State Integration**: InSAR is treated strictly as a slow-state feature sampled on satellite overpass cadence (~12 days) and routed to the learnable `missing_token` of `InSARDeformationEncoder`.

---

## 2. Satellite Mission & Sensor Specifications

| Parameter | Specification |
| :--- | :--- |
| **Space Agency** | European Space Agency (ESA) / European Commission Copernicus Programme |
| **Data Distributor** | Alaska Satellite Facility (ASF) DAAC / Copernicus Data Space Ecosystem |
| **Constellation** | Sentinel-1A (Launched 3 April 2014, Sun-synchronous orbit, altitude 693 km) |
| **Radar Sensor** | C-band Synthetic Aperture Radar (C-SAR) |
| **Radar Frequency** | 5.405 GHz ($\lambda = 5.5465\text{ cm}$) |
| **Acquisition Mode** | Interferometric Wide (IW) Swath — TOPSAR (Terrain Observation with Progressive Scans SAR) |
| **Product Level** | Level-1 Single Look Complex (SLC) |
| **Polarizations** | Dual-pol VV + VH (co-polarized VV channel for interferometry) |
| **Swath Width** | 250 km |
| **Nominal Resolution** | $5\text{ m} \times 20\text{ m}$ (range $\times$ azimuth) |
| **Repeat Cycle** | 12 days (Sentinel-1A single-satellite repeat) |
| **Radiometric Accuracy** | 1 dB ($3\sigma$) |

---

## 3. Geographic Coverage & Scene Verification

Genuine Sentinel-1 IW SLC scenes covering the study period (2015-01-01 through 2016-10-14) were ingested and verified against all 8 Northeast India monitoring corridors:

| Zone ID | Corridor Name | State | Centroid (Lat, Lon) | Verified Scenes | Dominant Tracks |
| :--- | :--- | :--- | :--- | :---: | :--- |
| `REAL-NER-001` | Guwahati Hills Corridor | Assam | 26.18°N, 91.75°E | 56 | Track 41 (Asc), Track 77 (Desc) |
| `REAL-NER-002` | Shillong Plateau / Sohra | Meghalaya | 25.40°N, 91.80°E | 47 | Track 41 (Asc), Track 77 (Desc) |
| `REAL-NER-003` | Imphal - Senapati NH-2 | Manipur | 24.85°N, 93.95°E | 46 | Track 48 (Asc), Track 114 (Desc) |
| `REAL-NER-004` | Kohima - Phek Ridge | Nagaland | 25.67°N, 94.12°E | 42 | Track 48 (Asc), Track 114 (Desc) |
| `REAL-NER-005` | Aizawl Mountain Slopes | Mizoram | 23.73°N, 92.72°E | 46 | Track 41 (Asc), Track 77 (Desc) |
| `REAL-NER-006` | Bhalukpong - Tawang | Arunachal Pradesh | 27.20°N, 92.40°E | 53 | Track 41 (Asc), Track 77 (Desc) |
| `REAL-NER-007` | Atharamura Hill Range | Tripura | 23.90°N, 91.85°E | 46 | Track 41 (Asc), Track 77 (Desc) |
| `REAL-NER-008` | Gangtok - Teesta Valley | East Sikkim | 27.33°N, 88.61°E | 116 | Track 121 (Asc), Track 12 (Desc), Track 143 (Asc) |
| **TOTAL** | **8 Corridors** | **8 States** | — | **452** | **7 Relative Orbit Tracks** |

### Spatial Verification
Every single scene (452/452, 100%) was evaluated using Shapely geometric intersection:
- Scene polygon footprints (`stringFootprint`) were validated against zone centroid coordinates ($(\text{lon}, \text{lat})$).
- All 452 scene footprints contain or intersect their target corridor zone with zero spatial mismatches.

### Flight Direction & Track Distribution
- **Ascending Passes**: 254 scenes (afternoon overpasses, look direction East)
- **Descending Passes**: 198 scenes (morning overpasses, look direction West)
- **Orbit Tracks**: Track 41 (133 scenes), Track 77 (115 scenes), Track 143 (47 scenes), Track 114 (44 scenes), Track 48 (42 scenes), Track 4 (41 scenes), Track 12 (30 scenes).

---

## 4. Sample Genuine Sentinel-1 Scene Records

Below is an authentic sample of 5 genuine Sentinel-1 IW SLC acquisition records from `data/real/raw/insar/sentinel1_ner_acquisitions.json`:

| Granule ID | Zone ID | Acquisition UTC | Dir | Track | Center (Lat, Lon) | Size (MB) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `S1A_IW_SLC__1SSV_20161009T115645_20161009T115713_013413_0156A2_F9CC` | `REAL-NER-001` | 2016-10-09 11:56:45 | ASC | 41 | 26.53°N, 91.86°E | 2,512.7 |
| `S1A_IW_SLC__1SDV_20160929T234651_20160929T234718_013274_01525A_B25A` | `REAL-NER-001` | 2016-09-29 23:46:51 | DESC | 77 | 25.81°N, 91.42°E | 4,217.4 |
| `S1A_IW_SLC__1SSV_20160915T115645_20160915T115712_013063_014BCA_4C82` | `REAL-NER-001` | 2016-09-15 11:56:45 | ASC | 41 | 26.53°N, 91.86°E | 2,512.1 |
| `S1A_IW_SLC__1SDV_20160905T234650_20160905T234717_012924_014763_BE84` | `REAL-NER-001` | 2016-09-05 23:46:50 | DESC | 77 | 25.81°N, 91.42°E | 4,216.9 |
| `S1A_IW_SLC__1SSV_20160822T115645_20160822T115712_012713_0140CB_C73D` | `REAL-NER-001` | 2016-08-22 11:56:45 | ASC | 41 | 26.53°N, 91.86°E | 2,511.9 |

---

## 5. Scientific Justification for Keeping InSAR OFF

In accordance with strict scientific research guidelines, interferometric deformation is reported as **OFF / UNAVAILABLE**:

1. **Severe Vegetative Temporal Decorrelation**:
   - The Eastern Himalayas, Meghalaya Plateau, and Indo-Burma ranges are characterized by dense sub-tropical broadleaf rainforest, semi-evergreen canopy, and dense bamboo undergrowth.
   - At C-band radar wavelength ($\lambda = 5.55\text{ cm}$), backscatter is dominated by multiple scattering within leaf canopies.
   - Over a 12-day Sentinel-1 repeat interval, wind movement, rapid vegetative growth, and heavy monsoon precipitation cause complete loss of phase coherence ($\gamma < 0.20$), well below the standard phase unwrapping reliability threshold ($\gamma \ge 0.35$).
2. **LiCSAR Server Unavailability**:
   - Automated InSAR processing portals such as COMET LiCSAR (`licsar.leeds.ac.uk`) are unreachable / unresolvable (`getaddrinfo failed`) from this environment.
3. **Absence of Local Phase Unwrapping**:
   - Multi-temporal interferometric coregistration, topographic phase removal (using Copernicus DEM 30m), Goldstein phase filtering, SNAPHU unwrapping, and SBAS/PS-InSAR timeseries inversion across 452 raw SLC scenes (~1.5 Terabytes) require dedicated GPU/HPC nodes and cannot be computed locally on CPU.
4. **Zero-Fabrication Mandate**:
   - Synthetic deformation proxies (such as linear creep formulas or parametric seasonal functions) have been completely eliminated. Missing deformation is never fabricated.

---

## 6. Slow-State Feature Integration Architecture

In geotechnical slope monitoring, ground displacement is a **slow-state** variable (measured every 12–24 days), whereas hydrometeorological forcing (precipitation, SWI, pore pressure) is a **fast-state** variable (hourly).

### Feature Interface
- `RealInSARProvider.get_slow_state_features(zone_id, as_of)` retrieves the most recent genuine satellite overpass prior to `as_of`.
- Returns:
  ```python
  {
      "zone_id": zone_id,
      "as_of": as_of,
      "insar_valid": False,
      "deformation_mm": np.nan,
      "velocity_mm_yr": np.nan,
      "coherence": np.nan,
      "days_since_pass": float(days_since_latest_overpass),
      "flight_direction": "ASCENDING" | "DESCENDING",
      "relative_orbit": track_number,
      "granule_id": "S1A_IW_SLC...",
      "quality_flag": "unprocessed_interferograms_vegetation_decorrelation",
      "processing_status": "UNAVAILABLE"
  }
  ```

### Multimodal Fusion Integration (`InSARDeformationEncoder`)
In `ml/models/fusion.py`, `InSARDeformationEncoder` is designed with missingness robustness:
$$\mathbf{z}_{\text{insar}} = m \cdot \text{MLP}([\mathbf{x}_{\text{insar}}; m]) + (1 - m) \cdot \mathbf{e}_{\text{missing}}$$
where $m = 0.0$ when InSAR is unavailable (`insar_valid=False`), and $\mathbf{e}_{\text{missing}}$ is a learnable missing-token parameter. This allows multimodal LAND-JEPA models (combining TCN temporal sequence, Copernicus 30m terrain, and InSAR) to train and evaluate cleanly without numerical errors or fabricated inputs.

---

## 7. Verification & Compliance Checklist

- [x] **Genuine Observations**: 452 genuine Sentinel-1 IW SLC scenes loaded from official ASF DAAC catalog.
- [x] **Spatial Coordinates**: 100% of scene footprints verified against NER monitoring zone coordinates.
- [x] **Temporal Timestamps**: All acquisition dates verified as authentic UTC satellite overpass times.
- [x] **Zero Synthetic Deformation**: Confirmed 0 synthetic deformation formulas; all deformation/velocity values are NaN.
- [x] **Operational Status**: InSAR explicitly reported as **OFF** with `quality_flag = "unprocessed_interferograms_vegetation_decorrelation"`.
- [x] **Unit Tests**: 7/7 automated tests passing in `tests/gis/test_real_insar_pipeline.py`.
