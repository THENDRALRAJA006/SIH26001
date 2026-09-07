# LAND-JEPA Final Data Provenance Report
## SIH26001 — Team ZAIX — September 2026

> [!NOTE]
> This report covers all five verified scientific data streams used in the LAND-JEPA benchmark.
> No synthetic or demo data enters the scientific evaluation pipeline.

---

## 1. Dataset 1 — Landslide Ground Truth

| Field | Value |
|---|---|
| **Source Organization** | NASA Goddard Earth Sciences Data and Information Services Center (GES DISC) |
| **Dataset / Product** | Global Landslide Catalog (GLC) — Community Online Resource for Reporting Earth Slide Events (COOLR) v1.1 |
| **Original URL** | https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog-Export/dd9e-wu2v |
| **File Retrieved** | `data/real/raw/globallandslides.csv` |
| **Download Date** | 2026-09-02 (during project ingestion) |
| **License** | NASA Open Data (Public Domain) |
| **Geographic Coverage** | Global |
| **NER Spatial Filter** | Bounding box: Lat [21.5°N, 29.5°N], Lon [88.0°E, 97.5°E] |
| **Temporal Coverage (Global)** | 2007 – 2016 |
| **Temporal Coverage (NER)** | 2011 – 2016 |
| **Temporal Resolution** | Event-level (date stamps; all 177 NER records have `date_precision = 'day'`) |
| **Spatial Resolution** | Point events with GPS coordinates; matched to 8 NER zone corridors (≤35 km radius) |
| **Units** | Event occurrence dates (UTC), fatality counts, trigger categories |
| **Raw Records (Global)** | 11,033 global events |
| **NER Events (after spatial filter)** | 177 confirmed landslide events |
| **Ingestion Module** | `ml/ingestion/real/landslide_glc.py` |
| **Output Artifact** | `data/real/processed/real_ner_events.pkl` (177 × 17 columns) |

### Distribution by Year
| Year | Events |
|---|---|
| 2011 | 30 |
| 2012 | 17 |
| 2013 | 32 |
| 2014 | 17 |
| 2015 | 58 |
| 2016 | 23 |
| **Total** | **177** |

### Distribution by Zone
| Zone ID | Zone Name | Events |
|---|---|---|
| `REAL-NER-001` | Guwahati Hills Corridor, Assam | 37 |
| `REAL-NER-002` | Shillong Plateau / Sohra, Meghalaya | 4 |
| `REAL-NER-003` | Imphal - Senapati NH-2, Manipur | 28 |
| `REAL-NER-004` | Kohima - Phek Ridge, Nagaland | 38 |
| `REAL-NER-005` | Aizawl Mountain Slopes, Mizoram | 12 |
| `REAL-NER-006` | Bhalukpong - Tawang, Arunachal Pradesh | 4 |
| `REAL-NER-007` | Atharamura Hill Range, Tripura | 1 |
| `REAL-NER-008` | Gangtok - Teesta Valley, Sikkim | 53 |

---

## 2. Dataset 2 — Rainfall / Precipitation

| Field | Value |
|---|---|
| **Source Organization** | European Centre for Medium-Range Weather Forecasts (ECMWF) |
| **Dataset / Product** | ERA5-Land Hourly Data on Single Levels — Total Precipitation |
| **API Endpoint** | Open-Meteo Historical API (ERA5-Land backend): https://archive-api.open-meteo.com/v1/archive |
| **Variables Fetched** | `precipitation` (mm/h) |
| **Download Date** | 2026-08-20 to 2026-08-22 (progressive zone-by-zone retrieval) |
| **License** | Copernicus Climate Change Service (C3S) — Free for non-commercial use with attribution |
| **Attribution** | ERA5-Land hourly data from 1950 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). DOI:10.24381/cds.68d2bb30 |
| **Geographic Coverage** | 8 NER zone centroids (26.18°N–27.33°N, 88.61°E–94.12°E) |
| **Temporal Coverage** | 2011-01-01 00:00 UTC to 2016-10-15 23:00 UTC |
| **Temporal Resolution** | Hourly |
| **Spatial Resolution** | ~0.1° × 0.1° (~9 km) grid point nearest to centroid |
| **Units** | mm/h (total precipitation) |
| **Derived Features** | acc_1h, acc_3h, acc_6h, acc_12h, acc_24h, acc_48h, acc_72h, intensity_max_1h, dry_hours_streak, monsoon_flag |
| **Ingestion Module** | `ml/ingestion/real/rainfall_openmeteo.py` |
| **Raw Records Ingested** | 406,080 hourly records (8 zones × 50,760 hours) |

---

## 3. Dataset 3 — Surface Meteorology

| Field | Value |
|---|---|
| **Source Organization** | ECMWF |
| **Dataset / Product** | ERA5-Land Hourly — Temperature, Humidity, Wind Speed, Surface Pressure |
| **API Endpoint** | Open-Meteo Historical API (ERA5-Land backend): https://archive-api.open-meteo.com/v1/archive |
| **Variables Fetched** | `temperature_2m` (°C), `relative_humidity_2m` (%), `wind_speed_10m` (m/s), `surface_pressure` (hPa) |
| **Download Date** | 2026-08-20 to 2026-08-22 |
| **License** | Copernicus C3S — Free for non-commercial with attribution |
| **Geographic Coverage** | 8 NER zone centroids |
| **Temporal Coverage** | 2011-01-01 00:00 UTC to 2016-10-15 23:00 UTC |
| **Temporal Resolution** | Hourly |
| **Units** | °C, %, m/s, hPa |
| **Ingestion Module** | `ml/ingestion/real/weather_openmeteo.py` |
| **Raw Records Ingested** | 406,080 hourly records |

---

## 4. Dataset 4 — Soil Moisture

| Field | Value |
|---|---|
| **Source Organization** | ECMWF |
| **Dataset / Product** | ERA5-Land Hourly — Volumetric Soil Water Layer 1 (0–7 cm depth) |
| **API Endpoint** | Open-Meteo Historical API (ERA5-Land backend): https://archive-api.open-meteo.com/v1/archive |
| **Variables Fetched** | `soil_moisture_0_to_7cm` (m³/m³) |
| **Download Date** | 2026-08-20 to 2026-08-22 |
| **License** | Copernicus C3S — Free for non-commercial with attribution |
| **Geographic Coverage** | 8 NER zone centroids |
| **Temporal Coverage** | 2011-01-01 00:00 UTC to 2016-10-15 23:00 UTC |
| **Temporal Resolution** | Hourly |
| **Units** | m³/m³ (volumetric water content) |
| **Derived Features** | sm_volumetric, swi (Soil Wetness Index), pore_pressure_proxy, stability_indicator |
| **Ingestion Module** | `ml/ingestion/real/soil_moisture_real.py` |
| **Raw Records Ingested** | 406,080 hourly records |

---

## 5. Dataset 5 — Digital Elevation Model (Terrain)

| Field | Value |
|---|---|
| **Source Organization** | European Space Agency (ESA) / Copernicus Land Service |
| **Dataset / Product** | Copernicus DEM GLO-30 (Global 30m Digital Elevation Model) |
| **Original URL** | https://registry.opendata.aws/copernicus-dem/ (AWS S3: `s3://copernicus-dem-30m/`) |
| **Tile Naming** | `Copernicus_DSM_COG_10_N{lat}_00_E{lon}_00_DEM.tif` (1°×1° GeoTIFF) |
| **Download Date** | 2026-08-27 (during terrain pipeline execution) |
| **License** | ESA Copernicus DEM — Free unrestricted use with attribution |
| **Attribution** | © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved. |
| **Geographic Coverage** | 8 tiles covering NER corridors: N23E088, N23E091, N23E092, N24E093, N25E091, N25E094, N26E091, N27E088 |
| **Temporal Coverage** | Static terrain product (2010–2014 SRTM + additional stereo processing) |
| **Spatial Resolution** | 30 metres (1 arc-second) |
| **CRS** | EPSG:4326 (WGS84 geographic) |
| **Tile Sizes (MB)** | 28.9–55.8 MB per 1°×1° tile; 351.75 MB total |
| **Derived Features** | elevation_m, slope_deg, aspect_deg, curvature, tpi (Topographic Position Index), twi (Topographic Wetness Index) |
| **Derivation Algorithm** | Horn (1981) gradient for slope; TWI = ln(A/tan(β)); TPI = elev − mean(neighbors) |
| **Ingestion Module** | `ml/ingestion/real/terrain_real.py` |
| **Raw Rasters** | `data/real/raw/terrain/*.tif` (8 files) |
| **Output Artifact** | `data/real/processed/real_ner_terrain.pkl` (8 zones × 7 terrain features) |
| **Validation** | CRS verified EPSG:4326, resolution 0.000278° = 30.87m at equator, NoData values preserved |

### Terrain Values by Zone
| Zone ID | Elevation (m) | Slope (°) | Aspect (°) | Curvature | TPI | TWI |
|---|---|---|---|---|---|---|
| `REAL-NER-001` (Guwahati) | 55.6 | 2.98 | 176.1 | 0.00000 | -0.000 | 10.78 |
| `REAL-NER-002` (Shillong) | 1,599.5 | 18.23 | 188.2 | 0.00032 | 0.047 | 8.94 |
| `REAL-NER-003` (Imphal) | 786.8 | 1.37 | 215.5 | -0.00018 | -0.051 | 11.56 |
| `REAL-NER-004` (Kohima) | 1,324.4 | 22.24 | 183.2 | 0.00068 | 0.107 | 8.72 |
| `REAL-NER-005` (Aizawl) | 535.7 | 25.03 | 183.6 | 0.00132 | 0.154 | 8.59 |
| `REAL-NER-006` (Bhalukpong) | 2,111.7 | 30.07 | 188.6 | 0.00063 | 0.147 | 8.37 |
| `REAL-NER-007` (Atharamura) | 118.4 | 11.14 | 176.7 | -0.00003 | -0.131 | 9.45 |
| `REAL-NER-008` (Sikkim) | 1,599.6 | 28.73 | 161.0 | -0.00005 | -0.034 | 8.43 |

---

## 6. Dataset 6 — InSAR / Satellite SAR

| Field | Value |
|---|---|
| **Source Organization** | European Space Agency (ESA) / Copernicus; Alaska Satellite Facility (ASF) DAAC |
| **Dataset / Product** | Sentinel-1A C-SAR IW SLC (Level-1 Single Look Complex) |
| **Catalog Source** | ASF DAAC Search API: https://api.daac.asf.alaska.edu |
| **Sensor** | C-SAR (5.405 GHz, λ = 5.565 cm) |
| **Acquisition Mode** | Interferometric Wide (IW) TOPSAR |
| **Geographic Coverage** | 8 NER zones — 452 scenes verified (100% footprint spatial containment) |
| **Temporal Coverage** | 2015-01-01 to 2016-10-14 |
| **Number of Scenes** | 452 genuine Sentinel-1A acquisitions |
| **InSAR Processing Status** | **UNAVAILABLE / OFF** — Interferometric deformation not processed |
| **Scientific Justification** | C-band coherence < 0.20 in dense sub-tropical vegetation (threshold ≥ 0.35 for unwrapping); LiCSAR server unreachable; raw SLC processing requires GPU/HPC |
| **Deformation Values** | All NaN (`insar_valid = False`) |
| **Synthetic Deformation** | **NONE — strictly zero** |
| **Ingestion Module** | `ml/ingestion/real/insar_real.py` |
| **Output Artifact** | `data/real/processed/real_ner_insar.pkl` (452 × 24 columns of scene metadata) |

---

## 7. Label Construction

```
Context window: 168 hours (7 days)
Target horizon: 24 hours (configurable to 48h)
Window stride: 24 hours (non-overlapping daily forecast steps)

Label rules (LabelBuilder):
  y = 1  if ≥ 1 verified event in [context_end, context_end + 24h]
  y = 0  if no event in [context_end - 72h, context_end + 24h]
  y = -1 (excluded) if ambiguous pre-event window
           (i.e. event within 72h before context_end)

Date precision policy:
  'day'   → INCLUDED with full 24h event window
  'month' → EXCLUDED (too imprecise for hourly label construction)
  'year'  → EXCLUDED
  'unknown' → EXCLUDED

All 177 NER events have date_precision = 'day' — all are usable.
```

### Split Statistics (Seed 42, 100% Labels, Terrain Enabled)
| Split | Date Range (UTC) | Windows | Positives | Negatives | Excluded | Pos Rate |
|---|---|---|---|---|---|---|
| **Train** | 2011-01-01 to 2014-12-31 | 11,440 | 76 | 11,296 | ~321 | 0.66% |
| **Validation** | 2015-01-01 to 2015-12-31 | 2,842 | 35 | 2,799 | ~73 | 1.23% |
| **Test** | 2016-01-01 to 2016-10-15 | 2,261 | 18 | 2,238 | ~41 | 0.80% |

> [!CAUTION]
> The test holdout contains 18 positive windows from 2016 landslide events. With 23 raw GLC events in 2016, 5 are excluded due to the 72h ambiguity buffer (they fall within 72h of an adjacent window boundary). Statistical interpretations of results should account for this small absolute positive count.

---

## 8. Anti-Contamination Declarations

- ✅ **No synthetic data** enters the scientific benchmark pipeline
- ✅ **No demo providers** (`ml/ingestion/demo/`) are invoked during benchmark runs
- ✅ **No future data leakage**: all windows are causal (context strictly before target horizon)
- ✅ **No event leakage**: 72h pre-event buffer prevents positive events from contaminating negative windows
- ✅ **No InSAR fabrication**: deformation = NaN, insar_valid = False, quality_flag = "unprocessed_interferograms_vegetation_decorrelation"
- ✅ **No elevation API point-sampling**: terrain comes exclusively from verified Copernicus DEM 30m rasters
