# LAND-JEPA — Data Sources

## Overview

This document lists all data sources used or planned for LAND-JEPA, their availability,
licensing considerations, access methods, and the current integration status.

**Key rule**: If a real data source is unavailable, an explicit adapter interface exists
and demo/synthetic data is used for software integration testing ONLY. Demo data is
always clearly labelled `is_demo=true` in the database and flagged in API responses.

---

## Core Data Sources

### 1. Rainfall / Precipitation

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | IMD (India Meteorological Department) gridded rainfall    |
| Alternative    | CHIRPS (Climate Hazards Group InfraRed Precipitation)     |
| Alternative    | ERA5-Land (ECMWF Reanalysis, via CDS API)                 |
| Resolution     | IMD: 0.25° grid; CHIRPS: 0.05°; ERA5: 0.1°              |
| Temporal res.  | Daily (IMD), near-daily (CHIRPS), hourly (ERA5)          |
| Coverage       | Pan-India; NER specifically well-covered by IMD           |
| License        | IMD: government data (check current terms); CHIRPS: open  |
| Access         | IMD API / FTP; CHIRPS via Google Earth Engine / direct   |
| Demo fallback  | Synthetic seasonal sine wave + Gaussian noise, clearly labelled |

### 2. Weather (Temperature, Humidity, Wind)

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | Open-Meteo API (free, open-source, no API key required)   |
| Alternative    | ERA5-Land via CDS API                                     |
| Alternative    | IMD weather station data                                  |
| Resolution     | Open-Meteo: 1–11 km grid, hourly                         |
| Coverage       | Global                                                    |
| License        | Open-Meteo: open, CC BY 4.0                              |
| Access         | REST API: `https://api.open-meteo.com/v1/forecast`        |
| Demo fallback  | Synthetic weather time series                             |

### 3. Soil Moisture

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source (prim.) | ESA CCI Soil Moisture (COMBINED product)                  |
| Alternative    | ERA5-Land volumetric soil water                           |
| Alternative    | SMAP Level-3 (NASA, requires Earthdata account)           |
| Resolution     | ESA CCI: 0.25°; ERA5: 0.1°; SMAP: ~36 km                |
| Temporal res.  | Daily (ESA CCI); hourly (ERA5)                           |
| License        | ESA CCI: open (CC BY); ERA5: free with registration      |
| Access         | ESA CCI via Copernicus Data Space; ERA5 via CDS API       |
| Demo fallback  | Physics proxy: rainfall-based SWI estimate               |

### 4. DEM / Terrain (Elevation, Slope, Aspect, Curvature)

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source (prim.) | SRTM 30m (NASA Shuttle Radar Topography Mission)          |
| Alternative    | ALOS AW3D30 (12.5m, JAXA)                                |
| Alternative    | Copernicus DEM GLO-30 (30m, ESA/European Commission)      |
| Static data    | Terrain is static (no refresh needed for slope/aspect)   |
| License        | SRTM: open (NASA); ALOS: open; Copernicus DEM: free      |
| Access         | USGS EarthExplorer; Google Earth Engine; OpenTopography   |
| Derived layers | slope, aspect, curvature, TPI, TWI (computed via GDAL/richdem) |
| Demo fallback  | Synthetic DEM for NER representative terrain class       |

### 5. Historical Landslide Events

| Property       | Details                                                                    |
|----------------|----------------------------------------------------------------------------|
| Source (prim.) | Bhuvan Landslide Atlas (ISRO)                                              |
| Alternative    | NDMA historical records                                                    |
| Alternative    | Global Landslide Catalog (NASA GLC)                                        |
| Alternative    | GSI Landslide Susceptibility Map data                                      |
| Alternative    | Published academic databases (Kirschbaum et al., 2010+)                    |
| Known issue    | Many events have only date precision (not exact time). Document carefully. |
| License        | Bhuvan: open government; NASA GLC: open                                    |
| Access         | Bhuvan portal; NASA EarthData; direct dataset download                     |
| Demo fallback  | Synthetic event set with explicit `is_demo=true` flag                     |

**CRITICAL NOTE**: Historical landslide records often have:
- Unknown exact times (only date or month)
- Uncertain locations (village-level, not GPS point)
- Missing magnitude or casualty data

The `landslide_events.date_precision` field captures this uncertainty.
Label construction explicitly accounts for it.

---

## Optional Data Sources

### 6. Sentinel-1 / InSAR Deformation

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | Sentinel-1 SAR (ESA Copernicus Programme)                 |
| Revisit        | 6–12 days (not hourly)                                   |
| Processing     | Requires SBAS/PS-InSAR processing (SNAP, ISCE, MintPy)   |
| Output         | LOS displacement rasters in mm                           |
| License        | Free (Copernicus Sentinel open data policy)              |
| Access         | Copernicus Data Space Ecosystem                          |
| Integration    | Optional adapter; system operates without it             |
| Demo fallback  | Synthetic deformation raster, clearly labelled           |

**Note**: Raw Sentinel-1 InSAR processing is NOT performed by this platform.
The platform consumes pre-processed deformation products.

### 7. Sentinel-2 Optical Imagery

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | Sentinel-2 MSI (ESA Copernicus Programme)                 |
| Resolution     | 10–60m, 5-day revisit                                    |
| Use case       | Post-event damage assessment, land cover change          |
| Integration    | Optional via SatelliteProvider adapter                   |
| Demo fallback  | Not included in initial demo                             |

### 8. River / Water Level Data

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | CWC (Central Water Commission), India-WRIS               |
| Coverage       | Major rivers in NER (Brahmaputra system)                 |
| Temporal res.  | Daily                                                    |
| License        | Government data; check current terms                     |
| Access         | India-WRIS portal                                        |
| Integration    | Optional adapter (not in Checkpoint 1–8)                 |

### 9. Citizen Observations

| Property       | Details                                                   |
|----------------|-----------------------------------------------------------|
| Source         | LAND-JEPA citizen reporting module                       |
| Content        | Photo, GPS location, category, description               |
| Human review   | All reports require human verification before use        |
| Integration    | Built-in (Phase 18)                                      |

---

## Demo Data Policy

All demo/synthetic data used for software integration testing:

1. Is stored with `is_demo = TRUE` in the database.
2. Is tagged with `data_source = 'DEMO'` or similar explicit label.
3. Is returned in API responses with a `"demo": true` field.
4. Is shown with a visible "DEMO DATA" banner in all UI components.
5. Is stored in `data/demo/` and NEVER mixed with `data/raw/` or `data/processed/`.
6. Is documented in `data/demo/README.md` with generation parameters.

---

## Data Licensing Summary

| Source           | License              | Registration Required |
|------------------|----------------------|-----------------------|
| Open-Meteo       | CC BY 4.0            | No                    |
| CHIRPS           | Open / CC            | No                    |
| ERA5 / CDS API   | Copernicus License   | Yes (free)            |
| ESA CCI SM       | CC BY               | No                    |
| SRTM             | Open (NASA)          | No                    |
| ALOS AW3D30      | Open (JAXA)          | Yes (free)            |
| Copernicus DEM   | Open (EC)            | No                    |
| Bhuvan           | Open Government Data | No (check portal)     |
| NASA GLC         | Open                 | No                    |
| Sentinel-1/2     | Copernicus Open      | Yes (free)            |
| CWC / WRIS       | Government           | Check current terms   |

---

## Data Refresh Schedule (Configurable)

| Data type       | Default refresh    | Configured in          |
|-----------------|--------------------|------------------------|
| Rainfall        | Every 1 hour       | `scripts/scheduler.py` |
| Weather         | Every 1 hour       | `scripts/scheduler.py` |
| Soil moisture   | Every 24 hours     | `scripts/scheduler.py` |
| Risk prediction | Every 1 hour       | `scripts/scheduler.py` |
| InSAR           | Manual / 6–12 days | `scripts/scheduler.py` |
| Terrain         | Manual (static)    | Not scheduled          |

All schedules are configurable via environment variables and not hard-coded.
