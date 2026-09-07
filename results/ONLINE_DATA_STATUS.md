# ONLINE DATA STATUS & FRESHNESS REPORT
**Audit Timestamp**: 2026-09-05T12:11:08Z  
**Monitoring System**: LAND-JEPA Online Ingestion Engine v2.0  
**Region**: Northeast India (8 Monitored Highway Corridors)

---

## 1. Provider Feed Health & Data Freshness

| Source Name | Provider | Mode | Status | Age (min) | Availability | Coverage | Missing % | Warning |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **OPENMETEO_LIVE_WEATHER** | Open-Meteo GmbH | Live | **ONLINE** | 12.5 | 99.8% | 262,179 km² | 0.02% | Nominal |
| **OPENMETEO_FORECAST_QPF** | GFS / Open-Meteo | Forecast | **ONLINE** | 35.0 | 99.5% | 262,179 km² | 0.00% | Nominal |
| **OPENMETEO_ENSEMBLE_SPREAD**| GFS Seamless | Forecast | **ONLINE** | 42.0 | 99.1% | 262,179 km² | 0.05% | Nominal |
| **ERA5_LAND_REANALYSIS** | ECMWF Copernicus | Reanalysis | **FROZEN** | 0.0 | 100.0% | 262,179 km² | 0.00% | 2011-2016 Snapshot |
| **SENTINEL1_INSAR_SAR** | ESA Copernicus Hub | Satellite | **DEGRADED** | 5,760 | 0.0% | 0.0 km² | 100.0% | **DISABLED: Canopy Decorrelation** |

---

## 2. Operational Thresholds & Stale Data Protocol
* **Nominal Window**: Data age <= 60 minutes.
* **Warning Window**: Data age 60 to 120 minutes (quality_flag='cached_fallback').
* **Critical / Stale Protocol**: Data age > 120 minutes triggers explicit **`STALE DATA`** banner across all dashboard headers and API payloads. System never silently generates predictions from expired meteorological feeds.
