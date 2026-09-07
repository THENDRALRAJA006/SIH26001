# LAND-JEPA Daily System Health & Integration Diagnostic Report

**Generated**: 2026-09-07T17:04:54.935738+00:00  
**Overall System Status**: `DEGRADED`  
**Application Version**: `0.1.0`  
**Uptime**: `35446.1s`  
**Diagnostic Latency**: `7559.56ms`  

---

## 1. Executive Summary

- **Components Audited**: 41 / 30
- **Critical Failures**: 0 (None)
- **Degraded / Optional Offline Services**: 4 (Database, MapTiler, Sentinel-1/InSAR, INSAR PROCESSING)

---

## 2. Full Component Health Matrix

| Component | Category | Status | Latency (ms) | Data Age | Version | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Frontend** | Presentation & UI | * ONLINE | 0.49 | < 1s | v3.2.0-PROD | React 18 + Vite 8 |
| **Backend API** | Core Gateway | * ONLINE | 0.0 | < 1s | 0.1.0 | FastAPI (Async ASGI) |
| **Database** | Persistence & Storage | * DEGRADED | 807.27 | < 1s | PostgreSQL 15 / SQLite Fallback | Local Ledger / Memory Store Fallback |
| **Authentication** | Security & Identity | * ONLINE | 0.23 | < 1s | JWT HS256 | HS256 |
| **JWT/RBAC** | Security & Identity | * ONLINE | 0.12 | < 1s | Role-Based Access v2.5 | ALLOW (HTTP 200) |
| **LAND-JEPA model** | AI Inference | * ONLINE | 22.23 | < 1s | v2.5 Production | 1b877ff4a498caad |
| **v2.5 production model** | AI Inference | * ONLINE | 0.25 | Frozen Champion | v2.5-TRIGGER-AWARE-CHAMPION | ACTIVE PRODUCTION BENCHMARK |
| **v2.6.1 challenger** | AI Inference | * ONLINE | 0.38 | Shadow Mode Quarantined | v2.6.1-CHALLENGER | FROZEN PROSPECTIVE CHALLENGER |
| **Weather provider** | Atmospheric Telemetry | * ONLINE | 1132.22 | < 15 min | Open-Meteo API v1 | 19.5 |
| **Forecast provider** | Atmospheric Telemetry | * ONLINE | 0.0 | < 30 min | 72h Multi-Horizon QPF | ['6h', '12h', '24h', '48h', '72h'] |
| **Soil data** | Geotechnical Telemetry | * ONLINE | 0.0 | < 1 hour | ERA5-Land / Soil Moisture v2 | ['0-7cm volumetric', '7-28cm root-zone', 'saturation_pct'] |
| **Terrain** | Spatial Infrastructure | * ONLINE | 0.16 | Static Baseline | Copernicus GLO-30m DEM | 30 meters |
| **GIS** | Spatial Infrastructure | * ONLINE | 0.01 | Continuous | GeoJSON EPSG:4326 | 8 |
| **MapTiler** | Cartography & Basemaps | * DEGRADED | 0.01 | Live Vector Tiles | MapTiler Cloud Vector | False |
| **Sentinel-1/InSAR** | Satellite Geodesy | ** UNAVAILABLE | 7.83 | 12-day orbital repeat | Copernicus S1A C-Band SAR | DECORRELATED (coherence < 0.20) |
| **Seismic/PGA** | Geotechnical Telemetry | * ONLINE | 0.0 | Static Hazard Prior + Real-time Stream | GSI Zone V / USGS Earthquake Catalog | Zone V (Highest Indian Seismic Hazard) |
| **TECTONIC DATA** | Geological & Geodesy Infrastructure | * ONLINE | 0.04 | Annual Prior (2026 Reference) | ITRF2014 / GSI Geodetic Prior | 38.2 |
| **FAULT DATA** | Geological & Geodesy Infrastructure | * ONLINE | 0.01 | Static Geological Prior | GSI Seismotectonic Atlas NER | 8 |
| **SEISMIC DATA** | Geological & Geodesy Infrastructure | * ONLINE | 0.07 | 77.1h since last event | NCS India / USGS Real-time Catalog | 1 |
| **SENTINEL-1** | Geological & Geodesy Infrastructure | * ONLINE | 5570.43 | 1 days ago | Copernicus Sentinel-1 C-SAR | 5 |
| **INSAR PROCESSING** | Geological & Geodesy Infrastructure | * DEGRADED | 0.34 | 13.1 days | Two-Pass MCF InSAR Pipeline | LOW_COHERENCE_MARGINAL |
| **Road GIS** | Spatial Infrastructure | * ONLINE | 0.0 | Verified 2026 | BRO / NHAI Highway Vectors | 50m toe cut & 200m crest slope buffer |
| **Drainage/Culvert** | Hydrological Infrastructure | * ONLINE | 0.0 | Verified 2026 | HydroSHEDS / Culvert Crossings | Active physical gate in v2.6.1 |
| **Alert engine** | Civil Safety Dispatch | * ONLINE | 0.26 | < 1s | NDMA 3-Tier Alert Engine | TEST_ALERT evaluated and rolled back safely |
| **Citizen reporting** | Civil Safety Dispatch | * ONLINE | 0.02 | < 1s | Crowdsourced Incident Ingestion v2 | TEST_CITIZEN_REPORT ingested and cleared |
| **Offline sync** | Data Reliability | * ONLINE | 0.37 | < 1s | Append-Only Local Storage Sync | Client IndexedDB / localStorage fallback |
| **Notification service** | Civil Safety Dispatch | * ONLINE | 0.37 | < 1s | Multi-Channel Broadcast Dispatch v3.0 | ['SMS (DLT)', 'Push (VAPID/FCM)', 'In-App'] |
| **SMS Provider** | Civil Safety Dispatch | * OFFLINE | 0.01 | < 1s | Provider: NONE (DLT Compliant) | CREDENTIALS_MISSING |
| **Push Provider** | Civil Safety Dispatch | * OFFLINE | 0.01 | < 1s | Protocol: WEBPUSH | webpush |
| **Notification Queue** | Civil Safety Dispatch | * ONLINE | 0.01 | < 1s | In-Memory + JSONL Append-Only Queue | 0 |
| **Webhook** | Civil Safety Dispatch | * ONLINE | 0.0 | < 1s | HMAC/Secret Verified Delivery Receipts | /api/v1/notifications/webhook |
| **Template Service** | Civil Safety Dispatch | * ONLINE | 0.01 | < 1s | TRAI DLT Template Registry | ['English', 'Hindi', 'Assamese', 'Bengali', 'Manipuri'] |
| **Translation/i18n** | User Experience & Presentation | * ONLINE | 3.04 | Static Parity | i18n 5-Language Regional Matrix | ['English', 'Hindi', 'Assamese', 'Bengali', 'Manipuri'] |
| **Prediction ledger** | Persistence & Storage | * ONLINE | 0.36 | < 1 hour | Append-Only JSONL v2.5 | 64 |
| **Audit log** | Persistence & Storage | * ONLINE | 0.23 | < 1s | Immutable Audit Trail v3.2 | 68 |
| **Scheduled jobs** | Core Gateway | * ONLINE | 0.0 | 30s Cadence | AsyncIO Cron Poller | {'critical_infrastructure': 'every 5 min', 'external_telemetry': 'every 10 min', 'shadow_validation': 'every 30 min', 'daily_report': 'once daily'} |
| **Cache** | Data Reliability | * ONLINE | 0.01 | < 1s | In-Memory LRU Cache | 0 |
| **Storage** | Persistence & Storage | * ONLINE | 0.07 | < 1s | Local NVMe / SSD Storage | 173.83 |
| **Docker services** | Container & Deployment | * ONLINE | 0.13 | Configured | Docker Compose v2 | ['backend', 'frontend', 'postgres', 'nginx'] |
| **Reverse proxy** | Container & Deployment | * ONLINE | 0.12 | Configured | NGINX / Vite Gateway | 60 req/min per IP |
| **Prediction pipeline** | AI Inference | * ONLINE | 9.6 | < 1s | LAND-JEPA Multi-Horizon v2.5 | SYSTEM HEALTH TEST |

---

## 3. Model Governance & Scientific Rigor

- **v2.5-TRIGGER-AWARE-CHAMPION**: Verified active production benchmark checkpoint.
- **v2.6.1-CHALLENGER**: Verified frozen prospective challenger in quarantined shadow mode (`0.6531`, `0.7724`, `0.9550`).
- **Zero Synthetic Contamination**: Sentinel-1 InSAR accurately reflects vegetation decorrelation rather than fabricated deformation.
- **Causality Constraint**: All observations and forecasts satisfy `timestamp <= prediction_time`.
