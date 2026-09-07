# LAND-JEPA Incident Response & Health Runbook

**Project**: LAND-JEPA AI Landslide Early Warning System  
**SIH Problem ID**: SIH26001 | **Team**: ZAIX  
**Coverage**: 8 Strategic Highway Corridors, Northeast India (NER)  

---

## 1. Quick Reference: System Status Levels

| Status Level | Indicator Color | Officer Topbar Icon | System Action | Operator Action |
| :--- | :--- | :--- | :--- | :--- |
| **OPERATIONAL** | 🟢 Green | `● SYSTEM OPERATIONAL` | Full automated multi-horizon inference and telemetry active | Normal surveillance operations |
| **DEGRADED** | 🟡 Amber | `⚠ SYSTEM DEGRADED` | Primary model active; secondary telemetries on verified fallback | Review degraded card on `/system-status`; verify backup data |
| **CRITICAL** | 🔴 Red | `✖ SYSTEM CRITICAL` | Automated dispatches suspended to prevent erroneous alerts | Immediate triage required; execute Runbook Section 3 |

---

## 2. Common Remediation Procedures

### Incident 01: Database Degraded (PostgreSQL Unreachable)
- **Symptom**: `Database` card shows `DEGRADED | PostgreSQL port 5432 not listening; local transaction ledger fallback active`.
- **Root Cause**: The local PostgreSQL service is either stopped or in container restart loop.
- **Automated Fallback**: The backend automatically persists transactions to `results/predictions_ledger.jsonl`, `results/UI_INTERACTION_AUDIT.csv`, and in-memory stores. Zero data is lost.
- **Remediation Steps**:
  1. Check PostgreSQL container status:
     ```bash
     docker ps -a | grep postgres
     ```
  2. Start service if offline:
     ```bash
     docker start landjepa_postgres || net start postgresql-x64-15
     ```
  3. Verify connection:
     ```bash
     python scripts/run_full_system_health_check.py
     ```

---

### Incident 02: Sentinel-1 InSAR Flagged as UNAVAILABLE
- **Symptom**: `Sentinel-1/InSAR` card displays `UNAVAILABLE | DECORRELATED (coherence < 0.20)`.
- **Root Cause**: Scientific constraint enforcement. The Northeast Indian Himalayan corridors feature dense sub-tropical vegetation cover where C-band SAR experiences temporal decorrelation.
- **Automated Policy**:
  > [!IMPORTANT]
  > **DO NOT ATTEMPT TO FORCE THIS STATUS TO ONLINE.**
  > The platform strictly prohibits fabricating synthetic deformation numbers. The multi-modal fusion layer automatically gates InSAR tokens OFF and relies on verified geotechnical slope stability rasters and rainfall triggers.
- **Operator Verification**:
  - Confirm in `ml/ingestion/real/insar_real.py` that `INSAR_AVAILABLE = False`.
  - Confirm the UI displays the transparent disclosure banner.

---

### Incident 03: MapTiler Degraded (CARTO Basemap Active)
- **Symptom**: `MapTiler` card shows `DEGRADED | CARTO Positron / Dark Matter basemaps active`.
- **Root Cause**: The environment variable `VITE_MAPTILER_API_KEY` is not set or the key exceeded quota.
- **Automated Fallback**: Leaflet automatically falls back to CARTO dark/light GIS basemaps without throwing errors.
- **Remediation Steps**:
  1. Set the key in `frontend/dashboard/.env`:
     ```env
     VITE_MAPTILER_API_KEY=your_key_here
     ```
  2. Rebuild frontend bundle:
     ```bash
     cd frontend/dashboard && npm run build
     ```

---

### Incident 04: Weather Provider Timeout
- **Symptom**: `Weather provider` shows `DEGRADED | HTTP Timeout`.
- **Root Cause**: Transient upstream network latency to `api.open-meteo.com`.
- **Automated Fallback**: System automatically switches to local ERA5-Land reanalysis climatology for the corridor.
- **Remediation Steps**:
  1. Test external connectivity from server:
     ```bash
     curl -I "https://api.open-meteo.com/v1/forecast?latitude=25.5788&longitude=91.8933"
     ```
  2. If DNS failure, switch local DNS resolver to `1.1.1.1` or `8.8.8.8`.

---

### Incident 05: Causality Violation (Critical Failure)
- **Symptom**: `Forecast provider` or `Prediction pipeline` flags `CRITICAL | Temporal Causality Violation`.
- **Root Cause**: Data packet timestamp is ahead of server UTC wall-clock ($t_{\text{observation}} > t_{\text{prediction}}$).
- **Automated Safety Action**: The prediction engine halts operational inference to prevent look-ahead bias and false warnings.
- **Remediation Steps**:
  1. Check system NTP clock synchronization:
     ```bash
     w32tm /query /status || timedatectl
     ```
  2. Sync clock with Indian Standard Time NTP pool (`pool.ntp.org`).

---

## 3. Automated Diagnostic Verification

To run a full self-diagnostic sweep and verify all 30 components from the command line:

```bash
python scripts/run_full_system_health_check.py
```

Expected output ends with:
```
LAND-JEPA Frontend         ✅
LAND-JEPA Backend          ✅
LAND-JEPA Database         ⚠️ DEGRADED / OPTIONAL FALLBACK
LAND-JEPA AI Model         ✅
LAND-JEPA Weather          ✅
LAND-JEPA Forecast         ✅
LAND-JEPA GIS              ✅
LAND-JEPA MapTiler         ⚠️ DEGRADED / OPTIONAL FALLBACK
LAND-JEPA InSAR            ⚠️ UNAVAILABLE / OPTIONAL FALLBACK
LAND-JEPA Seismic          ✅
LAND-JEPA Alerts           ✅
LAND-JEPA Citizen          ✅
LAND-JEPA Officer          ✅
LAND-JEPA Translation      ✅
LAND-JEPA Prediction       ✅
LAND-JEPA Security         ✅
LAND-JEPA E2E              ✅
====================================================================
Overall Status: OPERATIONAL / DEGRADED
```
