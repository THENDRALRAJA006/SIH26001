# LAND-JEPA System Health & Diagnostic Architecture

**Project**: LAND-JEPA AI Landslide Early Warning System  
**SIH Problem ID**: SIH26001 | **Team**: ZAIX  
**Coverage**: 8 Strategic Highway Corridors, Northeast India (NER)  

---

## 1. Architectural Philosophy & Zero-Fake Mandate

The LAND-JEPA Health & Integration Monitoring Subsystem is engineered around a strict core principle: **Scientific Honesty and Operational Realism**. 

Unlike conventional health checks that merely check if an HTTP port responds or if a static page renders, LAND-JEPA implements active, multi-layer diagnostic probes that continuously verify:
- Actual mathematical forward passes of deep vision-temporal encoders.
- Numerical validity (asserting probabilities $\in [0, 1]$, zero `NaN` or `Inf`).
- Genuine external meteorological feeds (Open-Meteo 11km NWP).
- Strict non-fabrication disclosure of satellite radar (Sentinel-1 InSAR decorrelation due to dense Himalayan canopy).
- Causality integrity ($t_{\text{obs}} \le t_{\text{pred}}$, $t_{\text{issue}} \le t_{\text{pred}}$, $t_{\text{SAR}} \le t_{\text{pred}}$).
- Non-polluting database transaction isolation using `BEGIN` $\to$ canary write $\to$ `ROLLBACK`.

```mermaid
graph TD
    A[Diagnostic Initiator / Automated Poller] --> B[SystemHealthService Singleton]
    
    subgraph "Layer 1: Edge & Client Presentation"
        B --> C1[Frontend Bundle & Routes Probe]
        B --> C2[Reverse Proxy & CORS Probe]
        B --> C3[i18n 5-Language Dictionary Parity Probe]
    end
    
    subgraph "Layer 2: Core Gateway & Security"
        B --> D1[Backend FastAPI Uptime & Latency]
        B --> D2[JWT Signature & Token Lifecycle]
        B --> D3[RBAC Permission Boundary Denials]
        B --> D4[Storage & Audit Log Write Integrity]
    end

    subgraph "Layer 3: Scientific AI & Physics Engines"
        B --> E1[v2.5 Champion Checkpoint & Weights Hash]
        B --> E2[v2.6.1 Frozen Challenger Configuration]
        B --> E3[Forward Pass Logits & NaN/Inf Verification]
        B --> E4[End-to-End Synthetic Prediction Cycle]
        B --> E5[Strict Temporal Causality Verification]
    end

    subgraph "Layer 4: Real Geotechnical & Spatial Data"
        B --> F1[Open-Meteo High-Resolution NWP Weather]
        B --> F2[72h Quantitative Precipitation Forecast]
        B --> F3[ERA5-Land Root-Zone Soil Saturation]
        B --> F4[Copernicus 30m Global DEM Rasters]
        B --> F5[BRO Highway Corridors & Road Cut Buffers]
        B --> F6[Sentinel-1 SAR Catalog & Decorrelation Flag]
        B --> F7[GSI Zone V Regional Seismic Shaking Prior]
    end

    subgraph "Layer 5: Civil Safety & Dispatch"
        B --> G1[NDMA 3-Tier Alert Engine Isolation Test]
        B --> G2[Crowdsourced Citizen Report Isolation Test]
        B --> G3[Multi-Channel Emergency Notification Pipeline]
    end

    B --> H[results/SYSTEM_HEALTH_STATUS.json]
    B --> I[results/SYSTEM_HEALTH_HISTORY.csv]
    B --> J[results/DAILY_SYSTEM_HEALTH_REPORT.md]
    B --> K[/system-status Dashboard]
    B --> L[Officer Command Topbar Badge]
```

---

## 2. 30 Component Probes Catalog

| ID | Component | Layer | Probe Method | Nominal Threshold | Fallback Behavior |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **01** | Frontend | Presentation | Validates `index.html`, 18 routes, and bundle compilation | File exists & valid | Static asset served |
| **02** | Backend API | Gateway | Checks process memory, ASGI loop, and server uptime | $< 5$ ms latency | Process monitor restart |
| **03** | Database | Persistence | Checks PostgreSQL port 5432; writes canary and rolls back | Connection active | Local memory & append-only file ledger |
| **04** | Authentication | Security | Generates and decodes HS256 JWT for health test officer | Signature valid | Deny unauthorized calls |
| **05** | JWT/RBAC | Security | Asserts Citizen denied Officer endpoint (401/403) | Enforced | Strict 403 Forbidden |
| **06** | LAND-JEPA Model | AI Engine | Runs forward pass with dummy tensors; checks for no NaNs | Shape `[1, 1]` logits | Tabular XGBoost / rule baseline |
| **07** | v2.5 Champion | AI Governance | Verifies `land_jepa_weights.pt` presence and SHA-256 hash | Hash matches | Failover to frozen baseline |
| **08** | v2.6.1 Challenger | AI Governance | Validates `PROSPECTIVE_CONFIG_FREEZE.json` thresholds | `0.6531, 0.7724, 0.9550` | Quarantined shadow mode |
| **09** | Weather Provider | Atmospheric | Pings Open-Meteo API for Shillong coordinates (2.5s timeout) | HTTP 200, valid units | ERA5-Land climatology baseline |
| **10** | Forecast Provider | Atmospheric | Evaluates 72h QPF horizons; verifies $t_{\text{issue}} \le t_{\text{pred}}$ | Freshness $< 30$ min | Last known NWP cycle |
| **11** | Soil Data | Geotechnical | Checks volumetric moisture (0-7cm, 7-28cm) and saturation | Valid float ranges | Antecedent Precipitation Index |
| **12** | Terrain | Spatial | Verifies Copernicus GLO-30m tiles, slope and curvature rasters | Resolution 30m | Static GSI regional slope prior |
| **13** | GIS | Spatial | Validates 8 NER corridor geometries in `gis/real_zones.py` | 8 active zones | EPSG:4326 bounding fallback |
| **14** | MapTiler | Cartography | Verifies `VITE_MAPTILER_API_KEY` configuration; masks key | Key configured | CARTO Positron / Dark Matter tiles |
| **15** | Sentinel-1/InSAR | Satellite | Queries Copernicus S1A catalog; checks coherence | Coherence $< 0.20$ | Disclose UNAVAILABLE (Decorrelated) |
| **16** | Seismic/PGA | Geotechnical | Checks USGS live feed with GSI Zone V static prior (0.36g) | Zero fabrication | GSI Zone V hazard factor |
| **17** | Road GIS | Spatial | Checks 50m highway cut buffers along NH-27, NH-10, NH-29 | Valid geometry | Standard chainage buffers |
| **18** | Drainage/Culvert | Hydrological | Evaluates HydroSHEDS streams and culvert blockage gates | Valid physical index | Regional drainage density proxy |
| **19** | Alert Engine | Safety | Creates isolated `TEST_ALERT`, validates Red level, rolls back | Isolated & deleted | Human-in-the-loop review |
| **20** | Citizen Reporting | Safety | Ingests `TEST_CITIZEN_REPORT`, validates bounds, clears record | Isolated & deleted | Local offline storage queue |
| **21** | Offline Sync | Reliability | Checks atomic append permission to local prediction ledger | File writable | In-memory cache buffer |
| **22** | Notification | Dispatch | Asserts delivery channel configuration (Siren, SMS, Push) | Isolated demo mode | Suppress live SMS to public |
| **23** | Translation/i18n | Presentation | Audits key parity across `en`, `hi`, `as`, `bn`, and `mni` | 0 missing keys | English root dictionary fallback |
| **24** | Prediction Ledger | Storage | Verifies line counts and read access to `predictions_ledger.jsonl` | File accessible | Memory ledger buffer |
| **25** | Audit Log | Storage | Verifies line counts and read access to `UI_INTERACTION_AUDIT.csv` | File accessible | Log warning alert |
| **26** | Scheduled Jobs | Gateway | Asserts asyncio background poller intervals (5m, 10m, 30m, 24h) | Timers active | Re-register poller tasks |
| **27** | Cache | Reliability | Inspects in-memory LRU cache size and TTL expiry (900s) | Active entries | Live re-fetch from source |
| **28** | Storage | Persistence | Checks disk capacity ($> 1$ GB free space on NVMe storage) | Space $> 1.0$ GB | Clean old transient caches |
| **29** | Docker Services | Infrastructure| Checks presence of `docker-compose.yml` and container health | Configured | Standalone host deployment |
| **30** | Reverse Proxy | Infrastructure| Asserts NGINX gateway rate limiting (60 req/min) & CORS rules | Valid config | Direct ASGI local binding |

---

## 3. Master Status Decision Tree

The system evaluates diagnostic results according to strict deterministic criteria:

```
IF any of [Backend API, LAND-JEPA Model, Prediction Pipeline, Authentication] is OFFLINE
   OR Temporal Causality Violation detected (t_obs > t_pred)
   ===> OVERALL STATUS = CRITICAL (Dispatches Blocked, Red Pulsing Banner)

ELSE IF any of [Database, Weather API, MapTiler, InSAR, Soil Data] is DEGRADED or UNAVAILABLE
   ===> OVERALL STATUS = DEGRADED (Operational Fallbacks Active, Amber Warning Banner)

ELSE
   ===> OVERALL STATUS = OPERATIONAL (All Services Nominal, Green Banner)
```
