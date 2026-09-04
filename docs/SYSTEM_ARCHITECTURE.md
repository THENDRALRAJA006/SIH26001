# LAND-JEPA — System Architecture

## Overview

LAND-JEPA is structured as a layered, service-oriented platform. Each layer communicates
through well-defined interfaces. No layer assumes direct knowledge of another layer's
internal implementation.

```
┌─────────────────────────────────────────────────────────────────┐
│                      LAND-JEPA PLATFORM                         │
│                                                                 │
│   ┌──────────────────────┐    ┌──────────────────────────────┐  │
│   │      WEB APP         │    │       MOBILE APP             │  │
│   │  React + TypeScript  │    │  React Native + Expo         │  │
│   │  Vite + Tailwind     │    │  SQLite Offline Store        │  │
│   └──────────┬───────────┘    └──────────────┬───────────────┘  │
│              │                               │                  │
│              └───────────────┬───────────────┘                  │
│                              │ HTTPS / REST / WebSocket         │
│                              ▼                                  │
│                   ┌──────────────────────┐                      │
│                   │      FastAPI          │                      │
│                   │  Auth  |  Routing     │                      │
│                   │  Rate Limit | CORS    │                      │
│                   └──────────┬───────────┘                      │
│                              │                                  │
│         ┌────────────────────┼────────────────────┐             │
│         │                   │                    │             │
│         ▼                   ▼                    ▼             │
│  ┌─────────────┐   ┌────────────────┐   ┌──────────────┐       │
│  │  PostgreSQL  │   │   AI Engine    │   │  GIS Engine  │       │
│  │  + PostGIS   │   │  (PyTorch +   │   │ (GeoPandas + │       │
│  │             │   │   XGBoost)     │   │  PostGIS)    │       │
│  └──────┬──────┘   └───────┬────────┘   └──────┬───────┘       │
│         │                  │                   │               │
│         └──────────────────┼───────────────────┘               │
│                            │                                    │
│                            ▼                                    │
│                   ┌──────────────────────┐                      │
│                   │     RISK ENGINE      │                      │
│                   │  current / 24h / 48h │                      │
│                   │  probability         │                      │
│                   │  confidence          │                      │
│                   │  leading factors     │                      │
│                   └──────────┬───────────┘                      │
│                              │                                  │
│              ┌───────────────┼───────────────┐                  │
│              │               │               │                  │
│              ▼               ▼               ▼                  │
│      ┌──────────────┐  ┌──────────┐  ┌─────────────────┐        │
│      │   GIS MAP    │  │  ALERT   │  │ PRIORITY ENGINE │        │
│      │  (Leaflet /  │  │  ENGINE  │  │                 │        │
│      │  MapLibre)   │  │  SMS/Push│  │ P1 | P2 | P3    │        │
│      └──────────────┘  └──────────┘  └────────┬────────┘        │
│                                               │                 │
│                                               ▼                 │
│                                    Authorities | Communities    │
│                                    Field Officers               │
└─────────────────────────────────────────────────────────────────┘
```

## Service Decomposition

### 1. Data Ingestion Layer

**Purpose**: Collect environmental data from external sources.

**Components**:
- `RainfallProvider` — fetches precipitation data
- `WeatherProvider` — fetches temperature, humidity, wind
- `SoilMoistureProvider` — fetches or estimates soil moisture
- `TerrainProvider` — fetches DEM, slope, aspect, curvature
- `LandslideInventoryProvider` — loads historical event records
- `SatelliteProvider` — optional Sentinel-2 imagery
- `InSARProvider` — optional Sentinel-1 deformation data

Each provider implements: `fetch()`, `validate()`, `transform()`, `store()`

**Critical rule**: Demo providers are in separate classes from real providers.
They are never silently mixed.

### 2. Data Quality Layer

**Purpose**: Validate and standardize all ingested data before it enters the ML pipeline.

**Checks**:
- Duplicate detection
- Missing-value detection and reporting
- Coordinate bounds validation
- Timestamp ordering validation
- Value range validation per variable
- Unit normalization
- Timezone normalization (all to UTC)
- Spatial consistency checks
- Source provenance metadata attachment

### 3. ML / AI Engine

**Purpose**: Train models and generate risk representations.

**Sub-components**:
- `FeatureEngineer` — builds tabular features for XGBoost
- `WindowGenerator` — builds time-series windows for TCN/JEPA
- `LabelBuilder` — builds landslide occurrence labels (with documented uncertainty)
- `XGBoostBaseline` — classical gradient-boosting model
- `SupervisedTCN` — causal TCN trained on labelled data
- `JEPAPretrainer` — self-supervised JEPA-TCN pre-training
- `JEPADownstream` — fine-tuned risk head on JEPA representation
- `PhysicsStateEstimator` — lightweight infiltration/pore-pressure proxy
- `InSARAdapter` — optional slow-deformation state (removable)
- `RiskEnsemble` — combines model output with physics state for final risk

### 4. Risk Engine

**Purpose**: Produce the final, calibrated risk output for consumption.

**Inputs**: AI representation + terrain + weather + soil moisture + historical state + optional deformation
**Outputs**:
```json
{
  "zone_id": "...",
  "current_risk": 0.0–1.0,
  "risk_24h": 0.0–1.0,
  "risk_48h": 0.0–1.0,
  "confidence": 0.0–1.0,
  "risk_level": "LOW|MEDIUM|HIGH",
  "leading_factors": [...]
}
```
**Risk thresholds**: Configurable in `ml/configs/risk_thresholds.yaml`.

### 5. Priority Engine

**Purpose**: Rank zones for emergency response.

**Inputs**: risk_level + population_exposure + road_criticality + infrastructure_criticality + accessibility + confidence
**Outputs**: Priority 1 / 2 / 3 with human-readable explanation.

### 6. Alert Engine

**Purpose**: Generate and deliver risk notifications.

**Features**:
- Configurable threshold triggers
- Severity classification
- Multilingual template rendering (EN, HI)
- Delivery channels: web notification, mobile push, email, SMS adapter
- **DEMO MODE**: logs/simulates alerts; never sends real emergency alerts during development.
- Delivery tracking per alert.

### 7. GIS Engine

**Purpose**: Spatial data management and map layer generation.

**Layers**:
- Current risk heatmap
- 24h / 48h risk heatmap
- Historical landslides
- Roads, villages, infrastructure
- Citizen reports
- Optional InSAR deformation overlay
- Terrain (slope, elevation)

### 8. Reporting Service

**Purpose**: Handle citizen and field-officer reports.

**Features**:
- Photo + video + GPS + timestamp + category + description
- Human review queue (reports not directly merged without verification)
- Sync status tracking
- Offline queue management

### 9. Auth Service

**Purpose**: Secure all API access.

**Features**:
- JWT-based authentication
- Role-based access control (Admin, Authority, Field Officer, Citizen)
- Audit logging

### 10. Sync Service

**Purpose**: Handle offline→online data synchronization.

**Flow**: local_store → sync_queue → server acknowledgement → mark_synced
**Status values**: Offline | Pending Sync | Syncing | Synced | Failed

## Data Flow Summary

```
External Sources
      │
      ▼
Ingestion Layer (providers)
      │
      ▼
Data Quality Layer (validation + provenance)
      │
      ├─────────────────────────────────┐
      │                                 │
      ▼                                 ▼
PostgreSQL + PostGIS               ML Pipeline
(raw + processed tables)          (features → training → inference)
      │                                 │
      └─────────────┬───────────────────┘
                    │
                    ▼
               Risk Engine
                    │
          ┌─────────┼─────────┐
          │         │         │
          ▼         ▼         ▼
        GIS      Alerts    Priority
          │         │         │
          └────┬────┘─────────┘
               │
               ▼
           FastAPI
               │
        ┌──────┴──────┐
        │             │
        ▼             ▼
      Web App     Mobile App
```

## Security Boundaries

- All external input is validated before storage.
- File uploads: type-checked, size-limited, virus-scan hook available.
- API: JWT on all protected routes.
- Secrets: environment variables only, never committed.
- CORS: configured per environment.
- Rate limiting: on auth and report submission endpoints.
- Audit log: all write operations recorded.

## Scalability Notes

- FastAPI is stateless; horizontally scalable behind a load balancer.
- PostgreSQL read replicas can be added without application changes.
- ML inference is synchronous initially; can be moved to async task queue (Celery/Redis).
- GIS tile caching can be added via a tile server (Martin, pg_tileserv) without changing the API.
- Mobile sync is designed for eventual consistency.

## Deployment Topology (Initial)

```
Docker Compose (development / demo):
  - backend    (FastAPI)
  - db         (PostgreSQL + PostGIS)
  - frontend   (Vite dev server / Nginx)
  - worker     (optional Celery worker)
```

Production targets (documented, not implemented during hackathon):
- Kubernetes with separate pods per service
- Managed PostgreSQL (AWS RDS / GCP CloudSQL)
- Object storage for uploads (S3 / GCS)
