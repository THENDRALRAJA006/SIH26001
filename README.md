# LAND-JEPA: AI-Powered Landslide Early Warning & Risk Monitoring System

**Northeast India Region (NER)**  
> **Team:** ZAIX | **Problem Statement:** SIH26001 | **Theme:** Disaster Management  
> **Repository:** `d:/SIH26001` | **Status:** All 9 Checkpoints Complete (Production Demo Ready)

---

## ⚠️ Safety & Demo Notice

> [!IMPORTANT]
> **RESEARCH & DEMO PLATFORM — NOT FOR OPERATIONAL LIFE-SAFETY DISPATCH**  
> This platform is an advanced AI and GIS engineering demonstration designed for the Smart India Hackathon. All hazard indices, zone risk scores, citizen reports, and alert dispatches operate with `is_demo=True`. It does **not** replace official advisories issued by the Geological Survey of India (GSI), India Meteorological Department (IMD), National Disaster Management Authority (NDMA), or State Disaster Management Authorities (SDMAs).

---

## Core System Architecture

```
                                    ┌─────────────────────────────────────────────────────────┐
                                    │                     CLIENT INTERFACES                   │
                                    │  ┌────────────────────────┐  ┌───────────────────────┐  │
                                    │  │ React / Vite Dashboard │  │ Expo Mobile App (RN)  │  │
                                    │  │ (Dark mode, GIS Map,   │  │ (Offline Sync Queue,  │  │
                                    │  │  SHAP, Alerts, Models) │  │  Multilingual EN/HI)  │  │
                                    │  └───────────┬────────────┘  └───────────┬───────────┘  │
                                    └──────────────┼───────────────────────────┼──────────────┘
                                                   │   HTTP / JSON (FastAPI)   │
                                                   ▼                           ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 FASTAPI APPLICATION GATEWAY                                 │
│  - JWT Bearer Authentication & Role-Based Access Control (RBAC)                             │
│  - Rate Limiting (SlowAPI) & Input Schema Validation (Pydantic v2)                          │
│  - Gated Alert Dispatcher (ALERT_DEMO_ONLY=True, Cooldown Suppression)                       │
│  - Citizen & Field Reporting (Mandatory Human Review Flags)                                 │
└──────┬──────────────────────┬─────────────────────────────┬───────────────────────────┬─────┘
       │                      │                             │                           │
       ▼                      ▼                             ▼                           ▼
┌──────────────┐      ┌──────────────┐              ┌───────────────┐           ┌──────────────┐
│  AI / ML     │      │ GIS PIPELINE │              │ PERSISTENCE   │           │ ORCHESTRATION│
│  ENGINE      │      │              │              │               │           │              │
│ - JEPA-TCN   │      │ - Horn 1981  │              │ PostgreSQL 15 │           │ Docker       │
│   Self-Super │      │   Slope/Asp  │              │ + PostGIS 3.4 │           │ Compose      │
│ - Sup. TCN   │      │ - Z&T Curv   │              │ (22 Normalized│           │ Multi-stage  │
│ - XGBoost    │      │ - Weiss TPI  │              │  Schema       │           │ Backend &    │
│   Baseline   │      │ - WLC Suscep │              │  Tables)      │           │ Frontend     │
│ - SHAP Trees │      │ - InSAR (Opt)│              │               │           │ Containers   │
└──────────────┘      └──────────────┘              └───────────────┘           └──────────────┘
```

---

## Checkpoint Delivery Summary

| Checkpoint | Scope & Components | Status | Verification & Tests |
|---|---|---|---|
| **CP 1** | System specs, architecture docs, safety policy, folder hierarchy | ✅ Complete | Complete docs in `docs/` |
| **CP 2** | PostGIS schema (22 tables), ingestion, preprocessing, validation, sliding windows | ✅ Complete | 146 unit & integration tests pass |
| **CP 3** | XGBoost baseline model with temporal CV, hyperparameter tuning, SHAP explainability | ✅ Complete | Verified on temporal splits |
| **CP 4** | Supervised Temporal Convolutional Network (TCN) with causal dilated layers & focal loss | ✅ Complete | Training & inference pipelines verified |
| **CP 5** | Self-Supervised JEPA (Joint Embedding Predictive Architecture) + Label Efficiency (5%–100%) | ✅ Complete | EMA target encoder & context masking verified |
| **CP 6** | Production FastAPI backend (REST API, JWT auth, risk prediction, citizen reporting, alerts) | ✅ Complete | 42 API & service tests pass |
| **CP 7** | Interactive React / Vite dashboard (Leaflet GIS map, risk cards, SHAP graphs, demo banner) | ✅ Complete | Vite production bundle verified |
| **CP 8** | React Native Expo mobile app (Offline SQLite/AsyncStorage sync queue, multi-lingual EN/HI) | ✅ Complete | 11 Jest unit tests pass |
| **CP 9** | GIS pipeline (morphometrics, WLC susceptibility, InSAR adapter), Docker, E2E tests, README | ✅ Complete | 54 GIS tests pass, E2E suite verified |

**Total Cumulative Tests Passing:** **253 tests** (242 Python pytest + 11 JS Jest).

---

## GIS Pipeline & Morphometrics

The GIS subsystem (`gis/`) operates on Digital Elevation Models (DEM) with pure vectorized NumPy operations:

1. **Slope & Aspect (`Horn 1981`):** 3×3 finite-difference convolution yielding terrain slope in degrees $[0, 90)$ and flow aspect $[0, 360)$. Flat areas safely handle NaN aspect angles.
2. **Profile Curvature (`Zevenbergen & Thorne 1987`):** Fourth-order polynomial surface fitting to identify concave (convergent flow) vs convex (divergent flow) slopes.
3. **Topographic Position Index (`Weiss 2001`):** Local elevation deviation relative to a uniform filter neighborhood window, isolating ridges from valleys.
4. **Hillshade:** Illumination simulation (sun azimuth 315°, altitude 45°) for visualization without runtime warnings on flat topography.
5. **Weighted Linear Combination (WLC) Susceptibility:** Multi-criteria static evaluation combining slope (0.35), curvature (0.15), aspect (0.10), TPI (0.15), lithology (0.15), and land-use (0.10).
6. **InSAR Surface Deformation Adapter:** Standardized Sentinel-1 LOS displacement reader that gracefully degrades to `NaN` when disabled or data is absent, fulfilling the project mandate that InSAR remains optional.
7. **8 NER Demo Zones:** High-risk zones across Northeast Indian states (Kamrup Metro Hills, East Khasi Hills, Kohima Ridge, Aizawl Slopes, Imphal Rim, West Arunachal, Upper Brahmaputra, North Tripura).

---

## Quick Start (Docker Deployment)

Launch the entire stack with Docker Compose:

```bash
# 1. Clone repository
git clone <repo-url>
cd SIH26001

# 2. Configure environment
cp .env.example .env

# 3. Start PostgreSQL/PostGIS, FastAPI backend, and React dashboard
docker compose up -d

# 4. Run database migrations & seed demo monitoring zones
docker compose exec backend alembic upgrade head
docker compose exec backend python scripts/seed_demo.py

# 5. Access services
# Web Dashboard: http://localhost:3000
# FastAPI Docs:  http://localhost:8000/docs
# API Health:    http://localhost:8000/health
```

---

## Native Local Development Setup

### 1. Backend (FastAPI + PyTorch)

```powershell
cd d:\SIH26001
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

# Run backend test suite (242 tests)
python -m pytest tests/ml/ tests/backend/ tests/gis/ -v

# Start FastAPI dev server
cd backend
uvicorn app.main:app --reload --port 8000
```

### 2. Web Dashboard (React + Vite)

```powershell
cd d:\SIH26001\frontend\dashboard
npm install
npm run dev
# Dashboard runs at http://localhost:5173
```

### 3. Mobile Application (React Native + Expo)

```powershell
cd d:\SIH26001\mobile\landjepa
npm install

# Run Jest offline sync tests (11 tests)
npm test

# Start Expo dev server
npx expo start
```

### 4. End-to-End Smoke Tests

```powershell
# Ensure backend is running on port 8000, then run:
python -m pytest tests/test_e2e_smoke.py -v
```

---

## Key API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, database connection, demo status |
| `GET` | `/api/v1/risk/zones` | Current risk summary for all 8 NER monitoring zones |
| `GET` | `/api/v1/risk/zones/{zone_id}` | Detailed zone prediction (current, 24h, 48h) + SHAP features |
| `POST`| `/api/v1/risk/predict` | Batch prediction for multiple zone IDs |
| `GET` | `/api/v1/risk/history/{zone_id}` | Time-series risk history (default 7 days) |
| `GET` | `/api/v1/alerts/` | Filterable list of active and past alerts |
| `POST`| `/api/v1/alerts/reports` | Citizen/field report submission with geofence validation |

---

## Project Team & Credits

- **Team:** ZAIX
- **Theme:** Disaster Management
- **Hackathon:** Smart India Hackathon (SIH26001)
- **Scientific Foundation:**
  - Joint Embedding Predictive Architecture (JEPA) — LeCun et al.
  - Temporal Convolutional Networks (TCN) — Bai, Kolter, Koltun
  - Horn (1981) & Zevenbergen-Thorne (1987) DEM morphometrics
