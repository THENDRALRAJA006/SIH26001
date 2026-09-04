# LAND-JEPA

**AI-Based Early Warning and Landslide Risk Monitoring System — Northeast India**

> Team: **ZAIX** | SIH Problem Statement: **SIH26001** | Theme: Disaster Management

---

## ⚠️ Disclaimer

This is a **research and demonstration platform** built for the Smart India Hackathon.
It is **not** a certified operational early-warning system. All risk outputs require
validation by domain experts. It does **not** replace official government agencies
(GSI, IMD, NDMA, State DMAs). See [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md) for full details.

---

## Overview

LAND-JEPA (Landslide Analysis via Neural Deep - Joint Embedding Predictive Architecture)
investigates the following research question:

> "Can JEPA-style self-supervised temporal representation learning with a TCN encoder
> improve landslide-risk prediction when labelled landslide events are scarce?"

The platform provides:
- Environmental data ingestion (rainfall, weather, soil moisture, terrain)
- XGBoost and supervised TCN baselines
- JEPA-style self-supervised pre-training with a TCN encoder
- Label-efficiency evaluation (5% → 100% of labels)
- Current / 24h / 48h risk prediction per zone
- GIS-based risk visualization
- Authority decision-support dashboard
- Citizen and field-officer reporting (with offline support)
- Emergency prioritization engine
- Multilingual alerts (EN, HI)
- Optional InSAR deformation integration
- React Native mobile application

---

## Architecture

```
WEB APP (React + Vite) ──┐
                         ├──► FastAPI ──► PostgreSQL + PostGIS
MOBILE APP (React Native)┘         │
                                   ├──► AI Engine (PyTorch + XGBoost)
                                   ├──► Risk Engine
                                   ├──► GIS Engine (GeoPandas)
                                   ├──► Alert Engine (DEMO MODE)
                                   └──► Priority Engine
```

See [`docs/SYSTEM_ARCHITECTURE.md`](docs/SYSTEM_ARCHITECTURE.md) for full details.

---

## Quick Start

```bash
# Prerequisites: Docker, Docker Compose

git clone <repository-url>
cd LAND-JEPA

cp .env.example .env
# Edit .env if needed (default is demo mode)

docker compose up -d
docker compose exec backend alembic upgrade head
docker compose exec backend python scripts/seed_demo.py

# Open http://localhost:3000
```

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for full deployment instructions.

---

## Documentation

| Document                  | Description                                    |
|---------------------------|------------------------------------------------|
| [PROJECT_SPEC.md](docs/PROJECT_SPEC.md) | Mission, rules, scope, technology |
| [SYSTEM_ARCHITECTURE.md](docs/SYSTEM_ARCHITECTURE.md) | Full system architecture |
| [AI_ARCHITECTURE.md](docs/AI_ARCHITECTURE.md) | XGBoost, TCN, JEPA-TCN architecture |
| [DATA_SCHEMA.md](docs/DATA_SCHEMA.md) | Database schema (all 22 tables) |
| [DATA_SOURCES.md](docs/DATA_SOURCES.md) | Data sources, licensing, demo policy |
| [API_SPEC.md](docs/API_SPEC.md) | Full API specification |
| [GIS_SPEC.md](docs/GIS_SPEC.md) | GIS layers, processing, visualization |
| [MOBILE_SPEC.md](docs/MOBILE_SPEC.md) | Mobile app screens and offline architecture |
| [SECURITY.md](docs/SECURITY.md) | Auth, RBAC, validation, audit logging |
| [EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md) | ML experiment design and protocols |
| [DEPLOYMENT.md](docs/DEPLOYMENT.md) | Docker setup and deployment notes |
| [LIMITATIONS.md](docs/LIMITATIONS.md) | Known limitations (required reading) |

---

## Repository Structure

```
LAND-JEPA/
├── docs/           Architecture and specification documents
├── data/           raw/ interim/ processed/ demo/
├── ml/             ML pipeline: ingestion, preprocessing, baselines, JEPA
├── backend/        FastAPI application
├── frontend/       React + TypeScript + Vite web app
├── mobile/         React Native + Expo mobile app
├── gis/            GIS processing scripts and styles
├── tests/          Unit, integration, frontend, mobile tests
├── scripts/        Data ingestion, scheduling, seeding scripts
├── notebooks/      Exploratory analysis notebooks
├── models/         Saved model artifacts
├── results/        Experiment results and plots
└── docker/         Dockerfiles and service configs
```

---

## Demo Mode

The system ships with DEMO MODE enabled by default.

In demo mode:
- All data is clearly labelled as `DEMO DATA`
- No real SMS, push, or email alerts are sent
- The `/demo` route shows a guided scenario walkthrough

**Demo scenario**: Rainfall increase → Risk elevation → HIGH zone → Priority 1 →
Citizen report → Human review → Alert generation → Authority response

---

## Research Context

This project applies the JEPA (Joint Embedding Predictive Architecture) framework
[LeCun et al., Meta AI, 2022] to geophysical time-series data for landslide risk prediction.
**This project did not invent JEPA.** The research contribution is the domain application
and label-efficiency evaluation.

---

## License

MIT License. See `LICENSE` file.

Data source licenses vary — see [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).
