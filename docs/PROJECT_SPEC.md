# LAND-JEPA — Project Specification

## Identity

| Field              | Value                                                         |
|--------------------|---------------------------------------------------------------|
| Project Name       | LAND-JEPA                                                     |
| Team               | ZAIX                                                          |
| Problem Statement  | SIH26001                                                      |
| Theme              | Disaster Management                                           |
| Category           | Software                                                      |
| Target Region      | Northeast India (NER)                                         |
| Competition        | Smart India Hackathon (SIH)                                   |

## Problem Statement

Northeast India (NER) is one of the world's most landslide-prone regions due to its
mountainous terrain, high-intensity monsoon rainfall, seismically active zones, and
fragile geological conditions. Existing early-warning systems are sparse, uncoordinated,
and insufficiently data-driven. Communities in remote areas receive little actionable
warning before a landslide event.

**SIH26001**: AI-Based Early Warning and Landslide Risk Monitoring System in NER.

## Mission

Build a complete, modular, production-style software platform that:

1. Collects landslide-relevant environmental data from configurable data sources.
2. Preprocesses and validates time-series and geospatial data rigorously.
3. Implements an XGBoost classical baseline model.
4. Implements a supervised Temporal Convolutional Network (TCN) baseline model.
5. Implements a JEPA-style self-supervised temporal pre-training framework using a TCN encoder.
6. Evaluates JEPA-TCN performance under scarce labelled landslide data (label-efficiency experiment).
7. Generates current, 24-hour, and 48-hour landslide risk probability estimates.
8. Provides GIS-based risk visualization for decision-makers.
9. Provides authority decision-support dashboards.
10. Provides citizen and field-officer reporting workflows.
11. Supports offline reporting with synchronization.
12. Provides emergency zone prioritization.
13. Provides multilingual warning and notification support.
14. Supports optional InSAR surface-deformation data integration.
15. Provides model explanations for all predictions.
16. Provides a React Native mobile application.
17. Keeps the full system modular, configurable, and extensible.

## Research Question

> "Can JEPA-style self-supervised temporal representation learning with a TCN encoder
> improve landslide-risk prediction when labelled landslide events are scarce?"

## Non-Negotiable Rules

### DO NOT

- Fabricate real datasets, landslide events, accuracy numbers, or API responses.
- Claim JEPA was invented by this project.
- Claim quantum advantage.
- Claim 100% accuracy or guaranteed warning time.
- Claim the system replaces GSI or any other authority.
- Use future information to construct past model inputs (no data leakage).
- Place any VQC in the critical alert path.
- Make InSAR mandatory for system operation.
- Create unnecessary AI models beyond the specified scope.
- Hide errors silently.

### DO

- Use real data where publicly available.
- Clearly label all synthetic/demo data.
- Document all assumptions.
- Make components configuration-driven.
- Write automated tests.
- Log all experiments with reproducible seeds.
- Use Git for version control.
- Create modular, independently testable services.
- Validate all data, APIs, and model outputs.
- Provide error handling at every boundary.
- Use human verification for citizen image reports.
- Explain all model limitations clearly.

## Scope Boundaries

| In Scope                                     | Out of Scope                                      |
|----------------------------------------------|---------------------------------------------------|
| NER landslide risk prediction                | Real-time government integration                  |
| Environmental data ingestion pipeline        | Operational deployment to GSI/NDMA systems        |
| XGBoost, TCN, JEPA-TCN models               | Quantum computing components                      |
| Web dashboard (React + TypeScript + Vite)    | Guaranteed evacuation workflow automation         |
| Mobile app (React Native + Expo)             | Real InSAR satellite acquisition                  |
| Offline reporting + sync                     | Full physics-based geotechnical simulation        |
| Multilingual alerts (EN, HI)                 | Causal explanation (only model-contribution SHAP) |
| Demo mode (safely labelled)                  |                                                   |
| Docker deployment                            |                                                   |

## Technology Stack Summary

| Layer             | Technology                                    |
|-------------------|-----------------------------------------------|
| Backend API       | Python 3.11+, FastAPI, Uvicorn                |
| Database          | PostgreSQL 15+, PostGIS 3                     |
| ML Framework      | PyTorch, XGBoost, SHAP                        |
| GIS               | GeoPandas, PostGIS, Leaflet/MapLibre           |
| Frontend          | React 18, TypeScript, Vite, Tailwind CSS       |
| Mobile            | React Native, Expo, SQLite                    |
| Offline Storage   | IndexedDB (web), SQLite (mobile)              |
| Container         | Docker, Docker Compose                        |
| Task Scheduling   | APScheduler / Celery (configurable)           |
| Auth              | JWT (python-jose)                             |
| Testing           | Pytest, Vitest, Jest                          |

## Disclaimer

This system is a research and demonstration platform built for the Smart India Hackathon.
It is **not** a certified operational early-warning system. It does **not** replace official
government agencies (GSI, IMD, NDMA, State DMAs). All risk outputs should be treated as
decision-support information requiring expert validation before operational use.
