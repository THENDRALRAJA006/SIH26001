# LAND-JEPA — System Architecture (ERD + Frontend + Implementation Roadmap)

## Entity-Relationship Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        LAND-JEPA DATABASE ERD                                  │
│                                                                                │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐                                  │
│  │  users  │◄───│user_roles│───►│  roles   │                                  │
│  └────┬────┘    └──────────┘    └──────────┘                                  │
│       │                                                                        │
│       ├──────────────────────┬──────────────────────────┐                     │
│       ▼                      ▼                          ▼                     │
│  ┌──────────────┐   ┌────────────────┐      ┌──────────────────┐             │
│  │citizen_reports│   │ field_reports  │      │   audit_logs     │             │
│  └──────┬───────┘   └───────┬────────┘      └──────────────────┘             │
│         │                   │                                                  │
│         └──────┬────────────┘                                                 │
│                │                                                               │
│                ▼                                                               │
│          ┌──────────┐                                                          │
│          │  zones   │◄─────────────────────────────────────────────┐          │
│          └────┬─────┘                                              │          │
│               │                                                    │          │
│    ┌──────────┼──────────┬──────────┬──────────┬────────┐         │          │
│    ▼          ▼          ▼          ▼          ▼        ▼         │          │
│ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌─────────┐        │          │
│ │rainfall│ │weather │ │soil_moist│ │terrain │ │insar_obs│        │          │
│ └────────┘ └────────┘ └──────────┘ └────────┘ └─────────┘        │          │
│                                                                    │          │
│  ┌──────────────────┐    ┌────────────────┐                        │          │
│  │ landslide_events │───►│    zones       │                        │          │
│  └──────────────────┘    └────────────────┘                        │          │
│                                                                    │          │
│  ┌──────────────┐    ┌────────────────┐    ┌──────────────────┐   │          │
│  │ model_runs   │◄───│risk_predictions│───►│  risk_factors    │   │          │
│  └──────────────┘    └───────┬────────┘    └──────────────────┘   │          │
│                              │                                     │          │
│                         zone_id ───────────────────────────────────┘          │
│                                                                               │
│  ┌──────────────────┐    ┌────────────────┐    ┌──────────────────────┐      │
│  │ priority_scores  │    │    alerts      │◄───│   alert_deliveries   │      │
│  └──────────────────┘    └────────────────┘    └──────────────────────┘      │
│                                                                               │
│  ┌──────────────┐   ┌──────────┐   ┌──────────────┐                          │
│  │   villages   │   │  roads   │   │infrastructure│                          │
│  └──────────────┘   └──────────┘   └──────────────┘                          │
│                                                                               │
│  ┌──────────────┐                                                             │
│  │  sync_queue  │                                                             │
│  └──────────────┘                                                             │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Frontend Page Map

### Authority Dashboard (/app)

```
/app
├── /dashboard              ← KPI cards, map, active alerts, weather trends
├── /map                    ← Full GIS map with all layers
├── /forecast               ← 24h / 48h risk forecast charts per zone
├── /historical             ← Historical landslide event browser
├── /roads                  ← Road network criticality map
├── /infrastructure         ← Infrastructure risk exposure
├── /priorities             ← Priority 1/2/3 zone list with explanations
├── /reports                ← Citizen & field report review queue
│   ├── /reports/:id        ← Report detail + review action
├── /alerts                 ← Alert history + manual alert creation
├── /analytics              ← Label efficiency charts, model performance
├── /model                  ← Model status, version, explanation viewer
└── /settings               ← Thresholds, ingestion config, user management
```

### Public Citizen Portal (/portal)

```
/portal
├── /                       ← Public risk summary, active alerts
├── /map                    ← Simplified public risk map
├── /report                 ← Citizen report submission form
├── /report/status/:id      ← Report submission + sync status
└── /alerts                 ← Public alert list
```

### Demo Route

```
/demo                       ← Guided demo scenario walkthrough
```

### Auth

```
/login
/register
/forgot-password
```

---

## Implementation Roadmap

### CHECKPOINT 1 — Architecture + Documentation ✅
**Status**: In progress (this document)

Deliverables:
- [x] Directory structure created
- [x] PROJECT_SPEC.md
- [x] SYSTEM_ARCHITECTURE.md
- [x] AI_ARCHITECTURE.md
- [x] DATA_SCHEMA.md
- [x] DATA_SOURCES.md
- [x] API_SPEC.md
- [x] GIS_SPEC.md
- [x] MOBILE_SPEC.md
- [x] SECURITY.md
- [x] EXPERIMENT_PLAN.md
- [x] DEPLOYMENT.md
- [x] LIMITATIONS.md
- [x] ERD + Frontend page map + Roadmap (this file)
- [ ] .env.example
- [ ] docker-compose.yml (skeleton)
- [ ] README.md

---

### CHECKPOINT 2 — Database + Data Pipeline
Deliverables:
- [ ] Alembic migration setup
- [ ] All 22 tables created via migration
- [ ] PostGIS extensions initialized
- [ ] All data provider interfaces defined
- [ ] Demo data providers implemented
- [ ] Data quality validators
- [ ] Dataset builder (window + label)
- [ ] Unit tests: data validation, window generation, label building

---

### CHECKPOINT 3 — XGBoost Baseline
Deliverables:
- [ ] Feature engineering pipeline
- [ ] XGBoost training script
- [ ] Evaluation with all metrics
- [ ] SHAP explainability
- [ ] Checkpoint saving
- [ ] Unit tests: feature eng, training, evaluation
- [ ] Git commit: `feat: xgboost baseline complete`

---

### CHECKPOINT 4 — Supervised TCN Baseline
Deliverables:
- [ ] TCN model implementation (causal, dilated residual)
- [ ] Training script
- [ ] Same evaluation protocol as XGBoost
- [ ] Unit tests: TCN forward pass, shapes, no-leakage
- [ ] Git commit: `feat: supervised TCN baseline complete`

---

### CHECKPOINT 5 — JEPA-TCN
Deliverables:
- [ ] Context TCN encoder
- [ ] Target TCN encoder (EMA copy)
- [ ] Predictor (lightweight MLP)
- [ ] JEPA loss (SmoothL1 in latent space)
- [ ] EMA update logic
- [ ] Pre-training script (no labels)
- [ ] Collapse detection utilities
- [ ] Downstream fine-tuning script
- [ ] Unit tests: shapes, EMA, collapse detection
- [ ] Git commit: `feat: JEPA-TCN pretraining + downstream complete`

---

### CHECKPOINT 6 — Research Evaluation
Deliverables:
- [ ] Label-efficiency experiment (5/10/25/50/100% fractions)
- [ ] label_efficiency.csv generated
- [ ] 4 plots generated
- [ ] Results honest (no cherry-picking)
- [ ] Git commit: `feat: label efficiency experiment complete`

---

### CHECKPOINT 7 — FastAPI + Database Integration
Deliverables:
- [ ] All 25+ API endpoints implemented
- [ ] Auth (JWT + RBAC)
- [ ] Risk API integration with AI engine
- [ ] OpenAPI docs auto-generated
- [ ] Integration tests: API → database → response
- [ ] Git commit: `feat: FastAPI backend complete`

---

### CHECKPOINT 8 — GIS Web Dashboard
Deliverables:
- [ ] React + Vite + Tailwind setup
- [ ] All authority dashboard pages
- [ ] GIS map with all layers
- [ ] Zone click popup
- [ ] Risk heatmap
- [ ] Layer controls
- [ ] Responsive design
- [ ] Git commit: `feat: authority web dashboard complete`

---

### CHECKPOINT 9 — Citizen / Field Reporting
Deliverables:
- [ ] Citizen report portal UI
- [ ] Photo/GPS/category form
- [ ] Human review workflow (authority dashboard)
- [ ] Report status tracking
- [ ] Git commit: `feat: citizen reporting complete`

---

### CHECKPOINT 10 — Offline Sync
Deliverables:
- [ ] IndexedDB integration (web)
- [ ] Offline report saving
- [ ] Sync queue management
- [ ] Sync status badges
- [ ] Conflict resolution
- [ ] Unit tests: offline save, sync, status
- [ ] Git commit: `feat: offline sync complete`

---

### CHECKPOINT 11 — Alerts + Priority
Deliverables:
- [ ] Alert engine (threshold triggers)
- [ ] Multilingual template rendering (EN + HI)
- [ ] DEMO MODE (logs only, no real delivery)
- [ ] Priority engine (P1/P2/P3 with explanations)
- [ ] Alert delivery tracking
- [ ] Unit tests: alert logic, priority scoring
- [ ] Git commit: `feat: alert and priority engine complete`

---

### CHECKPOINT 12 — Mobile App
Deliverables:
- [ ] React Native + Expo setup
- [ ] All citizen screens
- [ ] Field officer screens
- [ ] SQLite offline storage
- [ ] Push notification integration (DEMO MODE)
- [ ] Sync with backend
- [ ] Git commit: `feat: mobile app complete`

---

### CHECKPOINT 13 — Optional InSAR
Deliverables:
- [ ] InSAR adapter interface
- [ ] Demo InSAR data provider
- [ ] Deformation feature integration (optional)
- [ ] GIS overlay layer
- [ ] Graceful missing-data handling
- [ ] Git commit: `feat: optional InSAR adapter complete`

---

### CHECKPOINT 14 — Security + Testing
Deliverables:
- [ ] Full test suite (unit + integration + frontend + mobile)
- [ ] Security checklist verified
- [ ] Rate limiting tested
- [ ] Audit logging verified
- [ ] Input validation edge cases
- [ ] Git commit: `feat: full test suite and security hardening`

---

### CHECKPOINT 15 — Docker + Deployment
Deliverables:
- [ ] docker-compose.yml complete
- [ ] All Dockerfiles complete
- [ ] .env.example complete
- [ ] `docker compose up` → full system running
- [ ] Git commit: `feat: docker deployment complete`

---

### CHECKPOINT 16 — Final Demo
Deliverables:
- [ ] /demo route with guided scenario
- [ ] Demo data seeded
- [ ] Full end-to-end: rainfall → risk → alert → report → authority response
- [ ] DEMO_GUIDE.md
- [ ] Final documentation pass
- [ ] Git commit: `release: LAND-JEPA SIH demo ready`

---

## Acceptance Criteria Tracker

See `docs/LIMITATIONS.md` and the FINAL ACCEPTANCE CRITERIA in the project spec.
This roadmap is the single source of truth for tracking checkpoint progress.
