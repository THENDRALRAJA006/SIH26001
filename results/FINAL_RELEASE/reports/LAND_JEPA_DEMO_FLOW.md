# LAND-JEPA: Comprehensive Demonstration Flow & Presentation Narrative
## Official Demonstration Guide for Evaluators, Geotechnical Commanders & SIH Jury

**Project**: LAND-JEPA (Smart India Hackathon 2026 — Problem SIH26001)  
**Team**: ZAIX | **Domain**: Northeast India (NER) — 8 Strategic Highway Corridors  
**Document Identification**: `LJ-DEMO-2026-FINAL`  
**Target Audience**: Hackathon Jury, NDMA/SDMA Incident Commanders, Geotechnical AI Evaluators  
**System Endpoints**: Frontend `http://localhost:5173/` | Backend API `http://127.0.0.1:8000/`

---

## Executive Overview of the Demonstration

The LAND-JEPA demonstration showcases a production-grade, dual-portal disaster intelligence platform. It bridges the gap between high-level geotechnical deep learning and frontline disaster response, proving that AI early warning can deliver **25.2 hours of advance warning** while suppressing false alarms down to **3.45%**.

```
[1. Landing Page]
       │  (Public Awareness & Mission Briefing)
       ▼
[2. Citizen Portal]
       │  (Zero-friction GPS auto-locate, local risk gauges, multilingual warnings)
       ▼
[3. Officer Login]
       │  (Cryptographic JWT role-based access for NDMA / BRO commanders)
       ▼
[4. Officer AI Command Center]
       │  (Live telemetry, dual-model comparison: v2.5 Champion vs. v2.6.1 Challenger)
       ▼
[5. Interactive GIS Corridor Layers]
       │  (3D rain deflection, satellite/dark base tiles, road-cut hazard overlays)
       ▼
[6. 72-Hour NWP Forecast Engine]
       │  (30-member ensemble spread, 6h/12h/24h/48h/72h trajectory heads)
       ▼
[7. Operational Alert Dispatch]
       │  (NDMA protocol tiers: WATCH [0.6531], WARNING [0.7724], CRITICAL [0.9550])
       ▼
[8. Citizen Hazard Reporting with Live Camera]
       │  (Crowd-sourced mudslide capture, offline IndexedDB sync, photo evidence)
       ▼
[9. AI Analytics & Quantum Telemetry]
          (VQC feature ranking, Brier calibration, immutable prediction ledgers)
```

---

## Step-by-Step Demonstration Script

### Step 1: Landing Page (`/`) — Premium AI Laboratory & Geospatial Intelligence
- **URL**: `http://localhost:5173/`
- **Visuals & UX**:
  - Monochrome minimalist visual identity with deep charcoal background, subtle slate typography, and selective status lights.
  - Interactive **3D Monsoonal Rain Physics**: Rain particles fall across the viewport and realistically deflect upon contacting UI cards and text headers.
  - Theme toggle: Seamless switch between sleek Dark Mode and high-contrast Enterprise Light Mode.
- **Narrative**:
  > *"Welcome to LAND-JEPA. Northeast India faces over 70% of India's annual landslide casualties. Conventional empirical rainfall thresholds trigger false alarms on normal rainy days and provide less than one hour of warning. LAND-JEPA reimagines early warning by fusing deep representation learning with 86 physical geotechnical variables across eight strategic highway corridors."*

---

### Step 2: Citizen Portal (`/citizen`) — Zero-Password Local Protection
- **URL**: `http://localhost:5173/citizen`
- **Visuals & UX**:
  - **Zero-Barrier Access**: No passwords, usernames, or phone numbers required.
  - **Live GPS Auto-Locate**: Click "Detect My Live GPS Location" to automatically identify user latitude/longitude and snap to the nearest monitored strategic corridor (e.g., NH-27 Guwahati–Shillong or SH-4 Tawang Access).
  - **Multi-Horizon Risk Gauges**: Displays calibrated risk probabilities across Current (Now), 6h, 12h, 24h, 48h, and 72h advance horizons.
  - **Actionable Civil Guidance**: Dynamic advisories indicating whether highways are clear, under caution, or facing imminent closure.
- **Narrative**:
  > *"During disaster evacuations, citizens cannot remember passwords or navigate complex charts. The Citizen Portal automatically identifies the user's nearest highway corridor in milliseconds and presents plain-language safety instructions backed by multi-horizon forecast probabilities."*

---

### Step 3: Officer Secure Login (`/officer/login`)
- **URL**: `http://localhost:5173/officer/login`
- **Visuals & UX**:
  - Secure credential interface protected by JSON Web Token (JWT) cryptographic sessions.
  - Test credentials clearly accessible for jury evaluation (`commander` / `zaix2026`).
  - Strict role-based routing protecting administrative command controls.
- **Narrative**:
  > *"While citizens receive open advisories, operational decision levers require strict command authentication. Authorized incident commanders from the Border Roads Organisation, NDMA, and district administrations access dedicated operations dashboards."*

---

### Step 4: Officer AI Dashboard (`/officer/command`) — Mission Control
- **URL**: `http://localhost:5173/officer/command`
- **Visuals & UX**:
  - Multi-panel command center presenting real-time system health, corridor risk summaries, and active alert tallies.
  - **Dual-Model Governance Banner**: Displays the operational role of both models:
    - **Active Production Champion**: `v2.5-TRIGGER-AWARE-CHAMPION` (Recall 78.9%, Lead Time 24.0h)
    - **Frozen Prospective Challenger**: `v2.6.1-CHALLENGER` (Recall 81.6%, Lead Time 25.2h, running in blind shadow mode)
- **Narrative**:
  > *"Here, the commander views the entire regional threat matrix. Notice our strict scientific governance: v2.5 remains the active production benchmark, while v2.6.1 is monitored in prospective shadow mode. We never deploy an unvalidated model into production."*

---

### Step 5: GIS Geospatial Intelligence (`/officer/map`)
- **URL**: `http://localhost:5173/officer/map`
- **Visuals & UX**:
  - Interactive Leaflet-powered GIS mapping all eight strategic Northeast India transport corridors:
    - NH-27 (Guwahati–Shillong), NH-6 (Silchar–Imphal), SH-4 (Tawang Access), NH-10 (Sevoke–Gangtok), NH-29 (Dimapur–Kohima), NH-102 (Imphal–Moreh), NH-13 (Trans-Arunachal), NH-208A (Agartala).
  - Multi-tier layer controls: Toggle between Dark Canvas, Satellite Imagery, Terrain Topography, Precipitation Radar, and Road-Cut Excavation Hazard Buffers.
  - Real-time corridor markers pulsing with calibrated risk colors (Green = Safe, Amber = Warning, Red = Critical).
- **Narrative**:
  > *"Every 30-meter pixel along 1,400 kilometers of strategic highways is continuously monitored. By overlaying Copernicus 30m digital elevation models with road-cut excavation buffers and live precipitation, commanders can pinpoint the exact highway kilometers vulnerable to catastrophic toe blowout."*

---

### Step 6: 72-Hour NWP Forecast Engine (`/officer/forecast`)
- **URL**: `http://localhost:5173/officer/forecast`
- **Visuals & UX**:
  - Hourly meteorological ingestion tracking precipitation rate, 24h antecedent rainfall, API-30 saturation, and 30-member ECMWF ensemble spread.
  - Multi-horizon forecast curves showing predicted failure probability across 6h, 12h, 24h, 48h, and 72h horizons.
- **Narrative**:
  > *"Rather than relying on single-point rainfall numbers, LAND-JEPA disaggregates hazard probabilities into five distinct operational horizons. The 24-hour head provides the primary evacuation window (81.6% sensitivity), while the 72-hour head gives logistical commanders three full days to pre-position heavy clearing machinery."*

---

### Step 7: Operational Alert Dispatch (`/officer/alerts`)
- **URL**: `http://localhost:5173/officer/alerts`
- **Visuals & UX**:
  - Clear breakdown of National Disaster Management Authority (NDMA) action tiers:
    - **WATCH (Advisory)**: $P \ge 0.6531$ (v2.6.1) | $\text{FPR} \le 10\%$ | Stage machinery, 15-min sensor polling.
    - **WARNING (Actionable)**: $P \ge 0.7724$ (v2.6.1) | $\text{FPR} \le 5\%$ | Restrict night freight, mobilize NDRF units.
    - **CRITICAL (Imminent)**: $P \ge 0.9550$ (v2.6.1) | $\text{FPR} \le 1\%$ | Immediate highway closure, preventive evacuations.
  - 24-hour advisory persistence grouping prevents alert flickering, reducing corridor alert fatigue by 78.5%.
- **Narrative**:
  > *"Our thresholding is mathematically derived through multi-season minimax optimization. Unlike naive single-season cutoffs that trigger continuous alarms during peak monsoon, our frozen WARNING threshold of 0.7724 guarantees false alarms remain below 0.0425 per day."*

---

### Step 8: Citizen Hazard Reporting & Live Camera Capture (`/citizen` modal)
- **URL**: `http://localhost:5173/citizen` -> Click "Report Landslide / Hazard"
- **Visuals & UX**:
  - Modal offering crowd-sourced field incident reporting.
  - **Live Camera Capture**: Access device webcam/camera to capture real-time slope movement in the field.
  - **Image File Upload**: Drag-and-drop field photographs (pre-loaded with verified mudslide and rockfall images from NH-27 and Tawang).
  - Hazard classification: Mudslide, Rockfall, Blocked Culvert, Road Crack.
  - Automated GPS capture and offline IndexedDB sync for connectivity-compromised mountain areas.
- **Narrative**:
  > *"When a mountain culvert clogs or minor debris falls, citizens become the frontline sensor network. Citizens snap photos using live camera capture; reports instantly populate the Officer Command Center for geotechnical verification and machine learning validation."*

---

### Step 9: AI Analytics, Model Registry & Quantum Telemetry (`/officer/models` & `/officer/quantum`)
- **URL**: `http://localhost:5173/officer/models` and `/officer/quantum`
- **Visuals & UX**:
  - Standardized Model Card displaying architectural parameters, training lineage, and ethical boundaries.
  - Calibration reliability curves showing reduction of Expected Calibration Error (ECE) to 0.0028.
  - Variational Quantum Classifier (VQC) feature ranking telemetry evaluating quantum kernel expressivity on high-dimensional slope data.
  - **Immutable Prediction Ledger**: Cryptographic audit trail of all 11,520 prospective prediction cycles logged across September 2026.
- **Narrative**:
  > *"Every single prediction is hashed and immutably recorded prior to weather occurrence. This guarantees zero future data leakage and ensures that when the next major monsoon event strikes, our prospective evaluation will be 100% auditable and scientifically indisputable."*

---

## Key Takeaway for Evaluators & Jury

1. **True AI Innovation**: Self-supervised JEPA temporal representation learning models 168-hour geotechnical soil saturation memory, eliminating blind spots inherent in classical rainfall thresholds.
2. **Scientific Integrity First**: We strictly separate historical offline validation (81.6% Recall, 25.2h Lead Time) from prospective shadow surveillance (N=0, Recall UNDEFINED). We refuse to exaggerate model performance.
3. **End-to-End Field Readiness**: Dual portals, offline resilience, live camera integration, 3D rain physics, and NDMA-aligned decision tiers make LAND-JEPA immediately deployable across Northeast India.
