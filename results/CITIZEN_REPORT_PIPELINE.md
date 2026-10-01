# LAND-JEPA — Citizen Hazard Visual Evidence Verification Pipeline
**SIH26001 · Team ZAIX · Northeast India**  
**Version:** v3.0-GEOTEMPORAL  
**Module:** Citizen Hazard Evidence Verification Subsystem  

---

## 1. Executive Summary & Purpose

The **LAND-JEPA Citizen Visual Evidence Subsystem** provides an automated, objective, and ethical verification pipeline for citizen- and field-worker-submitted hazard photos across the 8 monitored highway corridors of Northeast India.

Instead of relying on unverified text claims or attempting to pass judgment on citizen honesty, the subsystem applies **Ultralytics YOLOv8 object detection**, **perceptual image hashing**, **EXIF telemetry inspection**, and **spatial corridor alignment** to calculate an objective **Evidence Strength Score (0.0 to 1.0)**.

```
┌─────────────────────────┐
│ Citizen / Field Officer │
│ (Live Camera / Upload)  │
└────────────┬────────────┘
             │ Photo + GPS + Description
             ▼
┌────────────────────────────────────────────────────────┐
│      CITIZEN VERIFICATION SERVICE PIPELINE             │
│                                                        │
│  1. Image Hashing (SHA-256 + 64-bit dHash)            │
│     ├── Identical SHA-256 match → EXACT_DUPLICATE      │
│     └── Hamming distance ≤ 4 → NEAR_DUPLICATE          │
│                                                        │
│  2. EXIF Telemetry Extraction                          │
│     ├── Camera Make, Model, Timestamp                  │
│     ├── EXIF GPS vs Reported GPS (Haversine distance)  │
│     └── Missing EXIF → EXIF_UNAVAILABLE (NOT fraud)    │
│                                                        │
│  3. Spatial Corridor Alignment                         │
│     └── Nearest LAND-JEPA Monitored Corridor (Km 0–8)  │
│                                                        │
│  4. Ultralytics YOLOv8 Vision Inference                │
│     ├── Classes: rockfall (0), landslides (1), tunnel  │
│     ├── Bounding Boxes & Area Coverage Ratio           │
│     └── Inference Latency: 8.3ms on CPU                │
│                                                        │
│  5. Multi-Modal Evidence Strength Score Formulation    │
│     └── Score ∈ [0.0, 1.0]                             │
└────────────┬───────────────────────────────────────────┘
             │ Verified Evidence Card + Receipt
             ▼
┌────────────────────────────────────────────────────────┐
│     OFFICER COMMAND CENTER (HUMAN-IN-THE-LOOP)         │
│                                                        │
│  Triage Status Categories:                             │
│  - STRONG_EVIDENCE (Score ≥ 70%, Priority Review)      │
│  - MODERATE_EVIDENCE (Score ≥ 45%, Standard Inspection)│
│  - NEEDS_REVIEW (Low Confidence, Awaiting Patrol)      │
│  - LOCATION_MISMATCH (EXIF GPS mismatch > 15 km)       │
│  - DUPLICATE (Linked to original incident)             │
│                                                        │
│  Officer Actions:                                      │
│  [VERIFY] | [REJECT] | [REQUEST_INFO] | [DUPLICATE]   │
└────────────┬───────────────────────────────────────────┘
             │ Verified Incident Layer Sync
             ▼
┌────────────────────────────────────────────────────────┐
│  LIVE ARCGIS & AUDIT LEDGER (DECOUPLED ARCHITECTURE)   │
│                                                        │
│  ✓ Logged in append-only disaster ledger               │
│  ✓ Visible on Officer Live GIS as Ground Truth Incident │
│  ✕ LAND-JEPA neural risk probability remains DECOUPLED │
└────────────────────────────────────────────────────────┘
```

---

## 2. Core Operational & Ethical Guardrails

### Rule 1: Evidence Verification, Not Citizen Fraud
The model inspects **physical features in pixels** (angular rock fragmentation, slope scarps, mud runout, retaining wall damage). It **never** labels a human citizen reporter as "fraudulent", "fake", or "malicious".

### Rule 2: Non-Punitive EXIF Policy
Many mobile messaging applications (WhatsApp, Telegram, Signal) and web browser canvas uploaders intentionally strip EXIF metadata to protect user privacy. In accordance with LAND-JEPA operational guidelines:
- If EXIF is missing: Status is cataloged as `EXIF_UNAVAILABLE` (Privacy Preserved).
- Missing EXIF is **never** considered grounds for rejecting a report.
- Location verification falls back to browser GPS telemetry and corridor axis proximity.

### Rule 3: Decoupled Architectural Isolation
Citizen reports provide critical observational context and ground-truth validation for civil defense. However, **citizen reports do not automatically modify or overwrite the LAND-JEPA neural risk probability**. 
- Neural forecasting is driven by physics-informed joint-embedding predictive architecture (geological, hydrological, seismic, and InSAR data).
- Verified citizen reports enter the disaster ledger and live GIS layer as **verified ground incidents**, requiring human officer sign-off before altering civil defense alert tiers.

---

## 3. Mathematical Formulation of the Evidence Strength Score

The Evidence Strength Score $S \in [0.0, 1.0]$ aggregates four orthogonal evidence vectors:

$$S = S_{\text{vision}} + S_{\text{area}} + S_{\text{spatial}} - P_{\text{mismatch}} - P_{\text{duplicate}}$$

### 1. Vision Confidence Component ($S_{\text{vision}} \le 0.60$)
$$S_{\text{vision}} = \max_{d \in \text{hazards}}(\text{conf}_d) \times 0.60$$
Where $\text{hazards} \in \{\text{rockfall}, \text{landslides}\}$.

### 2. Visible Hazard Area Bonus ($S_{\text{area}} \le 0.15$)
$$S_{\text{area}} = \min(1.5 \times \text{Ratio}_{\text{hazard\_area}}, 0.15)$$
Gives greater weight to mass wasting events that occupy a substantial portion of the camera frame.

### 3. Spatial Corroboration Component ($S_{\text{spatial}} \le 0.25$)
- $+0.15$ if EXIF GPS matches reported location within $2.0\text{ km}$.
- $+0.08$ if EXIF GPS is within $15.0\text{ km}$.
- $+0.05$ baseline if EXIF is unavailable but reported within monitored corridor boundary.
- $+0.10$ if within $10.0\text{ km}$ of one of the 8 monitored Northeast highway corridors.

### 4. Penalties
- **Location Mismatch Penalty ($P_{\text{mismatch}} = 0.25$):** Applied only when valid camera EXIF GPS explicitly contradicts the reported location by $> 25\text{ km}$.
- **Duplication Cap ($P_{\text{duplicate}}$):** If an image has identical SHA-256 or dHash Hamming distance $\le 4$ with an existing submission, the score is capped at $0.10$ and flagged as `DUPLICATE`.

---

## 4. API Endpoints Reference

| Method | Path | Description | Access |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/citizen/reports` | Submit citizen report with photo evidence | Public / Citizen |
| `POST` | `/api/v1/citizen/reports/upload` | Multipart file upload for live mobile camera | Public / Field |
| `GET` | `/api/v1/citizen/reports` | Triage list with filters (`status`, `corridor_id`) | Officer Command |
| `GET` | `/api/v1/citizen/reports/{id}` | Detailed incident report card with YOLO detections | Officer Command |
| `POST` | `/api/v1/citizen/reports/{id}/action` | Officer review decision (`VERIFY`, `REJECT`, etc.) | Officer Command |
| `POST` | `/api/v1/citizen/verify-image` | Instant visual hazard detection probe | Public / Officer |
| `GET` | `/api/v1/citizen/stats` | Triage metrics, duplicate count, avg score | Officer Command |

---

## 5. Security & Confidentiality

1. `ULTRALYTICS_API_KEY` is loaded strictly on the backend via environment variables (`.env`).
2. The key is never exposed to the frontend, never returned in API payloads, and never logged in console outputs.
3. User privacy is maintained: citizen submissions do not require PII, and image metadata is handled according to disaster management data protection standards.
