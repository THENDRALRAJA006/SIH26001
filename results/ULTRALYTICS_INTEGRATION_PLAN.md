# LAND-JEPA — Ultralytics Real API + Citizen Visual Verification Integration Plan

**Project:** LAND-JEPA v3.0-GEOTEMPORAL  
**Module:** Citizen Visual Evidence Verification & Ultralytics Computer-Vision Subsystem  
**Team:** Team ZAIX · Northeast India  
**Date:** September 2026  

---

## 1. Existing Architecture & Reusable Components

| Subsystem | Existing Implementation | Reusability in Visual Verification |
| :--- | :--- | :--- |
| **Backend Framework** | FastAPI application (`backend/app/main.py`), Pydantic v2 schemas | Mount new `/api/v1/citizen/reports` endpoints alongside `/alerts/citizen-report`. |
| **Citizen Reporting Model** | `CitizenReport` ORM (`backend/app/models/reports.py`) | Extend schema / database fields with `image_hash`, `perceptual_hash`, `visual_detections`, `evidence_score`, `metadata_status`, `location_status`, `temporal_status`, `duplicate_status`. |
| **Officer Review Workflow** | `GET /api/v1/alerts/reports`, `POST /api/v1/alerts/reports/{id}/action` | Extend to return visual evidence, bounding boxes, hashes, and support triage filters (`NEW`, `NEEDS_REVIEW`, `STRONG_EVIDENCE`, `LOCATION_MISMATCH`, `DUPLICATE`, `VERIFIED`, `REJECTED`). |
| **Storage Subsystem** | `UPLOAD_DIR: str = "data/uploads"` in `config.py` | Store uploaded citizen images in `data/uploads/citizen/` with SHA-256 filenames on the 180GB D: partition. |
| **Geospatial & Corridors** | `gis/real_zones.py`, `HIGHWAY_CORRIDORS` in `arcgisService.js` | Validate reported coordinates against nearest monitored MoRTH corridor for location consistency. |
| **Prediction Pipeline** | `RiskPipelineService`, `GeoTemporalInferenceService` | Kept strictly separated. Citizen visual evidence informs field operational context and event catalogs, **never** directly skewing the multi-modal JEPA probability. |
| **Citizen UI** | `ReportHazardModal.jsx` | Send image + metadata to `/api/v1/citizen/reports`, display evidence feedback ("Potential hazard evidence detected. Awaiting verification"). |
| **Officer Dashboard** | `OfficerLayout.jsx` Citizen Triage section | Enhanced visual inspection drawer: Image + Map + YOLO Bounding Boxes + Confidence + Metadata + Duplicate Status + Action Buttons. |

---

## 2. Security & Credential Isolation

- **Secret Variable**: `ULTRALYTICS_API_KEY` stored exclusively in root `.env`.
- **Git Protection**: Root `.env` is committed to `.gitignore` (verified line 4).
- **Template**: `.env.example` updated with `ULTRALYTICS_API_KEY=YOUR_ULTRALYTICS_KEY`.
- **Leakage Prevention**:
  - `ULTRALYTICS_API_KEY` is loaded strictly on the Python backend via `os.getenv("ULTRALYTICS_API_KEY")`.
  - Never passed into React / Vite public environment (`VITE_*`).
  - Never returned in API responses.
  - Never logged to console, disk logs, or audit records.
  - Zero exposed credentials verified across repository.

---

## 3. Data Flow Architecture

```
                       CITIZEN REPORTING FLOW
                               │
                       [ Citizen Upload ]
                    (Photo, GPS, Desc, Time)
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       1. Image Integrity Verification        │
        │   • SHA-256 Checksum calculation             │
        │   • dHash / pHash perceptual similarity      │
        │   • Duplicate detection against image ledger │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       2. Metadata & EXIF Analysis            │
        │   • Extract EXIF DateTimeOriginal & GPS      │
        │   • Status: AVAILABLE | EXIF_UNAVAILABLE     │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       3. Spatio-Temporal Consistency         │
        │   • Haversine distance to nearest corridor   │
        │   • Time discrepancy: capture vs submission  │
        │   • Status: MATCH | POSSIBLE_MISMATCH        │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │          4. YOLO Vision Detection            │
        │   • Ultralytics YOLO model inference         │
        │   • Output: classes, confidence, bbox coords │
        │   • Detects visible hazard evidence          │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │       5. Evidence Strength Scoring           │
        │   • Transparent composite evidence score     │
        │   • Assigns triage status:                   │
        │     - STRONG_VISUAL_EVIDENCE                 │
        │     - PROBABLE_HAZARD                        │
        │     - NEEDS_REVIEW                           │
        │     - LOW_EVIDENCE / DUPLICATE               │
        └──────────────────────┬───────────────────────┘
                               │
                               ▼
        ┌──────────────────────────────────────────────┐
        │    6. Officer Triage (HUMAN-IN-THE-LOOP)     │
        │   • Officer inspects Image, Boxes, Map       │
        │   • Decision: VERIFY | REJECT | REQUEST_INFO │
        └──────────────────────┬───────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
 [ Verified Hazard Event ]              [ Unverified / Rejected ]
 (Added to Operational Event Ledger)    (Archived with Audit Log)
            │
            ▼
 [ Future Model Retraining Catalog ]
 (Controlled, Human-Gated Approval)
```

---

## 4. Dataset & Model Specifications

- **Dataset**: `rockfall` (`ul://thendralraja-mj/datasets/rockfall`)
- **Recorded Counts**: 1,964 images, 5,236 annotations (Train: 1,316, Val: 390, Test: 258)
- **Model Version**: `citizen-vision-v1`
- **Base Architecture**: YOLO (YOLO26n / YOLOv5u / YOLOv8n)
- **Training Setup**:
  - Image size: 960 (or resource-adaptive 1280 depending on memory)
  - Epochs: 100
  - Batch: auto / adaptive
  - Target: Bounding box object detection for visible ground hazard evidence
- **Audit Deliverables**:
  - `results/CITIZEN_VISION_DATASET_AUDIT.csv`
  - `results/CITIZEN_VISION_DATASET_AUDIT.md`
  - `results/CITIZEN_VISION_TRAINING_RESULTS.csv`
  - `results/CITIZEN_VISION_TEST_RESULTS.csv`
  - `results/CITIZEN_VISION_HARD_NEGATIVES.csv`
  - `results/CITIZEN_VISION_FAILURE_ANALYSIS.csv`

---

## 5. Files to Create & Files to Modify

### Files to Create
1. `ml/models/citizen_vision_detector.py`: Ultralytics YOLO loader, inference runner, and bounding box extractor.
2. `backend/app/services/citizen_verification_service.py`: Automated pipeline: SHA-256, perceptual hash, EXIF parsing, spatial/temporal consistency, evidence scoring.
3. `backend/app/api/v1/citizen_vision.py`: Endpoints for `POST /api/v1/citizen/reports`, report evidence inspection, and officer triage actions.
4. `tests/ml/test_citizen_vision.py`: Comprehensive test suite covering integrity, EXIF, location check, YOLO inference, edge cases, and API integration.
5. `scripts/audit_citizen_dataset.py`: Comprehensive dataset auditor (corrupt images, invalid boxes, duplicates, train/val/test leakage).
6. `scripts/train_citizen_vision.py`: Training automation and metrics export script.
7. `scripts/evaluate_hard_negatives.py`: Hard-negative evaluation script against non-hazard environmental images.

### Files to Modify
1. `.env.example`: Add `ULTRALYTICS_API_KEY=YOUR_ULTRALYTICS_KEY`.
2. `.env`: Configure active `ULTRALYTICS_API_KEY`.
3. `backend/app/core/config.py`: Add `ULTRALYTICS_API_KEY`, `CITIZEN_VISION_MODEL_PATH`, and directory settings.
4. `backend/app/main.py`: Register citizen vision router and static upload file mount.
5. `frontend/dashboard/src/components/ReportHazardModal.jsx`: Wire submission to `/api/v1/citizen/reports` with evidence receipt display.
6. `frontend/dashboard/src/pages/OfficerLayout.jsx`: Enhance Citizen Report Triage with visual evidence inspection, bounding box viewer, and new filter categories.

---

## 6. Testing & Quality Assurance Plan

1. **Unit & Functional Tests (`test_citizen_vision.py`)**:
   - Verify model loading and class mapping.
   - Verify bounding box format `[x1, y1, x2, y2]` and confidence range $[0.0, 1.0]$.
   - Corrupt, empty, and non-image file handling.
   - Perceptual hash and SHA-256 duplicate identification.
   - Missing EXIF handled as `EXIF_UNAVAILABLE` without false fraud assumptions.
   - Distance calculation and `POSSIBLE_MISMATCH` logic.
2. **Synthetic Integration Test**: Single test report through full pipeline without database pollution.
3. **Real Image Inference**: Benchmark single real dataset image measuring CPU/GPU latency and throughput.
4. **Hard-Negative Evaluation**: Evaluate false positive rate against non-hazard mountain/road images.
