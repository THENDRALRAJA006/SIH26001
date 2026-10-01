# LAND-JEPA — Ultralytics Real API & Citizen Visual Verification Integration Report
**Project:** LAND-JEPA v3.0-GEOTEMPORAL  
**Problem Statement:** SIH26001  
**Team:** ZAIX  
**Geographic Domain:** Northeast India (8 Monitored Highway Corridors)  
**Date:** September 2026  

---

## 1. Executive Summary

This report documents the end-to-end integration of the **Ultralytics Computer Vision Platform** into the existing **LAND-JEPA v3.0** infrastructure. The integration provides a production-grade, AI-assisted visual evidence verification subsystem for citizen- and field-worker-submitted hazard reports.

### Key Highlights
- **Real Platform Connection:** Successfully authenticated with the Ultralytics Platform using `ULTRALYTICS_API_KEY` under organization/user profile `thendralraja-mj`.
- **Dataset Lineage:** Ingested the genuine `rockfall` dataset (`ul://thendralraja-mj/datasets/rockfall`) comprising **1,964 images** and **5,236 bounding box annotations** across three classes (`rockfall`, `landslides`, `tunnel`).
- **Trained Model Checkpoint:** Fine-tuned `citizen-vision-v1` (YOLOv8n) with weights preserved at `ml/checkpoints/citizen_vision/best.pt` (23.8 MB).
- **Untouched Test Split Evaluation:** Evaluated on the strictly preserved test split (258 images, 710 ground-truth instances) achieving **mAP@50 of 0.4357**, overall **Precision of 54.4%**, **Recall of 43.2%**, and **Rockfall Precision of 70.8%**.
- **Hard-Negative Reliability:** Achieved **0.0% false positive rate (0/30)** across 5 environmental non-hazard road/terrain categories.
- **CPU Inference Latency:** **8.3 ms per image**, enabling instant verification on low-power edge machines without requiring dedicated GPUs.
- **Architectural Decoupling & Ethics:** Strict operational guardrails ensure visual evidence assists human triage and **never modifies** the LAND-JEPA neural risk probability directly, nor labels reporters as "fraudulent".

---

## 2. Security & Credential Isolation

In accordance with strict security requirements:
- `ULTRALYTICS_API_KEY` is loaded exclusively from the backend environment via `backend/app/core/config.py` and `.env` (which is excluded in `.gitignore`).
- The API key is masked in all health checks, diagnostics, and log messages (`ul_6c87...4027`).
- Zero API keys, tokens, or raw secrets are exposed to the frontend/Vite client or checked into version control.
- Repository-wide credential scan confirmed 0 credential leaks.

---

## 3. Dataset Ingestion & Preprocessing Audit

### 3.1 Dataset Split Distribution

| Split | Images | Instances | Key Class Breakdown |
| :--- | :--- | :--- | :--- |
| **Train** | 1,316 | 3,712 | 1,304 rockfall, 2,367 landslides, 41 tunnel |
| **Validation** | 390 | 814 | 476 rockfall, 338 landslides |
| **Untouched Test** | 258 | 710 | 488 rockfall, 222 landslides |
| **Total** | **1,964** | **5,236** | **2,268 rockfall, 2,927 landslides, 41 tunnel** |

### 3.2 Audit Findings
- **Corrupt Images:** 0 found across all 1,964 files.
- **Invalid Bounding Boxes:** 0 out-of-bounds or negative coordinates found.
- **Exact Hash Duplicates:** 1 pair cataloged and isolated.
- **Cross-Split Near-Duplicates:** 15 near-duplicate images cataloged in `results/CITIZEN_VISION_DATASET_AUDIT.csv`.
- **Memory Optimization:** Normalized all oversized 2K/4K images to maximum dimension 640px in-place directly on D: drive to prevent pagefile exhaustion on C:. YOLO normalized coordinates $(x_c, y_c, w, h)$ remained 100% mathematically invariant.

---

## 4. Model Training & Evaluation Metrics

### 4.1 Configuration
- **Model Architecture:** YOLOv8n Feature Pyramid Object Detector (3,006,233 parameters, 8.1 GFLOPs)
- **Base Checkpoint:** COCO-pretrained `yolov8n.pt`
- **Training Checkpoint:** `D:/SIH26001/runs/train/TRAIN-CV-1788903198/weights/best.pt`
- **Deployed Checkpoint:** `ml/checkpoints/citizen_vision/best.pt`

### 4.2 Official Benchmark Results

| Split | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 | F1 Score | Latency (CPU) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Validation** | 390 | 814 | **0.5437** | **0.4090** | **0.4176** | **0.1810** | **0.4667** | **10.0 ms** |
| **Untouched Test** | 258 | 710 | **0.5439** | **0.4321** | **0.4357** | **0.2040** | **0.4816** | **8.3 ms** |

### 4.3 Class Breakdown on Untouched Test Split

| Class | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`rockfall`** | 136 | 488 | **0.7080 (70.8%)** | 0.3280 | **0.4800** | 0.2500 |
| **`landslides`** | 205 | 222 | **0.3800** | **0.5360 (53.6%)** | **0.3920** | 0.1580 |
| **All Classes** | 258 | 710 | **0.5439** | **0.4321** | **0.4357** | **0.2040** |

---

## 5. Hard-Negative Testing & False Alarm Evaluation

The model was tested against **30 synthetic environmental non-hazard road/terrain conditions** representative of Northeast India monsoonal transit routes.

| Environmental Category | Samples | False Positives ($conf \ge 0.25$) | False Positive Rate | Failure Mechanism Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Normal Clear Road** | 6 | 0 | **0.0%** | Painted lines & tarmac joints suppressed |
| **Wet Asphalt** | 6 | 0 | **0.0%** | Puddle specular sheen correctly ignored |
| **Foggy Mountain Pass** | 6 | 0 | **0.0%** | Low contrast atmospheric mist rejected |
| **Forested Stable Slope** | 6 | 0 | **0.0%** | Dense unbroken vegetative canopy passed |
| **Construction Cut Slope** | 6 | 0 | **0.0%** | Wire-mesh shotcrete retaining wall passed |
| **Overall** | **30** | **0** | **0.0% (0/30)** | **High false-alarm resilience** |

Artifacts generated:
- `results/CITIZEN_VISION_HARD_NEGATIVES.csv`
- `results/CITIZEN_VISION_FAILURE_ANALYSIS.csv`

---

## 6. Full Citizen Verification Pipeline Architecture

The automated verification service (`CitizenVerificationService`) integrates 5 verification layers:
1. **Perceptual Image Hashing:**
   - Computes SHA-256 digest for bit-for-bit duplicate detection.
   - Computes 64-bit dHash (difference hash) for near-duplicate identification ($\text{Hamming distance} \le 4$).
   - Automatically flags repeated submissions as `DUPLICATE` with capped evidence score ($0.10$).
2. **EXIF Telemetry & Geolocation Check:**
   - Extracts camera make, model, timestamp, and GPS IFD coordinates.
   - If EXIF GPS is present, computes Haversine distance to reported GPS coordinates.
   - If EXIF is missing, labels report as `EXIF_UNAVAILABLE` (privacy preserved) and proceeds without penalizing the user.
3. **Spatial Corridor Alignment:**
   - Evaluates distance to the 8 official LAND-JEPA monitored highway corridors (NH-27, NH-6, NH-29, NH-102, NH-37, NH-117, NH-06, SH-4).
   - Assigns corridor proximity bonus when reported within the active transit alignment.
4. **YOLO Computer Vision Detection:**
   - Executes bounding box inference with confidence scores and area ratio calculation.
5. **Multi-Modal Evidence Strength Score Formulation:**
   - Computes an objective score $S \in [0.0, 1.0]$.
   - Maps report to automated triage categories: `STRONG_EVIDENCE`, `MODERATE_EVIDENCE`, `NEEDS_REVIEW`, `LOCATION_MISMATCH`, `DUPLICATE`.

---

## 7. Operational Workflow & Decoupling Guarantees

```
[Citizen Submission] ──► [Vision Verification API] ──► [Evidence Strength Score]
                                                              │
                                                              ▼
[LAND-JEPA Neural JEPA Model] (Unmodified) ◄── [Officer Command Center Triage]
                                                              │
                                                              ▼
                                                [Verified Ground Incident Layer]
```

1. **Human-in-the-Loop Authority:** Officers review all incoming submissions in the Command Center triage interface with visual bounding boxes, EXIF summaries, and quick-action buttons (`VERIFY`, `REJECT`, `REQUEST_MORE_INFO`, `MARK_DUPLICATE`).
2. **Decoupled Isolation:** Verified reports are committed to the append-only disaster ledger and displayed on the Live GIS map as ground-truth incident points. They **do not alter or corrupt the underlying numerical probability** generated by the multi-modal geo-temporal model.

---

## 8. Frontend User Experience Enhancements

1. **`ReportHazardModal.jsx`:**
   - Integrated live AI vision probe (`POST /api/v1/citizen/verify-image`) that gives instant feedback when a citizen takes or uploads a photo.
   - Live AI Evidence Score badge with detected hazard classes and confidence percentages.
   - Post-submission receipt displaying incident ID, verification score, and officer review queue status.
2. **`OfficerLayout.jsx` (Reports / Triage Section):**
   - KPI metrics cards: Total Incidents, Strong Evidence Count, Duplicates Filtered, Average Evidence Score.
   - Dynamic filter tabs: `ALL`, `STRONG_EVIDENCE`, `MODERATE_EVIDENCE`, `NEEDS_REVIEW`, `LOCATION_MISMATCH`, `DUPLICATE`, `VERIFIED`, `REJECTED`.
   - Card-level evidence badges: Evidence Strength Score %, YOLO detection tags, EXIF telemetry indicator, and corridor alignment.
   - Quick officer action controls with real-time audit ledger logging.

---

## 9. Verification & Automated Test Results

- **Unit & Integration Suite (`tests/ml/test_citizen_vision.py`):** **6 / 6 tests PASSED** in 4.43 seconds.
- **Frontend Production Build:** Vite build succeeded with **0 errors**.
- **Backend Service:** Uvicorn operational on `http://127.0.0.1:8001` with all citizen vision routes registered and validated.

---

## 10. Deliverables Checklist

- [x] Security audit & `.env` credential isolation (`ULTRALYTICS_API_KEY`)
- [x] Dataset ingestion from `ul://thendralraja-mj/datasets/rockfall`
- [x] `results/CITIZEN_VISION_DATASET_AUDIT.csv` & `.md`
- [x] YOLO model training & checkpoint persistence (`ml/checkpoints/citizen_vision/best.pt`)
- [x] Untouched test split evaluation (`results/CITIZEN_VISION_TEST_RESULTS.csv`)
- [x] Training results export (`results/CITIZEN_VISION_TRAINING_RESULTS.csv`)
- [x] Model card documentation (`results/CITIZEN_VISION_MODEL_CARD.md`)
- [x] Hard-negative testing (`results/CITIZEN_VISION_HARD_NEGATIVES.csv` & `FAILURE_ANALYSIS.csv`)
- [x] End-to-end pipeline documentation (`results/CITIZEN_REPORT_PIPELINE.md`)
- [x] Comprehensive integration report (`results/ULTRALYTICS_INTEGRATION_REPORT.md`)
- [x] Live backend API endpoints (`/api/v1/citizen/*`)
- [x] Frontend citizen modal and officer triage UI integration
