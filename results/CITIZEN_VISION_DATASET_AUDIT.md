# LAND-JEPA Citizen Visual Evidence — Scientific Dataset Audit Report

**Dataset:** `ul://thendralraja-mj/datasets/rockfall`  
**Target Subsystem:** Citizen Hazard Visual Verification  
**Evaluation Standard:** Zero-Leakage & Class-Balance Integrity  
**Date:** September 2026 · SIH26001 · Team ZAIX  

---

## 1. Executive Dataset Summary

| Metric | Measured Value | Requirement / Target | Audit Status |
| :--- | :--- | :--- | :--- |
| **Total Images** | **1,964** | ≥ 1,000 images | **PASS** |
| **Total Annotations** | **5,236** | ≥ 3,000 instances | **PASS** |
| **Annotated Classes** | **3** | Defined Ground Truth | **PASS** |
| **Corrupt / Unreadable Images** | **0** | 0 | **PASS** |
| **Invalid Bounding Boxes** | **0** | 0 normalized errors | **PASS** |
| **Exact Duplicates (SHA-256)** | **1** | Tracked / Monitored | **AUDITED** |
| **Train/Val/Test Split Leakage** | **15** | 0 leakage | **FLAGGED** |

---

## 2. Partition Splits & Class Distribution

### Split Partitioning
- **Train Split:** 1,316 images (67.0%)
- **Validation Split:** 390 images (19.9%)
- **Test Split (Untouched):** 258 images (13.1%)

### Per-Class Instance Distribution

| Class ID | Class Label | Train Instances | Val Instances | Test Instances | Total Instances | Class Share |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | `rockfall` | 2,675 | 476 | 488 | **3,639** | 69.5% |
| **1** | `landslides` | 1,031 | 338 | 222 | **1,591** | 30.4% |
| **2** | `tunnel` | 6 | 0 | 0 | **6** | 0.1% |

---

## 3. Dataset Scope & Domain Appropriateness Assessment

> [!IMPORTANT]
> **Audit Finding on Dataset Scope:**
> The `rockfall` dataset encompasses three distinct, verified hazard and infrastructure classes:
> 1. `rockfall`: Detached rock fragments, boulder deposits, and road-blocking rocks.
> 2. `landslides`: Slope movement, scarp failures, mud/debris washouts, and active slope cuts.
> 3. `tunnel`: Mountain highway tunnels and structural portals (critical infrastructure in landslide corridors).
>
> **Domain Determination:**
> The dataset is **appropriate for Highway Rockfall & Landslide Hazard Verification**, providing direct visual evidence coverage for road-blocking slope failures across Northeast India highway alignments.

---

## 4. Split Leakage & Geometry Analysis

- **Bounding Box Integrity:** All bounding box coordinates are strictly bounded within normalized $[0.0, 1.0]$.
- **Cross-Split Leakage:** 15 cross-split duplicate instances identified and logged into `CITIZEN_VISION_DATASET_AUDIT.csv`.
- **Test Set Protection:** The 258 test images are quarantined for final unbiased validation report.

Generated automatically by `scripts/audit_citizen_dataset.py`.
