"""
LAND-JEPA — Citizen Hazard Visual Verification Model Training & Evaluation
Model: citizen-vision-v1
SIH26001 · Team ZAIX · Northeast India

Executes:
1. Training run of YOLO detector on the audited rockfall dataset
2. Model weight export to ml/checkpoints/citizen_vision/best.pt
3. Validation metrics extraction (Precision, Recall, mAP50, mAP50-95, F1)
4. Evaluation on untouched test set (258 images)
5. Generates:
   - results/CITIZEN_VISION_TRAINING_RESULTS.csv
   - results/CITIZEN_VISION_TEST_RESULTS.csv
   - results/CITIZEN_VISION_MODEL_CARD.md
"""

import os
import sys
import csv
import time
import shutil
from pathlib import Path
from ultralytics import YOLO

# Unicode stdout
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = Path("D:/SIH26001/data/datasets/rockfall/data.yaml")
CHECKPOINT_DIR = ROOT / "ml" / "checkpoints" / "citizen_vision"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "citizen-vision-v1"
DATASET_VERSION = "rockfall-v1"
RUN_ID = f"TRAIN-CV-{int(time.time())}"

EPOCHS = 5
IMGSZ = 480
BATCH = 8
DEVICE = "cpu"
WORKERS = 0

print(f"============================================================")
print(f" LAND-JEPA CITIZEN VISION TRAINING — {MODEL_VERSION}")
print(f" Dataset: {DATA_YAML}")
print(f" Run ID: {RUN_ID} | Epochs: {EPOCHS} | ImgSz: {IMGSZ} | Batch: {BATCH}")
print(f"============================================================")

t_start = time.time()
model = YOLO("yolov8n.pt")

# Execute Training
results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=WORKERS,
    project="D:/SIH26001/runs/train",
    name=RUN_ID,
    exist_ok=True,
    verbose=True,
    plots=True,
)
t_duration = round(time.time() - t_start, 2)

# Copy best.pt to ml/checkpoints/citizen_vision/best.pt
best_weights_src = Path(f"D:/SIH26001/runs/train/{RUN_ID}/weights/best.pt")
best_weights_dst = CHECKPOINT_DIR / "best.pt"
if best_weights_src.exists():
    shutil.copy2(best_weights_src, best_weights_dst)
    print(f"\n[OK] Model weights preserved to {best_weights_dst} ({best_weights_dst.stat().st_size:,} bytes)")
else:
    print(f"[WARN] Expected weights {best_weights_src} not found, checking last.pt...")
    last_src = Path(f"D:/SIH26001/runs/train/{RUN_ID}/weights/last.pt")
    if last_src.exists():
        shutil.copy2(last_src, best_weights_dst)

# ── 1. Validation Set Evaluation ─────────────────────────────────────────
print("\n[VALIDATION] Running official validation pass on 390 val images...")
val_model = YOLO(str(best_weights_dst))
val_metrics = val_model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    plots=True,
)

val_mp = float(val_metrics.box.mp)
val_mr = float(val_metrics.box.mr)
val_map50 = float(val_metrics.box.map50)
val_map50_95 = float(val_metrics.box.map)
val_f1 = float(2 * val_mp * val_mr / (val_mp + val_mr + 1e-8))

print(f"Validation Metrics: P={val_mp:.4f}, R={val_mr:.4f}, mAP50={val_map50:.4f}, mAP50-95={val_map50_95:.4f}, F1={val_f1:.4f}")

# Save Validation / Training Results CSV
train_csv_path = RESULTS_DIR / "CITIZEN_VISION_TRAINING_RESULTS.csv"
with open(train_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "model_version", "dataset_version", "training_run_id", "epochs",
        "image_size", "batch_size", "training_duration_sec", "precision",
        "recall", "mAP50", "mAP50_95", "f1_score"
    ])
    writer.writerow([
        MODEL_VERSION, DATASET_VERSION, RUN_ID, EPOCHS,
        IMGSZ, BATCH, t_duration, f"{val_mp:.4f}",
        f"{val_mr:.4f}", f"{val_map50:.4f}", f"{val_map50_95:.4f}", f"{val_f1:.4f}"
    ])
print(f"[SAVED] Training Results CSV: {train_csv_path}")

# ── 2. Untouched Test Set Evaluation ─────────────────────────────────────
print("\n[TEST SET] Evaluating on untouched test split (258 images)...")
test_metrics = val_model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    plots=True,
)

test_mp = float(test_metrics.box.mp)
test_mr = float(test_metrics.box.mr)
test_map50 = float(test_metrics.box.map50)
test_map50_95 = float(test_metrics.box.map)
test_f1 = float(2 * test_mp * test_mr / (test_mp + test_mr + 1e-8))

print(f"Test Metrics: P={test_mp:.4f}, R={test_mr:.4f}, mAP50={test_map50:.4f}, mAP50-95={test_map50_95:.4f}, F1={test_f1:.4f}")

# Save Test Results CSV
test_csv_path = RESULTS_DIR / "CITIZEN_VISION_TEST_RESULTS.csv"
with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "split", "model_version", "dataset_version", "test_images",
        "precision", "recall", "mAP50", "mAP50_95", "f1_score"
    ])
    writer.writerow([
        "test", MODEL_VERSION, DATASET_VERSION, 258,
        f"{test_mp:.4f}", f"{test_mr:.4f}", f"{test_map50:.4f}", f"{test_map50_95:.4f}", f"{test_f1:.4f}"
    ])
print(f"[SAVED] Test Results CSV: {test_csv_path}")

# ── 3. Generate Citizen Vision Model Card ────────────────────────────────
model_card_path = RESULTS_DIR / "CITIZEN_VISION_MODEL_CARD.md"
model_card_md = f"""# LAND-JEPA Citizen Vision Verification — Model Card

**Model Version:** `{MODEL_VERSION}`  
**Architecture:** YOLOv8n Feature Pyramid Object Detector  
**Training Run ID:** `{RUN_ID}`  
**Dataset:** `ul://thendralraja-mj/datasets/rockfall` (Version: `{DATASET_VERSION}`)  
**Task:** 2D Bounding Box Object Detection (`detect`)  
**Deployment Scope:** AI-Assisted Visual Evidence Verification for Citizen Hazard Reports  
**Date:** September 2026 · Team ZAIX · Northeast India  

---

## 1. Intended Use & Ethical Operational Guardrails

- **Intended Purpose:** Detect visible physical hazard evidence (detached boulders, active slope failures, structural highway portals) in user-submitted photos to compute an objective Evidence Strength Score.
- **Strict Limitation:** The model detects **visual evidence only**. It **never** classifies a human reporter as "truthful" or "fraudulent".
- **Decoupled Architecture:** Detections do **not** automatically trigger emergency civil defense warnings or directly modify LAND-JEPA geo-temporal hazard probabilities. All reports require **human officer review**.

---

## 2. Training Hyperparameters & System Profile

| Parameter | Configured Value |
| :--- | :--- |
| **Base Weights** | `yolov8n.pt` (COCO pretrained) |
| **Epochs** | {EPOCHS} |
| **Image Resolution** | {IMGSZ} × {IMGSZ} px |
| **Batch Size** | {BATCH} |
| **Optimizer** | AdamW (auto-learning rate scheduling) |
| **Compute Device** | {DEVICE.upper()} (AMD Ryzen 7 8845HS) |
| **Training Duration** | {t_duration} seconds |
| **Weight Size** | 6.2 MB |

---

## 3. Performance Metrics

### Validation Split (390 images)
- **Precision:** {val_mp:.4f}
- **Recall:** {val_mr:.4f}
- **mAP@50:** {val_map50:.4f}
- **mAP@50-95:** {val_map50_95:.4f}
- **F1 Score:** {val_f1:.4f}

### Untouched Test Split (258 images)
- **Precision:** {test_mp:.4f}
- **Recall:** {test_mr:.4f}
- **mAP@50:** {test_map50:.4f}
- **mAP@50-95:** {test_map50_95:.4f}
- **F1 Score:** {test_f1:.4f}

---

## 4. Class Mappings

| Class ID | Target Feature | Visual Indicator in Citizen Submissions |
| :--- | :--- | :--- |
| `0` | `rockfall` | Loose boulders, gravel fields, rock fragments on tarmac or berm. |
| `1` | `landslides` | Scarp fissures, mud washouts, vegetative debris, slope mass slips. |
| `2` | `tunnel` | Highway tunnel portals, retaining portals, culvert entrances. |

---

Generated by `scripts/train_citizen_vision.py`.
"""

with open(model_card_path, "w", encoding="utf-8") as f:
    f.write(model_card_md)
print(f"[SAVED] Citizen Vision Model Card: {model_card_path}")
print("\n[SUCCESS] Training, validation, test evaluation, and model card generation completed!")
