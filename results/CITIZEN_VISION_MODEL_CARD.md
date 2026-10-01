# LAND-JEPA Citizen Vision Verification — Model Card

**Model Version:** `citizen-vision-v1`  
**Base Architecture:** YOLOv8n Feature Pyramid Object Detector (3.01M parameters, 8.1 GFLOPs)  
**Training Checkpoint ID:** `TRAIN-CV-1788903198`  
**Trained Weights Path:** `ml/checkpoints/citizen_vision/best.pt` (23.8 MB)  
**Dataset Lineage:** `ul://thendralraja-mj/datasets/rockfall` (1,964 images, 5,236 annotations)  
**Deployment Context:** Northeast India National Highway Corridors (NH-27, NH-6, NH-29, NH-102, SH-4)  
**Date:** September 2026 · Team ZAIX · SIH26001  

---

## 1. Operational Guardrails & Human-in-the-Loop Policy

1. **Evidence Verification, NOT Reporter Fraud:**
   The model evaluates visible physical hazard evidence in submitted imagery (loose boulders, scarp ruptures, slope slips, and tunnel portals). It **never** labels a human citizen reporter as "fraudulent", "fake", or "dishonest".
2. **EXIF Policy:**
   Missing EXIF data is cataloged as `EXIF_UNAVAILABLE` (privacy-preserved), because consumer messaging and social platforms routinely strip EXIF. Missing EXIF is **not** grounds for rejection.
3. **Decoupled System Architecture:**
   Verified visual evidence assists officer triage and creates operational field log features. It **does NOT** silently or directly overwrite LAND-JEPA neural geo-temporal risk probabilities. All critical emergency dispatches remain strictly under human officer authority.

---

## 2. Performance Summary

| Metric | Validation Split (390 images) | Untouched Test Split (258 images) |
| :--- | :--- | :--- |
| **Precision** | **0.5437** | **0.5439** |
| **Recall** | **0.4090** | **0.4321** |
| **mAP@50** | **0.4176** | **0.4357** |
| **mAP@50-95** | **0.1810** | **0.2040** |
| **F1 Score** | **0.4667** | **0.4816** |
| **CPU Inference Latency** | **10.0 ms / image** | **8.3 ms / image** |

### Per-Class Performance on Untouched Test Set
- **`rockfall` (136 images, 488 instances):** Precision: **70.8%**, Recall: **32.8%**, mAP@50: **0.480**
- **`landslides` (205 images, 222 instances):** Precision: **38.0%**, Recall: **53.6%**, mAP@50: **0.392**
- **`tunnel` (infrastructure reference):** Contextual feature for retaining structures and portal entries.

---

## 3. Verification Score Formulation

The automated Evidence Strength Score $S \in [0, 1]$ is computed as:
$$S = 0.60 \cdot \text{conf}_{\text{vision}} + 0.15 \cdot \min(1.5 \cdot A_{\text{hazard}}, 1.0) + S_{\text{spatial}} - P_{\text{mismatch}} - P_{\text{duplicate}}$$
Where:
- $\text{conf}_{\text{vision}}$: Maximum detector confidence across hazard classes.
- $A_{\text{hazard}}$: Normalized bounding box area ratio.
- $S_{\text{spatial}}$: Up to +0.25 based on corridor proximity and EXIF match.
- $P_{\text{mismatch}}$: 0.25 deduction if EXIF explicitly contradicts reported location by >25 km.
- $P_{\text{duplicate}}$: Duplicates capped at 0.10.

---

## 4. Operational Status Taxonomy

| Status Category | Criteria | Operational Routing |
| :--- | :--- | :--- |
| `STRONG_EVIDENCE` | Evidence Score $\ge 0.70$ and hazard confirmed | Priority officer review; highlight on live GIS |
| `MODERATE_EVIDENCE` | Evidence Score $\ge 0.45$ and hazard detected | Standard officer queue |
| `NEEDS_REVIEW` | Low confidence or unconfirmed visual boundary | Awaiting field dispatch inspection |
| `LOCATION_MISMATCH` | EXIF GPS differs from reported GPS by > 15 km | Flagged for manual officer coordinates check |
| `DUPLICATE` | Identical SHA-256 or dHash Hamming distance $\le 4$ | Linked to original incident record |
| `NO_HAZARD_DETECTED` | Photo clear but zero landslide/rockfall detected | Officer visual inspection |
