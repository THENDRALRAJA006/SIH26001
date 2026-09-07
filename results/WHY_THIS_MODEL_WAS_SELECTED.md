# WHY THIS MODEL WAS SELECTED AS THE PRODUCTION SYSTEM
## LAND-JEPA (Fused LAND-JEPA) — SIH26001 / Team ZAIX

---

## 1. Model Selection Criteria

Production model selection was made on the following criteria **in strict priority order**:

| Priority | Criterion | Rationale |
|---|---|---|
| 1 | **PR-AUC** | Primary discrimination metric under class imbalance (<1% positive rate) |
| 2 | **Recall at ≤5% FPR** | Safety constraint: missed landslides (FN) cost lives; false alarms erode trust |
| 3 | **FNR (False Negative Rate)** | Directly measures missed disaster events |
| 4 | **Label efficiency** | Performance at low label fractions tests representation quality |
| 5 | **Brier score** | Probabilistic calibration; essential for threshold-based operational alerts |
| 6 | **Inference latency** | Must support real-time alerts; target <10 ms per zone |

---

## 2. Model Comparison Summary

Four models were evaluated against the 2016 real holdout test (18 positive windows from 23 NASA GLC events):

| Model | Architecture | Pretraining | Terrain | Physics |
|---|---|---|---|---|
| **XGBoost** | Gradient Boosted Trees | Supervised | ✅ Tabular | ✅ Tabular |
| **Supervised TCN** | Dilated Causal TCN | Supervised (scratch) | ❌ No | ❌ No |
| **JEPA-TCN** | TCN + JEPA Encoder | Self-supervised | ❌ No | ❌ No |
| **Fused LAND-JEPA** | TCN + Fusion + Terrain + Physics | Self-supervised | ✅ Latent | ✅ Latent |

> Results at 100% labels across seeds 42, 123, 456 — see `results/final_model_comparison.csv` for exact numbers.

---

## 3. Selection Decision

**Selected Model: Fused LAND-JEPA**

### 3.1 Primary Technical Justification

**Self-supervised representation learning is uniquely suited to this problem:**
- Landslide labeling in NER is sparse: only 177 confirmed events over 5 years across 8 zones
- ERA5-Land provides 406,080 hours of continuous environmental observations
- JEPA pretraining exploits ALL this unlabeled data to learn atmospheric dynamics — without a single landslide label

**Why Fused > JEPA-TCN alone:**
- Fused LAND-JEPA additionally conditions predictions on verified Copernicus DEM-30 terrain features
- Slope gradient (22°–30° in NER ridgelines) is a critical geomechanical predictor of slope instability
- Physics-aware states (SWI, pore-pressure proxy, stability indicator) provide interpretable hydrological context

**Why Fused > Supervised TCN:**
- Supervised TCN trains only on 76 labeled windows — insufficient to generalize
- JEPA pretraining provides a rich prior that transfers to sparse labeled regimes

**Why Fused > XGBoost:**
- XGBoost cannot model temporal dependencies (ordering of rainfall accumulations over 168h)
- XGBoost degrades significantly at <10% label fractions
- JEPA-TCN and Fused LAND-JEPA consistently outperform at low label fractions

### 3.2 Label Efficiency Advantage

JEPA-TCN's principal scientific contribution is demonstrated in the **label efficiency curves**:
- At 10% labels (~7 positive training events), JEPA-TCN maintains higher PR-AUC than Supervised TCN
- At 1% labels (~1 positive training event), supervised models collapse to near-random while JEPA-TCN retains meaningful signal
- This advantage directly addresses the operational reality that NER landslide inventories are incomplete

### 3.3 Operational Constraints Met

| Constraint | Status |
|---|---|
| Inference latency <10ms per zone | ✅ Measured <1ms on CPU |
| Supports multi-horizon forecasting (0h, 24h, 48h) | ✅ Three independent heads |
| Calibrated probability output | ✅ ECE < 0.05 verified |
| Missing InSAR handled gracefully | ✅ `missing_token` engaged automatically |
| Interpretable leading factors | ✅ Rainfall, SWI, slope attribution |
| Demo mode flag in API response | ✅ `is_demo: false` in production output |

### 3.4 InSAR Caveat

InSAR is currently `UNAVAILABLE` due to C-band vegetation decorrelation in sub-tropical NER terrain. The Fused LAND-JEPA architecture is InSAR-ready: when coherent interferograms become available (e.g. via LiCSAR processing or L-band NISAR), the `InSARDeformationEncoder` will engage with zero architectural changes. The `missing_token` ensures the model degrades gracefully in the current InSAR-absent state.

---

## 4. Threshold Selection Protocol

Operating threshold is selected using the FPR-constrained protocol:

```
For each model, on the validation split (2015):
  1. Sweep thresholds from 0.05 to 0.95 in steps of 0.005
  2. For each threshold t:
       a. Compute preds = (val_probs >= t)
       b. Compute FPR = FP / (TN + FP)
       c. If FPR <= 0.05: record Recall
  3. Select the threshold maximising Recall subject to FPR <= 0.05
  4. Apply this threshold to test set (never re-fitted on test)
```

This is the **same protocol for all four models** — no model is given an advantage through manual threshold tuning.

---

## 5. Comparison to Alternatives Considered

| Alternative | Rejected Because |
|---|---|
| Transformer (Attention-based) | Requires significantly more data; causal masking complexity; not shown to outperform TCN on short (<200h) sequences |
| LSTM/GRU | Vanishing gradients at 168h sequences; no dilated reception field; slower than TCN |
| Graph Neural Network | Requires verified spatial adjacency graph between NER zones; insufficient edge data |
| Reconstruction-based SSL (BERT-style masking) | Pixel-space reconstruction is unnecessary for latent-space downstream; JEPA avoids redundant low-level feature prediction |
| Contrastive SSL (SimCLR) | Requires strong data augmentation; environmental time series augmentations are non-trivial and can distort meteorological signals |
| Random Forest | No temporal modeling; similar degradation as XGBoost in sparse-label regimes |

---

## 6. Production Deployment Decision

**Fused LAND-JEPA is the LAND-JEPA production model.**

Weights are stored at: `ml/checkpoints/land_jepa_production/land_jepa_weights.pt`

The production inference class is [`LandJEPARiskModel`](file:///d:/SIH26001/ml/models/land_jepa_model.py) which exposes:
- `forward()` — raw logits for training/calibration
- `predict_risk()` — calibrated probabilities, risk levels, confidence, leading factors

The FastAPI backend exposes this via:
- `GET /api/v1/risk/{zone_id}` — current risk prediction
- `POST /api/v1/risk/predict` — on-demand inference with custom input
- `GET /api/v1/model/status` — model health and version metadata

**Version: LAND-JEPA-v1.0.0**
