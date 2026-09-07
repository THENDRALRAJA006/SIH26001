# VQC Data Leakage Audit

**SIH26001 / Team ZAIX**

> [!IMPORTANT]
> EXPERIMENTAL RESEARCH BRANCH — NOT FOR PRODUCTION USE

---

## Data Split Policy

| Split | Temporal Cutoff |
|---|---|
| Train | context_end < 2015-01-01 |
| Validation | 2015-01-01 ≤ context_end < 2016-01-01 |
| Test | context_end ≥ 2016-01-01 |

Source: `ml/configs/data_config.yaml` — identical to production benchmark.

---

## PCA Leakage Check

| Check | Status |
|---|---|
| PCA.fit() called on training split only | ✅ PASS |
| val/test transformed with frozen training PCA | ✅ PASS |
| Test statistics NOT used in PCA | ✅ PASS |
| PCA saved before val/test transform | ✅ PASS |

Implementation: `ml/quantum/quantum_features.py:QuantumFeatureReducer`

---

## Feature Scaler Leakage Check

| Check | Status |
|---|---|
| MinMaxScaler.fit() on training split only | ✅ PASS |
| val/test use frozen training scaler | ✅ PASS |
| Test statistics NOT used for normalization | ✅ PASS |

Implementation: `ml/quantum/quantum_features.py:QuantumFeatureScaler`

---

## Label Leakage Check

| Check | Status |
|---|---|
| LAND-JEPA model frozen during VQC experiment | ✅ PASS |
| No test labels used in PCA or scaling | ✅ PASS |
| No test labels used in threshold selection | ✅ PASS |
| Label fraction applied within training split only | ✅ PASS |
| Test split accessed exactly once (final evaluation) | ✅ PASS |

---

## Threshold Leakage Check

| Check | Status |
|---|---|
| Threshold selected on validation split only | ✅ PASS |
| Test evaluation uses frozen val threshold | ✅ PASS |
| No grid search or optimization on test set | ✅ PASS |

Implementation: `ml/quantum/vqc_trainer.py:VQCTrainer.train_vqc()`
Strategy: `select_threshold_on_val(y_val, val_probs, strategy='f1')`

---

## Temporal Leakage Check

| Check | Status |
|---|---|
| No future rainfall/weather/soil data in feature window | ✅ PASS |
| Label constructed from post-window landslide occurrence | ✅ PASS |
| Temporal ordering preserved in all splits | ✅ PASS |

Inherited from production dataset builder — same checks as main benchmark.

---

## LAND-JEPA Embedding Leakage Check

| Check | Status |
|---|---|
| Production LAND-JEPA weights frozen (no_grad=True) | ✅ PASS |
| VQC training does NOT update LAND-JEPA weights | ✅ PASS |
| Embeddings extracted with model.eval() | ✅ PASS |
| Same embedding extractor used for all splits | ✅ PASS |

---

## Audit Conclusion

**No data leakage identified.**

All PCA, scaler, and threshold statistics are derived exclusively from
the training split. The test split is accessed once, using frozen
val-selected thresholds.

The VQC experiment inherits the temporal and spatial separation from
the validated production LAND-JEPA data pipeline.
