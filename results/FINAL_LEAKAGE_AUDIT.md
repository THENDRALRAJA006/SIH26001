# LAND-JEPA Final Leakage Audit
## SIH26001 — Team ZAIX — September 2026

> [!IMPORTANT]
> This is a formal scientific leakage audit for the LAND-JEPA benchmark pipeline.
> Each item is verified against source code, with file references.

---

## 1. Temporal Future Leakage

### 1.1 Weather / Rainfall / Soil Moisture Leakage
**Status: CLEAN ✅**

- Context window: `[t - 168h, t]` (past 7 days of observations)
- Target horizon: `[t, t + 24h]` (future 24 hours)
- All feature extraction in [`ml/features/window_generator.py`](file:///d:/SIH26001/ml/features/window_generator.py) is strictly causal
- `WindowGenerator.generate_arrays()` uses only the context slice `[context_start:context_end]`
- No rolling features span into the target window
- ERA5-Land data retrieval is historical (not forecast data)

### 1.2 Rainfall Accumulation Feature Leakage
**Status: CLEAN ✅**

Features `acc_1h`, `acc_3h`, `acc_6h`, `acc_12h`, `acc_24h`, `acc_48h`, `acc_72h` are computed in [`ml/ingestion/real/rainfall_openmeteo.py`](file:///d:/SIH26001/ml/ingestion/real/rainfall_openmeteo.py) using backward-looking rolling windows (`df.rolling(window, min_periods=1).sum()`). The context window is extracted after this computation, so no future rainfall enters the context.

### 1.3 SWI / Physics State Leakage  
**Status: CLEAN ✅**

[`ml/features/physics_state.py`](file:///d:/SIH26001/ml/features/physics_state.py) computes SWI as an exponentially-weighted moving average of past precipitation only. Pore-pressure proxy and stability indicator depend on SWI and current soil moisture. All operations are backward-looking.

---

## 2. Label Construction Leakage

### 2.1 Post-Event Information in Context Window
**Status: CLEAN ✅**

[`ml/features/label_builder.py`](file:///d:/SIH26001/ml/features/label_builder.py) enforces a strict 72h pre-event exclusion buffer:
- Windows where `context_end` falls within `[event_time - 72h, event_time]` are labeled `y = -1` (excluded)
- This ensures no context window "peeks ahead" at an imminent landslide
- `negative_buffer_hours = 72` in `LabelConfig`

### 2.2 Event Split Across Train/Test Boundaries  
**Status: CLEAN ✅**

The temporal split is determined by `context_end` timestamp:
- `test_mask`: `context_end >= 2016-01-01 UTC`
- `val_mask`: `context_end >= 2015-01-01 UTC` AND `context_end < 2016-01-01 UTC`
- `train_mask`: `context_end < 2015-01-01 UTC`

Since events occur in 2011–2016, a 2016 event's label is ONLY in the test split. The same event cannot appear as both train and test label.

### 2.3 Duplicate Event Leakage  
**Status: CLEAN ✅**

[`ml/ingestion/real/landslide_glc.py`](file:///d:/SIH26001/ml/ingestion/real/landslide_glc.py) deduplicates events by `(date, zone_id, lat_round, lon_round)` before storing. No duplicate events contaminate the training signal.

---

## 3. Normalization Leakage

### 3.1 Train-Only Normalization
**Status: CLEAN ✅**

In [`scripts/run_final_benchmark.py`](file:///d:/SIH26001/scripts/run_final_benchmark.py):
```python
tnorm = TemporalNormalizer()
tr_seq = tnorm.fit_transform(train.X_sequence)   # fit ONLY on train
va_seq = tnorm.transform(val.X_sequence)          # transform with train statistics
te_seq = tnorm.transform(test.X_sequence)         # transform with train statistics
```
The `FeatureNormalizer` for XGBoost is similarly fit only on training data before applying to val/test.

### 3.2 Global Statistics Leakage
**Status: CLEAN ✅**

No global normalization (mean/std computed across all data) is applied. [`ml/preprocessing/normalizers.py`](file:///d:/SIH26001/ml/preprocessing/normalizers.py) explicitly requires `fit()` to be called on training data only.

---

## 4. Self-Supervised Pretraining Leakage

### 4.1 JEPA Uses No Landslide Labels
**Status: CLEAN ✅**

In `pretrain_jepa()` in [`scripts/run_final_benchmark.py`](file:///d:/SIH26001/scripts/run_final_benchmark.py):
- JEPA pretraining filters to `merged_df[merged_df["observed_at"] < 2015-01-01]` (training period only)
- The pretraining objective is purely self-supervised: predicting future latent representations from context representations
- No `events_df` is loaded or referenced during pretraining
- No `y` labels are accessible in the pretraining code path

### 4.2 JEPA Pretraining Data Boundary
**Status: CLEAN ✅**

Pretraining uses only `observed_at < 2015-01-01 UTC`, strictly within the training period. No validation or test sequences (2015–2016) are used for pretraining.

---

## 5. Threshold Selection Leakage

### 5.1 Threshold Selected on Validation Set Only
**Status: CLEAN ✅**

In [`ml/evaluation/metrics.py`](file:///d:/SIH26001/ml/evaluation/metrics.py), `select_threshold_on_val(y_val, val_probs)` maximizes F1 score using only validation predictions. The test set is never touched during threshold selection.

### 5.2 Identical Threshold Procedure for All Models
**Status: CLEAN ✅**

All 4 models (XGBoost, Supervised TCN, JEPA-TCN, Fused LAND-JEPA) use the same `select_threshold_on_val()` procedure. No model receives a manually tuned threshold.

---

## 6. Test Set Protection

### 6.1 Test Set Untouched Until Final Evaluation
**Status: CLEAN ✅**

The `test` split object is never passed to `fit()`, `fit_transform()`, or `select_threshold_on_val()`. Test data only enters at the final `compute_metrics(test.y, test_probs, threshold)` call.

### 6.2 Bootstrap CI on Test Set (Post-Evaluation)
**Status: CLEAN ✅**

1,000-resample bootstrap confidence intervals are computed on the 2016 held-out test predictions. No model re-training or threshold re-selection occurs during bootstrap. Bootstrap only resamples the fixed test set predictions.

---

## 7. Data Mode Isolation

### 7.1 Demo Data Isolation
**Status: CLEAN ✅**

- Real providers reside in `ml/ingestion/real/` — all have `is_demo = False`
- Demo providers reside in `ml/ingestion/demo/` — never imported by real benchmark scripts
- `load_final_real_data()` asserts `is_demo` is False before proceeding
- `run_final_benchmark.py` does not import any `demo/` module

### 7.2 Demo Mode Flag Propagation
**Status: CLEAN ✅**

Every `ProviderMetadata` object carries `is_demo = False`. The `RiskPredictionOutput` dataclass exposes `is_demo` in API responses, enabling the frontend to label DEMO data clearly in production.

---

## 8. Spatial Leakage (Leave-One-Zone-Out Experiment)

### 8.1 Spatial Generalization Setup
**Status: CLEAN ✅**

In the leave-one-zone-out experiment (Phase 13):
- Training uses only the 7 non-held-out zones
- Testing uses only the held-out zone
- No zone's data appears in both train and test
- Normalization fit on the 7 training zones only

---

## Summary

| Leakage Category | Status |
|---|---|
| Future weather in context | ✅ CLEAN |
| Future rainfall in context | ✅ CLEAN |
| Future soil moisture in context | ✅ CLEAN |
| Post-event context information | ✅ CLEAN (72h exclusion buffer) |
| Event split across train/test | ✅ CLEAN (strict temporal cut at 2016-01-01) |
| Duplicate event contamination | ✅ CLEAN (deduplication in GLC ingestion) |
| Train-only normalization | ✅ CLEAN |
| JEPA pretraining sees labels | ✅ CLEAN (zero labels used) |
| JEPA pretraining sees future data | ✅ CLEAN (train period < 2015 only) |
| Threshold selection on test | ✅ CLEAN (val only) |
| Same threshold procedure for all models | ✅ CLEAN |
| Test set protection | ✅ CLEAN |
| Demo/synthetic data contamination | ✅ CLEAN |
| Spatial leakage (LOZO) | ✅ CLEAN |

**Verdict: No data leakage identified in the LAND-JEPA scientific benchmark pipeline.**
