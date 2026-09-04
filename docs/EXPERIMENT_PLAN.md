# LAND-JEPA — ML Experiment Plan

## Research Question

> "Can JEPA-style self-supervised temporal representation learning with a TCN encoder
> improve landslide-risk prediction when labelled landslide events are scarce?"

---

## Experimental Setup

### Data

| Split        | Composition                              | Leakage control                    |
|--------------|------------------------------------------|------------------------------------|
| Pre-train    | Unlabelled environmental time series     | No labels used                     |
| Train        | Labelled fraction of training events     | Temporal split: no future leakage  |
| Validation   | Held-out labelled events                 | Not used during training           |
| Test         | Final held-out set, never seen           | Used ONCE for final evaluation     |

**Temporal split strategy**:
- All events before a cutoff date → Train/Validation
- All events after cutoff → Test
- NO random shuffling across the temporal boundary

**Label-efficiency fractions**: 5%, 10%, 25%, 50%, 100% of training labels.
- Stratified sampling: each fraction preserves the class imbalance ratio.
- 5 fixed random seeds per fraction for variance estimation.
- XGBoost and TCN re-trained from scratch at each fraction.
- JEPA-TCN: pre-training uses full unlabelled data; only fine-tuning fraction varies.

---

## Models

### Experiment 1: XGBoost Baseline

**Configuration file**: `ml/configs/xgboost_config.yaml`

| Hyperparameter           | Search range / default     |
|--------------------------|----------------------------|
| n_estimators             | 100–1000 (early stopping)  |
| max_depth                | 3–8                        |
| learning_rate            | 0.01–0.3                   |
| subsample                | 0.6–1.0                    |
| colsample_bytree         | 0.6–1.0                    |
| scale_pos_weight         | n_neg / n_pos (imbalance)  |

**Feature set** (tabular):
- Rainfall accumulations: 1h, 3h, 6h, 12h, 24h, 48h, 72h
- Max hourly intensity (24h window)
- Temperature mean (24h)
- Humidity mean (24h)
- Wind speed mean (24h)
- Soil moisture current, 7d anomaly, 30d anomaly
- Elevation, slope, aspect, curvature, TPI, TWI
- Historical susceptibility score
- Event count within 5km in last 10 years (spatial indicator)
- Physics proxy: SWI (soil wetness index)
- Optional: InSAR deformation trend

**Validation**: 5-fold TimeSeriesSplit (time-aware cross-validation)

**Output**: `ml/checkpoints/xgboost/`

---

### Experiment 2: Supervised TCN

**Configuration file**: `ml/configs/tcn_config.yaml`

| Hyperparameter   | Default  | Notes                            |
|------------------|----------|----------------------------------|
| context_len      | 168      | 7 days × 24h                     |
| hidden_dim       | 64       | TCN channel width                |
| num_blocks       | 4        | Dilation: 1, 2, 4, 8            |
| kernel_size      | 3        |                                  |
| dropout          | 0.1      |                                  |
| lr               | 1e-3     | Adam optimizer                   |
| batch_size       | 64       |                                  |
| epochs           | 100      | With early stopping (patience 10)|
| loss             | BCEWithLogitsLoss | With pos_weight for imbalance |

**Input features**: Same as XGBoost but structured as time series
(temporal dimension: hours; feature dimension: environmental variables).

**Output**: `ml/checkpoints/tcn_supervised/`

---

### Experiment 3: JEPA Pre-training

**Configuration file**: `ml/configs/jepa_config.yaml`

| Parameter          | Default  | Notes                                    |
|--------------------|----------|------------------------------------------|
| context_len        | 168      | Past 7 days (hourly)                     |
| target_len         | 24       | Future 24h window                        |
| latent_dim         | 128      | Embedding dimension                      |
| ema_decay          | 0.999    | Target encoder EMA decay rate            |
| predictor_hidden   | 256      | Predictor MLP hidden size                |
| predictor_layers   | 3        | Predictor MLP depth                      |
| lr                 | 3e-4     | Adam optimizer                           |
| batch_size         | 128      |                                          |
| epochs             | 200      | With LR warmup (10 epochs)               |
| loss               | SmoothL1 | Between predicted and target latent      |
| augmentation       | disabled | Gaussian noise optional                  |

**Collapse detection**: Monitor variance of z_c embeddings.
Alert if variance < threshold (configurable `collapse_variance_threshold`).

**Output**: `ml/checkpoints/jepa_pretrained/`

---

### Experiment 4: JEPA-TCN Downstream

**Configuration file**: `ml/configs/downstream_config.yaml`

| Parameter          | Options           | Default  |
|--------------------|-------------------|----------|
| encoder_mode       | frozen/partial/full | partial |
| unfreeze_blocks    | 1–num_blocks      | 2        |
| risk_head_hidden   | 64–256            | 128      |
| dropout            | 0.1–0.3           | 0.2      |
| lr_encoder         | 1e-5–1e-3         | 1e-4     |
| lr_head            | 1e-4–1e-3         | 5e-4     |
| epochs             | 50                | With early stopping |

---

## Metrics

All models evaluated with identical test protocol on the same held-out test set.

| Metric                   | Rationale                                                |
|--------------------------|----------------------------------------------------------|
| Recall (sensitivity)     | Primary metric — missing a landslide is dangerous        |
| False Negative Rate (FNR)| 1 - Recall; must be explicitly reported                  |
| Precision                |                                                          |
| F1                       | Harmonic mean of precision and recall                   |
| PR-AUC                   | Preferred over ROC-AUC for imbalanced classes            |
| Brier Score              | Calibration of probability estimates                     |
| ECE                      | Expected Calibration Error                               |
| Inference Latency (ms)   | Per-zone prediction time on test hardware                |

**Operating threshold**: Selected to maximize recall while maintaining precision ≥ 0.3
(configurable; not optimized post-hoc on test set).

---

## Outputs and Artifacts

```
results/
  label_efficiency.csv            ← all metrics × all models × all fractions
  label_efficiency_recall.png     ← recall vs label fraction
  label_efficiency_pr_auc.png     ← PR-AUC vs label fraction
  label_efficiency_f1.png         ← F1 vs label fraction
  label_efficiency_fnr.png        ← FNR vs label fraction
  experiment_logs/
    <run_id>.json                 ← per-run metrics and config snapshot
  plots/
    calibration_curves.png
    shap_summary_xgboost.png
    shap_summary_jepa_tcn.png
```

---

## Reproducibility Requirements

1. All runs must log:
   - Git commit hash at time of run
   - Configuration snapshot (full YAML)
   - Random seed(s) used
   - Package versions (`requirements.txt` snapshot)

2. Training script: `python ml/baselines/xgboost/train_xgboost.py --config ml/configs/xgboost_config.yaml --seed 42`

3. Validation utilities run before each training:
   - Shape checks on all tensors
   - NaN detection in input features
   - Label distribution check
   - No-leakage check (temporal ordering asserted)

4. Results in `label_efficiency.csv` are the source of truth for the research report.

---

## Interpretation Policy

- Report actual experimental results.
- If JEPA-TCN does NOT outperform supervised TCN at any fraction, report that honestly.
- Possible explanations for negative or neutral results are documented.
- No cherry-picked thresholds on the test set.
- Results section of research report includes 95% confidence intervals where sample size permits.

---

## Assumptions and Limitations

1. **Label uncertainty**: Historical landslide events with `date_precision != 'exact'` are
   handled conservatively. Label uncertainty is documented in results.

2. **Class imbalance**: Landslide events are rare. Heavy imbalance compensation required.
   The label-efficiency experiment uses stratified sampling to preserve class ratio.

3. **Spatial autocorrelation**: Nearby zones may have correlated inputs. This is not
   fully controlled for in the initial experiment. Documented as a limitation.

4. **Generalization**: The model is trained and evaluated on NER data (real or demo).
   Generalization to other regions is not tested in this hackathon scope.

5. **Demo data limitation**: If real event records are unavailable, experiments run on
   synthetic demo data. Results from demo data are explicitly marked as
   "SOFTWARE INTEGRATION TEST — NOT SCIENTIFIC RESULTS" in the output.
