# LAND-JEPA — AI Architecture

## Overview

The AI layer of LAND-JEPA contains three model families, a physics-aware state estimator,
and an explainability service. All model families produce landslide risk probabilities for
current, 24-hour, and 48-hour horizons.

---

## 1. XGBoost Baseline

**Purpose**: Strong classical gradient-boosting baseline for tabular features.

**Input**: Single-row tabular feature vector per zone per timestamp.

**Feature categories**:
| Group              | Features                                               |
|--------------------|--------------------------------------------------------|
| Rainfall           | acc_1h, acc_3h, acc_6h, acc_12h, acc_24h, acc_48h, acc_72h, intensity_max_1h |
| Weather            | temp_mean_24h, humidity_mean_24h, wind_speed_mean_24h  |
| Soil moisture      | sm_current, sm_anomaly_7d, sm_anomaly_30d              |
| Terrain (static)   | elevation, slope_deg, aspect_deg, curvature, tpi       |
| Historical         | susceptibility_score, prior_event_count_5km_10yr       |
| Optional InSAR     | deformation_trend_mm_30d (if available)               |

**Output**: `p_landslide` ∈ [0, 1]

**Training protocol**:
- Temporal split (no future leakage)
- Class imbalance handling via `scale_pos_weight`
- Hyperparameter tuning via cross-validation on training folds
- SHAP values for feature attribution

**Evaluation metrics**: Precision, Recall, F1, PR-AUC, FNR, Brier Score, ECE (calibration)

---

## 2. Supervised TCN Baseline

**Purpose**: Sequential model baseline that uses the full temporal context window.

**Architecture**:
```
Input tensor: (batch, time_steps, n_features)
      │
      ▼
Conv1D (causal, kernel_size=3, dilation=1)
      │
      ▼
Residual TCN Block (dilation=1)   ← uses weight normalization
      │
      ▼
Residual TCN Block (dilation=2)
      │
      ▼
Residual TCN Block (dilation=4)
      │
      ▼
Residual TCN Block (dilation=8)
      │
      ▼
Global temporal pooling (mean of last N steps)
      │
      ▼
Dropout
      │
      ▼
Linear → sigmoid → p_landslide
```

**Residual TCN Block detail**:
```
Input
  │
  ├─ Conv1D (causal, dilation=d) → WeightNorm → ReLU → Dropout
  │   → Conv1D (causal, dilation=d) → WeightNorm → ReLU → Dropout
  │
  ├─ (Residual connection: 1x1 Conv if dims differ)
  │
  └─ Add → Output
```

**Hyperparameters** (all configurable in `ml/configs/tcn_config.yaml`):
- `input_dim`: number of input features
- `hidden_dim`: TCN channel width
- `num_blocks`: number of dilation doublings
- `kernel_size`: convolution kernel size
- `dropout`: dropout rate
- `context_len`: input time-series window length (default 168 = 7 days × 24h)

---

## 3. JEPA-TCN (Core Research Contribution)

### 3.1 Conceptual Framework

JEPA (Joint Embedding Predictive Architecture) is a self-supervised learning framework
introduced by LeCun et al. (Meta AI, 2022). This project applies the JEPA principle to
**temporal environmental time series** for landslide risk pre-training.

**This project does NOT claim to have invented JEPA.**
**This project applies JEPA to the domain of geophysical time-series.**

### 3.2 Architecture

```
JEPA-TCN Pre-training

  Past Context Window              Future Target Window
  (no labels needed)               (no labels needed)
         │                                │
         ▼                                ▼
  ┌─────────────────┐             ┌───────────────────┐
  │  Context TCN    │             │   Target TCN      │
  │  Encoder        │             │   Encoder         │
  │  (trainable)    │             │   (EMA copy of    │
  └────────┬────────┘             │    Context TCN)   │
           │                     └────────┬──────────┘
           │  z_c (context latent)        │  z_t (target latent)
           ▼                             │
  ┌────────────────────┐                │
  │     Predictor      │◄───────────────┘
  │  (lightweight MLP  │    Predict z_t from z_c
  │   or Transformer)  │    (Stop-gradient on target path)
  └────────┬───────────┘
           │  ẑ_t (predicted target latent)
           ▼
   Loss = smooth_l1(ẑ_t, sg(z_t))
```

**Key design choices**:
- The **target encoder** is an **Exponential Moving Average (EMA)** of the context encoder.
  It is never directly trained via gradient descent.
- The **stop-gradient** is applied to the target latent before computing loss.
  This prevents representation collapse without requiring contrastive negatives.
- The predictor is lightweight (2–3 MLP layers). The context TCN carries the main capacity.
- The loss operates in **latent space**, not pixel/raw-observation space.
  The model learns **future environmental representations**, not exact future values.

### 3.3 JEPA Pre-training Data

- **No labels required**. Any contiguous environmental time series can be used.
- Context window = past 7 days (configurable).
- Target window = next 24–48 hours (configurable).
- Sampling: hourly.
- Augmentation: mild Gaussian noise on non-terrain inputs (configurable, off by default).

### 3.4 JEPA Downstream Fine-tuning

```
Pretrained Context TCN (weights loaded)
         │
         ▼
  Temporal Representation z_c
         │
         ├─── Terrain/Static Features (concatenated)
         │
         ├─── Soil Moisture State
         │
         └─── Optional: InSAR Deformation State
         │
         ▼
  Risk Head (MLP):
    Linear → ReLU → Dropout → Linear → Sigmoid
         │
         ├──► current risk probability
         ├──► risk_24h probability
         └──► risk_48h probability
```

**Fine-tuning modes** (selected by validation performance):
- `frozen`: context TCN weights frozen, only risk head trained
- `partial`: last N TCN blocks unfrozen
- `full`: all weights fine-tuned with lower learning rate

### 3.5 Label-Efficiency Experiment

The core research experiment tests whether JEPA pre-training improves performance
when only a small fraction of labelled events are available.

**Fractions**: 5%, 10%, 25%, 50%, 100%

**Models compared** at each fraction:
1. XGBoost (re-fitted at each fraction)
2. Supervised TCN (trained from scratch at each fraction)
3. JEPA-TCN (pre-trained on full unlabelled data; fine-tuned at each fraction)

**Controlled sampling**: stratified temporal sampling with fixed random seeds per fraction.

**Metrics collected**: Recall, FNR, PR-AUC, F1, Brier Score, ECE, Latency

**Output files**:
```
results/
  label_efficiency.csv
  label_efficiency_recall.png
  label_efficiency_pr_auc.png
  label_efficiency_f1.png
  label_efficiency_fnr.png
```

**Interpretation policy**: Report actual experiment results. Do not assume JEPA wins.

---

## 4. Physics-Aware State Estimator

**Purpose**: Provide a lightweight, interpretable soil-saturation proxy
as an additional risk feature. This is NOT a full geotechnical simulator.

**Concept**:
```
Cumulative rainfall (configurable window)
         │
         ▼
Infiltration model (simplified Green-Ampt or empirical)
         │
         ▼
Soil wetness index (SWI) ∈ [0, 1]
         │
         ▼
Pore-pressure proxy (SWI × terrain factor)
         │
         ▼
Stability indicator (SWI threshold crossings)
```

**Assumptions** (documented):
- Uniform soil properties per zone (first approximation)
- No lateral subsurface flow
- No evapotranspiration modeling (conservative, overestimates wetness)

**Removability**: This module is injected as an optional feature. The risk engine
operates without it if removed from configuration.

---

## 5. InSAR Adapter (Optional)

**Purpose**: Integrate surface-deformation signals from Sentinel-1 SAR data as a
slow-deformation state feature.

**Data characteristics**:
- InSAR observations are **not** hourly. Typical revisit: 6–12 days.
- Deformation is expressed in mm/month line-of-sight displacement.
- Ascending and descending passes may differ in availability.

**Adapter responsibilities**:
- Accept pre-processed deformation rasters (SNRI/SBAS output not produced here)
- Interpolate/extrapolate to zone boundaries
- Compute deformation trend over configurable window
- Return: `deformation_trend_mm`, `deformation_coherence`, `data_age_days`

**Missing data handling**: If deformation data is unavailable, the feature is set
to NaN and masked in downstream models. The system continues to operate.

---

## 6. Explainability Service

**Purpose**: Explain model predictions to operators and authorities.

**Method**: SHAP (SHapley Additive exPlanations)

| Model type      | SHAP method             |
|-----------------|-------------------------|
| XGBoost         | TreeExplainer           |
| TCN / JEPA-TCN  | GradientExplainer / KernelExplainer (configurable) |

**Output per prediction**:
```json
{
  "top_factors": [
    {"name": "rainfall_acc_24h", "shap_value": 0.31, "direction": "increase_risk"},
    {"name": "slope_deg",        "shap_value": 0.18, "direction": "increase_risk"},
    {"name": "sm_current",       "shap_value": 0.12, "direction": "increase_risk"}
  ],
  "disclaimer": "SHAP values represent model feature contributions, not causal effects."
}
```

**Critical disclaimer** (always displayed): SHAP values represent model-learned
feature correlations and contributions. They do NOT establish physical causation.

---

## 7. Model Configuration System

All model hyperparameters are stored in `ml/configs/`:

```
ml/configs/
  data_config.yaml      ← window sizes, horizons, feature lists
  xgboost_config.yaml   ← all XGBoost hyperparameters
  tcn_config.yaml       ← TCN architecture hyperparameters
  jepa_config.yaml      ← JEPA pre-training config (EMA decay, latent dim)
  downstream_config.yaml ← fine-tuning mode, learning rates
  risk_thresholds.yaml  ← LOW/MEDIUM/HIGH thresholds
  physics_config.yaml   ← infiltration model parameters
```

No hyperparameter is hard-coded in model source files.

---

## 8. Experiment Logging

All training runs log:
- Run ID (UUID)
- Git commit hash
- Random seed
- Configuration snapshot (full YAML copy)
- Per-epoch metrics
- Final evaluation metrics
- Checkpoint path

Storage: JSON Lines files in `results/experiment_logs/`
Format compatible with future MLflow or Weights & Biases integration.

---

## 9. Validation Utilities

JEPA pre-training includes automatic checks:
- Tensor shape assertions at each forward pass (debug mode)
- NaN detection in loss and gradients
- Gradient norm monitoring for explosion detection
- Representation collapse detection: monitors variance of z_c embeddings
  (alert if variance collapses below configurable threshold)

These checks run in training mode and are disabled in production inference.
