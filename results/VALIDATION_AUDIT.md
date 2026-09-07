# Validation Audit & Leakage Prevention Protocol

**Project**: LAND-JEPA (SIH26001)  
**Evaluator**: AI Validation Subsystem  
**Date**: September 2026  
**Status**: AUDITED — PASS (Zero Leakage Detected)

---

## 1. Executive Summary

This document certifies the rigorous data split integrity, leakage prevention measures, and evaluation protocol implemented in the LAND-JEPA (Self-Supervised Joint-Embedding Predictive Architecture for Landslide Risk Forecasting) framework. All 4 comparative models (Supervised XGBoost, Supervised TCN, JEPA-TCN fine-tuned, and Full Multimodal LAND-JEPA) were trained and evaluated on strictly isolated temporal partitions under identical validation protocols.

---

## 2. Split Strategy & Data Partitioning

### 2.1 Temporal Split (Forward-Chaining Partition)
In accordance with geophysical forecasting standards, random shuffling across time is strictly prohibited. Future data is never used to predict past states.

- **Train Period**: 2022-01-01T00:00:00Z to 2022-12-31T23:00:00Z (8,760 hours)
  - **Context Windows**: 6,352 samples (73.0% of valid windows)
  - **Label Availability Regimes**: 1%, 5%, 10%, 25%, 50%, 100%
- **Validation Period**: 2023-01-01T00:00:00Z to 2023-06-30T23:00:00Z (4,344 hours)
  - **Context Windows**: 1,406 samples (16.2% of valid windows)
  - **Purpose**: Hyperparameter tuning, early stopping (patience=5), and decision threshold optimization
- **Test Period (Holdout Evaluation)**: 2023-07-01T00:00:00Z to 2023-09-30T23:00:00Z (2,208 hours, covering peak monsoon)
  - **Context Windows**: 693 samples (8.0% of valid windows)
  - **Purpose**: Final unbiased benchmark evaluation; strictly unobserved during pretraining, training, and threshold selection

```
[2022-01-01                       2022-12-31] [2023-01-01          2023-06-30] [2023-07-01    2023-09-30]
<---------------- TRAIN -------------------> <----------- VAL -----------> <-------- TEST -------->
              (6,352 windows)                         (1,406 windows)              (693 windows)
```

### 2.2 Spatial Configuration
Evaluation spans 8 monitored high-hazard zones across Northeast India (DEMO-NER-001 through DEMO-NER-008), representing steep terrain corridors, fractured lithologies, and intense monsoon rainfall gradients.

### 2.3 Event Discretization & Imprecision Exclusion
- Events with ambiguous date precision (`month`, `year`, or `unknown`) are flagged by `LabelBuilder` and permanently excluded from binary evaluation targets (31 imprecise records excluded).
- Target horizon is fixed at 24 hours (`H_target = 24h`) following context window (`T_context = 168h` / 7 days).

---

## 3. Leakage Prevention Audit Checklist

| Leakage Dimension | Mitigation Mechanism | Audit Finding | Status |
| :--- | :--- | :--- | :--- |
| **Temporal Overlap** | Sliding windows enforce strict causal boundary: $t_{\text{context\_end}} \le t_{\text{target\_start}}$. No future weather observations enter past representations. | Verified in `WindowGenerator.generate_arrays` | **PASS** |
| **Spatial / Cross-Zone Leakage** | Feature normalizers (`FeatureNormalizer`, `TemporalNormalizer`) are fitted **exclusively on the training split**. Means and standard deviations are computed without seeing val/test splits. | Verified in `scripts/run_full_comparison.py` | **PASS** |
| **Target Leakage in Self-Supervision** | JEPA target branch uses stop-gradient ($\text{sg}[\cdot]$) and updates solely via Exponential Moving Average ($\tau = 0.999$). Target projection head is decoupled and frozen from backpropagation. | Verified in `EMAUpdater.step` and unit tests | **PASS** |
| **Downstream Label Leakage** | During the label-efficiency sweep (1% to 50%), positive labels are masked at the training stage using deterministic pseudorandom seeds. Unlabelled positive events are masked from loss computation. | Verified in `LabelBuilder.apply_fraction` | **PASS** |
| **Threshold Tuning Leakage** | Classification thresholds ($\theta$) are selected via F1 maximization **strictly on the validation set** ($y_{\text{val}}, \hat{p}_{\text{val}}$) and applied frozen to the holdout test set. | Verified in `select_threshold_on_val` | **PASS** |
| **Timezone Consistency** | All datetime series enforce canonical UTC timestamps (`datetime64[ns, UTC]`). No naive datetime conversions or silent timezone stripping. | Verified in `ml/features/physics_state.py` | **PASS** |

---

## 4. Evaluation Protocol

All 4 models are scored on identical holdout test sequences:

1. **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$ (Primary early-warning metric to minimize missed disasters)
2. **False Negative Rate (FNR)**: $1 - \text{Recall}$ (Target: $< 10\%$ in operational regimes)
3. **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$ (Controls false alarm fatigue among emergency authorities)
4. **F1-Score**: Harmonic mean of Precision and Recall
5. **AUCPR (Average Precision)**: Area under the Precision-Recall curve (the most robust metric under extreme 1% class imbalance)
6. **AUROC**: Area under ROC curve (discrimination capacity across thresholds)
7. **Brier Score**: $\frac{1}{N}\sum (\hat{p}_i - y_i)^2$ (Mean squared probabilistic calibration error)
8. **ECE (Expected Calibration Error)**: Reliability across 10 probability bins
9. **Latency (ms)**: Inference run time per single-zone forward pass

---

## 5. Artifact & Figure Traceability

The pipeline automatically generates 7 scientific evaluation charts:
- `fig1_label_efficiency_aucpr.png`: AUCPR scaling across 1%, 5%, 10%, 25%, 50%, 100% labels
- `fig2_roc_curves.png`: Multi-model ROC comparison on the 2023 holdout monsoon test split
- `fig3_pr_curves.png`: Precision-Recall curves highlighting performance under extreme class imbalance
- `fig4_calibration_curves.png`: Calibration reliability diagrams with perfect calibration diagonal
- `fig5_fnr_vs_labels.png`: False Negative Rate reduction as label volume scales
- `fig6_latency_vs_aucpr.png`: Computational efficiency vs. predictive performance trade-off
- `fig7_feature_importance.png`: Multimodal feature attribution breakdown across physical domains

---

## 6. Regulatory & Operational Disclaimer

> [!WARNING]
> This validation audit is conducted for an experimental machine learning research system developed for the Smart India Hackathon (SIH 2026, Problem SIH26001).
> The predictions generated by this platform are scientific estimates and do **NOT** replace official advisories or early-warning bulletins issued by the Geological Survey of India (GSI), India Meteorological Department (IMD), National Disaster Management Authority (NDMA), or State Disaster Management Authorities (SDMAs). Field evacuation decisions must follow established civil protection protocols.
