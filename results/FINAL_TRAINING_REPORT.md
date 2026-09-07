# LAND-JEPA: Final Training & Architecture Report (Phases 1–22)

**Project**: LAND-JEPA  
**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India  
**Baseline Model**: `v2.2-PREDICTION-OPTIMIZED`  
**Candidate Model**: `Two-Stage LAND-JEPA (Candidate)`  
**Operational Decision**: **RETAIN_BASELINE_V22**  

---

## 1. Executive Summary

This report documents the rigorous training cycle for the new **Two-Stage LAND-JEPA Architecture** designed specifically for genuine early-warning landslide forecasting across 8 high-risk Northeast India highway corridors.

The architecture cleanly decouples:
1. **Stage 1 (Static Susceptibility Prior)**: Computes terrain vulnerability $S(x)$ from high-resolution Copernicus 30m DEM slope, aspect, curvature, TPI, TWI, and relief.
2. **Stage 2 (Dynamic Temporal Event Risk)**: Extracts causal temporal environmental latent representations $Z_{temp}(t)$ from JEPA-TCN processing 168h sequences of rainfall, SWI, and soil moisture saturation.
3. **Multimodal Gating**: Dynamically weights susceptibility vs temporal event risk based on Numerical Weather Prediction (NWP) forecast uncertainty $U(t, H)$.

---

## 2. Master Head-to-Head Benchmark (24-Hour Horizon, FPR <= 5%)

Evaluated across 5 random seeds (42, 123, 456, 789, 1011) on unseen 2016 hold-out data:

| Architecture | 24h PR-AUC | 24h Recall @ FPR<=5% | 24h Event Recall | 24h FNR | Median Lead Time | Brier Score | False Alarms/Day |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Two-Stage LAND-JEPA (Candidate)** | **0.0610** | **19.1%** | **27.5%** | **80.9%** | **23.6h** | **0.1449** | **0.0578** |
| **v2.2-PREDICTION-OPTIMIZED (Baseline)** | 0.0644 | 28.7% | 36.7% | 71.3% | 19.9h | 0.1509 | 0.0788 |
| **Balanced Logistic Regression** | 0.1633 | 35.2% | 33.3% | 64.8% | 16.7h | 0.2144 | 0.0820 |
| **Regularized XGBoost** | 0.0343 | 25.9% | 45.6% | 74.1% | 25.0h | 0.0578 | 0.0940 |
| **JEPA-TCN** | 0.0404 | 27.8% | 43.9% | 72.2% | 22.7h | 0.0470 | 0.0870 |
| **Fused LAND-JEPA** | 0.0338 | 25.9% | 38.6% | 74.1% | 22.9h | 0.0578 | 0.0940 |
| **Supervised TCN** | 0.0325 | 24.1% | 33.3% | 75.9% | 24.3h | 0.0625 | 0.0930 |
| **Published-Methodology Threshold** | 0.0797 | 22.2% | 24.6% | 77.8% | 20.5h | 0.0126 | 0.1020 |
| **No-Forecast Persistence** | 0.0348 | 22.2% | 26.3% | 77.8% | 1.0h | 0.1267 | 0.0980 |

---

## 3. Systematic Feature & Architecture Ablations

From `results/FINAL_ABLATION.csv`:
- **Context Length**: Extending JEPA temporal receptive field from 72h to 168h increased 24h PR-AUC from 0.0578 to 0.0614 and Event Recall from 44.2% to 47.4%.
- **Multimodal Fusion**: Dynamic Gated Fusion outperformed Attention Fusion and Learned Modality Weighting by suppressing false alarms when forecast spread is elevated.
- **Two-Stage Modeling**: Decoupling static terrain susceptibility from dynamic rainfall saturation improved false alarm suppression to 0.0578 fa/day (-26.6% reduction).
- **6-Subset Hard Negatives**: Sample reweighting across extreme non-landslide rainfalls yielded over 85% false trigger suppression.
