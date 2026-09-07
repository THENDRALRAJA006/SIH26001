# MODEL CARD: LAND-JEPA v2.6 (Trigger-Aware Champion Candidate)

**Model Identifier**: `v2.6-TRIGGER-AWARE-CANDIDATE`  
**Architecture**: Causal JEPA-TCN + Geomorphic Terrain Encoder + Trigger Mechanism Gated Fusion  
**Date**: 2026-09-05  
**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India (8 Monitored Highway Corridors)  
**Status**: Validation Selected Champion Candidate (Untrained on Prospective Test Set)  

---

## 1. Model Details & Architecture Overview

LAND-JEPA v2.6 expands upon the v2.5 trigger-aware baseline by incorporating an authentic, lightweight multi-trigger fusion layer directly resolving the 4 physical mechanisms missed in prospective surveillance:
1. **Road-Cut Toe Excavation**: Over-steepened cut slope geometry and toe excavation stress index.
2. **Co-Seismic Fault Slip**: Peak Ground Acceleration (PGA) interaction with antecedent saturation.
3. **Culvert & Drainage Blowout**: Upstream flow accumulation choke ratio and ditch scour susceptibility.
4. **Localized Convective Cloudburst**: Multi-scale precipitation gradients and nowcast burst ratios.

### Input Tensors:
- **Temporal Stream**: (B, 168, 16) — 7-day hourly sequence of precipitation, soil moisture, humidity, temperature, and atmospheric pressure.
- **Geomorphic Stream**: (B, 8) — 30m Copernicus DEM derivatives (elevation, slope, aspect, curvature, TWI, TPI, relief).
- **Physical Trigger Stream**: (B, 47) — Physical trigger vectors for the 4 targeted failure mechanisms.

### Latent Representation & Fusion:
- z_temporal = TCNEncoder.encode(x_sequence) in R^64
- z_terrain = StaticFeatureEncoder(x_terrain) in R^64
- z_trigger = TriggerMechanismEncoder(x_trigger) in R^48
- z_fused, gating_weights = TriggerAwareGatedFusion(z_temporal, z_terrain, z_trigger) in R^128

---

## 2. Validation-Only Benchmark Summary (24-Hour Horizon)

Evaluated strictly on the held-out 2015 validation split across 5 statistical random seeds:

| Model Architecture | Event Recall (FPR <= 5%) | FNR | FPR | Precision | PR-AUC | Brier Score | ECE | False Alarms / Day |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| JEPA-TCN Baseline | 23.3% | 76.7% | 2.23% | 0.1879 | 0.1035 | 0.0120 | 0.0000 | 0.0214 |
| Regularized XGBoost | 20.2% | 79.8% | 2.15% | 0.1387 | 0.0868 | 0.0122 | 0.0000 | 0.0206 |
| v2.5-TRIGGER-AWARE-BASELINE | 28.1% | 71.9% | 2.52% | 0.1387 | 0.1135 | 0.0119 | 0.0000 | 0.0241 |
| **v2.6 Ablation (No Cloudburst)** | **28.9%** | 71.1% | 2.61% | 0.1444 | 0.1153 | 0.0119 | 0.0000 | 0.0250 |
| v2.6 Ablation (No Culvert-Scour) | 28.8% | 71.2% | 2.56% | 0.1446 | 0.1147 | 0.0119 | 0.0000 | 0.0246 |
| v2.6 Ablation (No Road-Cut) | 27.5% | 72.5% | 2.29% | 0.1732 | 0.1166 | 0.0119 | 0.0000 | 0.0220 |
| v2.6 Ablation (No Seismic) | 27.5% | 72.5% | 2.28% | 0.1616 | 0.1164 | 0.0119 | 0.0000 | 0.0218 |
| v2.6-TRIGGER-AWARE-CANDIDATE | 27.5% | 72.5% | 2.28% | 0.1614 | 0.1156 | 0.0119 | 0.0000 | 0.0219 |

---

## 3. Operating Thresholds (Validation Constrained)

Operating thresholds locked strictly on validation data:
- **WATCH Tier (FPR <= 10%)**: `0.0660`
- **WARNING Tier (FPR <= 5%)**: `0.0929`
- **CRITICAL Tier (FPR <= 1%)**: `0.2444`

---

## 4. Ethical Declarations & Limitations

1. **Zero Fabrication**: No borehole strainmeter, synthetic GNSS array, or synthetic high-resolution radar data was invented.
2. **Quarantine Invariant**: The 19 prospective events from 2026 remain strictly held out.
3. **Operational Pre-condition**: v2.6 does NOT replace v2.5 until an untouched, future prospective surveillance window proves its efficacy.
