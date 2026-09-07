# Model Card: LAND-JEPA (Fused Landslide Early Warning Architecture)

---

## Model Details

- **Model Name**: Fused LAND-JEPA (`LandJEPARiskModel`)
- **Version**: 1.0.0 (Production Release)
- **Organization**: Team ZAIX — Smart India Hackathon (SIH26001)
- **Model Type**: Multimodal Self-Supervised Joint Embedding Predictive Architecture (JEPA) + Temporal Convolutional Network (TCN) + Static Terrain MLP + Gated Multimodal Fusion
- **Release Date**: September 2026
- **Checkpoint Location**: `ml/checkpoints/land_jepa_production/land_jepa_weights.pt`
- **Pretrained Encoder Weights**: `ml/checkpoints/jepa_pretrained_final/context_encoder_weights.pt`
- **License**: MIT / Open Research

---

## Intended Use

### Primary Intended Uses:
- **Landslide Early Warning in Northeast India (NER)**: Forecasting regional slope failure probability at 0h, 24h, and 48h horizons across monitoring corridors.
- **Geotechnical Decision Support**: Providing calibrated probability scores and feature attributions (rainfall accumulation, soil moisture anomaly, slope gradient) to disaster management authorities (NDMA, SDMA).
- **Offline Field Screening**: Edge-compatible deployment on citizen report queues and mobile inspection devices.

### Out-of-Scope & Prohibited Uses:
- **Autonomous Emergency Evacuation**: The model MUST NOT be connected directly to automated siren or mass evacuation triggers without mandatory human-in-the-loop analyst review (`requires_human_review = True`).
- **Uncalibrated Global Deployment**: The model is trained on Northeast Indian monsoon patterns and Himalayan geomorphology. It must not be deployed in arid or non-monsoonal regions without local retraining.
- **Micro-Scale Site Engineering**: The model provides regional corridor hazard estimation (30m DEM resolution), not structural foundation engineering.

---

## Training Data & Provenance

- **Atmospheric Data**: ECMWF ERA5-Land via Copernicus Climate Change Service (406,080 hourly records, 2011–2016).
- **Terrain Data**: ESA Copernicus DEM GLO-30 (30m spatial resolution GeoTIFF rasters).
- **Ground Truth**: NASA Global Landslide Catalog (GLC v1.1, 177 confirmed NER occurrences, date precision: day).
- **Split Policy**:
  - Train: 2011-01-01 to 2014-12-31 (11,440 windows, 76 positives)
  - Validation: 2015-01-01 to 2015-12-31 (2,842 windows, 35 positives)
  - Test (Hold-out): 2016-01-01 to 2016-10-15 (2,261 windows, 18 positives)
- **Leakage Prevention**: 72-hour pre-event exclusion buffer ($y = -1$); normalizers fit strictly on training splits.

---

## Benchmark Performance & Real Metrics

All metrics evaluated on hold-out 2016 test set:

| Evaluation Metric | Measured Value | Benchmark Baseline (XGBoost) | Improvement |
|---|---|---|---|
| **PR-AUC (24h Primary)** | **0.1285** | 0.0405 | **+217% (3.17×)** |
| **AUROC** | **0.7968** | 0.3630 | **+119%** |
| **Brier Score (Error)** | **0.0136** | 0.1459 | **-90.7% (10.7× lower error)** |
| **Expected Calib. Error (ECE)** | **0.0493** | 0.3135 | **-84.3%** |
| **Recall @ FPR $\le$ 5%** | **16.67% – 27.78%** | 0.00% | **+16.7% – 27.8%** |
| **Operational False Positive Rate**| **1.20%** ($\tau = 0.2877$) | 3.61% | **3× fewer false alarms** |
| **False Alarms per Day (8 Zones)** | **0.0092 fa/day** | 0.0270 fa/day | **3× lower false alarm rate** |
| **Precision** | **10.00%** | 4.00% | **2.5× higher precision** |
| **Inference Latency** | **0.185 ms / sample** | 0.001 ms | **Real-time capable (>5,400/s)** |

### Multi-Horizon Forecast Operating Parameters

| Horizon | Primary Lead Time | Detection Rate (FPR $\le$ 5%) | False Alarms / Day | Operational Response |
|---|---|---|---|---|
| **6h** | 0 to 6 hours | 27.3% | 0.0085 fa/day | Urgent roadblock closures, bus halts |
| **12h** | 6 to 12 hours | 30.8% | 0.0090 fa/day | School closures, travel advisories |
| **24h** | 12 to 24 hours | 27.8% | 0.0092 fa/day | NDRF staging, heavy equipment positioning |
| **48h** | 24 to 48 hours | 22.2% | 0.0110 fa/day | Supply chain rerouting, hospital alerts |
| **72h** | 48 to 72 hours | 18.5% | 0.0145 fa/day | District council preparedness advisories |

---

## Ethical & Safety Considerations

1. **Cry-Wolf Hazard Prevention**: In operational disaster management, unconstrained models reporting "100% recall" trigger false alarms on virtually every rainy day ($\text{FPR} = 100\%$), leading the public to ignore real warnings. LAND-JEPA enforces an operational operating point at $\text{FPR} \le 5\%$, yielding strictly 27 false alarms across 2,216 negative monitoring windows.
2. **Human-in-the-Loop Safeguard**: All citizen field reports and high-risk alerts enforce `requires_human_review = True`. Automated systems assist, but do not replace, geotechnical engineers.
3. **Sensor Reality & Honesty**: InSAR is explicitly disabled (`insar_mask = 0`) because Sentinel-1 C-band decorrelates over dense NER vegetation. Synthetic proxies have been permanently eliminated.
4. **VQC Quarantined**: The experimental Variational Quantum Classifier (VQC) is strictly partitioned in research mode and excluded from the operational warning path.

---

## How to Load and Run

```python
import torch
from ml.models.land_jepa_model import LandJEPARiskModel

# Load production model
model = LandJEPARiskModel(
    temporal_dim=18,
    terrain_dim=6,
    insar_dim=2,
    physics_dim=3,
    tcn_hidden_dim=64,
    tcn_num_blocks=4,
    tcn_kernel_size=3,
    terrain_hidden_dim=64,
    insar_hidden_dim=32,
    fused_dim=128,
    fusion_mode="gated",
    dropout=0.1,
)

state_dict = torch.load("ml/checkpoints/land_jepa_production/land_jepa_weights.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# Run inference on input batch
with torch.no_grad():
    predictions = model(x_temporal, x_terrain, x_physics=x_physics)
    prob_24h = torch.sigmoid(predictions["logits_24h"])
```
