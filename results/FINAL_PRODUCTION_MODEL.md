# LAND-JEPA — Final Production Model Specification
## SIH26001 — Team ZAIX — September 2026

---

## 1. Selected Production Model

| Parameter | Specification |
|---|---|
| **Model Name** | **Fused LAND-JEPA** (`LandJEPARiskModel`) |
| **Architecture** | Multimodal Joint Embedding Predictive Architecture (TCN Temporal Encoder + DEM Terrain Encoder + SWI Physics Feature Concatenation + Gated Fusion) |
| **Checkpoint Path** | `ml/checkpoints/land_jepa_production/land_jepa_weights.pt` |
| **Pretrained Backbone** | `ml/checkpoints/jepa_pretrained_final/context_encoder_weights.pt` |
| **Model Registry Status** | **PRODUCTION / ACTIVE** |
| **Operating Device** | CPU / CUDA (Edge & Server compatible) |
| **Input Shape** | Temporal: `(B, 168, 18)`, Terrain: `(B, 6)`, Physics: `(B, 3)`, InSAR: `(B, 2)` (masked) |
| **Output Horizons** | 0h (`head_0h`), 24h (`head_24h`), 48h (`head_48h`) |
| **Production Threshold** | **$\tau = 0.2877$** (Operational FPR $\le 5\%$ operating point) |

---

## 2. Selection Rationale: Why Fused LAND-JEPA Won

The production model was selected by evaluating four candidates across 8 operational criteria on real Northeast India scientific benchmark data (NASA GLC 177 events, ERA5-Land 406,080 hours, Copernicus DEM GLO-30):

```
                        SELECTION MATRIX
┌──────────────────────┬──────────┬────────────────┬──────────┬──────────────────┐
│ Criterion            │ XGBoost  │ Supervised TCN │ JEPA-TCN │ Fused LAND-JEPA  │
├──────────────────────┼──────────┼────────────────┼──────────┼──────────────────┤
│ PR-AUC               │  0.0405  │     0.0579     │  0.0418  │    0.1285 (WIN)  │
│ Brier Score (Error)  │  0.1459  │     0.0360     │  0.0202  │    0.0136 (WIN)  │
│ Expected Calib. Error│  0.3135  │     0.1561     │  0.0612  │    0.0493 (WIN)  │
│ Recall @ FPR <= 5%   │   0.00%  │     12.50%     │  16.67%  │    16.67%–27.8%  │
│ Operational FPR      │   3.61%  │      4.22%     │   1.74%  │     1.20% (WIN)  │
│ Precision            │   4.00%  │      1.97%     │   7.14%  │    10.00% (WIN)  │
│ 1% Label Efficiency  │   0.00%  │     12.50%     │  12.50%  │  Non-collapsing  │
│ Inference Latency    │ 0.001 ms │    0.242 ms    │ 0.189 ms │  0.185 ms (FAST) │
└──────────────────────┴──────────┴────────────────┴──────────┴──────────────────┘
```

### Decisive Factors:
1. **Multimodal Super-Additivity**: Combining temporal atmospheric features with Copernicus DEM geomorphology and Soil Water Index (SWI) physics increased PR-AUC from 0.0418 to **0.1285** (a **3.07× increase**).
2. **Probability Calibration**: Fused LAND-JEPA achieved the lowest Brier score in the entire benchmark (**0.0136** vs. XGBoost 0.1459), guaranteeing that alert probabilities directly correlate with true ground risk rather than overconfident uncalibrated logits.
3. **Controlled False-Positive Rate**: At threshold $\tau = 0.2877$, the model produces an FPR of **1.20%**, correctly suppressing false alarms across 2,216 true-negative operational monitoring windows.
4. **Sub-millisecond Latency**: 0.185 ms inference enables real-time regional risk mapping (>5,400 zone evaluations per second on standard commodity CPU).

---

## 3. Real NER Benchmark Performance

All metrics measured on the independent hold-out test set (real 2016 landslide occurrences, strictly post-cutoff):

- **PR-AUC**: **0.1285** (Baseline random prevalence: 0.0080; 16× above no-skill)
- **AUROC**: **0.7968**
- **Precision**: **0.1000** (1 true positive per 10 alerts in 1:125 rare-event setting)
- **Recall (at FPR $\le 5\%$)**: **16.67% – 27.78%** across seeds
- **False Negative Rate (FNR)**: **83.33%** under strict 1.20% FPR constraint
- **False Positive Rate (FPR)**: **0.0120** (1.20%)
- **Brier Score**: **0.0136**
- **Expected Calibration Error (ECE)**: **0.0493**
- **Latency**: **0.185 ms / prediction**

> [!NOTE]
> **Scientific Integrity Note on Recall**: Unconstrained models can report "100% recall" only by setting the decision threshold to 0.0001, which triggers 685 false alarms (100% FPR), causing total alert fatigue. Under an operationally valid policy ($\text{FPR} \le 5\%$), true positive recall is **16.7% – 27.8%**.

---

## 4. Operational Checkpoint Verification

- State dict tensor count: **77 parameters**
- Checkpoint verification test: `python -c "from ml.models.land_jepa_model import LandJEPARiskModel; ..."` passed.
- Forward pass output shape: `logits_0h: (B, 1)`, `logits_24h: (B, 1)`, `logits_48h: (B, 1)`.
- Backend initialization log:
  ```
  Loaded production LandJEPARiskModel from D:\SIH26001\ml\checkpoints\land_jepa_production\land_jepa_weights.pt (temp=18, terr=6, insar=2)
  RiskPipelineService ready | land_jepa=✓
  ```

---

## 5. InSAR & VQC Policy

1. **InSAR**:
   - Status: **DISABLED / MASKED** (`insar_mask = 0`).
   - Rationale: C-band Sentinel-1 phase coherence is destroyed by dense sub-tropical vegetation canopy in Northeast India. No real interferometric deformation products exist. Synthetic proxies are completely excised. Architecture remains InSAR-ready for future L-band NISAR products.
2. **VQC (Variational Quantum Classifier)**:
   - Status: **EXPERIMENTAL RESEARCH ONLY** (`/model/quantum`).
   - Rationale: Multi-seed sweep confirmed VQC PR-AUC $\approx 0.009$, matching random class prevalence. Classical Logistic Regression (PR-AUC 0.088) and MLP (PR-AUC 0.079) strongly outperform VQC with $1400\times$ lower latency.
   - VQC is strictly quarantined from the emergency alert pipeline.
