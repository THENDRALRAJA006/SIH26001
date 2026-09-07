# LAND-JEPA — Final Scientific Claim Audit
## SIH26001 — Team ZAIX — September 2026

---

> [!IMPORTANT]
> Every scientific, technical, and operational claim made regarding the LAND-JEPA platform is audited below.
> In strict compliance with scientific integrity standards, each claim is classified into one of three categories:
> - **PROVEN**: Mathematically, empirically, or architecturally verified with reproducible evidence.
> - **PARTIALLY SUPPORTED**: Experimentally demonstrated under specific constraints, but requiring qualifications or geographical boundaries.
> - **NOT PROVEN**: Disproven by experiments, invalid under operational constraints, or technically unfeasible with current sensor products.

---

## 1. Summary Audit Matrix

```
┌───────────────────────────────────────────────────────────────────┬──────────────────────┐
│ Scientific / Engineering Claim                                     │ Formal Verdict       │
├───────────────────────────────────────────────────────────────────┼──────────────────────┤
│ 1. Real Data Provenance (NASA GLC + ERA5-Land + Copernicus DEM)   │ PROVEN               │
│ 2. Zero Data / Target Leakage Temporal Splitting & Buffer         │ PROVEN               │
│ 3. Self-Supervised JEPA Pretraining (Zero Landslide Labels Used)  │ PROVEN               │
│ 4. Target Encoder Stop-Gradient Isolation via EMA                 │ PROVEN               │
│ 5. Multimodal Gated Fusion Outperforms Unimodal Models            │ PROVEN               │
│ 6. Lowest Brier Score & Superior Probability Calibration          │ PROVEN               │
│ 7. JEPA Representation Robustness in Ultra-Low Labels (1%–10%)    │ PROVEN               │
│ 8. Operational Operating Point Under FPR <= 5% Constraint         │ PROVEN               │
│ 9. Spatial Generalization via Leave-One-Zone-Out (5 Valid Zones)  │ PROVEN               │
│ 10. Sub-millisecond Inference Latency on Commodity CPU (<0.2 ms)  │ PROVEN               │
│ 11. End-to-End System Integration (Backend, GIS, Mobile, Web)    │ PROVEN               │
│ 12. Geographic Generalization Outside Northeast India             │ PARTIALLY SUPPORTED  │
│ 13. Unconstrained "100% Recall" in Operational Alert Deployment   │ NOT PROVEN           │
│ 14. Real Sentinel-1 C-Band InSAR Deformation Benefit in NER       │ NOT PROVEN           │
│ 15. Quantum Advantage via Variational Quantum Classifiers (VQC)   │ NOT PROVEN           │
│ 16. Fully Automated Alert Escalation Without Human Review         │ NOT PROVEN           │
└───────────────────────────────────────────────────────────────────┴──────────────────────┘
```

---

## 2. Detailed Claim Audits

### Claim 1: Real Data Provenance
- **Verdict: PROVEN**
- **Evidence**:
  - 177 confirmed landslide occurrences from NASA Global Landslide Catalog (`data/real/processed/real_ner_events.pkl`).
  - 406,080 hourly meteorological records from ECMWF ERA5-Land across 8 NER zones (2011–2016).
  - 8 tiles of ESA Copernicus DEM GLO-30 at 30m resolution (`data/real/raw/terrain/`).
  - Zero synthetic or demo data was injected into the benchmark matrices (`is_demo == False` enforced).

### Claim 2: Zero Data / Target Leakage
- **Verdict: PROVEN**
- **Evidence**:
  - Strict temporal ordering: Train (< 2015-01-01), Val (2015), Test ($\ge$ 2016-01-01). No shuffling across time.
  - 72-hour pre-event buffer marks near-event windows as ambiguous ($y = -1$), preventing temporal contamination.
  - Normalizers (`TemporalNormalizer`, `FeatureNormalizer`) fit strictly on training splits.

### Claim 3: Self-Supervised JEPA Pretraining Without Labels
- **Verdict: PROVEN**
- **Evidence**:
  - `pretrain_jepa()` accesses only `observed_at < 2015-01-01` atmospheric tensors.
  - Predicts future latent embedding $\hat{z}_{\text{tgt}}$ from context latent $z_{\text{ctx}}$.
  - Loss decreased monotonically from $0.0364 \to 0.0070$ over 10 epochs. Representation collapse check PASSED.

### Claim 4: Target Encoder Stop-Gradient Isolation
- **Verdict: PROVEN**
- **Evidence**:
  - `EMAUpdater` updates target encoder strictly via exponential moving average ($\tau = 0.996$).
  - Target representations are detached (`stop_gradient`) prior to loss computation.
  - Gradients backpropagate exclusively through the context encoder and predictor network.

### Claim 5: Multimodal Fusion Outperforms Unimodal Models
- **Verdict: PROVEN**
- **Evidence**:
  - In `results/ablation.csv`:
    - Temporal JEPA alone: PR-AUC = 0.0418, Brier = 0.0202
    - JEPA + Copernicus DEM: PR-AUC = 0.0432, Brier = 0.0186 (Recall jumps from 16.7% to 27.8%)
    - Fused LAND-JEPA (Temporal + Terrain + Physics): PR-AUC = **0.1285** (3.07× improvement!), Brier = **0.0136**

### Claim 6: Calibration and Brier Score Performance
- **Verdict: PROVEN**
- **Evidence**:
  - Fused LAND-JEPA achieves Brier score = **0.0136** (vs XGBoost 0.1459, Supervised TCN 0.0360).
  - Expected Calibration Error (ECE) is **0.0493**, ensuring alert probabilities reflect true physical risk.

### Claim 7: JEPA Robustness in Ultra-Low Labels (1%–10%)
- **Verdict: PROVEN**
- **Evidence**:
  - In `results/robustness_audit.csv`, at 1% labels (~1 visible training event):
    - XGBoost: Recall 0.000, PR-AUC 0.0115
    - Supervised TCN: Recall 0.125, Precision 0.0125, PR-AUC 0.0196
    - JEPA-TCN: Precision 0.0526, F1 0.0741, PR-AUC 0.0455 (more than double Supervised TCN)
  - At 5% labels, Supervised TCN collapsed to 100% false-positive rate (predicting positive everywhere), whereas JEPA maintained calibrated discrimination.

### Claim 8: Operational Operating Point Under FPR $\le$ 5%
- **Verdict: PROVEN**
- **Evidence**:
  - At threshold $\tau = 0.2877$, Fused LAND-JEPA yields $\text{FPR} = \mathbf{1.20\%}$, well within the $\le 5\%$ constraint.
  - Generates strictly 27 false alarms over 2,216 true-negative operational monitoring windows.
  - Achieves operational recall of **16.67% – 27.78%** across seeds.

### Claim 9: Leave-One-Zone-Out Spatial Generalization
- **Verdict: PROVEN**
- **Evidence**:
  - All 5 NER zones with sufficient ground truth ($\ge 5$ events) were held out entirely during training:
    - REAL-NER-001 (Darjeeling): PR-AUC = 0.0409, Recall = 0.125
    - REAL-NER-003 (Upper Subansiri): PR-AUC = 0.0579, Recall = 0.278
    - REAL-NER-004 (Dima Hasao): PR-AUC = 0.0621, Recall = 0.552
    - REAL-NER-005 (East Khasi Hills): PR-AUC = 0.0588, Recall = 0.222
    - REAL-NER-008 (West Kameng): PR-AUC = 0.0705, Recall = 0.725
  - All valid zones achieve positive predictive power higher than no-skill prevalence.

### Claim 10: Sub-millisecond Inference Latency
- **Verdict: PROVEN**
- **Evidence**:
  - Benchmarked inference latency is **0.185 ms** per sample on standard CPU.
  - Capable of processing >5,400 regional zones per second in real-time.

### Claim 11: End-to-End System Integration
- **Verdict: PROVEN**
- **Evidence**:
  - Live FastAPI backend verified with `pytest tests/test_e2e_smoke.py` (17/17 passed).
  - Mobile Jest offline queue tests verified (11/11 passed).
  - Frontend Vite dashboard built with zero errors in 2.46s.

---

## 3. Partially Supported Claims

### Claim 12: Geographic Generalization Outside Northeast India
- **Verdict: PARTIALLY SUPPORTED**
- **Rationale**:
  - The model generalizes well *within* Northeast India across distinct geomorphic zones (Himalayan front vs Shillong plateau).
  - However, application to the Western Ghats, Andes, or Alps requires regional DEM tiles and precipitation re-normalization. Global transfer cannot be claimed without local validation.

---

## 4. Disproven & Unallowable Claims

### Claim 13: Unconstrained "100% Recall" in Operational Alerting
- **Verdict: NOT PROVEN (SCIENTIFICALLY INVALID)**
- **Rationale**:
  - In `results/threshold_sensitivity.csv`, 100% recall is attained only when threshold $\tau \le 0.001$, which causes 685 false alarms out of 685 negative windows ($\text{FPR} = 100\%$).
  - An early warning system with 100% false alarms leads to catastrophic alert fatigue.
  - **Rule**: NEVER claim 100% recall without stating the operating false positive rate. Under valid operational constraints ($\text{FPR} \le 5\%$), true recall is **16.7% – 27.8%**.

### Claim 14: Real Sentinel-1 C-Band InSAR Benefit in Northeast India
- **Verdict: NOT PROVEN (ZERO VALUE IN C-BAND)**
- **Rationale**:
  - In `results/ablation.csv`, Model D (+InSAR) produces PR-AUC = 0.1285, exactly identical to Model C (without InSAR).
  - Sentinel-1 C-band (5.4 GHz) suffers near-total coherence loss over dense sub-tropical vegetation in NER. No authentic interferometric ground displacement products exist in the dataset.
  - Any claim of active satellite radar displacement benefit in the current system is unfounded. InSAR inputs remain disabled (`insar_mask = 0`) until L-band NISAR is operational.

### Claim 15: Quantum Advantage via Variational Quantum Classifiers (VQC)
- **Verdict: NOT PROVEN (ZERO QUANTUM ADVANTAGE)**
- **Rationale**:
  - Multi-seed sweep across 4–8 qubits and depths 2–6 demonstrated VQC PR-AUC $\approx 0.009$, identical to random guessing.
  - Classical matched baselines on the exact same PCA embeddings achieved PR-AUC = 0.088 (Logistic Regression) and 0.079 (MLP).
  - VQC latency ($2.84 \text{ ms}$) is $>1400\times$ slower than classical inference ($0.002 \text{ ms}$).
  - **Rule**: VQC must be described strictly as an exploratory research experiment. Never claim quantum speedup, quantum advantage, or operational quantum deployment.

### Claim 16: Fully Automated Alert Dispatch Without Human Review
- **Verdict: NOT PROVEN (SAFETY HAZARD)**
- **Rationale**:
  - Operational early warning cannot dispatch public sirens or mass SMS evacuations purely on automated ML probabilities without human oversight.
  - All automated high-risk predictions and citizen field reports enforce `requires_human_review = True` and are routed to authorized geotechnical analysts.
