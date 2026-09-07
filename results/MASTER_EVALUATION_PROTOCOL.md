# MASTER EVALUATION PROTOCOL: LAND-JEPA BENCHMARK STANDARDIZATION

**Project**: LAND-JEPA  
**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India (8 Monitored Corridors)  
**Status**: IMMUTABLE MASTER BENCHMARK SPECIFICATION  

---

## 1. Executive Summary & Benchmark Reconciliation

Across previous experimental cycles, reported early-warning metrics (specifically Event Recall) showed numerical divergence across benchmark iterations. This document provides the authoritative reconciliation, establishes the mathematical and procedural root causes, and fixes an immutable specification for all current and future evaluations.

### 1.1 Root Causes of Historical Divergence
1. **Denominator Definition (Unique Events vs Seed-Pooled Evaluations)**:
   - In earlier reports (e.g. *Final Forecast Report v2.1*), the 2016 blind test set contained **19 unique physical landslide events** in the temporal coverage window. When evaluated across 3 random seeds, $19 \times 3 = 57$ event-evaluations occurred. Detecting 27 out of 57 instances yielded:
     $$\text{Event Recall} = \frac{27}{57} = 47.37\% \approx 47.4\%$$
   - When the same 19 events were evaluated across 5 seeds, $19 \times 5 = 95$ event-evaluations occurred, yielding $45 / 95 = 47.37\% \approx 47.4\%$.
   - In the subsequent *High-Performance Model Training* run, an expanded spatial matching radius (75 km vs 60 km) captured **24 events** in the temporal window. Evaluating the baseline v2.2 on these 24 events caught ~9 events per seed:
     $$\text{Event Recall} = \frac{9}{24} = 37.5\% \approx 36.7\% \text{ (mean across 5 seeds)}$$
2. **Catalog Spatial Radius**:
   - Initial catalog (`real_ner_events.pkl`): 60 km corridor radius $\to$ 177 total events (2011–2016), with **19 events** in the active 2016 ERA5 test period.
   - Expanded catalog (`expanded_ner_events.pkl`): 75 km corridor radius $\to$ 170 deduplicated events (2011–2016), with **24 events** in the active 2016 ERA5 test period.
3. **Threshold Selection Discipline**:
   - In some early scripts, thresholds were computed on sliding windows and then applied at the event level. Under the Master Protocol, thresholds are optimized **exclusively on validation data (2015)** under operational FPR ceilings ($\text{FPR} \le 1\%, 5\%, 10\%$).

---

## 2. Immutable Benchmark Specification

### 2.1 Monitored Corridors (8 High-Risk NER Highways)
- `REAL-NER-001`: Guwahati Hills Corridor (Assam)
- `REAL-NER-002`: Shillong Plateau / Sohra (Meghalaya)
- `REAL-NER-003`: Imphal - Senapati NH-2 Corridor (Manipur)
- `REAL-NER-004`: Kohima - Phek Ridge (Nagaland)
- `REAL-NER-005`: Aizawl Mountain Slopes (Mizoram)
- `REAL-NER-006`: Bhalukpong - Tawang Corridor (Arunachal Pradesh)
- `REAL-NER-007`: Atharamura Hills (Tripura)
- `REAL-NER-008`: Gangtok - Teesta Valley (Sikkim)

### 2.2 Chronological Split Boundary
- **Training Set**: 2011-01-01 00:00:00 UTC to 2014-12-31 23:00:00 UTC (4 complete monsoon seasons)
- **Validation Set**: 2015-01-01 00:00:00 UTC to 2015-12-31 23:00:00 UTC (1 complete monsoon season, for all threshold tuning, hyperparameter selection, and calibration fitting)
- **Blind Test Set**: 2016-01-01 00:00:00 UTC to 2016-10-14 23:00:00 UTC (Held out completely unseen until final inference)

### 2.3 Strict Anti-Leakage & Provenance Invariants
1. **Zero Future Leakage**: For every prediction at time $T$, $\max(t_{\text{input}}) \le T$. No future observations are ever accessible.
2. **Validation-Only Tuning**:
   - Operating thresholds ($\text{FPR} \le 1\%, 5\%, 10\%$) fitted **strictly on validation set (2015)**.
   - Temperature scaling parameter $T$ fitted **strictly on validation set (2015)**.
   - Robust feature scalers fitted **strictly on training set (2011–2014)**.
   - Test set (2016) is never accessed during training, scaling, or thresholding.
3. **No InSAR Ground Deformation in Production**: InSAR displacement channels remain strictly disabled due to C-band vegetative decorrelation in Northeast India.
4. **No Quantum Variational Classifiers in Production**: VQC remains strictly research-only.

### 2.4 Mathematical Definition of Evaluation Metrics
For each horizon $H \in \{6\text{h}, 12\text{h}, 24\text{h}, 48\text{h}, 72\text{h}\}$ across 5 seeds:

1. **Sliding-Window PR-AUC**:
   $$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) P_k$$
   Calculated on sliding-window predictions against binary ground truth $y_t$.
2. **Physical Event Recall**:
   $$\text{Event Recall} = \frac{\text{Number of Unique Confirmed Events Warned with Lead Time } \Delta t \ge 1.0\text{h}}{\text{Total Number of Confirmed Physical Events in Test Period}}$$
   Both **Unique Physical Event Recall** and **Seed-Averaged Event Recall** must be explicitly documented.
3. **Advance Warning Lead Time**:
   $$\text{Lead Time} = t_{\text{event}} - t_{\text{first\_warning}}$$
   Where $t_{\text{first\_warning}}$ is the timestamp of the earliest prediction triggering $p \ge \theta_{\text{FPR}\le 5\%}$ within $[t_{\text{event}} - H - 12\text{h}, t_{\text{event}})$.
4. **False Alarms Per Day**:
   $$\text{False Alarms / Day} = \frac{\text{Total Distinct False Alert Clusters in Non-Event Periods}}{\text{Total Monitored Days in Test Period}}$$
5. **False Negative Rate (FNR)**:
   $$\text{FNR} = 1.0 - \text{Recall @ FPR } \le 5\%$$
6. **Calibration Quality**:
   - **Brier Score**: $\frac{1}{N} \sum (p_i - y_i)^2$
   - **Expected Calibration Error (ECE)**: $\sum_{b=1}^{10} \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$

---

## 3. Operational Promotion Rules
Any new candidate model will be promoted over baseline `v2.2-PREDICTION-OPTIMIZED` **ONLY IF** it achieves a statistically superior Pareto-optimal combination on:
- **Event Recall @ 24h** ($\ge \text{Baseline}$)
- **Recall @ FPR } \le 5\%$ ($\ge \text{Baseline}$)
- **False Negative Rate** ($\le \text{Baseline}$)
- **False Alarms Per Day** ($\le \text{Baseline}$)
- **Advance Lead Time** ($\ge \text{Baseline}$)
- **Brier Calibration Score** ($\le \text{Baseline}$)

If the candidate fails on Event Recall or FNR, **`v2.2-PREDICTION-OPTIMIZED` MUST BE RETAINED WITHOUT EXAGGERATION**.
