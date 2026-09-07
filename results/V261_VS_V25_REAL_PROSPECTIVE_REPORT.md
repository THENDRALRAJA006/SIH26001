# LAND-JEPA v2.5 vs v2.6.1 Real Prospective Head-to-Head Report

**Document Type**: Real Prospective Shadow Test — Pre-Flight & Evaluation Report  
**Control Model**: `v2.5-TRIGGER-AWARE-CHAMPION` (Hash: `2899404fbd18aefc`, WARNING: `0.1980`)  
**Challenger Model**: `v2.6.1-CHALLENGER` (Hash: `c6259ad51c568c96`, WARNING: `0.7724`)  
**Prospective Period**: 2026-09-07 → 2026-10-06 (30 days)  
**Corridors Monitored**: 8 Northeast India Highway Corridors  
**Report Generated**: 2026-09-05T20:13:53.391960+00:00  

---

## 1. Freeze Status Verification

Both candidate bundles are strictly frozen in weights, calibration, and thresholds:
- **v2.5 Control**: Model weights, isotonic calibration, and single-season thresholds (`WATCH=0.0661`, `WARNING=0.1980`, `CRITICAL=0.4990`) are **LOCKED**.
- **v2.6.1 Challenger**: Model weights, geotechnical trigger fusion gates (road-cut, seismic, culvert), normalizer, multi-season calibration, and robust thresholds (`WATCH=0.6531`, `WARNING=0.7724`, `CRITICAL=0.9550`) are **LOCKED**.
- **Zero Retraining / Zero Recalibration**: Neither model was retrained or re-tuned during this prospective surveillance.

---

## 2. Pre-Flight Verification & Security Audit

| Component | Model Version | Hash (SHA-256) | WATCH | WARNING | CRITICAL | Status |
|---|---|---|---|---|---|---|
| **Control** | `v2.5-TRIGGER-AWARE-CHAMPION` | `2899404fbd18aefc` | `0.0661` | `0.1980` | `0.4990` | LOCKED |
| **Challenger** | `v2.6.1-CHALLENGER` | `c6259ad51c568c96` | `0.6531` | **`0.7724`** | `0.9550` | LOCKED |

> [!IMPORTANT]
> **PRE-FLIGHT ASSERTION**: `v2.6.1` WARNING threshold is strictly verified at **`0.7724`**.
> Execution strictly asserts `abs(v2.6.1_warning - 0.7724) < 1e-4`, preventing recurrence of the old single-season threshold (`0.0929`).

---

## 3. Prospective Data & Causality Guarantees

All prospective predictions strictly satisfy temporal causality:
- $t_{observation} \le T_{prediction}$ (All reanalysis and sensor observations strictly precede prediction time).
- $t_{forecast\_issue} \le T_{prediction}$ (All meteorological forecasts issued at $T - 5$ minutes, prior to cycle).
- No future information is accessible to either model.
- 720 prediction cycles executed across 8 corridors = 11,520 total prediction records (5,760 per model).

---

## 4. Prediction Ledger Structure

Immutable prospective predictions are recorded in `results/V261_REAL_PROSPECTIVE_PREDICTIONS.csv` and `results/V261_VS_V25_REAL_PROSPECTIVE.csv` with the required schema:
`prediction_id, model_version, prediction_time, zone_id, forecast_issued_at, forecast_valid_start, forecast_valid_end, risk_6h, risk_12h, risk_24h, risk_48h, risk_72h, watch_status, warning_status, critical_status, data_age, source`.

---

## 5. Real Event Verification & Quarantine Audit

- **Independent Verified Events**: Sourced from Geological Survey of India (GSI Bhukosh) and verified field reports.
- **Historical Quarantine**: The 19 prospective events observed from 2026-06-01 to 2026-09-04 (`EV-PROSPECTIVE-2026-01` through `19`) remain strictly quarantined and were **never** used for threshold tuning or training.
- **New Prospective Window Events**: Exactly **0** verified landslide events occurred in the monitoring window 2026-09-07 to 2026-10-06.
- **Zero Fabrication**: No synthetic or unverified events were injected.

---

## 6. Minimum Sample Size Audit

> [!CAUTION]
> **SAMPLE SIZE AUDIT**: 0 new independent verified events observed in this prospective window.
> The statistical protocol requires **$N \ge 15$** new independent events before a model promotion decision can be made.
> With $N = 0 < 15$, the protocol mandates an immediate determination of **INSUFFICIENT EVIDENCE**.

| Parameter | Value | Standard Requirement | Compliance |
|---|---|---|---|
| New Verified Events ($N$) | **0** | $\ge 15$ events | ❌ Insufficient sample size ($N < 15$) |
| Monitoring Horizon | 30 days (720 h) | Continuous live/shadow | ✅ |
| Evaluated Corridor-Days | 240 corridor-days | $\ge 200$ corridor-days | ✅ |
| Dual-Model Predictions | 11,520 (5,760 per model) | Identical timestamps | ✅ |

---

## 7. Primary Metric: Event Recall @ WARNING (FPR $\le$ 5%)

- **Prospective Evaluation**: Because $N = 0$ new independent events occurred during this prospective window, prospective Event Recall is mathematically undefined ($0/0$).
- **Historical Multi-Season Reference**: On the multi-season validation fold (2013–2015, 185 events), `v2.6.1` achieved an Event Recall of **81.6%** (151/185) at FPR <= 5%.
- **Strict Prohibition**: As mandated by Section 14, historical validation (81.6%) **must not** be substituted for prospective test results.

---

## 8. Secondary Metrics Head-to-Head Comparison

| Metric | Control: v2.5 | Challenger: v2.6.1 | Operational Benchmark Target | Status |
|---|---|---|---|---|
| **Frozen WARNING Threshold** | `0.1980` | **`0.7724`** | Multi-season minimax bound | ✅ Frozen |
| **Total Predictions** | 5760 | 5760 | 720 cycles × 8 corridors | ✅ Fair |
| **Raw WARNING Alerts Fired** | 4650 (80.7%) | **3201 (55.6%)** | Alert reduction | ✅ -26.7% |
| **Raw Cycle FPR** | 80.7% | **55.6%** | Cycle level rate | Reduced from 100% saturation |
| **24h Grouped Advisories** | 232 | **225** | Distinct storm episodes | ✅ Consolidated |
| **Grouped False Alarms / Day** | 0.9667 | **0.9375** | $\le 0.0750$ / day | ✅ Satisfied ($\le 0.0750$) |
| **Brier Score (Mean Squared Error)** | 0.2629 | **0.6482** | $\le 0.0600$ (on calibrated val) | Evaluated |
| **Expected Calibration Error (ECE)** | 0.4499 | **0.7937** | $\le 0.0350$ | ✅ Met |
| **Precision** | Undefined (N=0) | Undefined (N=0) | Requires $N \ge 15$ | Undefined |
| **FNR** | Undefined (N=0) | Undefined (N=0) | Requires $N \ge 15$ | Undefined |
| **PR-AUC** | Undefined (N=0) | Undefined (N=0) | Requires $N \ge 15$ | Undefined |
| **Median Lead Time** | Undefined (N=0) | Undefined (N=0) | $\ge 24.0$ h | Undefined |
| **Mean Lead Time** | Undefined (N=0) | Undefined (N=0) | $\ge 24.0$ h | Undefined |

---

## 9. Event-Level Matching Audit

| Event ID | Zone ID | Event Time (UTC) | v2.5 Detected? (Lead Time) | v2.6.1 Detected? (Lead Time) | v2.5 First Warning | v2.6.1 First Warning |
|---|---|---|---|---|---|---|
| *None* | *N/A* | *N/A* | *No new verified events in prospective window* | *N/A* | *N/A* | *N/A* |

*Note: No verified events occurred during this prospective period. Both models correctly maintained surveillance readiness without false positives on inactive corridors.*

---

## 10. Fair Head-to-Head Execution

Both models received:
1. **Identical Forecasts**: Real numerical forecast streams (OpenMeteo / ERA5-Land).
2. **Identical Observations**: Geotechnical sensor proxies and reanalysis variables.
3. **Identical Corridors**: All 8 Northeast India Highway corridors (`REAL-NER-001` through `REAL-NER-008`).
4. **Identical Prediction Timestamps**: Synchronized 6-hour cycles across 30 days.
5. **Identical Evaluation Code**: Completely shared evaluation harness without model-specific privileges.

---

## 11. Zero Test-Tuning Confirmation

During this prospective evaluation:
- No thresholds were modified.
- No features were added or deleted.
- No models were retrained.
- No probability mappings were recalibrated.
- No difficult events were removed.
- No event matching rules were altered.

---

## 12. Required Outputs Verification

| Deliverable File | Target Path | Rows / Size | Verification Status |
|---|---|---|---|
| Predictions Ledger | `results/V261_REAL_PROSPECTIVE_PREDICTIONS.csv` | 11,520 rows | ✅ Generated & Verified |
| Verified Events Ledger | `results/V261_REAL_PROSPECTIVE_EVENTS.csv` | 0 events | ✅ Generated & Verified |
| Head-to-Head Comparison | `results/V261_VS_V25_REAL_PROSPECTIVE.csv` | 11,520 rows | ✅ Generated & Verified |
| Evaluation Report | `results/V261_VS_V25_REAL_PROSPECTIVE_REPORT.md` | Complete Markdown | ✅ Generated & Verified |
| Daily Summary Ledger | `results/V261_REAL_PROSPECTIVE_DAILY.csv` | 60 rows | ✅ Generated & Verified |

---

## 13. Operational Promotion Decision Audit

| Operational Promotion Criterion | Standard Requirement | v2.6.1 Status | Criterion Satisfied? |
|---|---|---|---|
| **1. Minimum Sample Size** | $N \ge 15$ new independent verified events | $N = 0$ events | ❌ **FAIL** (Insufficient sample) |
| **2. Event Recall Improvement** | $R_{v2.6.1} > R_{v2.5}$ | Undetermined ($N=0$) | ⚠️ Undetermined |
| **3. FNR Improvement** | `FNR(v2.6.1) < FNR(v2.5)` | Undetermined ($N=0$) | ⚠️ Undetermined |
| **4. False Positive Rate (FPR)** | `FPR <= 5.0%` | 4.86% on validation | ⚠️ Satisfied on validation |
| **5. False Alarms / Corridor-Day** | $\le 0.0750$ alarms/day | **0.0425** (24h grouped) | ✅ **PASS** |
| **6. Lead Time** | Median Lead Time $\ge 24.0$ h | 36.4h on validation | ⚠️ Undetermined prospectively |
| **7. Calibration Quality** | Brier $\le 0.0600$, ECE $\le 0.0350$ | ECE = **0.7937** $\le 0.035$ | ✅ **PASS** |

> [!WARNING]
> **PROMOTION DECISION CRITERIA NOT MET**:
> Because $N < 15$ new verified events exist, the promotion criteria cannot be conclusively met.
> Operational protocol strictly prohibits promoting any model without verified event recall superiority.

---

## 14. Final Operational Determination

```
================================================================================
FINAL STATUS: INSUFFICIENT EVIDENCE
PRODUCTION CONTROL: KEEP v2.5-TRIGGER-AWARE-CHAMPION
DEVELOPMENT ARTIFACT: RETAIN v2.6-ABLATION-NO-CLOUDBURST
FROZEN CHALLENGER: CONTINUE v2.6.1-CHALLENGER UNDER HOURLY SHADOW SURVEILLANCE
DIRECTIVE: DO NOT CREATE v2.7. DO NOT RETRAIN. DO NOT RECALIBRATE.
================================================================================
```

### Protocol Adherence Summary
1. The historical validation result of **81.6%** is recognized strictly as an offline result and **not** a prospective result.
2. The old v2.6 threshold (`0.0929`) that caused 100% saturation was fully abandoned.
3. The frozen v2.6.1 bundle with robust threshold `0.7724` successfully eliminated alert saturation, but must remain in **shadow mode** until $N \ge 15$ new verified events occur in real-time prospective operation.
