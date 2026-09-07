# LAND-JEPA v2.5 vs v2.6 Real Prospective Head-to-Head Evaluation

**Document Type**: Prospective Shadow Test — Blind Evaluation Report  
**Control Model**: `v2.5-TRIGGER-AWARE-CHAMPION` (WARNING threshold=0.1980, FPR≤5%)  
**Challenger Model**: `v2.6-ABLATION-NO-CLOUDBURST` (WARNING threshold=0.0929, FPR≤5%)  
**Prospective Period**: 2026-09-07 → 2026-09-05 (30 days)  
**Zones**: 8 Northeast India Highway Corridors  
**Report Generated**: 2026-09-05T19:31:31.240306+00:00  

> [!CAUTION]
> **PROSPECTIVE SHADOW MODE — ACTIVE**: This is a blind evaluation. Neither model's thresholds, features, calibration, nor fusion weights have been modified after seeing prospective outcomes. The 2026 Jun–Sep quarantined events were NOT used for any tuning.

---

## 1. Event Sample Size

> [!CAUTION]
> **INSUFFICIENT SAMPLE SIZE**: Only 0 new independent verified events in the prospective window (minimum required: 15). Metrics are indicative only. Continue surveillance before making a promotion decision.

| Metric | Value |
|---|---|
| New Independent Verified Events | **0** |
| Minimum Required for Strong Decision | 15 |
| Prospective Duration (days) | 30 |
| Corridor-Days Evaluated | 240 |
| Previous Quarantined Events | 19 (Jun–Sep 2026, strictly excluded) |

---

## 2. Head-to-Head Metric Comparison

| Metric | v2.5 (Control) | v2.6 (Challenger) | Better |
|---|---|---|---|
| **Events (N)** | 0 | 0 | — |
| **Detected** | 0/0 | 0/0 | — |
| **Event Recall @ WARNING** | 0.0% | **0.0%** | v2.6 |
| **False Negative Rate (FNR)** | 0.0% | **0.0%** | v2.6 |
| **False Positive Rate (FPR)** | **51.4%** | 88.9% | v2.5 |
| **False Alarms / Day** | **0.5292** | 4.0000 | v2.5 |
| **Median Lead Time (h)** | 0.0h | **0.0h** | v2.6 |
| **PR-AUC** | 0.0650 | **0.0650** | v2.6 |
| **Brier Score** | 0.0179 | 0.6440 | v2.5 |
| **ECE** | 0.0049 | 0.0049 | v2.6 |

---

## 3. Promotion Criteria Assessment

| Criterion | Required | v2.5 | v2.6 | Pass? |
|---|---|---|---|---|
| Event Recall > v2.5 | >0.0% | 0.0% | 0.0% | ✅ |
| FNR ≤ v2.5 | <0.0% | 0.0% | 0.0% | ✅ |
| FPR ≤ 5% | ≤5.00% | 51.4% | 88.9% | ❌ |
| False Alarms/Day ≤ 0.075 | ≤0.0750 | 0.5292 | 4.0000 | ❌ |
| Median Lead Time ≥ 24h | ≥24.0h | 0.0h | 0.0h | ❌ |
| Brier ≤ 0.060 | ≤0.0600 | 0.0179 | 0.6440 | ❌ |
| N Events ≥ 15 | ≥15 | — | 0 | ❌ |

---

## 4. Final Verdict

> [!NOTE]
> **VERDICT: INSUFFICIENT EVIDENCE** — Fewer than 15 new independent events. Continue prospective surveillance.

---

## 5. Protocol Compliance

- **No retraining performed**: ✅
- **No threshold changes after observing outcomes**: ✅
- **Previous 2026 quarantined events excluded**: ✅ (19 events strictly isolated)
- **Causality enforced** (all predictions use only t_input ≤ T): ✅
- **Both models ran on identical inputs, zones, timestamps, events**: ✅
- **Shadow mode active** (no automated public alerts dispatched): ✅
