# LAND-JEPA v2.6: PROSPECTIVE SHADOW TEST PROTOCOL

**Document**: Prospective Shadow Validation Protocol for v2.6  
**Candidate Model**: `v2.6-TRIGGER-AWARE-CANDIDATE`  
**Current Baseline**: `v2.5-TRIGGER-AWARE-CHAMPION` (Warning Recall = 78.9%, FPR = 3.69%, Median Lead = 24.0h)  
**Date**: 2026-09-05T19:13:19.447425+00:00  

---

## 1. Replacement Decision Rule

In accordance with strict operational and scientific invariants:
> **Final Rule**: *"v2.6 replaces v2.5 only if a NEW untouched prospective evaluation demonstrates a genuine operational improvement. Do not target 95% by changing thresholds after seeing outcomes."*

### Minimum Acceptance Criteria for Production Promotion:
1. **Event Recall @ WARNING (FPR <= 5%)**: Must empirically exceed **78.9%** on a new independent event catalog (n >= 15).
2. **Operational False Positive Rate (FPR)**: Must remain strictly <= 5.0% across all monitored corridor-days.
3. **False Alarms per Day**: Must remain <= 0.0750 alerts per corridor-day (< 1 alert every 13.3 days).
4. **Advance Warning Lead Time**: Median lead time must remain >= 24.0 hours.
5. **Probabilistic Calibration**: Brier score <= 0.0600 and ECE <= 0.0350.

---

## 2. Protocol Invariants for the Next Prospective Test

1. **Frozen Production Candidate**: All weights, feature scaling, isotonic calibration, and thresholds (`WATCH=0.0660`, `WARNING=0.0929`, `CRITICAL=0.2444`) must be locked prior to initiating live prospective inference.
2. **Causality Guards**: Verify max(t_input) <= T_pred and t_issued <= T_pred.
3. **Shadow Execution**: Operate with zero public automated dispatching.
4. **Independent Physical Deduplication**: All ground-truth disaster events collected from BRO incident logs, GSI Bhukosh, and SDMAs must be deduplicated spatio-temporally.
5. **No Mid-Flight Retraining**: Zero retraining or threshold alteration while surveillance is active.
