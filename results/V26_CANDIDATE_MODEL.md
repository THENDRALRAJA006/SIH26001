# LAND-JEPA v2.6 CANDIDATE MODEL SPECIFICATION

**Selected Candidate**: `v2.6-ABLATION-NO-CLOUDBURST`  
**Selection Method**: Weighted Multi-Horizon Validation-Only (2015 Holdout, 5 Seeds, horizons 6/12/24/48/72h)  
**Date**: 2026-09-05T19:13:19.446956+00:00  
**Status**: Ready for Future Independent Prospective Shadow Testing  
**Selection Rationale**: Removing the convective divergence cloudburst gate improved weighted multi-horizon recall on 2015 validation (28.9% vs 27.5% for full v2.6). The cloudburst proxy adds informative signal at 6h/48h but introduces FPR pressure at 24h on the 2015 validation partition. Road-cut, seismic, and culvert-scour trigger gates are retained.  

---

## 1. Candidate Selection Statement

In strict adherence to protocol:
> *"Select exactly one v2.6 candidate using validation only. Then STOP. Do not evaluate v2.6 on the old 90-day test as a tuning loop. Wait for a new independent prospective evaluation period."*

The single champion model selected is **`v2.6 Ablation (No Cloudburst)`**.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        CANDIDATE MODEL VALIDATION METRICS                              │
├───────────────────────────────────────┬────────────────────────────────────────────────┤
│ Metric (24-Hour Horizon)              │ Validation Value (Mean over 5 Seeds)           │
├───────────────────────────────────────┼────────────────────────────────────────────────┤
│ Event Recall @ WARNING (FPR <= 5%)    │ 28.9%                                          │
│ False Negative Rate (FNR)             │ 71.1%                                          │
│ False Positive Rate (FPR)             │ 2.61%                                          │
│ Precision                             │ 0.1444                                         │
│ PR-AUC                                │ 0.1153                                         │
│ Brier Calibration Loss                │ 0.0119                                         │
│ Expected Calibration Error (ECE)      │ 0.0000                                         │
│ False Alarms / Corridor-Day           │ 0.0250 (1 alert / 40.0 days)                    │
│ Median Advance Warning Lead Time      │ 24.7 hours                                    │
└───────────────────────────────────────┴────────────────────────────────────────────────┘
```

---

## 2. Locked Operating Configuration

- **Model Version**: `v2.6-ABLATION-NO-CLOUDBURST`
- **Feature Count**: 86 physical variables across 8 trigger families
- **Isotonic Calibration**: Fitted on validation probabilities
- **Locked Thresholds**:
  - `WATCH` (FPR <= 10%): `0.0660`
  - `WARNING` (FPR <= 5%): `0.0929`
  - `CRITICAL` (FPR <= 1%): `0.2444`
- **Quarantine Guarantee**: Zero parameters, weights, or thresholds were adjusted against the 19 prospective events of the 2026 test.
