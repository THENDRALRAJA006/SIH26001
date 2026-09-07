# LAND-JEPA: Final Model Selection & Operational Promotion Decision

**Project**: LAND-JEPA | **Team**: ZAIX | **Problem**: SIH26001  
**Decision**: **RETAIN_BASELINE_V22**  

---

## Operational Promotion Checklist (24-Hour Horizon, FPR <= 5%)

| Operational Objective | Baseline v2.2 | Candidate Two-Stage | Delta / Relative Change | Criterion Met? |
|:---|:---:|:---:|:---:|:---:|
| **Physical Event Recall @ 24h** | 36.7% | 27.5% | -9.2% abs | **NO** |
| **Sliding-Window Recall @ FPR <= 5%** | 28.7% | 19.1% | -9.6% abs | **NO** |
| **False Negative Rate (Missed Disasters)** | 71.3% | 80.9% | +9.6% abs | **NO** |
| **Daily False Alarm Rate** | 0.0788 fa/day | 0.0578 fa/day | -26.6% | **YES** |
| **Median Advance Warning Lead Time** | 19.9 hours | 23.6 hours | +3.6 hours | **YES** |
| **Probability Calibration (Brier Score)**| 0.1509 | 0.1449 | -4.0% | **YES** |
| **Precision-Recall AUC (PR-AUC)** | 0.0644 | 0.0610 | -5.3% | **NO** |

---

### Official Promotion Statement & Scientific Verdict
Under the project's **Critical Rule**:
> *"DO NOT FORCE THE NEW MODEL TO WIN. If v2.2 is still better: keep v2.2. If the new model improves: promote it. If the improvement is small: do not exaggerate. Never fabricate events, forecasts, InSAR deformation, or metrics."*

While the **Two-Stage LAND-JEPA Candidate** achieved outstanding false-alarm suppression (**0.0578 vs 0.0788 fa/day, -26.6% reduction**) and longer advance warning lead time (**23.6h vs 19.9h, +3.7h**), its physical Event Recall at 24h (**27.5% vs 36.7%**) and FNR (**80.9% vs 71.3%**) did not beat the baseline.

Under Phase 20 requirements, promotion requires improvement across the operational objective (particularly event recall and missed disaster rates), and PR-AUC alone is not sufficient. Therefore, the candidate model is **NOT** promoted to production.

The production deployment strictly retains **`v2.2-PREDICTION-OPTIMIZED`** as the reigning operational model.
