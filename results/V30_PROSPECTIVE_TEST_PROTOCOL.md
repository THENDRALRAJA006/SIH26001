# Prospective Test Protocol & Promotion Mandate: LAND-JEPA v3.0-GEOTEMPORAL
**Governing Section**: Benchmark Protocol Section 30, 31, 32, 33, 40  
**Candidate Model**: `LAND-JEPA v3.0-GEOTEMPORAL`  
**Protocol Status**: **LOCKED & IMMUTABLE**  

---

## 1. Mandatory Model Freeze

Following model selection and multi-season validation, the candidate model `LAND-JEPA v3.0-GEOTEMPORAL` is hereby **FROZEN**:
- **Weights & Architecture**: Frozen (`v3.0-GEOTEMPORAL`).
- **Feature Schema**: Frozen (`v3.0-geotemporal-x102`, 102 continuous input channels).
- **Thresholds**: Frozen (`WATCH = 0.3000`, `WARNING = 0.5500`, `CRITICAL = 0.8000`).
- **Calibration Parameters**: Frozen (Isotonic regression fitted on 2015 validation partition).

> [!CAUTION]
> Under no circumstances may model weights, thresholds, or calibration tables be updated, tuned, or retrained after seeing prospective event outcomes.

---

## 2. Prospective Surveillance Period & Isolation Rules

- **Surveillance Window**: September 7, 2026 to December 6, 2026 (90-day prospective surveillance).
- **Quarantined Events**: All 19 historical events from `EV-PROSPECTIVE-2026-01` through `EV-PROSPECTIVE-2026-19` remain strictly isolated in quarantine to prevent benchmark leakage.
- **Pre-Event Persistence**: Every prospective prediction must be written to `results/predictions_ledger.jsonl` and timestamped **BEFORE** ground-truth event outcomes are reported by field teams.
- **Identical Input Protocol**: During head-to-head surveillance, all three models (`v2.5`, `v2.6.1`, and `v3.0`) run in shadow mode on the exact same live API observations, GFS forecast issuances, and corridor coordinates.

---

## 3. The 15-Event Promotion Mandate (Section 32 & 33)

To prevent premature promotion driven by small-sample statistical noise or temporary weather anomalies:
- **Sample Size Requirement**: A minimum of **$N \ge 15$ NEW, independently verified, satellite- or field-confirmed landslide events** must occur within the prospective monitoring window before a binding promotion decision can be rendered.
- **Mandatory Promotion Criteria**:
  1. **Higher Event Recall**: Event Recall @ FPR $\le$ 5% must exceed production champion `v2.5` ($78.9\%$).
  2. **FPR Constraint**: False Positive Rate must strictly satisfy $\text{FPR} \le 5.0\%$.
  3. **False Alarm Ceiling**: False alarms must not exceed $0.10\text{ alerts/day}$ per corridor.
  4. **Warning Lead Time**: Median lead time must satisfy $T_{\text{lead}} \ge 24.0\text{ Hours}$.
  5. **Calibration Integrity**: Brier score $\le 0.0080$ and $\text{ECE} \le 0.0050$.

---

## 4. Current Prospective Status & Verdict

In strict adherence to Section 40:

$$\textbf{PROSPECTIVE STATUS} = \textbf{INSUFFICIENT\_EVIDENCE}$$

- **Reason**: The prospective surveillance window has commenced, but the mandatory threshold of $N \ge 15$ new independent verified events has not yet been accrued.
- **Governing Precaution**: Historical validation superiority (86.8% vs 78.9%) is documented as a scientific milestone, but `v2.5-TRIGGER-AWARE-CHAMPION` remains the active production deployment until prospective criteria are fulfilled.

```
+-------------------------------------------------------------------------------+
|                        OFFICIAL STATUS DECLARATIONS                           |
+-------------------------------------------------------------------------------+
| BEST HISTORICAL VALIDATION MODEL:  LAND-JEPA v3.0-GEOTEMPORAL                 |
| BEST OPERATIONAL CANDIDATE:       LAND-JEPA v3.0-GEOTEMPORAL                 |
| PROSPECTIVE STATUS:               INSUFFICIENT_EVIDENCE                       |
| EVIDENCE QUALITY:                 HIGH (Audited Historical / Active Shadow)   |
+-------------------------------------------------------------------------------+
```
