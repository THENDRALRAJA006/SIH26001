# FORECAST GAP ANALYSIS: RETROSPECTIVE VS. PROSPECTIVE EARLY WARNING
**Project**: LAND-JEPA | **Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India (NER)  
**Date**: September 2026 | **Scope**: Systematic Diagnostic of Early-Warning Performance Degradation

---

## 1. Executive Summary

During Phase 1 of the Final Improvement Cycle, we conducted an in-depth audit comparing LAND-JEPA's retrospective benchmark against its prospective forecast-backtest results. 

In retrospective evaluation (where clean ERA5-Land reanalysis is available through the event window), models demonstrated strong discriminative capacity (e.g., Fused LAND-JEPA achieved PR-AUC $\approx 0.1285$ on 2016 hold-out testing). However, in prospective forecast backtesting (where only pre-event antecedent context is known and future rainfall must be predicted via simulated NWP Quantitative Precipitation Forecasts), PR-AUC ranged from $0.0197$ to $0.0731$, and recall at $\text{FPR} \le 5\%$ varied between $18.2\%$ and $47.6\%$.

This document outlines the **10 structural causes** of this performance divergence and defines the concrete algorithmic countermeasures implemented in this improvement cycle.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 THE FORECAST GAP MECHANISM                             │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ RETROSPECTIVE REANALYSIS (Clean, Uniform, Latent Geotechnical State Observable)        │
│   Antecedent Rain (t <= T) + Actual Event Rain (T to T+H) + Reanalysis Soil Moisture   │
│   ──► Model operates on precise physical mass-balance                                  │
│   ──► Result: High PR-AUC (~0.1285), Clean Probability Separation                      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ PROSPECTIVE FORECASTING (Noisy, Phase-Shifted, Geotechnical State Unobservable)        │
│   Antecedent Rain (t <= T) + Forecast Rain QPF (T to T+H) + Antecedent Soil State Only │
│   ──► NWP timing/intensity errors displace geotechnical failure boundary               │
│   ──► Future soil saturation, pore-pressure rise, and SWI cannot be directly measured  │
│   ──► Result: Moderate PR-AUC (0.02 - 0.07), Increased False Alarm Rate in Dry Fronts │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Audit of the 10 Structural Failure Modes

### 2.1 Reanalysis-to-Forecast Distribution Shift
* **Diagnostic Finding**: ERA5-Land reanalysis uses 4D-Var data assimilation to reconcile satellite radiances and surface rain gauges, creating spatially smoothed, physically consistent hourly fields. In contrast, operational NWP forecasts (e.g., NCUM, GFS, ECMWF IFS) exhibit localized convective over-prediction, spatial displacement of monsoonal troughs across the Patkai/Himalayan foothills, and high-frequency noise.
* **Impact**: A model trained on smooth reanalysis precipitation learns tight decision thresholds ($>35\text{ mm/24h}$) that misfire when applied to noisy forecast rain fields.
* **Countermeasure**: Expose the model during training to realistic forecast error sampling rather than clean historical reanalysis.

### 2.2 Forecast Rainfall Uncertainty Scaling with Horizon
* **Diagnostic Finding**: In the initial prospective backtest, a constant Gaussian noise factor ($\sigma = 30\%$) was applied uniformly across all lead times. In reality, NWP forecast error grows non-linearly with lead time:
  - $H=6\text{h}$: Correlation $r \approx 0.88$, mean absolute percentage error $\approx 15\%$
  - $H=12\text{h}$: $r \approx 0.81$, $\text{MAPE} \approx 20\%$
  - $H=24\text{h}$: $r \approx 0.72$, $\text{MAPE} \approx 30\%$
  - $H=48\text{h}$: $r \approx 0.55$, $\text{MAPE} \approx 45\%$
  - $H=72\text{h}$: $r \approx 0.40$, $\text{MAPE} \approx 60\%$
* **Impact**: Short horizons ($6\text{h}$) were artificially penalized with excessive noise, while extended horizons ($72\text{h}$) were artificially granted too much predictive certainty.
* **Countermeasure**: Parameterize forecast uncertainty as a horizon-conditioned and season-conditioned error distribution: $\sigma(H, \text{season})$.

### 2.3 Feature Availability Mismatch
* **Diagnostic Finding**: Retrospective feature extractors computed physics proxies (Soil Water Index, pore-pressure proxy, geotechnical factor of safety proxy) using variables across the entire event window. In prospective forecasting at time $T$, future soil moisture and pore pressure are unobservable; only antecedent soil moisture at $t \le T$ and forecast precipitation for $[T, T+H]$ exist.
* **Impact**: Models heavily reliant on concurrent soil saturation suffered an information collapse when transitioning to forecast mode.
* **Countermeasure**: Enforce strict feature boundary separation. Implement forward hydrologic projection: estimate future soil saturation $\widehat{\text{SWI}}_{T+H}$ as a function of antecedent $\text{SWI}_T$ and forecast cumulative rainfall $\sum_{t=T}^{T+H} \text{QPF}_t$.

### 2.4 Target-Label Imbalance & Sparsity
* **Diagnostic Finding**: Landslides in Northeast India are localized, episodic mass movements. In the 2016 test split (2,259 sliding windows), positive events were severely sparse:
  - $H=6\text{h}$: 11 positive windows ($0.48\%$)
  - $H=12\text{h}$: 14 positive windows ($0.62\%$)
  - $H=24\text{h}$: 18 positive windows ($0.80\%$)
  - $H=48\text{h}$: 32 positive windows ($1.42\%$)
  - $H=72\text{h}$: 46 positive windows ($2.04\%$)
* **Impact**: At $H=6\text{h}$, a single false positive or false negative shifts PR-AUC and recall drastically, creating wide bootstrap confidence bounds.
* **Countermeasure**: Implement balanced class re-weighting ($\text{scale\_pos\_weight}$) and multi-horizon multi-task training so heads with sparse labels share representations with broader-horizon heads.

### 2.5 Horizon Mismatch & Physical Failure Mechanisms
* **Diagnostic Finding**: Single-head architectures attempted to predict multiple horizons with the same temporal dynamics. However, physical landslide triggers are horizon-dependent:
  - **Short Horizons ($6\text{h}\text{--}12\text{h}$)**: Triggered by flash convective rainfall intensity ($I_{\text{max}} > 25\text{ mm/h}$) causing rapid shallow debris flows and mudslides.
  - **Extended Horizons ($48\text{h}\text{--}72\text{h}$)**: Triggered by cumulative monsoonal infiltration ($>200\text{ mm/72h}$) raising the phreatic water table and triggering deep-seated rotational slides.
* **Impact**: A single scalar output cannot capture both high-intensity flash triggers and long-duration saturation triggers.
* **Countermeasure**: Introduce 5 specialized horizon prediction heads ($6\text{h}, 12\text{h}, 24\text{h}, 48\text{h}, 72\text{h}$) branching from the shared JEPA latent representation.

### 2.6 Threshold Instability under False-Alarm Constraints
* **Diagnostic Finding**: Disaster management protocols require an operating false-alarm budget $\text{FPR} \le 5\%$ (or $\le 1\%$). Under extreme class imbalance, the $95\text{th}$ percentile threshold falls in the steep tail of the sigmoid score distribution. Minute perturbations in input rainfall shift scores across the threshold, causing large recall oscillations.
* **Impact**: Recall dropped from $45.5\%$ (in persistence) to $18.2\%$ (in prospective fused model) at $H=6\text{h}$ because the threshold was overly conservative.
* **Countermeasure**: Fit operating thresholds strictly on the validation set using smooth percentile interpolation and report performance across three operational budgets: $\text{FPR} \le 1\%$, $\le 5\%$, and $\le 10\%$.

### 2.7 Calibration Drift in Probability Tails
* **Diagnostic Finding**: Raw uncalibrated neural network outputs produce clustered probabilities near 0 and 1, with poor reliability in the critical decision range $[0.10, 0.45]$. Brier scores degraded from $0.013$ at $24\text{h}$ to $0.156$ at $72\text{h}$.
* **Impact**: Emergency managers cannot interpret raw model probabilities as genuine frequentist event likelihoods.
* **Countermeasure**: Apply validation-only Temperature Scaling and isotonic regression to guarantee well-calibrated reliability curves on the blind test split.

### 2.8 Hard Negative Contamination
* **Diagnostic Finding**: Monsoonal storms in Northeast India regularly dump $>50\text{ mm}$ of rain in 24 hours over stable, well-vegetated, crystalline bedrock slopes (e.g., Shillong Plateau granite) without initiating failure. Models trained solely on random negatives learn a naive rule: "high rain = high risk", generating false alarms during every monsoonal depression.
* **Impact**: High false alarm rates during the peak monsoon months (June–August).
* **Countermeasure**: Explicitly mine 1,123 hard negative windows (steep slope + high rainfall + high antecedent SWI + zero landslide) and include them in both pretraining contrastive contexts and classifier fine-tuning.

### 2.9 Event Clustering across Sliding Windows
* **Diagnostic Finding**: A physical landslide event lasting 6 hours is sampled multiple times by a 24-hour sliding window stride, generating 3–5 contiguous positive windows. Evaluating purely at the window level double-counts extended failures while masking missed point-events.
* **Impact**: Artificially distorts precision and recall metrics away from operational disaster response units.
* **Countermeasure**: Introduce event-based evaluation: group consecutive alerts by physical event ID and calculate event recall, event precision, and first-alert lead time.

### 2.10 Train/Test Information Mismatch
* **Diagnostic Finding**: Standard models were trained exclusively on clean reanalysis sequences and tested on noisy forecast-augmented sequences. This represents classic covariate shift ($P_{\text{train}}(X) \ne P_{\text{test}}(X)$).
* **Impact**: The encoder's internal representations experienced out-of-distribution feature combinations at test time.
* **Countermeasure**: Forecast-aware training: inject stochastic forecast error representations during training epochs, ensuring the encoder learns invariant representations that tolerate forecast uncertainty.

---

## 3. Systematic Action Plan

| Phase | Problem Addressed | Algorithmic Solution | Verification Metric |
|:---|:---|:---|:---|
| **Phase 2 & 3** | Distribution shift & leakage | Dedicated `ForecastProvider` with strict schema & timestamp assertions | $\max(t_{\text{input}}) \le t_{\text{pred}}$ assertion |
| **Phase 4 & 5** | Constant noise & unmodeled error | Calibrated $\sigma(H, \text{season})$ + 5 uncertainty features | Error residual distribution matching NWP skill |
| **Phase 6** | Feature availability mismatch | Rich antecedent features (API, rainfall anomaly, saturation proxy) | Feature correlation with failure onset |
| **Phase 7 & 8** | Horizon mismatch | Shared JEPA encoder + 5 horizon-specific heads ($6\text{h}\text{--}72\text{h}$) | Multi-horizon PR-AUC & Recall |
| **Phase 9** | False alarms on monsoonal rain | Hard negative mining (high rain + steep slope + $y=0$) | Hard negative rejection rate |
| **Phase 10** | Window clustering artifact | Event-level evaluator (first warning time, event recall) | Event recall & lead-time distribution |
| **Phase 11 & 12** | Threshold instability & poor Brier | Validation-only threshold optimization & temperature scaling | Brier score & Expected Calibration Error |
| **Phase 13 & 14** | Unfair comparisons | 8-model fair leaderboard + Published-Methodology baseline | Controlled benchmark table |

---

*This audit forms the scientific foundation for all subsequent phases of the Forecast-Aware Improvement Cycle.*
