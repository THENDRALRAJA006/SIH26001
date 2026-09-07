# MASTER DATASET V2 SPECIFICATION & CENSUS

**Project**: LAND-JEPA  
**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India (8 Monitored Highway Corridors)  
**Status**: Authoritative Dataset Census & Label Protocol Specification  

---

## 1. Dataset Census & Partition Overview

The complete reconciled dataset covers 8 critical highway transportation corridors across Northeast India over the 6-year period from 2011 to 2016.

| Split Name | Time Range | Total Windows | Confirmed Physical Events | Positive Window Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Training Set** | 2011-01-01 to 2014-12-31 | 11,540 | 86 | 0.82% |
| **Validation Set** | 2015-01-01 to 2015-12-31 | 2,754 | 53 | 1.96% |
| **Blind Test Set** | 2016-01-01 to 2016-10-14 | 2,249 | 19 (Primary) / 24 (Extended) | 0.80% / 1.02% |
| **Full Master Dataset** | **2011-01-01 to 2016-10-14** | **16,543** | **170 Deduplicated Events** | **1.03%** |

---

## 2. Reconciled Triple-Window Label Definition

To prevent label ambiguity and false negative penalties on scarred slopes, the Master Dataset strictly enforces the **Triple-Window Event-Aware Labeling Protocol**:

1. **Pre-Event Early Warning Window**:
   $$t \in [t_{\text{event}} - H, t_{\text{event}}]$$
   - **Label**: $y = 1$ (Positive).
   - **Operational Objective**: Early advance warning prior to failure release.
2. **Event Window**:
   $$t = t_{\text{event}}$$
   - **Label**: $y = 1$ (Positive).
   - **Operational Objective**: Direct failure detection.
3. **Post-Event Exclusion Window**:
   $$t \in [t_{\text{event}}, t_{\text{event}} + 48\text{h}]$$
   - **Label**: $y = -1$ (Masked / Excluded).
   - **Operational Rationale**: After a landslide releases, the scarred slope experiences ongoing debris shifting, emergency response, and localized altered drainage. Labeling this period as a clean negative ($y=0$) falsely penalizes predictive models for detecting real active ground instability. These samples are completely excluded from negative loss calculations.
4. **Unambiguous Negatives**:
   $$t \notin [t_{\text{event}} - H, t_{\text{event}} + 48\text{h}]$$
   - **Label**: $y = 0$ (Negative).
   - **Operational Objective**: True non-event background conditions.

---

## 3. The 6 Hard-Negative Challenge Subsets

Hard negatives are defined as difficult environmental conditions where triggers are severe but **no slope failure occurred**:

| Subset ID | Challenge Name | Exact Quantitative Criteria | Sample Count | Fraction of Negatives | Operational Failure Prevention Goal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `HN-01` | Extreme Rainfall Without Failure | $\text{acc\_24h} \ge 40.0\text{mm}, y=0$ | 714 | 4.4% | Suppresses false alarms during heavy downpours when slope friction holds. |
| `HN-02` | High Soil Moisture Saturation | $\theta_{\text{soil}} \ge 0.38\text{ m}^3/\text{m}^3, y=0$ | 10836 | 66.1% | Prevents false alarms when soil is near saturation but shear strength remains adequate. |
| `HN-03` | Highly Susceptible Steep Terrain | $\text{Slope} \ge 20.0^\circ, y=0$ | 8125 | 49.6% | Prevents static topography over-weighting in dry or low-rain conditions. |
| `HN-04` | Compound Severe Trigger | $\text{acc\_24h} \ge 40\text{mm} \land \text{Slope} \ge 20^\circ, y=0$ | 442 | 2.7% | Toughest challenge: steep slopes under downpours that successfully resisted failure. |
| `HN-05` | Prolonged Antecedent Infiltration | $\text{acc\_72h} \ge 100.0\text{mm}, y=0$ | 938 | 5.7% | Multi-day persistent monsoon rainfall without immediate mass wasting. |
| `HN-06` | Monsoon Active Dry Spell | $\text{Monsoon}=1 \land \text{acc\_24h} < 5\text{mm}, y=0$ | 1484 | 9.1% | Prevents seasonal baseline overfitting during monsoon dry spells. |

---

## 4. Feature Dimensionality (28 Engineered Channels)
The final feature space includes 28 physical, temporal, and spatial descriptors:
- **Rainfall Dynamics (7)**: acc_1h, acc_3h, acc_6h, acc_12h, acc_24h, acc_48h, acc_72h
- **Atmospheric Conditions (6)**: intensity_max_1h, dry_hours_streak, monsoon_flag, temperature_c, humidity_pct, wind_speed_ms
- **Soil & Geotechnical State (4)**: sm_volumetric, swi, pore_pressure_proxy, stability_indicator
- **Copernicus 30m DEM Terrain (6)**: elevation_m, slope_deg, aspect_deg, curvature, tpi, twi
- **NWP Forecast & Hydromechanical Proxies (5)**: forecast_rain_mean_mm, forecast_spread, forecast_uncertainty, antecedent_precipitation_index, factor_of_safety_proxy
