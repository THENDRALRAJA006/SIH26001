# LAND-JEPA — Known Limitations

> This document is required by the project specification. All limitations must be
> documented honestly. This system is a research and demonstration platform for the
> Smart India Hackathon. It is not a certified operational early-warning system.

---

## 1. This System Does Not Replace Authorities

LAND-JEPA is a **decision-support tool**, not a replacement for:
- Geological Survey of India (GSI)
- India Meteorological Department (IMD)
- National Disaster Management Authority (NDMA)
- State Disaster Management Authorities (SDMAs)
- Central Water Commission (CWC)

All risk outputs require validation by domain experts before operational use.
No evacuation, response, or policy decision should be made based on this system alone.

---

## 2. AI Model Limitations

### 2.1 No Guaranteed Accuracy or Warning Time

- Landslide occurrence is a complex geophysical process. No AI model can achieve
  100% recall or guarantee a specific warning time.
- The system may miss events (false negatives), especially rare or unusual events.
- The system may generate false alarms (false positives), especially during extreme
  rainfall events that do not result in landslides.

### 2.2 Label Scarcity

- Historical landslide event databases for NER are sparse and incomplete.
- Many events are undocumented, especially in remote areas.
- The model trained on incomplete labels will systematically underestimate risk in
  under-documented zones.

### 2.3 Label Uncertainty

- Many historical events have only date precision (not exact hour). This introduces
  uncertainty in the training labels.
- The `date_precision` field documents this uncertainty. The label builder
  accounts for it conservatively but cannot eliminate it.

### 2.4 Spatial Autocorrelation

- Adjacent zones have correlated environmental conditions. The current experiment
  design does not fully correct for spatial autocorrelation in the test set.
- Reported metrics may be optimistic for spatially similar zones.

### 2.5 SHAP ≠ Causation

- SHAP values measure model feature contributions (correlation-based).
- They do NOT establish physical causal relationships.
- High SHAP value for slope does not mean slope "caused" the landslide.
- This disclaimer is displayed on every explanation output.

### 2.6 JEPA is Not Invented Here

- The JEPA (Joint Embedding Predictive Architecture) framework was introduced by
  Meta AI / LeCun et al. This project applies JEPA to geophysical time series.
- The research contribution of this project is the domain application and
  label-efficiency evaluation, not the JEPA framework itself.

### 2.7 Generalization

- Models are trained and evaluated on NER data. Generalization to other geographic
  regions has not been tested.

---

## 3. Data Limitations

### 3.1 Rainfall Data

- IMD gridded rainfall is at 0.25° resolution (≈25 km grid). This may miss
  localized intense rainfall that triggers slope-specific failures.
- Satellite-derived rainfall (CHIRPS, GPM) has additional uncertainty in complex
  terrain due to orographic effects.

### 3.2 Soil Moisture

- Remote-sensing soil moisture (ESA CCI, SMAP) measures only the surface layer
  (top 5 cm). Root-zone and deep soil moisture, which are more directly relevant
  to landslide mechanics, are estimated (not directly observed).
- The physics proxy (SWI) is a simplified estimate with documented assumptions.

### 3.3 InSAR

- InSAR observations are not hourly. Revisit is 6–12 days.
- InSAR measures line-of-sight displacement, not absolute vertical movement.
- Dense vegetation in NER reduces InSAR coherence significantly.
- InSAR integration is optional and the system operates without it.
- Raw SAR processing is not performed by this platform.

### 3.4 Terrain (DEM)

- SRTM 30m is the primary DEM. In NER's complex terrain, 30m may be insufficient
  for capturing small-scale slope features that control individual landslide paths.
- Terrain is treated as static. Post-earthquake or post-slide terrain changes
  are not automatically updated.

### 3.5 Historical Events

- Historical landslide event databases are known to be incomplete.
- Under-reporting is common in remote areas and pre-2000 records.
- The absence of a recorded event does NOT mean a zone is safe.

---

## 4. Physics Module Limitations

- The physics-aware state estimator (SWI / pore-pressure proxy) uses simplified
  Green-Ampt or empirical infiltration with:
  - Uniform soil properties per zone (spatial heterogeneity ignored)
  - No lateral subsurface flow
  - No evapotranspiration modeling
  - No bedrock depth modeling
- This is NOT a geotechnical stability analysis.
- This module adds interpretability and may improve calibration; it is not a
  substitute for site-specific geotechnical surveys.

---

## 5. Operational Limitations

### 5.1 Not a Certified System

- LAND-JEPA has not been validated against certified operational early-warning
  standards (IMD, WMO, etc.)
- Before any operational deployment, independent validation against documented
  historical events and expert review are mandatory.

### 5.2 Alert System

- The alert system is in DEMO MODE during development.
- Real SMS, push notifications, or emergency broadcasts are NOT sent during development.
- Production alert delivery requires integration with official communication channels
  (NDMA's CAP-based systems, state DMA communication networks), which is beyond
  the scope of this hackathon project.

### 5.3 Offline Mode

- Offline mode only buffers data submission. Risk predictions are not computed offline.
- Field officers in offline areas cannot receive updated risk predictions until
  connectivity is restored.

### 5.4 Performance

- Model inference latency has not been optimized for edge/low-resource devices.
- GIS tile loading performance depends on server hardware and data volume.

---

## 6. Demo Data

- All demo/synthetic data is clearly labelled.
- Any performance metrics or visualizations using demo data are labelled
  "DEMO DATA — NOT SCIENTIFIC RESULTS" and are for software integration testing only.
- Demo scenarios do not represent any real past or future landslide event.

---

## 7. Ethical Limitations

### 7.1 False Negatives Risk

- Missing a landslide event (false negative) can have life-safety consequences.
- The system's recall performance at different label fractions is explicitly reported.
- Users and authorities must understand that AI-based risk monitoring supplements,
  not replaces, community knowledge and expert judgment.

### 7.2 False Positives and Alert Fatigue

- Frequent false alarms can cause alert fatigue, reducing community trust.
- The configurable threshold system allows operators to tune precision/recall tradeoff.
- Threshold configuration must be done by domain experts, not algorithmic optimization alone.

### 7.3 Equity

- Risk predictions rely on data coverage, which may be unequal across the region.
- Zones with fewer sensors or data sources may have lower confidence predictions.
- Confidence scores must be displayed alongside risk scores to communicate uncertainty.
- Low-confidence zones should receive more human review, not less.
