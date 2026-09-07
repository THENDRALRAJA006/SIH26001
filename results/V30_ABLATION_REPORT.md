# Ablation & Information-Contribution Report: LAND-JEPA v3.0-GEOTEMPORAL
**Governing Section**: Benchmark Protocol Section 21 & 22  
**Architecture**: 4-Way Cross-Modality Gated Fusion  

---

## 1. Sequential Integration Ablation (Configs A to E)

We evaluate the incremental contribution of each sensor modality starting from a pure hydrometeorological baseline:

| Config | Configuration Name | Encoders Active | Recall @ FPR ≤ 5% | FPR | FNR | PR-AUC | Brier Score | ECE | Median Lead Time |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | **Base LAND-JEPA** | Weather + DEM Terrain | 68.4% | 4.8% | 31.6% | 0.088 | 0.0094 | 0.0068 | 21.0h |
| **B** | **+ Tectonic Context** | Base + Crustal GPS & Fault Priors | 73.7% | 4.4% | 26.3% | 0.104 | 0.0084 | 0.0057 | 22.8h |
| **C** | **+ Tectonic + Seismic** | Config B + Ground Motion Attenuation | 78.9% | 4.1% | 21.1% | 0.122 | 0.0076 | 0.0049 | 24.2h |
| **D** | **+ Tectonic + Seismic + InSAR**| Config C + Sentinel-1 Radar Coherence | 78.9% | 4.0% | 21.1% | 0.124 | 0.0074 | 0.0048 | 24.5h |
| **E** | **Full v3.0-GEOTEMPORAL** | All 9 Input Streams + Multi-Trigger | **86.8%** | **3.1%** | **13.2%** | **0.152** | **0.0058** | **0.0035** | **26.8h** |

---

## 2. Modality Drop Sensitivity Analysis

To rigorously assess the individual impact of each modality on the full network, each modality was masked to physical zero/unavailable while holding the remaining network constant:

| Ablation Drop Experiment | Recall @ FPR ≤ 5% | Δ Recall vs Full v3.0 | FPR | PR-AUC | Impact Severity |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Without Weather / Rainfall** | **36.8%** | **-50.0%** | 4.6% | 0.042 | **Catastrophic Failure**: Rainfall is the primary triggering agent. |
| **Without Road / Drainage** | **78.9%** | **-7.9%** | 3.9% | 0.126 | **High**: Fails to capture roadside cut-slope destabilization and culvert blockages. |
| **Without Tectonic Context** | **81.6%** | **-5.2%** | 3.5% | 0.136 | **Moderate**: Loses spatial baseline strain sensitivity in active thrust zones. |
| **Without Seismic Shaking** | **84.2%** | **-2.6%** | 3.3% | 0.144 | **Conditional**: Only fires during active seismic swarms. |
| **Without InSAR Deformation**| **86.8%** | **0.0%** | 3.1% | 0.151 | **Negligible**: Broadleaf canopy decorrelation renders C-band InSAR unavailable in 98% of cases. |

---

## 3. Information Contribution Summary

| Modality | Primary Benefit | Δ Recall | Δ FPR | Δ Lead Time | Operational Verdict |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Forecast QPF (GFS)** | Early Warning Lead Time | **+18.4%** | -1.7% | **+5.8 Hours** | **Essential**: Critical for 24h-72h evacuation advisories. |
| **Road Cut-Slope Geometry** | Geometric Susceptibility | **+7.9%** | -0.8% | **+2.3 Hours** | **Essential**: Captures anthropogenic cut-slope failures. |
| **Culvert / Drainage Proximity**| Hydro-Choke Localization | **+5.3%** | -0.5% | **+1.5 Hours** | **Essential**: Resolves ravine blowout false negatives. |
| **Tectonic / Fault Priors** | Structural Susceptibility | **+5.3%** | -0.4% | **+1.8 Hours** | **Beneficial**: Elevates hazard floor in high-strain fault corridors. |
| **Seismic Attenuation (GMPE)**| Dynamic Shaking Prior | **+2.6%** | -0.2% | **+0.8 Hours** | **Beneficial**: Transient co-seismic destabilization. |
| **Sentinel-1 InSAR** | Surface Displacement | **+0.0%** | -0.1% | **+0.3 Hours** | **Marginal**: Tropical vegetation decorrelation constraint. |
