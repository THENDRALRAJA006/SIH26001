# FINAL FORECAST-AWARE MODEL CARD
**Model**: Fused LAND-JEPA Forecast-Aware Early Warning Model v2.0  
**Team**: ZAIX | **Problem**: SIH26001 | **Region**: Northeast India  
**Release Date**: 2026-09-05T12:11:08Z

---

## Model Architecture
* **Temporal Stream**: 4-block Causal Temporal Convolutional Network (TCN) with dilated convolutions (dilation factors 1, 2, 4, 8) and LayerNorm.
* **Geomorphic Stream**: Non-linear multi-layer perceptron encoding Copernicus 30m DEM derivatives (slope, aspect, profile curvature, plan curvature, TPI, TWI).
* **Fusion Layer**: Gated cross-modal projection yielding a 128-dimensional latent slope representation.
* **Specialized Hazard Heads**: 5 independent hazard prediction heads branching for 6h, 12h, 24h, 48h, and 72h horizons.
* **Uncertainty Features**: Integrated forward-looking numerical weather forecast mean, ensemble spread, lead-time confidence, and expected 1-sigma error.

## Operating Constraints
* **Primary Operating Threshold**: Calibrated on validation set at FPR <= 5% (threshold = 0.285).
* **Probability Interpretation**: All outputs represent probabilistic hazard levels (P in [0, 1]), NOT binary certainties.
* **InSAR Status**: Disabled (insar_available=False) due to vegetative decorrelation.
* **Quantum Status**: VQC is excluded from production.
