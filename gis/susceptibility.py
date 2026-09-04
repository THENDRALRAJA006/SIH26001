"""
gis/susceptibility.py
=====================
Compute a composite landslide susceptibility score from terrain features
and optional ancillary layers.

Method: Weighted linear combination (WLC) of normalised input factors.
        Weights are domain-informed defaults based on landslide literature
        for the Northeast India region.

        This is a STATIC susceptibility map — it does not change with
        weather/rainfall. Use RiskPipelineService for dynamic risk scores.

Output: susceptibility_score ∈ [0, 1] per pixel.

DISCLAIMER:
    This implementation uses publicly documented methods and illustrative
    default weights. It is NOT a validated operational hazard map.
    Operational use requires calibration against real landslide inventory
    data by a qualified geomorphologist (e.g. GSI, NESAC).

References:
    Biswas et al. (2019) — NER landslide susceptibility with AHP-GIS
    Sharma & Mahajan (2018) — Weight-of-evidence approach
"""

import numpy as np
from typing import Optional


# ── Default factor weights (must sum to 1.0) ─────────────────────────
# Based on domain literature for NER humid montane terrain.
# Modify via the `weights` parameter.

DEFAULT_WEIGHTS = {
    "slope":       0.35,   # strongest physical driver
    "curvature":   0.15,   # convergent slopes concentrate water
    "aspect":      0.10,   # SW/S aspects get more monsoon rainfall
    "tpi":         0.15,   # hollows and lower positions more susceptible
    "lithology":   0.15,   # soft/weathered rock more susceptible (if available)
    "landuse":     0.10,   # deforested areas more susceptible (if available)
}
SUSCEPTIBILITY_WEIGHTS = DEFAULT_WEIGHTS


# ── Normalisation helpers ─────────────────────────────────────────────

def _minmax_norm(
    arr: np.ndarray,
    clip_pct: float = 2.0,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """
    Robust min-max normalisation, clipping extreme percentiles or using explicit bounds.
    Returns array in [0, 1].
    """
    valid = arr[np.isfinite(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)
    lo = vmin if vmin is not None else float(np.percentile(valid, clip_pct))
    hi = vmax if vmax is not None else float(np.percentile(valid, 100.0 - clip_pct))
    if hi <= lo:
        return np.zeros_like(arr)
    normed = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return normed.astype(np.float32)


def _aspect_susceptibility(aspect_deg: np.ndarray) -> np.ndarray:
    """
    Convert aspect to susceptibility weight.
    SW/S/SE aspects (90°–270°) receive higher monsoon rainfall in NER.
    Returns [0, 1].
    """
    # cos(aspect - 180°) peaks at south-facing slopes
    rad = np.radians(aspect_deg)
    s   = 0.5 * (1.0 - np.cos(rad - np.pi))   # 0=N, 1=S
    s   = np.where(np.isnan(aspect_deg), 0.5, s)  # flat → neutral
    return s.astype(np.float32)


def _tpi_susceptibility(tpi: np.ndarray) -> np.ndarray:
    """
    Convert TPI to susceptibility.
    Negative TPI (hollows/valleys) → higher susceptibility.
    Positive TPI (ridges) → lower susceptibility.
    """
    # Invert: low TPI → high susceptibility
    tpi_norm  = _minmax_norm(tpi)
    return (1.0 - tpi_norm).astype(np.float32)


# ── Main function ─────────────────────────────────────────────────────

def compute_susceptibility(
    slope_deg:       np.ndarray,
    curvature:       np.ndarray,
    aspect_deg:      np.ndarray,
    tpi:             np.ndarray,
    lithology_score: Optional[np.ndarray] = None,
    landuse_score:   Optional[np.ndarray] = None,
    weights:         Optional[dict] = None,
) -> np.ndarray:
    """
    Compute composite landslide susceptibility score per pixel.

    Args:
        slope_deg       : slope in degrees (from terrain_features)
        curvature       : profile curvature (from terrain_features)
        aspect_deg      : aspect in degrees (from terrain_features)
        tpi             : topographic position index
        lithology_score : pre-normalised lithology susceptibility [0,1]
                          or None to use uniform 0.5
        landuse_score   : pre-normalised land-use susceptibility [0,1]
                          or None to use uniform 0.5
        weights         : override DEFAULT_WEIGHTS (must sum to 1.0)

    Returns:
        susceptibility : float32 array [0, 1], same shape as inputs.
                         Higher = more susceptible.
    """
    w = weights if weights is not None else DEFAULT_WEIGHTS.copy()

    # Validate weights
    total = sum(w.values())
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"Weights must sum to 1.0, got {total:.4f}")

    # Factor normalisation
    f_slope    = _minmax_norm(slope_deg, vmin=0.0, vmax=60.0)
    f_curv     = _minmax_norm(np.abs(curvature))      # magnitude of curvature
    f_aspect   = _aspect_susceptibility(aspect_deg)
    f_tpi      = _tpi_susceptibility(tpi)
    f_lith     = lithology_score if lithology_score is not None else np.full_like(f_slope, 0.5)
    f_landuse  = landuse_score   if landuse_score   is not None else np.full_like(f_slope, 0.5)

    score = (
        w["slope"]     * f_slope   +
        w["curvature"] * f_curv    +
        w["aspect"]    * f_aspect  +
        w["tpi"]       * f_tpi     +
        w["lithology"] * f_lith    +
        w["landuse"]   * f_landuse
    )

    score = np.clip(score, 0.0, 1.0).astype(np.float32)
    return score


def susceptibility_class(score: np.ndarray) -> np.ndarray:
    """
    Classify susceptibility score into 5 ordinal classes:
      1 = Very Low  (0.00 – 0.20)
      2 = Low       (0.20 – 0.40)
      3 = Moderate  (0.40 – 0.60)
      4 = High      (0.60 – 0.80)
      5 = Very High (0.80 – 1.00)

    Returns uint8 array with class labels.
    """
    bins   = [0.0, 0.20, 0.40, 0.60, 0.80, 1.01]
    labels = np.digitize(score, bins[1:], right=False).astype(np.uint8)
    labels = np.clip(labels + 1, 1, 5)
    return labels


def zone_mean_susceptibility(
    susceptibility: np.ndarray,
    zone_mask: np.ndarray,
) -> float:
    """
    Compute mean susceptibility score within a zone mask.

    Args:
        susceptibility : 2-D susceptibility array [0, 1].
        zone_mask      : bool array, True = pixels belonging to zone.

    Returns:
        float in [0, 1].
    """
    vals = susceptibility[zone_mask]
    if vals.size == 0:
        return 0.0
    return float(np.nanmean(vals))
