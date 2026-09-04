"""
gis/insar_adapter.py
====================
Optional InSAR (Interferometric SAR) surface deformation adapter.

InSAR provides mm-scale surface displacement measurements that can
detect slope creep before visible landslide failure.

IMPORTANT CONSTRAINTS (from PROJECT_SPEC.md):
  - InSAR integration is OPTIONAL and must NOT be mandatory.
  - System must function fully without InSAR data.
  - When InSAR is unavailable, deformation_trend_mm_30d = NaN.
  - Never fabricate deformation values.

Supported data sources (when available):
  - Sentinel-1 ascending / descending line-of-sight (LOS) displacement
  - ALOS-2 PALSAR-2 coherence
  - Custom CSV/NetCDF from processing chains (e.g. SNAP, StaMPS, MintPy)

All values in this module are LINE-OF-SIGHT displacement in mm.
Conversion to vertical/horizontal requires incidence angle — not done here.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import warnings


INSAR_DISABLED_MSG = (
    "InSAR integration is disabled (INSAR_ENABLED=False in config). "
    "System will operate without deformation features."
)


# ── Data classes ──────────────────────────────────────────────────────

@dataclass
class InSARMeasurement:
    """
    A single InSAR LOS displacement observation for a zone.

    All values in millimetres (mm).
    Negative = movement away from satellite (typically subsidence/downslope).
    Positive = movement toward satellite (typically uplift).
    """
    zone_id:              str
    timestamp:            str          # ISO-8601
    los_displacement_mm:  float        # LOS displacement (mm)
    coherence:            float        # [0, 1] — 1 = perfect coherence
    incidence_angle_deg:  float        # sensor incidence angle
    track:                str = "asc"  # "asc" or "desc" (Sentinel-1 pass)
    is_demo:              bool = True  # always True in demo mode

    def is_reliable(self, coherence_threshold: float = 0.3) -> bool:
        """Return True if coherence is above threshold."""
        return self.coherence >= coherence_threshold


@dataclass
class InSARZoneSummary:
    """
    Aggregated InSAR statistics for a zone over a time window.
    Used as input feature to the risk models.
    """
    zone_id:                  str
    n_observations:           int
    deformation_trend_mm_30d: Optional[float]   # linear trend (mm/30d)
    max_los_mm:               Optional[float]   # max absolute displacement
    mean_coherence:           Optional[float]   # average coherence
    data_available:           bool = False
    is_demo:                  bool = True

    @property
    def feature_value(self) -> float:
        """
        Return the deformation_trend_mm_30d value for use in ML features.
        Returns NaN if data unavailable or unreliable.
        """
        if not self.data_available or self.deformation_trend_mm_30d is None:
            return float("nan")
        return self.deformation_trend_mm_30d


# ── Adapter ───────────────────────────────────────────────────────────

class InSARAdapter:
    """
    Adapter that loads and parses InSAR displacement data.

    When disabled (enabled=False), all methods return empty/NaN results
    without raising exceptions — system degrades gracefully.

    Usage:
        adapter = InSARAdapter(enabled=False)  # disabled (default)
        summary = adapter.get_zone_summary("DEMO-NER-001", days=30)
        feat = summary.feature_value  # → nan
    """

    def __init__(self, enabled: bool = False, data_dir: Optional[str] = None):
        self.enabled  = enabled
        self.data_dir = data_dir
        if not enabled:
            warnings.warn(INSAR_DISABLED_MSG, UserWarning, stacklevel=2)

    def get_zone_summary(
        self,
        zone_id: str,
        days:    int = 30,
    ) -> InSARZoneSummary:
        """
        Return InSAR summary for a zone over the last `days` days.
        If disabled or data unavailable, returns a summary with
        data_available=False and feature_value=NaN.

        Args:
            zone_id : zone identifier string.
            days    : lookback window in days.

        Returns:
            InSARZoneSummary with data_available flag.
        """
        if not self.enabled:
            return InSARZoneSummary(
                zone_id=zone_id,
                n_observations=0,
                deformation_trend_mm_30d=None,
                max_los_mm=None,
                mean_coherence=None,
                data_available=False,
                is_demo=True,
            )

        # Real data loading would happen here:
        # measurements = self._load_from_file(zone_id, days)
        raise NotImplementedError(
            "Real InSAR data loading requires a configured data_dir "
            "with processed Sentinel-1 or ALOS-2 displacement products. "
            "Set INSAR_ENABLED=False to run without InSAR."
        )

    def compute_trend(
        self,
        times_days:       np.ndarray,
        displacements_mm: np.ndarray,
        coherences:       Optional[np.ndarray] = None,
        coherence_threshold: float = 0.3,
    ) -> Optional[float]:
        """
        Compute linear deformation trend (mm/30d) from time series.
        Low-coherence observations are excluded before fitting.

        Args:
            times_days        : array of observation times (days since epoch).
            displacements_mm  : array of LOS displacement values (mm).
            coherences        : optional coherence weights [0,1].
            coherence_threshold: minimum coherence to include.

        Returns:
            trend_mm_per_30d : float, or None if insufficient data.
        """
        if len(times_days) < 3:
            return None

        mask = np.ones(len(times_days), dtype=bool)
        if coherences is not None:
            mask &= coherences >= coherence_threshold

        t  = times_days[mask]
        d  = displacements_mm[mask]

        if len(t) < 3:
            return None

        # Linear least squares
        coeffs = np.polyfit(t, d, 1)
        trend_per_day = coeffs[0]
        return float(trend_per_day * 30.0)   # normalise to mm/30d

    @staticmethod
    def displacement_to_vertical(
        los_mm: float,
        incidence_angle_deg: float,
    ) -> float:
        """
        Approximate LOS → vertical displacement conversion.
        Assumes pure vertical motion (ignores horizontal component).

        vertical_mm ≈ LOS_mm / cos(incidence_angle)

        Args:
            los_mm              : LOS displacement in mm.
            incidence_angle_deg : sensor incidence angle in degrees.

        Returns:
            vertical_mm : approximate vertical displacement in mm.
        """
        cos_inc = np.cos(np.radians(incidence_angle_deg))
        if abs(cos_inc) < 1e-6:
            return float("nan")
        return float(los_mm / cos_inc)
