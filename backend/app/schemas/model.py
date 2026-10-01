"""
LAND-JEPA — Pydantic Schemas: Advanced Model Inference & Metadata

Schemas for the full LAND-JEPA multimodal risk model API.
Includes multi-horizon predictions, physics-state breakdown,
InSAR deformation tracking, leading factor attributions, and registry metadata.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class ModelHorizonHours(int, Enum):
    NOW = 0
    H6  = 6
    H12 = 12
    H24 = 24
    H48 = 48
    H72 = 72


class LeadingFactor(BaseModel):
    name: str = Field(description="Feature or factor name")
    importance: float = Field(description="Relative importance or attribution score")
    direction: str = Field(description="'increase_risk' or 'decrease_risk'")
    category: str = Field(default="temporal", description="temporal, terrain, insar, or physics")
    description: Optional[str] = Field(default=None, description="Human-readable factor summary")


class PhysicsStateDetail(BaseModel):
    soil_saturation: float = Field(description="Normalized soil saturation [0, 1]")
    pore_pressure_ratio: float = Field(description="Estimated pore water pressure ratio")
    factor_of_safety_proxy: float = Field(description="Physics-based stability indicator (lower = less stable)")
    is_critical: bool = Field(description="Whether physics threshold indicates imminent instability")


class InSARStatusDetail(BaseModel):
    available: bool = Field(description="Whether InSAR ground deformation was available for this zone")
    mean_velocity_mm_yr: Optional[float] = Field(default=None, description="Mean LOS velocity in mm/year")
    coherence: Optional[float] = Field(default=None, description="InSAR interferometric coherence [0, 1]")
    quality_flag: str = Field(default="nominal", description="nominal, low_coherence, or missing")


class ModelPredictionRequest(BaseModel):
    zone_id: str = Field(..., description="Zone identifier, e.g. DEMO-NER-001")
    horizon_hours: ModelHorizonHours = Field(
        default=ModelHorizonHours.NOW,
        description="Target prediction horizon (0, 24, or 48 hours)"
    )
    # Optional manual feature overrides (if omitted, fetched from internal features)
    temporal_sequence: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional (seq_len, 24) temporal feature sequence"
    )
    static_features: Optional[List[float]] = Field(
        default=None,
        description="Optional 6-element static terrain vector [slope, aspect, elevation, curvature, tpi, tri]"
    )
    insar_deformation: Optional[List[float]] = Field(
        default=None,
        description="Optional 2-element InSAR vector [los_velocity_mm_yr, coherence]"
    )
    soil_saturation: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional soil saturation override [0, 1]"
    )
    pore_pressure: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Optional pore pressure ratio override"
    )


class ModelPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    zone_id: str
    horizon_hours: int
    risk_score: float = Field(description="Risk probability for requested horizon [0, 1]")
    risk_level: str = Field(description="LOW, MEDIUM, or HIGH")
    confidence: float = Field(description="Confidence proxy [0, 1]")
    multi_horizon: Dict[str, float] = Field(
        description="Risk probabilities across all horizons: {'0h': p0, '24h': p24, '48h': p48}"
    )
    leading_factors: List[LeadingFactor] = Field(
        default_factory=list,
        description="Top leading risk factors and attributions"
    )
    physics_state: PhysicsStateDetail
    insar_status: InSARStatusDetail
    model_name: str
    model_version: str
    computed_at: datetime
    is_demo: bool
    disclaimer: str = (
        "This output is from the LAND-JEPA research model and requires expert validation. "
        "Do NOT use for emergency decisions without validation by GSI/NDMA."
    )


class CollapseMetricsDetail(BaseModel):
    variance: float
    cosine_similarity: float
    effective_rank: float
    collapsed: bool


class ModelStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_loaded: bool
    model_name: str
    model_version: str
    device: str
    parameter_count: int
    architecture: str
    fusion_type: str
    pretraining_collapse_metrics: Optional[CollapseMetricsDetail] = None
    registered_at: Optional[str] = None
    is_demo: bool


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    version: str
    status: str
    registered_at: str
    metrics: Dict[str, Any]
    checkpoint_path: str
    description: str


class ExplanationResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    zone_id: str
    horizon_hours: int
    risk_score: float
    risk_level: str
    leading_factors: List[LeadingFactor]
    physics_state: PhysicsStateDetail
    insar_status: InSARStatusDetail
    narrative: str = Field(description="Human-readable explanation of risk drivers")
    disclaimer: str
