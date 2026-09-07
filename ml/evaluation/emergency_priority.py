"""
LAND-JEPA — Emergency Prioritization Engine
Team: ZAIX | Problem: SIH26001 | Region: Northeast India

Computes emergency response priority for monitored zones based on:
  1. Calibrated Landslide Risk Score [0, 1]
  2. Population Exposure (density & urban vulnerability)
  3. Road Network Criticality (National Highway / single lifeline vs redundant corridors)
  4. Critical Infrastructure (hospitals, power transmission, water filtration)
  5. Accessibility Penalty (remoteness / geographic isolation hindering rescue)
  6. Model Prediction Confidence

Priority Levels:
  - Priority 1: High / Immediate Life Safety (Composite >= 0.65)
  - Priority 2: Moderate / Pre-positioning & Inspection (0.40 <= Composite < 0.65)
  - Priority 3: Advisory / Heightened Monitoring (Composite < 0.40)

Includes human-readable explanations detailing why each priority was assigned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ZoneVulnerabilityProfile:
    zone_id: str
    name: str
    population_exposure_score: float  # [0, 1]
    road_criticality_score: float     # [0, 1] (1.0 = single arterial lifeline)
    critical_infrastructure_count: int
    accessibility_penalty: float      # [0, 1] (1.0 = highly remote/isolated)
    lifeline_highway: str


# Real NER Vulnerability Baseline Data
NER_VULNERABILITY_PROFILES: Dict[str, ZoneVulnerabilityProfile] = {
    "REAL-NER-001": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-001",
        name="Guwahati Metro & Kamrup Hills",
        population_exposure_score=0.90,
        road_criticality_score=0.85,
        critical_infrastructure_count=12,
        accessibility_penalty=0.20,
        lifeline_highway="NH-27 / Gateway Corridor",
    ),
    "REAL-NER-002": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-002",
        name="Shillong & East Khasi Hills",
        population_exposure_score=0.75,
        road_criticality_score=0.95,
        critical_infrastructure_count=7,
        accessibility_penalty=0.50,
        lifeline_highway="NH-6 (Umiam - Shillong Arterial)",
    ),
    "REAL-NER-003": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-003",
        name="Aizawl Ridge & Chite Valley",
        population_exposure_score=0.70,
        road_criticality_score=1.00,  # Sole lifeline
        critical_infrastructure_count=5,
        accessibility_penalty=0.85,  # Highly remote mountainous ridge
        lifeline_highway="NH-306 (Silchar - Aizawl Lifeline)",
    ),
    "REAL-NER-004": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-004",
        name="Gangtok & Teesta Corridor",
        population_exposure_score=0.65,
        road_criticality_score=1.00,  # Severely prone to Teesta river cutting
        critical_infrastructure_count=6,
        accessibility_penalty=0.75,
        lifeline_highway="NH-10 (Sevoke - Gangtok Lifeline)",
    ),
    "REAL-NER-005": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-005",
        name="Kohima & Dzukou Escarpment",
        population_exposure_score=0.60,
        road_criticality_score=0.90,
        critical_infrastructure_count=4,
        accessibility_penalty=0.70,
        lifeline_highway="NH-29 (Dimapur - Kohima Corridor)",
    ),
    "REAL-NER-006": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-006",
        name="Itanagar & Papum Pare",
        population_exposure_score=0.50,
        road_criticality_score=0.75,
        critical_infrastructure_count=5,
        accessibility_penalty=0.60,
        lifeline_highway="NH-415 (Banderdewa - Itanagar)",
    ),
    "REAL-NER-007": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-007",
        name="Imphal Valley & Kangpokpi Slopes",
        population_exposure_score=0.70,
        road_criticality_score=0.85,
        critical_infrastructure_count=8,
        accessibility_penalty=0.65,
        lifeline_highway="NH-37 & NH-2 Corridors",
    ),
    "REAL-NER-008": ZoneVulnerabilityProfile(
        zone_id="REAL-NER-008",
        name="Agartala & West Tripura Foothills",
        population_exposure_score=0.60,
        road_criticality_score=0.65,
        critical_infrastructure_count=6,
        accessibility_penalty=0.35,
        lifeline_highway="NH-8 Corridor",
    ),
}

# Also alias DEMO zones for backward compatibility in demo mode
for i in range(1, 9):
    real_key = f"REAL-NER-{i:03d}"
    demo_key = f"DEMO-NER-{i:03d}"
    if real_key in NER_VULNERABILITY_PROFILES:
        prof = NER_VULNERABILITY_PROFILES[real_key]
        NER_VULNERABILITY_PROFILES[demo_key] = ZoneVulnerabilityProfile(
            zone_id=demo_key,
            name=f"Demo {prof.name}",
            population_exposure_score=prof.population_exposure_score,
            road_criticality_score=prof.road_criticality_score,
            critical_infrastructure_count=prof.critical_infrastructure_count,
            accessibility_penalty=prof.accessibility_penalty,
            lifeline_highway=prof.lifeline_highway,
        )


@dataclass
class PriorityResult:
    zone_id: str
    zone_name: str
    priority_level: str  # "Priority 1", "Priority 2", "Priority 3"
    composite_score: float
    risk_score: float
    confidence_score: float
    population_score: float
    road_criticality_score: float
    infrastructure_score: float
    accessibility_penalty: float
    lifeline_highway: str
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EmergencyPriorityEngine:
    """Computes operational emergency priority ranking across monitored zones."""

    WEIGHT_RISK = 0.40
    WEIGHT_POPULATION = 0.20
    WEIGHT_ROAD = 0.15
    WEIGHT_INFRASTRUCTURE = 0.10
    WEIGHT_ACCESSIBILITY = 0.10
    WEIGHT_CONFIDENCE = 0.05

    @classmethod
    def calculate_priority(
        cls,
        zone_id: str,
        risk_score: float,
        confidence_score: float = 0.85,
    ) -> PriorityResult:
        profile = NER_VULNERABILITY_PROFILES.get(
            zone_id,
            ZoneVulnerabilityProfile(
                zone_id=zone_id,
                name=f"Zone {zone_id}",
                population_exposure_score=0.50,
                road_criticality_score=0.50,
                critical_infrastructure_count=3,
                accessibility_penalty=0.50,
                lifeline_highway="State Highway Corridor",
            ),
        )

        infra_norm = min(profile.critical_infrastructure_count / 10.0, 1.0)

        composite = (
            cls.WEIGHT_RISK * risk_score
            + cls.WEIGHT_POPULATION * profile.population_exposure_score
            + cls.WEIGHT_ROAD * profile.road_criticality_score
            + cls.WEIGHT_INFRASTRUCTURE * infra_norm
            + cls.WEIGHT_ACCESSIBILITY * profile.accessibility_penalty
            + cls.WEIGHT_CONFIDENCE * confidence_score
        )
        composite = round(min(max(composite, 0.0), 1.0), 3)

        if composite >= 0.65:
            priority = "Priority 1"
            action_summary = "CRITICAL / IMMEDIATE ACTION REQUIRED"
        elif composite >= 0.40:
            priority = "Priority 2"
            action_summary = "MODERATE / PRE-POSITION EQUIPMENT & INSPECT"
        else:
            priority = "Priority 3"
            action_summary = "ADVISORY / CONTINUOUS SENSOR MONITORING"

        # Generate human-readable explanation of key driving factors
        reasons = []
        if risk_score >= 0.40:
            reasons.append(f"High landslide probability ({risk_score:.2f}) from rainfall and saturated terrain")
        else:
            reasons.append(f"Moderate landslide risk score ({risk_score:.2f})")

        if profile.road_criticality_score >= 0.90:
            reasons.append(f"Arterial lifeline vulnerability on {profile.lifeline_highway} with no redundant detour")

        if profile.population_exposure_score >= 0.70:
            reasons.append("High downstream population density and settlement vulnerability")

        if profile.accessibility_penalty >= 0.70:
            reasons.append("Severe geographic isolation and steep terrain hindering rapid search and rescue")

        if profile.critical_infrastructure_count >= 5:
            reasons.append(f"{profile.critical_infrastructure_count} critical facilities (hospitals, power/water stations) in hazard buffer")

        explanation = f"[{action_summary}]: " + "; ".join(reasons) + "."

        return PriorityResult(
            zone_id=zone_id,
            zone_name=profile.name,
            priority_level=priority,
            composite_score=composite,
            risk_score=risk_score,
            confidence_score=confidence_score,
            population_score=profile.population_exposure_score,
            road_criticality_score=profile.road_criticality_score,
            infrastructure_score=infra_norm,
            accessibility_penalty=profile.accessibility_penalty,
            lifeline_highway=profile.lifeline_highway,
            explanation=explanation,
        )

    @classmethod
    def rank_all_zones(cls, zone_risk_map: Dict[str, float]) -> List[PriorityResult]:
        results = [
            cls.calculate_priority(zid, score)
            for zid, score in zone_risk_map.items()
        ]
        # Sort descending by composite score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        return results
