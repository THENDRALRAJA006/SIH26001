"""
tests/ml/test_citizen_vision.py
===============================
LAND-JEPA Citizen Vision Verification Subsystem Tests
SIH26001 · Team ZAIX · Northeast India

Tests:
1. Detector initialization, inference schema, and latency bounds
2. Image integrity hashing (SHA-256 and dHash duplicate detection)
3. EXIF handling policy (EXIF_UNAVAILABLE is not fraud)
4. Geospatial corridor proximity matching
5. Evidence Strength Score formulation (bounded 0.0 to 1.0)
6. Decoupled operational architecture (risk probability remains untouched)
"""

import io
import time
import pytest
from PIL import Image, ImageDraw

from ml.models.citizen_vision_detector import CitizenVisionDetector
from backend.app.services.citizen_verification_service import (
    CitizenVerificationService,
    compute_dhash,
    hamming_distance,
    haversine_km,
)


def create_synthetic_test_image(color=(120, 100, 80), size=(320, 240)) -> bytes:
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 150, 150], fill=(60, 50, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_detector_schema_and_inference():
    detector = CitizenVisionDetector(conf_threshold=0.25)
    img_bytes = create_synthetic_test_image()

    res = detector.detect(img_bytes)
    assert res["success"] is True
    assert "detections" in res
    assert "hazard_detected" in res
    assert "top_hazard_type" in res
    assert "max_confidence" in res
    assert "total_hazard_area_ratio" in res
    assert "inference_latency_ms" in res
    assert res["inference_latency_ms"] >= 0.0
    assert 0.0 <= res["max_confidence"] <= 1.0


def test_dhash_and_duplicate_detection():
    img_bytes_1 = create_synthetic_test_image(color=(120, 100, 80))
    img_bytes_2 = create_synthetic_test_image(color=(121, 101, 81))  # Almost identical
    img_bytes_3 = create_synthetic_test_image(color=(240, 20, 20))   # Completely different

    img1 = Image.open(io.BytesIO(img_bytes_1))
    img2 = Image.open(io.BytesIO(img_bytes_2))
    img3 = Image.open(io.BytesIO(img_bytes_3))

    h1 = compute_dhash(img1)
    h2 = compute_dhash(img2)
    h3 = compute_dhash(img3)

    assert len(h1) == 16
    assert len(h2) == 16
    assert len(h3) == 16

    # Near duplicate should have very small Hamming distance
    dist_near = hamming_distance(h1, h2)
    assert dist_near <= 4

    # Verification service duplicate detection test
    svc = CitizenVerificationService()
    rep1 = svc.verify_report(image_bytes=img_bytes_1, reported_lat=25.57, reported_lng=91.88)
    assert rep1["duplicate_type"] == "UNIQUE"

    # Second submission with identical bytes must trigger EXACT_DUPLICATE
    rep2 = svc.verify_report(image_bytes=img_bytes_1, reported_lat=25.57, reported_lng=91.88)
    assert rep2["duplicate_type"] == "EXACT_DUPLICATE"
    assert rep2["duplicate_match_id"] == rep1["report_id"]
    assert rep2["evidence_strength_score"] <= 0.10


def test_exif_unavailability_policy():
    """
    CRITICAL POLICY: Missing EXIF is EXIF_UNAVAILABLE and NEVER classified as fraud.
    """
    svc = CitizenVerificationService()
    img_without_exif = create_synthetic_test_image()

    rep = svc.verify_report(
        image_bytes=img_without_exif,
        reported_lat=25.57,
        reported_lng=91.88,
        description="Rockfall blocking road",
    )

    assert rep["exif"]["exif_status"] == "EXIF_UNAVAILABLE"
    assert rep["automated_status"] != "FRAUD"
    assert rep["automated_status"] != "REJECTED"
    assert "fraud" not in rep["officer_recommendation"].lower()


def test_spatial_proximity_and_haversine():
    # Distance between Shillong (25.57, 91.88) and Guwahati (26.14, 91.73) ~65km
    dist = haversine_km(25.57, 91.88, 26.14, 91.73)
    assert 60.0 < dist < 75.0

    svc = CitizenVerificationService()
    rep = svc.verify_report(
        reported_lat=25.57,
        reported_lng=91.88,
        description="Corridor test",
    )
    assert rep["corridor_id"] == "REAL-NER-001"
    assert rep["corridor_distance_km"] < 1.0


def test_evidence_strength_score_bounds():
    svc = CitizenVerificationService()
    img_bytes = create_synthetic_test_image()

    rep = svc.verify_report(
        image_bytes=img_bytes,
        reported_lat=25.57,
        reported_lng=91.88,
        description="Rockfall reported on NH-27",
    )

    score = rep["evidence_strength_score"]
    assert 0.0 <= score <= 1.0
    assert "automated_status" in rep
    assert rep["officer_status"] == "NEW"


def test_officer_review_workflow_and_decoupling():
    svc = CitizenVerificationService()
    rep = svc.verify_report(
        reported_lat=25.57,
        reported_lng=91.88,
        description="Test verification flow",
    )
    rep_id = rep["report_id"]

    # Verify action
    res = svc.officer_action(
        report_id=rep_id,
        action="VERIFY",
        officer_id="OFFICER-NER-01",
        notes="Confirmed by field unit with bulldozer clearance underway.",
    )
    assert res["success"] is True
    assert res["new_status"] == "VERIFIED"

    updated = svc.get_report(rep_id)
    assert updated["officer_status"] == "VERIFIED"
    assert updated["reviewed_by"] == "OFFICER-NER-01"
    # Decoupling guarantee
    assert updated.get("numerical_model_unmodified") is True
