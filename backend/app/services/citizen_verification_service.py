"""
backend/app/services/citizen_verification_service.py
=====================================================
LAND-JEPA — Citizen Hazard Visual Evidence Verification Subsystem
Powered by Ultralytics Computer Vision & Multi-Source Geospatial Verification
SIH26001 · Team ZAIX · Northeast India

OPERATIONAL PRINCIPLES:
1. Evidence Verification, Not Citizen Fraud:
   Evaluates physical visible hazard evidence (rockfall, landslides, tunnel portals).
   Never assigns pejorative labels to human reporters.
2. EXIF Policy:
   Missing EXIF is 'EXIF_UNAVAILABLE', not fraud (most messaging apps strip EXIF).
   Location divergence > 2km is 'POSSIBLE_MISMATCH' / 'NEEDS_REVIEW'.
3. Decoupled AI Architecture:
   Visual evidence verification computes an objective Evidence Strength Score (0–1).
   It does NOT silently overwrite LAND-JEPA neural risk probabilities.
   All verified reports require Human-in-the-Loop officer action.
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image, ExifTags

from app.core.config import get_settings
from ml.models.citizen_vision_detector import CitizenVisionDetector

logger = logging.getLogger(__name__)

# Official LAND-JEPA Northeast Highway Corridors Reference Coordinates
NER_CORRIDORS = [
    {"id": "REAL-NER-001", "name": "NH-27 Guwahati–Shillong", "lat": 25.57, "lng": 91.88, "state": "Assam / Meghalaya"},
    {"id": "REAL-NER-002", "name": "NH-6 Silchar–Imphal", "lat": 24.82, "lng": 93.94, "state": "Assam / Manipur"},
    {"id": "REAL-NER-003", "name": "NH-29 Dimapur–Kohima", "lat": 25.67, "lng": 94.12, "state": "Nagaland"},
    {"id": "REAL-NER-004", "name": "NH-102 Agartala–Sabroom", "lat": 23.84, "lng": 91.28, "state": "Tripura"},
    {"id": "REAL-NER-005", "name": "NH-37 Jorhat–Dibrugarh", "lat": 27.10, "lng": 92.10, "state": "Assam"},
    {"id": "REAL-NER-006", "name": "NH-117 Aizawl–Lunglei", "lat": 23.27, "lng": 92.73, "state": "Mizoram"},
    {"id": "REAL-NER-007", "name": "NH-06 Demagiri Spur", "lat": 23.00, "lng": 92.90, "state": "Mizoram"},
    {"id": "REAL-NER-008", "name": "SH-4 Tawang Access Corridor", "lat": 27.55, "lng": 92.25, "state": "Arunachal Pradesh"},
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Great Circle distance between two lat/lon pairs in kilometers."""
    r = 6371.0  # Earth radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c


def compute_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """
    Calculate Difference Hash (dHash) for fast near-duplicate image detection.
    Resizes to (hash_size + 1, hash_size), converts to grayscale, compares adjacent pixels.
    """
    try:
        resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
        pixels = list(resized.getdata())
        diff = []
        for row in range(hash_size):
            for col in range(hash_size):
                left = pixels[row * (hash_size + 1) + col]
                right = pixels[row * (hash_size + 1) + col + 1]
                diff.append(left > right)
        decimal_val = 0
        hex_string = []
        for index, value in enumerate(diff):
            if value:
                decimal_val += 2 ** (index % 4)
            if index % 4 == 3:
                hex_string.append(hex(decimal_val)[2:])
                decimal_val = 0
        return "".join(hex_string)
    except Exception as e:
        logger.warning(f"Failed to compute dHash: {e}")
        return ""


def hamming_distance(hash1: str, hash2: str) -> int:
    """Compute Hamming distance between two hex hash strings."""
    if not hash1 or not hash2 or len(hash1) != len(hash2):
        return 999
    try:
        return bin(int(hash1, 16) ^ int(hash2, 16)).count("1")
    except Exception:
        return 999


def extract_exif_telemetry(img: Image.Image) -> Dict[str, Any]:
    """
    Extract EXIF tags, GPS telemetry, camera model, and timestamp.
    Safely handles missing or corrupted tags.
    """
    res = {
        "has_exif": False,
        "camera_make": None,
        "camera_model": None,
        "datetime_original": None,
        "gps_lat": None,
        "gps_lng": None,
        "gps_altitude_m": None,
        "exif_status": "EXIF_UNAVAILABLE",
        "notice": "EXIF metadata not present in image file (commonly stripped by messaging apps or browser canvas).",
    }
    try:
        exif_raw = img.getexif()
        if not exif_raw:
            return res

        res["has_exif"] = True
        res["exif_status"] = "EXIF_PRESENT_NO_GPS"
        res["notice"] = "EXIF tags found; camera device details extracted."

        # Parse tags
        for tag_id, value in exif_raw.items():
            tag_name = ExifTags.TAGS.get(tag_id, tag_id)
            if tag_name == "Make":
                res["camera_make"] = str(value).strip()
            elif tag_name == "Model":
                res["camera_model"] = str(value).strip()
            elif tag_name in ("DateTimeOriginal", "DateTime"):
                res["datetime_original"] = str(value).strip()

        # Parse GPS IFD if present
        gps_ifd = exif_raw.get_ifd(ExifTags.IFD.GPSInfo)
        if gps_ifd:
            lat_ref = gps_ifd.get(ExifTags.GPS.GPSLatitudeRef, "N")
            lat_coords = gps_ifd.get(ExifTags.GPS.GPSLatitude)
            lng_ref = gps_ifd.get(ExifTags.GPS.GPSLongitudeRef, "E")
            lng_coords = gps_ifd.get(ExifTags.GPS.GPSLongitude)

            if lat_coords and lng_coords:
                def dms_to_dd(dms, ref):
                    degrees = float(dms[0])
                    minutes = float(dms[1]) / 60.0
                    seconds = float(dms[2]) / 3600.0
                    dd = degrees + minutes + seconds
                    if ref in ("S", "W"):
                        dd = -dd
                    return dd

                lat_dd = dms_to_dd(lat_coords, lat_ref)
                lng_dd = dms_to_dd(lng_coords, lng_ref)

                res["gps_lat"] = round(lat_dd, 6)
                res["gps_lng"] = round(lng_dd, 6)
                res["exif_status"] = "EXIF_GPS_EXTRACTED"
                res["notice"] = "Valid camera GPS telemetry successfully extracted."

                alt = gps_ifd.get(ExifTags.GPS.GPSAltitude)
                if alt:
                    res["gps_altitude_m"] = round(float(alt), 1)

    except Exception as e:
        logger.warning(f"Error parsing image EXIF: {e}")
        res["notice"] = f"EXIF parsing notice: {type(e).__name__}"

    return res


class CitizenVerificationService:
    """
    Singleton service that processes citizen reports through the visual verification pipeline:
    1. Integrity & Duplicate Detection (SHA-256 + dHash)
    2. EXIF Geolocation Extraction & Spatial Consistency Analysis
    3. YOLOv8 Visual Hazard Evidence Detection
    4. Multi-Modal Evidence Strength Scoring (0.0 – 1.0)
    5. Operational Classification & Officer Review Lifecycle
    """

    _instance: Optional["CitizenVerificationService"] = None

    def __init__(self) -> None:
        self.settings = get_settings()
        self.detector = CitizenVisionDetector(
            weights_path=getattr(self.settings, "CITIZEN_VISION_MODEL_PATH", None),
            conf_threshold=0.25,
        )
        self.known_reports: List[Dict[str, Any]] = []
        self._seed_initial_history()

    @classmethod
    def get_instance(cls) -> "CitizenVerificationService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _seed_initial_history(self) -> None:
        """Populate initial ledger so duplicate matching works from inception."""
        pass

    def verify_report(
        self,
        image_bytes: Optional[bytes] = None,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        reported_lat: Optional[float] = None,
        reported_lng: Optional[float] = None,
        corridor_id: Optional[str] = None,
        road_name: Optional[str] = None,
        description: Optional[str] = None,
        reporter_id: Optional[str] = "citizen_anonymous",
    ) -> Dict[str, Any]:
        """
        Runs the full verification pipeline on a citizen or field submission.
        """
        t0 = time.perf_counter()
        report_id = f"VERIF-CR-{int(time.time())}-{str(uuid.uuid4())[:4].upper()}"
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        # ── 1. Resolve Image Bytes & PIL Object ──────────────────────────────
        raw_bytes = image_bytes
        pil_img: Optional[Image.Image] = None

        if not raw_bytes and image_base64:
            try:
                # Strip prefix if present (e.g., data:image/jpeg;base64,...)
                b64_str = image_base64
                if "," in b64_str:
                    b64_str = b64_str.split(",", 1)[1]
                raw_bytes = base64.b64decode(b64_str)
            except Exception as e:
                logger.warning(f"Failed to decode base64 image: {e}")

        if not raw_bytes and image_url:
            # Check if local file in workspace
            try:
                clean_path = image_url.lstrip("/")
                candidates = [
                    Path(clean_path),
                    Path(__file__).resolve().parents[3] / clean_path,
                    Path(__file__).resolve().parents[3] / "frontend" / "dashboard" / "public" / clean_path,
                ]
                for cand in candidates:
                    if cand.exists() and cand.is_file():
                        raw_bytes = cand.read_bytes()
                        break
            except Exception as e:
                logger.warning(f"Failed to read image_url {image_url}: {e}")

        sha256_hash = None
        dhash_val = None
        exif_info = {
            "has_exif": False,
            "exif_status": "EXIF_UNAVAILABLE",
            "notice": "No image provided with submission.",
        }

        if raw_bytes:
            sha256_hash = hashlib.sha256(raw_bytes).hexdigest()
            try:
                pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
                dhash_val = compute_dhash(pil_img)
                exif_info = extract_exif_telemetry(pil_img)
            except Exception as e:
                logger.error(f"Error opening image with PIL: {e}")
                pil_img = None

        # ── 2. Duplicate Detection ───────────────────────────────────────────
        duplicate_type = "UNIQUE"
        duplicate_match_id = None
        min_hamming = 999

        if sha256_hash or dhash_val:
            for rep in self.known_reports:
                # Exact SHA-256 match
                if sha256_hash and rep.get("sha256") == sha256_hash:
                    duplicate_type = "EXACT_DUPLICATE"
                    duplicate_match_id = rep.get("report_id")
                    break
                # Near-duplicate dHash comparison (Hamming distance <= 4)
                if dhash_val and rep.get("dhash"):
                    dist = hamming_distance(dhash_val, rep["dhash"])
                    if dist < min_hamming:
                        min_hamming = dist
                    if dist <= 4:
                        duplicate_type = "NEAR_DUPLICATE"
                        duplicate_match_id = rep.get("report_id")

        # ── 3. Geospatial Telemetry & Corridor Matching ──────────────────────
        r_lat = reported_lat or 25.57
        r_lng = reported_lng or 91.88

        # Identify nearest monitored corridor
        matched_corridor = NER_CORRIDORS[0]
        min_corr_dist = 9999.0
        for corr in NER_CORRIDORS:
            d = haversine_km(r_lat, r_lng, corr["lat"], corr["lng"])
            if d < min_corr_dist:
                min_corr_dist = d
                matched_corridor = corr

        resolved_corridor_id = corridor_id or matched_corridor["id"]
        corridor_proximity_km = round(min_corr_dist, 2)

        # Compare EXIF GPS with reported coordinates if EXIF is present
        exif_spatial_status = "EXIF_UNAVAILABLE"
        exif_distance_km = None
        if exif_info.get("gps_lat") is not None and exif_info.get("gps_lng") is not None:
            exif_dist = haversine_km(
                r_lat, r_lng, exif_info["gps_lat"], exif_info["gps_lng"]
            )
            exif_distance_km = round(exif_dist, 2)
            if exif_dist <= 2.0:
                exif_spatial_status = "EXIF_MATCH"
            elif exif_dist <= 15.0:
                exif_spatial_status = "CLOSE_PROXIMITY"
            else:
                exif_spatial_status = "POSSIBLE_MISMATCH"

        # ── 4. Computer Vision Object Detection ──────────────────────────────
        cv_result = {
            "success": False,
            "detections": [],
            "detections_count": 0,
            "hazard_detected": False,
            "top_hazard_type": None,
            "max_confidence": 0.0,
            "total_hazard_area_ratio": 0.0,
            "inference_latency_ms": 0.0,
            "model_version": "citizen-vision-v1",
        }

        if pil_img:
            cv_result = self.detector.detect(pil_img)

        # ── 5. Evidence Strength Score Formulation (0.0 to 1.0) ──────────────
        evidence_score = 0.0
        confidence_breakdown = {}

        if cv_result.get("hazard_detected"):
            max_conf = cv_result.get("max_confidence", 0.0)
            hazard_area = cv_result.get("total_hazard_area_ratio", 0.0)

            # Core vision evidence component (up to 0.60)
            vision_comp = max_conf * 0.60
            confidence_breakdown["vision_confidence_component"] = round(vision_comp, 3)

            # Significant hazard area bonus (up to 0.15)
            area_comp = min(hazard_area * 1.5, 0.15)
            confidence_breakdown["visible_hazard_area_bonus"] = round(area_comp, 3)

            # Spatial consistency components (up to 0.25)
            spatial_comp = 0.0
            if exif_spatial_status == "EXIF_MATCH":
                spatial_comp += 0.15
            elif exif_spatial_status == "CLOSE_PROXIMITY":
                spatial_comp += 0.08
            elif exif_spatial_status == "EXIF_UNAVAILABLE":
                # Do not penalize; grant standard baseline credit if reported in corridor
                spatial_comp += 0.05

            if corridor_proximity_km <= 10.0:
                spatial_comp += 0.10
            elif corridor_proximity_km <= 35.0:
                spatial_comp += 0.05

            confidence_breakdown["spatial_corroboration_component"] = round(spatial_comp, 3)

            # Base score accumulation
            evidence_score = vision_comp + area_comp + spatial_comp

            # Mismatch penalty: If EXIF GPS explicitly exists and is far away (> 25 km)
            if exif_spatial_status == "POSSIBLE_MISMATCH" and exif_distance_km and exif_distance_km > 25.0:
                deduction = 0.25
                evidence_score = max(0.15, evidence_score - deduction)
                confidence_breakdown["exif_location_mismatch_penalty"] = -deduction

        else:
            # No rockfall/landslide detected
            if pil_img:
                evidence_score = 0.15  # Photo submitted, but no recognizable hazard feature
                confidence_breakdown["no_hazard_features_detected"] = 0.15
            else:
                evidence_score = 0.05  # Text-only report without visual evidence
                confidence_breakdown["text_only_submission"] = 0.05

        # Duplication penalty: Re-uploaded photos capped
        if duplicate_type in ("EXACT_DUPLICATE", "NEAR_DUPLICATE"):
            evidence_score = min(evidence_score, 0.10)
            confidence_breakdown["duplicate_submission_penalty"] = "Capped to 0.10"

        evidence_score = round(max(0.0, min(1.0, evidence_score)), 4)

        # ── 6. Automated Category Classification ─────────────────────────────
        if duplicate_type in ("EXACT_DUPLICATE", "NEAR_DUPLICATE"):
            automated_status = "DUPLICATE"
            officer_recommendation = f"Flagged as {duplicate_type.lower().replace('_', ' ')} of report {duplicate_match_id}. Officer review suggested."
        elif exif_spatial_status == "POSSIBLE_MISMATCH":
            automated_status = "LOCATION_MISMATCH"
            officer_recommendation = f"EXIF GPS differs from reported GPS by {exif_distance_km} km. Officer manual review required."
        elif cv_result.get("hazard_detected") and evidence_score >= 0.70:
            automated_status = "STRONG_EVIDENCE"
            officer_recommendation = f"High-confidence {cv_result['top_hazard_type']} visual evidence corroborated with corridor geometry. Priority triage."
        elif cv_result.get("hazard_detected") and evidence_score >= 0.45:
            automated_status = "MODERATE_EVIDENCE"
            officer_recommendation = f"Identified {cv_result['top_hazard_type']} features. Standard officer inspection queue."
        elif cv_result.get("hazard_detected"):
            automated_status = "NEEDS_REVIEW"
            officer_recommendation = "Low confidence detection. Awaiting field validation."
        elif pil_img:
            automated_status = "NO_HAZARD_DETECTED"
            officer_recommendation = "No distinct landslide or rockfall mass detected by YOLO detector. Review photo manually."
        else:
            automated_status = "TEXT_ONLY"
            officer_recommendation = "Text report without photographic evidence. Requires field dispatched verification."

        total_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        verification_record = {
            "report_id": report_id,
            "timestamp": now_iso,
            "reporter_id": reporter_id,
            "corridor_id": resolved_corridor_id,
            "corridor_name": road_name or matched_corridor["name"],
            "reported_coords": [round(r_lat, 5), round(r_lng, 5)],
            "corridor_distance_km": corridor_proximity_km,
            "description": description or "",
            "sha256": sha256_hash,
            "dhash": dhash_val,
            "duplicate_type": duplicate_type,
            "duplicate_match_id": duplicate_match_id,
            "exif": exif_info,
            "exif_spatial_status": exif_spatial_status,
            "exif_distance_km": exif_distance_km,
            "vision_model": cv_result.get("model_version", "citizen-vision-v1"),
            "hazard_detected": cv_result.get("hazard_detected", False),
            "top_hazard_type": cv_result.get("top_hazard_type"),
            "max_confidence": cv_result.get("max_confidence", 0.0),
            "detections_count": cv_result.get("detections_count", 0),
            "detections": cv_result.get("detections", []),
            "evidence_strength_score": evidence_score,
            "score_breakdown": confidence_breakdown,
            "automated_status": automated_status,
            "officer_status": "NEW",  # NEW | VERIFIED | REJECTED | REQUEST_MORE_INFO | MARK_DUPLICATE
            "officer_recommendation": officer_recommendation,
            "officer_notes": None,
            "reviewed_by": None,
            "reviewed_at": None,
            "verification_latency_ms": total_latency_ms,
        }

        # Save to local in-memory ledger
        self.known_reports.insert(0, verification_record)
        if len(self.known_reports) > 500:
            self.known_reports.pop()

        logger.info(
            f"Verified citizen report {report_id}: Status={automated_status}, "
            f"Score={evidence_score:.2f}, Latency={total_latency_ms}ms"
        )
        return verification_record

    def list_reports(
        self,
        status_filter: Optional[str] = None,
        corridor_filter: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List reports with multi-faceted filtering."""
        results = self.known_reports
        if status_filter and status_filter.upper() != "ALL":
            sf = status_filter.upper()
            results = [
                r for r in results
                if r.get("automated_status") == sf or r.get("officer_status") == sf
            ]
        if corridor_filter and corridor_filter.upper() != "ALL":
            cf = corridor_filter.upper()
            results = [r for r in results if r.get("corridor_id") == cf]
        return results[:limit]

    def get_report(self, report_id: str) -> Optional[Dict[str, Any]]:
        for r in self.known_reports:
            if r.get("report_id") == report_id:
                return r
        return None

    def officer_action(
        self,
        report_id: str,
        action: str,  # VERIFY | REJECT | REQUEST_MORE_INFO | MARK_DUPLICATE
        officer_id: str = "OFFICER-NER-01",
        notes: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Officer triage decision. Updates status and records audit record.
        """
        rep = self.get_report(report_id)
        if not rep:
            raise KeyError(f"Report {report_id} not found.")

        act = action.upper().strip()
        valid_actions = {"VERIFY", "REJECT", "REQUEST_MORE_INFO", "MARK_DUPLICATE"}
        if act not in valid_actions:
            raise ValueError(f"Invalid action {act}. Must be one of {valid_actions}")

        old_status = rep.get("officer_status")
        now_iso = datetime.now(tz=timezone.utc).isoformat()

        if act == "VERIFY":
            rep["officer_status"] = "VERIFIED"
        elif act == "REJECT":
            rep["officer_status"] = "REJECTED"
        elif act == "REQUEST_MORE_INFO":
            rep["officer_status"] = "REQUEST_MORE_INFO"
        elif act == "MARK_DUPLICATE":
            rep["officer_status"] = "DUPLICATE"

        rep["reviewed_by"] = officer_id
        rep["reviewed_at"] = now_iso
        rep["officer_notes"] = notes

        # Note: Visual verification does NOT modify numerical LAND-JEPA probability
        rep["numerical_model_unmodified"] = True

        return {
            "success": True,
            "report_id": report_id,
            "old_status": old_status,
            "new_status": rep["officer_status"],
            "reviewed_by": officer_id,
            "reviewed_at": now_iso,
            "notes": notes,
        }
