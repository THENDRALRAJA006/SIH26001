"""
LAND-JEPA — Citizen Hazard Visual Evidence Detector
Model: citizen-vision-v1 (YOLOv8n object detection)
SIH26001 · Team ZAIX · Northeast India

Operational Purpose:
Provides automated visual evidence extraction from citizen-submitted hazard photos.
Detects visible physical landslide, rockfall, and structural tunnel features.
Does NOT classify human integrity or assign 'fake citizen' labels.
Output is consumed by the CitizenVerificationService to compute an Evidence Strength Score.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: "rockfall",
    1: "landslides",
    2: "tunnel",
}

HAZARD_CLASSES = {"rockfall", "landslides"}


class CitizenVisionDetector:
    """
    Ultralytics YOLOv8 detector for citizen hazard visual verification.
    Thread-safe inference on CPU or GPU.
    """

    def __init__(
        self,
        weights_path: Union[str, Path, None] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "cpu",
        model_version: str = "citizen-vision-v1",
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model_version = model_version
        self.model = None

        # Resolve weights path
        root = Path(__file__).resolve().parent.parent.parent
        default_path = root / "ml" / "checkpoints" / "citizen_vision" / "best.pt"

        if weights_path and Path(weights_path).exists():
            self.weights_path = Path(weights_path)
        elif default_path.exists():
            self.weights_path = default_path
        else:
            # Fallback to local runs or base yolov8n if training in progress
            runs_candidates = list(Path("D:/SIH26001/runs/train").glob("*/weights/best.pt"))
            if runs_candidates:
                self.weights_path = max(runs_candidates, key=lambda p: p.stat().st_mtime)
            else:
                self.weights_path = Path("yolov8n.pt")

        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            logger.info(f"Loading Citizen Vision detector from: {self.weights_path}")
            self.model = YOLO(str(self.weights_path))
            logger.info(f"Citizen Vision detector loaded successfully (device={self.device})")
        except Exception as e:
            logger.error(f"Failed to load Citizen Vision detector: {e}")
            self.model = None

    def reload_weights(self, weights_path: Union[str, Path]) -> bool:
        """Hot-reload model weights after a training run finishes."""
        p = Path(weights_path)
        if p.exists():
            try:
                from ultralytics import YOLO
                self.weights_path = p
                self.model = YOLO(str(p))
                logger.info(f"Reloaded Citizen Vision model from {p}")
                return True
            except Exception as e:
                logger.error(f"Error reloading weights: {e}")
        return False

    def detect(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
        conf: Union[float, None] = None,
    ) -> dict[str, Any]:
        """
        Run inference on image input and extract visual hazard evidence.
        
        Returns:
            dict containing detections, hazard_detected, top_hazard_type,
            max_confidence, total_hazard_area_ratio, and inference_latency_ms.
        """
        conf_to_use = conf if conf is not None else self.conf_threshold

        if self.model is None:
            return {
                "success": False,
                "error": "Model not loaded",
                "detections": [],
                "hazard_detected": False,
                "top_hazard_type": None,
                "max_confidence": 0.0,
                "total_hazard_area_ratio": 0.0,
                "inference_latency_ms": 0.0,
                "model_version": self.model_version,
            }

        # Convert bytes to PIL Image if needed
        import io
        img = image_input
        if isinstance(img, bytes):
            try:
                img = Image.open(io.BytesIO(img)).convert("RGB")
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Invalid image bytes: {e}",
                    "detections": [],
                    "hazard_detected": False,
                    "top_hazard_type": None,
                    "max_confidence": 0.0,
                    "total_hazard_area_ratio": 0.0,
                    "inference_latency_ms": 0.0,
                    "model_version": self.model_version,
                }

        t_start = time.perf_counter()
        try:
            results = self.model.predict(
                source=img,
                conf=conf_to_use,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
            latency_ms = round((time.perf_counter() - t_start) * 1000, 2)
        except Exception as e:
            logger.error(f"Inference error in CitizenVisionDetector: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "hazard_detected": False,
                "top_hazard_type": None,
                "max_confidence": 0.0,
                "total_hazard_area_ratio": 0.0,
                "inference_latency_ms": 0.0,
                "model_version": self.model_version,
            }

        detections = []
        max_conf = 0.0
        top_hazard = None
        total_hazard_area = 0.0

        if results and len(results) > 0:
            res = results[0]
            orig_shape = res.orig_shape  # (height, width)
            h, w = orig_shape[0], orig_shape[1]
            total_img_area = float(max(h * w, 1))

            if res.boxes is not None and len(res.boxes) > 0:
                for box in res.boxes:
                    cls_id = int(box.cls.item())
                    score = float(box.conf.item())
                    cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = [round(v, 2) for v in xyxy]

                    # Normalized coordinates [0, 1]
                    norm_x1 = round(max(0.0, min(1.0, x1 / w)), 4)
                    norm_y1 = round(max(0.0, min(1.0, y1 / h)), 4)
                    norm_x2 = round(max(0.0, min(1.0, x2 / w)), 4)
                    norm_y2 = round(max(0.0, min(1.0, y2 / h)), 4)

                    box_w = max(0.0, x2 - x1)
                    box_h = max(0.0, y2 - y1)
                    area_ratio = round((box_w * box_h) / total_img_area, 4)

                    det = {
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": round(score, 4),
                        "bbox": [x1, y1, x2, y2],
                        "bbox_normalized": [norm_x1, norm_y1, norm_x2, norm_y2],
                        "area_ratio": area_ratio,
                        "is_hazard": cls_name in HAZARD_CLASSES,
                    }
                    detections.append(det)

                    if cls_name in HAZARD_CLASSES:
                        total_hazard_area += area_ratio
                        if score > max_conf:
                            max_conf = score
                            top_hazard = cls_name

        hazard_detected = (top_hazard is not None and max_conf >= conf_to_use)

        return {
            "success": True,
            "detections": detections,
            "detections_count": len(detections),
            "hazard_detected": hazard_detected,
            "top_hazard_type": top_hazard,
            "max_confidence": round(max_conf, 4),
            "total_hazard_area_ratio": round(min(1.0, total_hazard_area), 4),
            "inference_latency_ms": latency_ms,
            "model_version": self.model_version,
        }
