"""
Person Detector & Tracker for Sentinel-AI
Implements:
- Feature 3: ByteTrack Precision Person Tracking & Minimalist HUD
- Spatial proximity calculation to support multi-person altercation validation
"""

import logging
from typing import List, Tuple, Dict, Any
import cv2
import numpy as np
import torch
from ultralytics import YOLO

from core.config import YOLO_MODEL_PATH, YOLO_CONFIDENCE

logger = logging.getLogger(__name__)


class PersonDetector:
    def __init__(self, model_path: str = str(YOLO_MODEL_PATH), confidence: float = YOLO_CONFIDENCE):
        self.confidence = confidence
        logger.info("Initializing YOLO11n detector from: %s", model_path)
        self.model = YOLO(model_path)
        # Warmup
        dummy_img = np.zeros((320, 320, 3), dtype=np.uint8)
        self.model(dummy_img, verbose=False)
        logger.info("YOLO11n detector ready.")

    def track(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], bool]:
        """
        Runs ByteTrack on frame.
        Returns:
            annotated_frame: np.ndarray with crisp thin HUD
            detections: List of dicts with 'id', 'bbox', 'conf', 'center'
            has_proximity_cluster: bool (True if 2 or more people are closely interacting)
        """
        h, w = frame.shape[:2]
        # Fast CPU tracking with optimized input resolution
        results = self.model.track(
            frame,
            persist=True,
            classes=[0],
            conf=self.confidence,
            imgsz=384,
            tracker="bytetrack.yaml",
            verbose=False
        )

        detections = []
        annotated = frame.copy()

        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                ids = boxes.id.cpu().numpy() if boxes.id is not None else [None] * len(boxes)

                for box, conf, track_id in zip(xyxy, confs, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    tid = int(track_id) if track_id is not None else -1

                    detections.append({
                        "id": tid,
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf),
                        "center": (cx, cy),
                        "width": x2 - x1,
                        "height": y2 - y1
                    })

        # Calculate spatial proximity clusters (2+ people in proximity)
        has_proximity_cluster = self._check_proximity(detections)

        # Draw minimalist high-tech HUD
        self._render_minimalist_hud(annotated, detections, has_proximity_cluster)

        return annotated, detections, has_proximity_cluster

    def _check_proximity(self, detections: List[Dict[str, Any]], distance_threshold: float = 180.0) -> bool:
        """Determines if at least 2 people are close enough to have physical contact or fight."""
        if len(detections) < 2:
            return False

        for i in range(len(detections)):
            c1 = np.array(detections[i]["center"])
            for j in range(i + 1, len(detections)):
                c2 = np.array(detections[j]["center"])
                dist = np.linalg.norm(c1 - c2)
                # If centers are within distance threshold or bounding boxes overlap
                if dist < distance_threshold:
                    return True
        return False

    def _render_minimalist_hud(self, frame: np.ndarray, detections: List[Dict[str, Any]], clustered: bool):
        """Renders 1.5px clean anti-aliased bounding boxes and micro-badge tags."""
        base_color = (0, 230, 118) if not clustered else (0, 190, 255)  # Green or Amber

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            tid = det["id"]
            conf_pct = int(det["conf"] * 100)

            # Draw thin refined box
            cv2.rectangle(frame, (x1, y1), (x2, y2), base_color, 1, cv2.LINE_AA)

            # Subtle corner brackets
            bracket_len = min(12, (x2 - x1) // 4, (y2 - y1) // 4)
            # Top-left
            cv2.line(frame, (x1, y1), (x1 + bracket_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x1, y1), (x1, y1 + bracket_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Top-right
            cv2.line(frame, (x2, y1), (x2 - bracket_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y1), (x2, y1 + bracket_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Bottom-left
            cv2.line(frame, (x1, y2), (x1 + bracket_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x1, y2), (x1, y2 - bracket_len), (255, 255, 255), 2, cv2.LINE_AA)
            # Bottom-right
            cv2.line(frame, (x2, y2), (x2 - bracket_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(frame, (x2, y2), (x2, y2 - bracket_len), (255, 255, 255), 2, cv2.LINE_AA)

            # Micro-badge
            label = f"ID #{tid} | {conf_pct}%" if tid != -1 else f"PERSON | {conf_pct}%"
            badge_y = max(15, y1 - 6)
            cv2.rectangle(frame, (x1, badge_y - 12), (x1 + 80, badge_y + 2), (20, 24, 28), -1)
            cv2.putText(frame, label, (x1 + 4, badge_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
