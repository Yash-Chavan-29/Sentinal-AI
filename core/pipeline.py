"""
Unified Surveillance Pipeline for Sentinel-AI
Dual-Threaded Architecture for Zero-Lag Fluid Video:
- Thread 1: High-Speed Camera Streaming Loop (30-60 FPS) with 15 FPS temporal frame feeding
- Thread 2: Asynchronous AI Inference Worker (YOLO11n + X3D-M + Kinetic Energy)
"""

import time
import datetime
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import cv2
import numpy as np

from core.config import SNAPSHOT_DIR, DEFAULT_CAMERA_SOURCE
from core.stream_manager import StreamManager
from core.detector import PersonDetector
from core.fight_classifier import FightClassifier

logger = logging.getLogger(__name__)


class SurveillancePipeline:
    def __init__(self, camera_source: Any = DEFAULT_CAMERA_SOURCE):
        self.stream_mgr = StreamManager(initial_source=camera_source)
        self.detector = PersonDetector()
        self.classifier = FightClassifier()

        # Shared AI State
        self.lock = threading.Lock()
        self.latest_detections: List[Dict[str, Any]] = []
        self.latest_proximity: bool = False
        self.latest_threat: float = 0.0
        self.latest_risk: str = "Normal"
        self.latest_alarm: bool = False
        self.latest_jpeg: Optional[bytes] = None

        # Evidence & Telemetry
        self.incident_log: List[Dict[str, Any]] = []
        self.last_snapshot_time: float = 0.0
        self.snapshot_cooldown: float = 3.5
        self.ai_latency_ms: float = 0.0

        self.current_telemetry: Dict[str, Any] = {
            "fps": 30.0,
            "latency_ms": 0.0,
            "person_count": 0,
            "threat_score": 0.0,
            "risk_level": "Normal",
            "is_alarm": False,
            "camera_connected": False,
            "camera_source": str(camera_source),
            "timestamp": ""
        }

        self.stop_event = threading.Event()
        self.stream_thread: Optional[threading.Thread] = None
        self.ai_thread: Optional[threading.Thread] = None

    def start(self):
        """Starts video ingestion and background processing threads."""
        self.stream_mgr.start()
        self.stop_event.clear()

        # 1. Asynchronous AI Worker Thread
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.ai_thread.start()

        # 2. High-Speed Frame Rendering & Streaming Thread
        self.stream_thread = threading.Thread(target=self._streaming_render_loop, daemon=True)
        self.stream_thread.start()

        logger.info("SurveillancePipeline multi-threaded engine started.")

    def stop(self):
        """Stops all threads."""
        self.stop_event.set()
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.5)
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.5)
        self.stream_mgr.stop()
        logger.info("SurveillancePipeline stopped.")

    def set_camera(self, source: Any) -> bool:
        """Dynamically switch camera source."""
        success = self.stream_mgr.set_source(source)
        with self.lock:
            self.current_telemetry["camera_source"] = str(source)
        return success

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns pre-encoded latest JPEG at native 30 FPS."""
        with self.lock:
            return self.latest_jpeg

    def _ai_worker_loop(self):
        """Asynchronous AI inference running without blocking video playback."""
        while not self.stop_event.is_set():
            raw_frame, connected, _ = self.stream_mgr.get_latest_frame()

            if not connected or raw_frame is None:
                time.sleep(0.05)
                continue

            t0 = time.time()

            # Downscale frame for lightning-fast inference
            h, w = raw_frame.shape[:2]
            proc_frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR) if (w != 640 or h != 360) else raw_frame

            # 1. ByteTrack Person Detection
            annotated_dummy, detections, has_proximity = self.detector.track(proc_frame)
            person_count = len(detections)

            # 2. Action & Kinetic Energy Analysis
            threat, risk, is_alarm, meta = self.classifier.analyze(has_proximity, person_count)

            dt_ms = (time.time() - t0) * 1000.0

            # Update shared state
            with self.lock:
                self.latest_detections = detections
                self.latest_proximity = has_proximity
                self.latest_threat = threat
                self.latest_risk = risk
                self.latest_alarm = is_alarm
                self.ai_latency_ms = round(dt_ms, 1)

                # Snapshot evidence check
                now = time.time()
                if is_alarm and (now - self.last_snapshot_time > self.snapshot_cooldown):
                    self.last_snapshot_time = now
                    self._save_incident_snapshot(proc_frame, threat, person_count)

            # Yield to CPU
            time.sleep(0.02)

    def _streaming_render_loop(self):
        """High-speed rendering loop maintaining solid 30-60 FPS delivery to browser."""
        frame_counter = 0

        while not self.stop_event.is_set():
            t_loop = time.time()
            raw_frame, connected, stream_fps = self.stream_mgr.get_latest_frame()

            if raw_frame is None:
                display_frame = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(display_frame, "AWAITING STREAM...", (200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                h, w = raw_frame.shape[:2]
                display_frame = cv2.resize(raw_frame, (640, 360), interpolation=cv2.INTER_LINEAR) if (w != 640 or h != 360) else raw_frame.copy()

            if connected and raw_frame is not None:
                # Feed frame into action classifier buffer at ~15 FPS (every 2nd frame)
                frame_counter += 1
                if frame_counter % 2 == 0:
                    self.classifier.add_frame(display_frame)

                # Grab latest AI predictions from background worker
                with self.lock:
                    dets = list(self.latest_detections)
                    clustered = self.latest_proximity
                    risk = self.latest_risk
                    is_alarm = self.latest_alarm
                    threat = self.latest_threat
                    lat = self.ai_latency_ms

                # Render HUD bounding boxes and badges
                self.detector._render_minimalist_hud(display_frame, dets, clustered)

                # Render CCTV Watermark overlay
                self._render_cctv_overlay(display_frame, stream_fps, risk, is_alarm)
                p_count = len(dets)
            else:
                p_count = 0
                threat = 0.0
                risk = "Offline"
                is_alarm = False
                lat = 0.0

            # Encode JPEG directly for streaming
            ret, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if ret:
                with self.lock:
                    self.latest_jpeg = buf.tobytes()
                    self.current_telemetry = {
                        "fps": stream_fps if stream_fps > 0 else 30.0,
                        "latency_ms": lat,
                        "person_count": p_count,
                        "threat_score": round(threat * 100, 1),
                        "risk_level": risk,
                        "is_alarm": is_alarm,
                        "camera_connected": connected,
                        "camera_source": str(self.stream_mgr.source),
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }

            # Solid 30 FPS timing
            elapsed = time.time() - t_loop
            sleep_time = max(0.005, 0.033 - elapsed)
            time.sleep(sleep_time)

    def _save_incident_snapshot(self, frame: np.ndarray, score: float, people: int):
        """Saves high-res snapshot of detected altercation to disk."""
        ts_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"INCIDENT_{ts_str}_score{int(score*100)}.jpg"
        filepath = SNAPSHOT_DIR / filename
        cv2.imwrite(str(filepath), frame)

        incident_info = {
            "id": len(self.incident_log) + 1,
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "severity": "CRITICAL" if score > 0.45 else "WARNING",
            "score": round(score * 100, 1),
            "people_involved": people,
            "filename": filename,
            "path": str(filepath)
        }
        self.incident_log.insert(0, incident_info)
        logger.warning("🚨 INCIDENT LOGGED: %s (Threat: %.1f%%)", filename, score * 100)

    def _render_cctv_overlay(self, frame: np.ndarray, fps: float, risk: str, is_alarm: bool):
        """Renders CCTV watermark, live timestamp, and corner brackets."""
        h, w = frame.shape[:2]
        time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Top Bar Background
        cv2.rectangle(frame, (0, 0), (w, 26), (15, 18, 22), -1)

        # Status Dot
        dot_color = (0, 0, 255) if is_alarm else ((0, 165, 255) if risk == "Suspicious" else (0, 230, 118))
        cv2.circle(frame, (14, 13), 5, dot_color, -1)

        # System Label
        cv2.putText(frame, "SENTINEL-AI LIVE", (26, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Risk Badge
        status_txt = f"STATUS: {risk.upper()}"
        cv2.putText(frame, status_txt, (155, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dot_color, 1, cv2.LINE_AA)

        # FPS & Clock
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 200, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, time_str, (w - 145, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

        # Pulsing Red Perimeter Alert on Altercation
        if is_alarm:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
            cv2.putText(frame, "ALERT: PHYSICAL ALTERCATION DETECTED", (w // 2 - 170, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2, cv2.LINE_AA)
