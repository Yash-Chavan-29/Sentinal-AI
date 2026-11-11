"""
Stream Manager for Sentinel-AI
Implements:
- Feature 1: Zero-Lag Asynchronous Frame Processing Pipeline
- Feature 5: Resilient Camera Auto-Reconnect Engine
- Feature 7: Dynamic Multi-Source Camera Switcher (Webcam, Mobile IP Cam, RTSP, Simulated Live Feed)
"""

import time
import logging
import threading
from pathlib import Path
from typing import Union, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)
SAMPLE_VIDEO_PATH = Path(__file__).resolve().parent.parent / "Sentinel-AI" / "videos" / "sample_fixed.mp4"


class StreamManager:
    def __init__(self, initial_source: Union[int, str] = 0):
        self.source: Union[int, str] = self._normalize_source(initial_source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self.connected: bool = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Latest frame buffer (Queue size 1 design)
        self.latest_frame: Optional[np.ndarray] = self._generate_placeholder_frame("System Initializing...")
        self.last_frame_time: float = time.time()
        self.fps: float = 30.0
        self.fps_smoothing: float = 0.9

        # Auto-reconnection state
        self.reconnect_delay: float = 2.5
        self.last_reconnect_attempt: float = 0.0
        self.stop_event = threading.Event()

    @staticmethod
    def _normalize_source(source: Union[int, str]) -> Union[int, str]:
        if isinstance(source, str):
            clean = source.strip()
            if clean.isdigit():
                return int(clean)
            if clean.lower() in ["sample", "test", "simulated", "simulation"]:
                return "sample"
            # Auto-prepend http:// if missing from IP address
            if not clean.startswith("http://") and not clean.startswith("https://") and not clean.startswith("rtsp://"):
                clean = "http://" + clean
            # IP Webcam / DroidCam stream paths
            if (":8080" in clean or ":4747" in clean) and not (clean.endswith("/video") or clean.endswith("/mjpeg")):
                clean = clean.rstrip("/") + "/video"
            return clean
        return source

    def set_source(self, new_source: Union[int, str]) -> bool:
        """Switch camera stream source dynamically on the fly."""
        norm_source = self._normalize_source(new_source)
        with self.lock:
            if self.source == norm_source and self.connected:
                return True
            logger.info("Switching camera source to: %s", norm_source)
            self.source = norm_source
            self._close_capture()
            self.last_reconnect_attempt = 0.0
        return True

    def start(self):
        """Start the background asynchronous capture thread."""
        if self.running:
            return
        self.running = True
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        logger.info("StreamManager background capture started.")

    def stop(self):
        """Stop background capture and release resources."""
        self.running = False
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._close_capture()
        logger.info("StreamManager stopped.")

    def _open_capture(self) -> bool:
        """Attempts to open capture with optimal buffer size."""
        try:
            if str(self.source).lower() in ["test", "simulation", "simulated", "sample"]:
                if SAMPLE_VIDEO_PATH.is_file():
                    cap = cv2.VideoCapture(str(SAMPLE_VIDEO_PATH))
                    if cap.isOpened():
                        self.cap = cap
                        self.connected = True
                        logger.info("Opened simulated live camera feed from: %s", SAMPLE_VIDEO_PATH)
                        return True

            if isinstance(self.source, int):
                cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(self.source)
            else:
                cap = cv2.VideoCapture(self.source)

            if cap.isOpened():
                # Test first frame to verify it actually reads
                ret, test_f = cap.read()
                if ret and test_f is not None:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap = cap
                    self.connected = True
                    logger.info("Successfully opened and verified camera source: %s", self.source)
                    return True
                else:
                    cap.release()
                    self.connected = False
                    return False
            else:
                self.connected = False
                return False
        except Exception as e:
            logger.error("Error opening camera %s: %s", self.source, e)
            self.connected = False
            return False

    def _close_capture(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.connected = False

    def _generate_placeholder_frame(self, message: str) -> np.ndarray:
        """Generates a high-tech tactical standby screen when camera is offline."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Tactical grid pattern
        frame[::40, :] = (20, 24, 30)
        frame[:, ::40] = (20, 24, 30)

        # Pulse circle
        center = (320, 200)
        pulse_val = int(127 + 127 * np.sin(time.time() * 3))
        cv2.circle(frame, center, 60, (0, pulse_val, 255), 2)
        cv2.circle(frame, center, 6, (0, 0, 255), -1)

        # Status text
        cv2.putText(frame, "SENTINEL-AI CCTV SYSTEM", (160, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        cv2.putText(frame, f"[ {message.upper()} ]", (150, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 165, 255), 2)
        cv2.putText(frame, f"Active Target: {self.source}", (190, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, "Plug in camera or enter Phone IP Stream URL (DroidCam/RTSP)", (100, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (140, 140, 140), 1)
        cv2.putText(frame, "Or click 'Simulated Live Feed' to preview altercation detection", (105, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 230, 118), 1)

        return frame

    def _capture_loop(self):
        """Asynchronous producer loop running independently from AI inference."""
        while not self.stop_event.is_set():
            now = time.time()
            if not self.connected or self.cap is None or not self.cap.isOpened():
                if now - self.last_reconnect_attempt > self.reconnect_delay:
                    self.last_reconnect_attempt = now
                    if not self._open_capture():
                        placeholder = self._generate_placeholder_frame("Waiting for camera feed...")
                        with self.lock:
                            self.latest_frame = placeholder
                time.sleep(0.05)
                continue

            # Read frame
            success, frame = self.cap.read()
            if not success or frame is None:
                # If simulated video, loop it
                if str(self.source).lower() in ["test", "simulation", "simulated", "sample"]:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue

                logger.warning("Camera connection dropped. Will reconnect in %.1fs...", self.reconnect_delay)
                self._close_capture()
                placeholder = self._generate_placeholder_frame("Reconnecting to camera...")
                with self.lock:
                    self.latest_frame = placeholder
                time.sleep(0.5)
                continue

            dt = now - self.last_frame_time
            if dt > 0:
                instant_fps = 1.0 / dt
                self.fps = self.fps_smoothing * self.fps + (1.0 - self.fps_smoothing) * instant_fps
            self.last_frame_time = now

            with self.lock:
                self.latest_frame = frame

            # Regulate capture rate if simulated feed to avoid 500 FPS
            if str(self.source).lower() in ["test", "simulation", "simulated", "sample"]:
                time.sleep(0.03)
            else:
                time.sleep(0.005)

    def get_latest_frame(self) -> Tuple[Optional[np.ndarray], bool, float]:
        """Returns (frame, is_connected, current_fps). Thread-safe and zero-wait."""
        with self.lock:
            frame = self.latest_frame.copy() if self.latest_frame is not None else None
            return frame, self.connected, round(self.fps, 1)
