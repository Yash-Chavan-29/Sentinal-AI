
import cv2
import time
import logging
import threading

from collections import deque
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

try:
    from temporal_analyzer import (
        _sample_window,
        _pack_slowfast,
        risk_level,
    )
except ImportError:
    from .temporal_analyzer import (
        _sample_window,
        _pack_slowfast,
        risk_level,
    )

logger = logging.getLogger(__name__)

# CPU optimization
CPU_THREADS = 4

if not torch.cuda.is_available():
    torch.set_num_threads(CPU_THREADS)


def live_camera_predictions(
    stream_url: str,
    model: torch.nn.Module,
    device: torch.device,
    *,
    backbone: str = "x3d_m",
    num_frames: int = 16,
    frame_size: int = 224,
    threshold: float = 0.4,
    inference_interval: float = 1.5,
) -> Iterator[tuple]:
    """
    CPU-optimized live violence detection.

    Camera capture runs in a background thread.
    Gradio receives preview frames independently of inference.
    """

    if not stream_url and stream_url != 0:
        raise ValueError("Camera stream URL or camera index is required.")

    # Parse numeric camera index or stream URL
    clean_source = str(stream_url).strip()
    source: int | str = int(clean_source) if clean_source.isdigit() else clean_source

    stop_event = threading.Event()
    lock = threading.Lock()

    latest_frame = None
    frame_buffer = deque(maxlen=num_frames)

    prediction = {
        "probability": 0.0,
        "risk": "Waiting",
        "status": "Connecting",
    }

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Cannot connect to camera. Check the stream URL and network."
        )

    # Request a smaller capture buffer where supported.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def capture_worker():
        nonlocal latest_frame

        while not stop_event.is_set():
            success, frame = cap.read()

            if not success:
                prediction["status"] = "Camera disconnected"
                stop_event.set()
                break

            # Reduce display resolution to lower CPU and transfer overhead.
            height, width = frame.shape[:2]

            if width > 640:
                scale = 640 / width
                frame = cv2.resize(
                    frame,
                    (640, int(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )

            with lock:
                latest_frame = frame.copy()
                frame_buffer.append(frame.copy())

        logger.info("Camera capture thread stopped.")

    capture_thread = threading.Thread(
        target=capture_worker,
        daemon=True,
    )

    model.eval()

    capture_thread.start()

    last_inference = 0.0

    logger.info("Live camera connected.")

    try:
        while not stop_event.is_set():

            with lock:
                frame = (
                    latest_frame.copy()
                    if latest_frame is not None
                    else None
                )

                enough_frames = len(frame_buffer) >= num_frames

                inference_frames = (
                    list(frame_buffer)
                    if enough_frames
                    else None
                )

            if frame is None:
                time.sleep(0.02)
                continue

            current_time = time.monotonic()

            # Run inference independently from the camera capture.
            if (
                inference_frames is not None
                and current_time - last_inference >= inference_interval
            ):

                last_inference = current_time

                try:
                    tensor = _sample_window(
                        inference_frames,
                        0,
                        len(inference_frames),
                        num_frames=num_frames,
                        frame_size=(frame_size, frame_size),
                        mean=(0.45, 0.45, 0.45),
                        std=(0.225, 0.225, 0.225),
                    ).to(device)

                    inputs = (
                        _pack_slowfast(tensor)
                        if "slowfast" in backbone
                        else tensor
                    )

                    with torch.inference_mode():
                        logits = model(inputs)

                        probabilities = F.softmax(
                            logits,
                            dim=1,
                        )[0].cpu().numpy()

                    violent_prob = float(probabilities[1])
                    current_risk, _ = risk_level(violent_prob)

                    with lock:
                        prediction["probability"] = violent_prob
                        prediction["risk"] = current_risk
                        prediction["status"] = "Monitoring"

                except Exception as inf_err:
                    logger.warning("Live inference step skipped: %s", inf_err)
                    with lock:
                        prediction["status"] = "Inference warning"

            with lock:
                current_risk = prediction["risk"]
                violent_prob = prediction["probability"]
                status = prediction["status"]

            display_frame = frame.copy()

            cv2.putText(
                display_frame,
                f"Risk: {current_risk.upper()}",
                (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                display_frame,
                f"Violence Probability: {violent_prob * 100:.1f}%",
                (15, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )

            frame_rgb = cv2.cvtColor(
                display_frame,
                cv2.COLOR_BGR2RGB,
            )

            yield (
                frame_rgb,
                current_risk,
                violent_prob,
                status,
            )

            # Limit Gradio update rate to avoid flooding the browser.
            time.sleep(0.04)

    finally:
        stop_event.set()

        capture_thread.join(timeout=2.0)

        cap.release()

        logger.info("Live camera released.")

