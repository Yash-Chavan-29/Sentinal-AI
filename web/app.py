"""
FastAPI Application for Sentinel-AI Live Surveillance
Provides:
- High-throughput MJPEG live video feed (Decoupled 30+ FPS)
- Real-time telemetry API
- Camera source switching API
- Incident evidence vault & snapshots
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import cv2

# Project path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from core.config import SNAPSHOT_DIR, DEFAULT_CAMERA_SOURCE
from core.pipeline import SurveillancePipeline

app = FastAPI(title="Sentinel-AI Live Surveillance")

# Mount snapshots
app.mount("/snapshots", StaticFiles(directory=str(SNAPSHOT_DIR)), name="snapshots")

# Initialize pipeline
pipeline = SurveillancePipeline(camera_source=DEFAULT_CAMERA_SOURCE)
pipeline.start()


class CameraSourceRequest(BaseModel):
    source: Any


@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    html_file = Path(__file__).resolve().parent / "dashboard.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


def generate_frames():
    """Generates continuous MJPEG frames instantly from pre-encoded buffer."""
    while True:
        jpeg_bytes = pipeline.get_latest_jpeg()
        if jpeg_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n')
        time.sleep(0.033)  # Solid 30 FPS


@app.get("/video_feed")
async def video_feed():
    """MJPEG Streaming endpoint."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns current system telemetry."""
    return pipeline.current_telemetry


@app.post("/api/camera")
async def switch_camera(req: CameraSourceRequest):
    """Switches the camera source dynamically."""
    success = pipeline.set_camera(req.source)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to switch camera source")
    return {"success": True, "source": str(req.source)}


@app.get("/api/snapshots")
async def list_snapshots():
    """Lists saved incident snapshots."""
    return pipeline.incident_log


@app.on_event("shutdown")
def shutdown_event():
    pipeline.stop()
