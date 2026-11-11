"""
Sentinel-AI Core Configuration
"""
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent

# Paths to models
YOLO_MODEL_PATH = BASE_DIR / "Sentinel-AI" / "yolo11n.pt"
X3D_MODEL_PATH = BASE_DIR / "Sentinel-AI" / "final" / "final_x3d_realtime.pt"
SNAPSHOT_DIR = BASE_DIR / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# Hardware setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Video and Tracking Parameters
DEFAULT_CAMERA_SOURCE = "sample"  # Starts with high-quality sample feed, can be switched to webcam or phone IP instantly
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
YOLO_CONFIDENCE = 0.35

# Action / Violence Recognition Parameters
X3D_NUM_FRAMES = 16
X3D_FRAME_SIZE = 224
X3D_THRESHOLD = 0.40  # 0.40 calibrated post-ablation threshold
CONSECUTIVE_ALERT_WINDOWS = 2
MIN_PROXIMITY_PERSON_COUNT = 2  # Spatial filter: Require 2+ people in proximity for fight alarm

# Server Settings
WEB_HOST = "0.0.0.0"
WEB_PORT = 7860
