"""
Sentinel-AI Unified Launcher
Runs the tactical live surveillance dashboard and AI processing pipeline.
Usage:
    python run.py
    python run.py --camera 0
    python run.py --camera "http://192.168.1.100:8080/video" --port 7860
"""

import sys
import argparse
import uvicorn
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import WEB_HOST, WEB_PORT, DEFAULT_CAMERA_SOURCE


def main():
    parser = argparse.ArgumentParser(description="Sentinel-AI Live Threat Detection System")
    parser.add_argument("--host", default=WEB_HOST, help=f"Host address (default: {WEB_HOST})")
    parser.add_argument("--port", type=int, default=WEB_PORT, help=f"Port number (default: {WEB_PORT})")
    parser.add_argument("--camera", default=str(DEFAULT_CAMERA_SOURCE), help="Camera source (device index or IP stream URL)")
    args = parser.parse_args()

    print("=" * 65)
    print("      🛡️  SENTINEL-AI LIVE SURVEILLANCE & THREAT DETECTION")
    print("=" * 65)
    print(f" ▸ Host Address    : http://{args.host}:{args.port}")
    print(f" ▸ Initial Camera  : {args.camera}")
    print(f" ▸ Pipeline        : YOLO11n ByteTrack + X3D-M 3D CNN")
    print("=" * 65)

    uvicorn.run("web.app:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
