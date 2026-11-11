"""
Sentinel AI - Unified People Tracking & Real-Time Fight Detection
Combines YOLO11n ByteTrack object tracking with X3D-M 3D CNN Violence Detection.
"""

import sys
import time
import argparse
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# Project root and engine setup
ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "fight_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from fight_engine.model_loader import ViolenceModelLoader
from fight_engine.temporal_analyzer import _sample_window, _pack_slowfast, risk_level

# Object classes
CLASS_NAMES = {
    0: "Person",
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

PERSON_COLOR = (0, 255, 0)
VEHICLE_COLOR = (255, 150, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Sentinel AI - Unified Tracking & Fight Detection")
    parser.add_argument(
        "--mode",
        choices=["video", "webcam", "phone_wifi", "phone_usb"],
        default="video",
        help="Input mode: video, webcam, phone_wifi, phone_usb (default: video)"
    )
    parser.add_argument(
        "--video",
        default=str(ROOT / "videos" / "sample_fixed.mp4"),
        help="Path to video file"
    )
    parser.add_argument(
        "--stream",
        default="http://10.219.151.72:8080/video",
        help="Phone WiFi stream URL"
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device index for webcam/USB (default: 0)"
    )
    parser.add_argument(
        "--model",
        default="final_x3d_realtime",
        help="Violence model key (default: final_x3d_realtime)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="YOLO detection confidence threshold (default: 0.4)"
    )
    parser.add_argument(
        "--yolo-size",
        type=int,
        default=320,
        help="YOLO input size (default: 320)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without GUI display window (headless mode)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=======================================================")
    print("    SENTINEL AI: UNIFIED TRACKING & FIGHT DETECTION   ")
    print("=======================================================")
    print(f"Device: {device}")
    print(f"Mode:   {args.mode}")

    # 1. Open Video Source
    if args.mode == "video":
        source = args.video
    elif args.mode == "phone_wifi":
        source = args.stream
    elif args.mode in ("webcam", "phone_usb"):
        source = args.camera
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"\n[ERROR] Unable to open video source: {source}")
        if args.mode == "phone_wifi":
            print("[HINT] Check phone webcam IP address or run with: python sentinel_fight_detection.py --mode video")
        sys.exit(1)

    if args.mode != "video":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or fps != fps:
        fps = 25.0
    frame_delay = max(1, int(1000 / fps))

    # 2. Load Models
    yolo_model_path = ROOT / "yolo11n.pt"
    print(f"\nLoading YOLO11n tracking model from {yolo_model_path}...")
    yolo_model = YOLO(str(yolo_model_path))

    print(f"Loading 3D CNN Violence Detection model ({args.model})...")
    loader = ViolenceModelLoader(device=device, max_cached=1)
    violence_model, spec = loader.load(args.model)
    violence_model.eval()
    print("All models loaded successfully!\n")

    # 3. Setup Fight Inference Shared State
    num_frames = spec.num_frames
    frame_size = spec.frame_size
    threshold = spec.threshold
    backbone = spec.backbone

    window_buffer = deque(maxlen=num_frames)
    fight_state = {
        "prob": 0.0,
        "risk": "Low",
        "label": "SAFE",
        "active": False,
        "updated_at": time.time(),
    }
    fight_lock = threading.Lock()
    stop_event = threading.Event()

    def violence_worker():
        """Background worker thread to continuously process violence inference."""
        while not stop_event.is_set():
            frames_to_process = None
            with fight_lock:
                if len(window_buffer) >= num_frames:
                    frames_to_process = list(window_buffer)

            if frames_to_process is None:
                time.sleep(0.05)
                continue

            try:
                tensor = _sample_window(
                    frames_to_process,
                    0,
                    len(frames_to_process),
                    num_frames=num_frames,
                    frame_size=(frame_size, frame_size),
                    mean=(0.45, 0.45, 0.45),
                    std=(0.225, 0.225, 0.225),
                ).to(device)

                inputs = _pack_slowfast(tensor) if "slowfast" in backbone else tensor
                with torch.inference_mode():
                    logits = violence_model(inputs)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                v_prob = float(probs[1])
                level, _ = risk_level(v_prob)

                if v_prob >= threshold:
                    lbl = "FIGHT DETECTED!"
                    active = True
                elif v_prob >= threshold * 0.75:
                    lbl = "SUSPICIOUS MOTION"
                    active = False
                else:
                    lbl = "SAFE"
                    active = False

                with fight_lock:
                    fight_state["prob"] = v_prob
                    fight_state["risk"] = level
                    fight_state["label"] = lbl
                    fight_state["active"] = active
                    fight_state["updated_at"] = time.time()

            except Exception as e:
                pass

            time.sleep(0.1)

    worker_thread = threading.Thread(target=violence_worker, daemon=True)
    worker_thread.start()

    # 4. Processing Loop
    frame_count = 0
    tracked_objects = []
    smooth_fps = fps
    prev_time = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                print("\nVideo playback complete or camera disconnected.")
                break

            frame_count += 1

            # Append frame copy for violence window
            with fight_lock:
                window_buffer.append(frame.copy())

            # YOLO Tracking every 3 frames
            if frame_count % 3 == 0:
                results = yolo_model.track(
                    frame,
                    persist=True,
                    classes=[0, 2, 3, 5, 7],
                    conf=args.conf,
                    imgsz=args.yolo_size,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                tracked_objects = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        class_id = int(box.cls[0])
                        obj_name = CLASS_NAMES.get(class_id, "Unknown")
                        obj_id = int(box.id[0]) if box.id is not None else None
                        tracked_objects.append((x1, y1, x2, y2, obj_id, obj_name))

            if not args.no_display:
                # ---------------- DRAW BOXES ----------------
                person_count = 0
                vehicle_count = 0

                for x1, y1, x2, y2, obj_id, obj_name in tracked_objects:
                    if obj_name == "Person":
                        color = PERSON_COLOR
                        person_count += 1
                    else:
                        color = VEHICLE_COLOR
                        vehicle_count += 1

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl = f"{obj_name} ID:{obj_id}" if obj_id is not None else obj_name
                    cv2.putText(
                        frame,
                        lbl,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                # ---------------- FPS ----------------
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - prev_time, 0.0001)
                prev_time = now
                smooth_fps = 0.85 * smooth_fps + 0.15 * instant_fps

                # ---------------- READ FIGHT STATE ----------------
                with fight_lock:
                    v_prob = fight_state["prob"]
                    risk = fight_state["risk"]
                    label = fight_state["label"]
                    is_fight = fight_state["active"]

                # ---------------- HUD BANNER ----------------
                overlay = frame.copy()
                cv2.rectangle(overlay, (10, 10), (450, 175), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (10, 10), (450, 175), (80, 80, 80), 1)

                cv2.putText(frame, "SENTINEL AI: SECURITY MONITOR", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

                cv2.putText(frame, f"Tracked: {person_count} People | {vehicle_count} Vehicles", (20, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.putText(frame, f"FPS: {smooth_fps:.1f} | Source: {args.mode}", (20, 92),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # Violence Alert Status
                if is_fight:
                    status_color = (0, 0, 255)      # Red
                    alert_txt = f"ALERT: {label} ({v_prob * 100:.1f}%)"
                    # Flashing border
                    if int(time.time() * 4) % 2 == 0:
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)
                elif v_prob >= threshold * 0.75:
                    status_color = (0, 165, 255)    # Amber
                    alert_txt = f"WARNING: {label} ({v_prob * 100:.1f}%)"
                else:
                    status_color = (0, 220, 0)      # Green
                    alert_txt = f"STATUS: {label} ({v_prob * 100:.1f}%)"

                cv2.putText(frame, alert_txt, (20, 125),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

                # Probability Progress Bar
                bar_x, bar_y, bar_w, bar_h = 20, 140, 280, 14
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * max(0.0, min(1.0, v_prob)))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

                cv2.imshow("Sentinel AI - People & Fight Detection", frame)

                if args.mode == "video":
                    elapsed_ms = (time.perf_counter() - loop_start) * 1000
                    wait_time = max(1, int(frame_delay - elapsed_ms))
                else:
                    wait_time = 1

                if cv2.waitKey(wait_time) & 0xFF == ord('q'):
                    break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        print("\nSentinel AI stopped cleanly.")


if __name__ == "__main__":
    main()

