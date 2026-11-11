"""
Sentinel AI - Fight & Violence Detection
Detects people and monitors for violent physical interactions in video.
"""

import sys
import time
import argparse
from pathlib import Path
from collections import deque

import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "fight_engine"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from fight_engine.model_loader import ViolenceModelLoader
from fight_engine.temporal_analyzer import _sample_window, _pack_slowfast, risk_level


def main():
    parser = argparse.ArgumentParser(description="Sentinel AI - Video Fight & Violence Detection")
    parser.add_argument(
        "--video",
        default=str(ROOT / "videos" / "sample_fixed.mp4"),
        help="Path to video file"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Person detection confidence (default: 0.45)"
    )
    parser.add_argument(
        "--model",
        default="final_x3d_realtime",
        help="Violence model name (default: final_x3d_realtime)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without displaying a GUI window"
    )
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n========================================")
    print("      SENTINEL AI: FIGHT DETECTION      ")
    print("========================================")
    print(f"Device: {device}")
    print(f"Video:  {video_path.name}")

    # Load YOLO
    yolo_path = ROOT / "yolo11n.pt"
    print(f"\nLoading person tracker from {yolo_path.name}...")
    yolo_model = YOLO(str(yolo_path))

    # Load Violence Model
    print(f"Loading violence detection model ({args.model})...")
    loader = ViolenceModelLoader(device=device, max_cached=1)
    violence_model, spec = loader.load(args.model)
    violence_model.eval()
    print("Models loaded successfully!")

    # Open video
    video = cv2.VideoCapture(str(video_path))
    if not video.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120 or fps != fps:
        fps = 25.0
    frame_delay = max(1, int(1000 / fps))

    num_frames = spec.num_frames
    threshold = spec.threshold
    frame_size = spec.frame_size
    backbone = spec.backbone

    frame_buffer = deque(maxlen=num_frames)
    tracked_people = []
    frame_count = 0
    violent_prob = 0.0
    risk = "Low"
    fight_detected = False

    print("\nStarting analysis (Press 'q' in video window to exit)...")

    try:
        while True:
            loop_start = time.perf_counter()
            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            frame_buffer.append(frame.copy())

            # Run YOLO person tracking every 3 frames
            if frame_count % 3 == 0:
                results = yolo_model.track(
                    frame,
                    persist=True,
                    classes=[0],
                    conf=args.conf,
                    imgsz=416,
                    tracker="bytetrack.yaml",
                    verbose=False
                )

                tracked_people = []
                if results and len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        person_id = int(box.id[0]) if box.id is not None else None
                        tracked_people.append((x1, y1, x2, y2, person_id))

            # Run Violence Model inference every num_frames // 2 frames
            if len(frame_buffer) == num_frames and frame_count % max(1, num_frames // 2) == 0:
                tensor = _sample_window(
                    list(frame_buffer),
                    0,
                    len(frame_buffer),
                    num_frames=num_frames,
                    frame_size=(frame_size, frame_size),
                    mean=(0.45, 0.45, 0.45),
                    std=(0.225, 0.225, 0.225),
                ).to(device)

                inputs = _pack_slowfast(tensor) if "slowfast" in backbone else tensor
                with torch.inference_mode():
                    logits = violence_model(inputs)
                    probs = F.softmax(logits, dim=1)[0].cpu().numpy()

                violent_prob = float(probs[1])
                risk, _ = risk_level(violent_prob)
                fight_detected = (violent_prob >= threshold)

            if not args.no_display:
                # Draw person boxes
                box_color = (0, 0, 255) if fight_detected else (0, 255, 0)
                for x1, y1, x2, y2, person_id in tracked_people:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                    lbl = f"Person {person_id}" if person_id is not None else "Person"
                    cv2.putText(
                        frame, lbl, (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2, cv2.LINE_AA
                    )

                # Draw top status banner
                overlay = frame.copy()
                cv2.rectangle(overlay, (15, 15), (420, 130), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
                cv2.rectangle(frame, (15, 15), (420, 130), (80, 80, 80), 1)

                cv2.putText(
                    frame, f"PEOPLE DETECTED: {len(tracked_people)}",
                    (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

                if fight_detected:
                    status_text = f"ALERT: FIGHT DETECTED ({violent_prob * 100:.1f}%)"
                    status_color = (0, 0, 255)
                    # Flashing screen border
                    if (frame_count // 5) % 2 == 0:
                        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 6)
                elif violent_prob >= threshold * 0.75:
                    status_text = f"WARNING: SUSPICIOUS ({violent_prob * 100:.1f}%)"
                    status_color = (0, 165, 255)
                else:
                    status_text = f"STATUS: NORMAL ({violent_prob * 100:.1f}%)"
                    status_color = (0, 220, 0)

                cv2.putText(
                    frame, status_text, (25, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA
                )

                # Violence Score Bar
                bar_x, bar_y, bar_w, bar_h = 25, 95, 260, 14
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 50, 50), -1)
                fill_w = int(bar_w * max(0.0, min(1.0, violent_prob)))
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), status_color, -1)
                cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (120, 120, 120), 1)

                cv2.imshow("Sentinel AI - Fight Detection", frame)

                elapsed_ms = (time.perf_counter() - loop_start) * 1000
                wait_time = max(1, int(frame_delay - elapsed_ms))
                if cv2.waitKey(wait_time) & 0xFF == ord("q"):
                    break

    finally:
        video.release()
        cv2.destroyAllWindows()
        print("\nAnalysis finished.")


if __name__ == "__main__":
    main()
