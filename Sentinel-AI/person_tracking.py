
import cv2
import sys
import time
from ultralytics import YOLO
from pathlib import Path

print("SENTINEL AI - HUMAN TRACKING")

# Project directory
project_dir = Path(__file__).resolve().parent

# Load YOLO model
model = YOLO(str(project_dir / "yolo11n.pt"))

# Open video
video_path = project_dir / "videos" / "sample_fixed.mp4"
cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    print(f"ERROR: Cannot open video: {video_path}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120 or fps != fps:
    fps = 25.0
frame_delay = max(1, int(1000 / fps))

print("Tracking started...")

while True:
    loop_start = time.perf_counter()
    success, frame = cap.read()

    if not success:
        break

    # Track people with persistent IDs
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        conf=0.4,
        tracker="bytetrack.yaml",
        verbose=False
    )

    # Draw tracking boxes and IDs
    annotated_frame = results[0].plot()

    # Count people
    person_count = len(results[0].boxes)

    cv2.putText(
        annotated_frame,
        f"People: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "SENTINEL AI - HUMAN TRACKING",
        annotated_frame
    )

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_time = max(1, int(frame_delay - elapsed_ms))
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Tracking completed!")
