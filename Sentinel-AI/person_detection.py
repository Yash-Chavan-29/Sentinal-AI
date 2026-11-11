
import cv2
import sys
import time
from ultralytics import YOLO
from pathlib import Path

# =====================================
# SENTINEL AI - PERSON DETECTION
# =====================================

print("Loading Sentinel AI...")

# Project directory
project_dir = Path(__file__).resolve().parent

# Video path
video_path = project_dir / "videos" / "sample_fixed.mp4"

# Load YOLO model
model = YOLO(str(project_dir / "yolo11n.pt"))

print("YOLO model loaded successfully!")

# Open video
video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print(f"ERROR: Cannot open video: {video_path}")
    sys.exit(1)

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120 or fps != fps:
    fps = 25.0
frame_delay = max(1, int(1000 / fps))

print("\nSentinel AI: Detecting people...")

while True:
    loop_start = time.perf_counter()
    success, frame = video.read()

    if not success:
        break

    # Detect only people (COCO class 0)
    results = model.predict(
        frame,
        classes=[0],
        conf=0.4,
        verbose=False
    )

    # Draw detection boxes
    annotated_frame = results[0].plot()

    # Count detected people
    person_count = len(results[0].boxes)

    # Display person count
    cv2.putText(
        annotated_frame,
        f"People Detected: {person_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Display video
    cv2.imshow(
        "SENTINEL AI - PERSON DETECTION",
        annotated_frame
    )

    # Press Q to exit
    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_time = max(1, int(frame_delay - elapsed_ms))
    if cv2.waitKey(wait_time) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

print("\nPerson detection completed!")
