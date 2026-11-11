import cv2
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
video_path = PROJECT_DIR / "videos" / "sample_fixed.mp4"
frames_dir = PROJECT_DIR / "frames"

frames_dir.mkdir(parents=True, exist_ok=True)

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print(f"ERROR: Cannot open video: {video_path}")
    sys.exit(1)

frame_count = 0
saved_count = 0

print(f"Starting frame capture from {video_path.name}...")

while True:
    success, frame = video.read()

    if not success:
        break

    # Save one frame every 30 frames
    if frame_count % 30 == 0:
        filename = frames_dir / f"frame_{frame_count}.jpg"
        cv2.imwrite(str(filename), frame)
        saved_count += 1
        print("Saved:", filename.name)

    frame_count += 1

video.release()

print("\nFrame capturing completed!")
print(f"Total frames processed: {frame_count} | Frames saved: {saved_count}")