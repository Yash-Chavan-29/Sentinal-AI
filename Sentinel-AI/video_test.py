import cv2
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
video_path = PROJECT_DIR / "videos" / "sample_fixed.mp4"

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print(f"ERROR: Cannot open video: {video_path}")
    sys.exit(1)

fps = video.get(cv2.CAP_PROP_FPS)
if fps <= 0 or fps > 120 or fps != fps:
    fps = 25.0
frame_delay = max(1, int(1000 / fps))

print(f"Sentinel AI: Playing video '{video_path.name}' at {fps:.1f} FPS (press 'q' to stop)...")

while True:
    loop_start = time.perf_counter()
    success, frame = video.read()

    if not success:
        break

    cv2.imshow("Sentinel AI - Video Monitoring", frame)

    elapsed_ms = (time.perf_counter() - loop_start) * 1000
    wait_time = max(1, int(frame_delay - elapsed_ms))
    if cv2.waitKey(wait_time) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

print("Video playback completed!")