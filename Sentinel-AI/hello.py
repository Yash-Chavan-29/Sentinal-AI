
import os
from pathlib import Path

project_dir = Path(__file__).resolve().parent
video_path = project_dir / "videos" / "sample_fixed.mp4"

print("Sentinel AI: Environment Check")
print("Project Directory:", project_dir)
print("Video exists:", video_path.exists())
print("Full path:", video_path)

videos_dir = project_dir / "videos"
if videos_dir.exists():
    print("Videos folder contents:", os.listdir(videos_dir))
