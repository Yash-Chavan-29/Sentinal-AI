"""
Download Checkpoints for Sentinel-AI
Downloads the fine-tuned X3D-M realtime violence detection model from Hugging Face.
"""
import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT_DIR = Path(__file__).resolve().parent
target_path = ROOT_DIR / "Sentinel-AI" / "final" / "final_x3d_realtime.pt"

print("=" * 60)
print("  Sentinel-AI Checkpoint Downloader")
print("=" * 60)

if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
    print(f"✅ Checkpoint already present at: {target_path}")
    print(f"✅ File size: {round(target_path.stat().st_size / (1024 * 1024), 2)} MB")
else:
    print("⏳ Downloading final_x3d_realtime.pt from HuggingFace Hub...")
    cached_path = hf_hub_download(
        repo_id="visionlab-ai/school-violence-detection-models",
        filename="final/final_x3d_realtime.pt"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached_path, target_path)
    print(f"✅ Download complete! Saved to {target_path}")

print("=" * 60)
