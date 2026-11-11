import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

PROJECT_DIR = Path(__file__).resolve().parent
local_target = PROJECT_DIR / "final" / "final_x3d_realtime.pt"

print("Sentinel AI: Checking Fight Detection Model...")

if local_target.exists() and local_target.stat().st_size > 1024 * 1024:
    print(f"Model already exists locally at: {local_target}")
    print(f"Model size: {round(local_target.stat().st_size / (1024 * 1024), 2)} MB")
else:
    print("Downloading Sentinel-AI Fight Detection Model from HuggingFace...")
    model_path = hf_hub_download(
        repo_id="visionlab-ai/school-violence-detection-models",
        filename="final/final_x3d_realtime.pt"
    )
    local_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, local_target)
    print("\nDownload successful!")
    print("Model cached at:", model_path)
    print("Model saved to:", local_target)
    print("Model size:", round(os.path.getsize(model_path) / (1024 * 1024), 2), "MB")