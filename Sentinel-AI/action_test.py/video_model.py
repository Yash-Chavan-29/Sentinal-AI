
import cv2
import torch
import numpy as np
import sys
from pathlib import Path
from torchvision.models.video import r3d_18, R3D_18_Weights

# ==========================================
# SENTINEL AI - ACTION RECOGNITION
# ==========================================

print("\n===================================")
print("       SENTINEL AI")
print("     ACTION RECOGNITION")
print("===================================\n")

# ==========================================
# 1. DEVICE CONFIGURATION
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# ==========================================
# 2. VIDEO PATH
# ==========================================

script_dir = Path(__file__).resolve().parent
project_dir = script_dir.parent

video_path = project_dir / "videos" / "sample_fixed.mp4"

print("Video path:", video_path)

if not video_path.exists():
    print("\nERROR: Video file not found!")
    sys.exit(1)

# ==========================================
# 3. LOAD ACTION RECOGNITION MODEL
# ==========================================

print("\nLoading action-recognition model...")

weights = R3D_18_Weights.DEFAULT

model = r3d_18(weights=weights)

model = model.to(device)
model.eval()

print("Model loaded successfully!")

# ==========================================
# 4. OPEN VIDEO
# ==========================================

print("\nOpening video...")

video = cv2.VideoCapture(str(video_path))

if not video.isOpened():
    print("ERROR: Could not open video.")
    sys.exit(1)

total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)

print("Total frames (metadata):", total_frames)
print("FPS:", fps)

# ==========================================
# 5. EXTRACT 16 FRAMES
# ==========================================

print("\nReading video frames...")

raw_frames = []
while True:
    success, frame = video.read()
    if not success:
        break
    raw_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

video.release()

print("Total frames read:", len(raw_frames))

if len(raw_frames) == 0:
    print("\nERROR: Video has 0 frames or could not be decoded.")
    sys.exit(1)

# Sample or pad to exactly 16 frames
if len(raw_frames) >= 16:
    indices = np.linspace(0, len(raw_frames) - 1, 16, dtype=int)
    frames = [raw_frames[i] for i in indices]
else:
    # Repeat / tile frames if video has fewer than 16 frames
    reps = int(np.ceil(16 / len(raw_frames)))
    tiled = (raw_frames * reps)[:16]
    frames = tiled

print("Frames extracted:", len(frames))

# ==========================================
# 6. PREPROCESS FRAMES
# ==========================================

print("\nPreprocessing video...")

clip = np.stack(frames)

# T, H, W, C -> T, C, H, W
clip = torch.from_numpy(clip).permute(
    0, 3, 1, 2
)

clip = clip.float()

preprocess = weights.transforms()

# Output: C, T, H, W
clip = preprocess(clip)

# Add batch dimension
clip = clip.unsqueeze(0)

clip = clip.to(device)

print("Tensor shape:", clip.shape)

# ==========================================
# 7. ACTION PREDICTION
# ==========================================

print("\nAnalysing video...")

with torch.no_grad():

    prediction = model(clip)

probabilities = torch.nn.functional.softmax(
    prediction[0],
    dim=0
)

top5 = torch.topk(
    probabilities,
    5
)

categories = weights.meta["categories"]

# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

print("\n===================================")
print("       ACTIVITY REPORT")
print("===================================\n")

for score, index in zip(
    top5.values,
    top5.indices
):

    activity = categories[index.item()]

    confidence = score.item() * 100

    print(
        f"{activity}: {confidence:.2f}%"
    )

print("\n===================================")
print("     VIDEO ANALYSIS COMPLETED")
print("===================================")
