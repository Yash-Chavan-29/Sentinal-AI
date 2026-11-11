"""
Video preprocessing utilities for the Gradio Space.

Tự chứa, không phụ thuộc vào repo source code.
"""

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

BROWSER_PLAYABLE_SUFFIXES = {".mp4", ".webm", ".ogg", ".ogv", ".m4v"}


def extract_uniform_frames(video_path: Union[str, Path], num_frames: int = 32) -> np.ndarray:
    """Đọc video và lấy uniformly num_frames frames.

    Returns:
        ndarray shape (T, H, W, 3) RGB uint8.
    """
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        # Fallback: đọc lần lượt
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise RuntimeError(f"No frames in video: {video_path}")
        total = len(frames)
        frames_np = np.stack(frames, axis=0)
    else:
        indices = np.linspace(0, max(total - 1, 0), num=num_frames).astype(int)
        target_set = set(indices.tolist())
        collected = {}
        idx = 0
        while idx <= max(indices):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in target_set:
                collected[idx] = frame
            idx += 1
        cap.release()

        # Lấp các index không đọc được bằng frame gần nhất
        frames_np = []
        last = None
        for i in indices:
            if i in collected:
                last = collected[i]
            if last is None:
                # Trường hợp video lỗi từ đầu
                raise RuntimeError(f"Failed to read frames from {video_path}")
            frames_np.append(last)
        frames_np = np.stack(frames_np, axis=0)

    # Nếu chưa đủ num_frames thì padding bằng repeat
    if frames_np.shape[0] < num_frames:
        reps = int(np.ceil(num_frames / frames_np.shape[0]))
        frames_np = np.tile(frames_np, (reps, 1, 1, 1))[:num_frames]
    elif frames_np.shape[0] > num_frames:
        indices = np.linspace(0, frames_np.shape[0] - 1, num=num_frames).astype(int)
        frames_np = frames_np[indices]

    # BGR -> RGB
    frames_np = frames_np[..., ::-1].copy()
    return frames_np


def preprocess_video(
    video_path: Union[str, Path],
    num_frames: int = 32,
    frame_size: Tuple[int, int] = (224, 224),
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225),
) -> torch.Tensor:
    """Đọc video và trả về tensor shape (1, 3, T, H, W) đã normalize."""
    frames = extract_uniform_frames(video_path, num_frames=num_frames)

    # Resize từng frame
    h, w = frame_size
    resized = np.stack(
        [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in frames],
        axis=0,
    )

    # (T, H, W, C) -> (C, T, H, W), float32 [0,1]
    tensor = torch.from_numpy(resized).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2).contiguous()

    # Normalize
    mean_t = torch.tensor(mean).view(3, 1, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1, 1)
    tensor = (tensor - mean_t) / std_t

    # Add batch dim
    return tensor.unsqueeze(0)


def _transcode_with_ffmpeg(src: Path, dst: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-i", str(src),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-an",
        str(dst),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    return dst.exists() and dst.stat().st_size > 0


def _transcode_with_opencv(src: Path, dst: Path) -> bool:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w <= 0 or h <= 0:
        cap.release()
        return False
    # mp4v: widely supported by OpenCV builds; produces an MP4 most browsers play
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    cap.release()
    return dst.exists() and dst.stat().st_size > 0


def ensure_browser_playable(video_path: Optional[Union[str, Path]]) -> Optional[str]:
    """Trả về đường dẫn video mà <video> tag HTML5 có thể phát.

    - Nếu file đã ở định dạng browser-playable (mp4/webm/ogg) → trả về nguyên.
    - Nếu là .avi (hoặc định dạng khác) → transcode sang .mp4 trong thư mục tạm
      bằng FFmpeg (ưu tiên) hoặc OpenCV (fallback).
    - Nếu mọi cách đều fail → trả về đường dẫn gốc (analyze vẫn chạy, chỉ là
      preview HTML5 sẽ không hiển thị video).
    """
    if video_path is None:
        return None
    src = Path(str(video_path))
    if not src.exists():
        return str(src)
    if src.suffix.lower() in BROWSER_PLAYABLE_SUFFIXES:
        return str(src)

    tmp_dir = Path(tempfile.gettempdir()) / "svd_playable"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / (src.stem + "__playable.mp4")
    if dst.exists() and dst.stat().st_size > 0:
        return str(dst)

    if _transcode_with_ffmpeg(src, dst):
        return str(dst)
    if _transcode_with_opencv(src, dst):
        return str(dst)
    return str(src)


def pack_slowfast_pathway(video_tensor: torch.Tensor, alpha: int = 4) -> list:
    """SlowFast cần input dạng [slow, fast]. Subsample fast pathway theo alpha."""
    # video_tensor: (B, C, T, H, W)
    fast = video_tensor
    slow_indices = torch.linspace(0, video_tensor.shape[2] - 1, video_tensor.shape[2] // alpha).long()
    slow = torch.index_select(video_tensor, dim=2, index=slow_indices)
    return [slow, fast]
