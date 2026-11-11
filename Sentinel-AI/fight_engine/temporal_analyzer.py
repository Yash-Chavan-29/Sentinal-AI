"""
Temporal analysis: chia video thành sliding windows, chạy inference từng window,
trả về timeline confidence + early-warning state.

Đây là **inference thật**, không phải mock — mỗi window chạy forward pass riêng.

Module này expose 2 cấp API:

* `analyze_video_timeline(...)` — synchronous one-shot, đơn giản nhất.
* `iter_video_timeline_steps(...)` — generator, yield mỗi bước để UI có thể
  cập nhật progress bar mà không cần spinner trên các output panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class TimelineWindow:
    """1 cửa sổ thời gian + confidence violent."""

    index: int
    start_sec: float
    end_sec: float
    violent_prob: float

    @property
    def mid_sec(self) -> float:
        return (self.start_sec + self.end_sec) / 2


@dataclass
class TimelineResult:
    """Kết quả phân tích timeline cho một video."""

    windows: List[TimelineWindow]
    duration_sec: float
    fps: float
    threshold: float
    consecutive_required: int
    inference_ms_total: float
    pre_ms_total: float
    aggregated_violent_prob: float  # max-pool across windows (smoothing decision rule)
    aggregated_non_violent_prob: float

    # Early warning state
    state: str  # "Safe" | "Warning" | "Violent"
    state_emoji: str
    consecutive_count: int  # max chuỗi window liên tiếp vượt threshold


def _read_video_frames(
    video_path: Union[str, Path],
) -> Tuple[List[np.ndarray], float, float]:
    """Đọc toàn bộ frames của video. Trả về (frames_bgr, fps, duration_sec)."""
    path = str(video_path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 0:
        fps = 30.0

    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames in video: {path}")

    duration = len(frames) / fps
    return frames, fps, duration


def _sample_window(
    frames: List[np.ndarray],
    start_idx: int,
    end_idx: int,
    num_frames: int,
    frame_size: Tuple[int, int],
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
) -> torch.Tensor:
    """Uniform-sample `num_frames` từ slice [start_idx, end_idx) và preprocess."""
    end_idx = min(end_idx, len(frames))
    start_idx = max(0, min(start_idx, end_idx - 1))
    indices = np.linspace(start_idx, end_idx - 1, num=num_frames).astype(int)

    h, w = frame_size
    arr = np.stack(
        [cv2.resize(frames[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in indices],
        axis=0,
    )
    # BGR -> RGB
    arr = arr[..., ::-1].copy()

    tensor = torch.from_numpy(arr).float() / 255.0
    tensor = tensor.permute(3, 0, 1, 2).contiguous()  # (C, T, H, W)

    mean_t = torch.tensor(mean).view(3, 1, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1, 1)
    tensor = (tensor - mean_t) / std_t

    return tensor.unsqueeze(0)  # (1, C, T, H, W)


def _pack_slowfast(video_tensor: torch.Tensor, alpha: int = 4) -> List[torch.Tensor]:
    """SlowFast 2-pathway input."""
    fast = video_tensor
    slow_indices = torch.linspace(
        0, video_tensor.shape[2] - 1, video_tensor.shape[2] // alpha
    ).long()
    slow = torch.index_select(video_tensor, dim=2, index=slow_indices)
    return [slow, fast]


def _plan_windows(
    n_frames: int, num_windows: int, overlap: float, min_window_len: int
) -> Tuple[List[int], int]:
    """Trả về (start_indices, window_len) cho sliding-window scheme."""
    if num_windows < 1:
        num_windows = 1
    if num_windows == 1:
        return [0], n_frames

    denom = 1 + (1 - overlap) * (num_windows - 1)
    window_len = max(int(round(n_frames / denom)), min_window_len)
    stride = max(int(round(window_len * (1 - overlap))), 1)
    starts = [min(i * stride, n_frames - window_len) for i in range(num_windows)]
    return starts, window_len


def _aggregate(
    windows: List[TimelineWindow],
    threshold: float,
    consecutive_required: int,
) -> Tuple[float, float, str, str, int]:
    """Aggregate per-window probs → (violent_p, non_violent_p, state, emoji, max_consec)."""
    violent_probs = np.array([w.violent_prob for w in windows]) if windows else np.array([0.0])
    aggregated_violent = float(violent_probs.max())
    aggregated_non_violent = 1.0 - aggregated_violent

    above = violent_probs >= threshold
    max_consec = 0
    cur = 0
    for flag in above:
        if flag:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    if max_consec >= consecutive_required:
        state, emoji = "Violent", "🚨"
    elif above.any() or aggregated_violent >= threshold * 0.75:
        state, emoji = "Warning", "⚠️"
    else:
        state, emoji = "Safe", "✅"

    return aggregated_violent, aggregated_non_violent, state, emoji, max_consec


def iter_video_timeline_steps(
    video_path: Union[str, Path],
    model: torch.nn.Module,
    device: torch.device,
    *,
    backbone: str,
    num_frames: int,
    frame_size: int,
    threshold: float,
    num_windows: int,
    overlap: float,
    consecutive_required: int,
    mean: Tuple[float, float, float] = (0.45, 0.45, 0.45),
    std: Tuple[float, float, float] = (0.225, 0.225, 0.225),
) -> Iterator[Tuple[str, float, "TimelineResult | None"]]:
    """Generator yield (step_label, percent, result_so_far_or_None).

    Steps:
      1. Loading video
      2. Sampling frames (window plan)
      3. Running 3D CNN inference (per-window, progress đếm theo window)
      4. Temporal smoothing
      5. Generating alert
    """
    import time

    # Step 1
    yield ("Loading video", 5.0, None)
    t0 = time.time()
    frames, fps, duration = _read_video_frames(video_path)
    n_frames = len(frames)
    yield ("Loading video", 20.0, None)

    # Step 2
    yield ("Sampling frames", 25.0, None)
    starts, window_len = _plan_windows(
        n_frames, num_windows=num_windows, overlap=overlap, min_window_len=num_frames
    )
    pre_total_ms = (time.time() - t0) * 1000
    yield ("Sampling frames", 35.0, None)

    # Step 3 — inference (chiếm phần lớn thời gian: progress từ 35 -> 80)
    windows: List[TimelineWindow] = []
    inf_total_ms = 0.0
    n_total = len(starts)
    inf_low, inf_high = 35.0, 80.0
    inf_range = inf_high - inf_low

    for i, start_idx in enumerate(starts):
        yield (
            f"Running 3D CNN inference (window {i + 1}/{n_total})",
            inf_low + inf_range * (i / max(n_total, 1)),
            None,
        )

        end_idx = min(start_idx + window_len, n_frames)

        t_pre = time.time()
        tensor = _sample_window(
            frames,
            start_idx,
            end_idx,
            num_frames=num_frames,
            frame_size=(frame_size, frame_size),
            mean=mean,
            std=std,
        ).to(device)
        pre_total_ms += (time.time() - t_pre) * 1000

        inputs = _pack_slowfast(tensor) if "slowfast" in backbone else tensor

        t_inf = time.time()
        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        inf_total_ms += (time.time() - t_inf) * 1000

        windows.append(
            TimelineWindow(
                index=i,
                start_sec=start_idx / fps,
                end_sec=end_idx / fps,
                violent_prob=float(probs[1]),
            )
        )

    yield (
        f"Running 3D CNN inference (window {n_total}/{n_total})",
        inf_high,
        None,
    )

    # Step 4
    yield ("Temporal smoothing", 88.0, None)
    aggregated_violent, aggregated_non_violent, state, emoji, max_consec = _aggregate(
        windows, threshold=threshold, consecutive_required=consecutive_required
    )

    # Step 5
    yield ("Generating alert", 95.0, None)
    result = TimelineResult(
        windows=windows,
        duration_sec=duration,
        fps=fps,
        threshold=threshold,
        consecutive_required=consecutive_required,
        inference_ms_total=inf_total_ms,
        pre_ms_total=pre_total_ms,
        aggregated_violent_prob=aggregated_violent,
        aggregated_non_violent_prob=aggregated_non_violent,
        state=state,
        state_emoji=emoji,
        consecutive_count=max_consec,
    )
    yield ("Analysis complete", 100.0, result)


def analyze_video_timeline(*args, **kwargs) -> TimelineResult:
    """One-shot convenience wrapper for callers that don't want progress."""
    result = None
    for _, _, r in iter_video_timeline_steps(*args, **kwargs):
        if r is not None:
            result = r
    assert result is not None
    return result


def risk_level(violent_prob: float) -> Tuple[str, str]:
    """Map xác suất violent → (level, emoji)."""
    if violent_prob >= 0.7:
        return "High", "🔴"
    if violent_prob >= 0.4:
        return "Medium", "🟠"
    return "Low", "🟢"
