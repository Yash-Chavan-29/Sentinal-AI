"""
Model loader for the Gradio Space.

- Tải checkpoint từ HF Hub model repo `visionlab-ai/school-violence-detection-models`.
- Build model wrapper từ package `sv_model` (đã copy từ source code).
- Cache LRU đơn giản giữ 1 model trong RAM cho HF Spaces free tier.
"""

from __future__ import annotations

import gc
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from pathlib import Path
from huggingface_hub import hf_hub_download

_THIS_DIR = Path(__file__).resolve().parent
_SV_DIR = _THIS_DIR / "sv_model"
import sys
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
if str(_SV_DIR) not in sys.path:
    sys.path.insert(0, str(_SV_DIR))

try:
    from sv_model import create_model
except ImportError:
    from .sv_model import create_model

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


MODEL_REPO_ID = os.environ.get(
    "VIOLENCE_MODEL_REPO_ID",
    "visionlab-ai/school-violence-detection-models",
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    backbone: str
    repo_path: str
    accuracy: float
    parameters_m: float
    description: str
    num_frames: int = 32
    frame_size: int = 224
    threshold: float = 0.5
    is_final: bool = False  # True nếu là model sau ablation


MODEL_SPECS: Dict[str, ModelSpec] = {
    # ── ⭐ FINAL (post-ablation) ──────────────────────────────────────
    "final_x3d_realtime": ModelSpec(
        name="final_x3d_realtime",
        display_name="⭐ Realtime — X3D",
        backbone="x3d_m",
        repo_path="final/final_x3d_realtime.pt",
        accuracy=0.93,
        parameters_m=3.0,
        description=(
            "X3D-M tuned for realtime (post-ablation, full retrain @16 frames). "
            "F1=0.93, GPU latency ~3.77 ms — model nhanh nhất với accuracy tốt nhất."
        ),
        num_frames=16,
        threshold=0.4,
        is_final=True,
    ),
    "final_best": ModelSpec(
        name="final_best",
        display_name="⭐ Best Accuracy — R(2+1)D",
        backbone="r2plus1d_18",
        repo_path="final/final_best.pt",
        accuracy=0.93,
        parameters_m=31.3,
        description=(
            "R(2+1)D-18 best config (Step 1 winner, Step 2 ablation applied). "
            "F1=0.93, GPU latency ~21 ms — accuracy cao nhất."
        ),
        num_frames=32,
        threshold=0.5,
        is_final=True,
    ),

    # ── Baselines (pre-ablation) ──────────────────────────────────────
    "x3d_m": ModelSpec(
        name="x3d_m",
        display_name="Baseline — X3D",
        backbone="x3d_m",
        repo_path="x3d_m/x3d_m_best.pt",
        accuracy=0.92,
        parameters_m=3.0,
        description="X3D-M baseline trước ablation, ~92% accuracy.",
    ),
    "i3d_r50": ModelSpec(
        name="i3d_r50",
        display_name="Baseline — I3D",
        backbone="i3d_r50",
        repo_path="i3d_r50/i3d_r50_best.pt",
        accuracy=0.9075,
        parameters_m=27.2,
        description="I3D Inflated 3D CNN ResNet-50 baseline, ~90.75% accuracy.",
    ),
    "slowfast_r50": ModelSpec(
        name="slowfast_r50",
        display_name="Baseline — SlowFast",
        backbone="slowfast_r50",
        repo_path="slowfast_r50/slowfast_best.pt",
        accuracy=0.925,
        parameters_m=33.6,
        description="SlowFast dual-pathway baseline, ~92.5% accuracy. Chậm trên CPU.",
    ),
    "r2plus1d_r18": ModelSpec(
        name="r2plus1d_r18",
        display_name="Baseline — R(2+1)D",
        backbone="r2plus1d_18",
        repo_path="r2plus1d_18/r2plus1d_best.pt",
        accuracy=0.93,
        parameters_m=31.3,
        description="R(2+1)D-18 baseline trước ablation, ~93% accuracy.",
    ),
}


# ── Early-warning config (giống nhau cho mọi model -- fair comparison) ──
# Threshold quyết định = spec.threshold (riêng cho final_x3d=0.4, còn lại 0.5).
# Warning được trigger khi `EARLY_WARNING_CONSECUTIVE` cửa sổ liên tiếp đều
# vượt ngưỡng (max-pool decision rule sau smoothing).
EARLY_WARNING_CONSECUTIVE = 2
TIMELINE_WINDOWS = 6  # số window chia video cho timeline confidence
TIMELINE_OVERLAP = 0.5  # window overlap (0 = không overlap, 0.5 = nửa)


def list_final_specs() -> List[ModelSpec]:
    return [s for s in MODEL_SPECS.values() if s.is_final]


def list_baseline_specs() -> List[ModelSpec]:
    return [s for s in MODEL_SPECS.values() if not s.is_final]


class ViolenceModelLoader:
    """Lazy loader giữ tối đa N model trong RAM."""

    def __init__(self, device: str = "cpu", max_cached: int = 1) -> None:
        self.device = torch.device(device)
        self.max_cached = max_cached
        self._cache: Dict[str, Tuple[nn.Module, ModelSpec]] = {}
        self._order: List[str] = []
        self.current_name: Optional[str] = None

    @property
    def specs(self) -> Dict[str, ModelSpec]:
        return MODEL_SPECS

    def list_display_names(self) -> List[str]:
        return [s.display_name for s in MODEL_SPECS.values()]

    def display_to_name(self, display_name: str) -> Optional[str]:
        for s in MODEL_SPECS.values():
            if s.display_name == display_name:
                return s.name
        return None

    def _evict_if_full(self) -> None:
        while len(self._cache) >= self.max_cached and self._order:
            oldest = self._order.pop(0)
            self._cache.pop(oldest, None)
            logger.info("Evicted %s from cache", oldest)
        gc.collect()

    def load(self, name: str) -> Tuple[nn.Module, ModelSpec]:
        if name not in MODEL_SPECS:
            raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_SPECS)}")

        spec = MODEL_SPECS[name]

        if name in self._cache:
            self._order.remove(name)
            self._order.append(name)
            self.current_name = name
            return self._cache[name]

        # Check local paths first before attempting remote download
        local_candidates = [
            _THIS_DIR / spec.repo_path,
            _THIS_DIR.parent / spec.repo_path,
            Path(spec.repo_path),
            _THIS_DIR / Path(spec.repo_path).name,
            _THIS_DIR.parent / "final" / Path(spec.repo_path).name,
        ]
        local_ckpt = next((p for p in local_candidates if p.is_file()), None)

        if local_ckpt:
            ckpt_path = str(local_ckpt)
            logger.info("Using local checkpoint for %s at %s", name, ckpt_path)
        else:
            logger.info("Downloading checkpoint for %s from %s ...", name, MODEL_REPO_ID)
            try:
                ckpt_path = hf_hub_download(
                    repo_id=MODEL_REPO_ID,
                    filename=spec.repo_path,
                )
                logger.info("Checkpoint cached at %s", ckpt_path)
            except Exception as dl_err:
                # If hub download fails, search for any local matching filename
                basename = Path(spec.repo_path).name
                fallback = next((p for p in [_THIS_DIR / basename, _THIS_DIR.parent / "final" / basename] if p.is_file()), None)
                if fallback:
                    ckpt_path = str(fallback)
                    logger.warning("Remote download failed (%s). Falling back to local file: %s", dl_err, ckpt_path)
                else:
                    raise

        # Build architecture (pretrained=False vì state_dict đã chứa weights)
        model = create_model(
            backbone=spec.backbone,
            num_classes=2,
            pretrained=False,
            dropout=0.5,
        )

        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning("Missing keys when loading %s: %d", name, len(missing))
        if unexpected:
            logger.warning("Unexpected keys when loading %s: %d", name, len(unexpected))

        model = model.to(self.device).eval()

        self._evict_if_full()
        self._cache[name] = (model, spec)
        self._order.append(name)
        self.current_name = name
        return model, spec

    def get_current(self) -> Optional[Tuple[nn.Module, ModelSpec]]:
        if self.current_name and self.current_name in self._cache:
            return self._cache[self.current_name]
        return None
