"""
Fight & Action Classifier for Sentinel-AI (Production Hackathon Edition)
Combines:
1. 3D CNN Temporal Recognition (X3D-M Backbone)
2. Illumination-Invariant Preprocessing (CLAHE on L-channel)
3. Dynamic Adaptive Thresholding (Rolling Ambient Baseline Mu + K*Sigma)
4. Long-Horizon Leaky Memory Accumulator (Multi-Window Threat Retention)
"""

import sys
import logging
from pathlib import Path
from collections import deque
from typing import Tuple, Dict, Any, Optional, List

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import (
    X3D_MODEL_PATH,
    DEVICE,
    X3D_NUM_FRAMES,
    X3D_FRAME_SIZE,
    X3D_THRESHOLD,
    MIN_PROXIMITY_PERSON_COUNT
)

# PyTorch 2.x Safe Globals
try:
    import numpy.dtypes
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([
            numpy.core.multiarray.scalar,
            numpy.dtype,
            numpy.dtypes.Float64DType,
            numpy.dtypes.Int64DType
        ])
except Exception:
    pass

logger = logging.getLogger(__name__)

# Include sv_model path
_BASE = Path(__file__).resolve().parent.parent / "Sentinel-AI" / "fight_engine"
if str(_BASE) not in sys.path:
    sys.path.insert(0, str(_BASE))
if str(_BASE / "sv_model") not in sys.path:
    sys.path.insert(0, str(_BASE / "sv_model"))

try:
    from sv_model import create_model
except ImportError:
    from fight_engine.sv_model import create_model


class FightClassifier:
    def __init__(self, weights_path: str = str(X3D_MODEL_PATH), device: str = DEVICE):
        self.device = torch.device(device)
        self.num_frames = X3D_NUM_FRAMES
        self.frame_size = X3D_FRAME_SIZE
        self.base_threshold = X3D_THRESHOLD  # 0.40 baseline

        self.frame_buffer = deque(maxlen=self.num_frames)
        self.prev_gray: Optional[np.ndarray] = None

        # Upgrade 1: CLAHE Illumination Equalizer
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Upgrade 2: Dynamic Ambient Baseline (Rolling statistics)
        self.ambient_kinetic_history = deque(maxlen=60)  # ~4 seconds of rolling environmental background
        self.ambient_mu = 0.05
        self.ambient_sigma = 0.02

        # Upgrade 3: Long-Horizon Leaky Memory Accumulator (Markov Threat State)
        self.threat_memory = 0.0
        self.memory_decay = 0.88  # Retains 88% of threat state across windows for sustained escalation detection
        self.instant_alpha = 0.55

        # Normalization constants (Kinetics-400)
        self.mean = torch.tensor([0.45, 0.45, 0.45], dtype=torch.float32).view(1, 3, 1, 1, 1).to(self.device)
        self.std = torch.tensor([0.225, 0.225, 0.225], dtype=torch.float32).view(1, 3, 1, 1, 1).to(self.device)

        self.model = self._load_model(weights_path)
        logger.info("Adaptive FightClassifier initialized on %s with CLAHE, Leaky Memory, and Dynamic Baseline", self.device)

    def _load_model(self, weights_path: str) -> nn.Module:
        model = create_model(backbone="x3d_m", num_classes=2, pretrained=False, dropout=0.5)
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict, strict=False)
        return model.to(self.device).eval()

    def _apply_clahe_normalization(self, bgr_frame: np.ndarray) -> np.ndarray:
        """Normalizes illumination via LAB color space CLAHE."""
        lab = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def add_frame(self, frame: np.ndarray):
        """Preprocesses frame with illumination equalization and updates rolling ambient baseline."""
        # Illumination normalization
        norm_frame = self._apply_clahe_normalization(frame)
        resized = cv2.resize(norm_frame, (self.frame_size, self.frame_size))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.frame_buffer.append(rgb)

        # Optical kinetic calculation
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            kinetic_val = float(np.mean(diff)) / 22.0
            self.ambient_kinetic_history.append(kinetic_val)
            if len(self.ambient_kinetic_history) >= 15:
                self.ambient_mu = float(np.mean(self.ambient_kinetic_history))
                self.ambient_sigma = float(np.std(self.ambient_kinetic_history))
        self.prev_gray = gray

    def analyze(self, has_proximity_cluster: bool, person_count: int) -> Tuple[float, str, bool, Dict[str, Any]]:
        """
        Executes mathematical fused analysis:
        P_fused = (w1 * P_x3d + w2 * E_kinetic_relative)
        M_t = gamma * M_{t-1} + (1 - gamma) * P_fused
        """
        if len(self.frame_buffer) < self.num_frames:
            return 0.0, "Buffering", False, {"buffered": len(self.frame_buffer), "total": self.num_frames}

        # 1. 5D Tensor Preprocessing
        frames_np = np.stack(list(self.frame_buffer), axis=0)
        tensor = torch.from_numpy(frames_np).float() / 255.0
        tensor = tensor.permute(3, 0, 1, 2).unsqueeze(0).to(self.device)
        tensor = (tensor - self.mean) / self.std

        # 2. X3D Forward Pass
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            raw_prob = float(probs[0, 1].item())

        # 3. Dynamic Thresholding relative to ambient scene noise
        latest_kinetic = self.ambient_kinetic_history[-1] if self.ambient_kinetic_history else 0.0
        # Kinetic deviation above baseline
        kinetic_z_score = max(0.0, (latest_kinetic - self.ambient_mu) / (self.ambient_sigma + 1e-5))
        relative_kinetic_prob = min(1.0, kinetic_z_score / 3.5)

        instant_fused = (0.65 * raw_prob) + (0.35 * relative_kinetic_prob)

        # 4. Long-Horizon Leaky Memory Accumulator
        # Retains threat state across windows; escalates quickly during sustained scuffles
        self.threat_memory = (self.memory_decay * self.threat_memory) + ((1.0 - self.memory_decay) * instant_fused)
        effective_threat = max(instant_fused, self.threat_memory)

        # 5. Adaptive Decision Boundary
        # Multi-person proximity lowers threshold; solo action requires higher confidence
        multi_person_cleared = (person_count >= MIN_PROXIMITY_PERSON_COUNT and has_proximity_cluster)
        
        dynamic_threshold = self.base_threshold if not multi_person_cleared else (self.base_threshold - 0.06)

        confirmed_alarm = False
        if effective_threat >= dynamic_threshold:
            risk = "Altercation"
            confirmed_alarm = True
        elif effective_threat >= (dynamic_threshold * 0.60):
            risk = "Suspicious"
        else:
            risk = "Normal"

        meta = {
            "x3d_prob": round(raw_prob, 3),
            "threat_memory": round(self.threat_memory, 3),
            "effective_threat": round(effective_threat, 3),
            "ambient_mu": round(self.ambient_mu, 3),
            "dynamic_threshold": round(dynamic_threshold, 3),
            "person_count": person_count
        }

        return effective_threat, risk, confirmed_alarm, meta
