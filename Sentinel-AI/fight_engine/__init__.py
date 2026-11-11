"""
Sentinel AI - Fight Detection Engine Package
"""

import sys
from pathlib import Path

# Ensure fight_engine and sv_model are in sys.path
_ENGINE_DIR = Path(__file__).resolve().parent
if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

_SV_MODEL_DIR = _ENGINE_DIR / "sv_model"
if str(_SV_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_SV_MODEL_DIR))

from .model_loader import ViolenceModelLoader, MODEL_SPECS
from .temporal_analyzer import analyze_video_timeline, iter_video_timeline_steps, risk_level

__all__ = [
    "ViolenceModelLoader",
    "MODEL_SPECS",
    "analyze_video_timeline",
    "iter_video_timeline_steps",
    "risk_level",
]

