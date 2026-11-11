"""
Model Package for Violence Detection

This package provides modular model implementations for violence detection
in video data. Each model architecture is in a separate file for better
organization and maintainability.

Available Models:
- X3D: Efficient 3D CNN (x3d_s, x3d_m, x3d_l)
- R(2+1)D: Factorized 3D CNN (r2plus1d_18)

Main Entry Point:
- Use `create_model()` factory function to instantiate models

Utilities:
- freeze_backbone(): Freeze/unfreeze backbone layers
- freeze_specific_layers(): Fine-grained layer control
- get_model_info(): Get model statistics
- print_model_summary(): Print formatted model info

Example:
    >>> from model import create_model
    >>> model = create_model('x3d_m', num_classes=2, pretrained=True)
    >>> model = create_model('r2plus1d_18', num_classes=2, pretrained=True)
    >>> # Or import specific model class
    >>> from model import ViolenceX3D, ViolenceR2Plus1D
    >>> model = ViolenceX3D('x3d_m', num_classes=2)

Author: Safe School AI Team
Date: November 2025
"""

# Import model classes (relative imports — package was renamed to `sv_model` to
# avoid clashing with the `model/` directory in the source repo)
from .x3d_model import ViolenceX3D, create_x3d_model
from .i3d_model import ViolenceI3D, create_i3d_model
from .slowfast_model import ViolenceSlowFast, create_slowfast_model
from .r2plus1d_model import ViolenceR2Plus1D, create_r2plus1d_model

# Import factory and utilities from violence_model
from .violence_model import (
    create_model,
    freeze_backbone,
    freeze_specific_layers,
    get_model_info,
    print_model_summary,
)

# Define what gets imported with "from model import *"
__all__ = [
    # Model classes
    'ViolenceX3D',
    'ViolenceI3D',
    'ViolenceSlowFast',
    'ViolenceR2Plus1D',
    
    # Factory functions
    'create_model',
    'create_x3d_model',
    'create_i3d_model',
    'create_slowfast_model',
    'create_r2plus1d_model',
    
    # Utility functions
    'freeze_backbone',
    'freeze_specific_layers',
    'get_model_info',
    'print_model_summary',
]

# Version info
__version__ = '1.0.0'
__author__ = 'Safe School AI Team'
