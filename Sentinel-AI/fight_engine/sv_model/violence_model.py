"""
Violence Detection Model Factory

This module serves as the main entry point for creating violence detection models.
It imports model classes from separate files and provides utilities for model management.

Supported models:
- X3D (x3d_s, x3d_m, x3d_l) - Efficient 3D CNN from PyTorchVideo- I3D (i3d_r50) - Inflated 3D CNN from PyTorchVideo- SlowFast (slowfast_r50, slowfast_r101) - Dual-pathway 3D CNN from PyTorchVideo
- R(2+1)D (r2plus1d_18) - Factorized 3D CNN from torchvision

Utilities:
- Model creation factory
- Freezing/unfreezing layers for transfer learning
- Model information and summaries

Author: Safe School AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import logging

# Import model classes from separate files
from .x3d_model import ViolenceX3D, create_x3d_model
from .i3d_model import ViolenceI3D, create_i3d_model
from .slowfast_model import ViolenceSlowFast, create_slowfast_model
from .r2plus1d_model import ViolenceR2Plus1D, create_r2plus1d_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Re-export model classes for backward compatibility
__all__ = [
    'ViolenceX3D',
    'ViolenceI3D',
    'ViolenceSlowFast',
    'ViolenceR2Plus1D',
    'create_model',
    'freeze_backbone',
    'freeze_specific_layers',
    'get_model_info',
    'print_model_summary'
]


def create_model(
    backbone: str = 'x3d_m',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> nn.Module:
    """
    Factory function to create violence detection model.
    
    Args:
        backbone: Model backbone name
                 X3D: 'x3d_s', 'x3d_m', 'x3d_l'
                 I3D: 'i3d_r50'
                 SlowFast: 'slowfast_r50', 'slowfast_r101'
                 R(2+1)D: 'r2plus1d_18', 'r2plus1d'
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        dropout: Dropout probability
    
    Returns:
        PyTorch model instance
    
    Example:
        >>> # Create X3D model
        >>> model = create_model('x3d_m', num_classes=2, pretrained=True)
        >>> # Create I3D model
        >>> model = create_model('i3d_r50', num_classes=2, pretrained=True)
        >>> # Create SlowFast model
        >>> model = create_model('slowfast_r50', num_classes=2, pretrained=True)
        >>> # Create R(2+1)D model
        >>> model = create_model('r2plus1d_18', num_classes=2, pretrained=True)
    """
    backbone = backbone.lower()
    
    if backbone in ['x3d_s', 'x3d_m', 'x3d_l', 'x3d_xs']:
        return create_x3d_model(
            model_name=backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif backbone in ['i3d_r50', 'i3d']:
        # Default to i3d_r50 if just 'i3d' is specified
        model_name = backbone if backbone.startswith('i3d_') else 'i3d_r50'
        return create_i3d_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif backbone in ['slowfast_r50', 'slowfast_r101', 'slowfast']:
        # Default to slowfast_r50 if just 'slowfast' is specified
        model_name = backbone if backbone.startswith('slowfast_') else 'slowfast_r50'
        return create_slowfast_model(
            model_name=model_name,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    elif backbone in ['r2plus1d_18', 'r2plus1d']:
        return create_r2plus1d_model(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}. "
                        f"Supported: x3d_s, x3d_m, x3d_l, i3d_r50, slowfast_r50, slowfast_r101, r2plus1d_18")


def freeze_backbone(model: nn.Module, freeze: bool = True) -> None:
    """
    Freeze or unfreeze the entire backbone (all layers except classification head).
    
    This is useful for transfer learning - freeze backbone to train only
    the classification head initially, then unfreeze to fine-tune.
    
    Args:
        model: ViolenceX3D model instance
        freeze: If True, freeze backbone; if False, unfreeze it
    
    Example:
        >>> model = create_model('x3d_m')
        >>> freeze_backbone(model, freeze=True)  # Freeze for head-only training
        >>> # Train for a few epochs...
        >>> freeze_backbone(model, freeze=False)  # Unfreeze for full fine-tuning
    """
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model
    
    # Freeze/unfreeze all blocks except the last one (classification head)
    if hasattr(backbone, 'blocks'):
        for i, block in enumerate(backbone.blocks[:-1]):
            for param in block.parameters():
                param.requires_grad = not freeze
        
        # Always keep the classification head trainable
        for param in backbone.blocks[-1].parameters():
            param.requires_grad = True
    else:
        # Fallback: freeze all parameters except those with 'proj' or 'fc' in name
        for name, param in backbone.named_parameters():
            if 'proj' not in name and 'fc' not in name:
                param.requires_grad = not freeze
    
    status = "frozen" if freeze else "unfrozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Backbone {status}. Trainable parameters: {trainable:,} / {total:,}")


def freeze_specific_layers(
    model: nn.Module,
    layer_names: List[str],
    freeze: bool = True
) -> None:
    """
    Freeze or unfreeze specific layers by name pattern.
    
    This provides fine-grained control over which layers to freeze/unfreeze.
    Useful for progressive unfreezing during fine-tuning.
    
    Args:
        model: ViolenceX3D model instance
        layer_names: List of layer name patterns to match (e.g., ['blocks.4', 'blocks.5'])
        freeze: If True, freeze matched layers; if False, unfreeze them
    
    Example:
        >>> model = create_model('x3d_m')
        >>> # Freeze only early blocks
        >>> freeze_specific_layers(model, ['blocks.0', 'blocks.1', 'blocks.2'], freeze=True)
        >>> # Unfreeze later blocks for fine-tuning
        >>> freeze_specific_layers(model, ['blocks.4', 'blocks.5'], freeze=False)
    """
    if hasattr(model, 'backbone'):
        backbone = model.backbone
    else:
        backbone = model
    
    matched_params = 0
    for name, param in backbone.named_parameters():
        # Check if any layer pattern matches this parameter name
        if any(pattern in name for pattern in layer_names):
            param.requires_grad = not freeze
            matched_params += 1
    
    status = "frozen" if freeze else "unfrozen"
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"{matched_params} parameters in {layer_names} {status}. "
               f"Total trainable: {trainable:,} / {total:,}")


def get_model_info(model: nn.Module) -> Dict[str, any]:
    """
    Get comprehensive information about the model.
    
    Args:
        model: PyTorch model instance
    
    Returns:
        Dictionary containing model statistics
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': frozen_params,
        'trainable_percentage': (trainable_params / total_params * 100) if total_params > 0 else 0,
        'model_size_mb': total_params * 4 / (1024 ** 2),  # Assuming float32
    }
    
    return info


def print_model_summary(model: nn.Module) -> None:
    """
    Print a formatted summary of the model architecture and parameters.
    
    Args:
        model: PyTorch model instance
    """
    info = get_model_info(model)
    
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print(f"Total parameters:      {info['total_parameters']:,}")
    print(f"Trainable parameters:  {info['trainable_parameters']:,}")
    print(f"Frozen parameters:     {info['frozen_parameters']:,}")
    print(f"Trainable percentage:  {info['trainable_percentage']:.2f}%")
    print(f"Model size:            {info['model_size_mb']:.2f} MB")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing X3D model creation...")
    model_x3d = create_model('x3d_m', num_classes=2, pretrained=False)
    print_model_summary(model_x3d)
    
    # Test freezing
    print("\nFreezing backbone...")
    freeze_backbone(model_x3d, freeze=True)
    print_model_summary(model_x3d)
    
    # Test specific layer freezing
    print("\nUnfreezing blocks 4 and 5...")
    freeze_specific_layers(model_x3d, ['blocks.4', 'blocks.5'], freeze=False)
    print_model_summary(model_x3d)
    
    # Test forward pass
    print("\nTesting forward pass...")
    dummy_input = torch.randn(1, 3, 16, 224, 224)  # (B, C, T, H, W)
    output = model_x3d(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n[OK] All tests passed!")
