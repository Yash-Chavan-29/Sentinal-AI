"""
X3D Model for Violence Detection

X3D is an efficient 3D CNN architecture for video classification.
This module provides X3D-based models (x3d_s, x3d_m, x3d_l) pretrained on Kinetics-400.

Author: Safe School AI Team
Date: November 2025
"""

import torch
import torch.nn as nn
import logging
from typing import Dict

# Check if pytorchvideo is available
try:
    import pytorchvideo  # noqa: F401
    PYTORCHVIDEO_AVAILABLE = True
except ImportError:
    PYTORCHVIDEO_AVAILABLE = False

logger = logging.getLogger(__name__)


class ViolenceX3D(nn.Module):
    """
    X3D-based violence detection model.
    
    X3D is an efficient 3D CNN architecture for video classification.
    This class wraps the pretrained X3D model from PyTorchVideo and
    adds a custom classification head for binary violence detection.
    
    Args:
        model_name: Name of X3D variant ('x3d_s', 'x3d_m', 'x3d_l')
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to load pretrained weights from Kinetics
        dropout: Dropout probability for classification head
    """
    
    def __init__(
        self,
        model_name: str = 'x3d_m',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Check if pytorchvideo is available
        if not PYTORCHVIDEO_AVAILABLE:
            raise ImportError(
                "pytorchvideo is not installed. Install it with: pip install pytorchvideo"
            )
        
        # Load X3D model using torch hub (recommended approach for pretrained models)
        try:
            logger.info(f"Loading {model_name} (pretrained={pretrained})...")

            # Normalize model name
            if model_name == 'x3d':
                model_name = 'x3d_m'

            valid_models = ['x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l']
            if model_name not in valid_models:
                logger.warning(f"Unknown model {model_name}, defaulting to x3d_m")
                model_name = 'x3d_m'

            # Load model using torch hub (best approach for pytorchvideo)
            self.backbone = torch.hub.load(
                'facebookresearch/pytorchvideo',
                model_name,
                pretrained=pretrained
            )
            logger.info(f"Loaded {model_name} via torch hub (pretrained={pretrained})")

        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Get the number of features from the original head
        if hasattr(self.backbone, 'blocks'):
            # X3D architecture
            if hasattr(self.backbone.blocks[-1], 'proj'):
                in_features = self.backbone.blocks[-1].proj.in_features
            elif hasattr(self.backbone.blocks[-1], 'linear'):
                in_features = self.backbone.blocks[-1].linear.in_features
            else:
                # Fallback - try to get from output layer
                in_features = 2048
        else:
            in_features = 2048  # Default fallback
        
        # Replace the classification head
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        logger.info(f"X3D model initialized: {model_name}, in_features={in_features}, "
                   f"num_classes={num_classes}, pretrained={pretrained}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B: batch size, C: channels (3), T: temporal frames,
               H: height, W: width
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (without classification head).
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Feature tensor of shape (B, feature_dim)
        """
        # Forward through all blocks except the final projection
        for i, block in enumerate(self.backbone.blocks[:-1]):
            x = block(x)
        
        # Apply pooling from last block if available
        if hasattr(self.backbone.blocks[-1], 'pool'):
            x = self.backbone.blocks[-1].pool(x)
        
        # Flatten
        x = x.flatten(1)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze backbone parameters.
        
        Args:
            freeze: If True, freeze backbone. If False, unfreeze.
        """
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        
        # Always keep classification head trainable
        for param in self.backbone.blocks[-1].proj.parameters():
            param.requires_grad = True
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Backbone {status} (classification head remains trainable)")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'frozen' parameter counts
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }


def create_x3d_model(
    model_name: str = 'x3d_m',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> ViolenceX3D:
    """
    Factory function to create X3D model.
    
    Args:
        model_name: X3D variant ('x3d_s', 'x3d_m', 'x3d_l')
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        dropout: Dropout probability
    
    Returns:
        ViolenceX3D model instance
    """
    return ViolenceX3D(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
