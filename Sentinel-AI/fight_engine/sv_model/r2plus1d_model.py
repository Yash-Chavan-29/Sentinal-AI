"""
R(2+1)D Model for Violence Detection

R(2+1)D decomposes 3D convolutions into separate spatial (2D) and temporal (1D) components,
providing better efficiency and accuracy compared to standard 3D CNNs.

Key Features:
- Uses pretrained R(2+1)D-18 from torchvision
- Separates spatial (2D) and temporal (1D) convolutions
- More efficient than standard 3D convolutions
- Good balance of speed and accuracy

References:
- Paper: "A Closer Look at Spatiotemporal Convolutions for Action Recognition" (CVPR 2018)
- https://arxiv.org/abs/1711.11248

Author: Safe School AI Team
Date: November 2025
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class ViolenceR2Plus1D(nn.Module):
    """
    R(2+1)D-18 model for violence detection.
    
    The R(2+1)D architecture decomposes 3D convolutions into:
    1. A 2D spatial convolution (1 × d × d)
    2. A 1D temporal convolution (t × 1 × 1)
    
    This decomposition:
    - Reduces parameters while maintaining expressiveness
    - Makes optimization easier
    - Often achieves better accuracy than full 3D convolutions
    
    Attributes:
        backbone: R(2+1)D-18 pretrained on Kinetics-400
        num_classes: Number of output classes
        dropout_rate: Dropout probability
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        """
        Initialize R(2+1)D model.
        
        Args:
            num_classes: Number of output classes (default: 2 for Violence/Non-Violence)
            pretrained: Whether to use Kinetics-400 pretrained weights
            dropout: Dropout probability for regularization
        """
        super(ViolenceR2Plus1D, self).__init__()
        
        # Load pretrained R(2+1)D-18
        if pretrained:
            weights = R2Plus1D_18_Weights.KINETICS400_V1
            self.backbone = r2plus1d_18(weights=weights)
            logger.info("Loading pretrained R(2+1)D-18 from torchvision (Kinetics-400)")
        else:
            self.backbone = r2plus1d_18(weights=None)
            logger.info("Initializing R(2+1)D-18 with random weights")
        
        # Get feature dimension from original fc layer
        in_features = self.backbone.fc.in_features
        
        # Replace classification head
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        logger.info(f"R(2+1)D model initialized: in_features={in_features}, "
                   f"num_classes={num_classes}, pretrained={pretrained}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               B = batch size
               C = 3 (RGB channels)
               T = 16 (temporal frames)
               H, W = 224, 224 (spatial dimensions)
        
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
        # Forward through all layers except fc
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        # Global average pooling
        x = self.backbone.avgpool(x)
        
        # Flatten
        x = x.flatten(1)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True):
        """
        Freeze/unfreeze backbone layers.
        
        Args:
            freeze: If True, freeze all layers except classification head
        """
        # Freeze all layers except fc
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:
                param.requires_grad = not freeze
        
        # Always keep fc trainable
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        status = "frozen" if freeze else "unfrozen"
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        logger.info(f"Backbone {status}. Trainable: {trainable:,} / {total:,}")
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get detailed parameter counts.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'frozen' parameter counts
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }


def create_r2plus1d_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> ViolenceR2Plus1D:
    """
    Factory function to create R(2+1)D model.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
    
    Returns:
        ViolenceR2Plus1D model instance
    
    Example:
        >>> model = create_r2plus1d_model(num_classes=2, pretrained=True)
        >>> x = torch.randn(2, 3, 16, 224, 224)
        >>> output = model(x)
        >>> print(output.shape)  # torch.Size([2, 2])
    """
    model = ViolenceR2Plus1D(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
    return model
