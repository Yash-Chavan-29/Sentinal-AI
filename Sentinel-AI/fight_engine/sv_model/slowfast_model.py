"""
SlowFast Model for Violence Detection

SlowFast is a state-of-the-art 3D CNN architecture for video understanding.
It uses a dual-pathway architecture:
- Slow pathway: Operates at low frame rate to capture spatial semantics
- Fast pathway: Operates at high frame rate to capture motion dynamics

This module provides SlowFast-R50 model pretrained on Kinetics-400.

Key Features:
- Dual-pathway design for efficient spatiotemporal modeling
- Higher accuracy than X3D but more memory intensive
- Pretrained on Kinetics-400 for strong transfer learning

References:
- Paper: "SlowFast Networks for Video Recognition" (ICCV 2019)
- https://arxiv.org/abs/1812.03982

Author: Safe School AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ViolenceSlowFast(nn.Module):
    """
    SlowFast-based violence detection model.
    
    SlowFast uses two pathways:
    1. Slow pathway (τ frames at low FPS) - captures spatial semantics
    2. Fast pathway (τ·α frames at high FPS) - captures temporal dynamics
    
    SlowFast-R50 configuration: α=4, β=8
    For 32 input frames:
    - Slow pathway: 8 frames (every 4th frame)
    - Fast pathway: 32 frames (all frames)
    
    This class wraps the pretrained SlowFast model from PyTorchVideo and
    adds a custom classification head for binary violence detection.
    
    Args:
        model_name: Name of SlowFast variant ('slowfast_r50', 'slowfast_r101')
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to load pretrained weights from Kinetics
        dropout: Dropout probability for classification head
    
    Attributes:
        backbone: SlowFast backbone network
        num_classes: Number of output classes
        model_name: Name of the model variant
    """
    
    def __init__(
        self,
        model_name: str = 'slowfast_r50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained SlowFast model from PyTorchVideo hub
        try:
            if pretrained:
                logger.info(f"Loading pretrained {model_name} from PyTorchVideo hub...")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    model=model_name,
                    pretrained=True
                )
            else:
                logger.info(f"Initializing {model_name} with random weights...")
                self.backbone = torch.hub.load(
                    'facebookresearch/pytorchvideo',
                    model=model_name,
                    pretrained=False
                )
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {e}")
            raise
        
        # Get the number of features from the original head
        # SlowFast architecture has blocks structure
        if hasattr(self.backbone, 'blocks'):
            # Get the last block's projection layer
            if hasattr(self.backbone.blocks[-1], 'proj'):
                if isinstance(self.backbone.blocks[-1].proj, nn.Linear):
                    in_features = self.backbone.blocks[-1].proj.in_features
                else:
                    # It's a Sequential or other module
                    for module in self.backbone.blocks[-1].proj.modules():
                        if isinstance(module, nn.Linear):
                            in_features = module.in_features
                            break
                    else:
                        in_features = 2304  # Default for SlowFast-R50
            else:
                in_features = 2304  # Default fallback
        else:
            in_features = 2304  # Default fallback
        
        # Replace the classification head with custom head for violence detection
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        logger.info(f"SlowFast model initialized: {model_name}, in_features={in_features}, "
                   f"num_classes={num_classes}, pretrained={pretrained}")
        logger.info(f"  Slow pathway: Low frame rate (spatial semantics)")
        logger.info(f"  Fast pathway: High frame rate (motion dynamics)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through SlowFast network.
        
        Args:
            x: Input tensor can be either:
               1. Single tensor (B, C, T, H, W) - will be split into slow/fast
               2. List [slow_tensor, fast_tensor] - already split
               
               Where:
               - B: batch size
               - C: channels (3 for RGB)
               - T: temporal frames (recommended: 32 for SlowFast-R50)
               - H: height (224)
               - W: width (224)
        
        Returns:
            Logits of shape (B, num_classes)
        
        Important:
            SlowFast-R50 uses alpha=4 (temporal ratio between pathways).
            For 32 input frames:
            - Slow pathway: 8 frames (every 4th frame)
            - Fast pathway: 32 frames (all frames)
        """
        # SlowFast expects input as list [slow_pathway, fast_pathway]
        if isinstance(x, list):
            # Already formatted correctly
            return self.backbone(x)
        
        # Split single tensor into slow and fast pathways
        # CRITICAL: SlowFast-R50 pretrained model uses alpha=4, not alpha=8!
        # alpha=4 means: fast_frames = 4 * slow_frames
        alpha = 4  # Temporal sampling ratio for SlowFast-R50
        
        B, C, T, H, W = x.shape
        
        # Validate that T is divisible by alpha
        if T % alpha != 0:
            # Provide helpful error message
            raise ValueError(
                f"Number of frames (T={T}) must be divisible by alpha ({alpha}).\n"
                f"For SlowFast-R50 with alpha=4:\n"
                f"  - Recommended: T=32 → slow=8 frames, fast=32 frames\n"
                f"  - Alternative: T=64 → slow=16 frames, fast=64 frames\n"
                f"  - Current: T={T} is not divisible by {alpha}"
            )
        
        # Slow pathway: subsample every alpha-th frame
        # For T=32, alpha=4: slow gets frames [0, 4, 8, 12, 16, 20, 24, 28] = 8 frames
        slow = x[:, :, ::alpha, :, :]  # (B, C, T/4, H, W)
        
        # Fast pathway: use all frames
        # For T=32: fast gets all 32 frames
        fast = x  # (B, C, T, H, W)
        
        # Create input list [slow, fast] as expected by SlowFast
        x = [slow, fast]
        
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (without classification head).
        
        Useful for:
        - Feature visualization
        - Transfer learning
        - Feature-based similarity search
        
        Args:
            x: Input tensor of shape (B, C, T, H, W) or list [slow, fast]
        
        Returns:
            Feature tensor of shape (B, feature_dim)
            where feature_dim = 2304 for SlowFast-R50
        """
        # Forward through all blocks except the final projection
        for i, block in enumerate(self.backbone.blocks[:-1]):
            x = block(x)
        
        # Apply pooling from last block if available
        if hasattr(self.backbone.blocks[-1], 'pool'):
            x = self.backbone.blocks[-1].pool(x)
        
        # Flatten spatial dimensions
        if isinstance(x, list):
            # If output is list (slow/fast pathways), concatenate them
            x = torch.cat([pathway.flatten(1) for pathway in x], dim=1)
        else:
            x = x.flatten(1)
        
        return x
    
    def freeze_backbone(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze the backbone network (except classification head).
        
        Useful for transfer learning:
        1. Freeze backbone initially, train only classification head
        2. Then unfreeze and fine-tune entire network
        
        Args:
            freeze: If True, freeze backbone. If False, unfreeze.
        """
        # Freeze all blocks except the last one (classification head)
        for i, block in enumerate(self.backbone.blocks[:-1]):
            for param in block.parameters():
                param.requires_grad = not freeze
        
        status = "frozen" if freeze else "unfrozen"
        logger.info(f"Backbone {status} (classification head remains trainable)")
    
    def get_num_parameters(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model.
        
        Returns:
            Dictionary with parameter counts:
            - total: Total parameters
            - trainable: Trainable parameters
            - frozen: Frozen parameters
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        return {
            'total': total,
            'trainable': trainable,
            'frozen': frozen
        }


def create_slowfast_model(
    model_name: str = 'slowfast_r50',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> ViolenceSlowFast:
    """
    Factory function to create SlowFast model.
    
    Args:
        model_name: SlowFast variant ('slowfast_r50', 'slowfast_r101')
        num_classes: Number of output classes (default: 2 for Fight/NonFight)
        pretrained: Whether to load pretrained weights from Kinetics-400
        dropout: Dropout probability for classification head
    
    Returns:
        ViolenceSlowFast model instance
    
    Example:
        >>> # Create SlowFast-R50 with pretrained weights
        >>> model = create_slowfast_model('slowfast_r50', num_classes=2, pretrained=True)
        >>> 
        >>> # Check model parameters
        >>> params = model.get_num_parameters()
        >>> print(f"Total parameters: {params['total']:,}")
        >>> 
        >>> # Freeze backbone for transfer learning
        >>> model.freeze_backbone(freeze=True)
    
    Notes:
        - SlowFast-R50 has ~34M parameters
        - Memory usage: ~5GB per batch (batch_size=4)
        - Recommended frames: 32 (compared to 16 for X3D)
        - Best for high-accuracy applications where speed is less critical
    """
    return ViolenceSlowFast(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )