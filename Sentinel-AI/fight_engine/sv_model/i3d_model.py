"""
I3D Model for Violence Detection

I3D (Inflated 3D ConvNet) is a 3D CNN architecture for video classification.
It inflates 2D convolution filters from pretrained image models (Inception-v1) 
into 3D, allowing transfer learning from images to videos.

This module provides I3D-R50 model pretrained on Kinetics-400.

Key Features:
- Inflated 3D convolutions from 2D Inception architecture
- Good balance of accuracy and efficiency
- Pretrained on Kinetics-400 for strong transfer learning
- Uses RGB input (can be extended to two-stream with optical flow)

References:
- Paper: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset" (CVPR 2017)
- https://arxiv.org/abs/1705.07750

Author: Safe School AI Team
Date: December 2025
"""

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ViolenceI3D(nn.Module):
    """
    I3D-based violence detection model.
    
    I3D (Inflated 3D ConvNet) inflates 2D convolution filters from ImageNet-pretrained
    Inception-v1 into 3D filters. This allows leveraging image classification knowledge
    for video understanding.
    
    Key aspects:
    1. 3D convolutions inflated from 2D Inception architecture
    2. Maintains spatial structure while modeling temporal information
    3. Can be used as single-stream (RGB) or two-stream (RGB + Flow)
    
    This class wraps the pretrained I3D model from PyTorchVideo and
    adds a custom classification head for binary violence detection.
    
    Args:
        model_name: Name of I3D variant ('i3d_r50' recommended)
        num_classes: Number of output classes (default: 2 for binary)
        pretrained: Whether to load pretrained weights from Kinetics
        dropout: Dropout probability for classification head
    
    Attributes:
        backbone: I3D backbone network
        num_classes: Number of output classes
        model_name: Name of the model variant
    """
    
    def __init__(
        self,
        model_name: str = 'i3d_r50',
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained I3D model from PyTorchVideo hub
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
        # I3D architecture has blocks structure similar to ResNet
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
                        in_features = 2048  # Default for I3D-R50
            else:
                in_features = 2048  # Default fallback
        else:
            in_features = 2048  # Default fallback
        
        # Replace the classification head with custom head for violence detection
        self.backbone.blocks[-1].proj = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
        
        logger.info(f"I3D model initialized: {model_name}, in_features={in_features}, "
                   f"num_classes={num_classes}, pretrained={pretrained}")
        logger.info(f"  Architecture: Inflated 3D convolutions from Inception-v1")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through I3D network.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
               Where:
               - B: batch size
               - C: channels (3 for RGB)
               - T: temporal frames (typically 32 or 64)
               - H: height (224)
               - W: width (224)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature representations (without classification head).
        
        Useful for:
        - Feature visualization
        - Transfer learning
        - Feature-based similarity search
        
        Args:
            x: Input tensor of shape (B, C, T, H, W)
        
        Returns:
            Feature tensor of shape (B, feature_dim)
            where feature_dim = 2048 for I3D-R50
        """
        # Forward through all blocks except the final projection
        for i, block in enumerate(self.backbone.blocks[:-1]):
            x = block(x)
        
        # Apply pooling from last block if available
        if hasattr(self.backbone.blocks[-1], 'pool'):
            x = self.backbone.blocks[-1].pool(x)
        
        # Flatten spatial and temporal dimensions
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
    
    def get_num_parameters(self) -> dict:
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


def create_i3d_model(
    model_name: str = 'i3d_r50',
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5
) -> ViolenceI3D:
    """
    Factory function to create I3D model.
    
    Args:
        model_name: I3D variant ('i3d_r50' is the standard variant)
        num_classes: Number of output classes (default: 2 for Fight/NonFight)
        pretrained: Whether to load pretrained weights from Kinetics-400
        dropout: Dropout probability for classification head
    
    Returns:
        ViolenceI3D model instance
    
    Example:
        >>> # Create I3D-R50 with pretrained weights
        >>> model = create_i3d_model('i3d_r50', num_classes=2, pretrained=True)
        >>> 
        >>> # Check model parameters
        >>> params = model.get_num_parameters()
        >>> print(f"Total parameters: {params['total']:,}")
        >>> 
        >>> # Freeze backbone for transfer learning
        >>> model.freeze_backbone(freeze=True)
    
    Notes:
        - I3D-R50 has ~28M parameters
        - Memory usage: ~4GB per batch (batch_size=4)
        - Recommended frames: 32 or 64
        - Good balance of accuracy and efficiency
        - Based on Inception-v1 architecture inflated to 3D
    """
    return ViolenceI3D(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )
