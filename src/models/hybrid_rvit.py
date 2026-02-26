# src/models/hybrid_rvit.py
"""
This Python script defines a hybrid Vision Transformer model that combines ResNet backbone 
with Vision Transformer encoder.

The architecture uses a CNN backbone to extract features from input images and passes them 
to a Transformer encoder for global reasoning. Configured dynamically via dictionaries 
passed during initialization.
"""

import torch
import torch.nn as nn
from einops import rearrange
from typing import Dict, Any, List

from .resnet import ResNet
from .vit import ViT


class HybridRViT(nn.Module):
    """
    Hybrid Architecture: ResNet backbone + Vision Transformer encoder.
    
    Combines a ResNet backbone for local feature extraction with a Vision Transformer 
    for capturing global dependencies. Features from a specified ResNet layer are 
    tokenized and processed through the ViT encoder.
    """

    def __init__(self, config: Dict[str, Any], linear_out_features: int) -> None:
        """
        Initialize the Hybrid RViT model.
        
        Args:
            config: Configuration dictionary containing 'resnet_config', 'vit_config',
                   and 'resnet_layer_out'.
            linear_out_features: Number of output classes.
        
        Raises:
            ValueError: If resnet_layer_out is not between 1 and 4.
            KeyError: If required config sections are missing.
        """
        super().__init__()
        
        # --- Extract Config Sections ---
        resnet_config: Dict[str, Any] = config.get("resnet_config", {})
        vit_config: Dict[str, Any] = config.get("vit_config", {})
        resnet_layer_out: int = config.get("resnet_layer_out", 4)
        
        if not resnet_config:
            raise KeyError("'resnet_config' section not found in configuration")
        if not vit_config:
            raise KeyError("'vit_config' section not found in configuration")
        
        # --- Validation ---
        if resnet_layer_out not in [1, 2, 3, 4]:
            raise ValueError(f"resnet_layer_out must be 1-4, got {resnet_layer_out}")
        
        self.resnet_layer_out = resnet_layer_out

        # --- Initialize ResNet Backbone ---
        resnet = ResNet(resnet_config, linear_out_features=linear_out_features)
        self.resnet_backbone = self._build_resnet_backbone(resnet, resnet_layer_out)
        
        # --- Extract Output Channels ---
        planes_list: List[int] = resnet_config.get("planes", [64, 128, 256, 512])
        resnet_out_channels = planes_list[resnet_layer_out - 1]

        # --- Initialize Vision Transformer ---
        vit_config_updated = vit_config.copy()
        vit_config_updated["in_channels"] = resnet_out_channels
        self.vit = ViT(vit_config_updated, linear_out_features=linear_out_features)

    def _build_resnet_backbone(self, resnet: ResNet, layer_out: int) -> nn.Sequential:
        """
        Build ResNet backbone up to a specified layer.
        
        Args:
            resnet: The ResNet model to extract backbone from.
            layer_out: Layer to output from (1-4).
        
        Returns:
            Sequential module containing ResNet layers.
        """
        layers = [
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        ]
        
        if layer_out >= 2:
            layers.append(resnet.layer2)
        if layer_out >= 3:
            layers.append(resnet.layer3)
        if layer_out >= 4:
            layers.append(resnet.layer4)
        
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Hybrid RViT model.
        
        Args:
            x: Input tensor of shape [Batch, Channels, Height, Width].
        
        Returns:
            Classification logits of shape [Batch, NumClasses].
        """
        # 1. Extract features from ResNet backbone
        features = self.resnet_backbone(x)
        
        # 2. Convert feature maps to image format for ViT
        # [B, C, H, W] -> [B, H, W, C] -> [B, H*W, C] -> [B, C, H, W]
        b, c, h, w = features.shape
        tokens = rearrange(features, 'b c h w -> b (h w) c')
        x = rearrange(tokens, 'b (h w) c -> b c h w', h=h, w=w)
        
        # 3. Pass through Vision Transformer (handles patch embedding, 
        #    attention, pooling, and classification)
        return self.vit(x)
