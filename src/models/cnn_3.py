# src/models/cnn_3.py
"""
Convolutional Neural Network (CNN_3) for image classification.

A flexible CNN architecture with three initial parallel convolutions that are
concatenated. Supports configurable layers, filter sizes, and kernel sizes.
Includes optional Batch Normalization, Dropout, and DropBlock regularization.
Configured dynamically via dictionary passed during initialization.
"""

import torch
from torch import nn
from torchvision.ops import DropBlock2d
from typing import Optional, Dict, Any, List, Tuple, Union

class Cnn_3(nn.Module):
    """
    Convolutional Neural Network (CNN_3) for classification.
    
    A flexible CNN architecture with three initial parallel convolutions that are 
    concatenated. Supports configurable layers, filter sizes, and kernel sizes. 
    Includes optional Batch Normalization, Dropout, and DropBlock regularization.
    """

    def __init__(self, config: Dict[str, Any], linear_out_features: int) -> None:
        """
        Initialize the CNN_3 model.
        
        Args:
            config: Configuration dictionary containing CNN parameters
                   (channels, kernel_sizes, padding, linear_in_features, etc.).
            linear_out_features: Number of output classes for classification.
        
        Raises:
            ValueError: If both Dropout and DropBlock are enabled simultaneously.
        """
        super().__init__()

        # --- Extract Configuration ---
        use_batchnorm: bool = config.get("use_batchnorm", False)
        use_dropout: bool = config.get("use_dropout", False)
        use_dropblock: bool = config.get("use_dropblock", False)
        p: float = config.get("p", 0.5)
        block_size: int = config.get("block_size", 5)
        bn_momentum: float = config.get("bn_momentum", 0.01)
        stride: int = config.get("stride", 1)
        
        channels: List[int] = config["channels"]
        kernel_sizes: List[int] = config["kernel_sizes"]
        padding: List[Union[int, Tuple[int, int]]] = config["padding"]
        linear_in_features: int = config["linear_in_features"]

        # --- Class Attributes ---
        self.use_batchnorm = use_batchnorm
        self.convs: nn.ModuleList
        self.bns: Optional[nn.ModuleList] = None
        self.drop_layers: Optional[nn.ModuleList] = None
        self.linear_at_output: nn.Linear

        if use_dropout and use_dropblock:
            raise ValueError("Cannot use both Dropout and DropBlock simultaneously.")

        # Three initial convolutions
        self.conv_01_a = nn.Conv2d(
            in_channels=1, out_channels=config["conv01_out"], 
            kernel_size=config["conv01_kernel_a"], 
            stride=stride, 
            padding=config["conv01_pad_a"], 
            dilation=config["conv01_dil_a"]
        )
        self.conv_01_b = nn.Conv2d(
            in_channels=1, out_channels=config["conv01_out"], 
            kernel_size=config["conv01_kernel_b"], 
            stride=stride, 
            padding=config["conv01_pad_b"], 
            dilation=config["conv01_dil_b"]
        )
        self.conv_01_c = nn.Conv2d(
            in_channels=1, out_channels=config["conv01_out"], 
            kernel_size=config["conv01_kernel_c"], 
            stride=stride, 
            padding=config["conv01_pad_c"], 
            dilation=config["conv01_dil_c"]
        )

        # Rest of the convolutional layers
        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            pad = padding[i]
            if isinstance(pad, list):
                pad = tuple(pad)
            
            conv = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel_sizes[i],
                stride=stride,
                padding=pad
            )
            self.convs.append(conv)

        # BatchNorm
        if use_batchnorm:
            self.bns = nn.ModuleList([
                nn.BatchNorm2d(channels[i + 1], momentum=bn_momentum)
                for i in range(len(channels) - 1)
            ])

        # Dropout / DropBlock
        if use_dropout:
            self.drop_layers = nn.ModuleList([
                nn.Dropout(p=p) for _ in range(len(self.convs))
            ])
        elif use_dropblock:
            self.drop_layers = nn.ModuleList([
                DropBlock2d(p=p, block_size=block_size) if i < 6 else None
                for i in range(len(self.convs))
            ])

        # Fully connected output
        self.linear_at_output = nn.Linear(
            in_features=linear_in_features,
            out_features=linear_out_features
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN_3 model.
        
        Args:
            inputs: Input tensor of shape [Batch, Channels, Height, Width].
        
        Returns:
            Classification logits of shape [Batch, NumClasses].
        """
        # 1. Three initial parallel convolutions concatenated
        x_a = self.conv_01_a(inputs)
        x_b = self.conv_01_b(inputs)
        x_c = self.conv_01_c(inputs)
        x = torch.cat((x_a, x_b, x_c), dim=1)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, (2, 2))

        flat_dim = self.linear_at_output.in_features

        # 2. Convolutional layers with optional regularization
        for i, conv in enumerate(self.convs):
            x = conv(x)

            if self.use_batchnorm and self.bns is not None:
                x = self.bns[i](x)

            x = nn.functional.relu(x)

            if self.drop_layers is not None:
                drop_layer = self.drop_layers[i]
                if drop_layer is not None:
                    x = drop_layer(x)

            if i < len(self.convs) - 1:
                x = nn.functional.max_pool2d(x, (2, 2))

        # 3. Flatten and classification head
        x = x.view(-1, flat_dim)
        logits = self.linear_at_output(x)
        
        return logits