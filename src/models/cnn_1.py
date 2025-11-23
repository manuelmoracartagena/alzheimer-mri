"""
This Python script defines a flexible Convolutional Neural Network for classification.
It is configured dynamically via a dictionary passed during initialization.
"""

import torch
from torch import nn
from torchvision.ops import DropBlock2d
from typing import Optional, Dict, Any

class Cnn_1(nn.Module):
    """
    Convolutional Neural Network for classification.
    Supports optional Batch Normalization, Dropout, and DropBlock regularization.
    """

    def __init__(self, config: Dict[str, Any], linear_out_features: int) -> None:
        super().__init__()

        # --- Extract configuration from the dictionary ---
        use_batchnorm = config.get("use_batchnorm", False)
        use_dropout = config.get("use_dropout", False)
        use_dropblock = config.get("use_dropblock", False)
        p = config.get("p", 0.5)
        block_size = config.get("block_size", 5)
        bn_momentum = config.get("bn_momentum", 0.01)
        
        channels = config["channels"]
        kernel_sizes = config["kernel_sizes"]
        padding = config["padding"]
        stride = config.get("stride", 1)
        linear_in_features = config["linear_in_features"]

        # --- Class Attributes ---
        self.convs: nn.ModuleList
        self.bns: Optional[nn.ModuleList]
        self.drop_layers: Optional[nn.ModuleList]
        self.linear_at_output: nn.Linear
        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_dropblock = use_dropblock

        if use_dropout and use_dropblock:
            raise ValueError("Cannot use both Dropout and DropBlock simultaneously.")

        self.convs = nn.ModuleList()

        # Create convolutional layers
        for i in range(len(channels) - 1):
            # Handle padding: YAML might load it as a list, convert to tuple if needed
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
        else:
            self.bns = None

        # Dropout / DropBlock
        if use_dropout:
            self.drop_layers = nn.ModuleList([nn.Dropout(p) for _ in range(len(self.convs))])
        elif use_dropblock:
            self.drop_layers = nn.ModuleList([
                DropBlock2d(p=p, block_size=block_size) if i < 6 else None
                for i in range(len(self.convs))
            ])
        else:
            self.drop_layers = None

        # Fully connected output
        self.linear_at_output = nn.Linear(
            in_features=linear_in_features,
            out_features=linear_out_features
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        
        # We need to flatten specifically to the size defined in config
        flat_dim = self.linear_at_output.in_features

        for i, conv in enumerate(self.convs):
            x = conv(x)

            if self.use_batchnorm and self.bns is not None:
                x = self.bns[i](x)

            x = nn.functional.relu(x)

            if self.drop_layers is not None:
                drop_layer = self.drop_layers[i]
                if drop_layer is not None:
                    x = drop_layer(x)

            # Max pooling except for the last layer
            if i < len(self.convs) - 1:
                x = nn.functional.max_pool2d(x, (2, 2))

        # Flatten
        x = x.view(-1, flat_dim)

        # Fully connected output
        logits = self.linear_at_output(x)
        return logits