"""
This Python script defines a flexible Convolutional Neural Network for classification.
It is configured dynamically via a dictionary passed during initialization.
"""

import torch
from torch import nn
from torchvision.ops import DropBlock2d
from typing import Optional, Dict, Any

class Cnn_3(nn.Module):
    """
    Convolutional Neural Network with three initial convolutions concatenated 
    for classification.
    """

    def __init__(self, config: Dict[str, Any], linear_out_features: int) -> None:
        super().__init__()

        # --- Extract configuration ---
        use_batchnorm = config.get("use_batchnorm", False)
        use_dropout = config.get("use_dropout", False)
        use_dropblock = config.get("use_dropblock", False)
        p = config.get("p", 0.5)
        block_size = config.get("block_size", 5)
        bn_momentum = config.get("bn_momentum", 0.01)
        stride = config.get("stride", 1)
        linear_in_features = config["linear_in_features"]

        self.use_batchnorm = use_batchnorm
        self.use_dropout = use_dropout
        self.use_dropblock = use_dropblock

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
        channels = config["channels"] 
        kernel_sizes = config["kernel_sizes"]
        padding = config["padding"]

        self.convs = nn.ModuleList()

        for i in range(len(channels) - 1):
            pad = padding[i]
            if isinstance(pad, list): pad = tuple(pad)
            
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

        self.linear_at_output = nn.Linear(
            in_features=linear_in_features,
            out_features=linear_out_features
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Three initial convolutions concatenated
        x_a = self.conv_01_a(inputs)
        x_b = self.conv_01_b(inputs)
        x_c = self.conv_01_c(inputs)
        x = torch.cat((x_a, x_b, x_c), dim=1)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, (2, 2))

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

            if i < len(self.convs) - 1:
                x = nn.functional.max_pool2d(x, (2, 2))

        x = x.view(-1, flat_dim)
        logits = self.linear_at_output(x)
        return logits