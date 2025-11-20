"""
This Python script defines a flexible Convolutional Neural Network for classification.
It supports optional Batch Normalization, Dropout, and DropBlock regularization.
It is configured via the 'configs/cnn_1_config.py' file.
"""

import torch
from torch import nn
from torchvision.ops import DropBlock2d
from typing import Optional

# Import the model configuration
from configs.cnn_1_config import MODEL_CONFIG

class Cnn_1(nn.Module):
    """
    Convolutional Neural Network for classification.
    Supports optional Batch Normalization, Dropout, and DropBlock regularization.
    """

    def __init__(
        self,
        use_batchnorm: bool = MODEL_CONFIG["use_batchnorm"],
        use_dropout: bool = MODEL_CONFIG["use_dropout"],
        use_dropblock: bool = MODEL_CONFIG["use_dropblock"],
        p: float = MODEL_CONFIG["p"],
        block_size: int = MODEL_CONFIG["block_size"],
        bn_momentum: float = MODEL_CONFIG["bn_momentum"],
        linear_out_features: int = MODEL_CONFIG["linear_out_features"]
    ) -> None:
        super().__init__()

        # --- Type hints for class attributes ---
        self.convs: nn.ModuleList
        self.bns: Optional[nn.ModuleList]
        self.drop_layers: Optional[nn.ModuleList]
        self.linear_at_output: nn.Linear
        self.use_batchnorm: bool = use_batchnorm
        self.use_dropout: bool = use_dropout
        self.use_dropblock: bool = use_dropblock
        # --- End type hints ---

        if use_dropout and use_dropblock:
            raise ValueError("Cannot use both Dropout and DropBlock simultaneously.")

        channels = MODEL_CONFIG["channels"]
        self.convs = nn.ModuleList()

        # Create convolutional layers
        for i in range(len(channels) - 1):
            kernel = MODEL_CONFIG["kernel_sizes"][i]
            pad = MODEL_CONFIG["padding"][i]
            conv = nn.Conv2d(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                kernel_size=kernel,
                stride=MODEL_CONFIG["stride"],
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
            in_features=MODEL_CONFIG["linear_in_features"],
            out_features=linear_out_features
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs

        # Loop through the the convolutional layers
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

        # Flatten before the linear layer
        x = x.view(-1, MODEL_CONFIG["linear_in_features"])

        # Fully connected output
        logits = self.linear_at_output(x)
        return logits