"""
This Python script defines a ResNet model for classification.
It supports optional Dropout regularization.
It is configured via the 'configs/resnet_config.py' file.
"""

import torch
import torch.nn as nn
from typing import Optional, Type, List

# Import the model configuration
from configs.resnet_config import MODEL_CONFIG 


class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet.
    Uses Conv -> BN -> ReLU -> Dropout -> Conv -> BN.
    The identity is added, followed by ReLU and another Dropout.
    """
    expansion = 1

    def __init__(
        self, 
        inplanes: int, 
        planes: int, 
        stride: int = 1, 
        downsample: Optional[nn.Sequential] = None,
        use_dropout: bool = False, 
        dropout_p: float = 0.3
    ) -> None:
        super().__init__()
        
        # --- Type hints for class attributes ---
        self.conv1: nn.Conv2d
        self.bn1: nn.BatchNorm2d
        self.relu: nn.ReLU
        self.use_dropout: bool = use_dropout
        self.dropout: Optional[nn.Dropout]
        self.conv2: nn.Conv2d
        self.bn2: nn.BatchNorm2d
        self.downsample: Optional[nn.Sequential] = downsample
        self.stride: int = stride
        # --- End type hints ---

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_dropout and self.dropout is not None:
            out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        
        if self.use_dropout and self.dropout is not None:
            out = self.dropout(out)

        return out


class ResNet(nn.Module):
    """
    ResNet implementation, configurable via MODEL_CONFIG.
    """

    def __init__(
        self,
        # Pull defaults from the imported configuration file
        linear_out_features: int = MODEL_CONFIG["linear_out_features"],
        in_channels: int = MODEL_CONFIG["in_channels"],
        block_type: str = MODEL_CONFIG["block_type"],
        layers: List[int] = MODEL_CONFIG["layers"],
        planes_list: List[int] = MODEL_CONFIG["planes"],
        initial_planes: int = MODEL_CONFIG["initial_planes"],
        use_dropout: bool = MODEL_CONFIG["use_dropout"],
        dropout_p: float = MODEL_CONFIG["dropout_p"]
    ) -> None:
        super().__init__()

        # --- Type hints for class attributes ---
        self.inplanes: int
        self.use_dropout: bool
        self.dropout_p: float
        self.conv1: nn.Conv2d
        self.bn1: nn.BatchNorm2d
        self.relu: nn.ReLU
        self.maxpool: nn.MaxPool2d
        self.layer1: nn.Sequential
        self.layer2: nn.Sequential
        self.layer3: nn.Sequential
        self.layer4: nn.Sequential
        self.avgpool: nn.AdaptiveAvgPool2d
        self.fc: nn.Linear
        # --- End type hints ---

        # Look up the block class from the string name
        block: Type[BasicBlock] # <- Se usa el tipo directamente aquí
        if block_type == "BasicBlock":
            block = BasicBlock
        else:
            raise ValueError(f"Unknown block_type: {block_type}")

        self.inplanes = initial_planes
        
        # Store regularization params to pass to _make_layer
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create ResNet layers dynamically from config
        self.layer1 = self._make_layer(block, planes_list[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes_list[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes_list[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes_list[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer
        # Input features are the last plane size * block expansion
        self.fc = nn.Linear(planes_list[-1] * block.expansion, linear_out_features)


    def _make_layer(
        self, 
        block: Type[BasicBlock],
        planes: int, 
        num_blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """Helper function to create a sequential ResNet layer."""
        downsample: Optional[nn.Sequential] = None

        # Check if downsampling is needed (stride != 1 or different dimensions)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        # Add the first block (which handles downsampling)
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            use_dropout=self.use_dropout, dropout_p=self.dropout_p
        ))

        self.inplanes = planes * block.expansion # Update inplanes for the next block

        # Add the rest of the blocks for this layer
        for _ in range(1, num_blocks):
            layers.append(block(
                self.inplanes, planes,
                use_dropout=self.use_dropout, dropout_p=self.dropout_p
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pooling and FC
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x