"""
This Python script defines a ResNet model for classification.
It is configured dynamically via a dictionary passed during initialization.
"""

import torch
import torch.nn as nn
from typing import Optional, Type, List, Dict, Any

class BasicBlock(nn.Module):
    """
    Basic Residual Block for ResNet.
    """
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, 
                 downsample: Optional[nn.Sequential] = None,
                 use_dropout: bool = False, dropout_p: float = 0.3) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        
        if self.use_dropout:
            self.dropout = nn.Dropout(p=dropout_p)
        else:
            self.dropout = None

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

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
    ResNet implementation. Configured via 'config' dictionary.
    """

    def __init__(self, config: Dict[str, Any], linear_out_features: int) -> None:
        super().__init__()

        # --- Extract Config ---
        in_channels = config.get("in_channels", 1)
        block_type_str = config.get("block_type", "BasicBlock")
        layers = config.get("layers", [2, 2, 2, 2])
        planes_list = config.get("planes", [64, 128, 256, 512])
        initial_planes = config.get("initial_planes", 64)
        use_dropout = config.get("use_dropout", False)
        dropout_p = config.get("dropout_p", 0.3)

        # Look up the block class from the string name
        block: Type[BasicBlock]
        if block_type_str == "BasicBlock":
            block = BasicBlock
        else:
            raise ValueError(f"Unknown block_type: {block_type_str}")

        self.inplanes = initial_planes
        self.use_dropout = use_dropout
        self.dropout_p = dropout_p

        # Initial convolution layer
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Create ResNet layers
        self.layer1 = self._make_layer(block, planes_list[0], layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes_list[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes_list[2], layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes_list[3], layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final fully connected layer
        self.fc = nn.Linear(planes_list[-1] * block.expansion, linear_out_features)

    def _make_layer(self, block: Type[BasicBlock], planes: int, num_blocks: int, stride: int = 1) -> nn.Sequential:
        downsample: Optional[nn.Sequential] = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers: List[nn.Module] = []
        layers.append(block(
            self.inplanes, planes, stride, downsample,
            use_dropout=self.use_dropout, dropout_p=self.dropout_p
        ))

        self.inplanes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(
                self.inplanes, planes,
                use_dropout=self.use_dropout, dropout_p=self.dropout_p
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x