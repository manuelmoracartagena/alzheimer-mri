"""
Configuration file for ResNet Model.

This file defines the default hyperparameters and architectural settings
used to initialize the ResNet model.
"""

MODEL_CONFIG = {
    # -------------------------------------------------------------------------
    # Regularization options
    # -------------------------------------------------------------------------
    "use_dropout": True,            # Enable Dropout within each BasicBlock (as in original resnet.py)

    # -------------------------------------------------------------------------
    # Regularization parameters
    # -------------------------------------------------------------------------
    "dropout_p": 0.3,               # Dropout probability for BasicBlock

    # -------------------------------------------------------------------------
    # ResNet architecture
    # -------------------------------------------------------------------------
    "block_type": "BasicBlock",     # Block type to use (e.g., "BasicBlock" or "Bottleneck")
    "layers": [2, 2, 2, 2],         # Number of blocks in each of the four main layers
    "planes": [64, 128, 256, 512],  # Number of channels (planes) for each of the four main layers
    "in_channels": 1,               # Input channels for the model (1 = grayscale)
    "initial_planes": 64,           # Number of channels after the very first convolutional layer

    # -------------------------------------------------------------------------
    # Fully connected layer
    # -------------------------------------------------------------------------
    # 'linear_in_features' is calculated automatically by the model
    "linear_out_features": 4        # Number of output classes
}