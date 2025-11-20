"""
Configuration file for Model_1 — a flexible Convolutional Neural Network (CNN)
for image classification with 4 output classes.

This file defines the default hyperparameters and architectural settings
used to initialize the model. It supports optional Batch Normalization,
Dropout, and DropBlock regularization mechanisms.
"""

MODEL_CONFIG = {
    # -------------------------------------------------------------------------
    # Regularization options
    # -------------------------------------------------------------------------
    "use_batchnorm": False,   # Enable Batch Normalization after each Conv layer
    "use_dropout": False,     # Enable Dropout regularization
    "use_dropblock": False,   # Enable DropBlock regularization
    # Note: Dropout and DropBlock cannot be enabled simultaneously.

    # -------------------------------------------------------------------------
    # Regularization parameters
    # -------------------------------------------------------------------------
    "bn_momentum": 0.01,      # Momentum for Batch Normalization
    "p": 0.5,                 # Dropout or DropBlock probability
    "block_size": 5,          # Block size for DropBlock2d

    # -------------------------------------------------------------------------
    # Convolutional architecture
    # -------------------------------------------------------------------------
    "channels": [1, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192], # Channel list (1=input)
    "kernel_sizes": [5, 3, 3, 3, 3, 3, 3, 3, 3], # Kernel sizes (1st=5x5, rest=3x3)
    "padding": [(2, 2)] + [(1, 1)] * 8,       # Padding (1st=(2,2), rest=(1,1))

    # Common convolution settings
    "stride": 1,              # Stride value for all convolutional layers
    "activation": "ReLU",     # Activation function used after each convolution
    "pooling": "MaxPool2d(2x2)",  # Pooling operation applied after each conv

    # -------------------------------------------------------------------------
    # Fully connected layer
    # -------------------------------------------------------------------------
    "linear_in_features": 8192,   # Flattened feature dimension before the FC layer
    "linear_out_features": 4      # Number of output classes
}