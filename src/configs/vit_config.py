"""
Configuration file for the Vision Transformer (ViT) Model.

This file defines the default hyperparameters and architectural settings
used to initialize the ViT model.
"""

MODEL_CONFIG = {
    # -------------------------------------------------------------------------
    # Input/Output Shape
    # -------------------------------------------------------------------------
    "image_size": 256,        # Image size (H, W). 256x256
    "patch_size": 16,         # Patch size. 16x16
    "in_channels": 1,         # Input channels (1 for grayscale)
    "linear_out_features": 4, # Number of classes (default value)

    # -------------------------------------------------------------------------
    # Transformer Architecture
    # -------------------------------------------------------------------------
    "dim": 768,               # Embedding dimension (ViT-Base)
    "depth": 6,               # Number of Transformer blocks (6 for "Small")
    "heads": 8,               # Number of attention heads
    "mlp_dim": 3072,          # FeedForward layer dimension (4 * dim)
    "dim_head": 64,           # Dimension of each head (optional, 768/8=96)

    # -------------------------------------------------------------------------
    # Pooling & Regularization
    # -------------------------------------------------------------------------
    "pool": "cls",            # 'cls' (CLS token) or 'mean' (average)
    "dropout": 0.1,           # Dropout in the Transformer
    "emb_dropout": 0.1        # Dropout in the embeddings
}