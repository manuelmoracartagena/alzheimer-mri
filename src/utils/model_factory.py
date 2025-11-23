# src/utils/model_factory.py
"""
Model Factory Module.

This module acts as a centralized factory for instantiating neural network models.
It manages a registry of available architectures (CNNs, ResNet, ViT) and their
corresponding configurations, providing a unified interface to create model instances
with specific parameters loaded from YAML files.
"""

from typing import List, Any

from models.cnn_1 import Cnn_1
from models.cnn_2 import Cnn_2
from models.cnn_3 import Cnn_3
from models.resnet import ResNet
from models.vit import ViT

# Import the config loader utility
from utils.config_loader import load_model_config

# Registry mapping model names to their Class and Config Filename
MODEL_REGISTRY = {
    "Cnn_1": {
        "class": Cnn_1,
        "config_file": "cnn_1_config.yaml"
    },
    "Cnn_2": {
        "class": Cnn_2,
        "config_file": "cnn_2_config.yaml"
    },
    "Cnn_3": {
        "class": Cnn_3,
        "config_file": "cnn_3_config.yaml"
    },
    "ResNet": {
        "class": ResNet,
        "config_file": "resnet_config.yaml"
    },
    "ViT": {
        "class": ViT,
        "config_file": "vit_config.yaml"
    }
}

def create_model(model_name: str, num_classes: int) -> Any:
    """
    Create a model instance based on model name.
    It loads the specific YAML config for the model automatically.

    Args:
        model_name (str): Name of the model to create.
        num_classes (int): Number of output classes.

    Returns:
        Any: An instance of the requested model class.
    """
    if model_name not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model '{model_name}' not found in registry. "
            f"Available models: {available_models}"
        )
    
    model_info = MODEL_REGISTRY[model_name]
    model_class = model_info["class"]
    config_filename = model_info["config_file"]
    
    # Load the specific configuration for this model from ROOT/configs/
    model_config = load_model_config(config_filename)
    
    # Instantiate the model passing the dictionary and num_classes
    # The models must be updated to accept (config, linear_out_features) in __init__
    model = model_class(
        config=model_config,
        linear_out_features=num_classes
    )
    
    return model


def get_available_models() -> List[str]:
    """
    Get list of available model names in the registry.
    
    Returns:
        List[str]: List of available model names.
    """
    return list(MODEL_REGISTRY.keys())