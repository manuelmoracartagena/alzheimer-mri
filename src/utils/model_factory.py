"""
Model Factory Module.

This module acts as a centralized factory for instantiating neural network models.
It manages a registry of available architectures (CNNs, ResNet, ViT) and their
corresponding configurations, providing a unified interface to create model instances
with specific parameters.
"""

from typing import List, Any

from models.cnn_1 import Cnn_1
from models.cnn_2 import Cnn_2
from models.cnn_3 import Cnn_3
from models.resnet import ResNet
from models.vit import ViT

from configs.cnn_1_config import MODEL_CONFIG as MODEL_1_CONFIG
from configs.cnn_2_config import MODEL_CONFIG as MODEL_2_CONFIG
from configs.cnn_3_config import MODEL_CONFIG as MODEL_3_CONFIG
from configs.resnet_config import MODEL_CONFIG as RESNET_CONFIG
from configs.vit_config import MODEL_CONFIG as VIT_CONFIG


MODEL_REGISTRY = {
    "Cnn_1": {
        "class": Cnn_1,
        "config": MODEL_1_CONFIG
    },
    "Cnn_2": {
        "class": Cnn_2,
        "config": MODEL_2_CONFIG
    },
    "Cnn_3": {
        "class": Cnn_3,
        "config": MODEL_3_CONFIG
    },
    "ResNet": {
        "class": ResNet,
        "config": RESNET_CONFIG
    },
    "ViT": {
        "class": ViT,
        "config": VIT_CONFIG
    }
}


def create_model(model_name: str, num_classes: int) -> Any:
    """
    Create a model instance based on model name.

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
    config = model_info["config"]
    
    model = None

    if model_name == "ResNet":
        model = model_class(
            linear_out_features=num_classes,
            use_dropout=config.get("use_dropout", False),
            dropout_p=config.get("dropout_p", 0.3)
        )

    elif model_name == "ViT":
        model = model_class(
            linear_out_features=num_classes,
            image_size=config.get("image_size", 256),
            patch_size=config.get("patch_size", 16),
            dim=config.get("dim", 768),
            depth=config.get("depth", 6),
            heads=config.get("heads", 8),
            mlp_dim=config.get("mlp_dim", 3072),
            dim_head=config.get("dim_head", 64),
            dropout=config.get("dropout", 0.1),
            emb_dropout=config.get("emb_dropout", 0.1),
            pool=config.get("pool", "cls")
        )

    elif model_name in ["Cnn_1", "Cnn_2", "Cnn_3"]:
        model = model_class(
            use_batchnorm=config.get("use_batchnorm", False),
            use_dropout=config.get("use_dropout", False),
            use_dropblock=config.get("use_dropblock", False),
            bn_momentum=config.get("bn_momentum", 0.01),
            p=config.get("p", 0.5),
            block_size=config.get("block_size", 5),
            linear_out_features=num_classes
        )
    
    else:
        raise NotImplementedError(
            f"Constructor for '{model_name}' is not implemented in the factory."
        )
    
    return model


def get_available_models() -> List[str]:
    """
    Get list of available model names in the registry.
    
    Returns:
        List[str]: List of available model names.
    """
    return list(MODEL_REGISTRY.keys())