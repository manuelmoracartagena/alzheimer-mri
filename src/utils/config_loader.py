# src/utils/config_loader.py
"""
Configuration Loader Module.
This module handles loading and merging of configuration files (YAML) for the project.
It provides functions to load the main configuration as well as specific model configurations,
resolving paths and merging dataset-specific settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Helper to load a YAML file safely."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def get_project_root() -> Path:
    """
    Returns the absolute path to the project root.
    Assumes this script is located at: ROOT/src/utils/config_loader.py
    """
    return Path(__file__).resolve().parent.parent.parent

def load_model_config(config_filename: str) -> Dict[str, Any]:
    """
    Loads a specific model configuration file from ROOT/configs/.
    
    Args:
        config_filename (str): The name of the yaml file (e.g., 'cnn_1_config.yaml')
        
    Returns:
        Dict: The 'model_config' section of the YAML.
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Model config not found at: {config_path}")
        
    data = load_yaml(config_path)
    return data["model_config"]

def load_main_config(config_filename: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads the main configuration, resolves absolute paths, and merges dataset settings.
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / config_filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Main config file not found at {config_path}")
        
    config = load_yaml(config_path)
    
    # --- Resolve Paths ---
    # The YAML contains relative string paths. We convert them to absolute Path objects.
    paths = config.get("paths", {})
    resolved_paths = {}
    
    for key, val in paths.items():
        # Join project_root with the relative path from yaml
        full_path = project_root / val
        resolved_paths[key] = full_path
        
        # Create directories for outputs (weights, logs, tmp)
        if key in ["save_dir", "tmp_dir", "logs_dir"]:
            full_path.mkdir(parents=True, exist_ok=True)

    # --- Merge Configurations ---
    # 1. Start with common hyperparameters
    final_config = config["common_hyperparameters"].copy()
    
    # 2. Add Active Dataset settings
    active_ds_key = config["active_dataset"]
    if active_ds_key not in config["datasets"]:
        raise ValueError(f"Active dataset '{active_ds_key}' not defined in 'datasets' section.")
        
    dataset_config = config["datasets"][active_ds_key]
    final_config.update(dataset_config)
    
    # 3. Add Resolved Paths
    final_config.update(resolved_paths)
    
    # 4. Resolve the specific 'data_base_dir' for the active dataset
    # The dataset config has a key 'data_base_dir_key' which points to a path name
    if "data_base_dir_key" in final_config:
        key_ref = final_config["data_base_dir_key"]
        if key_ref in resolved_paths:
            final_config["data_base_dir"] = resolved_paths[key_ref]
        else:
            raise ValueError(f"Path key '{key_ref}' not found in 'paths' section.")
            
    # 5. Add Project Root (useful for reference)
    final_config["project_root"] = project_root
    
    return final_config