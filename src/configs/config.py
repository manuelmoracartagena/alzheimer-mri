"""
Configuration file for model training settings and shared instances.

This file defines all necessary paths, selects the active dataset,
and builds the final training configuration dictionary (TRAIN_CONFIG)
by merging dataset-specific settings with common hyperparameters.
"""

import os
import torch
from pathlib import Path

# -------------------------------------------------------------------------
# 1. PATH CALCULATION & ENVIRONMENT SETUP
# -------------------------------------------------------------------------

# Define the absolute path to the project's root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# --- Data Paths ---
DATA_DS_KAGGLE_DIR = PROJECT_ROOT / "data" / "kaggle" / "resized"    # Path to Dataset 1 (4 classes, images)
DATA_DS_ADNI_DIR = PROJECT_ROOT / "data" / "adni"                     # Path to Dataset 2 (3 classes, .npy files loaded via .txt manifests)

# --- Output Paths ---
SAVE_DIR = PROJECT_ROOT / "data" / "weights"    # Directory to save trained model weights
TMP_DIR = PROJECT_ROOT / "data" / "tmp"         # Directory for temporary files (e.g., cached data)
LOGS_DIR = PROJECT_ROOT / "logs"                # Directory for logging outputs

# Create output directories if they don't exist
SAVE_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Set the directory for Weights & Biases (W&B) logs
os.environ["WANDB_DIR"] = str(LOGS_DIR)


# -------------------------------------------------------------------------
# 2. DETAILED DATASET CONFIGURATIONS
# -------------------------------------------------------------------------

# --- Define Dataset Keys ---
# Use these constants to set ACTIVE_DATASET in section 3.
KAGGLE_DATASET = "kaggle"  # Kaggle: 4 classes, .png/.jpg
ADNI_DATASET = "adni"      # ADNI: 3 classes, .npy from .txt split files


DATASET_CONFIGS = {
    
    # --- DATASET KAGGLE (Images, 4 Classes, loaded "on the fly") ---
    KAGGLE_DATASET: {
        "name": "kaggle_dataset",
        "data_base_dir": str(DATA_DS_KAGGLE_DIR),
        "num_classes": 4,
        "class_names": ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"],
        "input_type": "image",  # Loader will handle .png, .jpg, etc.
        
        # "on_the_fly": Scans 'data_base_dir' and splits data dynamically.
        "split_type": "on_the_fly", 
        "data_strategy": "cross_validation",  # "cross_validation" or "simple_split"
        
        # "on_the_fly" split settings
        "split_random_state": 42,   # For reproducible splits
        "test_split_size": 0.2,     # 20% of data for the final test set
        "val_split_size": 0.25,     # 25% of *remaining* (train+val) data for validation
        "n_splits_cv": 5,           # Number of folds for cross-validation
        
        # --- FOLD SELECTION ---
        # If data_strategy is "simple_split", this MUST be [0].
        # If "cross_validation", set to the folds you want to run (e.g., [0, 1, 2, 3, 4])
        "folds_to_run": [0], 

        # Dataset-specific options
        "augment_train": False,        # Apply data augmentation to the training set?
        "use_weighted_sampler": False  # Balance classes in the training loader?
    },
    
    # --- DATASET ADNI (NPY, 3 Classes, loaded from pre-defined TXT files) ---
    ADNI_DATASET: {
        "name": "adni_dataset",
        "data_base_dir": str(DATA_DS_ADNI_DIR),  # Loader will look for .txt files here
        "num_classes": 3,       
        "class_names": ["CN", "MCI", "AD"],      # Must match the label map
        "input_type": "numpy",                   # Loader will load .npy files specified in .txt
        
        # "pre_split_txt": Loads train/val/test paths from fixed .txt files.
        "split_type": "pre_split_txt",
        
        # These options are fixed for this dataset type
        "data_strategy": "simple_split", 
        "folds_to_run": [0],  # Only one "fold" exists (the pre-defined split)

        # Dataset-specific options
        "augment_train": False,     # Apply data augmentation to the training set?
        "use_weighted_sampler": True  # Balance classes in the training loader?
    }
}


# -------------------------------------------------------------------------
# 3. ACTIVE DATASET SELECTION
# -------------------------------------------------------------------------
ACTIVE_DATASET = ADNI_DATASET # Options: KAGGLE_DATASET or ADNI_DATASET


# -------------------------------------------------------------------------
# 4. FINAL TRAINING CONFIGURATION DICTIONARY
# -------------------------------------------------------------------------

# Start with the configuration of the selected dataset
TRAIN_CONFIG = DATASET_CONFIGS[ACTIVE_DATASET].copy()

# Add common hyperparameters and runtime settings
TRAIN_CONFIG.update({
    # --- Hyperparameters ---
    "batch_size": 16,
    "learning_rate": 0.000011,
    "epochs": 1,
    "optimizer": "Adam",
    "weight_decay": 0.0001,

    # --- Model Selection ---
    "models": ["Cnn_1"],      # Models to train (Options: "Cnn_1", "Cnn_2", "Cnn_3", "ResNet", "ViT".)
    
    # --- Environment & Runtime ---
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "save_weights": False,           # Save the best model weights?
    "save_dir": str(SAVE_DIR),
    "tmp_dir": TMP_DIR,
    
    # --- Logging ---
    "log_dir": str(LOGS_DIR),
    "wandb_project": "tfm"           # Weights & Biases project name
})