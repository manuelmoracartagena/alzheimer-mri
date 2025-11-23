"""
Main Execution Pipeline.

This script serves as the entry point for the training and evaluation pipeline.
It iterates over the specified models and data folds (or splits), manages 
Weights & Biases (WandB) logging, and orchestrates the training and testing phases.
"""

import copy
import torch
import wandb
import sys
import os
from collections import defaultdict
from pathlib import Path

# Adjust sys.path to ensure we can import modules if running from root
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# Import local modules
from utils.config_loader import load_main_config
from data.dataloader import get_dataloaders 
from utils.metrics import print_final_metrics
from utils.model_factory import create_model 
from train import run_training_fold
from eval import run_evaluation_fold

# --- Main Execution ---
def main() -> None:
    """
    Main function to execute the experiment pipeline.
    """
    
    # 1. Load Configuration (YAML) & Resolve Paths
    try:
        config = load_main_config()
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load configuration. {e}")
        return

    # 2. Setup Device
    device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    config["device"] = torch.device(device_name)
    
    # 3. Setup Environment Variables (e.g. for WandB)
    # We use the resolved 'logs_dir' path
    os.environ["WANDB_DIR"] = str(config["logs_dir"])

    strategy = config.get("data_strategy", "simple_split") 

    print(f"Selected device: {config['device']}")
    print(f"Project Root: {config['project_root']}")
    print(f"Weights Save Dir: {config['save_dir']}")

    # Dictionary to store aggregated metrics for all models
    all_model_metrics = defaultdict(lambda: defaultdict(list))

    # --- CONFIGURATION SETUP ---
    num_classes = config["num_classes"]
    folds_to_run = config["folds_to_run"]
    
    print(f"\nActive dataset configuration: {config['name']}")
    print(f"Detected number of classes: {num_classes}")
    # --- END CONFIGURATION SETUP ---

    print("\n--- Starting experiments ---")

    for model_name in config["models"]:
        print(f"\n--- Processing model: {model_name} ---")

        # --- DATA LOADING ---
        # The dataloader receives the full config, which now contains 
        # the absolute path to 'data_base_dir'
        try:
            all_fold_dataloaders = get_dataloaders(config, model_name=model_name)
        except FileNotFoundError as e:
            print(f"\nERROR: Could not load data for {model_name}. Error: {e}")
            continue
        except Exception as e:
            print(f"\nERROR: Unexpected error loading data: {e}")
            continue

        if len(folds_to_run) != len(all_fold_dataloaders):
            print("ERROR: Mismatch between 'folds_to_run' and loaded dataloaders.")
            continue
        
        # --- MODEL INITIALIZATION ---
        try:
            # Factory loads the specific model yaml from ROOT/configs
            model_instance = create_model(model_name, num_classes=num_classes)
        except Exception as e:
             print(f"WARNING: 'create_model' failed. Error: {e}")
             continue
        # --- END MODEL INITIALIZATION ---

        # --- FOLD LOOP ---
        for fold_index, (train_loader, val_loader, test_loader) in zip(folds_to_run, all_fold_dataloaders):
            
            fold_display_num = fold_index + 1
            run_label = f"Fold {fold_display_num}" if strategy == "cross_validation" else "Run"
            
            # Prepare config for WandB (convert Paths to strings, remove objects)
            config_for_wandb = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
            if "device" in config_for_wandb: del config_for_wandb["device"] 

            wandb_run_name = f"{model_name}_fold{fold_display_num}" if strategy == "cross_validation" else f"{model_name}_simple_split"
            
            print(f"\n- Executing {run_label}")

            # Initialize W&B
            run = wandb.init(
                project=config["wandb_project"],
                config=config_for_wandb,
                name=wandb_run_name, 
                reinit=True,
                dir=str(config["logs_dir"]) 
            )

            # Deep copy model to reset weights for this fold
            model_for_fold = copy.deepcopy(model_instance)

            print(f"{run_label} - Starting training...")
            best_model_path = run_training_fold(
                model_instance=model_for_fold,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                fold=fold_index,
                model_name=model_name
            )

            print(f"{run_label} - Starting testing...")
            fold_metrics = run_evaluation_fold(
                model_instance=model_for_fold, 
                test_loader=test_loader,
                model_path=best_model_path,
                config=config,
                fold=fold_index,
                model_name=model_name
            )

            # Aggregate metrics
            for key, value in fold_metrics.items():
                all_model_metrics[model_name][key].append(value)

            wandb.finish() # End the W&B run for this fold

    print("\n--- Experiments finished ---")
    print_final_metrics(all_model_metrics)


if __name__ == "__main__":
    main()