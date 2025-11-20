"""
Main Execution Pipeline.

This script serves as the entry point for the training and evaluation pipeline.
It iterates over the specified models and data folds (or splits), manages 
Weights & Biases (WandB) logging, and orchestrates the training and testing phases.
"""

import copy
import wandb
from collections import defaultdict

from configs.config import TRAIN_CONFIG as config
from utils.dataloader import get_dataloaders 
from utils.metrics import print_final_metrics
from utils.model_factory import create_model 
from utils.train import run_training_fold
from utils.eval import run_evaluation_fold

# --- Main Execution ---
def main() -> None:
    """
    Main function to execute the experiment pipeline.
    
    Steps:
    1. Reads configuration settings.
    2. Iterates through the list of models defined in the config.
    3. Generates DataLoaders specific to the model (handling resizing if necessary).
    4. Initializes the model architecture.
    5. Iterates through the requested folds (Cross-Validation or Simple Split).
    6. Runs training and evaluation for each fold.
    7. Aggregates and prints final metrics.
    """

    strategy = config.get("data_strategy", "simple_split") 

    print(f"Selected device: {config['device']}")

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
        try:
            all_fold_dataloaders = get_dataloaders(config, model_name=model_name)
        except FileNotFoundError as e:
            print(f"\nERROR: Could not load data for {model_name}. Error: {e}")
            continue

        if len(folds_to_run) != len(all_fold_dataloaders):
            print("ERROR: Mismatch between 'folds_to_run' and loaded dataloaders.")
            continue
        
        # --- MODEL INITIALIZATION ---
        try:
            model_instance = create_model(model_name, num_classes=num_classes)
        except Exception as e:
             print(f"WARNING: 'create_model' failed. Using a placeholder. Error: {e}")
             continue
        # --- END MODEL INITIALIZATION ---

        # --- FOLD LOOP ---
        for fold_index, (train_loader, val_loader, test_loader) in zip(folds_to_run, all_fold_dataloaders):
            
            fold_display_num = fold_index + 1
            run_label = f"Fold {fold_display_num}" if strategy == "cross_validation" else "Run"
            wandb_run_name = f"{model_name}_fold{fold_display_num}" if strategy == "cross_validation" else f"{model_name}_simple_split"
            
            print(f"\n- Executing {run_label}")

            # Initialize W&B for the current fold
            run = wandb.init(
                project=config["wandb_project"],
                config=config,
                name=wandb_run_name, 
                reinit=True 
            )

            # Create a deep copy of the model to ensure fresh weights for each fold
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

            # Aggregate metrics for final reporting
            for key, value in fold_metrics.items():
                all_model_metrics[model_name][key].append(value)

            wandb.finish() # End the W&B run for this fold

    print("\n--- Experiments finished ---")
    print_final_metrics(all_model_metrics)


if __name__ == "__main__":
    main()