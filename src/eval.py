# src/eval.py
"""
Evaluation Module.

This module handles the testing and evaluation phase of the model pipeline.
It includes functions to load trained model weights, execute testing loops
on the test dataset, calculate performance metrics, and generate confusion matrices.
"""

import torch
import torch.nn as nn
import wandb
import os
from tqdm import tqdm
from typing import Dict, Any, Tuple

from utils.metrics import calculate_metrics, plot_confusion_matrix

def test_model(model: nn.Module, 
               loader: torch.utils.data.DataLoader, 
               device: torch.device, 
               model_name: str, 
               strategy: str, 
               fold_display_num: int,
               config: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
    """
    Executes the testing loop and calculates final metrics.

    Args:
        model (nn.Module): The model to test.
        loader (DataLoader): DataLoader containing test data.
        device (torch.device): Computation device.
        model_name (str): Name of the model.
        strategy (str): Data strategy used (CV or split).
        fold_display_num (int): Current fold number for display.
        config (Dict): Configuration dictionary (needed for paths in plot_confusion_matrix).

    Returns:
        Tuple[Dict, str]: A dictionary of metrics and the path to the confusion matrix image.
    """
    model.eval()
    all_preds, all_labels = [], []
    
    if strategy == "cross_validation":
        prefix = f"Model: {model_name} Fold: {fold_display_num}"
    else:
        prefix = f"Model: {model_name}"

    desc = f"{prefix} [Test]"
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    
    with torch.no_grad(): # Disable gradient calculation
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    pbar.close()
    
    # Calculate metrics
    metrics = calculate_metrics(all_labels, all_preds)
    
    # Generate confusion matrix (requires config to know where to save 'tmp' files)
    cm_image_path = plot_confusion_matrix(all_labels, all_preds, config)
    
    return metrics, cm_image_path


def run_evaluation_fold(model_instance: nn.Module, 
                        test_loader: torch.utils.data.DataLoader, 
                        model_path: str, 
                        config: Dict[str, Any], 
                        fold: int, 
                        model_name: str) -> Dict[str, float]:
    """
    Loads the best saved model from disk and evaluates it on the test set.
    """
    device = config["device"]
    model = model_instance.to(device)

    strategy = config.get("data_strategy", "simple_split")
    fold_display_num = fold + 1
    run_label = f"Fold {fold_display_num}" if strategy == "cross_validation" else "Run"
    
    print(f"{run_label} - Loading best model for testing...")
    
    try:
        # Load weights
        # 'weights_only=True' is a security best practice in newer PyTorch versions
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Skipping test.")
        return {"error": "model file not found"}
    except Exception as e:
        print(f"ERROR: Could not load model from {model_path}: {e}. Skipping test.")
        return {"error": str(e)}

    # Run the test loop, passing the config dictionary
    fold_metrics, cm_image_path = test_model(
        model, test_loader, device,
        model_name, strategy, fold_display_num,
        config
    )
    
    f1_macro = fold_metrics.get("f1-score_macro", 0)

    # Log results to WandB
    wandb.log(fold_metrics)
    wandb.log({"confusion_matrix": wandb.Image(cm_image_path)})
    
    # Cleanup temporary confusion matrix file
    if os.path.exists(cm_image_path):
        os.remove(cm_image_path)
    else:
        # Just a safeguard in case the path logic fails subtly
        if "temp_cm.png" not in cm_image_path:
            print(f"Warning: Could not find {cm_image_path} to delete.")

    print(f"{run_label} - Testing complete.")

    # Handle weight preservation logic based on YAML config
    if not config.get("save_weights", False):
        print(f"Deleting model weights (save_weights is False)")
        if os.path.exists(model_path):
            os.remove(model_path)
    else:
        print(f"Model weights preserved at {model_path} (F1-Macro: {f1_macro:.4f})")

    return fold_metrics