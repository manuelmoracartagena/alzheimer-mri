# src/utils/metrics.py
"""
Metrics Calculation and Visualization Module.

This module calculates standard classification metrics (Accuracy, Precision, Recall, F1)
and handles the generation and saving of Confusion Matrices.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import wandb
from typing import Dict, List, Any
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score


def calculate_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    """
    Calculates various classification metrics.

    Args:
        y_true (List[int]): True ground truth labels.
        y_pred (List[int]): Predicted labels.

    Returns:
        Dict[str, float]: Dictionary containing calculated metrics.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    return {
        "accuracy": 100 * accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1-score_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1-score_macro": f1_macro
    }


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], config: Dict[str, Any]) -> str:
    """
    Generates and saves a visualization of the confusion matrix.

    Args:
        y_true (List): True labels.
        y_pred (List): Predicted labels.
        config (Dict): Configuration dictionary for paths.

    Returns:
        str: The file path of the saved confusion matrix image.
    """
    # --- NOTE ---
    # The order MUST match the 'enumerate' in datasplitter.py / dataloader
    # 0: "MildDemented"
    # 1: "ModerateDemented"
    # 2: "NonDemented"
    # 3: "VeryMildDemented"
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))            # Adjusted for manageable size
    plt.rcParams.update({'font.size': 10}) # Smaller font size
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.title("Confusion Matrix")

    tmp_dir = config["tmp_dir"]
    os.makedirs(tmp_dir, exist_ok=True)
    
    # Use a unique temporary filename to avoid collisions
    cm_image_path = tmp_dir / f"cm_{wandb.run.id if wandb.run else 'temp'}.png"
    plt.savefig(cm_image_path)
    plt.close()
    
    return str(cm_image_path) 


def print_final_metrics(model_metrics: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Prints the aggregated final metrics for each model (Mean +/- Std Dev).

    Args:
        model_metrics (Dict): Dictionary containing metrics lists per model.
    """
    print("\n--- Final Results per Model (Mean ± Standard Deviation) ---")
    
    for model_name, metrics in model_metrics.items():
        if not metrics:
            print(f"\nModel: {model_name} - No metrics recorded.")
            continue

        print(f"\n✨ Model: {model_name}")
        
        # Iterate over all collected metrics
        for metric_name, values in metrics.items():
            if values:
                mean = np.mean(values)
                std = np.std(values)
                # Format metric name for better readability
                display_name = metric_name.replace('_', ' ').replace('-', ' ').title()
                print(f"   - Mean {display_name}: {mean:.4f} ± {std:.4f}")