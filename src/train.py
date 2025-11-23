# src/train.py
"""
This script manages the training and validation loops for deep learning models using PyTorch.
It integrates Weights & Biases (WandB) for experiment tracking and saves the best model
state based on validation loss.

Key functions include single-epoch training, single-epoch evaluation, and the 
orchestration of the full training run for a specific data fold or split.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import copy
import os
from typing import Tuple, Dict, Any
from tqdm import tqdm
from utils.metrics import calculate_metrics

def train_epoch(model: nn.Module, 
                loader: torch.utils.data.DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device, 
                model_name: str, 
                strategy: str, 
                fold_display_num: int, 
                epoch_num: int, 
                total_epochs: int) -> float:
    """
    Executes a single training epoch, updating model weights based on loss.

    Args:
        model (nn.Module): The neural network model to train.
        loader (DataLoader): DataLoader containing the training dataset.
        criterion (nn.Module): Loss function (e.g., CrossEntropyLoss).
        optimizer (optim.Optimizer): Optimizer (e.g., Adam, SGD).
        device (torch.device): Device to run computations on (CPU or GPU).
        model_name (str): Name of the model for display purposes.
        strategy (str): Training strategy ('cross_validation' or 'simple_split').
        fold_display_num (int): Current fold number (for display).
        epoch_num (int): Current epoch number.
        total_epochs (int): Total number of epochs to run.

    Returns:
        float: The average training loss for the epoch.
    """
    model.train()
    running_train_loss = 0.0
    
    # Dynamically construct the progress bar description prefix based on strategy
    if strategy == "cross_validation":
        prefix = f"Model: {model_name} Fold: {fold_display_num}"
    else:
        prefix = f"Model: {model_name}"
    
    desc = f"{prefix} Epoch: {epoch_num}/{total_epochs} [Train]"
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device) # Move data to target device
        
        optimizer.zero_grad()                                 # Clear previous gradients
        logits = model(images)                                # Forward pass
        train_loss = criterion(logits, labels)                # Calculate loss
        train_loss.backward()                                 # Backward pass (gradients)
        optimizer.step()                                      # Update weights
        
        running_train_loss += train_loss.item()
        pbar.set_postfix({"train_loss": f"{train_loss.item():.4f}"}) # Update progress bar

    pbar.close()
    
    return running_train_loss / len(loader) if loader else 0.0


def evaluate_epoch(model: nn.Module, 
                   loader: torch.utils.data.DataLoader, 
                   criterion: nn.Module, 
                   device: torch.device, 
                   model_name: str, 
                   strategy: str, 
                   fold_display_num: int, 
                   epoch_num: int, 
                   total_epochs: int) -> Tuple[float, Dict[str, float]]:
    """
    Executes a validation epoch to evaluate model performance without updating weights.

    Args:
        model (nn.Module): The neural network model to evaluate.
        loader (DataLoader): DataLoader containing the validation dataset.
        criterion (nn.Module): Loss function.
        device (torch.device): Device to run computations on.
        model_name (str): Name of the model.
        strategy (str): Training strategy.
        fold_display_num (int): Current fold number.
        epoch_num (int): Current epoch number.
        total_epochs (int): Total number of epochs.

    Returns:
        Tuple[float, Dict]: A tuple containing the average validation loss and a dictionary of metrics.
    """
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Dynamically construct the progress bar description prefix
    if strategy == "cross_validation":
        prefix = f"Model: {model_name} Fold: {fold_display_num}"
    else:
        prefix = f"Model: {model_name}"
        
    desc = f"{prefix} Epoch: {epoch_num}/{total_epochs} [Val]"
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    
    with torch.no_grad():                                     # Disable gradient calculation for inference
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_loss = criterion(logits, labels)
            
            running_val_loss += val_loss.item()
            
            _, predicted = torch.max(logits, 1)               # Get class with highest probability
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

    pbar.close()
    
    # Calculate comprehensive metrics (Accuracy, F1, Precision, Recall)
    epoch_metrics = calculate_metrics(all_labels, all_preds)
    avg_val_loss = running_val_loss / len(loader) if loader else 0.0
    
    return avg_val_loss, epoch_metrics


def run_training_fold(model_instance: nn.Module, 
                      train_loader: torch.utils.data.DataLoader, 
                      val_loader: torch.utils.data.DataLoader, 
                      config: Dict[str, Any], 
                      fold: int, 
                      model_name: str) -> str:
    """
    Orchestrates the full training and validation lifecycle for a specific fold or split.
    Initializes the optimizer, handles the epoch loop, logs to WandB, and saves the best model.

    Args:
        model_instance (nn.Module): The model instance to train.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        config (Dict): Configuration dictionary containing hyperparameters.
        fold (int): Zero-indexed fold number.
        model_name (str): Name of the model architecture.

    Returns:
        str: The file path where the best model state dict was saved.
    """
    device = config["device"]
    model = model_instance.to(device)
    
    # Track the best model state to prevent overfitting
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = -1

    criterion = nn.CrossEntropyLoss()
    
    # Dynamically fetch optimizer class from string name in config
    optimizer_name = getattr(optim, config["optimizer"])
    optimizer = optimizer_name(model.parameters(), 
                               lr=config['learning_rate'], 
                               weight_decay=config["weight_decay"])
    
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    # Define naming conventions based on strategy
    strategy = config.get("data_strategy", "simple_split")
    fold_display_num = fold + 1
    run_label = f"Fold {fold_display_num}" if strategy == "cross_validation" else "Run"
    save_label = f"fold{fold_display_num}" if strategy == "cross_validation" else "simple_split"
    save_path = os.path.join(save_dir, f"{model_name}_{save_label}_best.pth")

    # --- Main Epoch Loop ---
    for epoch in range(config['epochs']):
        epoch_num = epoch + 1
        
        # Execute Training Step
        avg_train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            model_name, strategy, fold_display_num, epoch_num, config['epochs']
        )
        
        # Execute Validation Step
        avg_val_loss, epoch_metrics = evaluate_epoch(
            model, val_loader, criterion, device, 
            model_name, strategy, fold_display_num, epoch_num, config['epochs']
        )
        
        # Log metrics to Weights & Biases
        wandb.log({
            "epoch": epoch_num,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": epoch_metrics["accuracy"],
            "precision_weighted": epoch_metrics["precision_weighted"],
            "precision_macro": epoch_metrics["precision_macro"],
            "recall_weighted": epoch_metrics["recall_weighted"],
            "recall_macro": epoch_metrics["recall_macro"],
            "f1-score_weighted": epoch_metrics["f1-score_weighted"],
            "f1-score_macro": epoch_metrics["f1-score_macro"]
        })
        
        print(f"  Epoch {epoch_num}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {epoch_metrics['accuracy']:.2f}%")

        # Checkpoint: Save model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    # Save the best performing model to disk
    torch.save(best_model_state, save_path)
    
    print(f"{run_label} - Training complete. Best model (Epoch {best_epoch + 1}) saved to {save_path}")
    
    return save_path