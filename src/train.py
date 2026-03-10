# src/train.py
"""
This script manages the training and validation loops for deep learning models using PyTorch.
It integrates Weights & Biases (WandB) for experiment tracking and saves the best model
state based on validation loss.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import copy
from typing import Tuple, Dict, Any
from tqdm import tqdm

from utils.metrics import calculate_metrics


def apply_gradient_clipping(model: nn.Module, config: Dict[str, Any]) -> None:
    """
    Apply gradient clipping to model parameters based on configuration.
    
    Args:
        model (nn.Module): The model to apply clipping to.
        config (Dict[str, Any]): Configuration dictionary containing gradient_clipping settings.
    """
    grad_clip_config = config.get("gradient_clipping", {})
    
    if not grad_clip_config.get("enabled", False):
        return  # No clipping if disabled
    
    method = grad_clip_config.get("method", "norm")
    
    if method == "norm":
        max_norm = grad_clip_config.get("max_norm", 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    elif method == "value":
        max_value = grad_clip_config.get("max_value", 0.1)
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=max_value)
    else:
        print(f"WARNING: Unknown gradient clipping method '{method}'. Skipping clipping.")

def train_epoch(model: nn.Module, 
                loader: torch.utils.data.DataLoader, 
                criterion: nn.Module, 
                optimizer: optim.Optimizer, 
                device: torch.device, 
                model_name: str, 
                strategy: str, 
                fold_display_num: int, 
                epoch_num: int, 
                total_epochs: int,
                config: Dict[str, Any]) -> float:
    """
    Executes a single training epoch.
    """
    model.train()
    running_train_loss = 0.0
    
    if strategy == "cross_validation":
        prefix = f"Model: {model_name} Fold: {fold_display_num}"
    else:
        prefix = f"Model: {model_name}"
    
    desc = f"{prefix} Epoch: {epoch_num}/{total_epochs} [Train]"
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        train_loss = criterion(logits, labels)
        train_loss.backward()
        
        # Apply gradient clipping if enabled
        apply_gradient_clipping(model, config)
        
        optimizer.step()
        
        running_train_loss += train_loss.item()
        pbar.set_postfix({"train_loss": f"{train_loss.item():.4f}"})

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
    Executes a validation epoch.
    """
    model.eval()
    running_val_loss = 0.0
    all_preds = []
    all_labels = []
    
    if strategy == "cross_validation":
        prefix = f"Model: {model_name} Fold: {fold_display_num}"
    else:
        prefix = f"Model: {model_name}"
        
    desc = f"{prefix} Epoch: {epoch_num}/{total_epochs} [Val]"
    pbar = tqdm(loader, total=len(loader), desc=desc, leave=False)
    
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            val_loss = criterion(logits, labels)
            
            running_val_loss += val_loss.item()
            
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({"val_loss": f"{val_loss.item():.4f}"})

    pbar.close()
    
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
    Orchestrates the full training and validation lifecycle for a specific fold.
    """
    device = config["device"]
    model = model_instance.to(device)
    
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = -1

    criterion = nn.CrossEntropyLoss()
    
    # Fetch optimizer
    optimizer_name = getattr(optim, config["optimizer"])
    optimizer = optimizer_name(model.parameters(), 
                               lr=config['learning_rate'], 
                               weight_decay=config["weight_decay"])
    
    # Save Path Handling (save_dir is a Path object)
    save_dir = config["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    
    strategy = config.get("data_strategy", "simple_split")
    fold_display_num = fold + 1
    run_label = f"Fold {fold_display_num}" if strategy == "cross_validation" else "Run"
    save_label = f"fold{fold_display_num}" if strategy == "cross_validation" else "simple_split"
    
    # Use Path / operator
    save_path = save_dir / f"{model_name}_{save_label}_best.pth"

    # --- Main Epoch Loop ---
    for epoch in range(config['epochs']):
        epoch_num = epoch + 1
        
        avg_train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, 
            model_name, strategy, fold_display_num, epoch_num, config['epochs'], config
        )
        
        avg_val_loss, epoch_metrics = evaluate_epoch(
            model, val_loader, criterion, device, 
            model_name, strategy, fold_display_num, epoch_num, config['epochs']
        )
        
        wandb.log({
            "epoch": epoch_num,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "accuracy": epoch_metrics["accuracy"],
            "f1-score_macro": epoch_metrics["f1-score_macro"]
        })
        
        print(f"  Epoch {epoch_num}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {epoch_metrics['accuracy']:.2f}%")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    torch.save(best_model_state, save_path)
    
    print(f"{run_label} - Training complete. Best model (Epoch {best_epoch + 1}) saved to {save_path}")
    
    return str(save_path) # Return string for compatibility