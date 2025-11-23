# src/dataloader.py
"""
Data Loading and Processing Module.

This module manages dataset creation, data splitting strategies (Cross-Validation, 
Simple Split, or Pre-defined splits), and DataLoader instantiation. It handles 
both standard image files and pre-processed Numpy arrays, applying appropriate 
augmentations and transformations.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm


class MyDataset(Dataset):
    """
    Custom Dataset class that handles both image files and .npy slice files.
    It applies specific transformations based on the input type and augmentation flag.
    """
    def __init__(self, 
                 file_label_list: List[Tuple[Path, int]], 
                 config: Dict[str, Any], 
                 augment: bool = False, 
                 model_name: str = None):
        """
        Args:
            file_label_list (List): List of tuples (file_path, label_index).
            config (Dict): Configuration dictionary containing image sizes and settings.
            augment (bool): Whether to apply data augmentation.
            model_name (str): Name of the model (unused in logic but kept for consistency).
        """
        self.file_label_list = file_label_list
        self.augment = augment
        self.input_type = config.get("input_type", "image")
        self.model_name = model_name 

        cfg_image_size = int(config.get("image_size", 256))
        INPUT_SIZE = (cfg_image_size, cfg_image_size)
        self.input_size = INPUT_SIZE

        # --- Transforms for Standard Images (DS_Kaggle) ---
        self.transform_img_aug = transforms.Compose([
            transforms.Resize(INPUT_SIZE, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), #  
            transforms.ToTensor() 
        ])
        self.transform_img_no_aug = transforms.Compose([
            transforms.Resize(INPUT_SIZE, antialias=True), 
            transforms.ToTensor() 
        ])

        # --- Transforms for Numpy Arrays (DS_ADNI) ---
        self.transform_npy_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE, antialias=True),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05), # Add Gaussian noise
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)),         # Clip values
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.))          # Ensure [0, 1]
        ])
        self.transform_npy_no_aug = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(INPUT_SIZE, antialias=True), 
            transforms.ToTensor() 
        ])

    def __len__(self) -> int:
        return len(self.file_label_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves an item by index, processing it as an image or numpy array.
        """
        try:
            file_path, label = self.file_label_list[idx]

            if self.input_type == "image":
                image = Image.open(file_path).convert("L") # Convert to grayscale
                
                if self.augment:
                    image_tensor = self.transform_img_aug(image)
                else:
                    image_tensor = self.transform_img_no_aug(image)

            elif self.input_type == "numpy":
                slice2d = np.load(file_path)
                slice2d = slice2d.astype(np.float32)
                
                # Normalize to 0-1 range
                slice2d = (slice2d - np.min(slice2d)) / (np.max(slice2d) - np.min(slice2d) + 1e-8)
                
                if self.augment:
                    image_tensor = self.transform_npy_aug(slice2d)
                else:
                    image_tensor = self.transform_npy_no_aug(slice2d)
            
            else:
                raise ValueError(f"Unknown input_type: {self.input_type}")

            return image_tensor, label

        except Exception as e:
            print(f"ERROR loading {self.file_label_list[idx][0]}: {e}")
            
            # Return a black image on error to prevent crash
            h = self.input_size[0]
            return torch.zeros(1, h, h), -1


def _scan_all_files(data_base_dir: Path, class_folders: List[str]) -> Tuple[List[Path], List[int]]:
    """
    Scans the base directory recursively and returns lists of relative paths and labels.

    Args:
        data_base_dir (Path): Root directory of the dataset.
        class_folders (List[str]): List of subfolder names representing classes.

    Returns:
        Tuple[List, List]: A list of file paths and a list of corresponding label indices.
    """
    all_files_relative = []
    all_labels = []
    
    print(f"INFO: [On-the-fly] Scanning {data_base_dir} for images...")
    
    for label, folder in enumerate(class_folders): 
        folder_path = data_base_dir / folder
        if not folder_path.exists():
            print(f"Warning: Folder {folder_path} not found. Skipping...")
            continue
        
        desc = f"Scanning {folder}"
        for filename in tqdm(os.listdir(folder_path), desc=desc, leave=False):
            if filename.lower().endswith((".jpg", ".png", "jpeg")):
                all_files_relative.append(Path(folder) / filename)
                all_labels.append(label)
                
    if not all_files_relative:
        raise FileNotFoundError(f"No images found in {data_base_dir}. "
                                f"Check folders: {class_folders}")
                                
    print(f"INFO: [On-the-fly] Scan complete. Total images: {len(all_files_relative)}")
    return all_files_relative, all_labels


def _get_cv_splits(config: Dict[str, Any], 
                   all_files_relative: List[Path], 
                   all_labels: List[int]) -> List[Dict[str, List]]:
    """
    Generates K folds for Stratified Cross-Validation.
    """
    n_splits = config["n_splits_cv"]
    random_state = config["split_random_state"]
    val_split_size = config["val_split_size"] 
    data_base_dir = Path(config["data_base_dir"])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    all_splits_data = []

    print(f"INFO: [On-the-fly] Generating {n_splits} folds (CV)...")
    
    all_files_np = np.array(all_files_relative)
    all_labels_np = np.array(all_labels)

    # Loop through K-Fold splits
    for train_val_idx, test_idx in skf.split(all_files_np, all_labels_np):
        
        test_files = all_files_np[test_idx]
        test_labels = all_labels_np[test_idx]
        
        train_val_files = all_files_np[train_val_idx]
        train_val_labels = all_labels_np[train_val_idx]
        
        if len(train_val_files) == 0:
            print("ERROR: Not enough data for train/val split. Check n_splits.")
            continue
        
        # Split the Training set further to create a Validation set
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_files)),
            test_size=val_split_size,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        train_files = train_val_files[train_idx]
        train_labels = train_val_labels[train_idx]
        val_files = train_val_files[val_idx]
        val_labels = train_val_labels[val_idx]

        # Structure data for the current fold
        split_data = {
            "train": [(data_base_dir / train_files[i], train_labels[i]) for i in range(len(train_files))],
            "val": [(data_base_dir / val_files[i], val_labels[i]) for i in range(len(val_files))],
            "test": [(data_base_dir / test_files[i], test_labels[i]) for i in range(len(test_files))]
        }
        all_splits_data.append(split_data)
        
    print(f"INFO: [On-the-fly] {len(all_splits_data)} folds generated.")
    return all_splits_data


def _get_simple_split(config: Dict[str, Any], 
                      all_files_relative: List[Path], 
                      all_labels: List[int]) -> List[Dict[str, List]]:
    """
    Generates a single Train/Val/Test split (Hold-out strategy).
    """
    test_split_size = config["test_split_size"]
    val_split_size = config["val_split_size"]
    random_state = config["split_random_state"]
    data_base_dir = Path(config["data_base_dir"])

    print(f"INFO: [On-the-fly] Generating 1 simple split...")

    # First split: Separate Test set
    train_val_idx, test_idx = train_test_split(
        np.arange(len(all_files_relative)),
        test_size=test_split_size,
        random_state=random_state,
        stratify=all_labels
    )
    
    all_files_np = np.array(all_files_relative)
    all_labels_np = np.array(all_labels)

    test_files = all_files_np[test_idx]
    test_labels = all_labels_np[test_idx]
    
    train_val_files = all_files_np[train_val_idx]
    train_val_labels = all_labels_np[train_val_idx]

    # Second split: Separate Validation set from remaining data
    if len(train_val_files) > 0:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_val_files)),
            test_size=val_split_size,
            random_state=random_state,
            stratify=train_val_labels
        )
        
        train_files = train_val_files[train_idx]
        train_labels = train_val_labels[train_idx]
        val_files = train_val_files[val_idx]
        val_labels = train_val_labels[val_idx]
    else:
        print("WARNING: train_val set is empty. Train and Val will be empty.")
        train_files, train_labels = np.array([]), np.array([])
        val_files, val_labels = np.array([]), np.array([])

    split_data = {
        "train": [(data_base_dir / train_files[i], train_labels[i]) for i in range(len(train_files))],
        "val": [(data_base_dir / val_files[i], val_labels[i]) for i in range(len(val_files))],
        "test": [(data_base_dir / test_files[i], test_labels[i]) for i in range(len(test_files))]
    }
    
    print(f"INFO: [On-the-fly] Simple split generated.")
    print(f"INFO: Sizes -> Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
    
    return [split_data]


def _scan_pre_split_txt(txt_file_path: Path, label_map: Dict[str, int]) -> List[Tuple[str, int]]:
    """ 
    Scans a .txt file (e.g., 'train_preprocessed_paths.txt') and returns 
    a list of (absolute_path, label).
    """
    file_label_list = []
    if not os.path.exists(txt_file_path):
         raise FileNotFoundError(f"Split file not found: {txt_file_path}")

    with open(txt_file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split(',')
            if len(parts) >= 2:
                slice_path, group = parts[0], parts[1]
                
                if group not in label_map:
                    continue
                
                file_label_list.append((slice_path, label_map[group]))
                
    if not file_label_list:
        raise ValueError(f"No valid data found in {txt_file_path}")
        
    return file_label_list


def _get_pre_split_txt(config: Dict[str, Any]) -> List[Dict[str, List]]:
    """
    Loads splits from predefined .txt files instead of scanning directories.
    """
    data_base_dir = Path(config["data_base_dir"])
    
    label_map = {name: i for i, name in enumerate(config["class_names"])}
    
    print(f"INFO: [Pre-split TXT] Loading from .txt in {data_base_dir}...")
    
    try:
        train_txt = data_base_dir / 'train_preprocessed_paths.txt'
        val_txt = data_base_dir / 'val_preprocessed_paths.txt'
        test_txt = data_base_dir / 'test_preprocessed_paths.txt'

        train_files = _scan_pre_split_txt(train_txt, label_map)
        val_files = _scan_pre_split_txt(val_txt, label_map)
        test_files = _scan_pre_split_txt(test_txt, label_map)
        
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        raise
    
    print(f"INFO: [Pre-split TXT] Splits loaded. Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

    split_data = [{
        "train": train_files,
        "val": val_files,
        "test": test_files
    }]
    return split_data


def get_dataloaders(config: Dict[str, Any], model_name: str = None) -> List[Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Main entry point to initialize and return DataLoaders based on the config strategy.
    
    Args:
        config (Dict): Configuration dictionary.
        model_name (str): Optional model name for logging/dataset attribution.
        
    Returns:
        List[Tuple]: A list of tuples, where each tuple is (train_loader, val_loader, test_loader).
                     The list length depends on the number of folds to run.
    """
    batch_size = config["batch_size"]
    split_type = config.get("split_type", "on_the_fly") 
    
    all_splits_data = [] 
    
    # Determine how to gather data (scan folders vs read txt files)
    if split_type == "on_the_fly":
        data_base_dir = Path(config["data_base_dir"])
        class_folders = config["class_names"]
        strategy = config["data_strategy"]
        
        all_files_relative, all_labels = _scan_all_files(data_base_dir, class_folders)
        
        if strategy == "cross_validation":
            all_splits_data = _get_cv_splits(config, all_files_relative, all_labels)
        elif strategy == "simple_split":
            all_splits_data = _get_simple_split(config, all_files_relative, all_labels)
        else:
            raise ValueError(f"Unknown data strategy: '{strategy}'")

    elif split_type == "pre_split_txt":
        all_splits_data = _get_pre_split_txt(config)
        print(f"INFO: 'pre_split_txt' strategy selected. 1 split will be executed.")
        
    else:
        raise ValueError(f"Unknown split_type: '{split_type}'")

    dataloaders_list = []
    folds_to_run = config["folds_to_run"]
    
    print(f"INFO: Creating DataLoaders for {len(folds_to_run)} fold(s) to execute...")
    
    # Iterate over the selected folds to create DataLoaders
    for fold_index in folds_to_run:
        if fold_index >= len(all_splits_data):
            raise IndexError(f"Tried to load fold {fold_index+1} but only {len(all_splits_data)} splits were generated.")
            
        fold_data = all_splits_data[fold_index]
        
        augment_train = config.get("augment_train", False)
        
        train_dataset = MyDataset(fold_data["train"], config=config, augment=augment_train, model_name=model_name)
        val_dataset = MyDataset(fold_data["val"], config=config, augment=False, model_name=model_name)
        test_dataset = MyDataset(fold_data["test"], config=config, augment=False, model_name=model_name)

        sampler = None
        shuffle_train = True
        
        # Configure WeightedRandomSampler if requested to handle class imbalance
        if config.get("use_weighted_sampler", False):
            print("INFO: Using WeightedRandomSampler for training set.")
            
            if len(train_dataset) == 0:
                print("WARNING: Training dataset is empty. Cannot use WeightedSampler.")
            else:
                labels = torch.tensor([label for _, label in train_dataset.file_label_list])
                class_sample_count = torch.tensor(
                    [(labels == t).sum().item() for t in range(config["num_classes"])]
                )
                class_sample_count[class_sample_count[class_sample_count == 0].long()] = 1 
                
                weight = 1.0 / class_sample_count.float()
                samples_weight = torch.tensor([weight[t] for t in labels])
                
                sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)
                shuffle_train = False 
            
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle_train, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        dataloaders_list.append((train_loader, val_loader, test_loader))

    if not dataloaders_list:
        print(f"WARNING: No dataloaders were loaded. Check 'folds_to_run'.")
        
    return dataloaders_list