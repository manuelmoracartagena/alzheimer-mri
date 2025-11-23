# scripts/prepare_data_kaggle.py
"""
Optimized Preprocessing Script for Kaggle Image Dataset (Multiprocessing).

This script processes 2D images from input directories by converting them to
grayscale, padding them to a fixed 256x256 canvas, and saving them to
output directories. It uses parallel processing to maximize efficiency.

Pipeline Steps:
1.  Task Collection: Scans input directories to map source files to destination paths.
2.  Parallel Processing: Distributes image processing tasks across CPU cores.
3.  Image Transformation: Converts to grayscale -> Numpy Array -> Pads to 256x256.
"""

import os
import numpy as np
import concurrent.futures
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Optional

# --- Configuration ---
# Number of parallel processes. 
# Since 2D image processing is CPU-bound but lightweight, usually cpu_count() is safe.
MAX_WORKERS = os.cpu_count() or 1
TARGET_SIZE = (256, 256)

# --- Worker Function (Must be picklable) ---

def process_image_task(args: Tuple[str, str]) -> Optional[str]:
    """
    Worker function to process a single image.
    
    Args:
        args (tuple): A tuple containing (input_path, output_path).
        
    Returns:
        str: Error message if an exception occurs, None otherwise.
    """
    input_path, output_path = args
    
    try:
        # Open the image
        with Image.open(input_path) as img:
            # Convert to grayscale
            grayscale_image = img.convert("L")
            
            # Convert to NumPy array
            image_array = np.array(grayscale_image)
            
            h, w = image_array.shape
            
            # safety check: if image is larger than target, resize it first
            if h > TARGET_SIZE[0] or w > TARGET_SIZE[1]:
                grayscale_image = grayscale_image.resize(TARGET_SIZE)
                image_array = np.array(grayscale_image)
                h, w = image_array.shape

            # Calculate offsets to center the image
            offset_x = (TARGET_SIZE[0] - h) // 2
            offset_y = (TARGET_SIZE[1] - w) // 2
            
            # Create the canvas (black background)
            padded_array = np.zeros(TARGET_SIZE, dtype=image_array.dtype)
            
            # Embed the image into the canvas
            padded_array[offset_x:offset_x + h, offset_y:offset_y + w] = image_array
            
            # Convert back to PIL Image
            final_image = Image.fromarray(padded_array).convert("L")
            
            # Save the result
            final_image.save(output_path)
            
        return None # Success
        
    except Exception as e:
        return f"Error processing {os.path.basename(input_path)}: {e}"

# --- Helper Functions ---

def collect_tasks(input_folders: List[str], output_folders: List[str]) -> List[Tuple[str, str]]:
    """
    Scans directories and creates a list of (source, destination) tasks.
    """
    tasks = []
    print(f"\n{'='*60}")
    print(f" Scanning directories...")
    print(f"{'='*60}")

    for in_folder, out_folder in zip(input_folders, output_folders):
        if not os.path.exists(in_folder):
            print(f" [!] Warning: Input folder not found: {in_folder}")
            continue
            
        # Ensure output directory exists
        os.makedirs(out_folder, exist_ok=True)
        
        files = [f for f in os.listdir(in_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        print(f" Folder: {os.path.basename(in_folder)} -> Found {len(files)} images.")
        
        for file in files:
            src = os.path.join(in_folder, file)
            dst = os.path.join(out_folder, file)
            tasks.append((src, dst))
            
    return tasks

def main():
    # Enable multiprocessing support for Windows/macOS
    multiprocessing.freeze_support()

    print(f"\n{'#'*60}")
    print(f" KAGGLE DATASET PREPROCESSING (Parallel)")
    print(f" Active Workers: {MAX_WORKERS}")
    print(f"{'#'*60}")

    # --- Configuration: Paths ---
    project_root = Path(__file__).resolve().parent.parent
    
    # Input directories
    input_folders = [
        os.path.join(project_root, "data/kaggle/augmented/MildDemented"),
        os.path.join(project_root, "data/kaggle/augmented/ModerateDemented"),
        os.path.join(project_root, "data/kaggle/augmented/NonDemented"),
        os.path.join(project_root, "data/kaggle/augmented/VeryMildDemented")
    ]

    # Output directories
    output_folders = [
        os.path.join(project_root, "data/kaggle/resized/MildDemented"),
        os.path.join(project_root, "data/kaggle/resized/ModerateDemented"),
        os.path.join(project_root, "data/kaggle/resized/NonDemented"),
        os.path.join(project_root, "data/kaggle/resized/VeryMildDemented")
    ]

    # 1. Collect all files to process
    tasks = collect_tasks(input_folders, output_folders)
    
    if not tasks:
        print(" No images found to process. Exiting.")
        return

    print(f"\n Starting processing of {len(tasks)} images...")

    # 2. Execute in parallel
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Use chunksize to improve performance for large lists of small tasks
        results = list(tqdm(executor.map(process_image_task, tasks, chunksize=50), 
                            total=len(tasks), 
                            desc=" Processing Images", 
                            unit="img"))
        
        # Filter errors
        errors = [res for res in results if res is not None]

    # 3. Final Report
    print(f"\n{'='*60}")
    print(f" PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f" Processing finished successfully")
    print(f" Errors encountered:     {len(errors)}")
    
    if errors:
        print("\n Sample errors:")
        for err in errors[:5]:
            print(f" - {err}")
        print(f" (See logs for full details)")

if __name__ == "__main__":
    main()