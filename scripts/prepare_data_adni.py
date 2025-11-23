# scripts/prepare_data_adni.py
"""
Optimized Preprocessing Script for ADNI NIfTI Dataset (Multiprocessing).

Pipeline Steps:
0.  File Search: Recursively finds raw .nii/.nii.gz files (filtering out previous artifacts).
1.  Transposition: Reorients images to a standard shape (240, 256, 160).
2.  Padding: Pads images to a target XY plane size (256, 256).
3.  Data Split: Splits data into Train/Val/Test by PATIENT ID (preventing data leakage).
4.  Slice Extraction: Extracts central 2D slices, normalizes them, and saves as .npy.
"""

import pandas as pd
import nibabel as nib
import numpy as np
import os
import gc
import multiprocessing
import concurrent.futures
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict

# --- Configuration ---
# Adjustable number of workers. 
# Set to 4 or 8 for stability. Set higher (e.g., os.cpu_count()) only if you have >32GB RAM.
MAX_WORKERS = min(8, os.cpu_count() or 1) 

# --- Top-Level Worker Functions (Must be pickleable) ---

def get_output_path(input_path: str, suffix: str) -> str:
    """Generates the output path handling double extensions like .nii.gz."""
    base, ext = os.path.splitext(input_path)
    if ext == '.gz':
        base, ext2 = os.path.splitext(base)
        out_path = base + suffix + ext2 + ext
    else:
        out_path = base + suffix + ext
    return out_path

def worker_get_shape(file_path: str) -> Optional[Tuple[str, Tuple]]:
    """Worker for Step 0: Loads header to get image shape."""
    try:
        img = nib.load(file_path)
        return (file_path, img.shape)
    except Exception:
        return None

def worker_transpose(args) -> Tuple[str, str]:
    """Worker for Step 1: Transposes the 3D volume."""
    nii_path, target_shape = args
    try:
        img = nib.load(nii_path)
        # Skip if shape doesn't match the specific target for transposition
        if img.shape != target_shape:
             return (nii_path, nii_path)

        data = img.get_fdata()
        # Transpose from (160, 240, 256) -> (240, 256, 160)
        transposed_data = np.transpose(data, (1, 2, 0))
        
        new_img = nib.Nifti1Image(transposed_data, affine=img.affine, header=img.header)
        out_path = get_output_path(nii_path, '_transposed')
        nib.save(new_img, out_path)
        
        return (nii_path, out_path)
    except Exception as e:
        print(f" [!] Error transposing {os.path.basename(nii_path)}: {e}")
        return (nii_path, nii_path)

def worker_pad(args) -> Tuple[str, str]:
    """Worker for Step 2: Pads the image to target XY dimensions."""
    nii_path, target_xy = args
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        y, x, _ = data.shape

        if (y, x) != target_xy:
            pad_y = max(0, target_xy[0] - y)
            pad_x = max(0, target_xy[1] - x)
            
            pad_before_y = pad_y // 2
            pad_after_y = pad_y - pad_before_y
            pad_before_x = pad_x // 2
            pad_after_x = pad_x - pad_before_x

            pad_width = ((pad_before_y, pad_after_y), (pad_before_x, pad_after_x), (0, 0))
            padded_data = np.pad(data, pad_width, mode='constant', constant_values=0)
            
            out_path = get_output_path(nii_path, '_padded')
            new_img = nib.Nifti1Image(padded_data, affine=img.affine, header=img.header)
            nib.save(new_img, out_path)
            
            return (nii_path, out_path)
        else:
            return (nii_path, nii_path)
    except Exception as e:
        print(f" [!] Error padding {os.path.basename(nii_path)}: {e}")
        return (nii_path, nii_path)

def worker_process_slices(args) -> List[str]:
    """Worker for Step 4: Extracts, normalizes, and saves 2D slices."""
    img_path, group, patient_id, output_dir, num_slices = args
    results = []
    
    try:
        nii = nib.load(img_path)
        img = nii.get_fdata()
        
        Z = img.shape[2]
        start = (Z - num_slices) // 2
        end = start + num_slices
        
        # Boundary checks
        start = max(0, start)
        end = min(Z, end)

        for i, slice_idx in enumerate(range(start, end)):
            slice2d = img[:, :, slice_idx]
            
            # Min-Max Normalization
            slice2d = slice2d.astype(np.float32)
            min_val = np.min(slice2d)
            max_val = np.max(slice2d)
            denom = max_val - min_val
            if denom == 0: denom = 1e-8
            
            slice2d = (slice2d - min_val) / denom
            
            # Save slice
            slice_name = f"{patient_id}_slice{i:03d}.npy"
            slice_path = os.path.join(output_dir, slice_name)
            np.save(slice_path, slice2d)
            
            # Record metadata string
            results.append(f"{slice_path},{group},{patient_id}\n")
            
    except Exception as e:
        print(f" [!] Error processing slices for {os.path.basename(img_path)}: {e}")
        
    return results

# --- Pipeline Functions ---

def print_header(step_num: int, title: str):
    """Helper for console output."""
    print(f"\n{'='*60}")
    print(f" STEP {step_num}: {title}")
    print(f"{'='*60}")

def find_and_shape_files(root_dir: str) -> pd.DataFrame:
    print_header(0, "Recursive File Search (Parallel)")
    print(f" Root Directory: {root_dir}")
    
    all_files = []
    # 1. Scan directory (filtering out previous run artifacts)
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            # IMPORTANT: Skip processed files to avoid infinite loops/corruption
            if '_transposed' in file or '_padded' in file:
                continue
            
            if file.endswith('.nii') or file.endswith('.nii.gz'):
                all_files.append(os.path.join(dirpath, file))
    
    print(f" Found {len(all_files)} potential raw files.")
    print(" Reading headers in parallel to extract shapes...")
    
    # 2. Parallel Header Reading
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = list(tqdm(executor.map(worker_get_shape, all_files), 
                            total=len(all_files), 
                            desc="Reading Headers", 
                            unit="file"))
        
    results = [res for res in futures if res is not None]
    
    df = pd.DataFrame(results, columns=["Path", "Shape"])
    print(f" Step 0 Complete. {len(df)} valid files indexed.")
    return df

def transpose_files(df: pd.DataFrame, target_shape: tuple) -> pd.DataFrame:
    print_header(1, "Image Transposition")
    
    files_to_process = df[df['Shape'] == target_shape]['Path'].tolist()
    print(f" Target Shape: {target_shape}")
    print(f" Files to transpose: {len(files_to_process)}")
    
    if not files_to_process:
        print(" No files need transposition.")
        df['FinalPath'] = df['Path']
        return df

    args_list = [(f, target_shape) for f in files_to_process]
    transposed_map = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_transpose, args_list), 
                            total=len(args_list), 
                            desc="Transposing", 
                            unit="img"))
        
    for original, new_path in results:
        transposed_map[original] = new_path

    df['FinalPath'] = df['Path'].apply(lambda x: transposed_map.get(x, x))
    print(" Step 1 Complete.")
    return df

def pad_files(df: pd.DataFrame, target_xy: tuple) -> pd.DataFrame:
    print_header(2, "Image Padding")
    print(f" Target Dimensions: {target_xy}")
    
    paths = df['FinalPath'].tolist()
    args_list = [(p, target_xy) for p in paths]
    padded_map = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(worker_pad, args_list), 
                            total=len(args_list), 
                            desc="Applying Padding", 
                            unit="img"))

    for inp, out in results:
        padded_map[inp] = out
        
    df['PaddedPath'] = df['FinalPath'].map(padded_map)
    print(" Step 2 Complete.")
    return df

def split_data(df, adni_csvs, test_size, val_size, random_state):
    print_header(3, "Patient-Level Stratified Split")
    
    # Helper to extract ID
    def extract_patient_id(path: str) -> Optional[str]:
        norm_path = os.path.normpath(path)
        parts = norm_path.split(os.sep)
        for part in parts:
            # Assumes format '002_S_0295'
            if '_' in part and len(part.split('_')) == 3 and part.split('_')[0].isdigit():
                return part
        return None

    print(" Loading Clinical Metadata CSVs...")
    subject_to_group = {}
    for csv_file in adni_csvs:
        try:
            meta_df = pd.read_csv(csv_file)
            for _, row in meta_df.iterrows():
                subject_to_group[row['Subject']] = row['Group']
        except Exception as e:
            print(f" [!] Warning: Could not load {os.path.basename(csv_file)}")

    # Map IDs and Groups
    df['PatientID'] = df['PaddedPath'].apply(extract_patient_id)
    df['Group'] = df['PatientID'].apply(lambda pid: subject_to_group.get(pid, 'Unknown'))
    
    # Filter invalid entries
    initial_count = len(df)
    df = df[df['Group'] != 'Unknown'].dropna(subset=['PatientID'])
    print(f" Filtered {initial_count - len(df)} images without metadata.")
    
    # Stratify by PATIENT, not by image
    patients_df = df[['PatientID', 'Group']].drop_duplicates().reset_index(drop=True)
    print(f" Unique Patients found: {len(patients_df)}")
    
    # First Split: Train vs (Val + Test)
    train_patients, temp_patients = train_test_split(
        patients_df, test_size=test_size, random_state=random_state, stratify=patients_df['Group']
    )
    # Second Split: Val vs Test
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=val_size, random_state=random_state, stratify=temp_patients['Group']
    )
    
    # Assign splits back to the main DataFrame
    train_ids = set(train_patients['PatientID'])
    val_ids = set(val_patients['PatientID'])
    test_ids = set(test_patients['PatientID'])
    
    def assign_split(pid):
        if pid in train_ids: return 'train'
        elif pid in val_ids: return 'val'
        elif pid in test_ids: return 'test'
        return 'none'

    df['split'] = df['PatientID'].apply(assign_split)
    
    # Statistics
    print(f"\n Split Statistics (Images):")
    print(df['split'].value_counts().to_string())
    
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'val']
    test_df = df[df['split'] == 'test']
    
    print("\n Step 3 Complete.")
    return train_df, val_df, test_df, df

def preprocess_and_save_slices(split_dfs, base_output_dir, num_slices):
    print_header(4, "Slice Extraction & Normalization")
    print(" Note: Using chunksize=1 and periodic GC to protect RAM.")

    for split_name, split_df in split_dfs.items():
        print(f"\n Processing Set: {split_name.upper()} ({len(split_df)} volumes)")
        
        output_dir = os.path.join(base_output_dir, f'preprocessed_{split_name}')
        os.makedirs(output_dir, exist_ok=True)
        
        tasks = []
        for _, row in split_df.iterrows():
            tasks.append((
                row['PaddedPath'], 
                row['Group'], 
                row['PatientID'], 
                output_dir, 
                num_slices
            ))
            
        all_lines = []
        
        # Using ProcessPool with chunksize=1 for memory safety
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            results_iter = tqdm(executor.map(worker_process_slices, tasks, chunksize=1), 
                                total=len(tasks), 
                                desc=f"Extracting {split_name}",
                                unit="vol")
            
            for i, lines in enumerate(results_iter):
                all_lines.extend(lines)
                # Force Garbage Collection every 500 images processed
                if i % 500 == 0:
                    gc.collect()

        # Save Master TXT file
        new_txt_path = os.path.join(base_output_dir, f'{split_name}_preprocessed_paths.txt')
        with open(new_txt_path, 'w') as f:
            f.writelines(all_lines)
            
        print(f" Saved index: {os.path.basename(new_txt_path)}")
        print(f" Total slices: {len(all_lines)}")
        
        # Final cleanup for this split
        gc.collect()

    print("\n Step 4 Complete.")

def main():
    # Necessary for multiprocessing on Windows/MacOS
    multiprocessing.freeze_support()
    
    print(f"\n{'#'*60}")
    print(f" ADNI DATASET PREPROCESSING (Parallel")
    print(f" Active Workers: {MAX_WORKERS}")
    print(f"{'#'*60}")

    # --- CONFIGURATION ---
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    ADNI_DIR = PROJECT_ROOT / 'data' / 'adni'
    DATASET_ROOT_DIR = str(ADNI_DIR)
    
    ADNI_METADATA_CSVS = [
        str(ADNI_DIR / 'ADNI1_Complete_1Yr_3T_5_05_2025.csv'),
        str(ADNI_DIR / 'ADNI1_Complete_2Yr_3T_5_05_2025.csv'),
        str(ADNI_DIR / 'ADNI1_Complete_3Yr_3T_5_05_2025.csv'),
    ]
    
    BASE_OUTPUT_DIR = str(ADNI_DIR)
    TARGET_TRANSPOSE_SHAPE = (160, 240, 256)
    TARGET_PAD_SHAPE = (256, 256)
    NUM_SLICES_TO_EXTRACT = 80
    RANDOM_STATE = 45
    
    # --- EXECUTION ---
    
    # Step 0: Find Files
    main_df = find_and_shape_files(DATASET_ROOT_DIR)
    if main_df.empty:
        print(" No files found. Exiting.")
        return

    # Step 1: Transpose
    main_df = transpose_files(main_df, TARGET_TRANSPOSE_SHAPE)

    # Step 2: Pad
    main_df = pad_files(main_df, TARGET_PAD_SHAPE)

    # Step 3: Split (Patient Level)
    train_df, val_df, test_df, main_df = split_data(
        main_df, 
        ADNI_METADATA_CSVS, 
        test_size=0.4, 
        val_size=0.5, 
        random_state=RANDOM_STATE
    )

    # Step 4: Extract Slices
    split_dfs = {'train': train_df, 'val': val_df, 'test': test_df}
    preprocess_and_save_slices(split_dfs, BASE_OUTPUT_DIR, NUM_SLICES_TO_EXTRACT)

    # Final Wrap-up
    final_csv = os.path.join(BASE_OUTPUT_DIR, 'adni_full_processed.csv')
    main_df['Shape'] = main_df['Shape'].astype(str)
    main_df.to_csv(final_csv, index=False)
    print(f" Master CSV saved at: {final_csv}")
    
    print(f"\n{'#'*60}")
    print(f" Processing finished successfully")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    main()