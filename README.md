<div align="center">

# 🧠 Deep Learning for Alzheimer’s Diagnosis via Brain Neuroimaging

<!-- 📄 **Report**: [Insert Link to Paper/Report]   -->
💻 **Code**: This repository  
✍️ **Author**: Manuel Mora Cartagena  
🏆 **Goal**: Robust multiclass classification of Alzheimer's Disease brain neuroimaging data

<!-- ✨ **Focus**: CNNs, ResNets, Vision Transformers and Hybrid Resnet & Vision Transformers   -->

</div>

---

## 🚀 Introduction

<div align="justify">

Early diagnosis of Alzheimer's Disease is crucial for effective patient care. This repository hosts a **comprehensive Deep Learning pipeline** designed to classify Magnetic Resonance Imaging (MRI) slices into distinct stages of dementia.

This project explores the performance trade-offs between **Convolutional Neural Networks (CNNs)**, **Residual Neural Networks (ResNets)**, **Vision Transformers (ViTs)** and **Hybrid architectures such as Hybrid Residual Vision Transformers (HybridRViT)** applied to brain neuroimaging data. This repository provides a fully reproducible training pipeline equipped with automated reporting and Weights & Biases integration.

</div>


### 🔑 Key Features:
- **Multi-Architecture Support**: Seamless switching between custom CNNs, ResNet, ViT and HybridRViT.
- **Robust Evaluation**: Supports Stratified K-Fold Cross-Validation and Simple Splits (Train/Val/Test).
- **Advanced Regularization**: Implementations of DropBlock, Dropout, Batch Normalization and early stopping among others.
- **Dual Data Support**: Handles both standard image formats (`.jpg`, `.png`) and pre-processed Numpy arrays (`.npy`) for medical imaging depth.
- **Insightful Metrics**: Auto-generation of weighted and macro metrics (F1, Precision, Recall) and Confusion Matrices.

---

## 📂 Repository Structure

```plaintext
alzheimer-mri/

│── ⚙️ configs/           # Configuration for training and models
│── 🗂️ data/              # Dataset directory (Images or .npy files)
│── 📈 logs/              # Logs directory for wandb runs
│── 🛠️ scripts/           # Scripts for environment setup and dataset preprocessing
│── 📂 src/               # Core implementation
    ├── 📥 data/          # Dataloader for both datasets
│   ├── 🧠 models/        # Architecture definitions (CNNs, ResNet, ViT and HybridRViT)
│   ├── 🧩 utils/         # Config loader, metrics, model factory
│   ├── 📊 eval.py        # Evaluation script
│   ├── 🐍 main.py        # Main entry point for the pipeline
│   └── 🧪 train.py       # Training script
│── 🐳 Dockerfile         # Docker setup
│── ⚖️ LICENSE            # License information
│── 📘 README.md          # Project documentation
└── 📦 requirements.txt   # Python dependencies

```
---

## 🔧 Requirements

* **Python:** 3.8+
* **PyTorch:** >= 2.0
* **Libraries:** NumPy, Pillow, etc.
* **Logging:** Wandb

---

## 📦 Installation



### 🐳 Docker
You can set up a container by running any of the following:

**Using make target:**
```bash
make container
```

**Using the script directly:**
```bash
bash scripts/build_container.sh [IMAGE_NAME] [GPU_DEVICE] [SHM_SIZE_GB]
```

This will build and run a GPU Docker container with custom settings, with all necessary libraries pre-installed.

#### 📄 Arguments

| Argument | Description | Default | Required |
| :--- | :--- | :--- | :--- |
| `IMAGE_NAME` | Name to assign to the Docker image | `alzheimer-mri` | No |
| `GPU_DEVICE` | GPU ID to assign to the container | `0` | No |
| `SHM_SIZE_GB` | Shared memory size in GB | `8` | No |


### 🐍 Virtual Environment
You can create a Python virtual environment (venv) and install the necessary dependencies using any of the following:

**Using make target:**
```bash
make venv
```

**Using the script directly:**
```bash
bash scripts/build_venv.sh
```

This will create a Python virtual environment with PyTorch CUDA 12.4 and all project dependencies installed.

---

## 🧬 Models

This project implements and compares **5 distinct deep learning architectures** for MRI classification:

| Model | Description |
| :--- | :--- |
| **Convolutional Neural Network 1 (CNN_1)** | Base classic CNN architecture. |
| **Convolutional Neural Network 2 (CNN_2)** | Classic CNN architecture with 1 additional convolutional layer added at the initial stage. |
| **Convolutional Neural Network 3 (CNN_3)** | Classic CNN architecture with 2 additional convolutional layers added at the initial stage. |
| **Residual Network (ResNet)** | ResNet backbone with custom classification head. |
| **Vision Transformer (ViT)** | State-of-the-art ViT architecture for image classification. |
| **Hybrid architectures (HybridRViT)** | State-of-the-art Hybrid ResNet & ViT architecture for image classification. |

All models are configured to handle grayscale MRI images with adaptive input preprocessing. The system is fully automated and optimized for multiclass classification across dementia stages, dynamically adjusting to 3 or 4 classes depending on the dataset.

---

## 📊 Datasets

### ADNI Dataset
The **Alzheimer's Disease Neuroimaging Initiative (ADNI)** is a longitudinal neuroimaging study. This dataset focuses on clinical diagnosis stages:
- **Access**: Private repository (requires credentials)
- **Content**: 3T MRI scans with multiple timepoints (1-year, 2-year, 3-year follow-ups)
- **Classes (3)**: `CN` (Cognitively Normal), `MCI` (Mild Cognitive Impairment), `AD` (Alzheimer's Disease)
- **Patient Identifiers**: Unique patient IDs enable a proper separation of samples into training subsets, ensuring the robustness and reliability of the model.
- **Preprocessing**: Data available in Nifti format (`.nii`) converted to preprocessed slices
- **Usage**: High-quality clinical data with standardized acquisition protocols

### Kaggle Dataset
**Augmented Alzheimer MRI Dataset**
- **Source**: [Kaggle - Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset)
- **Content**: Augmented MRI brain scans categorized by dementia severity
- **Classes (4)**: `Non Demented`, `Very Mild Demented`, `Mild Demented`, `Moderate Demented`
- **Size**: Comprehensive collection of 2D slices from 3D MRI volumes
- **Formats**: Supports standard image formats (`.png`, `.jpg`) for flexible preprocessing

### Data Split Strategy
- **Train Set**: 60% of available data
- **Validation Set**: 20% of available data
- **Test Set**: 20% of available data
- **Cross-Validation**: Optional K-Fold stratified validation for robust evaluation

---

## 💻 Usage

### 📂 Data Preprocessing

Before training, the raw dataset must be preprocessed using the scripts in `src/tools/`:

**For Kaggle Dataset:**
1. Place the original images in `data/kaggle/augmented/` organized by class folders
2. Run the preprocessing script:
```bash
python3 scripts/prepare_data_kaggle.py
```
This will generate standardized images, all of the same size, in `data/kaggle/resized`.

**For ADNI Dataset:**
1. Place the raw ADNI data as provided from the original repository in `data/adni/ADNI/`
2. Ensure CSV files with patient IDs are present in `data/adni/` (e.g., `ADNI1_Complete_1Yr_3T_5_05_2025.csv`)
3. Run the preprocessing script:
```bash
python3 scripts/prepare_data_adni.py
```
This will convert Nifti files (`.nii`) to preprocessed slices and create train/val/test splits.

### 🧪 Basic Training
```bash
python3 src/main.py
```

### ⚙️ Configuration

**Training configuration** and **hyperparameters** can be customized in `configs/config.yaml`:
- Learning rate, batch size, epochs, optimizer, loss function, dataset, model selection, cross-validation.

**Model Architecture & Regularization** can be adjusted in individual model config files:
- `cnn_1_config.yaml`, `cnn_2_config.yaml`, `cnn_3_config.yaml`
- `resnet_config.yaml`
- `vit_config.yaml`
- `hybrid_rvit_config.yaml`

**Available Regularization Techniques:**
- Dropout, DropBlock, Batch Normalization, Early Stopping, Weight Decay, etc.

---

## 📈 Results & Monitoring

All training runs are logged to **Weights & Biases (WandB)** for:
- Real-time metric tracking
- Confusion matrices visualization
- Cross-validation aggregation
- Model performance comparison

---

## 🤝 Acknowledgements
This project was developed as part of the Master's in Artificial Intelligence and Big Data Analytics. Special thanks to the ADNI for making their dataset available for scientific purposes.

## 📝 License

This project is part of a Master's thesis in **Artificial Intelligence and Big Data Analytics**. 
See LICENSE file for details.


