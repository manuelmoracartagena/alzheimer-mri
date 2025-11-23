#!/usr/bin/env bash

# Initializes the local development environment by creating a Python virtual environment
# in the parent directory, upgrading pip, and forcibly installing PyTorch with CUDA 12.4
# support alongside the project dependencies from requirements.txt, ensuring the process
# halts immediately if any step fails.

# Exit immediately if any command fails
set -e

echo "=== Environment setup started ==="

# Set virtual environment directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
VENV_DIR="$SCRIPT_DIR/../venv"

echo "-> Creating Python virtual environment"
python3 -m venv "$VENV_DIR"

echo "-> Activating virtual environment"
source "$VENV_DIR/bin/activate"

echo "-> Upgrading pip"
python3 -m pip install --upgrade pip

echo "-> Installing PyTorch with CUDA 12.4"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "-> Installing dependencies from requirements.txt"
pip install -r "$SCRIPT_DIR/../requirements.txt"

source "$VENV_DIR/bin/activate"

echo "=== Environment setup complete ==="