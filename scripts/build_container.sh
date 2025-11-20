#!/bin/bash

# Automates the container deployment workflow by building the Docker image and immediately
# launching a detached, interactive container configured with specific GPU device access,
# adjustable shared memory size, and a volume mount that maps the current directory to
# the container's workspace for live development.

# Default image name if not provided
IMAGE_NAME=${1:-alzheimer-mri}
DEVICE=${2:-0}
SHM_SIZE=${3:-8}

# Build the Docker image
echo "Building Docker image as '$IMAGE_NAME'..."
docker build -t "$IMAGE_NAME" .

# Run the container
echo "Running container from image '$IMAGE_NAME'..."
docker run -itd --rm \
  --name "${IMAGE_NAME}-container" \
  --gpus device="$DEVICE" \
  -v "$(pwd)":/workspace/ \
  --shm-size="${SHM_SIZE}gb" \
  "$IMAGE_NAME"