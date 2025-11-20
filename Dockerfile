FROM nvcr.io/nvidia/pytorch:23.09-py3

# Set the working directory
WORKDIR /workspace

# Copy the requirements file (notice the dot '.' at the end is mandatory)
COPY requirements.txt .

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application
COPY . .