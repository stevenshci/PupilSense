#!/bin/bash

# Define the URL of the zip file
URL="https://storage.googleapis.com/stevenshci-public/PupilSense/models.zip"

echo "Downloading model..."
# Use curl to download the zip file
curl -o models.zip $URL

echo "Extracting files..."
# Unzip the file in the present working directory
unzip models.zip

echo "Cleanup downloaded zip file..."
# Remove the zip file after extraction
rm models.zip

python3 -m pip install pyyaml==5.1

# Copy example.env to .env if it doesn't already exist
if [ ! -f .env ]; then
    cp example.env .env
    echo "Created .env file. Please configure it with your specific settings."
else
    echo ".env file already exists. Please ensure it has the correct settings."
fi

# Clone Detectron2 repository
git clone https://github.com/facebookresearch/detectron2

# Add detectron2 to Python path
PYTHON_PATH=$(python3 -c "import os; print(os.path.abspath('./detectron2'))")
echo "export PYTHONPATH=\$PYTHONPATH:$PYTHON_PATH" >> ~/.bashrc

# Apply changes to current session
export PYTHONPATH=$PYTHONPATH:$PYTHON_PATH

echo "Setup completed successfully."

