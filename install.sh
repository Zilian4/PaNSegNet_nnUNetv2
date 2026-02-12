#!/bin/bash
# Installation script for PaNSegNet_nnUNetv2
# This script automates the installation process

set -e  # Exit on error

echo "=========================================="
echo "PaNSegNet_nnUNetv2 Installation Script"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "ERROR: Python >= 3.10 is required. Found: $python_version"
    exit 1
fi

echo "Python version: $(python3 --version)"

# Check if virtual environment should be created
read -p "Create a virtual environment? (recommended) [y/n]: " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "Virtual environment activated!"
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Install nnUNetv2 package
echo "Installing nnUNetv2 package..."
cd nnUNet
pip install -e .
cd ..

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Set up environment variables (see INSTALL.md)"
echo "2. Verify installation: nnUNetv2_plan_and_preprocess --help"
echo ""
echo "If you created a virtual environment, activate it with:"
echo "  source venv/bin/activate"

