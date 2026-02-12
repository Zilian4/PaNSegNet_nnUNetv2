# Installation Guide for PaNSegNet_nnUNetv2

This guide will help you install all required packages for PaNSegNet_nnUNetv2.

## Prerequisites

- Python >= 3.10
- pip (Python package manager)
- CUDA-capable GPU (recommended for training/inference)

## Installation Methods

### Method 1: Install with conda (Recommended)

```bash
# Create a conda environment
conda create -n PaNSegNet python=3.10
conda activate PaNSegNet

# Install PyTorch with CUDA (adjust CUDA version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install nnUNetv2 package
cd nnUNet
pip install -e .
```

### Method 2: Install from requirements.txt 

```bash
# Navigate to the project directory
cd PaNSegNet_nnUNetv2

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install nnUNetv2 package in editable mode
cd nnUNet
pip install -e .
```

### Method 3: Install using pip from pyproject.toml

```bash
# Navigate to the nnUNet directory
cd PaNSegNet_nnUNetv2/nnUNet

# Install the package (this will install all dependencies automatically)
pip install -e .
```


## Verify Installation

After installation, verify that nnUNetv2 is properly installed:

```bash
# Check if nnUNetv2 commands are available
nnUNetv2_plan_and_preprocess --help

# Or test Python import
python -c "import nnunetv2; print('nnUNetv2 installed successfully!')"
```

## Setting Up Environment Variables

nnUNetv2 requires environment variables to be set. Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

Then reload your shell:
```bash
source ~/.bashrc  # or source ~/.zshrc
```

## Troubleshooting

### CUDA/GPU Issues
- Make sure you have the correct PyTorch version with CUDA support
- Check CUDA compatibility: `python -c "import torch; print(torch.cuda.is_available())"`

### Package Conflicts
- Use a fresh virtual environment to avoid conflicts
- If issues persist, try installing packages one by one from requirements.txt

### Missing Dependencies
- Some packages may require system libraries (e.g., graphviz, SimpleITK)
- On Ubuntu/Debian: `sudo apt-get install graphviz`
- On macOS: `brew install graphviz`

## Additional Notes

- The `-e` flag installs the package in "editable" mode, so changes to the code are immediately reflected
- For production use, you may want to install without `-e` flag: `pip install .`
- Check the [nnUNet documentation](nnUNet/documentation/installation_instructions.md) for more details

