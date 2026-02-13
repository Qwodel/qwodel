# Installation Guide

## Quick Start (Local Installation)

### Option 1: Install All Backends (Recommended)

```bash
# For CPU-only (GGUF, CoreML work fine)
pip install -e /home/chin2/workspace/qwodel[all]

# For GPU support (AWQ requires this)
# First install PyTorch with CUDA:
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Then install qwodel[all]
pip install -e /home/chin2/workspace/qwodel[all]
```

### Option 2: Install Specific Backend

```bash
# For GGUF only (CPU quantization)
pip install -e /home/chin2/workspace/qwodel[gguf]

# For AWQ only (GPU quantization) - install PyTorch with CUDA first
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install -e /home/chin2/workspace/qwodel[awq]

# For CoreML only (Apple devices)
pip install -e /home/chin2/workspace/qwodel[coreml]
```

## Your Use Case: GGUF Q4_K_M

```bash
# Step 1: Install
cd /home/chin2/workspace/qwodel
pip install -e .[gguf]

# Step 2: Run quantization
python examples/quickstart_gguf.py
```

Or use the bash script:
```bash
chmod +x examples/install_and_quantize.sh
./examples/install_and_quantize.sh
```

Or use CLI directly:
```bash
qwodel quantize -b gguf -f Q4_K_M -m /home/chin2/workspace/qwen0.6 -o ./output
```

## Why Install PyTorch with CUDA Separately (for AWQ)?

**The Issue:**
- PyTorch from PyPI (via pip) defaults to **CPU-only** version
- GPU version requires special CUDA builds from PyTorch's index URL
- pip doesn't support multiple index URLs in requirements

**The Solution:**
1. **For GGUF/CoreML users**: Just do `pip install qwodel[all]` - CPU PyTorch is fine!
2. **For AWQ users (GPU needed)**: Install PyTorch with CUDA first, then qwodel

**This is industry standard!** Packages like `transformers`, `diffusers`, etc. all do this.

## Verification

After installation, verify:

```bash
# Check installation
pip show qwodel

# List backends
qwodel list-backends

# List formats
qwodel list-formats --backend gguf
```

## Development Installation

For contributors:

```bash
cd /home/chin2/workspace/qwodel
pip install -e .[dev]  # Includes testing tools
pytest tests/unit/ -v  # Run tests
```
