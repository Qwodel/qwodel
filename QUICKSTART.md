# Quick Start - GGUF Quantization

## Your Specific Use Case

### Model Path
`/home/chin2/workspace/qwen0.6`

### Installation & Usage

```bash
# 1. Install qwodel with GGUF backend
cd /home/chin2/workspace/qwodel
pip install -e .[gguf]

# 2. Quantize using CLI
qwodel quantize -b gguf -f Q4_K_M -m /home/chin2/workspace/qwen0.6 -o ./output

# Or use Python:
python examples/quickstart_gguf.py
```

### Alternative: Install ALL backends at once

```bash
# This installs GGUF, AWQ, and CoreML support
pip install -e .[all]

# PyTorch 2.1.2 will be installed automatically (CPU version)
# For GPU support with AWQ, install PyTorch with CUDA first:
# pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

## Why PyTorch with CUDA is separate

**Short answer**: Yes, PyTorch IS in the `[awq]` extras, but pip installs the **CPU version** by default.

**For GPU users only**: Install PyTorch with CUDA first using PyTorch's special index URL, then install qwodel.

**For GGUF users (you)**: Just do `pip install -e .[gguf]` or `pip install -e .[all]` - you don't need GPU!

## Verification

```bash
# Check if qwodel is installed
pip show qwodel

# Test the CLI
qwodel list-backends
qwodel list-formats --backend gguf
```

Expected output:
```
Available Backends:
┏━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Backend ┃ Status      ┃
┡━━━━━━━━━╇━━━━━━━━━━━━━┩
│ gguf    │ ✓ Available │
│ awq     │ ✓ Available │
│ coreml  │ ✓ Available │
└─────────┴─────────────┘
```
