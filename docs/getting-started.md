# Getting Started with Qwodel

This guide will help you get started with qwodel for model quantization.

## Installation

### Basic Installation

```bash
pip install qwodel
```

### With Specific Backend

```bash
# For GGUF (CPU quantization)
pip install qwodel[gguf]

# For AWQ (GPU quantization)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install qwodel[awq]

# For CoreML (Apple devices)
pip install qwodel[coreml]
```

## Quick Start

### Python API

```python
from qwodel import Quantizer

# Create quantizer
quantizer = Quantizer(
    backend="gguf",
    model_path="meta-llama/Llama-2-7b-hf",
    output_dir="./quantized"
)

# Quantize model
output_path = quantizer.quantize(format="Q4_K_M")
print(f"Quantized model: {output_path}")
```

### Command Line

```bash
# Quantize a model
qwodel quantize \
    --backend gguf \
    --format Q4_K_M \
    --model meta-llama/Llama-2-7b-hf \
    --output ./quantized

# List available formats
qwodel list-formats

# List available backends
qwodel list-backends
```

## Choosing a Backend

### GGUF - CPU Quantization
**Best for**: Most users, CPU-based deployment, broad compatibility

```python
quantizer = Quantizer(backend="gguf", model_path="./model", output_dir="./output")
quantizer.quantize(format="Q4_K_M")  # Recommended format
```

**Recommended formats**:
- `Q4_K_M`: Best balance (90% of users)
- `Q8_0`: High quality, larger size
- `Q2_K`: Maximum compression

### AWQ - GPU Quantization
**Best for**: NVIDIA GPU deployments, maximum speed

```python
quantizer = Quantizer(backend="awq", model_path="./model", output_dir="./output")
quantizer.quantize(format="int4")
```

**Requirements**: CUDA 12.1+, NVIDIA GPU

### CoreML - Apple Devices
**Best for**: iOS, macOS, iPadOS deployment

```python
quantizer = Quantizer(backend="coreml", model_path="./model", output_dir="./output")
quantizer.quantize(format="float16")  # Recommended for best compatibility
```

**Recommended formats**:
- `float16`: Universal compatibility
- `int4`: Maximum compression (iOS 18+)

## Progress Tracking

```python
def progress_handler(percent, stage, message):
    print(f"[{percent}%] {stage}: {message}")

quantizer = Quantizer(
    backend="gguf",
    model_path="./model",
    output_dir="./output",
    progress_callback=progress_handler
)

quantizer.quantize(format="Q4_K_M")
```

## Next Steps

- See [examples/](../examples/) for more usage examples
- Check [API Reference](api-reference.md) for detailed documentation
- Read [Backend Guides](backends/) for backend-specific information
