# Qwodel - Production-Grade Model Quantization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Qwodel** is a production-ready Python package for model quantization across multiple backends (AWQ, GGUF, CoreML). It provides a unified, intuitive API for quantizing large language models with minimal code.

## Features

- **Unified API** - Simple interface across all quantization backends
- **Multiple Backends** - AWQ (GPU), GGUF (CPU), CoreML (Apple devices)
- **Optional Dependencies** - Install only what you need
- **CLI & Python API** - Use via command line or programmatically
- **Type Safe** - Full type hints and mypy validation
- **Well Documented** - Comprehensive docs with examples

## Quick Start

### Installation

### Quick Install (All Backends)

```bash
pip install qwodel[all]
```

This installs **all backends** (GGUF, AWQ, CoreML) with PyTorch 2.1.2 (CPU version).

### GPU Support (for AWQ only)

If you need **GPU quantization with AWQ**, install PyTorch with CUDA first:

```bash
# 1. Install PyTorch with CUDA 12.1
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 2. Install qwodel
pip install qwodel[all]
```

> **Note**: GGUF and CoreML work perfectly fine with CPU-only PyTorch!

### Individual Backends

```bash
# GGUF only (CPU quantization - most popular!)
pip install qwodel[gguf]

# AWQ only (GPU quantization)
pip install qwodel[awq]

# CoreML only (Apple devices)
pip install qwodel[coreml]
```

### Local Development

```bash
# Clone and install locally
cd /path/to/qwodel
pip install -e .[all]
```

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
print(f"Quantized model saved to: {output_path}")
```

### CLI

```bash
# Quantize a model
qwodel quantize \
    --backend gguf \
    --format Q4_K_M \
    --model meta-llama/Llama-2-7b-hf \
    --output ./quantized

# List available formats
qwodel list-formats --backend gguf
```

## Supported Backends

### GGUF (CPU Quantization)
- **Use Case**: CPU inference, broad compatibility
- **Formats**: Q4_K_M, Q8_0, Q2_K, Q5_K_M, and more
- **Best For**: Most users, CPU-based deployment

### AWQ (GPU Quantization)
- **Use Case**: NVIDIA GPU inference
- **Formats**: INT4
- **Best For**: GPU deployments, maximum speed
- **Requires**: CUDA 12.1+

### CoreML (Apple Devices)
- **Use Case**: iOS, macOS, iPadOS deployment
- **Formats**: FLOAT16, INT8, INT4
- **Best For**: Apple device deployment

## Examples

### Batch Processing

```python
from qwodel import quantize

models = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"]

for model in models:
    quantize(
        model_path=model,
        backend="gguf",
        format="Q4_K_M",
        output_dir="./quantized"
    )
```

### Custom Progress Callback

```python
from qwodel import Quantizer

def progress_handler(progress: int, stage: str, message: str):
    print(f"[{progress}%] {stage}: {message}")

quantizer = Quantizer(
    backend="gguf",
    model_path="./my-model",
    output_dir="./output",
    progress_callback=progress_handler
)

quantizer.quantize(format="Q4_K_M")
```

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [CLI Reference](docs/CLI_REFERENCE.md)
- [API Reference](docs/API_REFERENCE.md)
- [CLI Reference](docs/CLI_REFERENCE.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Qwodel builds upon the excellent work of:
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for GGUF quantization
- [llm-compressor](https://github.com/vllm-project/llm-compressor) for AWQ quantization
- [CoreMLTools](https://github.com/apple/coremltools) for CoreML conversion
