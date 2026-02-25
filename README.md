# Qwodel

[![PyPI](https://img.shields.io/pypi/v/qwodel.svg)](https://pypi.org/project/Qwodel/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Community](https://img.shields.io/badge/License-Qwodel%20Community%20v1.0-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-docs.qwodel.com-blue.svg)](https://docs.qwodel.com/docs)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/qwodel?period=total&units=INTERNATIONAL_SYSTEM&left_color=BLACK&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/qwodel)

Production-grade model quantization SDK for AWQ, GGUF, and CoreML.

## Installation

```bash
# Pick the backend you need
pip install qwodel[gguf]    # CPU quantization
pip install qwodel[awq]     # GPU quantization (requires CUDA)
pip install qwodel[coreml]  # Apple devices
pip install qwodel[all]     # Everything
```

## Quick Example

```python
from qwodel import Quantizer

output = Quantizer(
    backend="gguf",
    model_path="meta-llama/Llama-2-7b-hf",
    output_dir="./quantized"
).quantize(format="Q4_K_M")

print(f"Saved to: {output}")
```

## Documentation

Full documentation — guides, API reference, CLI reference, and examples — is at:

**[docs.qwodel.com/docs](https://docs.qwodel.com/docs)**

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Qwodel Community License v1.0 — see [LICENSE](LICENSE).

## Acknowledgments

Built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp), [llm-compressor](https://github.com/vllm-project/llm-compressor), and [CoreMLTools](https://github.com/apple/coremltools).
