# API Reference

This document provides detailed reference documentation for the `qwodel` Python API.

## Core API

### `Quantizer`

The main entry point for quantization.

```python
from qwodel import Quantizer

quantizer = Quantizer(
    backend="coreml",
    model_path="./my-model",
    output_dir="./output"
)
```

#### Initialization

```python
def __init__(
    self,
    backend: str,
    model_path: str,
    output_dir: str = "./quantized_models",
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    **kwargs
)
```

| Parameter | Type | Description |
|Str | --- | --- |
| `backend` | `str` | The quantization backend to use. Supported: `"gguf"`, `"awq"`, `"coreml"`. |
| `model_path` | `str` | Path to the source model. Can be a local directory, a file (e.g., `.gguf`), or a HuggingFace model ID (e.g., `meta-llama/Llama-2-7b`). |
| `output_dir` | `str` | Directory where the quantized model will be saved. Defaults to `./quantized_models`. |
| `progress_callback` | `Callable` | Optional callback function for progress updates. Signature: `(percent: int, stage: str, message: str)`. |
| `**kwargs` | `Any` | Backend-specific initialization arguments (see Backend Specifics). |

#### Methods

##### `quantize`

Runs the quantization process.

```python
def quantize(self, format: str, **kwargs) -> Path
```

| Parameter | Type | Description |
| --- | --- | --- |
| `format` | `str` | The specific quantization format to use (e.g., `"int8_linear"`, `"Q4_K_M"`). |
| `**kwargs` | `Any` | Backend-specific quantization arguments. |

**Returns**: `Path` object pointing to the generated quantized model file (or directory for CoreML).

##### `get_model_info`

Retrieves metadata about the quantized model.

```python
def get_model_info(self) -> Dict[str, Any]
```

**Returns**: A dictionary containing:
- `source_model`: Path to input model.
- `quantized_model`: Path to output model.
- `backend`: Name of the backend used.
- `file_size`: Size of the output file in bytes.
- `input_format`: Detected format of the source model.

---

## Backends

### CoreML Backend (`backend="coreml"`)

Quantizes models for Apple platforms (iOS, macOS).

**Supported Formats**:
- `float16`: Half-precision (FP16).
- `int8_linear`: 8-bit linear quantization.
- `int8_symmetric`: 8-bit symmetric quantization.
- `int4`: 4-bit quantization (palettization).
- `int6`: 6-bit quantization (palettization).

**Initialization Arguments**:
- `compute_units` (str): `"ALL"` (default), `"CPU_ONLY"`, `"CPU_AND_GPU"`.
- `input_shape` (tuple): Shape for tracing, default `(1, 512)`.
- `seq_length` (int): Max sequence length, default `512`.

### GGUF Backend (`backend="gguf"`)

Quantizes models for llama.cpp compatible runners.

**Supported Formats**:
- `Q4_K_M` (Received): Balanced quality/size.
- `Q8_0`: High quality.
- `Q2_K`: Max compression.
- `Q3_K_M`, `Q4_0`, `Q5_K_M`, `Q6_K`, etc.

**Initialization Arguments**:
None specific.

### AWQ Backend (`backend="awq"`)

Quantizes models for GPU inference using AutoAWQ/llm-compressor.

**Supported Formats**:
- `int4`: 4-bit AWQ quantization.

**Initialization Arguments**:
- `calibration_dataset` (str): HuggingFace dataset ID (default: `"mit-han-lab/pile-val-backup"`).
- `calibration_split` (str): Dataset split (default: `"validation"`).
- `batch_size` (int): Manual batch size (overrides auto-config).
- `seq_length` (int): Manual sequence length (overrides auto-config).
- `num_samples` (int): Manual number of calibration samples (overrides auto-config).

---

## Logging

`qwodel` uses the standard Python `logging` module. You can configure logging verbosity and format in your application:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

The logger name is `qwodel`.
