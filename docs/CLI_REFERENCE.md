# CLI Reference

This document describes the Command Line Interface (CLI) tools provided by `qwodel`.

## `qwodel` CLI

The package provides a main CLI entry point for quantization tasks.

### Installation

Ensure the package is installed with the necessary extras:

```bash
# For CoreML support
pip install qwodel[coreml]

# For GGUF support
pip install qwodel[gguf]

# For AWQ support
pip install qwodel[awq]
```

### Usage

```bash
qwodel [OPTIONS] COMMAND [ARGS]...
```

### Commands

#### `quantize`

Quantize a model.

```bash
qwodel quantize [OPTIONS] MODEL_PATH
```

**Arguments:**

| Argument | Description | Required |
| --- | --- | --- |
| `MODEL_PATH` | Path to the input model (file or directory). | Yes |

**Options:**

| Option | Description | Default |
| --- | --- | --- |
| `--output-dir`, `-o` | Directory to save the output. | `./quantized_models` |
| `--backend`, `-b` | Backend to use (`coreml`, `gguf`, `awq`). | Auto-detected if possible, otherwise required. |
| `--format`, `-f` | Quantization format (e.g., `int8_linear`, `Q4_K_M`). | Backend default |
| `--verbose`, `-v` | Enable verbose logging. | False |

**Examples:**

Quantize to CoreML Int8:
```bash
qwodel quantize ./my-model -b coreml -f int8_linear -o ./output
```

Quantize to GGUF Q4_K_M:
```bash
qwodel quantize ./my-model -b gguf -f Q4_K_M
```

#### `list-formats`

List available quantization formats for a backend.

```bash
qwodel list-formats [BACKEND]
```

**Examples:**

```bash
qwodel list-formats coreml
```

#### `check`

Verify installation and dependencies.

```bash
qwodel check
```

---

## Example Scripts

The `examples/` directory contains helper scripts for specific use cases.

### `examples/coreml_quantization.py`

A dedicated script for CoreML quantization.

**Usage:**

```bash
python examples/coreml_quantization.py \
    --model_path <path_to_model> \
    --output_dir <output_directory> \
    --format <format>
```

**Arguments:**
- `--model_path`: Input model path.
- `--output_dir`: Output directory (default: `./quantized_models`).
- `--format`: Format string (default: `int8_linear`). Supported: `float16`, `int8_linear`, `int8_symmetric`, `int4`, `int6`.
