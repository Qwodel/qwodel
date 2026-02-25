# Contributing to Qwodel

Thank you for your interest in contributing to Qwodel — a production-grade model quantization SDK supporting AWQ, GGUF, and CoreML backends. This guide will walk you through everything you need to set up, build, and submit a high-quality contribution.

**Repository:** [github.com/Qwodel/qwodel](https://github.com/Qwodel/qwodel)

---

## Your First Contribution — Quick Checklist

New here? Follow these steps in order:

- [ ] **1.** [Fork](https://github.com/Qwodel/qwodel/fork) the repository on GitHub
- [ ] **2.** Clone your fork and add the upstream remote (see [§2](#2-branching-workflow))
- [ ] **3.** Install Python 3.10+ and run `pip install -e ".[dev]"` (see [§1](#1-prerequisites--environment-setup))
- [ ] **4.** Run `pre-commit install` to set up automatic linting on commit
- [ ] **5.** Create a branch following the naming convention (e.g. `feat/your-feature`)
- [ ] **6.** Read [§3 — Project Architecture](#3-project-architecture) before touching any files
- [ ] **7.** Make your changes and add tests in `tests/unit/`
- [ ] **8.** Run `ruff check src/ tests/` and `pytest tests/unit/` — both must pass
- [ ] **9.** Commit using Conventional Commits format (e.g. `feat(gguf): add Q8 support`)
- [ ] **10.** Open a Pull Request against **`develop`** at [Qwodel/qwodel](https://github.com/Qwodel/qwodel/pulls)

---

## Table of Contents

1. [Prerequisites & Environment Setup](#1-prerequisites--environment-setup)
2. [Branching Workflow](#2-branching-workflow)
3. [Project Architecture](#3-project-architecture)
4. [Coding Standards & Linting](#4-coding-standards--linting)
5. [Testing Protocols](#5-testing-protocols)
6. [Commit & PR Conventions](#6-commit--pr-conventions)
7. [Reporting Issues](#7-reporting-issues)

---

## 1. Prerequisites & Environment Setup

### System Requirements

- **Python 3.10, 3.11, or 3.12** (see `pyproject.toml`)
- `git` for version control
- A CUDA-capable GPU is required only if you are working on the **AWQ backend**
- macOS with Apple Silicon is required only if you are working on the **CoreML backend**

### Setting Up Your Environment

```bash
# 1. Clone your fork
git clone https://github.com/<your-username>/qwodel.git
cd qwodel

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install the base package in editable mode + dev tools
pip install -e ".[dev]"
```

### Backend-Specific Extras

Qwodel uses optional dependencies to keep the base install lightweight. Install only the backend(s) you plan to work on:

| Backend | Install command | Requires |
|---------|----------------|----------|
| AWQ | `pip install -e ".[awq]"` | CUDA GPU, PyTorch with CUDA |
| GGUF | `pip install -e ".[gguf]"` | CPU only |
| CoreML | `pip install -e ".[coreml]"` | macOS + Apple Silicon |
| All backends | `pip install -e ".[all]"` | All of the above |
| Docs | `pip install -e ".[docs]"` | None |

### Install Pre-commit Hooks


```bash
pre-commit install
```

This will automatically run `ruff` and other checks on every `git commit`, preventing style violations from reaching the PR stage.
a
---

## 2. Branching Workflow

Qwodel follows the **Fork and Pull Request** model. Direct pushes to `develop` or `main` are not permitted. **All PRs must target the `develop` branch** — never `main` directly.

### Step 1 — Fork

Click **Fork** on [github.com/Qwodel/qwodel](https://github.com/Qwodel/qwodel/fork) to create your own copy.

### Step 2 — Clone & Link Upstream

```bash
git clone https://github.com/<your-username>/qwodel.git
cd qwodel
git remote add upstream https://github.com/Qwodel/qwodel.git
```

### Step 3 — Sync with Upstream Before Starting Work

```bash
git fetch upstream
git checkout main
git merge upstream/main
```

### Step 4 — Create a Branch

Use the following naming conventions depending on your change type:

| Type | Example |
|------|---------|
| New feature | `feat/add-exllama2-backend` |
| Bug fix | `fix/awq-ignore-map-default` |
| Documentation | `docs/update-gguf-guide` |
| Chore / tooling | `chore/update-ruff-config` |
| Refactor | `refactor/core-base-class` |

```bash
git checkout -b feat/your-feature-name
```

---

## 3. Project Architecture

Understanding qwodel's layout is essential before making changes.

```
qwodel/
├── src/
│   └── qwodel/
│       ├── __init__.py          # Public API surface; re-exports key symbols
│       ├── api.py               # High-level Python API (QwodelAPI class)
│       ├── backends/
│       │   ├── _registry.py     # Backend auto-discovery and registration
│       │   ├── awq/             # AWQ quantization backend (GPU / llmcompressor)
│       │   ├── gguf/            # GGUF quantization backend (llama.cpp)
│       │   └── coreml/          # CoreML export backend (Apple devices)
│       ├── cli/
│       │   └── main.py          # Click-based CLI entrypoint (`qwodel` command)
│       └── core/
│           ├── base.py          # Abstract base class all backends inherit from
│           ├── constants.py     # All shared string constants, defaults, ignore maps
│           └── exceptions.py    # Custom exception hierarchy
├── tests/
│   ├── conftest.py              # Shared pytest fixtures
│   ├── unit/                    # Fast, isolated unit tests (no GPU/network)
│   └── integration/             # End-to-end tests requiring real models/hardware
├── docs/                        # MkDocs source files
├── pyproject.toml               # All project, tool, and dependency configuration
└── CONTRIBUTING.md              # This file
```

### Key Design Rules

- **Add constants to `core/constants.py`**, never hardcode strings inside backends.
- **Raise only custom exceptions** from `core/exceptions.py`. Do not raise raw `ValueError` or `RuntimeError` from within any backend.
- **All backends must subclass `core/base.py`** and implement its abstract interface. New backends must also register themselves in `backends/_registry.py`.
- **The CLI in `cli/main.py` calls `api.py`**, which in turn delegates to the registered backend. Do not import backend modules directly from the CLI.

---

## 4. Coding Standards & Linting

All code must pass `ruff` (linting + formatting) and `mypy` (static type checking) before a PR can be merged.

### Ruff (Linter & Formatter)

Configuration is in `pyproject.toml` under `[tool.ruff]`. Line length is **100 characters**.

```bash
# Check for lint errors
ruff check src/ tests/

# Auto-fix fixable issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

Enabled rule sets: `E`, `W` (pycodestyle), `F` (pyflakes), `I` (isort), `B` (bugbear), `C4` (comprehensions), `UP` (pyupgrade).

### Mypy (Type Checking)

```bash
mypy src/qwodel/
```

- All **new** functions and methods must include full type annotations on arguments and return values.
- The `disallow_untyped_defs` option is currently off — but aim for full annotation coverage in any file you touch.

### Docstrings

Every public function, class, and module must have a docstring. Use Google-style docstrings:

```python
def quantize(model_path: str, output_dir: str, bits: int = 4) -> None:
    """Quantize a model using the configured backend.

    Args:
        model_path: Path or HuggingFace Hub ID of the source model.
        output_dir: Directory where the quantized model will be saved.
        bits: Target bit-width for quantization. Defaults to 4.

    Raises:
        QwodelBackendError: If the backend fails to initialize.
        QwodelModelError: If the model architecture is unsupported.
    """
```

---

## 5. Testing Protocols

### Running the Test Suite

```bash
# Run all unit tests (fast, no GPU required)
pytest tests/unit/

# Run with coverage report (as configured in pyproject.toml)
pytest tests/unit/ --cov=qwodel --cov-report=term-missing

# Run integration tests (requires appropriate hardware/credentials)
pytest tests/integration/ -m integration

# Skip slow and GPU-dependent tests in CI-like environments
pytest tests/ -m "not slow and not gpu"
```

### Pytest Markers

Markers are defined in `pyproject.toml` under `[tool.pytest.ini_options]`:

| Marker | Meaning |
|--------|---------|
| `@pytest.mark.slow` | Test takes a long time (e.g. downloads a real model) |
| `@pytest.mark.gpu` | Test requires a CUDA device |
| `@pytest.mark.integration` | Full end-to-end test |

Apply markers to your tests accordingly:

```python
import pytest

@pytest.mark.gpu
@pytest.mark.slow
def test_awq_quantization_end_to_end():
    ...
```

### Writing Tests

- **Every new feature** must include at least one unit test in `tests/unit/`.
- **Every bug fix** must include a regression test that fails on the unfixed code and passes after your fix.
- Unit tests must not make network calls or require GPU hardware — mock them using `pytest` fixtures defined in `conftest.py`.
- Integration tests go in `tests/integration/` and should be marked with `@pytest.mark.integration`.

---

## 6. Commit & PR Conventions

### Conventional Commits

Commit messages must follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary>
```

**Type** must be one of:

| Type | When to use |
|------|-------------|
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation changes only |
| `refactor` | Code change that is neither a fix nor a feature |
| `test` | Adding or updating tests |
| `chore` | Build process, tooling, or dependency changes |
| `perf` | Performance improvements |

**Scope** (optional) refers to the part of qwodel being changed, e.g. `awq`, `gguf`, `coreml`, `cli`, `core`.

**Examples:**

```
feat(gguf): add support for Q8_0 quantization format
fix(awq): correct default ignore map for MistralForCausalLM
docs(cli): update qwodel convert --help description
chore: bump ruff to 0.4.0
```

### Submitting a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feat/your-feature-name
   ```
2. Open a Pull Request against the **`develop`** branch of [Qwodel/qwodel](https://github.com/Qwodel/qwodel/compare). PRs targeting `main` directly will be closed without review.
3. Fill out the PR template completely, including:
   - A link to the related GitHub issue (use `Closes #<issue-number>`)
   - A summary of **what** changed and **why**
   - Which backend(s) are affected
   - Test evidence (output of `pytest`, screenshots if CLI behaviour changed)
4. Ensure all CI checks pass (ruff, mypy, pytest) before requesting a review.
5. Address reviewer feedback promptly and push follow-up commits to the same branch.

---

## 7. Reporting Issues

If you find a bug or want to request a feature, [open an issue](https://github.com/Qwodel/qwodel/issues) and include:

- **For bugs**: qwodel version (`qwodel --version`), Python version, OS, the backend used (AWQ/GGUF/CoreML), the full traceback, and a minimal reproduction snippet.
- **For feature requests**: the use case you are trying to solve and any prior art or references.

---

## 8. Contributor License Agreement

Qwodel is distributed under the **Qwodel Community License v1.0**. By submitting a pull request, you agree that your contributions will be licensed under the same Qwodel Community License v1.0. 

You confirm that you have the right to submit the work and that it does not violate any third-party intellectual property rights.

---

Thank you for helping make Qwodel better!
