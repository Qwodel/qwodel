# Troubleshooting Guide

## Installation Issues

### Error: "build backend is missing the 'build_editable' hook"

**Issue:** You see this error when running `pip install -e .[gguf]`:
> Project file:///.../qwodel has a 'pyproject.toml' and its build backend is missing the 'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it cannot be installed in editable mode.

**Fix:** This happens with older `pip` or `setuptools` versions. We have added a minimal `setup.py` to fix this.

1. Ensure you have the latest code (with `setup.py`).
2. Upgrade your build tools:
   ```bash
   pip install --upgrade pip setuptools wheel
   ```
3. Try installing again:
   ```bash
   pip install -e .
   ```

### Error: "llama-quantize not found"

**Issue:** Running GGUF quantization fails with `llama-quantize not found`.

**Fix:**
This means `llama.cpp` tools are not in your PATH. If you installed via pip, the python binding is installed, but the C++ binary tools might be missing or not in PATH.

**Workaround:**
Ensure `llama-quantize` is compiled and available in your system path.

### Error: "CUDA not available" (for AWQ)

**Issue:** AWQ quantization fails or uses CPU fallback (which is slow/impossible).

**Fix:**
Ensure you installed PyTorch with CUDA support:
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

## Runtime Issues

### Initial Import Error

If you see import errors like `ModuleNotFoundError: No module named 'qwodel'`, ensure you installed in editable mode:
```bash
pip install -e .
```

### OOM (Out of Memory)

If you run out of memory during quantization:
1. Reduce batch size via smart config hack (AWQ handles this automatically)
2. Use a smaller model or format (e.g., Q4_K_M instead of Q8_0)
3. Close other GPU-intensive applications
