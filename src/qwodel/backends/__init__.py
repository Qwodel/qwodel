"""
Qwodel Backends Package

Quantization backend implementations.
"""

from qwodel.backends._registry import BackendRegistry, get_backend

# Eagerly import every backend so their module-level
# BackendRegistry.register() calls are executed.
# Each import is wrapped so that a missing optional dependency
# (e.g. torch/CUDA for AWQ, coremltools for CoreML) does not
# prevent the entire package from loading — the backend simply
# won't appear in BackendRegistry.list_backends() if its extras
# are not installed.

try:
    import qwodel.backends.gguf  # noqa: F401
except Exception:
    pass

try:
    import qwodel.backends.awq  # noqa: F401
except Exception:
    pass

try:
    import qwodel.backends.coreml  # noqa: F401
except Exception:
    pass

__all__ = [
    "BackendRegistry",
    "get_backend",
]
