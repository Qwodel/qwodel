"""
Qwodel - Production-Grade Model Quantization

A unified Python package for model quantization across AWQ, GGUF, and CoreML backends.
"""

# NOTE: Import order matters to avoid circular dependencies
from qwodel.core.constants import (
    QuantizationBackend,
    ModelFormat,
    GGUFFormat,
    AWQFormat,
    CoreMLFormat,
)

from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    ConversionError,
    BackendNotFoundError,
    FormatNotSupportedError,
)

# Conditional imports - only import if modules exist
try:
    from qwodel.__version__ import version as __version__
except ImportError:
    __version__ = "0.1.0.dev0"

# Main API
from qwodel.api import Quantizer, quantize

__all__ = [
    "__version__",
    # Main API
    "Quantizer",
    "quantize",
    # Constants
    "QuantizationBackend",
    "ModelFormat",
    "GGUFFormat",
    "AWQFormat",
    "CoreMLFormat",
    # Exceptions
    "QuantizationError",
    "ValidationError",
    "DependencyError",
    "ConversionError",
    "BackendNotFoundError",
    "FormatNotSupportedError",
]
