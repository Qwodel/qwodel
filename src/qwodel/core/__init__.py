"""
Qwodel Core Package

Provides core abstractions, constants, and utilities.
"""

from qwodel.core.constants import (
    QuantizationBackend,
    ModelFormat,
    GGUFFormat,
    AWQFormat,
    CoreMLFormat,
    FILE_EXTENSION_MAP,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMP_DIR,
    GGUF_FORMAT_DESCRIPTIONS,
    COREML_FORMAT_DESCRIPTIONS,
)

from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    ConversionError,
    DependencyError,
    ConfigurationError,
    BackendNotFoundError,
    FormatNotSupportedError,
)

__all__ = [
    # Constants
    "QuantizationBackend",
    "ModelFormat",
    "GGUFFormat",
    "AWQFormat",
    "CoreMLFormat",
    "FILE_EXTENSION_MAP",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_TEMP_DIR",
    "GGUF_FORMAT_DESCRIPTIONS",
    "COREML_FORMAT_DESCRIPTIONS",
    # Exceptions
    "QuantizationError",
    "ValidationError",
    "ConversionError",
    "DependencyError",
    "ConfigurationError",
    "BackendNotFoundError",
    "FormatNotSupportedError",
]
