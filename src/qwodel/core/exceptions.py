"""
Qwodel Core Exceptions

Custom exception hierarchy for the qwodel package.
Migrated from bekenstein-core/core/utils/exceptions.py
"""


class QuantizationError(Exception):
    """Base exception for quantization errors"""
    pass


class ValidationError(QuantizationError):
    """Raised when validation fails (e.g., invalid input format, unsupported architecture)"""
    pass


class ConversionError(QuantizationError):
    """Raised when model conversion fails"""
    pass


class DependencyError(QuantizationError):
    """Raised when required dependencies are missing or incompatible"""
    pass


class ConfigurationError(QuantizationError):
    """Raised when configuration is invalid"""
    pass


class BackendNotFoundError(QuantizationError):
    """Raised when a requested backend is not available"""
    pass


class FormatNotSupportedError(QuantizationError):
    """Raised when a quantization format is not supported by the backend"""
    pass
