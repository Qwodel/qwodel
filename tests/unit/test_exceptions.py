"""
Unit Tests for Core Exceptions

Tests exception hierarchy and usage.
"""

import pytest
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    ConversionError,
    DependencyError,
    ConfigurationError,
    BackendNotFoundError,
    FormatNotSupportedError,
)


class TestExceptionHierarchy:
    """Test exception inheritance."""
    
    def test_base_exception(self):
        """Test QuantizationError is base exception."""
        error = QuantizationError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"
    
    def test_validation_error(self):
        """Test ValidationError inherits from QuantizationError."""
        error = ValidationError("validation failed")
        assert isinstance(error, QuantizationError)
        assert isinstance(error, Exception)
    
    def test_conversion_error(self):
        """Test ConversionError inherits from QuantizationError."""
        error = ConversionError("conversion failed")
        assert isinstance(error, QuantizationError)
    
    def test_dependency_error(self):
        """Test DependencyError inherits from QuantizationError."""
        error = DependencyError("missing dependency")
        assert isinstance(error, QuantizationError)
    
    def test_configuration_error(self):
        """Test ConfigurationError inherits from QuantizationError."""
        error = ConfigurationError("invalid config")
        assert isinstance(error, QuantizationError)
    
    def test_backend_not_found_error(self):
        """Test BackendNotFoundError inherits from QuantizationError."""
        error = BackendNotFoundError("backend not found")
        assert isinstance(error, QuantizationError)
    
    def test_format_not_supported_error(self):
        """Test FormatNotSupportedError inherits from QuantizationError."""
        error = FormatNotSupportedError("format not supported")
        assert isinstance(error, QuantizationError)


class TestExceptionUsage:
    """Test exception raising and catching."""
    
    def test_raise_and_catch(self):
        """Test raising and catching exceptions."""
        with pytest.raises(QuantizationError):
            raise QuantizationError("test")
        
        with pytest.raises(ValidationError):
            raise ValidationError("test")
    
    def test_catch_base_exception(self):
        """Test catching specific error as base QuantizationError."""
        with pytest.raises(QuantizationError):
            raise ValidationError("test")
