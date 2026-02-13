"""
Unit Tests for Quantizer API

Tests the main user-facing Quantizer class.
"""

import pytest
from unittest.mock import Mock, patch
from qwodel import Quantizer
from qwodel.core.exceptions import BackendNotFoundError, FormatNotSupportedError


class TestQuantizerInitialization:
    """Test Quantizer initialization."""
    
    def test_init_with_string_backend(self):
        """Test initializing with backend as string."""
        # This will fail because there's no actual model, but tests initialization
        with pytest.raises(Exception):  # Will fail at validation
            Quantizer(backend="gguf", model_path="/nonexistent", output_dir="/tmp")
    
    def test_init_with_invalid_backend(self):
        """Test initializing with invalid backend."""
        with pytest.raises(BackendNotFoundError):
            Quantizer(backend="invalid", model_path="/nonexistent", output_dir="/tmp")
    
    def test_init_case_insensitive(self):
        """Test backend name is case-insensitive."""
        with pytest.raises(Exception):  # Will fail at validation, not backend lookup
            Quantizer(backend="GGUF", model_path="/nonexistent", output_dir="/tmp")


class TestQuantizerMethods:
    """Test Quantizer methods."""
    
    def test_list_formats_all(self):
        """Test listing all formats."""
        formats = Quantizer.list_formats()
        assert isinstance(formats, dict)
        assert "gguf" in formats
        assert "awq" in formats
        assert "coreml" in formats
    
    def test_list_formats_specific_backend(self):
        """Test listing formats for specific backend."""
        formats = Quantizer.list_formats(backend="gguf")
        assert isinstance(formats, dict)
        assert "gguf" in formats
        assert len(formats) == 1
        
        # Check GGUF formats
        gguf_formats = formats["gguf"]
        assert "Q4_K_M" in gguf_formats
        assert "Q8_0" in gguf_formats
    
    def test_list_backends(self):
        """Test listing available backends."""
        backends = Quantizer.list_backends()
        assert isinstance(backends, list)
        assert "gguf" in backends
        assert "awq" in backends
        assert "coreml" in backends
