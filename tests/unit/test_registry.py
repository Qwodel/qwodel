"""
Unit Tests for Backend Registry

Tests backend registration and retrieval.
"""

import pytest
from qwodel.backends import BackendRegistry, get_backend
from qwodel.core.base import BaseQuantizer
from qwodel.core.exceptions import BackendNotFoundError


class TestBackendRegistry:
    """Test BackendRegistry functionality."""
    
    def test_list_backends(self):
        """Test listing registered backends."""
        backends = BackendRegistry.list_backends()
        assert isinstance(backends, list)
        # At minimum, should have gguf, awq, coreml
        assert len(backends) >= 3
        assert "gguf" in backends
        assert "awq" in backends
        assert "coreml" in backends
    
    def test_get_backend(self):
        """Test getting registered backends."""
        # Test GGUF backend
        gguf_backend = BackendRegistry.get("gguf")
        assert issubclass(gguf_backend, BaseQuantizer)
        
        # Test AWQ backend
        awq_backend = BackendRegistry.get("awq")
        assert issubclass(awq_backend, BaseQuantizer)
        
        # Test CoreML backend
        coreml_backend = BackendRegistry.get("coreml")
        assert issubclass(coreml_backend, BaseQuantizer)
    
    def test_get_backend_case_insensitive(self):
        """Test backend retrieval is case-insensitive."""
        assert BackendRegistry.get("GGUF") == BackendRegistry.get("gguf")
        assert BackendRegistry.get("Awq") == BackendRegistry.get("awq")
    
    def test_get_nonexistent_backend(self):
        """Test getting non-existent backend raises error."""
        with pytest.raises(BackendNotFoundError) as exc_info:
            BackendRegistry.get("nonexistent")
        assert "not found" in str(exc_info.value).lower()
    
    def test_is_registered(self):
        """Test checking if backend is registered."""
        assert BackendRegistry.is_registered("gguf")
        assert BackendRegistry.is_registered("awq")
        assert BackendRegistry.is_registered("coreml")
        assert not BackendRegistry.is_registered("nonexistent")
    
    def test_get_backend_helper(self):
        """Test get_backend convenience function."""
        backend = get_backend("gguf")
        assert issubclass(backend, BaseQuantizer)
