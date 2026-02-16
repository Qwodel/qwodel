"""
Backend Registry

Manages registration and discovery of quantization backends.
"""

from typing import Dict, Type, List
from qwodel.core.base import BaseQuantizer
from qwodel.core.exceptions import BackendNotFoundError


class BackendRegistry:
    """
    Registry for quantization backends.
    
    Manages registration and retrieval of backend implementations.
    """
    
    _backends: Dict[str, Type[BaseQuantizer]] = {}
    
    @classmethod
    def register(cls, name: str, backend_class: Type[BaseQuantizer]) -> None:
        """
        Register a backend.
        
        Args:
            name: Backend name (e.g., "gguf", "awq", "coreml")
            backend_class: Backend quantizer class
        """
        cls._backends[name.lower()] = backend_class
    
    @classmethod
    def get(cls, name: str) -> Type[BaseQuantizer]:
        """
        Get a backend by name.
        
        Args:
            name: Backend name
            
        Returns:
            Backend quantizer class
            
        Raises:
            BackendNotFoundError: If backend not found
        """
        name_lower = name.lower()
        
        # Try to import backend if not registered
        if name_lower not in cls._backends:
            try:
                import importlib
                importlib.import_module(f"qwodel.backends.{name_lower}")
            except ImportError:
                # If import fails, we'll raise BackendNotFoundError below
                pass

        if name_lower not in cls._backends:
            available = list(cls._backends.keys())
            raise BackendNotFoundError(
                f"Backend '{name}' not found. Available: {available}"
            )
        return cls._backends[name_lower]
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """
        List all registered backends.
        
        Returns:
            List of backend names
        """
        return list(cls._backends.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if a backend is registered.
        
        Args:
            name: Backend name
            
        Returns:
            True if registered, False otherwise
        """
        return name.lower() in cls._backends


def get_backend(name: str) -> Type[BaseQuantizer]:
    """
    Convenience function to get a backend.
    
    Args:
        name: Backend name
        
    Returns:
        Backend quantizer class
    """
    return BackendRegistry.get(name)
