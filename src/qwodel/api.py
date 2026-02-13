"""
Qwodel Main API

User-facing API for quantization operations.
"""

from typing import Optional, Union, Callable, Dict
from pathlib import Path

from qwodel.core.constants import QuantizationBackend
from qwodel.backends import get_backend


class Quantizer:
    """
    Main quantization interface.
    
    This class provides a unified API for quantizing models across
    different backends (AWQ, GGUF, CoreML).
    
    Examples:
        >>> # GGUF quantization
        >>> quantizer = Quantizer(
        ...     backend="gguf",
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     output_dir="./output"
        ... )
        >>> output = quantizer.quantize(format="Q4_K_M")
        
        >>> # AWQ quantization (requires GPU)
        >>> quantizer = Quantizer(
        ...     backend="awq",
        ...     model_path="./my-model",
        ...     output_dir="./output"
        ... )
        >>> output = quantizer.quantize(format="int4")
        
        >>> # With progress callback
        >>> def progress(pct, stage, msg):
        ...     print(f"[{pct}%] {stage}: {msg}")
        >>> quantizer = Quantizer(
        ...     backend="gguf",
        ...     model_path="./model",
        ...     output_dir="./output",
        ...     progress_callback=progress
        ... )
        >>> output = quantizer.quantize(format="Q4_K_M")
    """
    
    def __init__(
        self,
        backend: Union[str, QuantizationBackend],
        model_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[int, str, str], None]] = None,
        **backend_kwargs
    ):
        """
        Initialize quantizer with specified backend.
        
        Args:
            backend: Backend name ("gguf", "awq", "coreml") or QuantizationBackend enum
            model_path: Path to source model
            output_dir: Output directory for quantized model
            progress_callback: Optional callback(progress: int, stage: str, message: str)
            **backend_kwargs: Backend-specific keyword arguments
        """
        # Normalize backend name
        if isinstance(backend, QuantizationBackend):
            self.backend_name = backend.value
        else:
            self.backend_name = str(backend).lower()
        
        # Get backend class
        backend_class = get_backend(self.backend_name)
        
        # Create backend instance
        self.backend_impl = backend_class(
            model_path=model_path,
            output_dir=output_dir,
            progress_callback=progress_callback,
            **backend_kwargs
        )
    
    def quantize(self, format: str, **format_kwargs) -> Path:
        """
        Execute quantization with specified format.
        
        Args:
            format: Quantization format (e.g., "Q4_K_M", "int4", "float16")
            **format_kwargs: Format-specific keyword arguments
            
        Returns:
            Path to quantized model
            
        Raises:
            QuantizationError: If quantization fails
            FormatNotSupportedError: If format is not supported
        """
        return self.backend_impl.quantize(format=format, **format_kwargs)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the quantized model.
        
        Returns:
            Dictionary with model metadata
        """
        return self.backend_impl.get_model_info()
    
    @staticmethod
    def list_formats(backend: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """
        List available quantization formats.
        
        Args:
            backend: Specific backend name, or None for all backends
            
        Returns:
            Dictionary mapping backend names to format dictionaries
            
        Examples:
            >>> # List formats for GGUF
            >>> Quantizer.list_formats("gguf")
            {'gguf': {'Q4_K_M': 'Best balance...', 'Q8_0': 'High Quality...'}}
            
            >>> # List all formats
            >>> Quantizer.list_formats()
            {'gguf': {...}, 'awq': {...}, 'coreml': {...}}
        """
        from qwodel.backends import BackendRegistry
        
        if backend:
            backend_class = get_backend(backend)
            return {backend: backend_class.list_formats()}
        
        # Return all backends
        result = {}
        for backend_name in BackendRegistry.list_backends():
            try:
                backend_class = get_backend(backend_name)
                result[backend_name] = backend_class.list_formats()
            except Exception:
                # Skip backends that fail to load
                continue
        
        return result
    
    @staticmethod
    def list_backends() -> list:
        """
        List all available backends.
        
        Returns:
            List of backend names
        """
        from qwodel.backends import BackendRegistry
        return BackendRegistry.list_backends()


def quantize(
    model_path: str,
    output_dir: str,
    backend: str,
    format: str,
    progress_callback: Optional[Callable[[int, str, str], None]] = None,
    **kwargs
) -> Path:
    """
    Convenience function for one-shot quantization.
    
    Args:
        model_path: Path to source model
        output_dir: Output directory
        backend: Backend name ("gguf", "awq", "coreml")
        format: Quantization format
        progress_callback: Optional progress callback
        **kwargs: Additional backend/format-specific arguments
        
    Returns:
        Path to quantized model
        
    Examples:
        >>> # Simple GGUF quantization
        >>> output = quantize(
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     output_dir="./output",
        ...     backend="gguf",
        ...     format="Q4_K_M"
        ... )
        
        >>> # AWQ quantization with progress
        >>> def progress(pct, stage, msg):
        ...     print(f"[{pct}%] {stage}")
        >>> output = quantize(
        ...     model_path="./model",
        ...     output_dir="./output",
        ...     backend="awq",
        ...     format="int4",
        ...     progress_callback=progress
        ... )
    """
    quantizer = Quantizer(
        backend=backend,
        model_path=model_path,
        output_dir=output_dir,
        progress_callback=progress_callback,
        **kwargs
    )
    return quantizer.quantize(format=format)
