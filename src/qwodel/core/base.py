"""
Qwodel Base Quantizer

Abstract base class for all quantization backends.
Provides common functionality including validation, logging, and orchestration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import logging

from qwodel.core.constants import ModelFormat, FILE_EXTENSION_MAP
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
)


def setup_logger(name: str) -> logging.Logger:
    """Setup logger for a quantizer class."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def detect_model_format(model_path: Path) -> ModelFormat:
    """
    Detect the format of a model based on its file extension or directory structure.
    
    Args:
        model_path: Path to model file or directory
        
    Returns:
        Detected ModelFormat
    """
    if not model_path.exists():
        return ModelFormat.UNKNOWN
    
    if model_path.is_file():
        suffix = model_path.suffix.lower()
        return FILE_EXTENSION_MAP.get(suffix, ModelFormat.UNKNOWN)
    elif model_path.is_dir():
        if (model_path / 'config.json').exists():
            return ModelFormat.HUGGINGFACE
        elif (model_path / 'saved_model.pb').exists():
            return ModelFormat.TENSORFLOW
    
    return ModelFormat.UNKNOWN


class BaseQuantizer(ABC):
    """
    Abstract base class for all quantizers.
    
    Provides common functionality:
    - Input validation
    - File management
    - Format detection
    - Logging
    - Error handling
    - Progress callbacks
    
    Subclasses must implement backend-specific quantization logic.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional[Callable[[int, str, str], None]] = None
    ):
        """
        Initialize the base quantizer.
        
        Args:
            model_path: Path to the source model file or directory
            output_dir: Directory to save quantized models
            progress_callback: Optional callback(progress: int, stage: str, message: str)
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.progress_callback = progress_callback
        self.logger = setup_logger(self.__class__.__name__)
        
        # Detect input format
        self.input_format = detect_model_format(self.model_path)
        
        # Validate inputs
        self._validate_inputs()
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _report_progress(self, progress: int, stage: str, message: str = "") -> None:
        """Report progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(progress, stage, message)
    
    def _validate_inputs(self) -> None:
        """
        Validate input parameters and paths.
        
        Raises:
            ValidationError: If validation fails
        """
        if not self.model_path.exists():
            raise ValidationError(f"Model path not found: {self.model_path}")
        
        if self.input_format == ModelFormat.UNKNOWN:
            raise ValidationError(
                f"Unknown model format. Path: {self.model_path}"
            )
        
        # Backend-specific validation
        self._validate_backend_compatibility()
        
        # Architecture validation
        self._validate_architecture()
    
    def _get_model_architecture(self) -> Optional[str]:
        """
        Get the model architecture from config.json if available.
        
        Returns:
            Architecture string (e.g. 'LlamaForCausalLM') or None
        """
        if self.input_format in (ModelFormat.HUGGINGFACE, ModelFormat.UNKNOWN):
            config_path = self.model_path
            if self.model_path.is_dir():
                config_path = self.model_path / "config.json"
            
            if config_path.exists() and config_path.is_file() and config_path.name == "config.json":
                try:
                    import json
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        if "architectures" in config and isinstance(config["architectures"], list):
                            if len(config["architectures"]) > 0:
                                return config["architectures"][0]
                except Exception as e:
                    self.logger.warning(f"Failed to parse config.json: {e}")
        
        return None
    
    @property
    def unsupported_architectures(self) -> List[str]:
        """
        List of architectures that are known to be unsupported.
        To be overridden by subclasses.
        """
        return []
    
    def _validate_architecture(self) -> None:
        """
        Validate that the model architecture is supported.
        
        Raises:
            ValidationError: If architecture is unsupported
        """
        arch = self._get_model_architecture()
        if arch:
            self.logger.info(f"Detected model architecture: {arch}")
            
            for unsupported in self.unsupported_architectures:
                if unsupported.endswith("*"):
                    prefix = unsupported[:-1]
                    if arch.startswith(prefix):
                        raise ValidationError(
                            f"Model architecture '{arch}' is not supported by "
                            f"{self.get_backend_name()} backend. "
                            f"(Matches unsupported pattern: {unsupported})"
                        )
                elif arch == unsupported:
                    raise ValidationError(
                        f"Model architecture '{arch}' is not supported by "
                        f"{self.get_backend_name()} backend."
                    )
    
    @abstractmethod
    def _validate_backend_compatibility(self) -> None:
        """
        Validate that the input format is compatible with this backend.
        
        Raises:
            ValidationError: If format is incompatible
        """
        pass
    
    @abstractmethod
    def _check_dependencies(self) -> None:
        """
        Check if required tools and libraries are available.
        
        Raises:
            DependencyError: If dependencies are missing
        """
        pass
    
    @abstractmethod
    def _run_quantization(self) -> None:
        """
        Execute the backend-specific quantization process.
        
        Raises:
            QuantizationError: If quantization fails
        """
        pass
    
    @abstractmethod
    def get_output_path(self) -> Path:
        """
        Get the output path for the quantized model.
        
        Returns:
            Path object for the output file
        """
        pass
    
    def quantize(self, format: str, **kwargs) -> Path:
        """
        Perform the complete quantization process.
        
        This is the main entry point that orchestrates:
        1. Dependency checking
        2. Format conversion (if needed)
        3. Quantization
        4. Validation
        
        Args:
            format: Quantization format string
            **kwargs: Format-specific arguments
        
        Returns:
            Path to the quantized model file
            
        Raises:
            QuantizationError: If any step fails
        """
        self.logger.info(f"Starting quantization with {self.__class__.__name__}")
        self.logger.info(f"Input: {self.model_path}")
        self.logger.info(f"Format: {format}")
        
        self._report_progress(0, "initializing", "Starting quantization")
        
        try:
            # Check dependencies
            self._report_progress(5, "checking_dependencies", "Validating dependencies")
            self._check_dependencies()
            
            # Run quantization (backend-specific)
            self._report_progress(10, "quantizing", "Running quantization")
            self._run_quantization()
            
            # Verify output
            output_path = self.get_output_path()
            if not output_path.exists():
                raise QuantizationError(
                    "Quantization completed but output file not found"
                )
            
            self._report_progress(100, "complete", "Quantization successful")
            self.logger.info(f"✅ Quantization successful: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"❌ Quantization failed: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the quantized model.
        
        Returns:
            Dictionary containing model metadata
        """
        output_path = self.get_output_path()
        return {
            "source_model": str(self.model_path),
            "quantized_model": str(output_path),
            "backend": self.get_backend_name(),
            "file_size": output_path.stat().st_size if output_path.exists() else None,
            "input_format": self.input_format.value,
        }
    
    @classmethod
    @abstractmethod
    def get_backend_name(cls) -> str:
        """
        Get the name of this quantization backend.
        
        Returns:
            Backend name string
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        """
        Get list of input formats supported by this backend.
        
        Returns:
            List of ModelFormat enum values
        """
        pass
    
    @classmethod
    @abstractmethod
    def list_formats(cls) -> Dict[str, str]:
        """
        List available quantization formats for this backend.
        
        Returns:
            Dictionary mapping format names to descriptions
        """
        pass
