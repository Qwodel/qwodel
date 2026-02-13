"""
GGUF Backend Implementation

Provides GGUF quantization using llama.cpp tools.
Supports all GGUF formats (Q4_K_M, Q8_0, Q2_K, etc.) in a unified implementation.
"""

from typing import Dict, List, Optional
from pathlib import Path
import subprocess
import tempfile
import shutil

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import ModelFormat, GGUFFormat, GGUF_FORMAT_DESCRIPTIONS
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)


class GGUFQuantizer(BaseQuantizer):
    """
    GGUF quantization backend using llama.cpp tools.
    
    Supports conversion from HuggingFace models and quantization
    to various GGUF formats (Q4_K_M, Q8_0, Q2_K, etc.).
    
    Examples:
        >>> quantizer = GGUFQuantizer(
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     output_dir="./output"
        ... )
        >>> output = quantizer.quantize(format="Q4_K_M")
    """
    
    # Unsupported architectures for llama.cpp
    _unsupported_architectures = [
        "XLMRobertaForTokenClassification",
        "BertForTokenClassification",
        "RobertaForTokenClassification",
        "DistilBertForTokenClassification",
        "MobileBertForTokenClassification",
    ]
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None
    ):
        """
        Initialize GGUF quantizer.
        
        Args:
            model_path: Path to source model (HuggingFace directory or GGUF file)
            output_dir: Output directory for quantized models
            progress_callback: Optional callback(progress: int, stage: str, message: str)
        """
        self._temp_gguf_path = None
        self._current_format = None
        super().__init__(model_path, output_dir, progress_callback)
    
    @property
    def unsupported_architectures(self) -> List[str]:
        """List of architectures unsupported by GGUF/llama.cpp."""
        return self._unsupported_architectures
    
    @classmethod
    def get_backend_name(cls) -> str:
        """Get backend name."""
        return "gguf"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        """Get supported input formats."""
        return [ModelFormat.GGUF, ModelFormat.HUGGINGFACE]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        """
        List available GGUF quantization formats.
        
        Returns:
            Dictionary mapping format names to descriptions
        """
        return {fmt.value: GGUF_FORMAT_DESCRIPTIONS[fmt] for fmt in GGUFFormat}
    
    def _validate_backend_compatibility(self) -> None:
        """Validate input format compatibility."""
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(
                f"GGUF backend does not support {self.input_format.value} format. "
                f"Supported formats: {[f.value for f in self.get_supported_input_formats()]}"
            )
        
        # Additional validation for HuggingFace models
        if self.input_format == ModelFormat.HUGGINGFACE:
            if self.model_path.is_dir():
                model_files = (
                    list(self.model_path.glob('*.bin')) +
                    list(self.model_path.glob('*.safetensors'))
                )
                if not model_files and not (self.model_path / 'pytorch_model.bin').exists():
                    raise ValidationError(
                        f"No model files found in HuggingFace directory: {self.model_path}"
                    )
    
    def _check_dependencies(self) -> None:
        """Check if llama.cpp tools are available."""
        try:
            # Check llama-quantize
            subprocess.run(
                ["llama-quantize", "--help"],
                capture_output=True,
                timeout=5
            )
        except FileNotFoundError:
            raise DependencyError(
                "llama-quantize not found. Please install llama.cpp tools.\n"
                "See: https://github.com/ggerganov/llama.cpp"
            )
        except subprocess.TimeoutExpired:
            # Tool exists but help command timed out - that's okay
            pass
        
        # Check conversion tool if needed
        if self.input_format == ModelFormat.HUGGINGFACE:
            try:
                subprocess.run(
                    ["python", "-c", "import gguf"],
                    capture_output=True,
                    check=True,
                    timeout=5
                )
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise DependencyError(
                    "gguf Python package not found. Install with: pip install qwodel[gguf]"
                )
    
    def _convert_to_gguf(self) -> Path:
        """
        Convert HuggingFace model to GGUF format.
        
        Returns:
            Path to converted GGUF file
        """
        if self.input_format == ModelFormat.GGUF:
            return self.model_path
        
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        gguf_path = temp_dir / f"{model_name}.gguf"
        
        self.logger.info(f"Converting HuggingFace model to GGUF format...")
        self.logger.info(f"Input: {self.model_path}")
        self.logger.info(f"Output: {gguf_path}")
        
        self._report_progress(20, "converting", "Converting to GGUF format")
        
        try:
            # Use convert_hf_to_gguf.py from llama.cpp
            cmd = [
                "python", "/usr/local/bin/convert_hf_to_gguf.py",
                str(self.model_path),
                "--outtype", "f16",
                "--outfile", str(gguf_path)
            ]
            
            self.logger.info(f"Conversion command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800  # 30 minute timeout
            )
            
            self.logger.info("Model conversion completed successfully")
            
            # Verify output exists
            if not gguf_path.exists():
                raise QuantizationError("Conversion completed but output file not found")
            
            self._temp_gguf_path = gguf_path
            return gguf_path
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Model conversion failed: {e.stderr}"
            self.logger.error(error_msg)
            # Clean up
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise QuantizationError(error_msg)
        except subprocess.TimeoutExpired:
            self.logger.error("Model conversion timed out")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise QuantizationError("Model conversion timed out after 30 minutes")
    
    def _run_quantization(self) -> None:
        """Execute GGUF quantization process."""
        # Convert to GGUF if needed
        self._report_progress(15, "converting", "Preparing model for quantization")
        gguf_path = self._convert_to_gguf()
        output_path = self.get_output_path()
        
        # Build quantization command
        self._report_progress(40, "preparing", "Preparing quantization")
        cmd = [
            "llama-quantize",
            str(gguf_path),
            str(output_path),
            self._current_format
        ]
        
        self.logger.info(f"Starting quantization with format: {self._current_format}")
        self.logger.info(f"Command: {' '.join(cmd)}")
        
        try:
            # Run quantization
            self._report_progress(50, "quantizing", f"Running {self._current_format} quantization")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=3600  # 1 hour timeout
            )
            
            self.logger.info("Quantization completed successfully")
            self.logger.info(f"Output saved to: {output_path}")
            
            # Verify output exists
            if not output_path.exists():
                raise QuantizationError("Quantization completed but output file not found")
            
            # Report file size
            size_mb = output_path.stat().st_size / (1024 * 1024)
            self._report_progress(95, "saving", f"Quantized model size: {size_mb:.2f} MB")
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Quantization failed: {e.stderr}"
            self.logger.error(error_msg)
            raise QuantizationError(error_msg)
        except subprocess.TimeoutExpired:
            self.logger.error("Quantization timed out")
            raise QuantizationError("Quantization timed out after 1 hour")
        finally:
            # Clean up temporary GGUF file
            if self._temp_gguf_path and self._temp_gguf_path.exists():
                temp_dir = self._temp_gguf_path.parent
                shutil.rmtree(temp_dir)
                self._temp_gguf_path = None
    
    def quantize(self, format: str, **kwargs) -> Path:
        """
        Quantize model to specified GGUF format.
        
        Args:
            format: GGUF format name (e.g., "Q4_K_M", "Q8_0")
            **kwargs: Additional arguments (unused for GGUF)
            
        Returns:
            Path to quantized model
            
        Raises:
            FormatNotSupportedError: If format is not supported
            QuantizationError: If quantization fails
        """
        # Validate format
        format_upper = format.upper()
        try:
            gguf_format = GGUFFormat[format_upper]
        except KeyError:
            available = [f.value for f in GGUFFormat]
            raise FormatNotSupportedError(
                f"Format '{format}' not supported by GGUF backend. "
                f"Available formats: {available}"
            )
        
        # Store format for use in quantization
        self._current_format = gguf_format.value
        
        # Call parent quantize which orchestrates the process
        return super().quantize(format=format, **kwargs)
    
    def get_output_path(self) -> Path:
        """Get output path for quantized model."""
        if not self._current_format:
            raise QuantizationError("Format not set. Call quantize() first.")
        
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        format_lower = self._current_format.lower().replace('_', '_')
        return self.output_dir / f"{model_name}-{format_lower}.gguf"


# Register GGUF backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("gguf", GGUFQuantizer)
