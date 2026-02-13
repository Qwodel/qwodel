"""
CoreML Backend Implementation

Provides CoreML quantization for Apple devices (iOS, macOS, iPadOS).
Supports FLOAT16, INT8, INT4, and INT6 quantization formats.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import torch

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import ModelFormat, CoreMLFormat, COREML_FORMAT_DESCRIPTIONS
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# Try to import CoreML dependencies
_COREML_AVAILABLE = False
_COREML_IMPORT_ERROR = None

try:
    import coremltools as ct
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OpPalettizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
        palettize_weights
    )
    from transformers import AutoModel, AutoConfig
    _COREML_AVAILABLE = True
except ImportError as e:
    _COREML_IMPORT_ERROR = str(e)


class CoreMLQuantizer(BaseQuantizer):
    """
    CoreML quantization backend for Apple devices.
    
    Supports conversion from HuggingFace/PyTorch models to CoreML .mlpackage
    with various quantization options (FLOAT16, INT8, INT4, INT6).
    
    Examples:
        >>> quantizer = CoreMLQuantizer(
        ...     model_path="./my-model",
        ...     output_dir="./output"
        ... )
        >>> output = quantizer.quantize(format="float16")
    """
    
    MLPACKAGE_EXTENSION = ".mlpackage"
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None,
        input_shape: tuple = (1, 512),
        compute_units: str = "ALL",
        seq_length: int = 512
    ):
        """
        Initialize CoreML quantizer.
        
        Args:
            model_path: Path to source model (HuggingFace directory or PyTorch file)
            output_dir: Output directory for quantized models
            progress_callback: Optional callback(progress: int, stage: str, message: str)
            input_shape: Input shape for tracing (batch, sequence_length)
            compute_units: CoreML compute units ("ALL", "CPU_ONLY", "CPU_AND_GPU")
            seq_length: Maximum sequence length for model export
        """
        self.input_shape = input_shape
        self.compute_units = compute_units
        self.seq_length = seq_length
        self._pytorch_model = None
        self._exported_model = None
        self._current_format = None
        super().__init__(model_path, output_dir, progress_callback)
    
    @classmethod
    def get_backend_name(cls) -> str:
        """Get backend name."""
        return "coreml"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        """Get supported input formats."""
        return [ModelFormat.PYTORCH, ModelFormat.HUGGINGFACE]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        """
        List available CoreML quantization formats.
        
        Returns:
            Dictionary mapping format names to descriptions
        """
        return {fmt.value: COREML_FORMAT_DESCRIPTIONS[fmt] for fmt in CoreMLFormat}
    
    def _validate_backend_compatibility(self) -> None:
        """Validate input format compatibility."""
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(
                f"CoreML backend does not support {self.input_format.value}. "
                f"Supported: {[f.value for f in self.get_supported_input_formats()]}"
            )
    
    def _check_dependencies(self) -> None:
        """Check if CoreML dependencies are available."""
        if not _COREML_AVAILABLE:
            raise DependencyError(
                f"CoreML dependencies not available. Install with: pip install qwodel[coreml]\n"
                f"Error: {_COREML_IMPORT_ERROR}"
            )
        
        self.logger.info(f"✓ coremltools {ct.__version__}")
        self.logger.info(f"✓ torch {torch.__version__}")
    
    def _load_pytorch_model(self) -> torch.nn.Module:
        """
        Load model from HuggingFace directory or PyTorch file.
        
        Returns:
            Loaded PyTorch model in eval mode
        """
        if self._pytorch_model is not None:
            return self._pytorch_model
        
        # Determine model directory
        if self.model_path.is_file():
            model_dir = self.model_path.parent
            self.logger.info(f"Loading model from file: {self.model_path}")
        else:
            model_dir = self.model_path
            self.logger.info(f"Loading model from directory: {self.model_path}")
        
        # Load config and model
        config = AutoConfig.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        
        model = AutoModel.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"  # Required for export
        )
        
        model.eval()
        self._pytorch_model = model
        self.logger.info(f"✓ Model loaded: {config.model_type}")
        return model
    
    def _create_stateless_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Create a stateless wrapper for export compatibility.
        
        Args:
            model: Base PyTorch model
            
        Returns:
            Wrapped stateless model
        """
        class StatelessWrapper(torch.nn.Module):
            """Stateless wrapper for model export"""
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
            
            def forward(self, input_ids, attention_mask):
                """Forward pass returning only tensor output"""
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                )
                return outputs.last_hidden_state
        
        wrapper = StatelessWrapper(model)
        wrapper.eval()
        self.logger.info("✓ Created stateless model wrapper")
        return wrapper
    
    def _export_model(self, model: torch.nn.Module) -> Any:
        """
        Export model using torch.export or torch.jit.trace.
        
        Args:
            model: Stateless wrapped model
            
        Returns:
            Exported model
        """
        if self._exported_model is not None:
            return self._exported_model
        
        self.logger.info("Exporting model with torch.export...")
        
        # Get vocab size
        if hasattr(model.model, "get_input_embeddings"):
            embedding_layer = model.model.get_input_embeddings()
            vocab_size = embedding_layer.weight.shape[0]
        else:
            vocab_size = 50000  # Fallback
        
        self.logger.info(f"Using vocab size: {vocab_size}")
        
        # Create example inputs
        batch_size, seq_length = self.input_shape
        example_input_ids = torch.randint(
            0, vocab_size,
            (batch_size, seq_length),
            dtype=torch.long
        )
        example_attention_mask = torch.ones(
            (batch_size, seq_length),
            dtype=torch.long
        )
        
        # Try torch.export, fallback to jit.trace
        try:
            exported = torch.export.export(
                model,
                args=(example_input_ids, example_attention_mask)
            )
            self.logger.info("✓ torch.export successful")
            self._exported_model = exported
            return exported
        except Exception as e:
            self.logger.warning(f"torch.export failed: {e}")
            self.logger.info("Falling back to torch.jit.trace...")
            
            with torch.no_grad():
                traced = torch.jit.trace(
                    model,
                    (example_input_ids, example_attention_mask),
                    strict=False,
                    check_trace=False
                )
            
            self.logger.info("✓ torch.jit.trace successful")
            self._exported_model = traced
            return traced
    
    def _convert_to_coreml(self, exported_model: Any) -> Any:
        """
        Convert exported PyTorch model to CoreML.
        
        Args:
            exported_model: Exported model
            
        Returns:
            CoreML model
        """
        import numpy as np
        
        self.logger.info("Converting to CoreML format...")
        
        # Get quantization config for initial conversion
        quant_config = self._get_initial_conversion_config()
        
        # Define input shapes
        batch_size = 1
        seq_length = ct.RangeDim(lower_bound=1, upper_bound=self.seq_length, default=128)
        
        try:
            coreml_model = ct.convert(
                exported_model,
                inputs=[
                    ct.TensorType(
                        name="input_ids",
                        shape=(batch_size, seq_length),
                        dtype=np.int32
                    ),
                    ct.TensorType(
                        name="attention_mask",
                        shape=(batch_size, seq_length),
                        dtype=np.int32
                    )
                ],
                outputs=[ct.TensorType(name="output")],
                compute_units=getattr(ct.ComputeUnit, self.compute_units, ct.ComputeUnit.ALL),
                skip_model_load=False,
                **quant_config
            )
            
            self.logger.info("✓ CoreML conversion successful")
            return coreml_model
            
        except Exception as e:
            raise QuantizationError(f"CoreML conversion failed: {e}")
    
    def _get_initial_conversion_config(self) -> Dict[str, Any]:
        """Get configuration for initial CoreML conversion."""
        # Float16 for initial conversion if format is float16, otherwise FP32
        if self._current_format and "float16" in self._current_format.lower():
            return {
                "compute_precision": ct.precision.FLOAT16,
                "minimum_deployment_target": ct.target.iOS15
            }
        return {
            "compute_precision": ct.precision.FLOAT32,
            "minimum_deployment_target": ct.target.iOS16
        }
    
    def _apply_quantization(self, coreml_model: Any) -> Any:
        """
        Apply post-conversion quantization based on format.
        
        Args:
            coreml_model: CoreML model
            
        Returns:
            Quantized CoreML model
        """
        if not self._current_format:
            return coreml_model
        
        format_lower = self._current_format.lower()
        
        # Float16 - already applied during conversion
        if "float16" in format_lower:
            return coreml_model
        
        # INT8 linear quantization
        elif "int8" in format_lower:
            self.logger.info("Applying INT8 linear quantization...")
            try:
                op_config = OpLinearQuantizerConfig(mode="linear", dtype="int8")
                config = OptimizationConfig(global_config=op_config)
                quantized_model = linear_quantize_weights(coreml_model, config=config)
                self.logger.info("✓ INT8 quantization applied")
                return quantized_model
            except Exception as e:
                self.logger.warning(f"INT8 quantization failed: {e}")
                return coreml_model
        
        # INT4 palettization
        elif "int4" in format_lower:
            self.logger.info("Applying INT4 palettization (k-means)...")
            try:
                op_config = OpPalettizerConfig(mode="kmeans", nbits=4)
                config = OptimizationConfig(global_config=op_config)
                quantized_model = palettize_weights(coreml_model, config=config)
                self.logger.info("✓ INT4 palettization applied")
                return quantized_model
            except Exception as e:
                self.logger.warning(f"INT4 palettization failed: {e}")
                return coreml_model
        
        # INT6 palettization
        elif "int6" in format_lower:
            self.logger.info("Applying INT6 palettization (k-means)...")
            try:
                op_config = OpPalettizerConfig(mode="kmeans", nbits=6)
                config = OptimizationConfig(global_config=op_config)
                quantized_model = palettize_weights(coreml_model, config=config)
                self.logger.info("✓ INT6 palettization applied")
                return quantized_model
            except Exception as e:
                self.logger.warning(f"INT6 palettization failed: {e}")
                return coreml_model
        
        return coreml_model
    
    def _run_quantization(self) -> None:
        """Execute the complete CoreML quantization pipeline."""
        # Load PyTorch model
        self._report_progress(15, "loading", "Loading PyTorch model")
        self.logger.info("Step 1/5: Loading PyTorch model...")
        pytorch_model = self._load_pytorch_model()
        
        # Create stateless wrapper
        self._report_progress(25, "wrapping", "Creating stateless wrapper")
        self.logger.info("Step 2/5: Creating stateless wrapper...")
        stateless_model = self._create_stateless_wrapper(pytorch_model)
        
        # Export model
        self._report_progress(45, "exporting", "Exporting model")
        self.logger.info("Step 3/5: Exporting model...")
        exported_model = self._export_model(stateless_model)
        
        # Convert to CoreML
        self._report_progress(70, "converting", "Converting to CoreML format")
        self.logger.info("Step 4/5: Converting to CoreML...")
        coreml_model = self._convert_to_coreml(exported_model)
        
        # Apply quantization
        self._report_progress(90, "quantizing", f"Applying {self._current_format} quantization")
        self.logger.info(f"Step 5/5: Applying {self._current_format} quantization...")
        quantized_model = self._apply_quantization(coreml_model)
        
        # Save model
        output_path = self.get_output_path()
        self.logger.info(f"Saving to: {output_path}")
        quantized_model.save(str(output_path))
        
        if not output_path.exists():
            raise QuantizationError("Model saved but file not found")
        
        # Calculate size
        size_bytes = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file())
        size_mb = size_bytes / (1024 * 1024)
        self.logger.info(f"✅ Quantization complete! Size: {size_mb:.2f} MB")
    
    def quantize(self, format: str, **kwargs) -> Path:
        """
        Quantize model to specified CoreML format.
        
        Args:
            format: CoreML format name ("float16", "int8_linear", "int4", "int6")
            **kwargs: Additional arguments (unused for CoreML)
            
        Returns:
            Path to quantized model (.mlpackage directory)
            
        Raises:
            FormatNotSupportedError: If format is not supported
            QuantizationError: If quantization fails
        """
        # Validate format
        format_lower = format.lower().replace('-', '_')
        try:
            coreml_format = CoreMLFormat[format_lower.upper()]
        except KeyError:
            available = [f.value for f in CoreMLFormat]
            raise FormatNotSupportedError(
                f"Format '{format}' not supported by CoreML backend. "
                f"Available formats: {available}"
            )
        
        # Store format
        self._current_format = coreml_format.value
        
        # Call parent quantize
        return super().quantize(format=format, **kwargs)
    
    def get_output_path(self) -> Path:
        """Get output path for quantized model."""
        if not self._current_format:
            raise QuantizationError("Format not set. Call quantize() first.")
        
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        format_clean = self._current_format.replace('_', '-')
        return self.output_dir / f"{model_name}_{format_clean}{self.MLPACKAGE_EXTENSION}"


# Register CoreML backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("coreml", CoreMLQuantizer)
