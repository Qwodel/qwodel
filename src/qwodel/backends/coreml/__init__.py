"""
CoreML Backend Implementation

Provides CoreML quantization for Apple devices (iOS, macOS, iPadOS).
Supports FLOAT16, INT8, INT4, and INT6 quantization formats.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import torch
import numpy as np

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import (
    ModelFormat, 
    CoreMLFormat, 
    COREML_FORMAT_DESCRIPTIONS,
    COREML_INPUT_NAME,
    COREML_MASK_NAME,
    COREML_OUTPUT_NAME,
    COREML_PACKAGE_EXTENSION
)
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# Optional imports
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
except ImportError:
    _COREML_AVAILABLE = False


class CoreMLQuantizer(BaseQuantizer):
    """
    CoreML quantization backend for Apple devices.
    Supports FLOAT16, INT8, INT4, INT6.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None,
        input_shape: tuple = (1, 512),
        compute_units: str = "ALL",
        seq_length: int = 512
    ):
        super().__init__(model_path, output_dir, progress_callback)
        self.input_shape = input_shape
        self.compute_units = compute_units
        self.seq_length = seq_length
        self._pytorch_model = None
        self._exported_model = None
        self._current_format = None
    
    @classmethod
    def get_backend_name(cls) -> str:
        return "coreml"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        return [ModelFormat.PYTORCH, ModelFormat.HUGGINGFACE]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        return {fmt.value: COREML_FORMAT_DESCRIPTIONS[fmt] for fmt in CoreMLFormat}
    
    def _validate_backend_compatibility(self) -> None:
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(f"CoreML backend does not support {self.input_format.value}")
    
    def _check_dependencies(self) -> None:
        if not _COREML_AVAILABLE:
            raise DependencyError("CoreML dependencies not available. Install with: pip install qwodel[coreml]")
    
    def _load_pytorch_model(self) -> torch.nn.Module:
        if self._pytorch_model:
            return self._pytorch_model
        
        path = self.model_path.parent if self.model_path.is_file() else self.model_path
        self.logger.info(f"Loading model from {path}")
        
        config = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            path,
            config=config,
            local_files_only=True,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager"
        )
        model.eval()
        self._pytorch_model = model
        return model
    
    def _create_stateless_wrapper(self, model: torch.nn.Module) -> torch.nn.Module:
        class StatelessWrapper(torch.nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.model = base_model
            
            def forward(self, input_ids, attention_mask):
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True
                ).last_hidden_state
        
        return StatelessWrapper(model).eval()
    
    def _export_model(self, model: torch.nn.Module) -> Any:
        if self._exported_model:
            return self._exported_model
            
        self.logger.info("Tracing model...")
        vocab_size = getattr(model.model.get_input_embeddings().weight, 'shape', [50000])[0]
        
        # Dummy inputs
        bs, seq = self.input_shape
        dummy_inputs = (
            torch.randint(0, vocab_size, (bs, seq), dtype=torch.long),
            torch.ones((bs, seq), dtype=torch.long)
        )
        
        try:
            self._exported_model = torch.export.export(model, args=dummy_inputs)
        except Exception as e:
            self.logger.warning(f"torch.export failed ({e}), falling back to jit.trace")
            with torch.no_grad():
                self._exported_model = torch.jit.trace(model, dummy_inputs, strict=False, check_trace=False)
                
        return self._exported_model

    def _convert_to_coreml(self, exported_model: Any) -> Any:
        self.logger.info("Converting to CoreML...")
        
        # Check explicit format requirements
        is_float16 = self._current_format and "float16" in self._current_format.lower()
        precision = ct.precision.FLOAT16 if is_float16 else ct.precision.FLOAT32
        target = ct.target.iOS15 if is_float16 else ct.target.iOS16
        
        inputs = [
            ct.TensorType(name=COREML_INPUT_NAME, shape=(1, ct.RangeDim(1, self.seq_length, default=128)), dtype=np.int32),
            ct.TensorType(name=COREML_MASK_NAME, shape=(1, ct.RangeDim(1, self.seq_length, default=128)), dtype=np.int32)
        ]
        
        try:
            return ct.convert(
                exported_model,
                inputs=inputs,
                outputs=[ct.TensorType(name=COREML_OUTPUT_NAME)],
                compute_units=getattr(ct.ComputeUnit, self.compute_units, ct.ComputeUnit.ALL),
                skip_model_load=False,
                compute_precision=precision,
                minimum_deployment_target=target
            )
        except Exception as e:
            raise QuantizationError(f"CoreML conversion failed: {e}")

    def _apply_quantization(self, ml_model: Any) -> Any:
        fmt = (self._current_format or "").lower()
        if not fmt or "float16" in fmt:
            return ml_model

        self.logger.info(f"Applying post-training quantization: {fmt}")
        try:
            if "int8" in fmt:
                config = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear", dtype="int8"))
                return linear_quantize_weights(ml_model, config=config)
            
            # Palettization (int4, int6)
            nbits = 4 if "int4" in fmt else (6 if "int6" in fmt else None)
            if nbits:
                config = OptimizationConfig(global_config=OpPalettizerConfig(mode="kmeans", nbits=nbits))
                return palettize_weights(ml_model, config=config)
                
        except Exception as e:
            self.logger.warning(f"Quantization optimization failed: {e}")
            
        return ml_model

    def _run_quantization(self) -> None:
        self._report_progress(10, "loading", "Loading PyTorch model")
        model = self._create_stateless_wrapper(self._load_pytorch_model())
        
        self._report_progress(40, "exporting", "Tracing model")
        exported = self._export_model(model)
        
        self._report_progress(60, "converting", "Converting to CoreML")
        ml_model = self._convert_to_coreml(exported)
        
        self._report_progress(80, "optimizing", f"Optimizing ({self._current_format})")
        final_model = self._apply_quantization(ml_model)
        
        self._report_progress(95, "saving", "Saving mlpackage")
        output_path = self.get_output_path()
        final_model.save(str(output_path))
        
        self.logger.info(f"Saved to {output_path}")

    def quantize(self, format: str, **kwargs) -> Path:
        format_upper = format.upper().replace('-', '_')
        if format_upper not in CoreMLFormat.__members__:
             raise FormatNotSupportedError(f"Format '{format}' not supported. Use: {list(self.list_formats().keys())}")
        
        self._current_format = CoreMLFormat[format_upper].value
        return super().quantize(format=format, **kwargs)

    def get_output_path(self) -> Path:
        if not self._current_format:
            raise QuantizationError("Format not set.")
        
        name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        return self.output_dir / f"{name}_{self._current_format.replace('_', '-')}{COREML_PACKAGE_EXTENSION}"


# Register backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("coreml", CoreMLQuantizer)
