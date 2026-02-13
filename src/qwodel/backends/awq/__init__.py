"""
AWQ Backend Implementation

Provides AWQ (Activation-aware Weight Quantization) using llm-compressor.
Supports GPU-based INT4 quantization with smart VRAM management.
"""

from typing import Dict, List, Optional
from pathlib import Path
import torch

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import ModelFormat, AWQFormat
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# Try to import AWQ dependencies
_AWQ_AVAILABLE = False
_AWQ_IMPORT_ERROR = None

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor import oneshot
    from datasets import load_dataset
    _AWQ_AVAILABLE = True
except ImportError as e:
    _AWQ_IMPORT_ERROR = str(e)


class AWQQuantizer(BaseQuantizer):
    """
    AWQ quantization backend using llm-compressor.
    
    Supports INT4 quantization for Causal LM models on NVIDIA GPUs.
    Includes smart VRAM management and OOM recovery.
    
    Examples:
        >>> quantizer = AWQQuantizer(
        ...     model_path="meta-llama/Llama-2-7b-hf",
        ...     output_dir="./output"
        ... )
        >>> output = quantizer.quantize(format="int4")
    """
    
    # Unsupported architectures for AWQ
    _unsupported_architectures = [
        "XLMRobertaForTokenClassification",
        "BertForTokenClassification",
        "RobertaForTokenClassification",
        "DistilBertForTokenClassification",
        "MobileBertForTokenClassification",
        "BertForMaskedLM",
        "RobertaForMaskedLM",
        # AWQ expects CausalLM models
    ]
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None
    ):
        """
        Initialize AWQ quantizer.
        
        Args:
            model_path: Path to source model (HuggingFace directory)
            output_dir: Output directory for quantized models
            progress_callback: Optional callback(progress: int, stage: str, message: str)
        """
        self._current_format = None
        super().__init__(model_path, output_dir, progress_callback)
    
    @property
    def unsupported_architectures(self) -> List[str]:
        """List of architectures unsupported by AWQ."""
        return self._unsupported_architectures
    
    @classmethod
    def get_backend_name(cls) -> str:
        """Get backend name."""
        return "awq"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        """Get supported input formats."""
        return [ModelFormat.HUGGINGFACE, ModelFormat.PYTORCH]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        """
        List available AWQ quantization formats.
        
        Returns:
            Dictionary mapping format names to descriptions
        """
        return {
            AWQFormat.INT4.value: "4-bit weight quantization with activation awareness (W4A16 scheme)"
        }
    
    def _validate_backend_compatibility(self) -> None:
        """Validate input format compatibility."""
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(
                f"AWQ backend requires HuggingFace/PyTorch model, "
                f"got {self.input_format.value}"
            )
    
    def _check_dependencies(self) -> None:
        """Check if AWQ dependencies are available."""
        if not _AWQ_AVAILABLE:
            raise DependencyError(
                f"AWQ dependencies not available. Install with: pip install qwodel[awq]\n"
                f"Error: {_AWQ_IMPORT_ERROR}"
            )
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. AWQ quantization requires GPU.")
            # Don't raise error to allow dry-run, but it will likely fail later
    
    def _get_model_params_gb(self) -> float:
        """Estimate model size in parameters (billions)."""
        try:
            # Calculate from file size
            total_size = sum(f.stat().st_size for f in self.model_path.glob("*.safetensors"))
            if total_size == 0:
                total_size = sum(f.stat().st_size for f in self.model_path.glob("*.bin"))
            
            size_gb = total_size / (1024**3)
            # Assuming FP16 (2 bytes per param), params = size / 2
            return size_gb / 2.0
        except Exception:
            return 7.0  # Default fallback
    
    def _get_gpu_vram_gb(self) -> float:
        """Get total GPU VRAM in GB."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024**3)
        return 0.0
    
    def _calculate_smart_config(self) -> Dict[str, int]:
        """
        Calculate dynamic quantization parameters based on available VRAM.
        
        Formula: Empty_VRAM = GPU_Total_VRAM - (Model_Params * 4.0)
        
        Returns:
            Dictionary with batch_size, seq_len, and samples
        """
        total_vram = self._get_gpu_vram_gb()
        model_params = self._get_model_params_gb()
        
        # 4.0 factor accounts for model weights + overhead + fragmentation
        empty_vram = total_vram - (model_params * 4.0)
        
        self.logger.info(
            f"Smart Config: Total VRAM={total_vram:.2f}GB, "
            f"Model~={model_params:.2f}B, Empty={empty_vram:.2f}GB"
        )
        
        if empty_vram < 6.0:
            # Slow & Safe
            config = {"batch_size": 1, "seq_len": 2048, "samples": 32}
            mode = "Slow & Safe"
        elif 6.0 <= empty_vram < 10.0:
            # Standard
            config = {"batch_size": 1, "seq_len": 4096, "samples": 64}
            mode = "Standard"
        elif 10.0 <= empty_vram < 16.0:
            # Balanced
            config = {"batch_size": 4, "seq_len": 4096, "samples": 128}
            mode = "Balanced"
        elif 16.0 <= empty_vram < 24.0:
            # High Quality
            config = {"batch_size": 8, "seq_len": 8192, "samples": 128}
            mode = "High Quality"
        else:  # > 24.0
            # Turbo
            config = {"batch_size": 16, "seq_len": 8192, "samples": 256}
            mode = "Turbo"
        
        self.logger.info(f"Selected Mode: {mode} -> {config}")
        return config
    
    def _get_dir_size(self, path: Path) -> float:
        """Calculate total size of a directory in MB."""
        total_size = 0
        path = Path(path)
        if path.is_file():
            total_size = path.stat().st_size
        else:
            for f in path.glob('**/*'):
                if f.is_file():
                    total_size += f.stat().st_size
        return total_size / (1024 * 1024)
    
    def _run_quantization(self) -> None:
        """Execute AWQ quantization using llm-compressor."""
        # Calculate smart config
        config = self._calculate_smart_config()
        
        self._report_progress(15, "loading", "Loading model for quantization")
        self.logger.info("Loading model for quantization...")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            trust_remote_code=True
        )
        
        # Load calibration dataset
        self._report_progress(25, "preparing", "Loading calibration dataset")
        self.logger.info("Loading calibration dataset...")
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        
        # Define AWQ recipe
        recipe = [
            AWQModifier(
                targets=["Linear"],
                ignore=["lm_head"],
                scheme="W4A16"
            )
        ]
        
        # Measure initial size
        initial_size_mb = self._get_dir_size(self.model_path)
        self.logger.info(f"Initial Model Size: {initial_size_mb:.2f} MB")
        
        # OOM Recovery Loop
        max_retries = 3
        self._report_progress(40, "quantizing", "Running AWQ quantization")
        
        for attempt in range(max_retries):
            try:
                self.logger.info(
                    f"Starting one-shot quantization (Attempt {attempt+1}/{max_retries}) "
                    f"with batch_size={config['batch_size']}..."
                )
                
                # Run quantization
                oneshot(
                    model=model,
                    dataset=dataset,
                    recipe=recipe,
                    num_calibration_samples=config['samples'],
                    max_seq_length=config['seq_len'],
                    batch_size=config['batch_size']
                )
                break  # Success!
                
            except torch.cuda.OutOfMemoryError:
                self.logger.warning(f"⚠️ CUDA OOM detected on attempt {attempt+1}!")
                if attempt == max_retries - 1:
                    self.logger.error("❌ Max retries reached. Failing job.")
                    raise
                
                # Recovery strategy
                torch.cuda.empty_cache()
                old_bs = config['batch_size']
                new_bs = max(1, old_bs // 2)
                config['batch_size'] = new_bs
                self.logger.info(f"📉 Reducing batch size: {old_bs} -> {new_bs} and retrying...")
                
            except Exception as e:
                # Check for OOM in generic exception message
                if "out of memory" in str(e).lower():
                    self.logger.warning(
                        f"⚠️ CUDA OOM detected (via generic exception) on attempt {attempt+1}!"
                    )
                    if attempt == max_retries - 1:
                        raise
                    torch.cuda.empty_cache()
                    old_bs = config['batch_size']
                    new_bs = max(1, old_bs // 2)
                    config['batch_size'] = new_bs
                    self.logger.info(f"📉 Reducing batch size: {old_bs} -> {new_bs} and retrying...")
                else:
                    raise
        
        # Save quantized model
        self._report_progress(85, "saving", "Saving quantized model")
        self.logger.info("Saving quantized model...")
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        
        # Calculate statistics
        final_size_mb = self._get_dir_size(self.output_dir)
        compression_ratio = initial_size_mb / final_size_mb if final_size_mb > 0 else 0
        size_drop = (1 - (final_size_mb / initial_size_mb)) * 100 if initial_size_mb > 0 else 0
        
        self.logger.info("-" * 40)
        self.logger.info("📊 QUANTIZATION STATISTICS")
        self.logger.info("-" * 40)
        self.logger.info(f"Original Size:     {initial_size_mb:.2f} MB")
        self.logger.info(f"Quantized Size:    {final_size_mb:.2f} MB")
        self.logger.info(f"Compression Ratio: {compression_ratio:.2f}x")
        self.logger.info(f"Size Reduction:    {size_drop:.2f}%")
        self.logger.info("-" * 40)
        
        self._report_progress(95, "complete", "Quantization complete")
    
    def quantize(self, format: str, **kwargs) -> Path:
        """
        Quantize model to specified AWQ format.
        
        Args:
            format: AWQ format name (currently only "int4" supported)
            **kwargs: Additional arguments (unused for AWQ)
            
        Returns:
            Path to quantized model directory
            
        Raises:
            FormatNotSupportedError: If format is not supported
            QuantizationError: If quantization fails
        """
        # Validate format
        format_lower = format.lower()
        try:
            awq_format = AWQFormat[format_lower.upper().replace('-', '_')]
        except KeyError:
            available = [f.value for f in AWQFormat]
            raise FormatNotSupportedError(
                f"Format '{format}' not supported by AWQ backend. "
                f"Available formats: {available}"
            )
        
        # Store format
        self._current_format = awq_format.value
        
        # Call parent quantize
        return super().quantize(format=format, **kwargs)
    
    def get_output_path(self) -> Path:
        """Get output path for quantized model (directory)."""
        return self.output_dir


# Register AWQ backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("awq", AWQQuantizer)
