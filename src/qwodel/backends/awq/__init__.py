"""
AWQ Backend Implementation

Provides AWQ (Activation-aware Weight Quantization) using llm-compressor.
Supports GPU-based INT4 quantization with smart VRAM management.
"""

import os
from typing import Dict, List, Optional
from pathlib import Path
import torch

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import (
    ModelFormat, 
    AWQFormat,
    DEFAULT_CALIBRATION_DATASET,
    DEFAULT_CALIBRATION_SPLIT,
    UNSUPPORTED_ARCHITECTURES,
    AWQ_DEFAULT_IGNORE_MAP
)
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# Optional imports for AWQ
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor.modifiers.awq import AWQModifier
    from llmcompressor import oneshot
    from datasets import load_dataset
    _AWQ_AVAILABLE = True
except ImportError:
    _AWQ_AVAILABLE = False


class AWQQuantizer(BaseQuantizer):
    """
    AWQ quantization backend using llm-compressor.
    Supports INT4 quantization for Causal LM models on NVIDIA GPUs.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None,
        calibration_dataset: str = DEFAULT_CALIBRATION_DATASET,
        calibration_split: str = DEFAULT_CALIBRATION_SPLIT,
        batch_size: Optional[int] = None,
        seq_length: Optional[int] = None,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        super().__init__(model_path, output_dir, progress_callback)
        self.calibration_dataset = calibration_dataset
        self.calibration_split = calibration_split
        self.manual_config = {
            "batch_size": batch_size,
            "seq_len": seq_length,
            "samples": num_samples,
            **kwargs
        }
        self.token = kwargs.get('token')
        self._current_format = None
    
    @property
    def unsupported_architectures(self) -> List[str]:
        return UNSUPPORTED_ARCHITECTURES
    
    @classmethod
    def get_backend_name(cls) -> str:
        return "awq"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        return [ModelFormat.HUGGINGFACE, ModelFormat.PYTORCH]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        return {
            AWQFormat.INT4.value: "4-bit weight quantization (W4A16)"
        }
    
    def _validate_backend_compatibility(self) -> None:
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(f"AWQ backend requires HuggingFace/PyTorch model, got {self.input_format.value}")
    
    def _check_dependencies(self) -> None:
        if not _AWQ_AVAILABLE:
            raise DependencyError("AWQ dependencies not available. Install with: pip install qwodel[awq]")
        
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available. AWQ quantization requires GPU.")
    
    def _get_ignore_list(self, model_config):
        return AWQ_DEFAULT_IGNORE_MAP.get(model_config.model_type, AWQ_DEFAULT_IGNORE_MAP["default"])

    def _get_model_size_gb(self) -> float:
        """Estimate model size in GB."""
        files = list(self.model_path.glob("*.safetensors")) or list(self.model_path.glob("*.bin"))
        total_bytes = sum(f.stat().st_size for f in files)
        return total_bytes / (1024**3)

    def _calculate_config(self) -> Dict[str, int]:
        """Calculate quantization parameters based on available VRAM."""
        # Defaults
        config = {"batch_size": 8, "seq_len": 4096, "samples": 128}

        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            model_size = self._get_model_size_gb()
            # Crude estimate: Model weights (FP16) + overhead ~ 2x model size
            # We want to leave room for activation and batches.
            # If we assume loading in FP16, we need strict VRAM management.
            
            # Simple heuristic levels based on free memory "headroom"
            # Headroom = Total VRAM - Model Size (FP16)
            headroom = total_vram - model_size 
            
            if headroom < 4.0:
                config = {"batch_size": 1, "seq_len": 2048, "samples": 32}
            elif headroom < 8.0:
                config = {"batch_size": 2, "seq_len": 4096, "samples": 64}
            elif headroom < 16.0:
                config = {"batch_size": 4, "seq_len": 4096, "samples": 128}
            elif headroom < 24.0:
                config = {"batch_size": 8, "seq_len": 8192, "samples": 128}
            else:
                config = {"batch_size": 16, "seq_len": 8192, "samples": 256}
                
            self.logger.info(f"VRAM Headroom: {headroom:.2f}GB. Selected config: {config}")

        # Overrides
        for k, v in self.manual_config.items():
            if v is not None:
                config[k] = v
                
        return config
    
    def _load_calibration_dataset(self):
        """
        Load calibration dataset with support for:
        1. Local files (json, jsonl, txt)
        2. HuggingFace datasets with subsets (repo:subset)
        3. Standard HuggingFace datasets
        """
        dataset_path = self.calibration_dataset
        
        # 1. Check for local file
        if os.path.isfile(dataset_path):
            self.logger.info(f"Loading local calibration dataset: {dataset_path}")
            ext = Path(dataset_path).suffix.lower()
            if ext in ['.json', '.jsonl']:
                # For JSONL, we usually want the 'train' split. 
                # load_dataset("json", ...) returns a DatasetDict usually.
                return load_dataset("json", data_files=dataset_path, split=self.calibration_split)
            elif ext in ['.txt']:
                return load_dataset("text", data_files=dataset_path, split=self.calibration_split)
            else:
                # Fallback for other types if supported by generic load_dataset, or fail
                pass

        # 2. Check for subset syntax (repo:subset)
        subset = None
        if ":" in dataset_path:
            dataset_path, subset = dataset_path.split(":", 1)
        elif "," in dataset_path:
            # Support comma as fallback/alternative separator
            dataset_path, subset = dataset_path.split(",", 1)
            
        self.logger.info(f"Loading HF dataset: {dataset_path} (subset={subset}, split={self.calibration_split})")
        return load_dataset(dataset_path, subset, split=self.calibration_split)

    def _run_quantization(self) -> None:
        config = self._calculate_config()
        
        self._report_progress(10, "loading", "Loading model")
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            token=self.token,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True, token=self.token)
        
        self._report_progress(20, "preparing", "Loading dataset")
        try:
            dataset = self._load_calibration_dataset()
        except Exception as e:
            raise QuantizationError(f"Failed to load dataset '{self.calibration_dataset}': {e}")
        
        # Determine ignore list: User override > Default map
        if "ignore" in self.manual_config:
            ignore_list = self.manual_config["ignore"]
            self.logger.info(f"Using manual ignore list: {ignore_list}")
        else:
            ignore_list = self._get_ignore_list(model.config)

        recipe = [
            AWQModifier(
                targets=["Linear"],
                ignore=ignore_list,
                scheme="W4A16"
            )
        ]
        
        self._report_progress(30, "quantizing", "Running AWQ")
        
        # Simple retry loop for OOM
        max_retries = 5
        import gc
        for i in range(max_retries):
            try:
                self.logger.info(f"Starting quantization (Attempt {i+1}/{max_retries}) bs={config['batch_size']}, seq_len={config['seq_len']}")
                oneshot(
                    model=model,
                    dataset=dataset,
                    recipe=recipe,
                    num_calibration_samples=config['samples'],
                    max_seq_length=config['seq_len']
                )
                break
            except Exception as e:
                # Check for OOM or related runtime errors from llmcompressor/torch
                is_oom = "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)
                
                if is_oom and i < max_retries - 1:
                    # Clear cache
                    torch.cuda.empty_cache()
                    gc.collect()
                    
                    # Strategy: Reduce Batch Size first, then Sequence Length
                    if config['batch_size'] > 1:
                        config['batch_size'] //= 2
                        self.logger.warning(f"OOM detected. Reducing batch size to {config['batch_size']}")
                    elif config['seq_len'] > 128:
                        config['seq_len'] //= 2
                        self.logger.warning(f"OOM detected. Batch size is 1. Reducing sequence length to {config['seq_len']}")
                    else:
                        self.logger.error("OOM detected but cannot reduce parameters further.")
                        raise e
                else:
                    raise e
        
        self._report_progress(90, "saving", "Saving model")
        model.save_pretrained(str(self.output_dir))
        tokenizer.save_pretrained(str(self.output_dir))
        self._report_progress(100, "complete", "Done")

    def quantize(self, format: str, **kwargs) -> Path:
        format_upper = format.upper().replace('-', '_')
        if format_upper not in AWQFormat.__members__:
             raise FormatNotSupportedError(f"Format '{format}' not supported. Use: {list(self.list_formats().keys())}")
        
        # Support runtime config overrides
        for key in ['batch_size', 'seq_len', 'samples', 'num_samples', 'ignore']:
            if key in kwargs:
                # Map num_samples to internal 'samples' key
                target_key = 'samples' if key == 'num_samples' else key
                if kwargs[key] is not None:
                    self.manual_config[target_key] = kwargs[key]
                    self.logger.info(f"Overriding config: {target_key}={kwargs[key]}")

        self._current_format = AWQFormat[format_upper].value
        return super().quantize(format=format, **kwargs)

    def get_output_path(self) -> Path:
        return self.output_dir


# Register backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("awq", AWQQuantizer)
