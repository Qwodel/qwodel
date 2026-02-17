"""
GGUF Backend Implementation

Provides GGUF quantization using llama.cpp tools (via llama-cpp-python or binary).
Supports automatic download of conversion scripts and logical fallback.
"""

from typing import Dict, List, Optional
from pathlib import Path
import subprocess
import shutil
import os
import sys

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import (
    ModelFormat, 
    GGUFFormat, 
    GGUF_FORMAT_DESCRIPTIONS,
    UNSUPPORTED_ARCHITECTURES,
    GGUF_CONVERSION_SCRIPT,
    GGUF_BINARY_NAME
)
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# Optional imports
try:
    import llama_cpp
    _HAS_PYTHON_BINDING = True
except ImportError:
    _HAS_PYTHON_BINDING = False


class GGUFQuantizer(BaseQuantizer):
    """
    GGUF quantization backend.
    
    Strategies:
    1. Conversion: Uses bundled or local convert_hf_to_gguf.py
    2. Quantization: Tries llama-quantize binary first, then falls back to python binding.
    """
    
    def __init__(
        self,
        model_path: str,
        output_dir: str = "./quantized_models",
        progress_callback: Optional = None
    ):
        super().__init__(model_path, output_dir, progress_callback)
        self._temp_gguf_path = None
        self._current_format = None
        self._check_binary()

    @property
    def unsupported_architectures(self) -> List[str]:
        return UNSUPPORTED_ARCHITECTURES
    
    @classmethod
    def get_backend_name(cls) -> str:
        return "gguf"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        return [ModelFormat.GGUF, ModelFormat.HUGGINGFACE]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        return {fmt.value: GGUF_FORMAT_DESCRIPTIONS[fmt] for fmt in GGUFFormat}
    
    def _validate_backend_compatibility(self) -> None:
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(f"GGUF backend does not support {self.input_format.value} format.")
        
        # Verify GGUF file input
        if self.input_format == ModelFormat.GGUF:
            if not self.model_path.exists() or not self.model_path.is_file():
                 raise ValidationError(f"Invalid GGUF file: {self.model_path}")
        
        # Verify HF Directory
        if self.input_format == ModelFormat.HUGGINGFACE and self.model_path.is_dir():
             has_model = any(self.model_path.glob('*.bin')) or any(self.model_path.glob('*.safetensors'))
             if not has_model and not (self.model_path / 'pytorch_model.bin').exists():
                 raise ValidationError(f"No model files found in: {self.model_path}")

    def _check_binary(self) -> None:
        self._has_binary = False
        try:
            subprocess.run([GGUF_BINARY_NAME, "--help"], capture_output=True, timeout=2)
            self._has_binary = True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _check_dependencies(self) -> None:
        if not _HAS_PYTHON_BINDING and not self._has_binary:
            raise DependencyError(f"Neither {GGUF_BINARY_NAME} binary nor llama-cpp-python found. Install with: pip install qwodel[gguf]")
            
    def _get_conversion_script(self) -> Path:
        """Find the conversion script."""
        # 1. Bundled script
        try:
            import qwodel.backends.gguf.scripts
            bundled = Path(qwodel.backends.gguf.scripts.__file__).parent / GGUF_CONVERSION_SCRIPT
            if bundled.exists(): return bundled
        except ImportError:
            pass
        
        # 2. Local fallback
        local = Path.home() / "llama.cpp" / GGUF_CONVERSION_SCRIPT
        if local.exists(): return local
        
        # 3. Cache fallback
        cache = Path.home() / ".cache" / "qwodel" / GGUF_CONVERSION_SCRIPT
        if cache.exists(): return cache
        
        raise DependencyError(f"Conversion script ({GGUF_CONVERSION_SCRIPT}) not found.")
    
    def _convert_to_gguf(self) -> Path:
        if self.input_format == ModelFormat.GGUF:
            return self.model_path
        
        script = self._get_conversion_script()
        temp_dir = Path(self.output_dir) / "temp_conversion"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        outfile = temp_dir / f"{self.model_path.name}.gguf"
        self.logger.info(f"Converting HF to GGUF: {self.model_path} -> {outfile}")
        
        # Inject custom vendor path for gguf_py if present
        env = os.environ.copy()
        try:
            import qwodel.backends.gguf.vendor.gguf_py
            vendor_path = Path(qwodel.backends.gguf.vendor.gguf_py.__file__).parent
            env["PYTHONPATH"] = f"{vendor_path}:{env.get('PYTHONPATH', '')}"
        except ImportError:
            pass

        cmd = [sys.executable, str(script), str(self.model_path), "--outtype", "f16", "--outfile", str(outfile)]
        
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800, env=env)
            self._temp_gguf_path = outfile # Mark for cleanup
            return outfile
        except subprocess.CalledProcessError as e:
            if temp_dir.exists(): shutil.rmtree(temp_dir, ignore_errors=True)
            raise QuantizationError(f"Conversion failed: {e.stderr}")

    def _quantize_binary(self, infile: Path, outfile: Path):
        cmd = [GGUF_BINARY_NAME, str(infile), str(outfile), self._current_format]
        self.logger.info(f"Running binary: {' '.join(cmd)}")
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)

    def _quantize_python(self, infile: Path, outfile: Path):
        self.logger.info("Running python binding quantization...")
        try:
            from llama_cpp import llama_model_quantize, llama_model_quantize_params
            
            # Simple format map fallback
            fmt_map = {
                "Q4_K_M": 15, "Q4_K_S": 14, "Q5_K_M": 17, "Q5_K_S": 16,
                "Q2_K": 10, "Q3_K_M": 12, "Q6_K": 18, "Q8_0": 7, "Q4_0": 2
            }
            ftype = fmt_map.get(self._current_format)
            
            # Try to lookup dynamically if possible
            if ftype is None:
                for name, val in vars(llama_cpp).items():
                    if name.startswith("LLAMA_FTYPE_") and self._current_format in name:
                         ftype = val
                         break
            
            if ftype is None:
                 raise FormatNotSupportedError(f"Could not map format {self._current_format} to llama-cpp ID")

            params = llama_model_quantize_params(
                nthread=os.cpu_count() or 4,
                ftype=ftype,
                allow_requantize=False,
                quantize_output_tensor=True,
                only_copy=False,
                pure=False,
                keep_split=False
            )
            
            res = llama_model_quantize(str(infile).encode(), str(outfile).encode(), params)
            if res != 0: raise QuantizationError(f"Quantization failed with code {res}")
            
        except Exception as e:
            raise QuantizationError(f"Python binding quantization error: {e}")

    def _run_quantization(self) -> None:
        self._report_progress(10, "preparing", "Preparing model")
        gguf_input = self._convert_to_gguf()
        output_path = self.get_output_path()
        
        self._report_progress(50, "quantizing", f"Quantizing to {self._current_format}")
        
        try:
            if self._has_binary:
                self._quantize_binary(gguf_input, output_path)
            elif _HAS_PYTHON_BINDING:
                self._quantize_python(gguf_input, output_path)
            else:
                raise DependencyError("No accessible quantization method found.")
            
            if not output_path.exists():
                raise QuantizationError("Output file missing after quantization.")
                
            self.logger.info(f"Quantization complete: {output_path}")
            
        finally:
            if self._temp_gguf_path and self._temp_gguf_path.exists():
                shutil.rmtree(self._temp_gguf_path.parent, ignore_errors=True)
                self._temp_gguf_path = None

    def quantize(self, format: str, **kwargs) -> Path:
        format_upper = format.upper()
        if format_upper not in GGUFFormat.__members__:
             raise FormatNotSupportedError(f"Format '{format}' not supported. Available: {list(self.list_formats().keys())}")
        
        self._current_format = GGUFFormat[format_upper].value
        return super().quantize(format=format, **kwargs)

    def get_output_path(self) -> Path:
        if not self._current_format:
            raise QuantizationError("Format not set")
        
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        return self.output_dir / f"{model_name}-{self._current_format.lower()}.gguf"


# Register backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("gguf", GGUFQuantizer)
