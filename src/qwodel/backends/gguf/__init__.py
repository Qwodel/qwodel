"""
GGUF Backend Implementation

Provides GGUF quantization using llama.cpp tools (via llama-cpp-python or binary).
Supports automatic download of conversion scripts and logical fallback.
"""

from typing import Dict, List, Optional
from pathlib import Path
import subprocess
import tempfile
import shutil
import os
import sys
import requests
import stat

from qwodel.core.base import BaseQuantizer
from qwodel.core.constants import ModelFormat, GGUFFormat, GGUF_FORMAT_DESCRIPTIONS
from qwodel.core.exceptions import (
    QuantizationError,
    ValidationError,
    DependencyError,
    FormatNotSupportedError,
)

# URL for the conversion script (using master for latest model support like Qwen3)
CONVERT_SCRIPT_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"


class GGUFQuantizer(BaseQuantizer):
    """
    GGUF quantization backend.
    
    Strategies:
    1. Conversion: Uses downloaded convert_hf_to_gguf.py
    2. Quantization: Tries llama-quantize binary first, then falls back to python binding if available.
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
        self._temp_gguf_path = None
        self._current_format = None
        self._conversion_script_path = None
        super().__init__(model_path, output_dir, progress_callback)
    
    @property
    def unsupported_architectures(self) -> List[str]:
        return self._unsupported_architectures
    
    @classmethod
    def get_backend_name(cls) -> str:
        return "gguf"
    
    @classmethod
    def get_supported_input_formats(cls) -> List[ModelFormat]:
        """Get supported input formats."""
        return [ModelFormat.GGUF, ModelFormat.HUGGINGFACE]
    
    @classmethod
    def list_formats(cls) -> Dict[str, str]:
        return {fmt.value: GGUF_FORMAT_DESCRIPTIONS[fmt] for fmt in GGUFFormat}
    
    def _validate_backend_compatibility(self) -> None:
        """Validate input format compatibility."""
        if self.input_format not in self.get_supported_input_formats():
            raise ValidationError(
                f"GGUF backend does not support {self.input_format.value} format."
            )
        
        # Verify GGUF file exists if input is GGUF
        if self.input_format == ModelFormat.GGUF:
            if not self.model_path.exists():
                raise ValidationError(f"GGUF model file not found: {self.model_path}")
            if not self.model_path.is_file():
                raise ValidationError(f"GGUF model path must be a file, not a directory: {self.model_path}")
        
        # Validate HuggingFace directory structure
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
        """Check for llama-cpp-python or llama-quantize binary."""
        # 1. Check for Python binding
        self._has_python_binding = False
        try:
            import llama_cpp
            self._has_python_binding = True
            self.logger.info("llama-cpp-python found")
        except ImportError:
            pass
            
        # 2. Check for binary
        self._has_binary = False
        try:
            subprocess.run(["llama-quantize", "--help"], capture_output=True, timeout=5)
            self._has_binary = True
            self.logger.info("llama-quantize binary found")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
            
        if not self._has_python_binding and not self._has_binary:
            raise DependencyError(
                "Neither llama-quantize binary nor llama-cpp-python found.\n"
                "Please install: pip install qwodel[gguf]"
            )
            
    def _get_conversion_script(self) -> Path:
        """Get the conversion script (bundled b7921 or local llama.cpp)."""
        # Priority 1: Use bundled script from package (llama.cpp b7921)
        try:
            import qwodel.backends.gguf.scripts
            scripts_dir = Path(qwodel.backends.gguf.scripts.__file__).parent
            script_path = scripts_dir / "convert_hf_to_gguf.py"
            
            if script_path.exists():
                self.logger.info("Using bundled conversion script (llama.cpp b7921)")
                return script_path
        except Exception as e:
            self.logger.warning(f"Could not load bundled script: {e}")
        
        # Priority 2: Check for local llama.cpp installation (may be newer)
        llamacpp_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
        if llamacpp_script.exists():
            self.logger.info("Using local llama.cpp installation")
            return llamacpp_script
        
        # Priority 3: Check cache (from previous versions/downloads)
        cache_dir = Path.home() / ".cache" / "qwodel"
        script_path = cache_dir / "convert_hf_to_gguf.py"  
        if script_path.exists():
            self.logger.info("Using cached conversion script")
            return script_path
        
        raise DependencyError(
            "Conversion script not found.\n"
            "This is likely a package installation issue.\n"
            "Try: pip install --force-reinstall qwodel[gguf]"
        )
    
    def _convert_to_gguf(self) -> Path:
        """Convert HuggingFace model to GGUF format (based on bekenstein's approach)."""
        if self.input_format == ModelFormat.GGUF:
            return self.model_path
        
        # Get conversion script
        script_path = self._get_conversion_script()
        
        # Create temporary directory for output
        temp_dir = Path(tempfile.mkdtemp())
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        gguf_path = temp_dir / f"{model_name}.gguf"
        
        self.logger.info(f"Converting HuggingFace model to GGUF...")
        self.logger.info(f"Input: {self.model_path}")
        self.logger.info(f"Output: {gguf_path}")
        
        # Prepare environment with vendored gguf library
        env = os.environ.copy()
        try:
            import qwodel.backends.gguf.vendor.gguf_py
            # qwodel/backends/gguf/vendor/gguf_py/__init__.py -> parent is gguf_py -> parent is vendor
            vendor_path = Path(qwodel.backends.gguf.vendor.gguf_py.__file__).parent
            
            # Prepend vendor path to PYTHONPATH so it takes precedence over system gguf
            current_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = f"{vendor_path}:{current_pythonpath}"
            self.logger.info(f"Injecting vendored gguf library from: {vendor_path}")
        except (ImportError, AttributeError) as e:
            self.logger.warning(f"Could not find vendored gguf library ({e}) - using system default")

        # Build command (use venv's python to ensure correct dependencies)
        cmd =  [
            sys.executable,  # Use current python (venv)
            str(script_path),
            str(self.model_path),
            "--outtype", "f16",
            "--outfile", str(gguf_path)
        ]
        
        self.logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=1800,  # 30 minutes max
                env=env  # Use modified environment
            )
            
            if not gguf_path.exists():
                raise QuantizationError("Conversion completed but output file not found")
            
            self._temp_gguf_path = gguf_path
            self.logger.info("Conversion successful")
            return gguf_path
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed: {e.stderr}")
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            raise QuantizationError(f"HF->GGUF conversion failed: {e.stderr}")



    def _run_quantization(self) -> None:
        """Run quantization using available method."""
        # Convert HF to GGUF if needed
        self._report_progress(15, "preparing", "Preparing model")
        gguf_path = self._convert_to_gguf()
        output_path = self.get_output_path()
        
        # Quantize
        self._report_progress(50, "quantizing", f"Running {self._current_format} quantization")
        
        try:
            # Try binary first (often faster)
            if self._has_binary:
                self._run_binary_quantization(gguf_path, output_path)
            # Fallback to Python binding
            elif self._has_python_binding:
                self._run_python_quantization(gguf_path, output_path)
            else:
                raise DependencyError("No quantization capability found")
                
            # Verify
            if not output_path.exists():
                raise QuantizationError("Output file not found after quantization")
                
            size_mb = output_path.stat().st_size / (1024 * 1024)
            self._report_progress(95, "saving", f"Size: {size_mb:.2f} MB")
            self.logger.info(f"Quantization complete: {output_path}")
        finally:
            # Cleanup temp GGUF file if created during conversion
            if self._temp_gguf_path and self._temp_gguf_path.exists():
                shutil.rmtree(self._temp_gguf_path.parent)
                self._temp_gguf_path = None

    def _run_binary_quantization(self, input_path: Path, output_path: Path):
        """Run quantization using llama-quantize binary."""
        cmd = [
            "llama-quantize",
            str(input_path),
            str(output_path),
            self._current_format
        ]
        self.logger.info(f"Running binary: {' '.join(cmd)}")
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=3600
        )

    def _run_python_quantization(self, input_path: Path, output_path: Path):
        """Run quantization using llama-cpp-python binding."""
        self.logger.info("Running python binding quantization...")
        
        # Map string format to integer ID expected by llama.cpp
        # This mapping might need to be adjusted based on llama-cpp-python version
        # For now, we rely on the generic 'quantize' function if available
        # or we might need to map format strings to format IDs.
        
        from llama_cpp import llama_model_quantize, llama_model_quantize_params
        
        # We need to map format string (e.g. Q4_K_M) to valid params
        # llama-cpp-python usually accepts the format name string or ID
        
        # For safety, let's look up the ID if possible, but the python binding 
        # usually handles the string parsing or we pass params.
        
        # Actually, standard llama_model_quantize takes:
        # fname_inp, fname_out, params
        
        # We need to construct params.
        # This part is tricky without docs, but let's try the high-level API if available.
        
        # Let's try passing the format string directly if supported, or map it.
        # Fallback mapping (common IDs):
        # Q4_K_M = 15, Q8_0 = 7, etc.
        # Ideally we use the binding's constants.
        
        import llama_cpp
        
        # Get format ID from string
        fmt_name = self._current_format
        # Try to find matching constant in llama_cpp
        # e.g. LLAMA_FTYPE_MOSTLY_Q4_K_M
        
        ftype = None
        target_attr = f"LLAMA_FTYPE_MOSTLY_{fmt_name}"
        if hasattr(llama_cpp, target_attr):
            ftype = getattr(llama_cpp, target_attr)
        else:
            # Try without MOSTLY prefix for older/newer versions?
             # List scan
            for attr in dir(llama_cpp):
                if attr.startswith("LLAMA_FTYPE_") and fmt_name in attr:
                    ftype = getattr(llama_cpp, attr)
                    break
        
        if ftype is None:
            # Fallback map for common formats
            fallback_map = {
                "Q4_K_M": 15, "Q4_K_S": 14, "Q5_K_M": 17, "Q5_K_S": 16,
                "Q2_K": 10, "Q3_K_M": 12, "Q6_K": 18, "Q8_0": 7, "Q4_0": 2
            }
            ftype = fallback_map.get(fmt_name)
            
        if ftype is None:
             raise FormatNotSupportedError(f"Could not map format {fmt_name} to llama-cpp ID")
             
        params = llama_model_quantize_params(
            nthread=os.cpu_count() or 4,
            ftype=ftype,
            allow_requantize=False,
            quantize_output_tensor=True,
            only_copy=False,
            pure=False,
            keep_split=False,
        )
        
        result = llama_model_quantize(
            str(input_path).encode("utf-8"),
            str(output_path).encode("utf-8"),
            params
        )
        
        if result != 0:
            raise QuantizationError(f"llama_model_quantize failed with code {result}")

    def quantize(self, format: str, **kwargs) -> Path:
        format_upper = format.upper()
        try:
            gguf_format = GGUFFormat[format_upper]
        except KeyError:
             available = [f.value for f in GGUFFormat]
             raise FormatNotSupportedError(f"Format '{format}' not supported. Available: {available}")
        
        self._current_format = gguf_format.value
        return super().quantize(format=format, **kwargs)

    def get_output_path(self) -> Path:
        if not self._current_format:
            raise QuantizationError("Format not set")
        model_name = self.model_path.stem if self.model_path.is_file() else self.model_path.name
        format_lower = self._current_format.lower().replace('_', '_')
        return self.output_dir / f"{model_name}-{format_lower}.gguf"


# Register GGUF backend
from qwodel.backends import BackendRegistry
BackendRegistry.register("gguf", GGUFQuantizer)
