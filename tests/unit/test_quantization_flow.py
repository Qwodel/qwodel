"""
Unit Tests for Quantization Flow

These tests verify the end-to-end quantization execution path for each backend
(GGUF, AWQ, CoreML) using mocks. No real model, GPU, or heavy dependency is
required — only the logic and orchestration in each backend is exercised.

Coverage:
- _check_dependencies is called before _run_quantization
- _run_quantization is called during quantize()
- progress callbacks fire at the right stages
- FormatNotSupportedError is raised for invalid formats
- DependencyError propagates cleanly when extras are missing
- Output path is returned on success
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from qwodel.core.exceptions import (
    DependencyError,
    FormatNotSupportedError,
    QuantizationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hf_model(tmp_path: Path, arch: str = "LlamaForCausalLM") -> Path:
    """Create a minimal fake HuggingFace model directory."""
    model_dir = tmp_path / "mock_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = {"architectures": [arch], "model_type": "llama", "vocab_size": 32000}
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "pytorch_model.bin").touch()
    return model_dir


def _make_gguf_file(tmp_path: Path) -> Path:
    """Create a minimal fake GGUF file."""
    f = tmp_path / "model.gguf"
    f.touch()
    return f


# ---------------------------------------------------------------------------
# GGUF backend
# ---------------------------------------------------------------------------

class TestGGUFQuantizationFlow:
    """Test the GGUF quantization execution path with mocks."""

    def _make_quantizer(self, model_path: Path, output_dir: Path):
        from qwodel.backends.gguf import GGUFQuantizer
        with patch.object(GGUFQuantizer, "_check_binary", return_value=None):
            q = GGUFQuantizer(
                model_path=str(model_path),
                output_dir=str(output_dir),
            )
            q._has_binary = True  # pretend binary exists
        return q

    def test_quantize_calls_run_quantization(self, tmp_path):
        """quantize() must call _run_quantization() exactly once."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        fake_output = out_dir / "model-q4_k_m.gguf"
        fake_output.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization", side_effect=lambda: fake_output.touch()), \
             patch.object(q, "get_output_path", return_value=fake_output):
            result = q.quantize("Q4_K_M")

        assert result == fake_output

    def test_progress_callback_fires(self, tmp_path):
        """Progress callback should be invoked during quantize()."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        callback = MagicMock()
        q.progress_callback = callback

        fake_output = out_dir / "model-q4_k_m.gguf"
        fake_output.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization", side_effect=lambda: fake_output.touch()), \
             patch.object(q, "get_output_path", return_value=fake_output):
            q.quantize("Q4_K_M")

        assert callback.call_count >= 2
        # First call must be the initializing stage
        first_call_args = callback.call_args_list[0]
        assert first_call_args == call(0, "initializing", "Starting quantization")

    def test_invalid_format_raises_error(self, tmp_path):
        """An unsupported format string must raise FormatNotSupportedError."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with pytest.raises(FormatNotSupportedError):
            q.quantize("INVALID_FORMAT_XYZ")

    def test_dependency_error_propagates(self, tmp_path):
        """DependencyError from _check_dependencies must bubble up cleanly."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with patch.object(q, "_check_dependencies",
                          side_effect=DependencyError("llama-quantize not found")):
            with pytest.raises(DependencyError, match="llama-quantize not found"):
                q.quantize("Q4_K_M")

    def test_missing_output_raises_quantization_error(self, tmp_path):
        """If _run_quantization completes but output is absent, raise QuantizationError."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        ghost = out_dir / "ghost.gguf"  # deliberately not created

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization"), \
             patch.object(q, "get_output_path", return_value=ghost):
            with pytest.raises(QuantizationError, match="output file not found"):
                q.quantize("Q4_K_M")

    def test_check_dependencies_called_before_run_quantization(self, tmp_path):
        """_check_dependencies must be invoked before _run_quantization."""
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        call_order = []
        fake_output = out_dir / "model-q4_k_m.gguf"
        fake_output.parent.mkdir(parents=True, exist_ok=True)

        with patch.object(q, "_check_dependencies",
                          side_effect=lambda: call_order.append("deps")), \
             patch.object(q, "_run_quantization",
                          side_effect=lambda: (call_order.append("run"), fake_output.touch())), \
             patch.object(q, "get_output_path", return_value=fake_output):
            q.quantize("Q4_K_M")

        assert call_order == ["deps", "run"]

    def test_valid_gguf_formats_accepted(self, tmp_path):
        """All formats declared in list_formats() must be accepted by quantize()."""
        from qwodel.backends.gguf import GGUFQuantizer
        model = _make_gguf_file(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        for fmt in GGUFQuantizer.list_formats():
            fake_output = out_dir / f"model-{fmt.lower()}.gguf"
            fake_output.parent.mkdir(parents=True, exist_ok=True)
            with patch.object(q, "_check_dependencies"), \
                 patch.object(q, "_run_quantization",
                              side_effect=lambda f=fake_output: f.touch()), \
                 patch.object(q, "get_output_path", return_value=fake_output):
                result = q.quantize(fmt)
            assert result == fake_output


# ---------------------------------------------------------------------------
# AWQ backend
# ---------------------------------------------------------------------------

class TestAWQQuantizationFlow:
    """Test the AWQ quantization execution path with mocks."""

    def _make_quantizer(self, model_path: Path, output_dir: Path):
        from qwodel.backends.awq import AWQQuantizer
        # Patch torch import so the backend loads even without CUDA
        with patch("qwodel.backends.awq._AWQ_AVAILABLE", True), \
             patch("torch.cuda.is_available", return_value=False):
            q = AWQQuantizer(
                model_path=str(model_path),
                output_dir=str(output_dir),
            )
        return q

    def test_quantize_calls_run_quantization(self, tmp_path):
        """quantize() must exercise _run_quantization() for AWQ."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        fake_output = out_dir
        fake_output.mkdir(parents=True, exist_ok=True)

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization"), \
             patch.object(q, "get_output_path", return_value=fake_output):
            result = q.quantize("INT4")

        assert result == fake_output

    def test_invalid_format_raises_error(self, tmp_path):
        """Unsupported AWQ format must raise FormatNotSupportedError."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with pytest.raises(FormatNotSupportedError):
            q.quantize("Q4_K_M")  # GGUF format, not valid for AWQ

    def test_dependency_error_propagates(self, tmp_path):
        """DependencyError raised by _check_dependencies must surface cleanly."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with patch.object(q, "_check_dependencies",
                          side_effect=DependencyError("pip install qwodel[awq]")):
            with pytest.raises(DependencyError):
                q.quantize("INT4")

    def test_progress_callback_fires(self, tmp_path):
        """Progress callback must be invoked during AWQ quantize()."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        callback = MagicMock()
        q.progress_callback = callback

        fake_output = out_dir
        fake_output.mkdir(parents=True, exist_ok=True)

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization"), \
             patch.object(q, "get_output_path", return_value=fake_output):
            q.quantize("INT4")

        assert callback.call_count >= 2

    def test_valid_awq_formats_accepted(self, tmp_path):
        """All formats declared in AWQQuantizer.list_formats() must be accepted."""
        from qwodel.backends.awq import AWQQuantizer
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        for fmt in AWQQuantizer.list_formats():
            fake_output = out_dir
            fake_output.mkdir(parents=True, exist_ok=True)
            with patch.object(q, "_check_dependencies"), \
                 patch.object(q, "_run_quantization"), \
                 patch.object(q, "get_output_path", return_value=fake_output):
                result = q.quantize(fmt)
            assert result == fake_output


# ---------------------------------------------------------------------------
# CoreML backend
# ---------------------------------------------------------------------------

class TestCoreMLQuantizationFlow:
    """Test the CoreML quantization execution path with mocks."""

    def _make_quantizer(self, model_path: Path, output_dir: Path):
        from qwodel.backends.coreml import CoreMLQuantizer
        with patch("qwodel.backends.coreml._COREML_AVAILABLE", True):
            q = CoreMLQuantizer(
                model_path=str(model_path),
                output_dir=str(output_dir),
            )
        return q

    def test_quantize_calls_run_quantization(self, tmp_path):
        """quantize() must exercise _run_quantization() for CoreML."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        fake_output = out_dir / "mock_model_float16.mlpackage"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.touch()

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization"), \
             patch.object(q, "get_output_path", return_value=fake_output):
            result = q.quantize("FLOAT16")

        assert result == fake_output

    def test_invalid_format_raises_error(self, tmp_path):
        """Unsupported CoreML format must raise FormatNotSupportedError."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with pytest.raises(FormatNotSupportedError):
            q.quantize("Q4_K_M")  # GGUF format, not valid for CoreML

    def test_dependency_error_propagates(self, tmp_path):
        """DependencyError from _check_dependencies must surface cleanly."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        with patch.object(q, "_check_dependencies",
                          side_effect=DependencyError("pip install qwodel[coreml]")):
            with pytest.raises(DependencyError):
                q.quantize("FLOAT16")

    def test_progress_callback_fires(self, tmp_path):
        """Progress callback must be invoked during CoreML quantize()."""
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        callback = MagicMock()
        q.progress_callback = callback

        fake_output = out_dir / "mock_model_float16.mlpackage"
        fake_output.parent.mkdir(parents=True, exist_ok=True)
        fake_output.touch()

        with patch.object(q, "_check_dependencies"), \
             patch.object(q, "_run_quantization"), \
             patch.object(q, "get_output_path", return_value=fake_output):
            q.quantize("FLOAT16")

        assert callback.call_count >= 2

    def test_valid_coreml_formats_accepted(self, tmp_path):
        """All formats in CoreMLQuantizer.list_formats() must be accepted."""
        from qwodel.backends.coreml import CoreMLQuantizer
        model = _make_hf_model(tmp_path)
        out_dir = tmp_path / "out"
        q = self._make_quantizer(model, out_dir)

        for fmt in CoreMLQuantizer.list_formats():
            fake_output = out_dir / f"model_{fmt}.mlpackage"
            fake_output.parent.mkdir(parents=True, exist_ok=True)
            fake_output.touch()
            with patch.object(q, "_check_dependencies"), \
                 patch.object(q, "_run_quantization"), \
                 patch.object(q, "get_output_path", return_value=fake_output):
                result = q.quantize(fmt.upper().replace("-", "_"))
            assert result == fake_output
