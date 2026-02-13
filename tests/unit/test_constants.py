"""
Unit Tests for Core Constants

Tests all constants, enums, and mappings.
"""

import pytest
from qwodel.core.constants import (
    QuantizationBackend,
    ModelFormat,
    GGUFFormat,
    AWQFormat,
    CoreMLFormat,
    FILE_EXTENSION_MAP,
    GGUF_FORMAT_DESCRIPTIONS,
    COREML_FORMAT_DESCRIPTIONS,
)


class TestEnums:
    """Test enum definitions."""
    
    def test_quantization_backend_values(self):
        """Test QuantizationBackend enum values."""
        assert QuantizationBackend.AWQ.value == "awq"
        assert QuantizationBackend.GGUF.value == "gguf"
        assert QuantizationBackend.COREML.value == "coreml"
    
    def test_model_format_values(self):
        """Test ModelFormat enum values."""
        assert ModelFormat.GGUF.value == "gguf"
        assert ModelFormat.HUGGINGFACE.value == "huggingface"
        assert ModelFormat.PYTORCH.value == "pytorch"
    
    def test_gguf_format_values(self):
        """Test GGUFFormat enum has all expected formats."""
        expected_formats = [
            "Q4_K_M", "Q8_0", "Q2_K", "Q5_K_M", "Q4_0",
            "Q6_K", "Q3_K_M", "Q4_K_S", "Q5_K_S", "IQ4_NL", "IQ3_M"
        ]
        for fmt in expected_formats:
            assert hasattr(GGUFFormat, fmt)
    
    def test_awq_format_values(self):
        """Test AWQFormat enum."""
        assert AWQFormat.INT4.value == "int4"
    
    def test_coreml_format_values(self):
        """Test CoreMLFormat enum."""
        assert CoreMLFormat.FLOAT16.value == "float16"
        assert CoreMLFormat.INT8_LINEAR.value == "int8_linear"
        assert CoreMLFormat.INT4.value == "int4"


class TestMappings:
    """Test mapping dictionaries."""
    
    def test_file_extension_map(self):
        """Test file extension to format mapping."""
        assert FILE_EXTENSION_MAP['.gguf'] == ModelFormat.GGUF
        assert FILE_EXTENSION_MAP['.safetensors'] == ModelFormat.HUGGINGFACE
        assert FILE_EXTENSION_MAP['.bin'] == ModelFormat.HUGGINGFACE
        assert FILE_EXTENSION_MAP['.pt'] == ModelFormat.PYTORCH
    
    def test_gguf_descriptions(self):
        """Test all GGUF formats have descriptions."""
        for fmt in GGUFFormat:
            assert fmt in GGUF_FORMAT_DESCRIPTIONS
            assert len(GGUF_FORMAT_DESCRIPTIONS[fmt]) > 0
    
    def test_coreml_descriptions(self):
        """Test all CoreML formats have descriptions."""
        for fmt in CoreMLFormat:
            assert fmt in COREML_FORMAT_DESCRIPTIONS
            assert len(COREML_FORMAT_DESCRIPTIONS[fmt]) > 0
