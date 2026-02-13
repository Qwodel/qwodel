"""
Test Configuration and Fixtures

Provides common fixtures and configuration for all tests.
"""

import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    # Cleanup
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture
def mock_model_path(temp_dir):
    """Create a mock HuggingFace model directory."""
    model_dir = temp_dir / "mock_model"
    model_dir.mkdir()
    
    # Create config.json
    import json
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": 32000
    }
    with open(model_dir / "config.json", "w") as f:
        json.dump(config, f)
    
    # Create a dummy model file
    (model_dir / "pytorch_model.bin").touch()
    
    return model_dir


@pytest.fixture
def mock_gguf_path(temp_dir):
    """Create a mock GGUF file."""
    gguf_file = temp_dir / "model.gguf"
    gguf_file.touch()
    return gguf_file
