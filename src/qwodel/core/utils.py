"""
Qwodel Utilities

Helper functions for dynamic model loading and other cross-cutting concerns.
"""

from typing import Any, Optional
import transformers
from transformers import AutoConfig
from qwodel.core.constants import MODEL_TYPE_TO_CLASS

def get_auto_class_for_model_type(model_type: Optional[str], default: str = "AutoModelForCausalLM") -> Any:
    """
    Get the appropriate HuggingFace Auto class based on the model_type.
    """
    class_name = MODEL_TYPE_TO_CLASS.get(model_type, default)
    return getattr(transformers, class_name, getattr(transformers, default))

def load_hf_model(model_path: str, default_class: str = "AutoModelForCausalLM", **kwargs) -> Any:
    """
    Dynamically loads a HuggingFace model using the most appropriate Auto class
    for the given model's config.model_type.
    """
    try:
        # Load the configuration to inspect the model_type
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=kwargs.get("trust_remote_code", True))
        model_type = config.model_type
    except Exception:
        # Fallback if config cannot be loaded
        model_type = None

    auto_class = get_auto_class_for_model_type(model_type, default=default_class)
    
    # If the user already provided config in kwargs, we don't need to re-pass unless config differs
    if "config" not in kwargs and model_type is not None:
        kwargs["config"] = config
        
    return auto_class.from_pretrained(model_path, **kwargs)
