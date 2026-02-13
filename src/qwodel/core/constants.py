"""
Qwodel Core Constants

Framework-wide constants, enums, and configuration values.
Migrated from bekenstein-core/core/utils/constants.py
"""

from enum import Enum
from typing import Dict


class QuantizationBackend(Enum):
    """
    Supported quantization backends.
    
    Each backend represents a different quantization framework/tool
    that can be used to quantize models.
    """
    AWQ = "awq"
    GGUF = "gguf"
    COREML = "coreml"


class ModelFormat(Enum):
    """
    Supported model input formats.
    
    Represents the format of the source model before quantization.
    Used for format detection and validation.
    """
    GGUF = "gguf"
    HUGGINGFACE = "huggingface"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    UNKNOWN = "unknown"


class GGUFFormat(Enum):
    """
    Supported GGUF quantization formats with descriptions.
    
    These formats represent different quantization strategies with
    varying trade-offs between model size, speed, and quality.
    """
    # Most Popular Formats
    Q4_K_M = "Q4_K_M"  # Best balance of speed vs. smarts. 90% of users want this.
    Q8_0 = "Q8_0"      # High Quality: Almost perfect quality for users with RAM to spare.
    Q2_K = "Q2_K"      # Maximum Shrink: For tiny devices, runs even if quality is reduced.
    Q5_K_M = "Q5_K_M"  # Better quality than Q4_K_M, slightly larger size.
    Q4_0 = "Q4_0"      # Slightly smaller than Q4_K_M, good for most use cases.
    Q6_K = "Q6_K"      # High quality between Q8_0 and Q4_K_M.
    
    # Additional K-Quant Formats
    Q3_K_M = "Q3_K_M"  # Medium 3-bit quantization, smaller size with acceptable quality.
    Q4_K_S = "Q4_K_S"  # Small 4-bit K-quant, slightly smaller than Q4_K_M.
    Q5_K_S = "Q5_K_S"  # Small 5-bit K-quant, good quality with smaller size.
    
    # IQ (Importance Quantization) Formats
    IQ4_NL = "IQ4_NL"  # 4.5 bpw non-linear quantization, good quality.
    IQ3_M = "IQ3_M"    # 3.66 bpw quantization mix, very compact.


class AWQFormat(Enum):
    """AWQ quantization formats"""
    INT4 = "int4"


class CoreMLFormat(Enum):
    """
    Supported CoreML quantization formats.
    
    CoreML provides various quantization formats for deploying models
    on Apple devices (iOS, macOS, iPadOS) with optimized performance.
    """
    FLOAT16 = "float16"                 # 16-bit half-precision, ~2x compression
    INT8_LINEAR = "int8_linear"         # 8-bit linear quantization, ~4x compression
    INT8_SYMMETRIC = "int8_symmetric"   # 8-bit symmetric linear, ~4x compression
    INT4 = "int4"                       # 4-bit quantization, ~8x compression (iOS 18+)
    INT6 = "int6"                       # 6-bit quantization, ~5x compression


# File Extension to Model Format Mapping
FILE_EXTENSION_MAP: Dict[str, ModelFormat] = {
    '.gguf': ModelFormat.GGUF,
    '.tflite': ModelFormat.TENSORFLOW,
    '.onnx': ModelFormat.ONNX,
    '.pt': ModelFormat.PYTORCH,
    '.pth': ModelFormat.PYTORCH,
    '.bin': ModelFormat.HUGGINGFACE,
    '.safetensors': ModelFormat.HUGGINGFACE,
}


# Default Configuration Values
DEFAULT_OUTPUT_DIR = "./quantized_models"
DEFAULT_TEMP_DIR = "/tmp/qwodel"
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# Quantization Format Descriptions
GGUF_FORMAT_DESCRIPTIONS: Dict[GGUFFormat, str] = {
    GGUFFormat.Q4_K_M: "Best balance of speed vs. smarts. Recommended for 90% of users.",
    GGUFFormat.Q8_0: "High Quality: Almost perfect quality for users with sufficient RAM.",
    GGUFFormat.Q2_K: "Maximum Shrink: For tiny devices, runs even with reduced quality.",
    GGUFFormat.Q5_K_M: "Better quality than Q4_K_M with slightly larger size.",
    GGUFFormat.Q4_0: "Compact and fast: Slightly smaller than Q4_K_M, good for most use cases.",
    GGUFFormat.Q6_K: "High quality: Between Q8_0 and Q4_K_M, near-perfect quality with reasonable size.",
    GGUFFormat.Q3_K_M: "Compact medium quality: 3-bit quantization with acceptable quality.",
    GGUFFormat.Q4_K_S: "Small 4-bit: Slightly smaller than Q4_K_M with good quality.",
    GGUFFormat.Q5_K_S: "Small 5-bit: Good quality with smaller size than Q5_K_M.",
    GGUFFormat.IQ4_NL: "IQ 4-bit non-linear: 4.5 bpw with importance-based quantization.",
    GGUFFormat.IQ3_M: "IQ 3-bit medium: 3.66 bpw very compact quantization mix.",
}


# CoreML Format Descriptions
COREML_FORMAT_DESCRIPTIONS: Dict[CoreMLFormat, str] = {
    CoreMLFormat.FLOAT16: "Half-precision: ~2x compression, minimal accuracy loss, universal compatibility",
    CoreMLFormat.INT8_LINEAR: "8-bit linear: ~4x compression, good accuracy, requires calibration",
    CoreMLFormat.INT8_SYMMETRIC: "8-bit symmetric: ~4x compression, simplified quantization, faster ops",
    CoreMLFormat.INT4: "4-bit: ~8x compression, iOS 18+ only, higher accuracy impact",
    CoreMLFormat.INT6: "6-bit: ~5x compression, balance between Int4 and Int8",
}


# Tool Names
LLAMA_QUANTIZE_TOOL = "llama-quantize"
LLAMA_CONVERT_TOOL = "llama-convert-hf"


# CoreML File Extensions
COREML_MODEL_EXTENSION = ".mlmodel"
COREML_PACKAGE_EXTENSION = ".mlpackage"


# HuggingFace Model Indicators
HUGGINGFACE_CONFIG_FILE = "config.json"
HUGGINGFACE_MODEL_FILE = "pytorch_model.bin"
HUGGINGFACE_VALID_EXTENSIONS = ['.bin', '.safetensors', '.ckpt']


# TensorFlow Model Indicators
TENSORFLOW_SAVED_MODEL = "saved_model.pb"


# API Configuration Constants
DEFAULT_PRESIGNED_URL_EXPIRATION = 3600  # 1 hour in seconds
DEFAULT_API_TIMEOUT = 30  # seconds
DEFAULT_STORAGE_REGION = "auto"  # Cloudflare R2 region


# HTTP Status Codes
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_SERVER_ERROR = 500
HTTP_SERVICE_UNAVAILABLE = 503
