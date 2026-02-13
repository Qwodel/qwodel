# Example: Basic GGUF Quantization

"""
This example demonstrates basic GGUF quantization using qwodel.
"""

from qwodel import Quantizer

def main():
    # Create quantizer for GGUF backend
    quantizer = Quantizer(
        backend="gguf",
        model_path="meta-llama/Llama-2-7b-hf",  # HuggingFace model ID or local path
        output_dir="./quantized"
    )
    
    # Quantize to Q4_K_M format (recommended for most users)
    output_path = quantizer.quantize(format="Q4_K_M")
    
    print(f"✅ Quantization complete!")
    print(f"📦 Output: {output_path}")
    
    # Get model information
    info = quantizer.get_model_info()
    print(f"📊 Backend: {info['backend']}")
    print(f"📊 File size: {info['file_size'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
