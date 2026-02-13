# Example: Batch Processing

"""
This example demonstrates batch processing multiple models.
"""

from qwodel import quantize

def main():
    # List of models to quantize
    models = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "meta-llama/Llama-2-7b-hf",
    ]
    
    # Quantize each model
    for model_path in models:
        print(f"\n🔄 Quantizing {model_path}...")
        
        try:
            output = quantize(
                model_path=model_path,
                backend="gguf",
                format="Q4_K_M",
                output_dir="./quantized"
            )
            print(f"✅ Success: {output}")
        except Exception as e:
            print(f"❌ Failed: {e}")


if __name__ == "__main__":
    main()
