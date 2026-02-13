"""
Quickstart: GGUF Quantization

This example shows how to quantize a model using GGUF Q4_K_M format.

Installation:
    pip install -e /home/chin2/workspace/qwodel[gguf]

Or for all backends:
    pip install -e /home/chin2/workspace/qwodel[all]
"""

from qwodel import Quantizer


def main():
    print("🔄 GGUF Quantization - Q4_K_M Format")
    print("=" * 50)
    
    # Configure quantization
    model_path = "/home/chin2/workspace/qwen0.6"
    output_dir = "/home/chin2/workspace/qwodel/output"
    
    print(f"\nModel: {model_path}")
    print(f"Output: {output_dir}")
    print(f"Format: Q4_K_M (recommended)\n")
    
    # Create quantizer with progress tracking
    def show_progress(percent, stage, message):
        bar_length = 40
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"\r[{bar}] {percent}% | {stage}", end="", flush=True)
        if percent == 100:
            print()
    
    quantizer = Quantizer(
        backend="gguf",
        model_path=model_path,
        output_dir=output_dir,
        progress_callback=show_progress
    )
    
    # Quantize!
    output_path = quantizer.quantize(format="Q4_K_M")
    
    # Show results
    print(f"\n✅ Success!")
    print(f"📁 Quantized model: {output_path}")
    
    # Get model info
    info = quantizer.get_model_info()
    if info.get("file_size"):
        size_mb = info["file_size"] / (1024 * 1024)
        print(f"📦 File size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
