# Example: Progress Callback

"""
This example demonstrates using a custom progress callback
to track quantization progress.
"""

from qwodel import Quantizer


def custom_progress(percent: int, stage: str, message: str = ""):
    """Custom progress callback with detailed logging."""
    bar_length = 40
    filled = int(bar_length * percent / 100)
    bar = "█" * filled + "░" * (bar_length - filled)
    
    print(f"\r[{bar}] {percent}% | {stage}", end="", flush=True)
    
    if message:
        print(f" - {message}", end="")
    
    if percent == 100:
        print()  # New line when complete


def main():
    print("🔄 Quantizing model with progress tracking...\n")
    
    quantizer = Quantizer(
        backend="gguf",
        model_path="meta-llama/Llama-2-7b-hf",
        output_dir="./quantized",
        progress_callback=custom_progress
    )
    
    output = quantizer.quantize(format="Q4_K_M")
    
    print(f"\n✅ Complete! Output: {output}")
    
    # Get model info
    info = quantizer.get_model_info()
    print(f"📊 Size: {info['file_size'] / 1024 / 1024:.2f} MB")


if __name__ == "__main__":
    main()
