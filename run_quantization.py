#!/usr/bin/env python3
"""
Direct Installation & Quantization Example

This script will:
1. Verify qwodel installation
2. Quantize /home/chin2/workspace/qwen0.6 using GGUF Q4_K_M
3. Show results

Usage:
    python run_quantization.py
"""

import sys
import subprocess
from pathlib import Path


def check_installation():
    """Check if qwodel is installed."""
    try:
        import qwodel
        print(f"✅ qwodel {qwodel.__version__} is installed")
        return True
    except ImportError:
        print("❌ qwodel is not installed")
        print("\nPlease install with:")
        print("  cd /home/chin2/workspace/qwodel")
        print("  pip install -e .[gguf]")
        return False


def run_quantization():
    """Run GGUF quantization on qwen0.6 model."""
    from qwodel import Quantizer
    
    # Configuration
    model_path = "/home/chin2/workspace/qwen0.6"
    output_dir = "/home/chin2/workspace/qwodel/output"
    format_name = "Q4_K_M"
    
    print("\n" + "="*60)
    print("🔄 GGUF Quantization")
    print("="*60)
    print(f"Model:  {model_path}")
    print(f"Output: {output_dir}")
    print(f"Format: {format_name}")
    print("="*60 + "\n")
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Error: Model path not found: {model_path}")
        return False
    
    # Progress callback
    def show_progress(percent, stage, message=""):
        # Create progress bar
        bar_length = 50
        filled = int(bar_length * percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Format stage name
        stage_name = stage.replace("_", " ").title()
        
        # Print progress
        print(f"\r[{bar}] {percent:3d}% | {stage_name:<20}", end="", flush=True)
        
        if message and percent % 10 == 0:  # Print messages at intervals
            print(f" - {message}", end="")
        
        if percent == 100:
            print()  # New line when complete
    
    try:
        # Create quantizer
        quantizer = Quantizer(
            backend="gguf",
            model_path=model_path,
            output_dir=output_dir,
            progress_callback=show_progress
        )
        
        # Run quantization
        output_path = quantizer.quantize(format=format_name)
        
        # Success!
        print("\n" + "="*60)
        print("✅ Quantization Complete!")
        print("="*60)
        print(f"📁 Output: {output_path}")
        
        # Get model info
        info = quantizer.get_model_info()
        if info.get("file_size"):
            size_mb = info["file_size"] / (1024 * 1024)
            print(f"📦 Size:   {size_mb:.2f} MB")
        
        print("="*60 + "\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    print("\n🚀 Qwodel Quantization Tool\n")
    
    # Check installation
    if not check_installation():
        sys.exit(1)
    
    # Run quantization
    success = run_quantization()
    
    if success:
        print("🎉 All done!")
        sys.exit(0)
    else:
        print("\n⚠️ Quantization failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
