#!/bin/bash
# Sample script to install qwodel and quantize a model

echo "🚀 Installing qwodel with GGUF backend..."

# Install qwodel with GGUF support
pip install -e "/home/chin2/workspace/qwodel[gguf]"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📊 Running GGUF quantization on qwen0.6..."
echo ""

# Run quantization using Python
python3 << 'EOF'
from qwodel import Quantizer

print("🔄 Starting quantization...")

# Create quantizer
quantizer = Quantizer(
    backend="gguf",
    model_path="/home/chin2/workspace/qwen0.6",
    output_dir="/home/chin2/workspace/qwodel/output"
)

# Quantize with Q4_K_M format
output_path = quantizer.quantize(format="Q4_K_M")

print(f"\n✅ Quantization complete!")
print(f"📁 Output: {output_path}")

# Get model info
info = quantizer.get_model_info()
if info.get("file_size"):
    size_mb = info["file_size"] / (1024 * 1024)
    print(f"📦 Size: {size_mb:.2f} MB")
EOF

echo ""
echo "🎉 Done!"
