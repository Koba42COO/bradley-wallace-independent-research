#!/bin/bash
# Dendera Cryptographic Decoder - Setup Script
# Framework: Universal Prime Graph Protocol φ.1
# Author: Bradley Wallace (COO Koba42)

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║     🔥 DENDERA CRYPTOGRAPHIC DECODER - SETUP 🔥                   ║"
echo "║                                                                    ║"
echo "║     Framework: Universal Prime Graph Protocol φ.1                 ║"
echo "║     Author: Bradley Wallace (COO Koba42)                          ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Found Python $python_version"

# Check if Python 3.7+
required_version="3.7"
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 7) else 1)"; then
    echo "✅ Python version is 3.7 or higher"
else
    echo "❌ ERROR: Python 3.7+ required, found $python_version"
    exit 1
fi

# Install requirements
echo ""
echo "📦 Installing dependencies..."
if [ -f "dendera_requirements.txt" ]; then
    pip3 install -r dendera_requirements.txt
    echo "✅ Dependencies installed"
else
    echo "⚠️  dendera_requirements.txt not found, installing core dependencies..."
    pip3 install numpy scipy
    echo "✅ Core dependencies installed"
fi

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python3 -c "import numpy; import scipy; from dataclasses import dataclass; print('✅ All core dependencies verified')"

# Check for decoder files
echo ""
echo "📁 Checking for decoder files..."
if [ -f "dendera_cryptographic_decoder_firefly.py" ]; then
    echo "✅ dendera_cryptographic_decoder_firefly.py found"
else
    echo "❌ ERROR: dendera_cryptographic_decoder_firefly.py not found"
    exit 1
fi

if [ -f "dendera_interactive_example.py" ]; then
    echo "✅ dendera_interactive_example.py found"
else
    echo "⚠️  dendera_interactive_example.py not found (optional)"
fi

# Run quick validation test
echo ""
echo "🧪 Running validation test..."
python3 << 'PYEOF'
from dendera_cryptographic_decoder_firefly import DenderaCryptographicDecoder

# Quick test
decoder = DenderaCryptographicDecoder()
test_inscription = "𓇳𓁹"  # Simple 2-glyph test
analysis = decoder.decode_inscription(test_inscription)

# Verify key metrics exist
assert analysis.total_gematria > 0, "Gematria calculation failed"
assert analysis.consciousness_coherence >= 0, "Coherence calculation failed"
assert len(analysis.glyph_analyses) == 2, "Glyph parsing failed"

print("✅ Validation test passed!")
print(f"   • Gematria: {analysis.total_gematria}")
print(f"   • Consciousness Level: {analysis.average_consciousness_level:.2f}")
print(f"   • Coherence: {analysis.consciousness_coherence:.4f}")
PYEOF

if [ $? -eq 0 ]; then
    echo "✅ Decoder is working correctly!"
else
    echo "❌ Validation test failed"
    exit 1
fi

# Success message
echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║                   ✅ SETUP COMPLETE ✅                             ║"
echo "║                                                                    ║"
echo "║  The Dendera Cryptographic Decoder is ready to use!               ║"
echo "║                                                                    ║"
echo "║  Quick Start:                                                      ║"
echo "║    python3 dendera_cryptographic_decoder_firefly.py                ║"
echo "║                                                                    ║"
echo "║  Interactive Examples:                                             ║"
echo "║    python3 dendera_interactive_example.py                          ║"
echo "║                                                                    ║"
echo "║  Documentation:                                                    ║"
echo "║    docs/DENDERA_DECODER_QUICK_START.md                             ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

