#!/bin/bash
# Setup script for AIVA benchmark testing environment

echo "🧠 Setting up AIVA Benchmark Testing Environment"
echo "=================================================="
echo ""

# Install required packages
echo "Installing required packages..."
pip install datasets huggingface-hub requests

echo ""
echo "✅ Benchmark environment ready!"
echo ""
echo "Available benchmarks:"
echo "  - MMLU (HuggingFace: cais/mmlu)"
echo "  - GSM8K (HuggingFace: gsm8k)"
echo "  - HumanEval (OpenAI GitHub)"
echo "  - MATH (Competition dataset)"
echo ""
echo "Run benchmarks with:"
echo "  python3 aiva_public_benchmark_integration.py"

