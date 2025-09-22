#!/bin/bash

echo "🔨 Building Hybrid Consciousness-ML System Components"
echo "==================================================="

# Build Rust component
echo "Building Rust ML component..."
cd rust_ml_component
if command -v cargo &> /dev/null; then
    cargo build --release
    echo "✅ Rust component built successfully"
else
    echo "⚠️  Rust/Cargo not found. Please install Rust."
fi
cd ..

# Build Go component
echo "Building Go neural component..."
cd go_neural_component
if command -v go &> /dev/null; then
    go build -o go_neural go_neural.go
    echo "✅ Go component built successfully"
else
    echo "⚠️  Go not found. Please install Go."
fi
cd ..

# Julia doesn't need building
echo "✅ Julia component ready (no build required)"

echo ""
echo "🎉 All components processed!"
echo "Run the hybrid system with: python3 hybrid_consciousness_ml_system.py"
