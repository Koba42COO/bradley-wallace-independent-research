#!/usr/bin/env python3
"""
Simple CUDNT Test - Debug broadcasting issues
"""

import numpy as np
from cudnt_production_system import create_cudnt_production

def test_dense_layer():
    """Test Dense layer broadcasting."""
    cudnt = create_cudnt_production()

    # Create a simple dense layer
    dense = cudnt.nn_layers.Dense(10, activation='relu')

    # Create input
    x = cudnt.tf_api.constant(np.random.rand(32, 64))  # batch_size=32, input_dim=64

    try:
        output = dense(x)
        print(f"✅ Dense layer worked: input {x.shape} -> output {output.shape}")
        return True
    except Exception as e:
        print(f"❌ Dense layer failed: {e}")
        return False

def test_simple_matmul():
    """Test simple matrix multiplication."""
    cudnt = create_cudnt_production()

    a = cudnt.tf_api.constant(np.random.rand(32, 64))
    b = cudnt.tf_api.constant(np.random.rand(64, 10))

    try:
        result = cudnt.tf_api.matmul(a, b)
        print(f"✅ Matmul worked: {a.shape} @ {b.shape} = {result.shape}")
        return True
    except Exception as e:
        print(f"❌ Matmul failed: {e}")
        return False

def test_bias_broadcasting():
    """Test bias broadcasting."""
    cudnt = create_cudnt_production()

    output = cudnt.tf_api.constant(np.random.rand(32, 10))  # batch_size=32, output_dim=10
    bias = cudnt.tf_api.constant(np.random.rand(10))  # bias shape should be (10,)

    try:
        # Try to broadcast bias
        batch_size = output.shape[0]
        bias_reshaped = cudnt.tf_api.reshape(bias, (1, 10))
        bias_broadcasted = cudnt.tf_api.tile(bias_reshaped, (batch_size, 1))
        result = cudnt.tf_api.add(output, bias_broadcasted)
        print(f"✅ Bias broadcasting worked: output {output.shape} + bias {bias.shape} -> {result.shape}")
        return True
    except Exception as e:
        print(f"❌ Bias broadcasting failed: {e}")
        return False

if __name__ == '__main__':
    print("🔧 CUDNT Simple Debug Test")
    print("=" * 30)

    test_simple_matmul()
    test_bias_broadcasting()
    test_dense_layer()
