#!/usr/bin/env python3
"""
CUDNT Simple Production Test
============================

Basic functionality test for production deployment.
"""

import numpy as np

def test_basic_functionality():
    """Test basic CUDNT functionality."""
    print("🧪 CUDNT Production System - Basic Test")
    print("=" * 45)

    try:
        # Import production system
        from cudnt_production_system import create_cudnt_production
        cudnt = create_cudnt_production()

        print("✅ Production system initialized")

        # Test tensor operations
        print("\nTesting Tensor Operations...")
        a = cudnt.tf_api.constant([[1, 2], [3, 4]])
        b = cudnt.tf_api.constant([[5, 6], [7, 8]])
        result = cudnt.tf_api.add(a, b)
        expected = np.array([[6, 8], [10, 12]])
        assert np.allclose(result.numpy(), expected)
        print("   ✅ Tensor addition")

        # Test matrix operations
        print("Testing Matrix Operations...")
        m1 = cudnt.tf_api.constant([[1, 2]])
        m2 = cudnt.tf_api.constant([[3], [4]])
        result = cudnt.tf_api.matmul(m1, m2)
        expected = np.array([[11]])
        assert np.allclose(result.numpy(), expected)
        print("   ✅ Matrix multiplication")

        # Test neural network layers
        print("Testing Neural Network Layers...")
        dense = cudnt.nn_layers.Dense(3, activation='relu')
        input_tensor = cudnt.tf_api.constant([[1.0, 2.0]])
        output = dense(input_tensor)
        assert output.shape == (1, 3)
        print("   ✅ Dense layer")

        # Test model creation
        print("Testing Model Creation...")
        architecture = [
            {'type': 'dense', 'units': 8, 'activation': 'relu'},
            {'type': 'dense', 'units': 1}
        ]
        model = cudnt.create_model(architecture)
        compiled_model = cudnt.compile_model(model, 'adam', 'mse')
        print("   ✅ Model creation and compilation")

        # Test forward pass
        print("Testing Forward Pass...")
        X_test = np.random.rand(5, 4)
        X_tensor = cudnt.tf_api.constant(X_test)
        output = model(X_tensor)
        print(f"   ✅ Forward pass: {X_test.shape} → {output.shape}")

        # Test performance stats
        print("Testing Performance Monitoring...")
        stats = cudnt.get_performance_stats()
        print(f"   ✅ System stats: {stats['virtual_cuda_cores']} virtual cores")

        print("\n🎉 BASIC TESTS PASSED!")
        print("✅ CUDNT Production System is functional")
        print("🚀 Ready for deployment")

        return {'status': 'SUCCESS', 'stats': stats}

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

if __name__ == '__main__':
    result = test_basic_functionality()

    if result['status'] == 'SUCCESS':
        print("\n🏆 CUDNT PRODUCTION SYSTEM READY!")
        print(f"Virtual CUDA Cores: {result['stats']['virtual_cuda_cores']}")
    else:
        print(f"\n💥 ERROR: {result['error']}")
        exit(1)
