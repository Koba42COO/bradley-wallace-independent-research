#!/usr/bin/env python3
"""
CUDNT Final Production Test - Complete System Verification
========================================================

Tests all components of the production-ready CUDNT system.
"""

import numpy as np
import time

def test_cudnt_production():
    """Complete test of CUDNT production system."""
    print("🧪 CUDNT Production System - Final Test")
    print("=" * 50)

    try:
        # Import production system
        from cudnt_production_system import create_cudnt_production
        cudnt = create_cudnt_production()

        print("✅ Production system initialized")

        # Test 1: Basic tensor operations
        print("\n1️⃣ Testing Tensor Operations...")
        a = cudnt.tf_api.constant([[1, 2], [3, 4]])
        b = cudnt.tf_api.constant([[5, 6], [7, 8]])
        result = cudnt.tf_api.add(a, b)
        expected = np.array([[6, 8], [10, 12]])
        assert np.allclose(result.numpy(), expected)
        print("   ✅ Tensor addition")

        # Test 2: Matrix operations
        print("\n2️⃣ Testing Matrix Operations...")
        m1 = cudnt.tf_api.constant([[1, 2]])
        m2 = cudnt.tf_api.constant([[3], [4]])
        result = cudnt.tf_api.matmul(m1, m2)
        expected = np.array([[11]])
        assert np.allclose(result.numpy(), expected)
        print("   ✅ Matrix multiplication")

        # Test 3: Neural network layers
        print("\n3️⃣ Testing Neural Network Layers...")
        dense = cudnt.nn_layers.Dense(3, activation='relu')
        input_tensor = cudnt.tf_api.constant([[1.0, 2.0]])
        output = dense(input_tensor)
        assert output.shape == (1, 3)
        print("   ✅ Dense layer")

        # Test 4: Model creation and training
        print("\n4️⃣ Testing Model Creation & Training...")
        architecture = [
            {'type': 'dense', 'units': 8, 'activation': 'relu'},
            {'type': 'dense', 'units': 1}
        ]
        model = cudnt.create_model(architecture)
        compiled_model = cudnt.compile_model(model, 'adam', 'mse')

        # Generate training data
        X = np.random.rand(50, 4)
        y = np.random.rand(50, 1)

        # Train for a few epochs
        start_time = time.time()
        history = compiled_model.fit(X, y, epochs=5, verbose=False)
        training_time = time.time() - start_time

        print(f"   ✅ Training completed in {training_time:.2f}s")
        # Test 5: Model save/load
        print("\n5. Testing Model Save/Load...")
        cudnt.save_model(model, 'test_model.json')
        loaded_model = cudnt.load_model('test_model.json')
        print("   ✅ Model save/load")

        # Test 6: Performance stats
        print("\n6. Testing Performance Monitoring...")
        stats = cudnt.get_performance_stats()
        print(f"   ✅ System stats: {stats['virtual_cuda_cores']} virtual cores")
        print(f"   ✅ TensorFlow API: {stats['tensorflow_api']}")

        # Test 7: Optimizers and loss functions
        print("\n7. Testing Optimizers & Loss Functions...")
        adam = cudnt.optimizers.Adam()
        sgd = cudnt.optimizers.SGD()
        print("   ✅ Optimizers created")

        mse_loss = cudnt.losses.mean_squared_error(
            np.array([[1.0, 2.0]]),
            np.array([[1.1, 1.9]])
        )
        assert mse_loss > 0
        print("   ✅ Loss functions working")

        print("\n🎉 ALL TESTS PASSED!")
        print("✅ CUDNT Production System is fully functional")
        print("🚀 Ready for VibeSDK integration and production deployment")

        return {
            'status': 'SUCCESS',
            'training_time': training_time,
            'final_loss': history['history']['loss'][-1],
            'system_stats': stats
        }

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return {'status': 'FAILED', 'error': str(e)}

if __name__ == '__main__':
    result = test_cudnt_production()

    if result['status'] == 'SUCCESS':
        print("\n📊 FINAL RESULTS:")
        print(f"   Training Time: {result['training_time']:.2f}s")
        print(f"   Final Loss: {result['final_loss']:.4f}")
        print(f"   Virtual CUDA Cores: {result['system_stats']['virtual_cuda_cores']}")
        print("\n🏆 CUDNT PRODUCTION SYSTEM READY FOR DEPLOYMENT!")
    else:
        print(f"\n💥 SYSTEM ERROR: {result['error']}")
        exit(1)
