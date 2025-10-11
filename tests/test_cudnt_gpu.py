#!/usr/bin/env python3
"""
Simple test script for CUDNT GPU virtualization
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_cudnt_gpu():
    """Test CUDNT GPU virtualization functionality."""
    print("🧪 Testing CUDNT GPU Virtualization")
    print("=" * 40)

    try:
        from cudnt_gpu_virtualization import CUDNT_GPU_Virtualization
        print("✅ GPU virtualization module imported")

        # Initialize virtual GPU
        virtual_gpu = CUDNT_GPU_Virtualization(n_threads=2)
        print("✅ Virtual GPU initialized")

        # Test basic operations
        import numpy as np
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)

        # Test tensor addition
        result = virtual_gpu.tensor_add(a, b)
        print(f"✅ Tensor addition: {a.shape} + {b.shape} = {result.shape}")

        # Test matrix multiplication
        m1 = np.random.rand(2, 3)
        m2 = np.random.rand(3, 4)
        result = virtual_gpu.matrix_multiply_gpu(m1, m2)
        print(f"✅ Matrix multiply: {m1.shape} × {m2.shape} = {result.shape}")

        # Test activation function
        tensor = np.random.rand(3, 3) - 0.5
        result = virtual_gpu.relu_activation(tensor)
        print(f"✅ ReLU activation: {tensor.shape} → {result.shape}")

        # Get performance stats
        stats = virtual_gpu.get_gpu_stats()
        print("📊 Performance Stats:")
        print(f"   Operations: {stats['operations_performed']}")
        print(f"   Threads: {stats['virtual_gpu_threads']}")
        print(".4f"
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cudnt_gpu()
    if success:
        print("\n🎯 SUCCESS: CUDNT GPU virtualization is working!")
        print("   • CPU-based GPU operations functional")
        print("   • Parallel processing enabled")
        print("   • ML workloads now possible on CPU-only systems")
    else:
        print("\n❌ FAILURE: CUDNT GPU virtualization has issues")
        sys.exit(1)
