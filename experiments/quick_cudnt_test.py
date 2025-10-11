#!/usr/bin/env python3
"""
Quick CUDNT Performance Test
"""

import numpy as np
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

def quick_performance_test():
    """Quick test of CUDNT GPU virtualization performance."""
    print("...")
    print("...")

    # Test 1: Basic tensor operations
    print("...")
    size = (1000, 1000)

    # NumPy baseline
    a = np.random.rand(*size)
    b = np.random.rand(*size)

    print("...")
    start = time.time()
    for _ in range(10):
        c = a + b
    numpy_time = (time.time() - start) / 10
    print("...")
    # CUDNT test
    try:
        from cudnt_enhanced_integration import create_enhanced_cudnt
        cudnt = create_enhanced_cudnt()

        print("...")
        start = time.time()
        for _ in range(10):
            c = cudnt.tensor_add(a, b)
        cudnt_time = (time.time() - start) / 10
        print("...")
        speedup = numpy_time / cudnt_time
        print("...")
    except Exception as e:
        print(f"   ❌ CUDNT failed: {e}")
        return False

    # Test 2: Matrix multiplication
    print("...")
    m, k, n = 500, 500, 500

    a = np.random.rand(m, k)
    b = np.random.rand(k, n)

    print("...")
    start = time.time()
    for _ in range(3):
        c = a @ b
    numpy_time = (time.time() - start) / 3
    print("...")
    print("...")
    start = time.time()
    for _ in range(3):
        c = cudnt.matrix_multiply(a, b)
    cudnt_time = (time.time() - start) / 3
    print("...")
    speedup = numpy_time / cudnt_time
    print("...")
    # Calculate GFLOPS
    operations = 2 * m * k * n  # multiply-add operations
    gflops_numpy = operations / (numpy_time * 1e9)
    gflops_cudnt = operations / (cudnt_time * 1e9)
    print("...")
    # Test 3: Neural network operations
    print("...")
    batch_size, features = 1000, 512
    tensor = np.random.rand(batch_size, features)

    print("...")
    start = time.time()
    result = cudnt.relu(tensor)
    relu_time = time.time() - start
    print("...")
    print("...")
    start = time.time()
    result = cudnt.batch_normalize(tensor)
    bn_time = time.time() - start
    print("...")
    print("...")
    # Test 4: ML pipeline
    print("...")
    try:
        # Generate simple data
        X = np.random.rand(1000, 10)
        y = np.random.rand(1000, 1)

        params = {'weights': np.random.rand(10, 1), 'bias': np.zeros(1)}

        print("...")
        start = time.time()

        for epoch in range(5):
            # Forward pass
            pred = cudnt.matrix_multiply(X, params['weights']) + params['bias']
            pred = cudnt.relu(pred)

            # Simple gradients
            error = pred - y
            dW = cudnt.matrix_multiply(X.T, error) / 1000
            db = np.mean(error, axis=0)

            # Update
            params = cudnt.gradient_step(params, {'weights': dW, 'bias': db}, 0.01)

        training_time = time.time() - start
        print("...")
        # Test 5: GPU stats
        print("...")
        if hasattr(cudnt, 'gpu_virtualizer'):
            stats = cudnt.gpu_virtualizer.get_gpu_stats()
            print(f"   Virtual GPU threads: {stats['virtual_gpu_threads']}")
            print(f"   Operations performed: {stats['operations_performed']}")
            print("...")
    except Exception as e:
        print(f"   ❌ ML pipeline failed: {e}")

    print("...")
    print("...")
    print("...")
    print("...")
    print("...")
    print("...")
    print("...")

    return True

if __name__ == "__main__":
    success = quick_performance_test()
    if not success:
        sys.exit(1)
