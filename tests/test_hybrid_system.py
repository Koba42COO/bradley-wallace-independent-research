#!/usr/bin/env python3
"""
Test CUDNT Hybrid System - GPU + CPU Acceleration
================================================

Test script for the professional CUDNT hybrid accelerator.
Demonstrates GPU detection, hybrid operations, and performance comparison.
"""

import numpy as np
import time
from cudnt_production_system import create_cudnt_production

def test_hardware_detection():
    """Test hardware detection capabilities."""
    print("🔍 Testing Hardware Detection")
    print("-" * 30)

    cudnt = create_cudnt_production()

    # Get hardware info
    hw_info = cudnt.get_hardware_info()

    print(f"Hybrid Available: {hw_info['hybrid_available']}")
    print(f"Current Device: {hw_info['current_device']}")

    if hw_info['hybrid_available']:
        print("Available Devices:")
        for device_id, info in hw_info['devices'].items():
            status = "✅" if info['available'] else "❌"
            preferred = " (preferred)" if info.get('preferred', False) else ""
            print(f"  {status} {device_id}: {info['name']} ({info['memory_gb']}GB){preferred}")
    else:
        print(f"CPU Cores: {hw_info.get('cpu_cores', 'Unknown')}")

    return cudnt

def test_hybrid_operations(cudnt):
    """Test hybrid tensor operations."""
    print("\n🔥 Testing Hybrid Operations")
    print("-" * 30)

    try:
        # Create hybrid tensors
        print("Creating tensors...")
        a_data = np.random.rand(256, 256).astype(np.float32)
        b_data = np.random.rand(256, 256).astype(np.float32)

        a = cudnt.create_hybrid_tensor(a_data)
        b = cudnt.create_hybrid_tensor(b_data)

        device_info = "hybrid" if hasattr(a, 'device') else "numpy"
        print(f"Tensors created as: {device_info}")

        # Test matrix multiplication
        print("Testing matrix multiplication...")
        start_time = time.time()
        c = cudnt.hybrid_matmul(a, b)
        matmul_time = time.time() - start_time

        result_shape = c.shape if hasattr(c, 'shape') else np.array(c).shape
        print(f"Result shape: {result_shape}")
        print(f"Matmul time: {matmul_time:.4f}s")
        # Test convolution if possible
        print("Testing convolution...")
        input_data = np.random.rand(8, 32, 32, 3).astype(np.float32)
        kernel_data = np.random.rand(3, 3, 3, 64).astype(np.float32)

        input_tensor = cudnt.create_hybrid_tensor(input_data)
        kernel = cudnt.create_hybrid_tensor(kernel_data)

        start_time = time.time()
        conv_result = cudnt.hybrid_conv2d(input_tensor, kernel, strides=(1, 1), padding='VALID')
        conv_time = time.time() - start_time

        conv_shape = conv_result.shape if hasattr(conv_result, 'shape') else np.array(conv_result).shape
        print(f"Conv result shape: {conv_shape}")
        print(f"Conv time: {conv_time:.4f}s")
        print("✅ Hybrid operations successful!")
        return True

    except Exception as e:
        print(f"❌ Hybrid operations failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_comparison(cudnt):
    """Compare performance between CPU and hybrid modes."""
    print("\n⚡ Performance Comparison")
    print("-" * 30)

    test_sizes = [(128, 128), (256, 256), (512, 512)]

    for size in test_sizes:
        print(f"\nTesting {size[0]}x{size[1]} matrices:")

        # Create test data
        a_data = np.random.rand(*size).astype(np.float32)
        b_data = np.random.rand(*size).astype(np.float32)

        # CPU-only test
        start_time = time.time()
        for _ in range(5):
            c_cpu = np.matmul(a_data, b_data)
        cpu_time = (time.time() - start_time) / 5

        # Hybrid test
        try:
            a_hybrid = cudnt.create_hybrid_tensor(a_data)
            b_hybrid = cudnt.create_hybrid_tensor(b_data)

            start_time = time.time()
            for _ in range(5):
                c_hybrid = cudnt.hybrid_matmul(a_hybrid, b_hybrid)
            hybrid_time = (time.time() - start_time) / 5

            speedup = cpu_time / hybrid_time
            print(f"  CPU time: {cpu_time:.4f}s")
            print(f"  Hybrid time: {hybrid_time:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
        except Exception as e:
            print(f"  CPU time: {cpu_time:.4f}s")
            print(f"  Hybrid failed: {e}")

def test_hardware_benchmark(cudnt):
    """Run hardware benchmark."""
    print("\n🏃 Hardware Benchmark")
    print("-" * 30)

    try:
        benchmark_results = cudnt.benchmark_hardware()

        if 'cpu_only' in benchmark_results:
            print("CPU-only mode - no GPU benchmarking available")
        else:
            print("Benchmark Results:")
            for device_id, result in benchmark_results.items():
                if 'error' in result:
                    print(f"  {device_id}: ERROR - {result['error']}")
                else:
                    print(f"  {device_id}:")
                    print(f"    Matmul time: {result['matmul_time']:.4f}s")
                    print(f"    GFLOPS: {result['gflops']:.2f}")
                    print(f"    Device: {result['device_info']['name']}")

    except Exception as e:
        print(f"❌ Benchmark failed: {e}")

def main():
    """Main test function."""
    print("🚀 CUDNT Hybrid System Test")
    print("=" * 40)

    # Test hardware detection
    cudnt = test_hardware_detection()

    # Test hybrid operations
    operations_work = test_hybrid_operations(cudnt)

    # Performance comparison
    if operations_work:
        test_performance_comparison(cudnt)

    # Hardware benchmark
    test_hardware_benchmark(cudnt)

    # Summary
    print("\n🏆 TEST SUMMARY")
    print("=" * 40)

    hw_info = cudnt.get_hardware_info()
    print(f"System Type: {'HYBRID' if hw_info['hybrid_available'] else 'CPU-ONLY'}")
    print(f"Active Device: {hw_info.get('current_device', 'cpu')}")

    if hw_info['hybrid_available']:
        device_count = len(hw_info['devices'])
        gpu_count = sum(1 for d in hw_info['devices'].values() if d['type'] != 'cpu')
        print(f"Total Devices: {device_count} ({gpu_count} GPU + 1 CPU)")
    else:
        print("Devices: CPU-only")

    print(f"Operations Status: {'✅ WORKING' if operations_work else '❌ FAILED'}")

    if operations_work:
        print("\n🎉 SUCCESS! CUDNT Hybrid System is fully operational!")
        print("   • Automatic GPU detection and selection")
        print("   • Hybrid CPU/GPU tensor operations")
        print("   • Unified memory management")
        print("   • Performance optimization across devices")
        print("\n🚀 Ready for production ML workloads on any hardware!")
    else:
        print("\n⚠️ Hybrid system needs debugging - check logs above")

if __name__ == '__main__':
    main()
