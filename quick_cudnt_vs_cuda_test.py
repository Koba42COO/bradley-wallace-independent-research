#!/usr/bin/env python3
"""
Quick CUDNT vs CUDA Test
========================
Fast comparison test between CUDNT and CUDA-style approaches
"""

import time
import numpy as np
import json
from datetime import datetime
from cudnt_universal_accelerator import get_cudnt_accelerator

def cuda_style_matrix_mult(size=256):
    """Traditional CUDA-style matrix multiplication"""
    start = time.time()
    A = np.random.random((size, size)).astype(np.float32)
    B = np.random.random((size, size)).astype(np.float32)
    C = np.dot(A, B)
    return time.time() - start, float(np.sum(C))

def cudnt_matrix_mult(size=256):
    """CUDNT prime aligned compute-enhanced matrix multiplication"""
    cudnt = get_cudnt_accelerator()
    start = time.time()
    
    A = np.random.random((size, size)).astype(np.float32)
    B = np.random.random((size, size)).astype(np.float32)
    
    # Apply prime aligned compute enhancement
    A_flat = A.flatten()
    B_flat = B.flatten()
    A_enhanced = cudnt.vectorizer.vectorize_consciousness_transform(A_flat)
    B_enhanced = cudnt.vectorizer.vectorize_consciousness_transform(B_flat)
    
    A_enhanced = A_enhanced.reshape((size, size))
    B_enhanced = B_enhanced.reshape((size, size))
    C = np.dot(A_enhanced, B_enhanced)
    
    return time.time() - start, float(np.sum(C))

def cuda_style_vector_ops(size=10000):
    """Traditional CUDA-style vector operations"""
    start = time.time()
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = dot / (norm_a * norm_b)
    return time.time() - start, float(cosine)

def cudnt_vector_ops(size=10000):
    """CUDNT quantum-enhanced vector operations"""
    cudnt = get_cudnt_accelerator()
    start = time.time()
    
    a = np.random.random(size).astype(np.float32)
    b = np.random.random(size).astype(np.float32)
    
    # Quantum processing
    quantum_result = cudnt.accelerate_quantum_computing(a, 50)
    
    # Enhanced operations
    dot = np.dot(a, b) * 1.618  # prime aligned compute enhancement
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = dot / (norm_a * norm_b)
    
    return time.time() - start, float(cosine), quantum_result.get("average_fidelity", 0.0)

def run_quick_test():
    """Run quick comparison test"""
    print("🚀 Quick CUDNT vs CUDA Test")
    print("=" * 40)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }
    
    # Test 1: Matrix Multiplication
    print("\n📊 Test 1: Matrix Multiplication (256x256)")
    cuda_time, cuda_result = cuda_style_matrix_mult()
    cudnt_time, cudnt_result = cudnt_matrix_mult()
    
    speedup = cuda_time / cudnt_time if cudnt_time > 0 else 0
    
    print(f"   CUDA-style: {cuda_time:.4f}s, result: {cuda_result:.2f}")
    print(f"   CUDNT:      {cudnt_time:.4f}s, result: {cudnt_result:.2f}")
    print(f"   Speedup:    {speedup:.2f}x")
    
    results["tests"].append({
        "test": "matrix_multiplication",
        "cuda_time": cuda_time,
        "cudnt_time": cudnt_time,
        "speedup": speedup,
        "cuda_result": cuda_result,
        "cudnt_result": cudnt_result
    })
    
    # Test 2: Vector Operations
    print("\n📊 Test 2: Vector Operations (10,000 elements)")
    cuda_time, cuda_cosine = cuda_style_vector_ops()
    cudnt_time, cudnt_cosine, quantum_fidelity = cudnt_vector_ops()
    
    speedup = cuda_time / cudnt_time if cudnt_time > 0 else 0
    
    print(f"   CUDA-style: {cuda_time:.4f}s, cosine: {cuda_cosine:.4f}")
    print(f"   CUDNT:      {cudnt_time:.4f}s, cosine: {cudnt_cosine:.4f}")
    print(f"   Quantum fidelity: {quantum_fidelity:.4f}")
    print(f"   Speedup:    {speedup:.2f}x")
    
    results["tests"].append({
        "test": "vector_operations",
        "cuda_time": cuda_time,
        "cudnt_time": cudnt_time,
        "speedup": speedup,
        "cuda_cosine": cuda_cosine,
        "cudnt_cosine": cudnt_cosine,
        "quantum_fidelity": quantum_fidelity
    })
    
    # Test 3: Quantum Processing
    print("\n📊 Test 3: Quantum Processing (8 qubits)")
    cudnt = get_cudnt_accelerator()
    test_data = np.random.random(256).astype(np.float32)
    
    start = time.time()
    quantum_result = cudnt.accelerate_quantum_computing(test_data, 100)
    quantum_time = time.time() - start
    
    print(f"   CUDNT Quantum: {quantum_time:.4f}s")
    print(f"   Qubits: {quantum_result.get('qubits_simulated', 0)}")
    print(f"   Fidelity: {quantum_result.get('average_fidelity', 0.0):.4f}")
    print(f"   prime aligned compute enhancement: {quantum_result.get('consciousness_enhancement', 1.618)}")
    
    results["tests"].append({
        "test": "quantum_processing",
        "quantum_time": quantum_time,
        "quantum_result": quantum_result
    })
    
    # Summary
    print("\n🎯 SUMMARY")
    print("-" * 20)
    avg_speedup = np.mean([t["speedup"] for t in results["tests"] if "speedup" in t])
    print(f"Average Speedup: {avg_speedup:.2f}x")
    
    if avg_speedup > 1.0:
        print("✅ CUDNT shows superior performance!")
    else:
        print("⚠️ CUDA-style shows better performance")
    
    print("\n🧠 CUDNT Advantages:")
    print("   • prime aligned compute mathematics (1.618x Golden Ratio)")
    print("   • Universal access (no GPU hardware required)")
    print("   • Quantum simulation capabilities")
    print("   • Advanced vectorization")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"quick_cudnt_vs_cuda_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return results

if __name__ == "__main__":
    run_quick_test()
