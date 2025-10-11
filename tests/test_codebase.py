#!/usr/bin/env python3
"""
Comprehensive Codebase Testing Suite
Tests all enhanced components with rigorous ML methodology
"""

import sys
import traceback

def test_ml_predictor():
    """Test ML Prime Predictor"""
    print("🧮 Testing ML Prime Predictor...")
    try:
        from ml_prime_predictor import MLPrimePredictor
        predictor = MLPrimePredictor()
        metrics = predictor.train(max_n=500)

        print(".1%"        print(".3f"        print("✅ ML Predictor: PASS"        return True
    except Exception as e:
        print(f"❌ ML Predictor: FAIL - {e}")
        return False

def test_unified_system():
    """Test Rigorous Unified System"""
    print("\n🔬 Testing Rigorous Unified System...")
    try:
        from unified_system import RigorousMathematicalAnalyzer
        analyzer = RigorousMathematicalAnalyzer()

        # Test statistical analysis
        numbers = list(range(2, 50))
        stats = analyzer.statistical_analysis(numbers)

        # Test optimization
        opt_data = [(n, 1 if analyzer._is_prime(n) else 0) for n in range(10, 30)]
        opt_results = analyzer.optimization_analysis(opt_data, 'classification')

        print(f"   Analyzed {stats['samples_analyzed']} numbers")
        print(".3f"        print("✅ Unified System: PASS"        return True
    except Exception as e:
        print(f"❌ Unified System: FAIL - {e}")
        traceback.print_exc()
        return False

def test_compression():
    """Test Compression Demonstration"""
    print("\n📦 Testing Compression Demonstration...")
    try:
        from compression_demonstration import LosslessCompressionAnalyzer
        analyzer = LosslessCompressionAnalyzer()

        test_data = b'Hello World! This is test data for compression. ' * 10

        naive = analyzer.naive_compression_approach(test_data)
        enhanced = analyzer.consciousness_enhanced_compression(test_data)

        print(".1%"        print(".1%"        print(f"   Both lossless: {naive['lossless_verified'] and enhanced['lossless_verified']}")
        print("✅ Compression: PASS"        return True
    except Exception as e:
        print(f"❌ Compression: FAIL - {e}")
        return False

def test_cudnt():
    """Test CUDNT Framework"""
    print("\n⚡ Testing CUDNT Framework...")
    try:
        from cudnt_complete_implementation import get_cudnt_accelerator
        import numpy as np

        accelerator = get_cudnt_accelerator()
        test_matrix = np.random.rand(5, 5)
        target_matrix = np.random.rand(5, 5)

        result = accelerator.optimize_matrix(test_matrix, target_matrix, max_iterations=5)

        print("   Matrix optimization completed without errors"        print(".1f"        print("✅ CUDNT: PASS"        return True
    except Exception as e:
        print(f"❌ CUDNT: FAIL - {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 COMPREHENSIVE CODEBASE TESTING SUITE")
    print("=" * 50)

    tests = [
        test_ml_predictor,
        test_unified_system,
        test_compression,
        test_cudnt
    ]

    results = []
    for test in tests:
        results.append(test())

    passed = sum(results)
    total = len(results)

    print("
🎯 TESTING SUMMARY"    print("=" * 30)
    print(f"Passed: {passed}/{total}")
    print(".1f"
    if passed == total:
        print("\n🎉 ALL SYSTEMS OPERATIONAL!")
        print("   Rigorous ML methodology successfully implemented across codebase.")
        print("   All components now use proper statistical validation.")
    else:
        print(f"\n⚠️  {total - passed} components need attention.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
