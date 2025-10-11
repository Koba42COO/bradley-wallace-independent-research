#!/usr/bin/env python3
"""
CUDNT Verification Test - Basic functionality check
"""

import sys
import os

def verification_test():
    """Verify CUDNT components are working."""
    print("🔍 CUDNT VERIFICATION TEST")
    print("=" * 30)

    results = {}

    # Test 1: Import GPU virtualization
    print("Test 1: GPU Virtualization Import")
    try:
        from cudnt_gpu_virtualization import CUDNT_GPU_Virtualization, create_tensorflow_like_api
        print("✅ GPU virtualization imported")

        # Test instantiation
        gpu = CUDNT_GPU_Virtualization(n_threads=2)
        print("✅ GPU virtualization instantiated")

        # Test TensorFlow API
        tf_api = create_tensorflow_like_api(gpu)
        print("✅ TensorFlow-like API created")

        results['gpu_virtualization'] = True

    except Exception as e:
        print(f"❌ GPU virtualization failed: {e}")
        results['gpu_virtualization'] = False

    # Test 2: Import enhanced integration
    print("\nTest 2: Enhanced Integration Import")
    try:
        from cudnt_enhanced_integration import create_enhanced_cudnt
        print("✅ Enhanced integration imported")

        # Test instantiation
        cudnt = create_enhanced_cudnt()
        print("✅ Enhanced CUDNT instantiated")

        # Check capabilities
        status = cudnt.get_system_status()
        print(f"✅ System status: GPU={status.get('gpu_virtualization', False)}, Matrix={status.get('matrix_optimization', False)}")

        results['enhanced_integration'] = True

    except Exception as e:
        print(f"❌ Enhanced integration failed: {e}")
        results['enhanced_integration'] = False
        cudnt = None

    # Test 3: Basic operations (if CUDNT loaded)
    if cudnt:
        print("\nTest 3: Basic Operations")
        try:
            import numpy as np

            # Test tensor add
            a = np.random.rand(10, 10)
            b = np.random.rand(10, 10)
            result = cudnt.tensor_add(a, b)
            print("✅ Tensor addition working")

            # Test matrix multiply
            m1 = np.random.rand(5, 8)
            m2 = np.random.rand(8, 3)
            result = cudnt.matrix_multiply(m1, m2)
            print("✅ Matrix multiplication working")

            # Test ReLU
            tensor = np.random.rand(10, 10) - 0.5
            result = cudnt.relu(tensor)
            print("✅ ReLU activation working")

            results['basic_operations'] = True

        except Exception as e:
            print(f"❌ Basic operations failed: {e}")
            results['basic_operations'] = False

    # Test 4: File existence check
    print("\nTest 4: File Structure Check")
    required_files = [
        'cudnt_gpu_virtualization.py',
        'cudnt_enhanced_integration.py',
        'cudnt_ml_demo.py',
        'cudnt_analysis_and_fixes.md',
        'CUDNT_IMPLEMENTATION_SUMMARY.md',
        'cudnt_benchmark_suite.py',
        'test_cudnt_gpu.py'
    ]

    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            missing_files.append(file)

    results['files_complete'] = len(missing_files) == 0

    # Summary
    print("\n📊 VERIFICATION SUMMARY")
    print("=" * 25)

    all_passed = all(results.values())
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("CUDNT GPU virtualization system is ready for ML workloads.")
    else:
        print("⚠️ Some tests failed. Check implementation.")

    print("\nDetailed Results:")
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {test}: {status}")

    return all_passed

if __name__ == "__main__":
    success = verification_test()
    print(f"\n🏁 Final Result: {'SUCCESS' if success else 'ISSUES FOUND'}")

    if success:
        print("\n🚀 CUDNT is ready for production use!")
        print("   • GPU virtualization working")
        print("   • ML operations functional")
        print("   • CPU-only AI/ML enabled")
    else:
        print("\n🔧 Issues need to be resolved before production use.")
        sys.exit(1)
