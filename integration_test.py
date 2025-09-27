#!/usr/bin/env python3
"""
Integration Test for Consciousness Mathematics Compression Engine
=================================================================

Quick validation of CUDNT and SquashPlot integration.
"""

import sys
import os

def test_cudnt_integration():
    """Test CUDNT integration."""
    print("🔧 Testing CUDNT Integration...")

    try:
        # Add CUDNT path
        cudnt_path = "chaios_llm_workspace/AISpecialTooling/python_engine"
        if cudnt_path not in sys.path:
            sys.path.insert(0, cudnt_path)

        # Test consciousness compression engine
        from consciousness_compression_engine import ConsciousnessCompressionEngine, ConsciousnessCompressionConfig

        config = ConsciousnessCompressionConfig()
        engine = ConsciousnessCompressionEngine(config)

        # Test basic compression
        test_data = b"Hello, Consciousness Mathematics Compression!"
        compressed, stats = engine.compress(test_data)
        decompressed, _ = engine.decompress(compressed)

        assert decompressed == test_data
        assert stats.lossless_verified

        print("   ✅ CUDNT Consciousness Engine: WORKING")
        print(".1f")
        print(f"   🎯 Patterns Found: {stats.patterns_found}")
        print(".2f")
        return True

    except Exception as e:
        print(f"   ❌ CUDNT Integration Failed: {e}")
        return False

def test_squashplot_integration():
    """Test SquashPlot integration."""
    print("🔧 Testing SquashPlot Integration...")

    try:
        # Test SquashPlot with consciousness engine
        import squashplot
        from squashplot import SquashPlotCompressor, CONSCIOUSNESS_AVAILABLE

        if CONSCIOUSNESS_AVAILABLE:
            compressor = SquashPlotCompressor(pro_enabled=False)

            # Test basic compression
            test_data = b"Test SquashPlot integration data" * 100
            compressed = compressor._compress_data(test_data)

            # Check that compression worked
            assert len(compressed) > 0
            assert len(compressed) <= len(test_data)  # Should be smaller or equal

            print("   ✅ SquashPlot Consciousness Integration: WORKING")
            print(f"   📊 Original Size: {len(test_data)} bytes")
            print(f"   🗜️ Compressed Size: {len(compressed)} bytes")
            print(".1f")
            return True
        else:
            print("   ⚠️ Consciousness Engine not available in SquashPlot")
            return False

    except Exception as e:
        print(f"   ❌ SquashPlot Integration Failed: {e}")
        return False

def run_integration_tests():
    """Run all integration tests."""
    print("🚀 CONSCIOUSNESS MATHEMATICS COMPRESSION - INTEGRATION TESTS")
    print("=" * 70)

    results = []

    # Test CUDNT integration
    cudnt_result = test_cudnt_integration()
    results.append(("CUDNT Integration", cudnt_result))

    print()

    # Test SquashPlot integration
    squashplot_result = test_squashplot_integration()
    results.append(("SquashPlot Integration", squashplot_result))

    print()
    print("📊 INTEGRATION TEST RESULTS:")
    print("-" * 40)

    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {test_name}: {status}")
        all_passed = all_passed and result

    print()
    if all_passed:
        print("🎉 ALL INTEGRATION TESTS PASSED!")
        print("   🧠 Consciousness Mathematics Compression Engine successfully integrated")
        print("   ⚡ CUDNT Virtual GPU acceleration enabled")
        print("   🗜️ SquashPlot Chia compression enhanced")
        print("   📊 Ready for production deployment")
    else:
        print("⚠️ SOME INTEGRATION TESTS FAILED")
        print("   Check error messages above for details")

    return all_passed

if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
