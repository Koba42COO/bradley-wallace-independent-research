#!/usr/bin/env python3
"""
SquashPlot Test Suite for Replit
================================

Quick tests to verify SquashPlot is working correctly on Replit.
"""

import os
import sys
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("📦 Testing imports...")

    try:
        from squashplot import SquashPlotCompressor
        print("✅ SquashPlotCompressor imported")
    except ImportError as e:
        print(f"❌ Failed to import SquashPlotCompressor: {e}")
        return False

    try:
        from whitelist_signup import WhitelistManager
        print("✅ WhitelistManager imported")
    except ImportError as e:
        print(f"❌ Failed to import WhitelistManager: {e}")
        return False

    try:
        from compression_validator import CompressionValidator
        print("✅ CompressionValidator imported")
    except ImportError as e:
        print(f"❌ Failed to import CompressionValidator: {e}")
        return False

    return True

def test_basic_compression():
    """Test basic compression functionality"""
    print("\n🗜️ Testing basic compression...")

    try:
        from squashplot import SquashPlotCompressor

        # Create test data
        test_data = b"Hello, this is test data for SquashPlot compression!" * 100

        # Test basic compression
        compressor = SquashPlotCompressor(pro_enabled=False)
        compressed = compressor._basic_compress(test_data)

        print(f"   Original size: {len(test_data)} bytes")
        print(f"   Compressed size: {len(compressed)} bytes")
        print(".1f")
        print("✅ Basic compression test passed")

        return True

    except Exception as e:
        print(f"❌ Basic compression test failed: {e}")
        return False

def test_pro_compression():
    """Test Pro compression functionality"""
    print("\n🚀 Testing Pro compression...")

    try:
        from squashplot import SquashPlotCompressor

        # Create test data
        test_data = b"Hello, this is test data for SquashPlot Pro compression!" * 100

        # Test Pro compression
        compressor = SquashPlotCompressor(pro_enabled=True)
        compressed = compressor._pro_compress(test_data)

        print(f"   Original size: {len(test_data)} bytes")
        print(f"   Compressed size: {len(compressed)} bytes")
        print(".1f")
        print("✅ Pro compression test passed")

        return True

    except Exception as e:
        print(f"❌ Pro compression test failed: {e}")
        return False

def test_whitelist():
    """Test whitelist functionality"""
    print("\n📋 Testing whitelist...")

    try:
        from whitelist_signup import WhitelistManager

        # Create whitelist manager
        wl = WhitelistManager()

        # Test adding user
        result = wl.add_to_whitelist("test@example.com")
        if result['success']:
            print("✅ Whitelist add test passed")
        else:
            print("❌ Whitelist add test failed")
            return False

        # Test checking user
        status = wl.check_status("test@example.com")
        if status['found']:
            print("✅ Whitelist check test passed")
        else:
            print("❌ Whitelist check test failed")
            return False

        return True

    except Exception as e:
        print(f"❌ Whitelist test failed: {e}")
        return False

def test_web_interface():
    """Test that web interface exists and is accessible"""
    print("\n🌐 Testing web interface...")

    web_file = Path("squashplot_web_interface.html")
    if web_file.exists():
        size = web_file.stat().st_size
        print(f"✅ Web interface found ({size:,} bytes)")

        # Check for key content
        with open(web_file, 'r') as f:
            content = f.read()

        if "SquashPlot" in content:
            print("✅ Web interface contains SquashPlot content")
        else:
            print("❌ Web interface missing SquashPlot content")
            return False

        return True
    else:
        print("❌ Web interface file not found")
        return False

def test_main_entry():
    """Test main entry point"""
    print("\n🚀 Testing main entry point...")

    try:
        import main
        print("✅ Main module can be imported")
        return True
    except ImportError as e:
        print(f"❌ Main module import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 SquashPlot Test Suite for Replit")
    print("=" * 40)

    tests = [
        ("Import Test", test_imports),
        ("Basic Compression", test_basic_compression),
        ("Pro Compression", test_pro_compression),
        ("Whitelist System", test_whitelist),
        ("Web Interface", test_web_interface),
        ("Main Entry", test_main_entry),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running: {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} failed")

    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! SquashPlot is ready to use.")
        print("\n🚀 Quick start:")
        print("   python main.py --web    # Start web interface")
        print("   python main.py --demo   # Run interactive demo")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
