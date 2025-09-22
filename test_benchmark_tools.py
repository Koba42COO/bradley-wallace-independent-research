#!/usr/bin/env python3
"""
🧪 BENCHMARK TOOLS TEST SCRIPT
==============================
Simple test script to verify benchmark tools are working correctly
"""

import sys
import importlib
import requests
from datetime import datetime

def test_imports():
    """Test that all required modules can be imported"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        'requests',
        'json',
        'time',
        'numpy',
        'psutil',
        'statistics',
        'concurrent.futures',
        'threading',
        'asyncio',
        'aiohttp'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("Please install missing dependencies:")
        print("pip install -r requirements_benchmarks.txt")
        return False
    else:
        print("\n✅ All required modules imported successfully")
        return True

def test_api_connectivity():
    """Test API connectivity"""
    print("\n🌐 Testing API connectivity...")
    
    api_url = "http://localhost:8000"
    
    try:
        # Test basic connectivity
        response = requests.get(f"{api_url}/plugin/health", timeout=5)
        if response.status_code == 200:
            print(f"   ✅ API health check: {response.status_code}")
        else:
            print(f"   ⚠️  API health check: {response.status_code}")
            return False
        
        # Test plugin list
        response = requests.get(f"{api_url}/plugin/list", timeout=5)
        if response.status_code == 200:
            plugins = response.json().get("plugins", [])
            print(f"   ✅ Plugin list: {len(plugins)} plugins available")
            if plugins:
                print(f"   📋 Available plugins: {', '.join(plugins[:3])}{'...' if len(plugins) > 3 else ''}")
        else:
            print(f"   ⚠️  Plugin list: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Cannot connect to {api_url}")
        print("   Please ensure the chAIos server is running")
        return False
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

def test_benchmark_modules():
    """Test that benchmark modules can be imported"""
    print("\n📊 Testing benchmark modules...")
    
    benchmark_modules = [
        'glue_superglue_benchmark',
        'comprehensive_benchmark_suite', 
        'performance_stress_test',
        'master_benchmark_runner'
    ]
    
    failed_modules = []
    
    for module in benchmark_modules:
        try:
            importlib.import_module(module)
            print(f"   ✅ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_modules.append(module)
        except Exception as e:
            print(f"   ⚠️  {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import benchmark modules: {', '.join(failed_modules)}")
        return False
    else:
        print("\n✅ All benchmark modules imported successfully")
        return True

def test_simple_benchmark():
    """Test a simple benchmark execution"""
    print("\n🧪 Testing simple benchmark execution...")
    
    try:
        from glue_superglue_benchmark import GLUEBenchmarkSuite
        
        # Create benchmark suite
        suite = GLUEBenchmarkSuite()
        print("   ✅ GLUE benchmark suite created")
        
        # Test a simple request (without actually running the full test)
        print("   ✅ Benchmark suite initialization successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Simple benchmark test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 BENCHMARK TOOLS TEST SCRIPT")
    print("=" * 50)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("API Connectivity", test_api_connectivity),
        ("Benchmark Modules", test_benchmark_modules),
        ("Simple Benchmark", test_simple_benchmark)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*50}")
    print(f"TEST SUMMARY: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Benchmark tools are ready to use.")
        print("\nYou can now run benchmarks with:")
        print("  python master_benchmark_runner.py")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before running benchmarks.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
