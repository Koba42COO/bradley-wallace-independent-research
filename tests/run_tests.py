#!/usr/bin/env python3
"""
Comprehensive Test Runner for Wallace Research Suite
Runs all product tests and generates reports
"""

import unittest
import sys
import os
import time
import json
from datetime import datetime
import subprocess

def run_test_suite(test_module, test_name):
    """Run a specific test suite and return results"""
    print(f"\n🧪 Running {test_name}...")

    try:
        # Import the test module
        module = __import__(f"tests.{test_module}", fromlist=[test_module])

        # Discover and run tests
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=1, stream=sys.stdout)
        result = runner.run(suite)

        return {
            'name': test_name,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(result.skipped),
            'success': result.wasSuccessful(),
            'failures_list': [str(f[0]) for f in result.failures],
            'errors_list': [str(e[0]) for e in result.errors]
        }

    except Exception as e:
        print(f"❌ Failed to run {test_name}: {e}")
        return {
            'name': test_name,
            'tests_run': 0,
            'failures': 0,
            'errors': 1,
            'skipped': 0,
            'success': False,
            'failures_list': [],
            'errors_list': [str(e)]
        }

def check_service_availability(service_name, url, expected_status=200):
    """Check if a service is available"""
    try:
        import requests
        response = requests.get(url, timeout=5)
        return response.status_code == expected_status
    except:
        return False

def generate_report(results, start_time, end_time):
    """Generate a comprehensive test report"""
    total_tests = sum(r['tests_run'] for r in results)
    total_failures = sum(r['failures'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    overall_success = all(r['success'] for r in results)

    report = {
        'timestamp': datetime.utcnow().isoformat(),
        'duration_seconds': end_time - start_time,
        'overall_success': overall_success,
        'summary': {
            'total_tests': total_tests,
            'total_failures': total_failures,
            'total_errors': total_errors,
            'total_skipped': total_skipped,
            'success_rate': (total_tests - total_failures - total_errors) / total_tests if total_tests > 0 else 0
        },
        'service_availability': {
            'wqrf_api': check_service_availability('WQRF API', 'http://localhost:5001/health'),
            'aiva_ide_server': check_service_availability('AIVA IDE Server', 'http://localhost:3001/api/health'),
            'aiva_ide_client': check_service_availability('AIVA IDE Client', 'http://localhost:3000'),
        },
        'test_results': results
    }

    return report

def save_report(report, filename="test_report.json"):
    """Save test report to file"""
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📊 Test report saved to {filename}")

def main():
    """Main test runner"""
    print("🚀 Wallace Research Suite - Comprehensive Test Suite")
    print("=" * 60)

    # Test suites to run
    test_suites = [
        ('test_wqrf_api', 'WQRF API Tests'),
        ('test_aiva_ide', 'AIVA IDE Tests'),
        ('test_cudnt', 'CUDNT Tests'),
    ]

    start_time = time.time()
    results = []

    # Check prerequisites
    print("🔍 Checking prerequisites...")

    # Check if required packages are installed
    required_packages = ['requests', 'numpy', 'scikit-learn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"⚠️  Missing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return 1

    print("✅ Prerequisites check passed")

    # Run each test suite
    for test_module, test_name in test_suites:
        if os.path.exists(f"tests/{test_module}.py"):
            result = run_test_suite(test_module, test_name)
            results.append(result)
        else:
            print(f"⚠️  Test file tests/{test_module}.py not found - skipping")
            results.append({
                'name': test_name,
                'tests_run': 0,
                'failures': 0,
                'errors': 0,
                'skipped': 1,
                'success': True,
                'failures_list': [],
                'errors_list': []
            })

    end_time = time.time()

    # Generate and display report
    report = generate_report(results, start_time, end_time)

    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Passed: {report['summary']['total_tests'] - report['summary']['total_failures'] - report['summary']['total_errors']}")
    print(f"Failed: {report['summary']['total_failures']}")
    print(f"Errors: {report['summary']['total_errors']}")
    print(f"Skipped: {report['summary']['total_skipped']}")
    print(".1%")
    print(".2f")
    print(f"Overall Status: {'✅ PASSED' if report['overall_success'] else '❌ FAILED'}")

    print("\n🌐 Service Availability:")
    for service, available in report['service_availability'].items():
        status = "✅ Available" if available else "❌ Unavailable"
        print(f"  {service}: {status}")

    print("\n📋 Detailed Results:")
    for result in report['test_results']:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} {result['name']}: {result['tests_run']} tests")

        if result['failures'] > 0:
            print(f"    Failures: {result['failures']}")
        if result['errors'] > 0:
            print(f"    Errors: {result['errors']}")

    # Save detailed report
    save_report(report)

    # Return appropriate exit code
    return 0 if report['overall_success'] else 1

if __name__ == '__main__':
    sys.exit(main())
