#!/usr/bin/env python3
"""
Validation script for structured_chaos_foundation
Runs tests and generates validation report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

def run_validation():
    """Run validation tests and generate report."""
    paper_dir = Path(__file__).parent.parent
    tests_dir = paper_dir / "tests"
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'paper': 'structured_chaos_foundation',
        'timestamp': datetime.now().isoformat(),
        'theorems_tested': 8,
        'tests': []
    }
    
    # Run test file if it exists
    test_file = tests_dir / f"test_structured_chaos_foundation.py"
    if test_file.exists():
        print(f"Running tests from {test_file}...")
        try:
            result = subprocess.run(
                [sys.executable, str(test_file)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            results['test_output'] = result.stdout
            results['test_errors'] = result.stderr
            results['test_returncode'] = result.returncode
            results['tests_passed'] = result.returncode == 0
            
            if result.returncode == 0:
                print("✅ All tests passed!")
            else:
                print("⚠️  Some tests failed")
        except subprocess.TimeoutExpired:
            results['test_timeout'] = True
            print("⚠️  Tests timed out")
        except Exception as e:
            results['test_error'] = str(e)
            print(f"⚠️  Error running tests: {e}")
    else:
        print(f"⚠️  Test file not found: {test_file}")
        results['test_file_missing'] = True
    
    # Save results
    results_file = output_dir / f"validation_results_structured_chaos_foundation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    report_file = output_dir / f"validation_log_structured_chaos_foundation.md"
    with open(report_file, 'w') as f:
        f.write(f"# Validation Log: structured_chaos_foundation\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Paper:** structured_chaos_foundation\n")
        f.write(f"**Total Theorems:** 8\n\n")
        f.write("## Test Execution Summary\n\n")
        
        if results.get('tests_passed'):
            f.write("✅ **Status:** All tests passed\n")
        elif results.get('test_file_missing'):
            f.write("⚠️  **Status:** Test file not found\n")
        else:
            f.write("❌ **Status:** Some tests failed\n")
        
        f.write("\n## Theorem Validation Results\n\n")
        for idx, thm in enumerate(theorems):
            f.write(f"### {idx+1}. {thm['name']} ({thm['type']})\n")
            f.write("**Status:** ⏳ Pending validation\n")
            f.write("**Validation Method:** Automated test suite\n\n")
        
        f.write("\n## Overall Statistics\n\n")
        f.write(f"- **Total Theorems:** {len(theorems)}\n")
        f.write("- **Tests Run:** {'Yes' if not results.get('test_file_missing') else 'No'}\n")
        f.write("- **Tests Passed:** {'Yes' if results.get('tests_passed') else 'No'}\n")
    
    print(f"\n✅ Validation complete! Results saved to {results_file}")
    print(f"📄 Report saved to {report_file}")

if __name__ == '__main__':
    run_validation()
