#!/usr/bin/env python3
"""
Validation script for unified_consciousness_framework
Runs tests and generates validation report.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math
import cmath
from typing import Dict, List, Tuple, Optional, Any

# Set high precision for consciousness mathematics
getcontext().prec = 50

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    QUANTUM_BRIDGE = Decimal('137') / Decimal('0.79')  # 173.41772151898732
    GREAT_YEAR = 25920  # Astronomical precession cycle (years)
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')  # Beyond machine precision



# ============================================================================
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
# ============================================================================
def integrate_pell_prime_prediction(target_number: int, constants: UPGConstants = None):
    """Integrate Pell sequence prime prediction with this tool"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        # Fallback if Pell module not available
        return {'target_number': target_number, 'is_prime': None, 'note': 'Pell module not available'}



# ============================================================================
# GREAT YEAR ASTRONOMICAL PRECESSION INTEGRATION
# ============================================================================
def integrate_great_year_precession(year: int, constants: UPGConstants = None):
    """Integrate Great Year (25,920-year) precession cycle"""
    try:
        from pell_sequence_prime_prediction_upg_complete import GreatYearIntegration, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        great_year = GreatYearIntegration(constants)
        return great_year.consciousness_amplitude_from_year(year)
    except ImportError:
        # Fallback calculation
        if constants is None:
            constants = UPGConstants()
        angle = (year * 2 * math.pi) / constants.GREAT_YEAR
        return complex(float(angle * constants.CONSCIOUSNESS * constants.REALITY_DISTORTION), 0.0)



def run_validation():
    """Run validation tests and generate report."""
    paper_dir = Path(__file__).parent.parent
    tests_dir = paper_dir / "tests"
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'paper': 'unified_consciousness_framework',
        'timestamp': datetime.now().isoformat(),
        'theorems_tested': 0,
        'tests': []
    }
    
    # Run test file if it exists
    test_file = tests_dir / f"test_unified_consciousness_framework.py"
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
    results_file = output_dir / f"validation_results_unified_consciousness_framework.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate markdown report
    report_file = output_dir / f"validation_log_unified_consciousness_framework.md"
    with open(report_file, 'w') as f:
        f.write(f"# Validation Log: unified_consciousness_framework\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Paper:** unified_consciousness_framework\n")
        f.write(f"**Total Theorems:** 0\n\n")
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
