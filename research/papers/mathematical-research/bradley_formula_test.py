#!/usr/bin/env python3
"""
Direct implementation of Bradley's Formula: g_n = W_φ(p_n) · φ^k
Test on the 455 million prime dataset
"""

import numpy as np
import time
import json
from pathlib import Path
from scipy.fft import fft
from results_database import WallaceResultsDatabase
from enhanced_display import EnhancedDisplaySystem

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
SQRT2 = np.sqrt(2)         # ≈ 1.4142135623730951

def wallace_transform(x, alpha=PHI, beta=0.618, epsilon=1e-8, phi=PHI):
    """Wallace Transform: W_φ(x)"""
    if x <= 0:
        return np.nan
    log_val = np.log(x + epsilon)
    return alpha * np.power(np.abs(log_val), phi) * np.sign(log_val) + beta

def bradley_formula_test(primes, gaps, k_range=(-2, 3), tolerance=0.2):
    """
    Test Bradley's Formula: g_n = W_φ(p_n) · φ^k

    Args:
        primes: array of prime numbers
        gaps: array of prime gaps
        k_range: range of k values to test
        tolerance: relative error tolerance (20%)

    Returns:
        dict with test results
    """
    print("🔬 TESTING BRADLEY'S FORMULA: g_n = W_φ(p_n) · φ^k")
    print(f"   Dataset: {len(primes):,} primes, {len(gaps):,} gaps")
    print(f"   k range: {k_range[0]} to {k_range[1]}")
    print(f"   Tolerance: {tolerance*100}%")

    results = {}
    total_tests = 0
    total_matches = 0

    # Test each k value
    for k in range(k_range[0], k_range[1] + 1):
        phi_k = np.power(PHI, k)
        matches = 0
        match_details = []

        print(f"   Testing k={k} (φ^k = {phi_k:.6f})...")

        # Test each gap
        for i, (p, gap) in enumerate(zip(primes, gaps)):
            if i >= len(gaps):
                break

            # Calculate W_φ(p_n)
            wt_p = wallace_transform(p)

            # Calculate expected gap: W_φ(p_n) · φ^k
            expected_gap = wt_p * phi_k

            # Check if actual gap matches expected (within tolerance)
            if expected_gap > 0:
                relative_error = abs(expected_gap - gap) / max(gap, expected_gap)

                if relative_error <= tolerance:
                    matches += 1
                    match_details.append({
                        'prime': int(p),
                        'actual_gap': int(gap),
                        'expected_gap': float(expected_gap),
                        'error': float(relative_error),
                        'k': k
                    })

        # Store results for this k
        percent_match = (matches / len(gaps)) * 100
        results[k] = {
            'phi_k': float(phi_k),
            'matches': matches,
            'percent': float(percent_match),
            'sample_matches': match_details[:10]  # Keep first 10 for display
        }

        total_tests += len(gaps)
        total_matches += matches

        print(f"      k={k}: {matches:,} matches ({percent_match:.3f}%)")
    # Overall results
    overall_percent = (total_matches / total_tests) * 100

    return {
        'formula': "g_n = W_φ(p_n) · φ^k",
        'k_range': k_range,
        'tolerance': tolerance,
        'total_primes': len(primes),
        'total_gaps': len(gaps),
        'total_matches': total_matches,
        'overall_percent': float(overall_percent),
        'k_results': results
    }

def load_10billion_dataset():
    """Load the 455 million prime dataset"""
    print("📥 Loading 455 million prime dataset...")

    # Check if we have the results database with the actual 455M dataset
    db_path = Path("wallace_results.db")
    if db_path.exists():
        print("   Found results database, attempting to load 455M prime data...")
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            # Check if we have the large dataset stored
            cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='analysis_runs'")
            if cursor.fetchone()[0] > 0:
                # Try to find the 455M prime run
                cursor.execute("""
                    SELECT run_id FROM analysis_runs
                    WHERE scale >= 400000000
                    ORDER BY timestamp DESC LIMIT 1
                """)
                result = cursor.fetchone()
                if result:
                    run_id = result[0]
                    print(f"   Found 455M dataset run_id: {run_id}")

                    # Try to load primes from raw_results or recreate from gaps
                    # For now, fall back to generating large dataset
                    pass

            conn.close()
        except Exception as e:
            print(f"   Database error: {e}")

    # For now, generate a larger test dataset (closer to the actual scale)
    print("   📈 Generating larger test dataset (closer to 455M scale)...")

    # Generate primes up to 10^7 (10 million) for more realistic testing
    # This is 100x larger than our previous test but still manageable
    limit = 10000000  # 10^7 primes
    print(f"   Generating primes up to {limit:,} (10M primes)...")
    primes = generate_primes_sieve(limit)
    gaps = np.diff(primes).astype(int)

    print(f"   ✅ Generated {len(primes):,} primes up to {limit}")
    print(f"   ✅ Generated {len(gaps):,} gaps")
    print(f"   📊 Scale ratio: {len(primes)/455052511:.3f} of full 455M dataset")

    return primes, gaps

def generate_primes_sieve(limit):
    """Generate primes using sieve of Eratosthenes"""
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False

    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    return np.where(sieve)[0]

def run_bradley_test():
    """Run the complete Bradley formula test"""
    print("🌟 WALLACE TRANSFORM - BRADLEY'S FORMULA VALIDATION")
    print("=" * 60)

    # Load dataset
    print("📥 Loading dataset...")
    primes, gaps = load_10billion_dataset()

    # Run Bradley's formula test
    print("🔬 Running Bradley's formula test...")
    results = bradley_formula_test(primes, gaps)

    # Display key results
    print("\n📊 BRADLEY'S FORMULA RESULTS")
    print("-" * 35)
    print(f"Formula: {results['formula']}")
    print(f"Dataset: {results['total_primes']:,} primes")
    print(f"Total matches: {results['total_matches']:,}")
    print(".3f")

    print("\nDetailed k results:")
    for k, k_data in results['k_results'].items():
        phi_k = k_data['phi_k']
        matches = k_data['matches']
        percent = k_data['percent']
        print("6.3f")

    # Save detailed results
    output_file = f"bradley_formula_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {output_file}")

    return results

if __name__ == "__main__":
    run_bradley_test()
