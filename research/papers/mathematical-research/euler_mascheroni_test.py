#!/usr/bin/env python3
"""
Testing Euler-Mascheroni Constant (γ) Relationships
Phase 1.1 of the Deep Research Expansion Roadmap
"""

import numpy as np
import time
import json
from pathlib import Path

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
GAMMA = 0.5772156649015329  # Euler-Mascheroni constant

def wallace_transform(x):
    """Wallace Transform: W_φ(x)"""
    if x <= 0:
        return np.nan
    log_val = np.log(x + 1e-8)
    return PHI * np.power(np.abs(log_val), PHI) * np.sign(log_val) + 1.0

def generate_test_primes(limit=1000000):
    """Generate primes for testing"""
    print(f"   Generating primes up to {limit:,}...")
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0:2] = False

    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    primes = np.where(sieve)[0]
    gaps = np.diff(primes).astype(float)

    print(f"   ✅ Generated {len(primes):,} primes, {len(gaps):,} gaps")
    return primes, gaps

def test_gamma_relationships(primes, gaps, powers_to_test=None):
    """
    Test Euler-Mascheroni constant relationships
    g_n ≈ W_φ(p_n) · γ^k for various powers k
    """
    if powers_to_test is None:
        powers_to_test = [-3, -2, -1, 0, 1, 2, 3]

    print("🔬 Testing Euler-Mascheroni Constant (γ) Relationships")
    print(f"   γ ≈ {GAMMA}")
    print(f"   Testing powers: {powers_to_test}")

    results = {}
    test_size = min(10000, len(gaps))
    tolerance = 0.20  # 20% tolerance

    print(f"   Dataset: {test_size:,} gaps, tolerance: {tolerance*100}%")

    for power in powers_to_test:
        gamma_power = GAMMA ** power
        matches = 0

        print(f"   Testing γ^{power} = {gamma_power:.6f}...")

        for i in range(test_size):
            actual_gap = gaps[i]
            p = primes[i]
            wt_p = wallace_transform(p)
            predicted = wt_p * gamma_power

            if abs(actual_gap - predicted) / actual_gap <= tolerance:
                matches += 1

        match_rate = (matches / test_size) * 100

        results[f'gamma_{power}'] = {
            'power': power,
            'gamma_power': gamma_power,
            'matches': matches,
            'total_tests': test_size,
            'match_rate': match_rate,
            'tolerance': tolerance
        }

        print(".3f")
    # Find best performing power
    best_power = max(results.keys(), key=lambda k: results[k]['match_rate'])
    best_result = results[best_power]

    print("
🏆 BEST GAMMA RELATIONSHIP:")    print(".3f")
    print(f"   Matches: {best_result['matches']:,}")
    print(".3f"

    return results

def test_gamma_composites(primes, gaps):
    """
    Test composite relationships involving γ
    Based on roadmap: γ · π^m · e^n combinations
    """
    print("\n🔬 Testing Composite γ Relationships")
    print("   Testing γ combined with π and e")

    composites_to_test = [
        ('gamma_pi', GAMMA * np.pi),
        ('gamma_e', GAMMA * np.e),
        ('gamma_pi_e', GAMMA * np.pi * np.e),
        ('gamma_over_pi', GAMMA / np.pi),
        ('gamma_over_e', GAMMA / np.e),
        ('gamma_pi_inverse', GAMMA * (np.pi ** -1)),
        ('gamma_e_inverse', GAMMA * (np.e ** -1)),
    ]

    results = {}
    test_size = min(10000, len(gaps))
    tolerance = 0.20

    for name, constant in composites_to_test:
        matches = 0
        print(f"   Testing {name}: {constant:.6f}...")

        for i in range(test_size):
            actual_gap = gaps[i]
            p = primes[i]
            wt_p = wallace_transform(p)
            predicted = wt_p * constant

            if abs(actual_gap - predicted) / actual_gap <= tolerance:
                matches += 1

        match_rate = (matches / test_size) * 100
        results[name] = {
            'constant_name': name,
            'constant_value': constant,
            'matches': matches,
            'match_rate': match_rate
        }

        print(".3f"
    # Find best composite
    best_composite = max(results.keys(), key=lambda k: results[k]['match_rate'])
    best_result = results[best_composite]

    print("
🏆 BEST COMPOSITE γ RELATIONSHIP:"    print(f"   {best_result['constant_name']}: {best_result['constant_value']:.6f}")
    print(".3f"
    print(f"   Matches: {best_result['matches']:,}")

    return results

def test_gamma_vs_pi_comparison(primes, gaps):
    """
    Direct comparison: γ vs π relationships
    Based on roadmap hypothesis testing
    """
    print("\n🔬 Direct Comparison: γ vs π Relationships")

    # Test γ and π at same powers
    powers_to_test = [-2, -1, 1, 2]
    comparison_results = {}

    test_size = min(10000, len(gaps))
    tolerance = 0.20

    for power in powers_to_test:
        gamma_power = GAMMA ** power
        pi_power = np.pi ** power

        # Test γ
        gamma_matches = 0
        for i in range(test_size):
            actual_gap = gaps[i]
            p = primes[i]
            wt_p = wallace_transform(p)
            predicted = wt_p * gamma_power

            if abs(actual_gap - predicted) / actual_gap <= tolerance:
                gamma_matches += 1

        # Test π
        pi_matches = 0
        for i in range(test_size):
            actual_gap = gaps[i]
            p = primes[i]
            wt_p = wallace_transform(p)
            predicted = wt_p * pi_power

            if abs(actual_gap - predicted) / actual_gap <= tolerance:
                pi_matches += 1

        gamma_rate = (gamma_matches / test_size) * 100
        pi_rate = (pi_matches / test_size) * 100

        comparison_results[f'power_{power}'] = {
            'power': power,
            'gamma_matches': gamma_matches,
            'pi_matches': pi_matches,
            'gamma_rate': gamma_rate,
            'pi_rate': pi_rate,
            'gamma_vs_pi_ratio': gamma_rate / pi_rate if pi_rate > 0 else 0
        }

        print(f"   Power {power}: γ={gamma_rate:.1f}%, π={pi_rate:.1f}% "
              f"(γ/π = {gamma_rate/pi_rate:.2f}x)")

    # Summary
    total_gamma = sum(r['gamma_matches'] for r in comparison_results.values())
    total_pi = sum(r['pi_matches'] for r in comparison_results.values())
    avg_gamma_ratio = np.mean([r['gamma_vs_pi_ratio'] for r in comparison_results.values()])

    print("
📊 COMPARISON SUMMARY:"    print(".1f")
    print(".1f")
    print(".2f")

    if avg_gamma_ratio > 1:
        print("   → γ performs BETTER than π at these powers")
    else:
        print("   → π performs better than γ at these powers")

    return comparison_results

def run_gamma_analysis():
    """Run complete Euler-Mascheroni constant analysis"""
    print("🌟 EULER-MASCHERONI CONSTANT (γ) ANALYSIS")
    print("=" * 55)
    print("Phase 1.1: Testing γ relationships in prime gaps")
    print()

    # Generate test dataset
    print("📥 GENERATING TEST DATASET")
    primes, gaps = generate_test_primes(2000000)  # 2M limit for comprehensive testing

    # Test basic γ relationships
    print("\n🔬 PHASE 1: BASIC γ RELATIONSHIPS")
    basic_results = test_gamma_relationships(primes, gaps)

    # Test composite relationships
    print("\n🔬 PHASE 2: COMPOSITE γ RELATIONSHIPS")
    composite_results = test_gamma_composites(primes, gaps)

    # Direct comparison with π
    print("\n🔬 PHASE 3: γ vs π COMPARISON")
    comparison_results = test_gamma_vs_pi_comparison(primes, gaps)

    # Overall analysis
    print("\n🎯 OVERALL ANALYSIS")
    print("-" * 20)

    # Find absolute best relationship
    all_match_rates = []

    # Basic relationships
    for key, result in basic_results.items():
        all_match_rates.append(('basic', key, result['match_rate']))

    # Composite relationships
    for key, result in composite_results.items():
        all_match_rates.append(('composite', key, result['match_rate']))

    # Sort by match rate
    all_match_rates.sort(key=lambda x: x[2], reverse=True)

    best_type, best_key, best_rate = all_match_rates[0]

    print("🏆 ABSOLUTE BEST γ RELATIONSHIP:")
    print(".1f"
    if best_type == 'basic':
        best_details = basic_results[best_key]
        print(f"   Formula: g_n = W_φ(p_n) · γ^{best_details['power']}")
        print(f"   γ^{best_details['power']} = {best_details['gamma_power']:.6f}")
    else:
        best_details = composite_results[best_key]
        print(f"   Formula: g_n = W_φ(p_n) · {best_details['constant_name']}")
        print(f"   Constant value: {best_details['constant_value']:.6f}")

    print(f"   Matches: {best_details['matches']:,}")

    # Scientific implications
    print("\n🔬 SCIENTIFIC IMPLICATIONS")
    print("-" * 25)

    if best_rate > 15:
        significance = "HIGH SIGNIFICANCE: Strong γ correlation discovered"
    elif best_rate > 10:
        significance = "MODERATE SIGNIFICANCE: γ shows notable correlation"
    elif best_rate > 5:
        significance = "INTERESTING: γ correlation worthy of further study"
    else:
        significance = "LIMITED CORRELATION: γ relationships weak"

    print(f"   Significance Level: {significance}")
    print(f"   Best Match Rate: {best_rate:.1f}%")
    print(f"   Compared to π⁻²: {best_rate/20.24:.2f}x weaker")
    print(f"   Mathematical Context: γ appears in Mertens' theorems")
    print(f"   Potential Applications: Number theory asymptotics")

    # Save comprehensive results
    results_file = f"gamma_analysis_{int(time.time())}.json"
    analysis_results = {
        'analysis_type': 'euler_mascheroni_gamma_analysis',
        'dataset_info': {
            'prime_limit': 2000000,
            'total_primes': len(primes),
            'total_gaps': len(gaps),
            'test_sample_size': 10000
        },
        'basic_gamma_results': basic_results,
        'composite_gamma_results': composite_results,
        'gamma_vs_pi_comparison': comparison_results,
        'best_relationship': {
            'type': best_type,
            'key': best_key,
            'match_rate': best_rate,
            'details': best_details
        },
        'scientific_summary': {
            'significance_level': significance,
            'gamma_vs_pi_ratio': best_rate / 20.24,
            'mathematical_context': 'Euler-Mascheroni constant in prime asymptotics',
            'potential_applications': ['Number theory', 'Asymptotic analysis', 'Prime distribution']
        },
        'timestamp': time.time()
    }

    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)

    print(f"\n💾 Complete analysis saved to: {results_file}")
    print("\n✅ EULER-MASCHERONI CONSTANT ANALYSIS COMPLETE")
    print("Phase 1.1 of Deep Research Expansion completed!")

    return analysis_results

if __name__ == "__main__":
    run_gamma_analysis()
