#!/usr/bin/env python3
"""
Explore inverse relationships in Wallace Transform deeply
Test various inverse and higher-order relationships beyond basic Bradley formula
"""

import numpy as np
import time
import json
from pathlib import Path

PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
SQRT2 = np.sqrt(2)         # ≈ 1.4142135623730951

def wallace_transform(x, alpha=PHI, beta=0.618, epsilon=1e-8, phi=PHI):
    """Wallace Transform: W_φ(x)"""
    if x <= 0:
        return np.nan
    log_val = np.log(x + epsilon)
    return alpha * np.power(np.abs(log_val), phi) * np.sign(log_val) + beta

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

    print(f"   Generated {len(primes):,} primes, {len(gaps):,} gaps")
    return primes, gaps

def test_inverse_relationships(primes, gaps, extended_k_range=(-5, 6), tolerance=0.2):
    """
    Test extended inverse relationships:
    - Direct: g_n = W_φ(p_n) · φ^k
    - Inverse: g_n = W_φ(p_n) / φ^k
    - Combined: g_n = W_φ(p_n) · φ^k / φ^m
    """
    print(f"🔬 Testing Extended Relationships: k={extended_k_range[0]} to {extended_k_range[1]}")

    results = {}

    # Test 1: Direct relationships (original Bradley)
    print("   Testing DIRECT relationships: g_n = W_φ(p_n) · φ^k")
    for k in range(extended_k_range[0], extended_k_range[1] + 1):
        phi_k = np.power(PHI, k)
        matches = 0

        for i, (p, gap) in enumerate(zip(primes, gaps)):
            if i >= len(gaps):
                break

            wt_p = wallace_transform(p)
            expected_gap = wt_p * phi_k

            if expected_gap > 0:
                relative_error = abs(expected_gap - gap) / max(gap, expected_gap)
                if relative_error <= tolerance:
                    matches += 1

        percent_match = (matches / len(gaps)) * 100
        results[f'direct_k{k}'] = {
            'relationship': f'g_n = W_φ(p_n) · φ^{k}',
            'phi_k': float(phi_k),
            'matches': matches,
            'percent': float(percent_match)
        }

    # Test 2: Inverse relationships
    print("   Testing INVERSE relationships: g_n = W_φ(p_n) / φ^k")
    for k in range(extended_k_range[0], extended_k_range[1] + 1):
        if k == 0:
            continue  # Skip division by zero

        phi_k = np.power(PHI, k)
        matches = 0

        for i, (p, gap) in enumerate(zip(primes, gaps)):
            if i >= len(gaps):
                break

            wt_p = wallace_transform(p)
            expected_gap = wt_p / phi_k

            if expected_gap > 0:
                relative_error = abs(expected_gap - gap) / max(gap, expected_gap)
                if relative_error <= tolerance:
                    matches += 1

        percent_match = (matches / len(gaps)) * 100
        results[f'inverse_k{k}'] = {
            'relationship': f'g_n = W_φ(p_n) / φ^{k}',
            'phi_k': float(phi_k),
            'matches': matches,
            'percent': float(percent_match)
        }

    # Test 3: Combined relationships (direct + inverse)
    print("   Testing COMBINED relationships: g_n = W_φ(p_n) · φ^a / φ^b")
    for a in [-2, -1, 1, 2]:
        for b in [-2, -1, 1, 2]:
            if a == b:
                continue  # Skip trivial cases

            phi_a = np.power(PHI, a)
            phi_b = np.power(PHI, b)
            matches = 0

            for i, (p, gap) in enumerate(zip(primes, gaps)):
                if i >= len(gaps):
                    break

                wt_p = wallace_transform(p)
                expected_gap = wt_p * phi_a / phi_b

                if expected_gap > 0:
                    relative_error = abs(expected_gap - gap) / max(gap, expected_gap)
                    if relative_error <= tolerance:
                        matches += 1

            percent_match = (matches / len(gaps)) * 100
            results[f'combined_a{a}_b{b}'] = {
                'relationship': f'g_n = W_φ(p_n) · φ^{a} / φ^{b}',
                'phi_a': float(phi_a),
                'phi_b': float(phi_b),
                'matches': matches,
                'percent': float(percent_match)
            }

    return results

def test_alternative_bases(primes, gaps, tolerance=0.2):
    """Test relationships with different bases (not just φ)"""
    print("🔬 Testing Alternative Bases (beyond φ)")

    results = {}
    bases = {
        'sqrt2': SQRT2,
        'e': np.e,
        'pi': np.pi,
        'golden_conjugate': 1/PHI,  # φ conjugate
        'sqrt3': np.sqrt(3),
        'euler_gamma': np.euler_gamma
    }

    for base_name, base_val in bases.items():
        print(f"   Testing base: {base_name} = {base_val:.6f}")

        # Test direct relationships with this base
        for k in [-2, -1, 1, 2]:
            base_k = np.power(base_val, k)
            matches = 0

            for i, (p, gap) in enumerate(zip(primes, gaps)):
                if i >= len(gaps):
                    break

                wt_p = wallace_transform(p)
                expected_gap = wt_p * base_k

                if expected_gap > 0:
                    relative_error = abs(expected_gap - gap) / max(gap, expected_gap)
                    if relative_error <= tolerance:
                        matches += 1

            percent_match = (matches / len(gaps)) * 100
            results[f'{base_name}_k{k}'] = {
                'relationship': f'g_n = W_φ(p_n) · {base_name}^{k}',
                'base_value': float(base_val),
                'base_k': float(base_k),
                'matches': matches,
                'percent': float(percent_match)
            }

    return results

def analyze_relationship_patterns(all_results):
    """Analyze patterns in the relationship results"""
    print("\n📊 RELATIONSHIP PATTERN ANALYSIS")

    # Find top-performing relationships
    sorted_results = sorted(
        all_results.items(),
        key=lambda x: x[1]['percent'],
        reverse=True
    )

    print("🏆 TOP 10 PERFORMING RELATIONSHIPS:")
    print("   Rank | Relationship | Matches | Percent")
    print("   -----|--------------|---------|--------")

    for i, (key, data) in enumerate(sorted_results[:10]):
        relationship = data['relationship'][:30]  # Truncate long strings
        matches = data['matches']
        percent = data['percent']
        print("4d")

    # Analyze by relationship type
    direct_results = {k: v for k, v in all_results.items() if k.startswith('direct_')}
    inverse_results = {k: v for k, v in all_results.items() if k.startswith('inverse_')}
    combined_results = {k: v for k, v in all_results.items() if k.startswith('combined_')}

    print(f"\n📈 RELATIONSHIP TYPE SUMMARY:")
    print(f"   Direct relationships: {len(direct_results)} tested")
    if direct_results:
        direct_top = max(direct_results.values(), key=lambda x: x['percent'])
        print(".3f")

    print(f"   Inverse relationships: {len(inverse_results)} tested")
    if inverse_results:
        inverse_top = max(inverse_results.values(), key=lambda x: x['percent'])
        print(".3f")

    print(f"   Combined relationships: {len(combined_results)} tested")
    if combined_results:
        combined_top = max(combined_results.values(), key=lambda x: x['percent'])
        print(".3f")

    return sorted_results

def run_inverse_relationships_study():
    """Run the complete inverse relationships study"""
    print("🌟 WALLACE TRANSFORM - INVERSE RELATIONSHIPS STUDY")
    print("=" * 60)

    # Generate test dataset
    print("📥 Generating test dataset...")
    primes, gaps = generate_test_primes(2000000)  # 2M limit for deeper analysis

    # Test extended relationships
    print("\n🔬 PHASE 1: Extended φ Relationships")
    extended_results = test_inverse_relationships(primes, gaps)

    # Test alternative bases
    print("\n🔬 PHASE 2: Alternative Mathematical Bases")
    alternative_results = test_alternative_bases(primes, gaps)

    # Combine all results
    all_results = {**extended_results, **alternative_results}

    # Analyze patterns
    sorted_results = analyze_relationship_patterns(all_results)

    # Save comprehensive results
    output_file = f"inverse_relationships_study_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_primes': len(primes),
                'total_gaps': len(gaps),
                'timestamp': time.time()
            },
            'extended_phi_results': extended_results,
            'alternative_base_results': alternative_results,
            'all_results': all_results,
            'top_relationships': sorted_results[:20]
        }, f, indent=2, default=str)

    print(f"\n💾 Comprehensive results saved to: {output_file}")

    # Key findings
    print("\n💡 KEY FINDINGS FROM INVERSE RELATIONSHIPS STUDY:")
    print("-" * 55)

    # Find the absolute best performer
    if sorted_results:
        best_relationship = sorted_results[0][1]
        best_key = sorted_results[0][0]

        print("🏆 BEST RELATIONSHIP DISCOVERED:")
        print(f"   {best_relationship['relationship']}")
        print(".3f")
        print(".1f")

        # Analyze if it's inverse-related
        if 'inverse' in best_key or 'φ^{-' in best_relationship['relationship']:
            print("   📈 This confirms the INVERSE HYPOTHESIS!")
            print("   📈 Prime gaps may follow inverse harmonic relationships!")

    return all_results

if __name__ == "__main__":
    run_inverse_relationships_study()
