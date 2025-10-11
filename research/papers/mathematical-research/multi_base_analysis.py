#!/usr/bin/env python3
"""
Multi-Base Wallace Transform Analysis
Tests self-referential hypothesis: Does log base determine scaling constant?
"""

import numpy as np
import json
import time
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Mathematical constants
CONSTANTS = {
    'e': np.e,
    'π': np.pi,
    'φ': (1 + np.sqrt(5)) / 2,
    'γ': 0.5772156649015329,
    '√2': np.sqrt(2)
}

PHI = (1 + np.sqrt(5)) / 2
EPSILON = 1e-12

def is_prime(n):
    """Check if number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_primes(limit):
    """Generate list of primes up to limit count"""
    primes = []
    n = 2
    while len(primes) < limit:
        if is_prime(n):
            primes.append(n)
        n += 1
    return primes

def wallace_transform(x, log_base):
    """Multi-base Wallace Transform"""
    log_conversion = np.log(log_base)
    log_val = np.log(x + EPSILON) / log_conversion
    return PHI * np.power(np.abs(log_val), PHI) * np.sign(log_val) + 1.0

def test_combination(primes, gaps, log_base, scaling_constant, test_size=5000):
    """Test single log_base × scaling_constant combination"""
    scaling_factor = np.power(scaling_constant, -2)
    matches = 0
    tolerance = 0.20

    for i in range(min(test_size, len(gaps))):
        actual_gap = gaps[i]
        p = primes[i]
        wt_p = wallace_transform(p, log_base)
        predicted = wt_p * scaling_factor

        if abs(actual_gap - predicted) / max(actual_gap, predicted) <= tolerance:
            matches += 1

    match_rate = (matches / min(test_size, len(gaps))) * 100
    return round(match_rate, 2)

def run_multi_base_analysis():
    """Run complete multi-base analysis"""
    print("🔬 MULTI-BASE WALLACE TRANSFORM ANALYSIS")
    print("=" * 50)
    print("Testing self-referential hypothesis: Does log base determine scaling constant?")
    print()

    # Generate dataset
    print("📊 Generating dataset...")
    primes = generate_primes(25000)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    print(f"✅ Generated {len(primes)} primes, {len(gaps)} gaps")
    print()

    # Test matrix
    log_bases = list(CONSTANTS.keys())
    scaling_constants = list(CONSTANTS.keys())

    print("🎯 Testing all combinations...")
    print(f"Matrix: {len(log_bases)} log bases × {len(scaling_constants)} scaling constants")
    print(f"Total combinations: {len(log_bases) * len(scaling_constants)}")
    print()

    results_matrix = {}
    start_time = time.time()

    for i, log_base_name in enumerate(log_bases):
        log_base_value = CONSTANTS[log_base_name]
        results_matrix[log_base_name] = {}

        for j, scale_const_name in enumerate(scaling_constants):
            scale_const_value = CONSTANTS[scale_const_name]

            print(f"Testing {log_base_name} logs × {scale_const_name}⁻²...", end=' ')

            match_rate = test_combination(primes, gaps, log_base_value, scale_const_value)
            results_matrix[log_base_name][scale_const_name] = match_rate

            print(f"{match_rate}%")

    total_time = time.time() - start_time
    print(f"⏱️  Total computation time: {total_time:.1f}s")
    # Analysis
    print("\n🔍 ANALYZING RESULTS...")
    print("-" * 30)

    # Extract diagonal (self-referential) values
    diagonal_values = []
    for base in log_bases:
        if base in results_matrix[base]:
            diagonal_values.append(results_matrix[base][base])

    # Extract off-diagonal values
    off_diagonal_values = []
    for log_base in log_bases:
        for scale_const in scaling_constants:
            if log_base != scale_const:
                off_diagonal_values.append(results_matrix[log_base][scale_const])

    # Statistics
    avg_diagonal = np.mean(diagonal_values)
    avg_off_diagonal = np.mean(off_diagonal_values)
    max_diagonal = max(diagonal_values)
    max_overall = max([max(row.values()) for row in results_matrix.values()])

    # Find best combinations
    best_overall = {'rate': 0, 'log_base': '', 'constant': ''}
    for log_base in log_bases:
        for scale_const in scaling_constants:
            rate = results_matrix[log_base][scale_const]
            if rate > best_overall['rate']:
                best_overall = {'rate': rate, 'log_base': log_base, 'constant': scale_const}

    # Self-referential test
    self_referential = bool(avg_diagonal > avg_off_diagonal + 1.0)  # 1% threshold

    print("\n📊 STATISTICAL ANALYSIS:")
    print(f"Average diagonal (self-ref): {avg_diagonal:.2f}%")
    print(f"Average off-diagonal: {avg_off_diagonal:.2f}%")
    print(f"Max diagonal: {max_diagonal:.2f}%")
    print(f"Max overall: {max_overall:.2f}%")
    print(f"Self-referential pattern: {'YES' if self_referential else 'NO'}")
    print(f"Best overall: {best_overall['log_base']} logs × {best_overall['constant']}⁻² = {best_overall['rate']}%")

    # Detailed matrix
    print("\n🎯 RESULTS MATRIX:")
    print("Log Base → Scaling Constant ↓")
    print("        " + " ".join([f"{c:>6}" for c in scaling_constants]))
    for log_base in log_bases:
        row = [f"{results_matrix[log_base][const]:>6.1f}" for const in scaling_constants]
        diagonal_marker = " ←" if log_base in scaling_constants and log_base == log_base else ""
        print(f"{log_base:>6}  {' '.join(row)}{diagonal_marker}")

    # Save results
    results_data = {
        'metadata': {
            'analysis_type': 'multi_base_wallace_transform',
            'timestamp': datetime.now().isoformat(),
            'primes_count': len(primes),
            'gaps_count': len(gaps),
            'log_bases_tested': log_bases,
            'scaling_constants_tested': scaling_constants,
            'total_combinations': len(log_bases) * len(scaling_constants),
            'computation_time': round(total_time, 2)
        },
        'results_matrix': results_matrix,
        'analysis': {
            'diagonal_values': [float(x) for x in diagonal_values],
            'off_diagonal_values': [float(x) for x in off_diagonal_values],
            'avg_diagonal': float(round(avg_diagonal, 2)),
            'avg_off_diagonal': float(round(avg_off_diagonal, 2)),
            'max_diagonal': float(max_diagonal),
            'max_overall': float(max_overall),
            'best_combination': best_overall,
            'self_referential': bool(self_referential),
            'self_referential_strength': float(round(avg_diagonal - avg_off_diagonal, 2))
        }
    }

    filename = f"multi_base_analysis_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(results_data, f, indent=2)

    print(f"\n💾 Results saved to: {filename}")

    # Create visualization
    create_visualization(results_matrix, log_bases, scaling_constants)

    return results_matrix, results_data

def create_visualization(matrix, log_bases, scaling_constants):
    """Create heatmap visualization"""
    # Prepare data for heatmap
    data = np.zeros((len(log_bases), len(scaling_constants)))
    for i, log_base in enumerate(log_bases):
        for j, scale_const in enumerate(scaling_constants):
            data[i, j] = matrix[log_base][scale_const]

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlBu_r',
                xticklabels=[f'{c}⁻²' for c in scaling_constants],
                yticklabels=[f'log({b})' for b in log_bases])

    plt.title('Multi-Base Wallace Transform Results\nLog Base vs Scaling Constant Match Rates (%)')
    plt.xlabel('Scaling Constant (c⁻²)')
    plt.ylabel('Logarithm Base (log_c)')

    # Highlight diagonal (self-referential)
    for i in range(min(len(log_bases), len(scaling_constants))):
        plt.plot([i+0.5, i+0.5], [i+0.5, len(log_bases)-i+0.5],
                color='gold', linewidth=3, alpha=0.7)

    plt.tight_layout()
    plt.savefig('multi_base_analysis_heatmap.png', dpi=300, bbox_inches='tight')
    print("📊 Heatmap saved as: multi_base_analysis_heatmap.png")

    # Create bar chart comparison
    plt.figure(figsize=(12, 6))

    x = np.arange(len(log_bases))
    width = 0.15

    for i, scale_const in enumerate(scaling_constants):
        values = [matrix[log_base][scale_const] for log_base in log_bases]
        plt.bar(x + i*width, values, width, label=f'{scale_const}⁻²', alpha=0.8)

    plt.xlabel('Logarithm Base')
    plt.ylabel('Match Rate (%)')
    plt.title('Multi-Base Wallace Transform: Match Rates by Combination')
    plt.xticks(x + width*2, [f'log({b})' for b in log_bases])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('multi_base_analysis_bars.png', dpi=300, bbox_inches='tight')
    print("📊 Bar chart saved as: multi_base_analysis_bars.png")

if __name__ == "__main__":
    print("🌟 STARTING MULTI-BASE WALLACE TRANSFORM ANALYSIS")
    print("Testing the self-referential hypothesis...")
    print("=" * 60)

    results_matrix, results_data = run_multi_base_analysis()

    print("\n" + "=" * 60)
    print("🎉 ANALYSIS COMPLETE!")
    print("=" * 60)

    # Final summary
    analysis = results_data['analysis']
    if analysis['self_referential']:
        print("🌀 SELF-REFERENTIAL PATTERN DETECTED!")
        print(f"   Diagonal advantage: {analysis['self_referential_strength']:.2f}%")
    else:
        print("🎯 NO STRONG SELF-REFERENTIAL PATTERN")
        print(f"   Diagonal advantage: {analysis['self_referential_strength']:.2f}%")
    print(f"\n🏆 Best combination: {analysis['best_combination']['log_base']} logs × {analysis['best_combination']['constant']}⁻² = {analysis['best_combination']['rate']}%")

    print("\n📊 Visualizations created:")
    print("   - multi_base_analysis_heatmap.png (correlation matrix)")
    print("   - multi_base_analysis_bars.png (comparison chart)")

    print(f"\n📄 Full results saved to JSON file")
    print("\n🔬 HYPOTHESIS TESTED: Does log base determine scaling constant?")
    print(f"   Result: {'CONFIRMED' if analysis['self_referential'] else 'NOT CONFIRMED'}")
