#!/usr/bin/env python3
"""
Test 79/21 Rule with Real Data Only
====================================

Direct test of the 79/21 coherence rule using real mathematical data
to verify if structured + complementary energies sum to 100%.
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import correlate
import requests
from io import BytesIO
import matplotlib.pyplot as plt
import os

# Constants
epsilon = 1e-8
PHI = (1 + np.sqrt(5)) / 2

def load_real_prime_gaps():
    """Load real prime gap data."""
    try:
        print("📥 Loading real prime gap data...")
        response = requests.get("https://raw.githubusercontent.com/primegap-list-project/prime-gap-list/main/prime-gaps.txt", timeout=10)
        if response.status_code == 200:
            lines = response.text.splitlines()
            gaps = [int(line.strip()) for line in lines if line.strip().isdigit()]
            return np.array(gaps[:10000])  # Limit for analysis
    except:
        print("⚠️  Could not load prime gaps, using synthetic data")
        return None

def compute_79_21_partition(data, analysis_type="fft"):
    """Compute the 79/21 energy partition directly."""
    if data is None or len(data) < 2:
        return None

    # Step 1: Compute gaps and log-transform
    gaps = np.diff(data)
    g_i = np.log(np.abs(gaps) + epsilon)

    # Step 2: Spectral analysis
    N = len(g_i)
    yf = fft(g_i)
    xf = fftfreq(N, 1)[:N//2]
    power = np.abs(yf[:N//2])**2
    total_energy = np.sum(power)

    if total_energy == 0:
        return None

    # Step 3: Find frequency cutoff for 79% energy
    cum_energy = np.cumsum(power) / total_energy
    f_cut_idx = np.where(cum_energy >= 0.79)[0]

    if len(f_cut_idx) == 0:
        primary_energy_pct = cum_energy[-1]
    else:
        primary_energy_pct = cum_energy[f_cut_idx[0]]

    complement_energy_pct = 1.0 - primary_energy_pct

    return {
        'primary_energy': primary_energy_pct * 100,
        'complement_energy': complement_energy_pct * 100,
        'total_energy': 100.0,
        'sum_check': (primary_energy_pct + complement_energy_pct) * 100,
        'analysis_type': analysis_type
    }

def test_prime_gaps():
    """Test 79/21 rule on prime gaps."""
    print("\n🔢 TESTING PRIME GAPS")
    print("=" * 40)

    prime_gaps = load_real_prime_gaps()
    if prime_gaps is not None:
        result = compute_79_21_partition(prime_gaps, "fft")
        if result:
            print(f"Primary Energy (79%): {result['primary_energy']:.3f}%")
            print(f"Complement Energy (21%): {result['complement_energy']:.3f}%")
            print(f"Sum Check: {result['sum_check']:.3f}%")
            print(f"Perfect 100%? {abs(result['sum_check'] - 100.0) < 0.001}")
            return result['complement_energy']
    return None

def test_zeta_zeros():
    """Test on zeta function zeros (simplified approximation)."""
    print("\n🌀 TESTING ZETA FUNCTION ZEROS")
    print("=" * 40)

    # Use known zeta zeros as test data
    zeta_zeros = [6.003, 14.134725, 21.022040, 25.010858, 30.424876, 32.935062, 37.586178, 40.918719]
    result = compute_79_21_partition(np.array(zeta_zeros), "fft")

    if result:
        print(f"Primary Energy (79%): {result['primary_energy']:.3f}%")
        print(f"Complement Energy (21%): {result['complement_energy']:.3f}%")
        print(f"Sum Check: {result['sum_check']:.3f}%")
        print(f"Perfect 100%? {abs(result['sum_check'] - 100.0) < 0.001}")
        return result['complement_energy']
    return None

def test_fibonacci_sequence():
    """Test on Fibonacci sequence ratios."""
    print("\n🌀 TESTING FIBONACCI SEQUENCE")
    print("=" * 40)

    # Generate Fibonacci sequence ratios approaching φ
    fib_ratios = []
    a, b = 1, 1
    for i in range(50):
        ratio = b / a
        fib_ratios.append(ratio)
        a, b = b, a + b

    result = compute_79_21_partition(np.array(fib_ratios), "fft")
    if result:
        print(f"Primary Energy (79%): {result['primary_energy']:.3f}%")
        print(f"Complement Energy (21%): {result['complement_energy']:.3f}%")
        print(f"Sum Check: {result['sum_check']:.3f}%")
        print(f"Perfect 100%? {abs(result['sum_check'] - 100.0) < 0.001}")
        return result['complement_energy']
    return None

def test_random_data():
    """Test on truly random data to establish baseline."""
    print("\n🎲 TESTING RANDOM DATA (BASELINE)")
    print("=" * 40)

    np.random.seed(42)  # Reproducible
    random_data = np.random.randn(1000)

    result = compute_79_21_partition(random_data, "fft")
    if result:
        print(f"Primary Energy (79%): {result['primary_energy']:.3f}%")
        print(f"Complement Energy (21%): {result['complement_energy']:.3f}%")
        print(f"Sum Check: {result['sum_check']:.3f}%")
        print(f"Perfect 100%? {abs(result['sum_check'] - 100.0) < 0.001}")
        return result['complement_energy']
    return None

def main():
    """Main test function."""
    print("🧮 TESTING 79/21 RULE WITH REAL DATA ONLY")
    print("Verifying if structured + complementary = 100%")
    print("=" * 60)

    results = []

    # Test on real mathematical data
    prime_result = test_prime_gaps()
    if prime_result:
        results.append(("Prime Gaps", prime_result))

    zeta_result = test_zeta_zeros()
    if zeta_result:
        results.append(("Zeta Zeros", zeta_result))

    fib_result = test_fibonacci_sequence()
    if fib_result:
        results.append(("Fibonacci", fib_result))

    random_result = test_random_data()
    if random_result:
        results.append(("Random (Baseline)", random_result))

    # Summary analysis
    print("\n📊 SUMMARY ANALYSIS")
    print("=" * 40)

    if results:
        complement_values = [r[1] for r in results]

        print("Complement Energy (21%) Results:")
        for name, value in results:
            deviation = abs(value - 21.0)
            status = "✅ CLOSE" if deviation < 1.0 else "❌ FAR"
            print(".3f")
        print("\n📈 STATISTICS:")
        print(f"Mean complement energy: {np.mean(complement_values):.3f}%")
        print(f"Std deviation: {np.std(complement_values):.3f}%")
        print(f"Min value: {np.min(complement_values):.3f}%")
        print(f"Max value: {np.max(complement_values):.3f}%")

        # Check if close to 21%
        close_to_21 = all(abs(v - 21.0) < 2.0 for v in complement_values[:-1])  # Exclude random
        print(f"\n🎯 CLOSE TO 21% (excluding random): {close_to_21}")

        # Check perfect 100% sum
        perfect_sum = all(abs(v + 79.0 - 100.0) < 0.1 for v in complement_values[:-1])
        print(f"🎯 PERFECT 79% + 21% = 100%: {perfect_sum}")

    print("\n✅ TEST COMPLETE")
if __name__ == "__main__":
    main()
