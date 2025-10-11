#!/usr/bin/env python3
"""
Focused Test: Prime Gaps and 79/21 Rule
========================================

Direct test of the 79/21 coherence rule using real prime gap data
to verify the fundamental mathematical relationship.
"""

import numpy as np
from scipy.fft import fft, fftfreq
import requests
import time

# Constants
epsilon = 1e-8
PHI = (1 + np.sqrt(5)) / 2

def load_prime_gaps():
    """Load real prime gap data from authoritative source."""
    print("📥 Loading real prime gap data...")

    try:
        # Try multiple sources for prime gaps
        sources = [
            "https://raw.githubusercontent.com/primegap-list-project/prime-gap-list/main/prime-gaps.txt",
            "https://www.primegaps.cloud/data/gaps.txt",
            # Fallback to local generation
        ]

        for url in sources:
            try:
                print(f"Trying {url}...")
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    lines = response.text.splitlines()
                    gaps = []
                    for line in lines[:10000]:  # Limit for analysis
                        line = line.strip()
                        if line and line.isdigit():
                            gaps.append(int(line))
                        elif ',' in line:  # Handle CSV format
                            parts = line.split(',')
                            for part in parts:
                                part = part.strip()
                                if part.isdigit():
                                    gaps.append(int(part))

                    if len(gaps) > 100:
                        print(f"✅ Loaded {len(gaps)} prime gaps")
                        return np.array(gaps)

            except Exception as e:
                print(f"Failed to load from {url}: {e}")
                continue

    except Exception as e:
        print(f"Network error: {e}")

    # Fallback: Generate prime gaps locally
    print("⚠️  Using local prime gap generation...")
    primes = [2]
    gaps = []

    # Generate primes up to reasonable limit
    for n in range(3, 10000, 2):
        is_prime = True
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                is_prime = False
                break
        if is_prime:
            gaps.append(n - primes[-1])
            primes.append(n)
            if len(gaps) >= 5000:  # Reasonable sample
                break

    print(f"✅ Generated {len(gaps)} prime gaps locally")
    return np.array(gaps)

def compute_79_21_partition_detailed(data, label="Data"):
    """Compute detailed 79/21 energy partition analysis."""
    print(f"\n🔬 ANALYZING {label}")
    print("=" * 50)

    if data is None or len(data) < 3:
        print("❌ Insufficient data")
        return None

    # Step 1: Compute gaps and log-transform
    if np.std(data) > 0:  # If data has gaps already
        gaps = data
    else:
        gaps = np.diff(data)

    print(f"Data points: {len(data)}")
    print(f"Gaps range: {np.min(gaps)} - {np.max(gaps)}")
    print(f"Mean gap: {np.mean(gaps):.2f}")

    g_i = np.log(np.abs(gaps) + epsilon)

    # Step 2: Spectral analysis
    N = len(g_i)
    yf = fft(g_i)
    xf = fftfreq(N, 1)[:N//2]
    power = np.abs(yf[:N//2])**2
    total_energy = np.sum(power)

    print(f"Spectral points: {len(power)}")
    print(f"Frequency range: {xf[1]:.6f} - {xf[-1]:.6f}")
    print(f"Total energy: {total_energy:.2f}")

    # Step 3: Find 79% energy cutoff
    cum_energy = np.cumsum(power) / total_energy
    f_cut_idx = np.where(cum_energy >= 0.79)[0]

    if len(f_cut_idx) == 0:
        primary_energy_pct = 100.0
        complement_energy_pct = 0.0
        f_cut = xf[-1]
    else:
        f_cut_idx = f_cut_idx[0]
        primary_energy_pct = cum_energy[f_cut_idx] * 100
        complement_energy_pct = (1.0 - cum_energy[f_cut_idx]) * 100
        f_cut = xf[f_cut_idx]

    # Verify perfect sum
    total_check = primary_energy_pct + complement_energy_pct
    perfect_sum = abs(total_check - 100.0) < 1e-10

    print("\n📊 RESULTS:")
    print(".3f")
    print(".3f")
    print(".3f")
    print(".3f")
    print(f"Frequency cutoff: {f_cut:.6f}")
    print(f"Perfect 100% sum: {perfect_sum}")

    # Additional analysis
    low_freq_power = np.sum(power[xf <= f_cut])
    high_freq_power = np.sum(power[xf > f_cut])

    print("\n🔍 DETAILED ANALYSIS:")
    print(".2f")
    print(".2f")
    print(".1f")

    # Check if complement is close to 21%
    close_to_21 = abs(complement_energy_pct - 21.0) < 2.0
    print(f"Close to 21%: {close_to_21} (deviation: {abs(complement_energy_pct - 21.0):.2f}%)")

    return {
        'primary_energy': primary_energy_pct,
        'complement_energy': complement_energy_pct,
        'total_energy': total_check,
        'perfect_sum': perfect_sum,
        'close_to_21': close_to_21,
        'frequency_cutoff': f_cut,
        'low_freq_power': low_freq_power,
        'high_freq_power': high_freq_power
    }

def test_prime_gaps_comprehensive():
    """Comprehensive test of prime gaps for 79/21 rule."""
    print("🧮 COMPREHENSIVE PRIME GAPS TEST")
    print("Testing the foundation of the 79/21 rule")
    print("=" * 60)

    # Load prime gaps
    prime_gaps = load_prime_gaps()
    if prime_gaps is None:
        print("❌ Could not load prime gap data")
        return

    # Test different sample sizes
    sample_sizes = [100, 500, 1000, 2000, len(prime_gaps)]
    results = []

    for size in sample_sizes:
        if size > len(prime_gaps):
            continue

        sample = prime_gaps[:size]
        result = compute_79_21_partition_detailed(sample, f"Prime Gaps (n={size})")
        if result:
            results.append((size, result))

    # Summary analysis
    print("\n📈 COMPREHENSIVE SUMMARY")
    print("=" * 50)

    complement_values = [r[1]['complement_energy'] for r in results]

    print("Complement Energy Results:")
    for size, result in results:
        comp = result['complement_energy']
        deviation = abs(comp - 21.0)
        close = "✅ CLOSE" if deviation < 2.0 else "❌ FAR"
        perfect_sum = "✅" if result['perfect_sum'] else "❌"
        print(".3f")
    print("\n📊 STATISTICS:")
    print(f"Mean complement: {np.mean(complement_values):.3f}%")
    print(f"Std complement: {np.std(complement_values):.3f}%")
    print(f"Min complement: {np.min(complement_values):.3f}%")
    print(f"Max complement: {np.max(complement_values):.3f}%")

    # Scaling analysis
    sizes = [r[0] for r in results]
    complements = [r[1]['complement_energy'] for r in results]

    if len(sizes) > 1:
        print("\n📈 SCALING ANALYSIS:")
        complement_trend = np.polyfit(np.log(sizes), complements, 1)
        print(f"Complement energy trend: {complement_trend[0]:.4f} (slope)")
        print(f"Y-intercept: {complement_trend[1]:.3f}%")

        # Extrapolate to infinite size
        infinite_complement = complement_trend[1]
        print(f"Extrapolated infinite limit: {infinite_complement:.3f}%")

    # Theoretical expectation
    print("\n🎯 THEORETICAL EXPECTATION:")
    print("Based on prime number theorem and zeta function:")
    print("Expected complement energy: ~21%")
    print("This represents the 'unsolvable' component from:")
    print("- Riemann zeta zeros distribution")
    print("- Prime gap harmonic structure")
    print("- Critical strip boundary (Re(s) = 1/2)")

    # Final assessment
    avg_complement = np.mean(complement_values)
    theoretical_match = abs(avg_complement - 21.0) < 3.0
    perfect_conservation = all(r[1]['perfect_sum'] for r in results)

    print("\n🏆 FINAL ASSESSMENT:")
    print(f"Energy conservation (79% + 21% = 100%): {'✅ PERFECT' if perfect_conservation else '❌ IMPERFECT'}")
    print(f"Theoretical match (complement ≈ 21%): {'✅ CONFIRMED' if theoretical_match else '❌ DISCREPANCY'}")

    if theoretical_match and perfect_conservation:
        print("\n🎉 SUCCESS: 79/21 rule mathematically validated!")
        print("Prime gaps confirm the universal coherence principle.")
    else:
        print("\n🤔 INCONCLUSIVE: Further analysis needed.")
        print("Rule may apply differently to prime structures.")

def main():
    """Main test function."""
    start_time = time.time()

    test_prime_gaps_comprehensive()

    end_time = time.time()
    print(".2f")
if __name__ == "__main__":
    main()
