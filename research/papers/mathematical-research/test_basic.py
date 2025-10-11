#!/usr/bin/env python3
"""
Basic test for Wallace Transform Analysis
"""

import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)

def generate_primes(limit):
    """Generate primes using sieve."""
    sieve = np.zeros(limit + 1, dtype=bool)
    sieve[2:] = True

    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = False

    return np.where(sieve)[0]

def test_basic_analysis():
    """Test basic spectral analysis."""
    print("🧪 Testing Basic Wallace Transform Analysis")
    print("=" * 50)

    # Generate small prime set
    print("Generating primes...")
    primes = generate_primes(100000)  # 100k limit
    print(f"Generated {len(primes)} primes up to {primes[-1]}")

    # Compute gaps
    gaps = np.diff(primes)
    print(f"Computed {len(gaps)} gaps, range: {np.min(gaps)} - {np.max(gaps)}")

    # Log transform
    log_gaps = np.log(gaps.astype(float) + 1e-8)
    print(".4f")

    # Basic FFT
    print("Running FFT...")
    N = len(log_gaps)
    fft_result = fft(log_gaps)

    # Get positive frequencies
    freqs = fftfreq(N)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]
    pos_mags = np.abs(fft_result[pos_mask]) / N

    # Find top peaks
    num_peaks = 8
    sorted_indices = np.argsort(pos_mags)[::-1][:num_peaks]

    print("Top frequency peaks:")
    print("Rank | Frequency | Magnitude | Ratio (exp(f))")
    print("-" * 50)

    for i, idx in enumerate(sorted_indices):
        freq = pos_freqs[idx]
        mag = pos_mags[idx]
        ratio = np.exp(freq)
        print(f"{i+1:4d} | {freq:.6f} | {mag:.4f} | {ratio:.4f}")

    # Find peaks near known ratios
    known_ratios = [1.0, PHI, SQRT2, 2.0]
    print("\nChecking known ratios:")
    for ratio in known_ratios:
        target_freq = np.log(ratio)
        # Find closest frequency
        closest_idx = np.argmin(np.abs(pos_freqs - target_freq))
        closest_freq = pos_freqs[closest_idx]
        closest_mag = pos_mags[closest_idx]
        distance = abs(closest_freq - target_freq)

        match = "✓" if distance < 0.1 else "✗"
        print(f"  {ratio:.1f} → f={target_freq:.3f}, closest_f={closest_freq:.3f}, dist={distance:.3f} {match}")

    print("\n✅ Basic analysis completed successfully!")

if __name__ == "__main__":
    test_basic_analysis()
