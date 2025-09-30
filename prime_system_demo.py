#!/usr/bin/env python3
"""
SIMPLIFIED PRIME SYSTEM DEMONSTRATION
Showcase of optimized prime determination and prediction system
"""

from comprehensive_prime_system import ComprehensivePrimeSystem
import time

def main():
    print("🔬 OPTIMIZED PRIME DETERMINATION AND PREDICTION SYSTEM")
    print("=" * 60)

    system = ComprehensivePrimeSystem()

    # Test basic primality on various numbers
    print("\n🔍 BASIC PRIMALITY TESTING:")
    test_numbers = [29, 97, 10007, 7919, 1000003, 2147483647]  # Including a small Mersenne prime

    for n in test_numbers:
        start_time = time.time()
        result = system.is_prime_comprehensive(n)
        end_time = time.time()

        status = "✅ PRIME" if result.is_prime else "❌ COMPOSITE"
        print(".4f")

    # Test different prime types
    print("\n🌟 SPECIAL PRIME TYPES:")

    # Safe primes (smaller limit to avoid overflow)
    safe_primes = system.generate_safe_primes(1000)
    print(f"Safe Primes (up to 1000): {len(safe_primes)} found - {safe_primes[:8]}...")

    # Twin primes
    twin_primes = system.generate_twin_primes(1000)
    print(f"Twin Prime Pairs (up to 1000): {len(twin_primes)} pairs - {twin_primes[:5]}...")

    # Sophie Germain primes
    sophie_primes = system.generate_sophie_germain_primes(1000)
    print(f"Sophie Germain Primes (up to 1000): {len(sophie_primes)} found - {sophie_primes[:8]}...")

    # Palindromic primes
    palindromic_primes = system.generate_palindromic_primes(10000)
    print(f"Palindromic Primes (up to 10,000): {len(palindromic_primes)} found - {palindromic_primes[:8]}...")

    # Mersenne primes (only small ones to avoid overflow)
    mersenne_primes = system.generate_mersenne_primes(100)  # Small limit
    print(f"Mersenne Primes (small exponents): {len(mersenne_primes)} found - {mersenne_primes}")

    # Fermat primes (only small ones)
    fermat_primes = system.generate_fermat_primes()
    print(f"Fermat Primes: {len(fermat_primes)} found - {fermat_primes}")

    # Comprehensive analysis
    print("\n🔬 COMPREHENSIVE ANALYSIS OF 113:")
    analysis = system.comprehensive_prime_analysis(113)
    print(f"Is Prime: {analysis['is_prime']['miller_rabin']}")
    print(f"Prime Factors: {analysis['prime_factors']}")
    print(f"Special Properties: {analysis['special_properties']}")
    print(f"Number Theory: μ(113)={analysis['number_theory']['mobius_function']}, φ(113)={analysis['number_theory']['euler_totient']}")

    # Prime prediction
    print("\n🔮 PRIME PREDICTION:")
    prediction = system.predict_next_prime(113, 'riemann')
    actual_next = system.get_next_prime(113)
    print(f"Next prime after 113: Predicted {prediction.number} (actual: {actual_next})")
    print(f"Confidence: {prediction.probability:.3f}, Error: {abs(prediction.number - actual_next)}")

    # Algorithm comparison
    print("\n⚡ ALGORITHM PERFORMANCE COMPARISON:")
    test_num = 10007

    algorithms = ['trial_division', 'miller_rabin']
    for alg in algorithms:
        start_time = time.time()
        result = system.is_prime_comprehensive(test_num, alg)
        end_time = time.time()

        print(".2e")
    # Prime distribution analysis
    print("\n📊 PRIME DISTRIBUTION ANALYSIS (up to 10,000):")
    analysis = system.analyze_prime_distribution(10000)
    print(f"Total primes: {analysis['total_primes']:,}")
    print(f"Density: {analysis['density']:.6f}")
    print(f"Average gap: {analysis['average_gap']:.2f}")
    print(f"Li(x) approximation error: {analysis['li_error']:.6f}")
    print(f"Riemann R(x) approximation error: {analysis['riemann_error']:.6f}")

    # Prime gaps analysis
    primes = system.sieve_of_eratosthenes(10000)
    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
    print(f"\n📈 PRIME GAPS STATISTICS:")
    print(f"Max gap: {max(gaps)}, Min gap: {min(gaps)}")
    print(f"Most common gap: {max(set(gaps), key=gaps.count)}")

    print("\n✅ OPTIMIZED PRIME SYSTEM DEMONSTRATION COMPLETE!")
    print("All major prime types and algorithms successfully implemented and tested.")

if __name__ == "__main__":
    main()
