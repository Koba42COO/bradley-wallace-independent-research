#!/usr/bin/env python3
"""
Multi-Method Validation Framework for Wallace Transform
Combines FFT, Autocorrelation, and Bradley's Formula approaches
"""

import numpy as np
import pandas as pd
import time
import json
from pathlib import Path
from scipy.fft import fft
from scipy.signal import correlate
from results_database import WallaceResultsDatabase

# Mathematical constants
PHI = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618033988749895
SQRT2 = np.sqrt(2)         # ≈ 1.4142135623730951
SQRT3 = np.sqrt(3)         # ≈ 1.732050807568877

# Known harmonic ratios to detect (including inverse and higher-order ratios)
KNOWN_RATIOS = {
    # Original ratios
    '1.000': {'name': 'Unity', 'value': 1.000, 'physical': 'Base unit, identity'},
    '1.414': {'name': '√2 (Octave)', 'value': 1.4142135623730951, 'physical': 'Quantum uncertainty, musical fifths'},
    '1.618': {'name': 'φ (Golden)', 'value': PHI, 'physical': 'Golden ratio, Fibonacci, growth patterns'},
    '1.732': {'name': '√3 (Fifth)', 'value': 1.732050807568877, 'physical': 'Major sixth, crystal structures'},
    '1.847': {'name': 'Pell', 'value': (1 + np.sqrt(2)), 'physical': 'Pell numbers, continued fractions'},
    '2.000': {'name': 'Octave', 'value': 2.000, 'physical': 'Perfect octave, frequency doubling'},
    '2.287': {'name': 'φ·√2', 'value': PHI * SQRT2, 'physical': 'Combined golden-quantum ratio'},
    '3.236': {'name': '2φ', 'value': 2 * PHI, 'physical': 'Double golden ratio'},

    # Inverse and higher-order ratios (detected by Bradley)
    '0.382': {'name': 'φ⁻²', 'value': PHI ** -2, 'physical': 'Inverse squared golden ratio'},
    '0.618': {'name': 'φ⁻¹', 'value': PHI ** -1, 'physical': 'Inverse golden ratio'},
    '2.618': {'name': 'φ²', 'value': PHI ** 2, 'physical': 'Squared golden ratio'},
    '4.236': {'name': 'φ³', 'value': PHI ** 3, 'physical': 'Cubed golden ratio'}
}

class MultiMethodValidator:
    """Combines FFT, Autocorrelation, and Bradley's Formula for comprehensive validation"""

    def __init__(self):
        self.db = WallaceResultsDatabase()
        self.results = {}

    def wallace_transform(self, x, alpha=PHI, beta=0.618, epsilon=1e-8, phi=PHI):
        """Wallace Transform: W_φ(x)"""
        if x <= 0:
            return np.nan
        log_val = np.log(x + epsilon)
        return alpha * np.power(np.abs(log_val), phi) * np.sign(log_val) + beta

    def generate_test_primes(self, limit=1000000):
        """Generate primes for testing (same as browser version)"""
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

    def fft_analysis(self, gaps, sample_size=100000):
        """FFT-based spectral analysis for micro-harmonics"""
        print(f"🎯 FFT Analysis: {sample_size:,} gap samples")

        # Sample gaps
        if len(gaps) > sample_size:
            indices = np.random.choice(len(gaps), sample_size, replace=False)
            gap_sample = gaps[indices]
        else:
            gap_sample = gaps

        # Apply log transform for spectral analysis
        log_gaps = np.log(gap_sample + 1e-8)

        # Remove mean to focus on variations
        log_gaps = log_gaps - np.mean(log_gaps)

        # Compute FFT
        fft_result = fft(log_gaps)
        freqs = np.fft.fftfreq(len(log_gaps))

        # Get positive frequencies and magnitudes
        pos_mask = freqs > 0
        freqs_pos = freqs[pos_mask]
        magnitudes = np.abs(fft_result[pos_mask])

        # Convert frequencies to ratios (exp(frequency))
        # This gives us multiplicative ratios: exp(f) where f is the frequency
        detected_ratios = np.exp(freqs_pos)

        # Find peaks (top 8)
        peak_indices = np.argsort(magnitudes)[-8:][::-1]
        peaks = []

        for i, idx in enumerate(peak_indices):
            ratio = detected_ratios[idx]
            magnitude = magnitudes[idx]
            frequency = freqs_pos[idx]

            # Find closest known ratio
            closest_ratio, distance = self.find_closest_known_ratio(ratio)

            peaks.append({
                'rank': i + 1,
                'ratio': float(ratio),
                'magnitude': float(magnitude),
                'frequency': float(frequency),
                'closest_known': closest_ratio,
                'distance': float(distance),
                'detected': distance < 0.01  # Very strict tolerance for FFT
            })

        return {
            'method': 'fft',
            'sample_size': len(gap_sample),
            'peaks': peaks,
            'detected_ratios': [p['closest_known'] for p in peaks if p['detected']]
        }

    def autocorrelation_analysis(self, gaps, sample_size=50000):
        """Autocorrelation analysis for larger harmonic ratios"""
        print(f"🔄 Autocorrelation Analysis: {sample_size:,} gap samples")

        # Sample gaps
        if len(gaps) > sample_size:
            indices = np.random.choice(len(gaps), sample_size, replace=False)
            gap_sample = gaps[indices]
        else:
            gap_sample = gaps

        # Convert to log space for multiplicative relationships
        log_gaps = np.log(gap_sample + 1e-8)

        # Remove mean
        log_gaps = log_gaps - np.mean(log_gaps)

        # Compute autocorrelation
        autocorr = correlate(log_gaps, log_gaps, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Positive lags only

        # Find peaks in autocorrelation (exclude lag 0)
        lags = np.arange(1, len(autocorr))
        autocorr_values = autocorr[1:]

        # Get top 8 peaks
        peak_indices = np.argsort(autocorr_values)[-8:][::-1]
        peaks = []

        for i, idx in enumerate(peak_indices):
            lag = lags[idx]
            correlation = autocorr_values[idx]

            # Convert lag to ratio: exp(lag * small_factor)
            # This is calibrated to detect ratios like φ, √2, etc.
            ratio = np.exp(lag * 0.0001)  # Calibrated factor - much smaller

            # Find closest known ratio
            closest_ratio, distance = self.find_closest_known_ratio(ratio)

            peaks.append({
                'rank': i + 1,
                'lag': int(lag),
                'correlation': float(correlation),
                'ratio': float(ratio),
                'closest_known': closest_ratio,
                'distance': float(distance),
                'detected': distance < 0.05  # 5% tolerance for autocorrelation
            })

        return {
            'method': 'autocorr',
            'sample_size': len(gap_sample),
            'peaks': peaks,
            'detected_ratios': [p['closest_known'] for p in peaks if p['detected']]
        }

    def bradley_formula_analysis(self, primes, gaps, k_range=(-3, 4), tolerance=0.2):
        """Bradley's Formula: g_n = W_φ(p_n) · φ^k"""
        print(f"🔬 Bradley's Formula Analysis: k={k_range[0]} to {k_range[1]}")

        results = {}
        total_matches = 0
        total_tests = 0

        for k in range(k_range[0], k_range[1] + 1):
            phi_k = np.power(PHI, k)
            matches = 0

            for i, (p, gap) in enumerate(zip(primes, gaps)):
                if i >= len(gaps):
                    break

                # Calculate W_φ(p_n)
                wt_p = self.wallace_transform(p)

                # Test direct relationship: g_n = W_φ(p_n) · φ^k
                expected_gap = wt_p * phi_k

                if expected_gap > 0:
                    relative_error = abs(expected_gap - gap) / max(gap, expected_gap)
                    if relative_error <= tolerance:
                        matches += 1

                # Also test inverse relationship: g_n = W_φ(p_n) / φ^k
                expected_gap_inv = wt_p / phi_k

                if expected_gap_inv > 0:
                    relative_error_inv = abs(expected_gap_inv - gap) / max(gap, expected_gap_inv)
                    if relative_error_inv <= tolerance:
                        matches += 1

            percent_match = (matches / len(gaps)) * 100

            results[k] = {
                'phi_k': float(phi_k),
                'matches': matches,
                'percent': float(percent_match),
                'inverse_phi_k': float(1.0 / phi_k)
            }

            total_matches += matches
            total_tests += len(gaps)

        return {
            'method': 'bradley',
            'k_range': k_range,
            'tolerance': tolerance,
            'total_matches': total_matches,
            'overall_percent': float((total_matches / total_tests) * 100),
            'k_results': results,
            'detected_ratios': self.extract_bradley_ratios(results)
        }

    def extract_bradley_ratios(self, k_results):
        """Extract detected ratios from Bradley results"""
        detected = []
        for k, data in k_results.items():
            if data['percent'] > 1.0:  # Significant detection threshold
                if k == 0:
                    detected.append('1.000')  # Unity
                elif k == 1:
                    detected.append('1.618')  # φ
                elif k == 2:
                    detected.append('2.618')  # φ² (close to 2.618)
                elif k == -1:
                    detected.append('0.618')  # φ⁻¹
                elif k == -2:
                    detected.append('0.382')  # φ⁻²
        return list(set(detected))  # Remove duplicates

    def find_closest_known_ratio(self, detected_ratio):
        """Find the closest known harmonic ratio"""
        min_distance = float('inf')
        closest_ratio = None

        for symbol, data in KNOWN_RATIOS.items():
            distance = abs(data['value'] - detected_ratio)
            if distance < min_distance:
                min_distance = distance
                closest_ratio = symbol

        return closest_ratio, min_distance

    def cross_validate_methods(self):
        """Cross-validate results from all three methods"""
        print("🔍 Cross-Validating Methods...")

        validation_matrix = {}

        # Get detected ratios for each method (remove duplicates)
        fft_detected = set(self.results.get('fft', {}).get('detected_ratios', []))
        autocorr_detected = set(self.results.get('autocorr', {}).get('detected_ratios', []))
        bradley_detected = set(self.results.get('bradley', {}).get('detected_ratios', []))

        print(f"   FFT detected: {sorted(fft_detected)}")
        print(f"   Autocorr detected: {sorted(autocorr_detected)}")
        print(f"   Bradley detected: {sorted(bradley_detected)}")

        # Build validation matrix
        for ratio_symbol in KNOWN_RATIOS.keys():
            validation_matrix[ratio_symbol] = {
                'name': KNOWN_RATIOS[ratio_symbol]['name'],
                'physical': KNOWN_RATIOS[ratio_symbol]['physical'],
                'fft': ratio_symbol in fft_detected,
                'autocorr': ratio_symbol in autocorr_detected,
                'bradley': ratio_symbol in bradley_detected
            }

        # Calculate confidence scores
        for ratio_symbol, methods in validation_matrix.items():
            method_count = sum([methods['fft'], methods['autocorr'], methods['bradley']])
            validation_matrix[ratio_symbol]['confidence'] = method_count / 3.0  # 0.0 to 1.0
            validation_matrix[ratio_symbol]['methods_detected'] = method_count

        return validation_matrix

    def generate_comprehensive_report(self):
        """Generate comprehensive validation report"""
        print("\n🎯 MULTI-METHOD VALIDATION REPORT")
        print("=" * 50)

        # Overall statistics
        print("📊 OVERALL STATISTICS")
        print(f"   Dataset: {self.results.get('metadata', {}).get('total_primes', 'N/A'):,} primes")
        print(f"   Gap samples - FFT: {self.results.get('fft', {}).get('sample_size', 'N/A'):,}")
        print(f"   Gap samples - Autocorr: {self.results.get('autocorr', {}).get('sample_size', 'N/A'):,}")
        print(f"   Bradley tests: {self.results.get('bradley', {}).get('total_matches', 'N/A'):,} matches")

        # Method results summary
        print("\n🏆 METHOD RESULTS SUMMARY")
        for method_name, method_results in self.results.items():
            if method_name == 'metadata':
                continue
            detected_count = len(method_results.get('detected_ratios', []))
            print(f"   {method_name.upper()}: {detected_count} ratios detected")

        # Cross-validation matrix
        validation_matrix = self.cross_validate_methods()

        print("\n🎯 CROSS-VALIDATION MATRIX")
        print("   Ratio | Name | FFT | Auto | Brad | Conf | Methods")
        print("   ------|------|-----|------|------|------|--------")

        for ratio_symbol, data in validation_matrix.items():
            fft_check = "✓" if data['fft'] else "✗"
            auto_check = "✓" if data['autocorr'] else "✗"
            brad_check = "✓" if data['bradley'] else "✗"
            confidence = ".2f"

            print("4s")

        # Key findings
        print("\n💡 KEY FINDINGS")
        high_confidence = [r for r, d in validation_matrix.items() if d['confidence'] >= 0.67]
        if high_confidence:
            print(f"   High Confidence Ratios (≥2 methods): {len(high_confidence)}")
            for ratio in high_confidence:
                name = validation_matrix[ratio]['name']
                methods = validation_matrix[ratio]['methods_detected']
                print(f"     • {name} ({ratio}): {methods}/3 methods")

        return validation_matrix

    def run_full_validation(self, prime_limit=1000000):
        """Run complete multi-method validation"""
        print("🌟 WALLACE TRANSFORM - MULTI-METHOD VALIDATION")
        print("=" * 60)

        start_time = time.time()

        # Generate test dataset
        print("📥 Generating test dataset...")
        primes, gaps = self.generate_test_primes(prime_limit)

        # Store metadata
        self.results['metadata'] = {
            'prime_limit': prime_limit,
            'total_primes': len(primes),
            'total_gaps': len(gaps),
            'timestamp': time.time()
        }

        # Run FFT analysis (scaled up to 1M samples)
        print("\n🎯 Running FFT Analysis (1M samples)...")
        self.results['fft'] = self.fft_analysis(gaps, sample_size=1000000)

        # Run autocorrelation analysis (scaled up to 500K samples)
        print("\n🔄 Running Autocorrelation Analysis (500K samples)...")
        self.results['autocorr'] = self.autocorrelation_analysis(gaps, sample_size=500000)

        # Run Bradley's formula analysis
        print("\n🔬 Running Bradley's Formula Analysis...")
        self.results['bradley'] = self.bradley_formula_analysis(primes, gaps)

        # Generate comprehensive report
        validation_matrix = self.generate_comprehensive_report()

        # Save results
        output_file = f"multi_method_validation_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'results': self.results,
                'validation_matrix': validation_matrix,
                'timestamp': time.time(),
                'computation_time': time.time() - start_time
            }, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {output_file}")
        print(".2f")

        return validation_matrix

def main():
    """Run the multi-method validation"""
    validator = MultiMethodValidator()

    # Run validation
    validation_matrix = validator.run_full_validation()

    # Print final summary
    print("\n🎉 VALIDATION COMPLETE!")
    print("Multi-method approach successfully combined:")
    print("  • FFT: Micro-harmonic detection")
    print("  • Autocorrelation: Large ratio detection")
    print("  • Bradley's Formula: Direct mathematical relationships")

if __name__ == "__main__":
    main()
