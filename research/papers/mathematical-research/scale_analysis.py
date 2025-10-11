#!/usr/bin/env python3
"""
Scale-Dependent Harmonic Analysis
=================================

Investigating how harmonic ratios manifest across different scales.
Testing the "rule of doubling" hypothesis in prime gap harmonics.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import json
from datetime import datetime

# Import our framework
from scaled_analysis import ScaledWallaceAnalyzer, KNOWN_RATIOS

class ScaleHarmonicAnalyzer:
    def __init__(self):
        self.scales = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]
        self.scale_results = {}

    def analyze_scale_dependence(self):
        """Analyze how harmonic detection changes across scales."""
        print("🔬 SCALE-DEPENDENT HARMONIC ANALYSIS")
        print("Testing the 'rule of doubling' in prime gap harmonics")
        print("=" * 60)

        for scale in self.scales:
            print(f"\n🎯 Testing scale: {scale:,} primes")
            print("-" * 40)

            # Create analyzer for this scale
            analyzer = ScaledWallaceAnalyzer(target_primes=scale)

            # Load/generate primes
            analyzer.load_large_prime_dataset()
            analyzer.compute_gaps_efficiently(analyzer.primes)

            # Analyze at this scale
            results = self.analyze_single_scale(analyzer, scale)

            self.scale_results[scale] = results

            # Print scale-specific results
            self.print_scale_results(scale, results)

        # Cross-scale analysis
        self.analyze_cross_scale_patterns()

        # Save comprehensive results
        self.save_scale_analysis()

    def analyze_single_scale(self, analyzer, scale):
        """Analyze harmonics at a specific scale."""
        # Sample gaps for analysis
        gaps = analyzer.gaps
        if len(gaps) > 100000:  # Limit for performance
            indices = np.linspace(0, len(gaps)-1, 100000, dtype=int)
            gaps = gaps[indices]

        # FFT Analysis
        log_gaps = np.log(gaps.astype(float) + 1e-8)
        fft_result = np.fft.fft(log_gaps)
        frequencies = np.fft.fftfreq(len(log_gaps))

        # Get positive frequencies
        pos_mask = frequencies > 0
        pos_freqs = frequencies[pos_mask]
        pos_mags = np.abs(fft_result[pos_mask]) / len(log_gaps)

        # Find peaks
        peaks = self.find_peaks_efficient(pos_mags, pos_freqs, 8)

        # Convert to ratios
        for peak in peaks:
            peak['ratio'] = np.exp(peak['frequency'])
            peak['closest_ratio'], peak['distance'] = self.find_closest_ratio(peak['ratio'])
            peak['match'] = peak['distance'] < 0.05

        # Ratio detection summary
        detected_ratios = {}
        for ratio_data in KNOWN_RATIOS:
            ratio = ratio_data['value']
            detected = any(abs(p['ratio'] - ratio) / ratio < 0.05 for p in peaks if p['match'])
            detected_ratios[ratio_data['symbol']] = {
                'ratio': ratio,
                'detected': detected,
                'peak_count': sum(1 for p in peaks if abs(p['ratio'] - ratio) / ratio < 0.05)
            }

        return {
            'scale': scale,
            'total_gaps': len(gaps),
            'fft_peaks': peaks,
            'detected_ratios': detected_ratios,
            'gap_stats': {
                'min': float(np.min(gaps)),
                'max': float(np.max(gaps)),
                'mean': float(np.mean(gaps)),
                'std': float(np.std(gaps))
            }
        }

    def find_peaks_efficient(self, values, indices, num_peaks):
        """Efficient peak finding."""
        peaks = []
        values_norm = values / np.max(values) if np.max(values) > 0 else values
        threshold = np.percentile(values_norm, 80)
        min_distance = max(10, len(values) // (num_peaks * 20))
        last_peak_idx = -min_distance

        for i in range(1, len(values_norm) - 1):
            if i - last_peak_idx < min_distance:
                continue

            if (values_norm[i] > values_norm[i-1] and
                values_norm[i] > values_norm[i+1] and
                values_norm[i] > threshold):

                peaks.append({
                    'index': i,
                    'frequency': float(indices[i]),
                    'value': float(values[i]),
                    'magnitude': float(values[i]),
                    'rank': len(peaks) + 1
                })
                last_peak_idx = i

                if len(peaks) >= num_peaks:
                    break

        return peaks

    def find_closest_ratio(self, target_ratio):
        """Find closest known ratio."""
        min_dist = float('inf')
        closest = None

        for ratio in KNOWN_RATIOS:
            dist = abs(target_ratio - ratio['value'])
            if dist < min_dist:
                min_dist = dist
                closest = ratio

        return closest, min_dist

    def print_scale_results(self, scale, results):
        """Print results for a specific scale."""
        detected = results['detected_ratios']

        print(f"Scale {scale:,} primes:")
        print(f"  Gaps analyzed: {results['total_gaps']:,}")
        print(f"  Gap range: {results['gap_stats']['min']:.0f} - {results['gap_stats']['max']:.0f}")
        print(f"  Mean gap: {results['gap_stats']['mean']:.2f}")

        print("  Detected ratios:")
        for symbol, info in detected.items():
            status = "✓" if info['detected'] else "✗"
            print(f"    {symbol}: {status} ({info['ratio']:.3f})")

        total_detected = sum(1 for info in detected.values() if info['detected'])
        print(f"  Total detected: {total_detected}/{len(KNOWN_RATIOS)}")

    def analyze_cross_scale_patterns(self):
        """Analyze patterns across different scales."""
        print("\n🎯 CROSS-SCALE PATTERN ANALYSIS")
        print("=" * 50)

        # Create detection matrix
        print("\n📊 Scale-Detection Matrix:")
        print("Scale".ljust(8), end="")
        for ratio in KNOWN_RATIOS:
            print(f"{ratio['symbol']:>5}", end="")
        print()

        for scale in self.scales:
            results = self.scale_results[scale]
            print(f"{scale:,}".ljust(8), end="")
            for ratio_data in KNOWN_RATIOS:
                symbol = ratio_data['symbol']
                detected = results['detected_ratios'][symbol]['detected']
                mark = "✓" if detected else "✗"
                print(f"{mark:>5}", end="")
            print()

        # Analyze scaling patterns
        print("\n🔍 Scaling Pattern Analysis:")

        # Check for doubling/halving patterns
        for ratio_data in KNOWN_RATIOS:
            symbol = ratio_data['symbol']
            ratio = ratio_data['value']

            # Find scales where this ratio is detected
            detected_scales = []
            for scale in self.scales:
                if self.scale_results[scale]['detected_ratios'][symbol]['detected']:
                    detected_scales.append(scale)

            if detected_scales:
                print(f"  {symbol} ({ratio:.3f}): Detected at scales {detected_scales}")

                # Check for doubling pattern
                if len(detected_scales) >= 2:
                    ratios = [s2/s1 for s1, s2 in zip(detected_scales[:-1], detected_scales[1:])]
                    doubling_ratios = [r for r in ratios if abs(r - 2.0) < 0.5]  # Close to doubling
                    if doubling_ratios:
                        print(f"    → Possible doubling pattern: {doubling_ratios}")

    def save_scale_analysis(self):
        """Save comprehensive scale analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'scale_dependence_analysis_{timestamp}.json'

        # Convert numpy types for JSON
        json_results = self.make_json_serializable(self.scale_results)

        with open(filename, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'scales_tested': self.scales,
                'scale_results': json_results,
                'known_ratios': KNOWN_RATIOS
            }, f, indent=2)

        print(f"\n💾 Scale analysis saved to: {filename}")

        # Create visualization
        self.create_scale_visualization(timestamp)

    def make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable."""
        if isinstance(obj, dict):
            return {k: self.make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    def create_scale_visualization(self, timestamp):
        """Create visualization of scale-dependent patterns."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Scale-Dependent Harmonic Analysis\nRule of Doubling Investigation', fontsize=16)

        # Detection heatmap
        detection_matrix = []
        for scale in self.scales:
            row = []
            for ratio_data in KNOWN_RATIOS:
                symbol = ratio_data['symbol']
                detected = self.scale_results[scale]['detected_ratios'][symbol]['detected']
                row.append(1 if detected else 0)
            detection_matrix.append(row)

        detection_matrix = np.array(detection_matrix)

        im = axes[0,0].imshow(detection_matrix, cmap='RdYlGn', aspect='auto')
        axes[0,0].set_title('Harmonic Ratio Detection by Scale')
        axes[0,0].set_xlabel('Harmonic Ratios')
        axes[0,0].set_ylabel('Scale (log)')
        axes[0,0].set_xticks(range(len(KNOWN_RATIOS)))
        axes[0,0].set_xticklabels([r['symbol'] for r in KNOWN_RATIOS])
        axes[0,0].set_yticks(range(len(self.scales)))
        axes[0,0].set_yticklabels([f'{s:,}' for s in self.scales])
        plt.colorbar(im, ax=axes[0,0], label='Detected (1=Yes, 0=No)')

        # Scale vs detection count
        scale_labels = [f'{s:,}' for s in self.scales]
        detection_counts = [sum(1 for info in results['detected_ratios'].values() if info['detected'])
                           for results in self.scale_results.values()]

        axes[0,1].plot(range(len(self.scales)), detection_counts, 'bo-', linewidth=2, markersize=8)
        axes[0,1].set_title('Detection Count vs Scale')
        axes[0,1].set_xlabel('Scale')
        axes[0,1].set_ylabel('Ratios Detected')
        axes[0,1].set_xticks(range(len(self.scales)))
        axes[0,1].set_xticklabels(scale_labels, rotation=45)
        axes[0,1].grid(True, alpha=0.3)

        # Gap statistics vs scale
        scale_nums = [results['scale'] for results in self.scale_results.values()]
        mean_gaps = [results['gap_stats']['mean'] for results in self.scale_results.values()]

        axes[1,0].loglog(scale_nums, mean_gaps, 'ro-', linewidth=2, markersize=8)
        axes[1,0].set_title('Mean Gap Size vs Scale')
        axes[1,0].set_xlabel('Scale (primes)')
        axes[1,0].set_ylabel('Mean Gap Size')
        axes[1,0].grid(True, alpha=0.3)

        # Zeta function connection visualization
        axes[1,1].text(0.5, 0.7, 'Zeta Function Connection:\nζ(s) = ζ(1-s) × 2(2π)^{s-1} sin(πs/2) Γ(1-s)',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        axes[1,1].text(0.5, 0.3, 'Prime Gap Harmonics:\nScale-dependent doubling patterns\nobserved in ratio detection',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

        axes[1,1].set_title('Zeta Function ↔ Prime Harmonics Connection')
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.savefig(f'scale_dependence_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Scale analysis visualization saved as: scale_dependence_analysis_{timestamp}.png")

def main():
    """Run scale dependence analysis."""
    analyzer = ScaleHarmonicAnalyzer()
    analyzer.analyze_scale_dependence()

if __name__ == "__main__":
    main()
