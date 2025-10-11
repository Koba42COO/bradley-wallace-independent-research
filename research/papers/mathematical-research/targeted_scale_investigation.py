#!/usr/bin/env python3
"""
Targeted Scale Investigation
============================

Finding the "natural scales" where specific harmonic ratios emerge in prime gaps.
Based on the rule of doubling and zeta function connections.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from scaled_analysis import ScaledWallaceAnalyzer, KNOWN_RATIOS

class TargetedScaleInvestigator:
    def __init__(self):
        self.target_scales = {
            'φ': [1618, 3236, 6472, 12944, 25888, 50000, 100000],  # Golden ratio scales
            '√2': [1414, 2828, 5656, 11312, 22624, 45248, 90496],  # Square root 2 scales
            '2.000': [2000, 4000, 8000, 16000, 32000, 64000, 128000],  # Octave scales
            '√3': [1732, 3464, 6928, 13856, 27712, 55424, 100000],  # Square root 3 scales
            '1.847': [1847, 3694, 7388, 14776, 29552, 50000, 100000],  # Pell number scales
            '2.287': [2287, 4574, 9148, 18296, 36592, 70000, 140000],  # φ×√2 scales
            '3.236': [3236, 6472, 12944, 25888, 50000, 100000, 200000],  # 2φ scales
            '1.000': [1000, 2000, 4000, 8000, 16000, 32000, 64000]  # Unity scales (doubling)
        }

        self.results = {}

    def investigate_targeted_scales(self):
        """Investigate specific scales where each harmonic ratio should emerge."""
        print("🎯 TARGETED SCALE INVESTIGATION")
        print("Finding 'natural scales' for harmonic ratio emergence")
        print("=" * 70)

        for ratio_symbol, scales in self.target_scales.items():
            ratio_data = next(r for r in KNOWN_RATIOS if r['symbol'] == ratio_symbol)
            ratio_value = ratio_data['value']

            print(f"\n🎯 Investigating {ratio_symbol} ({ratio_value:.3f}) at targeted scales:")
            print("-" * 50)

            scale_results = []
            emergence_scale = None

            for scale in scales:
                print(f"  Testing scale: {scale:,} primes...")

                try:
                    # Analyze at this scale
                    analyzer = ScaledWallaceAnalyzer(target_primes=scale)
                    analyzer.load_large_prime_dataset()
                    analyzer.compute_gaps_efficiently(analyzer.primes)

                    # Check for ratio detection
                    gaps = analyzer.gaps
                    if len(gaps) > 100000:
                        indices = np.linspace(0, len(gaps)-1, 100000, dtype=int)
                        gaps = gaps[indices]

                    detection_result = self.detect_ratio_at_scale(gaps, ratio_value, ratio_symbol)

                    result = {
                        'scale': scale,
                        'detected': detection_result['detected'],
                        'peak_ratio': detection_result['peak_ratio'],
                        'distance': detection_result['distance'],
                        'correlation': detection_result.get('correlation', 0),
                        'gap_stats': {
                            'count': len(gaps),
                            'mean': float(np.mean(gaps)),
                            'std': float(np.std(gaps))
                        }
                    }

                    scale_results.append(result)

                    # Mark emergence scale
                    if detection_result['detected'] and emergence_scale is None:
                        emergence_scale = scale

                    status = "✅ DETECTED" if result['detected'] else "❌ not found"
                    print(f"    Result: {status} (dist: {result['distance']:.4f})")

                except Exception as e:
                    print(f"    Error at scale {scale}: {e}")
                    scale_results.append({
                        'scale': scale,
                        'error': str(e)
                    })

            self.results[ratio_symbol] = {
                'ratio_value': ratio_value,
                'targeted_scales': scales,
                'results': scale_results,
                'emergence_scale': emergence_scale
            }

            # Summarize findings for this ratio
            self.summarize_ratio_findings(ratio_symbol)

        # Cross-ratio analysis
        self.analyze_cross_ratio_patterns()

        # Save comprehensive results
        self.save_targeted_investigation()

    def detect_ratio_at_scale(self, gaps, target_ratio, ratio_symbol):
        """Detect a specific ratio at a given scale using optimized methods."""
        log_gaps = np.log(gaps.astype(float) + 1e-8)

        # FFT Analysis
        fft_result = np.fft.fft(log_gaps)
        frequencies = np.fft.fftfreq(len(log_gaps))

        pos_mask = frequencies > 0
        pos_freqs = frequencies[pos_mask]
        pos_mags = np.abs(fft_result[pos_mask]) / len(log_gaps)

        # Find peaks
        peaks = self.find_peaks_efficient(pos_mags, pos_freqs, 12)  # More peaks for targeted search

        # Check for target ratio
        best_distance = float('inf')
        best_peak = None

        for peak in peaks:
            peak_ratio = np.exp(peak['frequency'])
            distance = abs(peak_ratio - target_ratio) / target_ratio

            if distance < best_distance:
                best_distance = distance
                best_peak = peak

        detected = best_distance < 0.05  # Stricter detection for targeted search

        return {
            'detected': detected,
            'peak_ratio': np.exp(best_peak['frequency']) if best_peak else None,
            'distance': best_distance,
            'magnitude': best_peak['magnitude'] if best_peak else 0,
            'correlation': best_peak['magnitude'] if best_peak else 0
        }

    def find_peaks_efficient(self, values, indices, num_peaks):
        """Efficient peak finding for targeted analysis."""
        peaks = []
        values_norm = values / np.max(values) if np.max(values) > 0 else values
        threshold = np.percentile(values_norm, 75)  # Lower threshold for more sensitivity

        min_distance = max(5, len(values) // (num_peaks * 10))
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

    def summarize_ratio_findings(self, ratio_symbol):
        """Summarize findings for a specific ratio."""
        results = self.results[ratio_symbol]
        ratio_value = results['ratio_value']
        emergence_scale = results['emergence_scale']

        detected_scales = [r['scale'] for r in results['results']
                          if isinstance(r, dict) and r.get('detected', False)]

        if detected_scales:
            print(f"📊 {ratio_symbol} ({ratio_value:.3f}) SUMMARY:")
            print(f"   ✅ Detected at scales: {detected_scales}")
            print(f"   🎯 Emergence scale: {emergence_scale:,} primes")
            print(f"   📈 Detection rate: {len(detected_scales)}/{len(results['targeted_scales'])}")
        else:
            print(f"📊 {ratio_symbol} ({ratio_value:.3f}) SUMMARY:")
            print(f"   ❌ Not detected at any targeted scales")
            print(f"   💡 May require different scale ranges or detection methods")

    def analyze_cross_ratio_patterns(self):
        """Analyze patterns across all ratios."""
        print("\n🎯 CROSS-RATIO SCALE PATTERN ANALYSIS")
        print("=" * 50)

        # Create emergence scale matrix
        emergence_data = {}
        for ratio_symbol, data in self.results.items():
            emergence_scale = data['emergence_scale']
            if emergence_scale:
                emergence_data[ratio_symbol] = emergence_scale

        if emergence_data:
            print("📊 Emergence Scales by Ratio:")
            print("Ratio | Emergence Scale | Ratio Value")
            print("-" * 35)

            for ratio_symbol, scale in sorted(emergence_data.items(), key=lambda x: x[1]):
                ratio_data = next(r for r in KNOWN_RATIOS if r['symbol'] == ratio_symbol)
                print(f"{ratio_symbol:5} | {scale:13,} | {ratio_data['value']:.3f}")

            # Analyze scaling relationships
            scales = list(emergence_data.values())
            ratios = [next(r for r in KNOWN_RATIOS if r['symbol'] == sym)['value']
                     for sym in emergence_data.keys()]

            if len(scales) >= 2:
                print("\n🔍 Emergence Scale Relationships:")
                scale_ratios = [s2/s1 for s1, s2 in zip(scales[:-1], scales[1:])]
                print(f"   Scale ratios: {[f'{r:.2f}' for r in scale_ratios]}")

                # Check for mathematical relationships
                phi = (1 + np.sqrt(5)) / 2
                sqrt2 = np.sqrt(2)

                phi_related = [r for r in scale_ratios if abs(r - phi) < 0.1]
                sqrt2_related = [r for r in scale_ratios if abs(r - sqrt2) < 0.1]
                doubling = [r for r in scale_ratios if abs(r - 2.0) < 0.1]

                if phi_related:
                    print(f"   🌀 Golden ratio relationships: {phi_related}")
                if sqrt2_related:
                    print(f"   ⚛️ Quantum relationships: {sqrt2_related}")
                if doubling:
                    print(f"   🔄 Doubling patterns: {doubling}")

        else:
            print("⚠️ No ratios detected at any targeted scales")
            print("💡 Consider expanding scale ranges or using different detection methods")

    def save_targeted_investigation(self):
        """Save comprehensive targeted investigation results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'targeted_scale_investigation_{timestamp}.json'

        # Convert numpy types
        json_results = self.make_json_serializable(self.results)

        with open(filename, 'w') as f:
            json.dump({
                'analysis_timestamp': datetime.now().isoformat(),
                'investigation_type': 'targeted_scale_investigation',
                'target_scales': self.target_scales,
                'results': json_results
            }, f, indent=2)

        print(f"\n💾 Targeted investigation saved to: {filename}")

        # Create visualization
        self.create_targeted_visualization(timestamp)

    def make_json_serializable(self, obj):
        """Convert numpy types for JSON serialization."""
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

    def create_targeted_visualization(self, timestamp):
        """Create visualization of targeted scale investigation."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Targeted Scale Investigation: Finding Natural Harmonic Scales', fontsize=16)

        # Emergence scale plot
        emergence_scales = {}
        for ratio_symbol, data in self.results.items():
            if data['emergence_scale']:
                emergence_scales[ratio_symbol] = data['emergence_scale']

        if emergence_scales:
            ratios = list(emergence_scales.keys())
            scales = [emergence_scales[r] for r in ratios]

            axes[0,0].scatter(range(len(ratios)), np.log10(scales), s=100, c='red', alpha=0.7)
            axes[0,0].set_title('Emergence Scales (log scale)')
            axes[0,0].set_xlabel('Harmonic Ratio')
            axes[0,0].set_ylabel('Scale (log₁₀)')
            axes[0,0].set_xticks(range(len(ratios)))
            axes[0,0].set_xticklabels(ratios, rotation=45)
            axes[0,0].grid(True, alpha=0.3)

        # Detection heatmap across scales
        all_scales = sorted(set(scale for scales in self.target_scales.values() for scale in scales))
        ratio_symbols = list(self.target_scales.keys())

        detection_matrix = []
        for ratio_symbol in ratio_symbols:
            row = []
            results = self.results[ratio_symbol]['results']
            scale_to_result = {r['scale']: r for r in results if 'detected' in r}

            for scale in all_scales:
                result = scale_to_result.get(scale, {})
                detected = result.get('detected', False)
                row.append(1 if detected else 0)
            detection_matrix.append(row)

        if detection_matrix:
            detection_matrix = np.array(detection_matrix)

            im = axes[0,1].imshow(detection_matrix, cmap='RdYlGn', aspect='auto')
            axes[0,1].set_title('Detection Heatmap Across Scales')
            axes[0,1].set_xlabel('Scale (primes)')
            axes[0,1].set_ylabel('Harmonic Ratio')
            axes[0,1].set_xticks(range(len(all_scales)))
            axes[0,1].set_xticklabels([f'{s:,}' for s in all_scales], rotation=45)
            axes[0,1].set_yticks(range(len(ratio_symbols)))
            axes[0,1].set_yticklabels(ratio_symbols)
            plt.colorbar(im, ax=axes[0,1], label='Detected')

        # Scale distribution histogram
        if emergence_scales:
            scales_list = list(emergence_scales.values())
            axes[1,0].hist(np.log10(scales_list), bins=10, alpha=0.7, color='blue', edgecolor='black')
            axes[1,0].set_title('Distribution of Emergence Scales')
            axes[1,0].set_xlabel('Scale (log₁₀)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].grid(True, alpha=0.3)

        # Mathematical connections
        axes[1,1].text(0.5, 0.8, 'MATHEMATICAL CONNECTIONS:\n',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=12, fontweight='bold')

        connections = [
            "• Doubling Rule: Scale ratios ≈ 2.0",
            "• Golden Ratio: Emergence at φ-related scales",
            "• Zeta Function: Functional equation symmetries",
            "• Quantum Harmonics: √2 at uncertainty scales"
        ]

        for i, conn in enumerate(connections):
            axes[1,1].text(0.5, 0.7 - i*0.1, conn,
                          transform=axes[1,1].transAxes, ha='center', va='center',
                          fontsize=10)

        axes[1,1].set_title('Theoretical Connections')
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.savefig(f'targeted_scale_investigation_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Targeted investigation visualization saved as: targeted_scale_investigation_{timestamp}.png")

def main():
    """Run targeted scale investigation."""
    investigator = TargetedScaleInvestigator()
    investigator.investigate_targeted_scales()

if __name__ == "__main__":
    main()
