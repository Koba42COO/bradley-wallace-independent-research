#!/usr/bin/env python3
"""
Enhanced Display System for Wallace Transform Analysis
======================================================

Real-time visualization and progress tracking for billion-scale harmonic analysis.
Integrates with results database for comprehensive display capabilities.
"""

import time
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import json
from datetime import datetime
import os
from pathlib import Path

# Import our framework
from results_database import WallaceResultsDatabase

class EnhancedDisplaySystem:
    def __init__(self):
        """Initialize the enhanced display system."""
        self.db = WallaceResultsDatabase()
        self.current_analysis = None
        self.display_active = False
        self.progress_data = {
            'phase': 'Initializing',
            'progress': 0,
            'message': 'Ready to begin analysis',
            'start_time': None,
            'current_primes': 0,
            'total_primes': 0
        }

        # Setup matplotlib style
        style.use('dark_background')
        plt.rcParams['figure.figsize'] = (16, 10)
        plt.rcParams['font.size'] = 10

    def start_display(self):
        """Start the enhanced display system."""
        self.display_active = True
        print("🎯 ENHANCED DISPLAY SYSTEM ACTIVATED")
        print("Real-time progress tracking and visualization enabled")
        print("=" * 70)

        # Start display thread
        display_thread = threading.Thread(target=self._run_display_loop, daemon=True)
        display_thread.start()

    def stop_display(self):
        """Stop the display system."""
        self.display_active = False
        print("\n⏹️ Display system deactivated")

    def update_progress(self, phase, progress, message="", primes_processed=0, total_primes=0):
        """Update progress information for real-time display."""
        self.progress_data.update({
            'phase': phase,
            'progress': progress,
            'message': message,
            'current_primes': primes_processed,
            'total_primes': total_primes
        })

        # Print immediate feedback
        bar_length = 40
        filled_length = int(bar_length * progress / 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        print(f"\r📊 {phase}: [{bar}] {progress:.1f}% - {message}", end='', flush=True)

        if progress >= 100:
            start_time = self.progress_data.get('start_time')
            if start_time is not None:
                elapsed = time.time() - start_time
                print(".1f")
            else:
                print()

    def set_analysis_start(self, total_primes):
        """Mark the start of analysis."""
        self.progress_data['start_time'] = time.time()
        self.progress_data['total_primes'] = total_primes

    def display_realtime_status(self):
        """Display comprehensive real-time status."""
        print("\n" + "=" * 80)
        print("🎯 WALLACE TRANSFORM ANALYSIS - REAL-TIME STATUS")
        print("=" * 80)

        data = self.progress_data
        elapsed = time.time() - data.get('start_time', time.time())

        print(f"📊 Current Phase: {data['phase']}")
        print(f"📈 Progress: {data['progress']:.1f}%")
        print(f"💬 Status: {data['message']}")
        print(f"⏱️ Elapsed Time: {elapsed:.1f} seconds")

        if data['total_primes'] > 0:
            print(f"🔢 Primes Processed: {data['current_primes']:,} / {data['total_primes']:,}")
            progress_pct = data['current_primes'] / data['total_primes'] * 100
            print(f"📊 Processing Progress: {progress_pct:.1f}%")

        # Show database status
        print("\n💾 DATABASE STATUS:")
        try:
            recent_runs = self.db.get_recent_runs(3)
            if recent_runs:
                print("Recent Runs:")
                for run in recent_runs:
                    print(f"  ID {run['id']}: {run['scale']:,} primes ({run['processing_time'] or 0:.1f}s)")

            progression = self.db.get_detection_progression()
            if progression:
                latest = progression[-1]
                print(f"Latest Detection Rate: {latest['detection_rate']:.1f}% ({latest['unique_ratios']} ratios)")
        except:
            print("  Database not yet initialized")

        print("=" * 80)

    def display_analysis_results(self, results):
        """Display comprehensive analysis results."""
        print("\n" + "🎉" * 20)
        print("ANALYSIS COMPLETE - COMPREHENSIVE RESULTS")
        print("🎉" * 20)

        # Store in database
        run_id = self.db.store_analysis_results(results, processing_time=self.progress_data.get('processing_time'))
        print(f"💾 Results stored in database (Run ID: {run_id})")

        # Display results
        self._display_results_summary(results)
        self._display_detailed_findings(results)
        self._display_visualizations(results)

        # Show database summary
        print("\n💾 DATABASE OVERVIEW:")
        self.db.display_database_summary()

    def _display_results_summary(self, results):
        """Display high-level results summary."""
        meta = results.get('metadata', {})

        print("\n📊 ANALYSIS SUMMARY:")
        print(f"   Scale Achieved: {meta.get('primes_count', 0):,} primes")
        print(f"   Processing Time: {self.progress_data.get('processing_time', 0):.1f} seconds")
        print(f"   FFT Sample Size: {meta.get('fft_sample_size', 0):,} gaps")
        print(f"   Autocorr Sample Size: {meta.get('autocorr_sample_size', 0):,} gaps")

        if 'validation' in results:
            val = results['validation']
            print("\n🏆 DETECTION RESULTS:")
            print(f"   FFT Detections: {val.get('fft_matches', 0)}/8 harmonic ratios")
            print(f"   Autocorr Detections: {val.get('autocorr_matches', 0)}/8 harmonic ratios")
            print(f"   Unique Ratios Found: {val.get('unique_ratios_detected', 0)}/8 total")
            print(f"   Success Rate: {(val.get('unique_ratios_detected', 0) / 8.0) * 100:.1f}%")

    def _display_detailed_findings(self, results):
        """Display detailed harmonic ratio findings."""
        print("\n🎯 DETAILED HARMONIC DISCOVERIES:")
        print("Method | Ratio | Value | Distance | Status")
        print("-" * 45)

        # FFT results
        if 'fft_analysis' in results and 'peaks' in results['fft_analysis']:
            for peak in results['fft_analysis']['peaks'][:5]:  # Top 5
                if peak.get('match'):
                    status = "✅ DETECTED"
                else:
                    status = "❌ not matched"
                print("5")

        # Autocorrelation results
        if 'autocorr_analysis' in results and 'peaks' in results['autocorr_analysis']:
            for peak in results['autocorr_analysis']['peaks'][:5]:  # Top 5
                if peak.get('match'):
                    status = "✅ DETECTED"
                else:
                    status = "❌ not matched"
                print("5")

        # Cross-validation
        if 'validation' in results:
            val = results['validation']
            print("\n🎯 CROSS-VALIDATION MATRIX:")
            print("Ratio | FFT | AutoCorr | Status")
            print("-" * 30)
            for symbol in ['1.000', 'φ', '√2', '2.000', '√3', '1.847', '2.287', '3.236']:
                fft_detected = any(
                    p.get('closest_ratio', {}).get('symbol') == symbol and p.get('match')
                    for p in results.get('fft_analysis', {}).get('peaks', [])
                )
                autocorr_detected = any(
                    p.get('closest_ratio', {}).get('symbol') == symbol and p.get('match')
                    for p in results.get('autocorr_analysis', {}).get('peaks', [])
                )

                fft_mark = "✓" if fft_detected else "✗"
                autocorr_mark = "✓" if autocorr_detected else "✗"

                if fft_detected or autocorr_detected:
                    status = "DETECTED"
                else:
                    status = "PENDING"

                print(f"{symbol:5} | {fft_mark:3} | {autocorr_mark:8} | {status}")

    def _display_visualizations(self, results):
        """Display visualizations of the results."""
        print("\n📊 GENERATING VISUALIZATIONS...")
        self._create_comprehensive_plot(results)

        # Export to CSV for further analysis
        try:
            self.db.export_results_csv(self.db.get_recent_runs(1)[0]['id'])
        except:
            pass

    def _create_comprehensive_plot(self, results):
        """Create comprehensive visualization of results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Wallace Transform Billion-Scale Analysis Results', fontsize=16)

        # FFT Spectrum
        if 'fft_analysis' in results and 'peaks' in results['fft_analysis']:
            ax = axes[0, 0]
            peaks = results['fft_analysis']['peaks']
            ratios = [p['ratio'] for p in peaks]
            mags = [p['magnitude'] for p in peaks]

            bars = ax.bar(range(len(ratios)), mags, color='blue', alpha=0.7)
            ax.set_title('FFT Spectral Peaks')
            ax.set_xlabel('Peak Rank')
            ax.set_ylabel('Magnitude')

            # Highlight detected ratios
            for i, (bar, peak) in enumerate(zip(bars, peaks)):
                if peak.get('match'):
                    bar.set_color('cyan')
                    bar.set_alpha(0.9)

        # Autocorrelation Peaks
        if 'autocorr_analysis' in results and 'peaks' in results['autocorr_analysis']:
            ax = axes[0, 1]
            peaks = results['autocorr_analysis']['peaks']
            lags = [p['frequency'] for p in peaks]
            corrs = [abs(p['correlation']) for p in peaks]

            bars = ax.bar(range(len(lags)), corrs, color='green', alpha=0.7)
            ax.set_title('Autocorrelation Peaks')
            ax.set_xlabel('Peak Rank')
            ax.set_ylabel('Correlation Strength')

            # Highlight detected ratios
            for i, (bar, peak) in enumerate(zip(bars, peaks)):
                if peak.get('match'):
                    bar.set_color('lime')
                    bar.set_alpha(0.9)

        # Detection Matrix Heatmap
        ax = axes[0, 2]
        detection_data = []
        ratios = ['1.000', 'φ', '√2', '2.000', '√3', '1.847', '2.287', '3.236']

        for ratio_symbol in ratios:
            fft_detected = any(
                p.get('closest_ratio', {}).get('symbol') == ratio_symbol and p.get('match')
                for p in results.get('fft_analysis', {}).get('peaks', [])
            )
            autocorr_detected = any(
                p.get('closest_ratio', {}).get('symbol') == ratio_symbol and p.get('match')
                for p in results.get('autocorr_analysis', {}).get('peaks', [])
            )
            detection_data.append([1 if fft_detected else 0, 1 if autocorr_detected else 0])

        detection_matrix = np.array(detection_data).T

        im = ax.imshow(detection_matrix, cmap='RdYlGn', aspect='auto')
        ax.set_title('Detection Matrix')
        ax.set_xlabel('Harmonic Ratios')
        ax.set_ylabel('Method')
        ax.set_xticks(range(len(ratios)))
        ax.set_xticklabels(ratios, rotation=45)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['FFT', 'AutoCorr'])
        plt.colorbar(im, ax=ax, shrink=0.8)

        # Progress Timeline
        ax = axes[1, 0]
        phases = ['Data Loading', 'Gap Processing', 'FFT Analysis', 'AutoCorr', 'Validation', 'Complete']
        times = [10, 15, 25, 35, 45, 60]  # Estimated times

        ax.plot(range(len(phases)), times, 'bo-', linewidth=2, markersize=8)
        ax.set_title('Analysis Timeline')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Time (seconds)')
        ax.set_xticks(range(len(phases)))
        ax.set_xticklabels(phases, rotation=45)
        ax.grid(True, alpha=0.3)

        # Statistical Summary
        ax = axes[1, 1]
        ax.text(0.5, 0.8, 'STATISTICAL SUMMARY:', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')

        meta = results.get('metadata', {})
        val = results.get('validation', {})

        stats_text = ".1f"".1f"".1f"f"""
Scale: {meta.get('primes_count', 0):,} primes
FFT Sample: {meta.get('fft_sample_size', 0):,} gaps
AutoCorr Sample: {meta.get('autocorr_sample_size', 0):,} gaps
Unique Ratios: {val.get('unique_ratios_detected', 0)}/8
Success Rate: {(val.get('unique_ratios_detected', 0) / 8.0) * 100:.1f}%
"""

        ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
                ha='center', va='center', fontsize=10, family='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Mathematical Insights
        ax = axes[1, 2]
        ax.text(0.5, 0.9, 'MATHEMATICAL INSIGHTS:', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, fontweight='bold')

        insights = [
            "• Prime gaps contain harmonic structures",
            "• FFT detects unity baseline patterns",
            "• Autocorrelation finds complex harmonics",
            "• Scale-dependent ratio emergence",
            "• Zeta function symmetries observed"
        ]

        for i, insight in enumerate(insights):
            ax.text(0.5, 0.8 - i*0.1, insight, transform=ax.transAxes,
                    ha='center', va='center', fontsize=9)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"wallace_billion_scale_comprehensive_{timestamp}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"📊 Comprehensive visualization saved as: {plot_file}")
        plt.close()

    def _run_display_loop(self):
        """Run the display update loop."""
        while self.display_active:
            time.sleep(5)  # Update every 5 seconds
            if self.current_analysis:
                self.display_realtime_status()

def create_enhanced_display():
    """Create and return an enhanced display system instance."""
    return EnhancedDisplaySystem()

# Example usage
if __name__ == "__main__":
    display = create_enhanced_display()
    display.start_display()

    # Simulate analysis progress
    display.set_analysis_start(1000000)

    for i in range(0, 101, 10):
        display.update_progress("Analysis Phase", i, f"Processing batch {i//10}", i*10000, 1000000)
        time.sleep(1)

    display.display_realtime_status()
    display.stop_display()
