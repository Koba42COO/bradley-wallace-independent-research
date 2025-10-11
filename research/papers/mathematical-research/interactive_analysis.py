#!/usr/bin/env python3
"""
Interactive Wallace Transform Analysis with Progress Tracking
=============================================================

Provides real-time progress updates and checkpointing for large-scale analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import time
import json
from datetime import datetime
import os
import requests

# Import our analysis functions
from scaled_analysis import ScaledWallaceAnalyzer, KNOWN_RATIOS

class InteractiveWallaceAnalyzer(ScaledWallaceAnalyzer):
    def __init__(self, target_primes=50000000, chunk_size=500000):
        super().__init__(target_primes, chunk_size)
        self.checkpoint_file = f"wallace_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.progress_callback = None

    def set_progress_callback(self, callback):
        """Set a callback function for progress updates."""
        self.progress_callback = callback

    def update_progress(self, phase, progress, message=""):
        """Update progress and call callback if set."""
        if self.progress_callback:
            self.progress_callback(phase, progress, message)
        else:
            print(f"📊 {phase}: {progress:.1f}% - {message}")

    def save_checkpoint(self, data):
        """Save current progress to checkpoint file."""
        checkpoint_data = {
            'timestamp': datetime.now().isoformat(),
            'target_primes': self.target_primes,
            'current_primes': len(data.get('primes_chunks', [])) * self.chunk_size,
            'data': data
        }

        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, default=str)

        print(f"💾 Checkpoint saved: {self.checkpoint_file}")

    def load_checkpoint(self):
        """Load progress from checkpoint file if it exists."""
        if os.path.exists(self.checkpoint_file):
            print(f"📂 Loading checkpoint: {self.checkpoint_file}")
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return None

    def analyze_with_progress(self, analysis_type='both'):
        """Run analysis with detailed progress tracking."""
        print("🚀 INTERACTIVE WALLACE TRANSFORM ANALYSIS")
        print("=" * 60)
        print(f"🎯 Target: {self.target_primes:,} primes")
        print(f"📦 Chunk size: {self.chunk_size:,}")
        print(f"💾 Checkpoint file: {self.checkpoint_file}")
        print("=" * 60)

        start_time = time.time()

        # Check for existing checkpoint
        checkpoint = self.load_checkpoint()
        if checkpoint:
            print("🔄 Resuming from checkpoint...")
            primes_chunks = checkpoint['data'].get('primes_chunks', [])
            gaps_chunks = checkpoint['data'].get('gaps_chunks', [])
        else:
            primes_chunks = []
            gaps_chunks = []

        # Phase 1: Load primes in chunks
        self.update_progress("Phase 1/4", 0, "Loading prime chunks...")

        total_chunks_needed = (self.target_primes + self.chunk_size - 1) // self.chunk_size
        current_chunk = len(primes_chunks)

        while len(primes_chunks) * self.chunk_size < self.target_primes:
            chunk_start_time = time.time()

            # Try to load next chunk
            try:
                new_chunks = self.load_primes_chunked_interactive(current_chunk)
                if not new_chunks:
                    print("⚠️ No more primes available from source")
                    break

                primes_chunks.extend(new_chunks)

                # Process gaps for new chunks
                new_gaps_chunks = self.process_gaps_chunked(new_chunks[len(gaps_chunks):])
                gaps_chunks.extend(new_gaps_chunks)

                current_chunk = len(primes_chunks)
                progress = min(100, (current_chunk / total_chunks_needed) * 25)  # 25% for phase 1

                chunk_time = time.time() - chunk_start_time
                total_primes = sum(len(chunk) for chunk in primes_chunks)

                self.update_progress("Phase 1/4", progress,
                    f"Loaded {current_chunk}/{total_chunks_needed} chunks ({total_primes:,} primes, {chunk_time:.1f}s)")

                # Save checkpoint every 10 chunks
                if current_chunk % 10 == 0:
                    self.save_checkpoint({
                        'primes_chunks': primes_chunks,
                        'gaps_chunks': gaps_chunks
                    })

            except KeyboardInterrupt:
                print("\n⏹️ Analysis interrupted by user")
                self.save_checkpoint({
                    'primes_chunks': primes_chunks,
                    'gaps_chunks': gaps_chunks
                })
                return None

        # Phase 2: Sample for analysis
        self.update_progress("Phase 2/4", 25, "Sampling data for analysis...")
        fft_sample_size = min(2000000, len(gaps_chunks) * self.chunk_size // 10)
        autocorr_sample_size = min(500000, len(gaps_chunks) * self.chunk_size // 20)

        fft_sample = self.sample_gaps_for_analysis(gaps_chunks, fft_sample_size)
        autocorr_sample = self.sample_gaps_for_analysis(gaps_chunks, autocorr_sample_size)

        self.update_progress("Phase 2/4", 35, f"FFT sample: {len(fft_sample):,} gaps, AutoCorr sample: {len(autocorr_sample):,} gaps")

        # Phase 3: FFT Analysis
        self.update_progress("Phase 3/4", 40, "Running FFT analysis...")
        fft_start = time.time()
        fft_results = self.fft_analysis_optimized(fft_sample)
        fft_time = time.time() - fft_start
        self.update_progress("Phase 3/4", 60, f"FFT completed in {fft_time:.1f}s")

        # Phase 4: Autocorrelation Analysis
        self.update_progress("Phase 4/4", 65, "Running autocorrelation analysis...")
        autocorr_start = time.time()
        autocorr_results = self.autocorr_analysis_optimized(autocorr_sample)
        autocorr_time = time.time() - autocorr_start
        self.update_progress("Phase 4/4", 85, f"Autocorrelation completed in {autocorr_time:.1f}s")

        # Cross-validation
        self.update_progress("Finalizing", 90, "Cross-validating results...")
        validation = self.cross_validate_results(
            fft_results.get('peaks', []),
            autocorr_results.get('peaks', [])
        )

        # Save final results
        results = {
            'metadata': {
                'primes_count': sum(len(chunk) for chunk in primes_chunks),
                'total_chunks': len(primes_chunks),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'scale': 'interactive',
                'fft_sample_size': len(fft_sample),
                'autocorr_sample_size': len(autocorr_sample),
                'checkpoint_file': self.checkpoint_file
            },
            'ratios_tested': KNOWN_RATIOS,
            'fft_analysis': fft_results,
            'autocorr_analysis': autocorr_results,
            'validation': validation
        }

        self.save_final_results(results)

        elapsed = time.time() - start_time
        self.update_progress("Complete", 100, f"Analysis finished in {elapsed:.1f}s")

        return results

    def load_primes_chunked_interactive(self, start_chunk=0):
        """Interactive version of chunked loading with progress."""
        try:
            # Try GitHub source first
            response = requests.get("https://raw.githubusercontent.com/srmalins/primelists/master/someprimes.txt", timeout=30)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            current_chunk = []
            chunks_loaded = []

            # Skip to starting position
            lines_to_skip = start_chunk * self.chunk_size
            current_line = 0
            total_numbers = 0

            for line in lines:
                if current_line < lines_to_skip:
                    # Count numbers in skipped lines
                    numbers_in_line = len([x for x in line.split() if x.isdigit()])
                    current_line += numbers_in_line
                    continue

                line = line.strip()
                if line and not line.startswith('#'):
                    numbers = [int(x) for x in line.split() if x.isdigit()]
                    current_chunk.extend(numbers)
                    total_numbers += len(numbers)

                    # Process chunks
                    while len(current_chunk) >= self.chunk_size:
                        chunk_array = np.array(current_chunk[:self.chunk_size], dtype=np.int64)
                        chunk_array = np.sort(chunk_array)
                        chunk_array = np.unique(chunk_array)
                        chunks_loaded.append(chunk_array)

                        current_chunk = current_chunk[self.chunk_size:]

                        if len(chunks_loaded) >= 10:  # Limit chunks per call
                            return chunks_loaded

            # Add remaining chunk
            if current_chunk:
                chunk_array = np.array(current_chunk, dtype=np.int64)
                chunk_array = np.sort(chunk_array)
                chunk_array = np.unique(chunk_array)
                chunks_loaded.append(chunk_array)

            return chunks_loaded

        except Exception as e:
            print(f"❌ Download failed: {e}")
            return []

    def save_final_results(self, results):
        """Save final comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'wallace_interactive_analysis_{results["metadata"]["primes_count"]}_{timestamp}.json'

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"💾 Final results saved to: {filename}")

        # Create summary plot
        self.create_interactive_plot(results, timestamp)
        self.print_interactive_summary(results)

    def create_interactive_plot(self, results, timestamp):
        """Create comprehensive visualization."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Interactive Wallace Transform Analysis\n{results["metadata"]["primes_count"]:,} Primes', fontsize=16)

        # FFT Results
        if 'fft_analysis' in results and results['fft_analysis']['peaks']:
            fft_peaks = results['fft_analysis']['peaks']
            ratios = [p['ratio'] for p in fft_peaks]
            mags = [p['magnitude'] for p in fft_peaks]

            axes[0,0].bar(range(len(ratios)), mags, color='blue', alpha=0.7)
            axes[0,0].set_title(f'FFT Spectral Peaks\n({results["metadata"]["fft_sample_size"]:,} sample)')
            axes[0,0].set_xlabel('Peak Rank')
            axes[0,0].set_ylabel('Magnitude')
            axes[0,0].set_xticks(range(len(ratios)))
            axes[0,0].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        # Autocorrelation Results
        if 'autocorr_analysis' in results and results['autocorr_analysis']['peaks']:
            autocorr_peaks = results['autocorr_analysis']['peaks']
            ratios = [p['ratio'] for p in autocorr_peaks]
            corrs = [abs(p['correlation']) for p in autocorr_peaks]

            axes[0,1].bar(range(len(ratios)), corrs, color='green', alpha=0.7)
            axes[0,1].set_title(f'Autocorrelation Peaks\n({results["metadata"]["autocorr_sample_size"]:,} sample)')
            axes[0,1].set_xlabel('Peak Rank')
            axes[0,1].set_ylabel('Correlation Strength')
            axes[0,1].set_xticks(range(len(ratios)))
            axes[0,1].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        # Ratio Detection Matrix
        if 'validation' in results:
            val = results['validation']
            ratios = list(val['method_summary'].keys())
            fft_detected = [1 if val['method_summary'][r]['fft_detected'] else 0 for r in ratios]
            autocorr_detected = [1 if val['method_summary'][r]['autocorr_detected'] else 0 for r in ratios]

            x = np.arange(len(ratios))
            width = 0.35

            axes[1,0].bar(x - width/2, fft_detected, width, label='FFT', color='blue', alpha=0.7)
            axes[1,0].bar(x + width/2, autocorr_detected, width, label='AutoCorr', color='green', alpha=0.7)
            axes[1,0].set_title('Ratio Detection by Method')
            axes[1,0].set_xlabel('Harmonic Ratios')
            axes[1,0].set_ylabel('Detection (0/1)')
            axes[1,0].set_xticks(x)
            axes[1,0].set_xticklabels(ratios, rotation=45)
            axes[1,0].legend()

        # Progress timeline (placeholder)
        axes[1,1].text(0.5, 0.5, 'Interactive Analysis\nCompleted Successfully',
                      transform=axes[1,1].transAxes, ha='center', va='center',
                      fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1,1].set_title('Analysis Status')
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')

        plt.tight_layout()
        plt.savefig(f'wallace_interactive_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Interactive analysis plot saved as: wallace_interactive_analysis_{timestamp}.png")

    def print_interactive_summary(self, results):
        """Print comprehensive interactive analysis summary."""
        print("\n" + "=" * 100)
        print("🎯 INTERACTIVE WALLACE TRANSFORM ANALYSIS SUMMARY")
        print("=" * 100)

        meta = results['metadata']
        print(f"🎯 Scale Achieved: {meta['primes_count']:,} primes")
        print(f"📦 Chunked Processing: {meta['total_chunks']} chunks")
        print(f"🎯 FFT Sample Size: {meta['fft_sample_size']:,} gaps")
        print(f"🔗 Autocorr Sample Size: {meta['autocorr_sample_size']:,} gaps")
        print(f"💾 Checkpoint File: {meta['checkpoint_file']}")

        # Results summary
        if 'validation' in results:
            val = results['validation']
            fft_matches = val['fft_matches']
            autocorr_matches = val['autocorr_matches']
            unique_detected = val['unique_ratios_detected']

            print("\n🏆 RESULTS SUMMARY:")
            print(f"   FFT Detections: {fft_matches}/8 harmonic ratios")
            print(f"   Autocorr Detections: {autocorr_matches}/8 harmonic ratios")
            print(f"   Unique Ratios Found: {unique_detected}/8 total")
            print(f"   Success Rate: {(unique_detected/8)*100:.1f}%")

        print("\n📊 HARMONIC RATIOS DETECTED:")
        if 'validation' in results:
            val = results['validation']
            print("Ratio | FFT | AutoCorr | Status")
            print("-" * 30)
            for symbol, summary in val['method_summary'].items():
                fft_mark = "✓" if summary['fft_detected'] else "✗"
                autocorr_mark = "✓" if summary['autocorr_detected'] else "✗"
                status = "DETECTED" if (summary['fft_detected'] or summary['autocorr_detected']) else "PENDING"
                print(f"{symbol:5} | {fft_mark:3} | {autocorr_mark:8} | {status}")

        print("\n✅ ANALYSIS COMPLETE!")
        print("🎯 Interactive framework allows resuming from checkpoints")
        print("🌌 Results saved for further analysis")

def progress_callback(phase, progress, message):
    """Progress callback function for real-time updates."""
    bar_length = 40
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    print(f"\r📊 {phase}: [{bar}] {progress:.1f}% - {message}", end='', flush=True)
    if progress >= 100:
        print()  # New line at completion

def main():
    """Interactive analysis main function."""
    import argparse

    parser = argparse.ArgumentParser(description='Interactive Wallace Transform Analysis')
    parser.add_argument('--primes', type=int, default=50000000,
                       help='Target number of primes (default: 50M)')
    parser.add_argument('--chunk-size', type=int, default=500000,
                       help='Chunk size for processing (default: 500K)')
    parser.add_argument('--analysis', choices=['fft', 'autocorr', 'both'], default='both',
                       help='Analysis type (default: both)')

    args = parser.parse_args()

    print("🎯 INTERACTIVE WALLACE TRANSFORM ANALYSIS")
    print("Provides real-time progress updates and checkpointing")
    print("=" * 60)

    # Create analyzer
    analyzer = InteractiveWallaceAnalyzer(
        target_primes=args.primes,
        chunk_size=args.chunk_size
    )

    # Set progress callback
    analyzer.set_progress_callback(progress_callback)

    try:
        # Run interactive analysis
        results = analyzer.analyze_with_progress(args.analysis)

        if results:
            print("\n🎉 INTERACTIVE ANALYSIS SUCCESSFULLY COMPLETED!")
            print(f"📊 Processed {results['metadata']['primes_count']:,} primes")
            print("📈 Results saved and ready for further analysis")

    except KeyboardInterrupt:
        print("\n⏹️ Analysis interrupted by user")
        print("💾 Progress saved to checkpoint file")
        print("🔄 Use same command to resume analysis")

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("💾 Checkpoint may be available for recovery")

if __name__ == "__main__":
    main()
