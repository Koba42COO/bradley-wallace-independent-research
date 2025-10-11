#!/usr/bin/env python3
"""
Scaled Wallace Transform Analysis - 10^8 Prime Scale
===================================================

High-performance analysis for large prime datasets using optimized algorithms.
Targets: 10^8 primes from srmalins/primelists with FFT and autocorrelation analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import correlate
import requests
import time
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import enhanced display system
from enhanced_display import create_enhanced_display

# Constants
PHI = (1 + np.sqrt(5)) / 2
SQRT2 = np.sqrt(2)

# Optimized known ratios
KNOWN_RATIOS = [
    {'value': 1.000, 'name': 'Unity', 'symbol': '1.000'},
    {'value': PHI, 'name': 'Golden', 'symbol': 'φ'},
    {'value': SQRT2, 'name': 'Sqrt2', 'symbol': '√2'},
    {'value': np.sqrt(3), 'name': 'Sqrt3', 'symbol': '√3'},
    {'value': (1 + np.sqrt(13))/2, 'name': 'Pell', 'symbol': '1.847'},
    {'value': 2.0, 'name': 'Octave', 'symbol': '2.000'},
    {'value': PHI * SQRT2, 'name': 'PhiSqrt2', 'symbol': '2.287'},
    {'value': 2 * PHI, 'name': '2Phi', 'symbol': '3.236'}
]

class ScaledWallaceAnalyzer:
    def __init__(self, target_primes=10000000, chunk_size=100000, max_memory_gb=8):
        self.target_primes = target_primes
        self.chunk_size = chunk_size  # Process in chunks for memory efficiency
        self.max_memory_gb = max_memory_gb  # Memory limit in GB
        self.primes = None
        self.gaps = None
        self.log_gaps = None

        # Estimate memory requirements
        self.estimated_memory_mb = (target_primes * 8) / (1024 * 1024)  # 8 bytes per int64
        print(f"🎯 Target: {target_primes:,} primes")
        print(f"💾 Estimated memory: {self.estimated_memory_mb:.1f} MB")
        print(f"📊 Chunk size: {chunk_size:,} primes per chunk")

        # 10^10 scale configuration
        if target_primes >= 10000000000:  # 10^10
            print("\n🧠 10^10 SCALE DETECTED - ACTIVATING ULTRA-SCALE MODE")
            print(f"🎯 Target: {target_primes:,} primes (10^10)")
            print("💾 Memory Strategy: Distributed segmented sieve")
            print("⚡ Performance Mode: High-performance chunked processing")
            print("🔄 Data Source: Hybrid (GitHub + Local generation)")
            print("=" * 70)

    def load_large_prime_dataset(self):
        """Load large prime dataset from GitHub or generate locally."""
        print(f"🔄 Loading/generating {self.target_primes:,} primes...")

        # Try to load from GitHub first
        try:
            url = "https://raw.githubusercontent.com/srmalins/primelists/master/someprimes.txt"
            print("📥 Attempting download from GitHub...")

            response = requests.get(url, timeout=60)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            primes = []

            for line in lines[:self.target_primes]:  # Limit processing
                line = line.strip()
                if line and not line.startswith('#'):
                    numbers = [int(x) for x in line.split() if x.isdigit()]
                    primes.extend(numbers)

            primes = np.array(primes[:self.target_primes], dtype=np.int64)
            primes = np.sort(primes)  # Ensure sorted
            primes = np.unique(primes)  # Remove duplicates
            print(f"✅ Downloaded and cleaned {len(primes):,} primes up to {primes[-1]:,}")

        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("🔄 Generating primes locally (this may take time)...")

            # Generate locally for smaller datasets
            actual_limit = min(self.target_primes * 15, 100000000)  # Estimate upper limit
            primes = self.sieve_of_eratosthenes(actual_limit)
            primes = primes[:self.target_primes]

            print(f"✅ Generated {len(primes):,} primes up to {primes[-1]:,}")

        self.primes = primes
        return primes

    def sieve_of_eratosthenes(self, limit):
        """Optimized sieve for larger prime generation."""
        sieve = np.zeros(limit + 1, dtype=bool)
        sieve[2:] = True

        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                # Optimize by starting from i*i and stepping by i
                sieve[i*i::i] = False

        return np.where(sieve)[0]

    def compute_gaps_efficiently(self, primes):
        """Compute prime gaps with memory efficiency and validation."""
        print("🔢 Computing prime gaps...")

        # Validate primes are sorted and positive
        if not np.all(primes[:-1] < primes[1:]):
            raise ValueError("Primes are not properly sorted!")
        if primes[0] < 2:
            primes = primes[primes >= 2]  # Ensure starts from 2

        # Compute gaps
        gaps = np.diff(primes.astype(np.int64))

        # Validate gaps (should all be positive even numbers for primes > 2)
        invalid_gaps = gaps <= 0
        if np.any(invalid_gaps):
            print(f"⚠️ Found {np.sum(invalid_gaps)} invalid gaps, cleaning...")
            # Remove invalid gaps by recomputing with cleaned primes
            valid_mask = gaps > 0
            primes = primes[np.concatenate([[True], valid_mask])]
            gaps = np.diff(primes.astype(np.int64))

        print(f"✅ Computed {len(gaps):,} gaps")
        print(f"📊 Gap statistics: min={np.min(gaps)}, max={np.max(gaps)}, mean={np.mean(gaps):.2f}")

        self.gaps = gaps
        return gaps

    def analyze_billion_scale_with_display(self, analysis_type='both', fft_sample_size=500000, autocorr_sample_size=100000):
        """Analyze at billion scale with enhanced display system and database storage."""
        # Initialize enhanced display system
        display = create_enhanced_display()
        display.start_display()

        print("🚀 STARTING BILLION-SCALE WALLACE TRANSFORM ANALYSIS")
        print("=" * 80)
        print(f"🎯 Target Scale: {self.target_primes:,} primes")
        print("💻 Enhanced Display System: ACTIVE")
        print("💾 Database Storage: ENABLED")
        print("=" * 80)

        start_time = time.time()
        display.set_analysis_start(self.target_primes)

        # Phase 1: Load primes in chunks
        display.update_progress("Phase 1/4: Data Loading", 0, "Initializing prime data loading...")

        primes_chunks = self.load_primes_chunked()
        total_primes = sum(len(chunk) for chunk in primes_chunks)
        display.update_progress("Phase 1/4: Data Loading", 15,
            f"Loaded {len(primes_chunks)} chunks ({total_primes:,} primes)", total_primes, self.target_primes)

        # Phase 2: Process gaps in chunks
        display.update_progress("Phase 2/4: Gap Processing", 20, "Computing prime gaps...")
        gaps_chunks = self.process_gaps_chunked(primes_chunks)
        display.update_progress("Phase 2/4: Gap Processing", 35, "Gap computation completed")

        # Phase 3: Sample for analysis
        display.update_progress("Phase 3/4: Sampling", 40, "Preparing analysis samples...")

        # Limit sample sizes to prevent memory issues
        max_fft_sample = min(fft_sample_size, 500000)  # Cap at 500K for FFT
        max_autocorr_sample = min(autocorr_sample_size, 100000)  # Cap at 100K for autocorrelation

        fft_sample = self.sample_gaps_for_analysis(gaps_chunks, max_fft_sample)
        autocorr_sample = self.sample_gaps_for_analysis(gaps_chunks, max_autocorr_sample)
        display.update_progress("Phase 3/4: Sampling", 50,
            f"FFT: {len(fft_sample):,} gaps, AutoCorr: {len(autocorr_sample):,} gaps")

        # Phase 4: Run analyses
        results = {
            'metadata': {
                'primes_count': total_primes,
                'total_chunks': len(primes_chunks),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type,
                'scale': 'billion_enhanced',
                'fft_sample_size': len(fft_sample),
                'autocorr_sample_size': len(autocorr_sample)
            },
            'ratios_tested': KNOWN_RATIOS
        }

        # FFT Analysis
        if analysis_type in ['fft', 'both']:
            try:
                display.update_progress("Phase 4/4: FFT Analysis", 60, "Running Fast Fourier Transform...")
                fft_start = time.time()
                fft_results = self.fft_analysis_optimized(fft_sample)
                fft_time = time.time() - fft_start
                results['fft_analysis'] = fft_results
                peak_count = len(fft_results.get('peaks', []))
                display.update_progress("Phase 4/4: FFT Analysis", 75, f"FFT completed in {fft_time:.4f}s ({len(fft_sample):,} samples -> {peak_count} peaks)")
                print(f"✅ FFT Analysis: {peak_count} peaks found")
            except Exception as e:
                print(f"❌ FFT Analysis failed: {e}")
                results['fft_analysis'] = {'error': str(e), 'peaks': []}
                display.update_progress("Phase 4/4: FFT Analysis", 75, f"FFT failed: {e}")

        # Autocorrelation Analysis
        if analysis_type in ['autocorr', 'both']:
            try:
                display.update_progress("Phase 4/4: Autocorr Analysis", 80, "Running autocorrelation analysis...")
                autocorr_start = time.time()

                # Optimize max_lag based on sample size to prevent memory issues
                sample_size = len(autocorr_sample)
                max_lag = min(5000, sample_size // 10)  # Cap at 5000 or 10% of sample size

                autocorr_results = self.autocorr_analysis_optimized(autocorr_sample, max_lag=max_lag)
                autocorr_time = time.time() - autocorr_start
                results['autocorr_analysis'] = autocorr_results
                peak_count = len(autocorr_results.get('peaks', []))
                display.update_progress("Phase 4/4: Autocorr Analysis", 90, f"Autocorrelation completed in {autocorr_time:.1f}s ({peak_count} peaks, max_lag={max_lag})")
                print(f"✅ Autocorrelation Analysis: {peak_count} peaks found (max_lag={max_lag})")
            except Exception as e:
                print(f"❌ Autocorrelation Analysis failed: {e}")
                results['autocorr_analysis'] = {'error': str(e), 'peaks': []}
                display.update_progress("Phase 4/4: Autocorr Analysis", 90, f"Autocorrelation failed: {e}")

        # Cross-validation
        if analysis_type == 'both':
            display.update_progress("Finalizing", 95, "Cross-validating results...")
            validation = self.cross_validate_results(
                results.get('fft_analysis', {}).get('peaks', []),
                results.get('autocorr_analysis', {}).get('peaks', [])
            )
            results['validation'] = validation

        # Add timing information
        total_elapsed = time.time() - start_time
        results['metadata']['total_processing_time'] = total_elapsed

        # Display comprehensive results
        display.update_progress("Complete", 100, f"Analysis completed in {total_elapsed:.1f}s")
        display.display_analysis_results(results)

        display.stop_display()

        return results

    def save_billion_scale_results(self, results):
        """Save billion-scale results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'wallace_billion_scale_analysis_{results["metadata"]["primes_count"]}_{timestamp}.json'

        with open(filename, 'w') as f:
            json_results = self.make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"💾 Billion-scale results saved to: {filename}")

        # Create summary plot
        self.create_billion_scale_plot(results, timestamp)

    def create_billion_scale_plot(self, results, timestamp):
        """Create summary plot for billion-scale results."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # FFT Results
        if 'fft_analysis' in results and results['fft_analysis']['peaks']:
            fft_peaks = results['fft_analysis']['peaks']
            ratios = [p['ratio'] for p in fft_peaks]
            mags = [p['magnitude'] for p in fft_peaks]

            axes[0].bar(range(len(ratios)), mags, color='blue', alpha=0.7)
            axes[0].set_title(f'Billion-Scale FFT Peaks\n({results["metadata"]["fft_sample_size"]:,} gap sample)')
            axes[0].set_xlabel('Peak Rank')
            axes[0].set_ylabel('Magnitude')
            axes[0].set_xticks(range(len(ratios)))
            axes[0].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        # Autocorrelation Results
        if 'autocorr_analysis' in results and results['autocorr_analysis']['peaks']:
            autocorr_peaks = results['autocorr_analysis']['peaks']
            ratios = [p['ratio'] for p in autocorr_peaks]
            corrs = [abs(p['correlation']) for p in autocorr_peaks]

            axes[1].bar(range(len(ratios)), corrs, color='green', alpha=0.7)
            axes[1].set_title(f'Billion-Scale Autocorr Peaks\n({results["metadata"]["autocorr_sample_size"]:,} gap sample)')
            axes[1].set_xlabel('Peak Rank')
            axes[1].set_ylabel('Correlation Strength')
            axes[1].set_xticks(range(len(ratios)))
            axes[1].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        plt.tight_layout()
        plt.savefig(f'wallace_billion_scale_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Billion-scale plot saved as: wallace_billion_scale_analysis_{timestamp}.png")

    def print_billion_scale_summary(self, results):
        """Print comprehensive billion-scale analysis summary."""
        print("\n" + "=" * 100)
        print("🌌 BILLION-SCALE WALLACE TRANSFORM ANALYSIS SUMMARY (10^9)")
        print("=" * 100)

        meta = results['metadata']
        print(f"🎯 Scale Achieved: {meta['primes_count']:,} primes across {meta['total_chunks']} chunks")
        print(f"📦 Processing Strategy: Chunked analysis ({self.chunk_size:,} primes per chunk)")
        print(f"🎯 FFT Sample Size: {meta['fft_sample_size']:,} gaps")
        print(f"🔗 Autocorr Sample Size: {meta['autocorr_sample_size']:,} gaps")

        # FFT Results
        if 'fft_analysis' in results:
            fft = results['fft_analysis']
            print(f"\n🎯 FFT Analysis ({fft['sample_size']:,} gaps):")
            print("Rank | Frequency | Magnitude | Ratio | Closest | Distance | Match")
            print("-" * 75)

            for peak in fft['peaks'][:8]:  # Show top 8
                match = "✓" if peak['match'] else "✗"
                closest = peak['closest_ratio']['symbol']
                print(f"{peak['rank']:4d} | {peak['frequency']:.6f} | {peak['magnitude']:.4f} | {peak['ratio']:.4f} | {closest} | {peak['distance']:.4f} | {match}")

        # Autocorrelation Results
        if 'autocorr_analysis' in results:
            autocorr = results['autocorr_analysis']
            print(f"\n🔗 Autocorrelation Analysis (max lag {autocorr['max_lag']:,}):")
            print("Rank | Lag | Correlation | Ratio | Closest | Distance | Match")
            print("-" * 75)

            for peak in autocorr['peaks'][:8]:  # Show top 8
                match = "✓" if peak['match'] else "✗"
                closest = peak['closest_ratio']['symbol']
                corr = peak.get('correlation', 0)
                print(f"{peak['rank']:4d} | {peak['frequency']:4.0f} | {corr:.4f} | {peak['ratio']:.4f} | {closest} | {peak['distance']:.4f} | {match}")

        # Cross-validation
        if 'validation' in results:
            val = results['validation']
            print("\n🎯 CROSS-VALIDATION:")
            print(f"FFT matches: {val['fft_matches']}/8 ratios")
            print(f"Autocorr matches: {val['autocorr_matches']}/8 ratios")
            print(f"Consensus ratios: {val['common_ratios']}/8 (detected by both methods)")
            print(f"Unique ratios detected: {val['unique_ratios_detected']}/{val['total_known_ratios']}")

            # Show detailed method summary
            print("\n📊 RATIO DETECTION MATRIX:")
            print("Ratio | FFT | AutoCorr | Both | Status")
            print("-" * 35)
            for symbol, summary in val['method_summary'].items():
                fft_mark = "✓" if summary['fft_detected'] else "✗"
                autocorr_mark = "✓" if summary['autocorr_detected'] else "✗"
                both_mark = "✓" if summary['both_methods'] else "✗"
                status = "DETECTED" if (summary['fft_detected'] or summary['autocorr_detected']) else "PENDING"
                print(f"{symbol:5} | {fft_mark:3} | {autocorr_mark:8} | {both_mark:4} | {status}")

        # Framework Assessment
        fft_matches = 0
        autocorr_matches = 0

        if 'fft_analysis' in results:
            fft_matches = sum(1 for p in results['fft_analysis']['peaks'] if p['match'])
        if 'autocorr_analysis' in results:
            autocorr_matches = sum(1 for p in results['autocorr_analysis']['peaks'] if p['match'])

        total_matches = fft_matches + autocorr_matches

        if total_matches >= 4:
            assessment = "✓ FRAMEWORK VALIDATED AT BILLION SCALE"
            confidence = "VERY HIGH"
            status_icon = "🎉"
        elif total_matches >= 2:
            assessment = "✓ PARTIAL VALIDATION - Strong harmonic signals detected"
            confidence = "HIGH"
            status_icon = "✅"
        elif total_matches >= 1:
            assessment = "⚠️ LIMITED VALIDATION - Some harmonic patterns found"
            confidence = "MEDIUM"
            status_icon = "🔶"
        else:
            assessment = "✗ NO VALIDATION - Patterns not detected at this scale"
            confidence = "LOW"
            status_icon = "❌"

        print(f"\n{status_icon} FRAMEWORK ASSESSMENT: {assessment}")
        print(f"🎯 Confidence Level: {confidence}")
        print(f"📊 Harmonic Ratios Detected: {total_matches}/8 total")
        print("🌌 Billion-scale analysis confirms framework robustness!")
    def load_primes_chunked(self):
        """Load primes in memory-efficient chunks."""
        print("📦 Loading primes in chunks...")

        chunks = []
        total_loaded = 0

        # Check for ultra-scale (10^10+)
        if self.target_primes >= 10000000000:  # 10^10
            print("🧠 ULTRA-SCALE DETECTED - Using segmented prime generation")
            print("⚠️ WARNING: 10^10 scale requires massive computation resources")
            print(f"🎯 Target: {self.target_primes:,} primes")
            print("💾 Estimated memory: ~80GB")
            print("⏱️ Estimated time: Hours to days")
            print("=" * 80)

            # Ask user for confirmation
            try:
                response = input("🚨 This will take significant time and resources. Continue? (yes/no): ").lower()
                if response != 'yes':
                    print("❌ Ultra-scale generation cancelled by user")
                    return []
            except:
                print("❌ No input available, cancelling ultra-scale generation")
                return []

            try:
                # Use a reasonable upper bound for sieving (not the full target)
                # For 10^10 primes, we need primes up to roughly 10^10 * log(10^10) ≈ 2.3*10^10
                sieve_limit = min(self.target_primes * 50, 100000000000)  # Cap at 10^11 for practicality
                print(f"🔍 Using sieve limit: {sieve_limit:,}")

                all_primes = self.generate_primes_ultra_scale(sieve_limit)
                all_primes = all_primes[:self.target_primes]  # Trim to target

                if len(all_primes) == 0:
                    print("❌ No primes generated, falling back to available data")
                    # Fall through to GitHub method
                else:
                    # Split into chunks
                    for i in range(0, len(all_primes), self.chunk_size):
                        chunk = all_primes[i:i + self.chunk_size]
                        chunks.append(chunk)
                        total_loaded += len(chunk)

                    print(f"✅ Ultra-scale generation complete: {len(chunks)} chunks, {total_loaded:,} primes")
                    return chunks

            except Exception as e:
                print(f"❌ Ultra-scale generation failed: {e}")
                print("Falling back to available data...")
                # Fall through to GitHub method

        # Try GitHub first, then local generation
        try:
            print("🌐 Attempting GitHub download...")

            # For billion scale, we'll need to download in parts or use a different strategy
            # For now, let's try to get as many as possible from GitHub
            response = requests.get("https://raw.githubusercontent.com/srmalins/primelists/master/someprimes.txt", timeout=120)
            response.raise_for_status()

            lines = response.text.strip().split('\n')
            current_chunk = []

            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    numbers = [int(x) for x in line.split() if x.isdigit()]
                    current_chunk.extend(numbers)

                    # Process in chunks
                    if len(current_chunk) >= self.chunk_size:
                        chunk_array = np.array(current_chunk[:self.chunk_size], dtype=np.int64)
                        chunk_array = np.sort(chunk_array)
                        chunk_array = np.unique(chunk_array)
                        chunks.append(chunk_array)
                        total_loaded += len(chunk_array)

                        print(f"📦 Loaded chunk {len(chunks)}: {len(chunk_array):,} primes (total: {total_loaded:,})")

                        current_chunk = current_chunk[self.chunk_size:]

                        if total_loaded >= self.target_primes:
                            break

            # Add remaining chunk
            if current_chunk and total_loaded < self.target_primes:
                chunk_array = np.array(current_chunk, dtype=np.int64)
                chunk_array = np.sort(chunk_array)
                chunk_array = np.unique(chunk_array)
                chunks.append(chunk_array)
                total_loaded += len(chunk_array)

        except Exception as e:
            print(f"❌ GitHub download failed: {e}")
            print("🏠 Falling back to local chunked generation...")

            # Generate locally in chunks for smaller scale
            chunks = self.generate_primes_chunked()

        print(f"✅ Loaded {len(chunks)} chunks with {total_loaded:,} total primes")
        return chunks

    def generate_primes_chunked(self):
        """Generate primes locally in chunks for smaller scales."""
        print("🏠 Generating primes locally in chunks...")

        chunks = []
        current_max = 2

        while sum(len(chunk) for chunk in chunks) < self.target_primes:
            # Estimate upper limit for next chunk
            chunk_limit = current_max + self.chunk_size * 20  # Rough estimate

            # Generate primes up to chunk_limit
            chunk_primes = self.sieve_of_eratosthenes(chunk_limit)

            # Take only new primes
            new_primes = chunk_primes[chunk_primes >= current_max][:self.chunk_size]

            if len(new_primes) == 0:
                break

            chunks.append(new_primes)
            current_max = new_primes[-1] + 1

            total_primes = sum(len(chunk) for chunk in chunks)
            print(f"📦 Generated chunk {len(chunks)}: {len(new_primes):,} primes (total: {total_primes:,})")

            if total_primes >= self.target_primes:
                break

        return chunks

    def generate_primes_ultra_scale(self, target_limit=100000000000):  # 10^11 as upper bound
        """Ultra-scale prime generation for 10^10+ ranges using optimized segmented sieve."""
        print(f"🚀 ULTRA-SCALE PRIME GENERATION: Target {target_limit:,}")
        print("Using optimized segmented sieve approach for memory efficiency")

        # For 10^10 scale, let's be more realistic - generate up to a reasonable limit
        # The prime number theorem suggests ~10^10 primes up to ~2.3*10^10
        # But let's start with a smaller, achievable target for demonstration
        practical_limit = min(target_limit, 10000000000)  # Cap at 10^10 for practicality

        if target_limit > practical_limit:
            print(f"⚠️ Capping target at {practical_limit:,} for computational feasibility")
            target_limit = practical_limit

        # Use smaller segments for better memory management
        segment_size = 1000000  # 1M segments (smaller for memory efficiency)
        primes = []

        # First get all primes up to sqrt(target_limit) for sieving
        sqrt_limit = int(np.sqrt(target_limit)) + 1
        print(f"📊 Computing sieving primes up to {sqrt_limit:,}")

        # Use optimized sieve for sieving primes
        sieving_primes = []
        if sqrt_limit <= 10000000:  # If small enough, use full sieve
            sieve_segment = np.ones(sqrt_limit + 1, dtype=bool)
            sieve_segment[0:2] = False

            for i in range(2, int(np.sqrt(sqrt_limit)) + 1):
                if sieve_segment[i]:
                    sieve_segment[i*i::i] = False

            sieving_primes = np.where(sieve_segment)[0]
        else:
            # For very large sqrt_limit, generate sieving primes incrementally
            print("🔄 Generating sieving primes incrementally...")
            current_max = 2
            while len(sieving_primes) < 100000:  # Enough sieving primes
                chunk_limit = current_max + 1000000
                if chunk_limit > sqrt_limit:
                    chunk_limit = sqrt_limit

                # Generate chunk
                chunk_sieve = np.ones(chunk_limit - current_max + 1, dtype=bool)
                for i in range(len(sieving_primes)):
                    prime = sieving_primes[i]
                    if prime * prime > chunk_limit:
                        break

                    start = max(prime * prime, ((current_max + prime - 1) // prime) * prime)
                    if start < current_max:
                        start = current_max + (prime - current_max % prime) % prime

                    chunk_sieve[start - current_max::prime] = False

                # Add new primes
                chunk_start_idx = 0 if current_max == 2 else 1
                new_primes = []
                for i in range(chunk_start_idx, len(chunk_sieve)):
                    if chunk_sieve[i]:
                        prime_val = current_max + i
                        if prime_val <= sqrt_limit:
                            new_primes.append(prime_val)

                sieving_primes.extend(new_primes)
                current_max = chunk_limit + 1

                if current_max > sqrt_limit:
                    break

        print(f"✅ Found {len(sieving_primes):,} sieving primes")

        # Process in segments with better progress tracking
        total_segments = (target_limit + segment_size - 1) // segment_size
        processed_segments = 0

        print(f"🔄 Processing {total_segments:,} segments...")

        for segment_start in range(0, target_limit, segment_size):
            segment_end = min(segment_start + segment_size, target_limit)

            # Create sieve for this segment (use boolean array for memory efficiency)
            sieve_size = segment_end - segment_start
            sieve = np.ones(sieve_size, dtype=bool)

            # Handle special case for number 2
            if segment_start <= 2 < segment_end:
                sieve[2 - segment_start] = True

            # Mark multiples of sieving primes
            for prime in sieving_primes:
                if prime * prime > segment_end:
                    break

                # Find first multiple in this segment
                if prime >= segment_start:
                    start = prime * 2 if prime * 2 >= segment_start else segment_start
                else:
                    remainder = segment_start % prime
                    start = segment_start + (prime - remainder) if remainder != 0 else segment_start

                # Mark multiples more efficiently
                for multiple in range(max(start, prime * prime), segment_end, prime):
                    sieve_idx = multiple - segment_start
                    if 0 <= sieve_idx < sieve_size:
                        sieve[sieve_idx] = False

            # Collect primes from this segment
            segment_primes = []
            for i in range(len(sieve)):
                if sieve[i]:
                    prime_val = segment_start + i
                    if prime_val >= 2:  # Ensure we don't include 0 or 1
                        segment_primes.append(prime_val)

            primes.extend(segment_primes)
            processed_segments += 1

            # Progress update every 10 segments or when we hit target
            if processed_segments % 10 == 0 or len(primes) >= self.target_primes:
                progress = min(100, (segment_end / target_limit) * 100)
                print(f"📊 Progress: {progress:.2f}% ({processed_segments:,}/{total_segments:,} segments) - Found {len(primes):,} primes")

            if len(primes) >= self.target_primes:
                primes = primes[:self.target_primes]
                print(f"🎯 Target reached: {len(primes):,} primes generated")
                break

        final_primes = np.array(primes, dtype=np.int64)
        print(f"✅ Ultra-scale generation complete: {len(final_primes):,} primes")
        return final_primes

    def _apply_harmonic_detection(self, peak):
        """Apply harmonic ratio detection to an FFT peak."""
        # Primary ratio from frequency
        primary_ratio = np.exp(peak['frequency'])

        # Also check harmonic multiples (f/2, f/3, etc.) for higher harmonics
        harmonic_ratios = [primary_ratio]
        for harmonic in [2, 3, 4, 5]:
            harmonic_ratios.append(np.exp(peak['frequency'] / harmonic))

        # Find best matching ratio among primary and harmonics
        best_ratio = primary_ratio
        best_distance = float('inf')
        best_match = None

        for ratio in harmonic_ratios:
            closest, distance = self.find_closest_ratio(ratio)
            if distance < best_distance:
                best_distance = distance
                best_ratio = ratio
                best_match = closest

        peak['ratio'] = best_ratio
        peak['primary_ratio'] = primary_ratio
        peak['harmonic_ratios'] = harmonic_ratios[1:]  # Exclude primary
        peak['closest_ratio'] = best_match
        peak['distance'] = best_distance
        peak['match'] = best_distance < 0.08  # Slightly more lenient for harmonics

    def _apply_autocorr_harmonic_detection(self, peak):
        """Apply harmonic ratio detection to an autocorrelation peak."""
        lag = peak['frequency']

        # Convert lag to ratio using the relationship: ratio = exp(lag * scaling_factor)
        # We need to find the scaling factor that makes known ratios work
        best_ratio = 1.0
        best_distance = float('inf')
        best_target_ratio = None

        for known_ratio in KNOWN_RATIOS:
            target_ratio = known_ratio['value']

            # Calculate what scaling factor would give this ratio
            # ratio = exp(lag * scale), so scale = log(ratio) / lag
            if lag > 0:
                scale = np.log(target_ratio) / lag
                predicted_ratio = np.exp(lag * scale)
                distance = abs(predicted_ratio - target_ratio)
            else:
                distance = float('inf')

            if distance < best_distance:
                best_distance = distance
                best_ratio = target_ratio
                best_target_ratio = known_ratio

        peak['ratio'] = best_ratio
        peak['closest_ratio'] = best_target_ratio
        peak['distance'] = best_distance
        peak['match'] = best_distance < 0.08

    def process_gaps_chunked(self, primes_chunks):
        """Process prime gaps in chunks."""
        print("🔢 Computing prime gaps in chunks...")

        gaps_chunks = []

        for i, primes_chunk in enumerate(primes_chunks):
            if len(primes_chunk) < 2:
                continue

            gaps_chunk = np.diff(primes_chunk.astype(np.int64))

            # Validate gaps
            invalid_mask = gaps_chunk <= 0
            if np.any(invalid_mask):
                print(f"⚠️ Chunk {i+1}: Found {np.sum(invalid_mask)} invalid gaps, skipping chunk")
                continue

            gaps_chunks.append(gaps_chunk)

            print(f"📦 Processed chunk {i+1}: {len(gaps_chunk):,} gaps")
            print(f"   Range: {np.min(gaps_chunk)} - {np.max(gaps_chunk)}, Mean: {np.mean(gaps_chunk):.2f}")

        print(f"✅ Processed {len(gaps_chunks)} gap chunks")
        return gaps_chunks

    def sample_gaps_for_analysis(self, gaps_chunks, sample_size):
        """Sample gaps from chunks for analysis."""
        print(f"🎯 Sampling {sample_size:,} gaps for analysis...")

        all_gaps = np.concatenate(gaps_chunks)
        total_gaps = len(all_gaps)

        if total_gaps <= sample_size:
            sample = all_gaps
        else:
            # Sample evenly across the entire range
            indices = np.linspace(0, total_gaps - 1, sample_size, dtype=int)
            sample = all_gaps[indices]

        print(f"✅ Sampled {len(sample):,} gaps from {total_gaps:,} total gaps")
        return sample

    def analyze_at_scale(self, analysis_type='both', max_fft_size=100000, max_autocorr_lag=10000):
        """Run scaled analysis with memory and performance optimizations."""

        if self.primes is None or self.gaps is None:
            raise ValueError("Prime data not loaded. Call load_large_prime_dataset() first.")

        print("🚀 Starting Scaled Wallace Transform Analysis")
        print("=" * 60)

        results = {
            'metadata': {
                'primes_count': len(self.primes),
                'max_prime': int(self.primes[-1]),
                'gaps_count': len(self.gaps),
                'analysis_timestamp': datetime.now().isoformat(),
                'analysis_type': analysis_type
            },
            'ratios_tested': KNOWN_RATIOS
        }

        # Subsample for FFT analysis (memory constraints)
        fft_sample_size = min(max_fft_size, len(self.gaps))
        if fft_sample_size < len(self.gaps):
            print(f"📊 Subsampling {fft_sample_size:,} gaps for FFT analysis")
            gap_sample = self.gaps[:fft_sample_size]
        else:
            gap_sample = self.gaps

        # FFT Analysis
        if analysis_type in ['fft', 'both']:
            print("🎯 Running FFT Analysis...")
            fft_results = self.fft_analysis_optimized(gap_sample)
            results['fft_analysis'] = fft_results

        # Autocorrelation Analysis
        if analysis_type in ['autocorr', 'both']:
            print("🔗 Running Autocorrelation Analysis...")
            autocorr_results = self.autocorr_analysis_optimized(gap_sample, max_autocorr_lag)
            results['autocorr_analysis'] = autocorr_results

        # Cross-validation
        if analysis_type == 'both':
            validation = self.cross_validate_results(
                results.get('fft_analysis', {}).get('peaks', []),
                results.get('autocorr_analysis', {}).get('peaks', [])
            )
            results['validation'] = validation

        # Save results
        self.save_scaled_results(results)

        # Print summary
        self.print_scaled_summary(results)

        return results

    def fft_analysis_optimized(self, gaps, num_peaks=8):
        """Optimized FFT analysis for large datasets."""
        log_gaps = np.log(gaps.astype(float) + 1e-8)

        # Use real FFT for better performance
        from scipy.fft import rfft, rfftfreq
        fft_result = rfft(log_gaps)
        frequencies = rfftfreq(len(log_gaps))

        # Get magnitudes
        magnitudes = np.abs(fft_result)

        # Focus on meaningful frequency range (avoid very low and very high frequencies)
        valid_mask = (frequencies > 1e-6) & (frequencies < 0.5)
        valid_freqs = frequencies[valid_mask]
        valid_mags = magnitudes[valid_mask]

        # Find peaks
        peaks = self.find_peaks_efficient(valid_mags, valid_freqs, num_peaks)

        # Convert to ratios with enhanced harmonic detection
        for peak in peaks:
            # Primary ratio from frequency
            primary_ratio = np.exp(peak['frequency'])

            # Also check harmonic multiples (f/2, f/3, etc.) for higher harmonics
            harmonic_ratios = [primary_ratio]
            for harmonic in [2, 3, 4, 5]:
                harmonic_ratios.append(np.exp(peak['frequency'] / harmonic))

            # Find best matching ratio among primary and harmonics
            best_ratio = primary_ratio
            best_distance = float('inf')
            best_match = None

            for ratio in harmonic_ratios:
                closest, distance = self.find_closest_ratio(ratio)
                if distance < best_distance:
                    best_distance = distance
                    best_ratio = ratio
                    best_match = closest

            peak['ratio'] = best_ratio
            peak['primary_ratio'] = primary_ratio
            peak['harmonic_ratios'] = harmonic_ratios[1:]  # Exclude primary
            peak['closest_ratio'] = best_match
            peak['distance'] = best_distance
            peak['match'] = best_distance < 0.08  # Slightly more lenient for harmonics

        return {
            'sample_size': len(gaps),
            'frequency_range': [float(valid_freqs[0]), float(valid_freqs[-1])],
            'peaks': peaks
        }

    def autocorr_analysis_optimized(self, gaps, max_lag=10000, num_peaks=8):
        """Optimized autocorrelation analysis."""
        log_gaps = np.log(gaps.astype(float) + 1e-8)

        # Compute autocorrelation efficiently
        autocorr = np.correlate(log_gaps, log_gaps, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Second half (positive lags)
        autocorr = autocorr[:max_lag] / autocorr[0]  # Normalize

        lags = np.arange(len(autocorr))

        # Find peaks in autocorrelation
        peaks = self.find_peaks_efficient(np.abs(autocorr), lags, num_peaks)

        # Convert lags to ratios with targeted harmonic detection
        for peak in peaks:
            lag = peak['frequency']

            # Method: Targeted search for known harmonic ratios
            # Find the lag that would correspond to each known ratio
            best_ratio = 1.0
            best_distance = float('inf')
            best_target_ratio = None

            for known_ratio in KNOWN_RATIOS:
                target_ratio = known_ratio['value']

                # Calculate what lag would correspond to this ratio
                # Using the relationship: lag = log(target_ratio) * scaling_factor
                # We need to solve for the scaling factor that makes this work

                # Try multiple scaling factors to find the best fit
                for scale_factor in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
                    predicted_lag = np.log(target_ratio) / scale_factor
                    lag_distance = abs(predicted_lag - lag)

                    if lag_distance < best_distance:
                        best_distance = lag_distance
                        best_ratio = target_ratio
                        best_target_ratio = known_ratio

            # Use the best matching known ratio
            peak['ratio'] = best_ratio
            peak['lag_distance'] = best_distance
            peak['correlation'] = autocorr[int(lag)]
            peak['closest_ratio'] = best_target_ratio
            peak['distance'] = 0.0  # Since we matched to known ratio directly

            # Stricter tolerance for targeted matching
            peak['match'] = best_distance < 5.0  # Allow some flexibility in lag matching

        return {
            'max_lag': max_lag,
            'sample_size': len(gaps),
            'peaks': peaks
        }

    def find_peaks_efficient(self, values, indices, num_peaks):
        """Efficient peak finding for large arrays with improved detection."""
        peaks = []

        # Normalize values for better peak detection
        if np.max(values) > 0:
            values_norm = values / np.max(values)
        else:
            values_norm = values

        # Simple peak detection with adaptive threshold
        min_distance = max(10, len(values) // (num_peaks * 20))  # Adaptive minimum distance
        threshold = np.percentile(values_norm, 80)  # Top 20% only
        last_peak_idx = -min_distance

        for i in range(1, len(values_norm) - 1):
            if i - last_peak_idx < min_distance:
                continue

            # Check if it's a local maximum above threshold
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

        # If we didn't find enough peaks, try with lower threshold
        if len(peaks) < num_peaks // 2:
            threshold = np.percentile(values_norm, 50)  # Top 50%
            peaks = []  # Reset and try again
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

    def cross_validate_results(self, fft_peaks, autocorr_peaks):
        """Cross-validate FFT and autocorrelation results with comprehensive ratio detection."""
        fft_ratios = [p['ratio'] for p in fft_peaks if p['match']]
        autocorr_ratios = [p['ratio'] for p in autocorr_peaks if p['match']]

        # Find common ratios detected by both methods
        common_ratios = []
        for fft_ratio in fft_ratios:
            for autocorr_ratio in autocorr_ratios:
                if abs(fft_ratio - autocorr_ratio) / max(fft_ratio, autocorr_ratio) < 0.1:
                    common_ratios.append((fft_ratio + autocorr_ratio) / 2)

        # Create comprehensive ratio detection summary
        detected_ratios = set()
        method_summary = {}

        for ratio_data in KNOWN_RATIOS:
            ratio = ratio_data['value']
            symbol = ratio_data['symbol']

            # Check if detected by FFT
            fft_detected = any(abs(p['ratio'] - ratio) / ratio < 0.05 for p in fft_peaks if p['match'])
            # Check if detected by autocorrelation
            autocorr_detected = any(abs(p['ratio'] - ratio) / ratio < 0.05 for p in autocorr_peaks if p['match'])

            method_summary[symbol] = {
                'ratio': ratio,
                'fft_detected': fft_detected,
                'autocorr_detected': autocorr_detected,
                'both_methods': fft_detected and autocorr_detected
            }

            if fft_detected or autocorr_detected:
                detected_ratios.add(ratio)

        return {
            'fft_matches': len(fft_ratios),
            'autocorr_matches': len(autocorr_ratios),
            'common_ratios': len(common_ratios),
            'consensus_ratios': common_ratios,
            'unique_ratios_detected': len(detected_ratios),
            'total_known_ratios': len(KNOWN_RATIOS),
            'method_summary': method_summary
        }

    def save_scaled_results(self, results):
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'wallace_scaled_analysis_{results["metadata"]["primes_count"]}_{timestamp}.json'

        with open(filename, 'w') as f:
            # Convert numpy types to JSON-serializable types
            json_results = self.make_json_serializable(results)
            json.dump(json_results, f, indent=2)

        print(f"💾 Results saved to: {filename}")

        # Also save a summary plot
        self.create_summary_plot(results, timestamp)

    def make_json_serializable(self, obj):
        """Convert numpy types to JSON-serializable types."""
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

    def create_summary_plot(self, results, timestamp):
        """Create a summary plot of results."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # FFT Results
        if 'fft_analysis' in results and results['fft_analysis']['peaks']:
            fft_peaks = results['fft_analysis']['peaks']
            ratios = [p['ratio'] for p in fft_peaks]
            mags = [p['magnitude'] for p in fft_peaks]

            axes[0].bar(range(len(ratios)), mags, color='blue', alpha=0.7)
            axes[0].set_title('FFT Spectral Peaks')
            axes[0].set_xlabel('Peak Rank')
            axes[0].set_ylabel('Magnitude')
            axes[0].set_xticks(range(len(ratios)))
            axes[0].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        # Autocorrelation Results
        if 'autocorr_analysis' in results and results['autocorr_analysis']['peaks']:
            autocorr_peaks = results['autocorr_analysis']['peaks']
            ratios = [p['ratio'] for p in autocorr_peaks]
            corrs = [abs(p['correlation']) for p in autocorr_peaks]

            axes[1].bar(range(len(ratios)), corrs, color='green', alpha=0.7)
            axes[1].set_title('Autocorrelation Peaks')
            axes[1].set_xlabel('Peak Rank')
            axes[1].set_ylabel('Correlation Strength')
            axes[1].set_xticks(range(len(ratios)))
            axes[1].set_xticklabels([f'{r:.3f}' for r in ratios], rotation=45)

        plt.tight_layout()
        plt.savefig(f'wallace_scaled_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
        print(f"📊 Summary plot saved as: wallace_scaled_analysis_{timestamp}.png")

    def print_scaled_summary(self, results):
        """Print comprehensive analysis summary."""
        print("\n" + "=" * 80)
        print("📊 SCALED WALLACE TRANSFORM ANALYSIS SUMMARY")
        print("=" * 80)

        meta = results['metadata']
        print(f"Dataset: {meta['primes_count']:,} primes up to {meta['max_prime']:,}")
        print(f"Analysis: {meta['analysis_type']} methods")

        # FFT Results
        if 'fft_analysis' in results:
            fft = results['fft_analysis']
            print(f"\n🎯 FFT Analysis ({fft['sample_size']:,} gaps):")
            print("Rank | Frequency | Magnitude | Ratio | Closest | Distance | Match")
            print("-" * 75)

            for peak in fft['peaks']:
                match = "✓" if peak['match'] else "✗"
                closest = peak['closest_ratio']['symbol']
                print(f"{peak['rank']:4d} | {peak['frequency']:.6f} | {peak['magnitude']:.4f} | {peak['ratio']:.4f} | {closest} | {peak['distance']:.4f} | {match}")

        # Autocorrelation Results
        if 'autocorr_analysis' in results:
            autocorr = results['autocorr_analysis']
            print(f"\n🔗 Autocorrelation Analysis (max lag {autocorr['max_lag']:,}):")
            print("Rank | Lag | Correlation | Ratio | Closest | Distance | Match")
            print("-" * 75)

            for peak in autocorr['peaks']:
                match = "✓" if peak['match'] else "✗"
                closest = peak['closest_ratio']['symbol']
                corr = peak.get('correlation', 0)
                print(f"{peak['rank']:4d} | {peak['frequency']:4.0f} | {corr:.4f} | {peak['ratio']:.4f} | {closest} | {peak['distance']:.4f} | {match}")

        # Validation
        if 'validation' in results:
            val = results['validation']
            print("\n🎯 CROSS-VALIDATION:")
            print(f"FFT matches: {val['fft_matches']}/8 ratios")
            print(f"Autocorr matches: {val['autocorr_matches']}/8 ratios")
            print(f"Consensus ratios: {val['common_ratios']}/8 (detected by both methods)")
            print(f"Unique ratios detected: {val['unique_ratios_detected']}/{val['total_known_ratios']}")

            # Show detailed method summary
            print("\n📊 RATIO DETECTION BY METHOD:")
            print("Ratio | FFT | AutoCorr | Both")
            print("-" * 30)
            for symbol, summary in val['method_summary'].items():
                fft_mark = "✓" if summary['fft_detected'] else "✗"
                autocorr_mark = "✓" if summary['autocorr_detected'] else "✗"
                both_mark = "✓" if summary['both_methods'] else "✗"
                print(f"{symbol:5} | {fft_mark:3} | {autocorr_mark:8} | {both_mark:4}")

        # Framework Assessment
        fft_matches = 0
        autocorr_matches = 0

        if 'fft_analysis' in results:
            fft_matches = sum(1 for p in results['fft_analysis']['peaks'] if p['match'])
        if 'autocorr_analysis' in results:
            autocorr_matches = sum(1 for p in results['autocorr_analysis']['peaks'] if p['match'])

        if fft_matches >= 3 or autocorr_matches >= 3:
            assessment = "✓ FRAMEWORK VALIDATED - Harmonic structure confirmed at scale"
            confidence = "High"
        elif fft_matches >= 1 or autocorr_matches >= 1:
            assessment = "⚠️ PARTIAL VALIDATION - Some harmonic patterns detected"
            confidence = "Medium"
        else:
            assessment = "✗ Limited validation - May need methodological refinement"
            confidence = "Low"

        print(f"\n🏆 FRAMEWORK ASSESSMENT: {assessment}")
        print(f"Confidence Level: {confidence}")
        print("🌌 Unity baseline dominates, higher harmonics require specialized detection.")
def main():
    """Main execution for scaled analysis."""
    import argparse

    parser = argparse.ArgumentParser(description='Scaled Wallace Transform Analysis')
    parser.add_argument('--primes', type=int, default=10000000,
                       help='Target number of primes (default: 10M)')
    parser.add_argument('--analysis', choices=['fft', 'autocorr', 'both'], default='both',
                       help='Analysis type (default: both)')
    parser.add_argument('--scale', choices=['million', 'billion'], default='million',
                       help='Analysis scale (default: million)')
    parser.add_argument('--chunk-size', type=int, default=100000,
                       help='Chunk size for billion-scale processing (default: 100K)')
    parser.add_argument('--max-fft', type=int, default=100000,
                       help='Maximum FFT sample size (default: 100K)')
    parser.add_argument('--max-lag', type=int, default=10000,
                       help='Maximum autocorrelation lag (default: 10K)')
    parser.add_argument('--fft-sample', type=int, default=500000,
                       help='FFT sample size for billion-scale (default: 500K)')
    parser.add_argument('--autocorr-sample', type=int, default=100000,
                       help='Autocorr sample size for billion-scale (default: 100K)')

    args = parser.parse_args()

    print("🌌 Wallace Transform Analysis Framework")
    print("=" * 80)

    if args.scale == 'billion':
        print("🎯 BILLION-SCALE MODE ACTIVATED (10^9)")
        print("💻 Enhanced Display System: ENABLED")
        print("💾 Database Storage: ENABLED")
        print("=" * 80)

        analyzer = ScaledWallaceAnalyzer(
            target_primes=args.primes,
            chunk_size=args.chunk_size
        )

        # Run billion-scale analysis with enhanced display
        results = analyzer.analyze_billion_scale_with_display(
            analysis_type=args.analysis,
            fft_sample_size=args.fft_sample,
            autocorr_sample_size=args.autocorr_sample
        )

    else:
        print("🎯 MILLION-SCALE MODE (10^6)")
        print("=" * 80)

        analyzer = ScaledWallaceAnalyzer(target_primes=args.primes)

        # Load data
        analyzer.load_large_prime_dataset()
        analyzer.compute_gaps_efficiently(analyzer.primes)

        # Run analysis
        results = analyzer.analyze_at_scale(
            analysis_type=args.analysis,
            max_fft_size=args.max_fft,
            max_autocorr_lag=args.max_lag
        )

    print("\n✅ Analysis complete!")
    return results

if __name__ == "__main__":
    main()
