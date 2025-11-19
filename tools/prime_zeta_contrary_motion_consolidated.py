# CONSOLIDATED FROM: prime_zeta_contrary_motion.py, prime_zeta_contrary_motion.py, prime_zeta_contrary_motion.py, prime_zeta_contrary_motion.py
#!/usr/bin/env python3
"""
Prime Distribution and Zeta Distribution in Contrary Motion

This script generates audio where prime number distribution and Riemann zeta function
values are played in contrary motion using Bram Cohen's synthesizer.
"""

import math
import wave
import struct
import numpy as np
from typing import List, Tuple, Dict, Any
import json
import sys
import os


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math
import cmath
from typing import Dict, List, Tuple, Optional, Any

# Set high precision for consciousness mathematics
getcontext().prec = 50

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    QUANTUM_BRIDGE = Decimal('137') / Decimal('0.79')  # 173.41772151898732
    GREAT_YEAR = 25920  # Astronomical precession cycle (years)
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')  # Beyond machine precision



# ============================================================================
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
# ============================================================================
def integrate_pell_prime_prediction(target_number: int, constants: UPGConstants = None):
    """Integrate Pell sequence prime prediction with this tool"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        # Fallback if Pell module not available
        return {'target_number': target_number, 'is_prime': None, 'note': 'Pell module not available'}



# ============================================================================
# GREAT YEAR ASTRONOMICAL PRECESSION INTEGRATION
# ============================================================================
def integrate_great_year_precession(year: int, constants: UPGConstants = None):
    """Integrate Great Year (25,920-year) precession cycle"""
    try:
        from pell_sequence_prime_prediction_upg_complete import GreatYearIntegration, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        great_year = GreatYearIntegration(constants)
        return great_year.consciousness_amplitude_from_year(year)
    except ImportError:
        # Fallback calculation
        if constants is None:
            constants = UPGConstants()
        angle = (year * 2 * math.pi) / constants.GREAT_YEAR
        return complex(float(angle * constants.CONSCIOUSNESS * constants.REALITY_DISTORTION), 0.0)



# Add Bram Cohen's synth to path
sys.path.append('bram_cohen_synth')

class PrimeZetaContraryMotion:
    """Generate audio of prime and zeta distributions in contrary motion"""

    def __init__(self):
        self.sample_rate = 44100
        self.target_total_duration = 142.0  # 2 minutes 22 seconds
        self.base_freq = 220  # A3

        # Musical scale for mapping values
        self.scale_degrees = [0, 2, 4, 5, 7, 9, 11, 12]  # Major scale

        # Generate primes and zeta values
        self.primes = self.generate_primes(100)
        self.zeta_values = self.generate_zeta_values(100)

    def generate_primes(self, n: int) -> List[int]:
        """Generate first n prime numbers"""
        primes = []
        num = 2
        while len(primes) < n:
            if self.is_prime(num):
                primes.append(num)
            num += 1
        return primes

    def is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    def generate_zeta_values(self, n: int) -> List[float]:
        """Generate Riemann zeta function values for first n integers"""
        zeta_values = []
        for k in range(1, n + 1):
            # Use approximation for zeta(k) = sum(1/i^k for i in 1 to infinity)
            zeta_sum = 0
            for i in range(1, 100):  # Approximate with first 100 terms
                zeta_sum += 1 / (i ** k)
            zeta_values.append(zeta_sum)
        return zeta_values

    def map_to_frequency(self, value: float, min_val: float, max_val: float,
                        octave_range: int = 2) -> float:
        """Map a mathematical value to a musical frequency"""
        # Normalize value to 0-1 range
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = (value - min_val) / (max_val - min_val)

        # Map to scale degrees
        scale_index = int(normalized * (len(self.scale_degrees) - 1))
        scale_degree = self.scale_degrees[scale_index]

        # Add octave variation
        octave = int(normalized * octave_range)

        # Calculate frequency
        freq = self.base_freq * (2 ** (octave + scale_degree / 12))
        return freq

    def create_contrary_motion_sequence(self, length: int = 50) -> Dict[str, List[float]]:
        """Create sequences for prime and zeta distributions in contrary motion"""

        # Get data for both sequences
        prime_data = self.primes[:length]
        zeta_data = self.zeta_values[:length]

        # Calculate prime gaps for timing alignment
        prime_gaps = [prime_data[0]]  # First prime has no gap, use value itself
        for i in range(1, len(prime_data)):
            gap = prime_data[i] - prime_data[i-1]
            prime_gaps.append(gap)

        # Normalize the data for frequency mapping
        prime_min, prime_max = min(prime_data), max(prime_data)
        zeta_min, zeta_max = min(zeta_data), max(zeta_data)

        # Calculate note durations to reach exact target total duration
        # Scale the gap-based durations so total equals target_total_duration
        gap_min, gap_max = min(prime_gaps), max(prime_gaps)
        if gap_max == gap_min:
            base_durations = [1.0] * len(prime_gaps)
        else:
            base_durations = []
            for gap in prime_gaps:
                # Map gap to duration: smaller gaps = shorter notes, larger gaps = longer notes
                base_duration = 0.2 + (gap - gap_min) / (gap_max - gap_min) * 0.8
                base_durations.append(base_duration)

        # Scale durations to match target total duration
        current_total = sum(base_durations)
        if current_total > 0:
            duration_scale = self.target_total_duration / current_total
            note_durations = [d * duration_scale for d in base_durations]
        else:
            note_durations = [self.target_total_duration / length] * length

        # Verify total duration
        actual_total = sum(note_durations)
        print(f"   Target duration: {self.target_total_duration:.1f}s")
        print(f"   Actual duration: {actual_total:.1f}s")

        prime_freqs = []
        zeta_freqs = []

        for i in range(length):
            # Prime distribution (ascending trend)
            prime_value = prime_data[i]
            prime_freq = self.map_to_frequency(prime_value, prime_min, prime_max)
            prime_freqs.append(prime_freq)

            # Zeta distribution (descending trend for contrary motion)
            zeta_value = zeta_data[length - 1 - i]  # Reverse order
            zeta_freq = self.map_to_frequency(zeta_value, zeta_min, zeta_max)
            zeta_freqs.append(zeta_freq)

        return {
            'prime_frequencies': prime_freqs,
            'zeta_frequencies': zeta_freqs,
            'prime_data': prime_data,
            'zeta_data': zeta_data,
            'prime_gaps': prime_gaps,
            'note_durations': note_durations,
            'target_duration': self.target_total_duration,
            'actual_duration': actual_total
        }

    def generate_sine_wave(self, frequency: float, duration: float,
                          amplitude: float = 0.3) -> List[float]:
        """Generate a sine wave at given frequency and duration"""
        num_samples = int(self.sample_rate * duration)
        samples = []

        for i in range(num_samples):
            t = i / self.sample_rate
            sample = amplitude * math.sin(2 * math.pi * frequency * t)
            samples.append(sample)

        return samples

    def generate_audio_sequence(self, frequencies: List[float], durations: List[float] = None,
                              timbre: int = 12) -> List[float]:
        """Generate audio sequence using Bram Cohen's synthesizer"""

        # Use fixed duration if no durations provided
        if durations is None:
            default_duration = self.target_total_duration / len(frequencies)
            durations = [default_duration] * len(frequencies)

        try:
            # Import Bram's synth functions
            from xen12synth import make_clean_whole_note, convert_wav_data

            # Set timbre
            import xen12synth
            xen12synth.timbre = timbre
            xen12synth.level = 1.4
            xen12synth.sample_rate = self.sample_rate

            full_audio = []

            for freq, duration in zip(frequencies, durations):
                # Generate note using Bram's synth
                num_samples = int(self.sample_rate * duration)
                note_audio = make_clean_whole_note(freq, num_samples)

                if note_audio:
                    # Apply fade out to avoid clicks (scale fade time with note duration)
                    fade_samples = int(self.sample_rate * min(0.1, duration * 0.2))  # Up to 20% of note duration
                    for i in range(min(fade_samples, len(note_audio))):
                        fade_factor = (fade_samples - i) / fade_samples
                        note_audio[-(i+1)] *= fade_factor

                    full_audio.extend(note_audio)
                else:
                    # Fallback to sine wave if synth fails
                    sine_audio = self.generate_sine_wave(freq, duration, 0.2)
                    full_audio.extend(sine_audio)

            return full_audio

        except Exception as e:
            print(f"Error using Bram's synth, falling back to sine waves: {e}")
            # Fallback to sine waves
            full_audio = []
            for freq, duration in zip(frequencies, durations):
                sine_audio = self.generate_sine_wave(freq, duration, 0.2)
                full_audio.extend(sine_audio)
            return full_audio

    def mix_audio_tracks(self, track1: List[float], track2: List[float]) -> List[float]:
        """Mix two audio tracks together"""
        mixed = []
        max_len = max(len(track1), len(track2))

        # Pad shorter track with zeros
        track1_padded = track1 + [0] * (max_len - len(track1))
        track2_padded = track2 + [0] * (max_len - len(track2))

        for i in range(max_len):
            mixed_sample = track1_padded[i] + track2_padded[i]
            # Prevent clipping
            mixed_sample = max(-1.0, min(1.0, mixed_sample))
            mixed.append(mixed_sample)

        return mixed

    def save_wav_file(self, audio_data: List[float], filename: str):
        """Save audio data as WAV file"""
        try:
            # Convert to 16-bit PCM
            wav_data = b''
            for sample in audio_data:
                # Convert to 16-bit signed integer
                sample_int = int(sample * 32767)
                sample_int = max(-32768, min(32767, sample_int))
                wav_data += struct.pack('<h', sample_int)

            # Write WAV file
            with wave.open(filename, 'wb') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(self.sample_rate)
                wav_file.writeframes(wav_data)

            print(f"✅ Saved audio to {filename}")

        except Exception as e:
            print(f"❌ Error saving {filename}: {e}")

    def run_contrary_motion_analysis(self) -> Dict[str, Any]:
        """Run the complete contrary motion analysis"""

        print("🎵 Generating Prime Distribution vs Zeta Distribution in Contrary Motion")
        print("=" * 75)

        results = {}

        # 1. Generate mathematical sequences
        print("1. Generating mathematical sequences...")

        # Calculate optimal number of notes for target duration
        # Assuming average note duration of ~4-5 seconds gives us ~30-35 notes for 142 seconds
        estimated_notes = int(self.target_total_duration / 4.5)
        sequences = self.create_contrary_motion_sequence(estimated_notes)
        results['sequences'] = sequences

        print(f"   Target duration: {self.target_total_duration:.1f} seconds")
        print(f"   Using {len(sequences['prime_data'])} notes")
        print(f"   Prime sequence: {sequences['prime_data'][:10]}...")
        print(f"   Zeta sequence: {sequences['zeta_data'][:10]}...")

        # 2. Generate audio tracks
        print("2. Generating audio tracks...")

        # Get note durations aligned with prime progression
        default_duration = self.target_total_duration / len(sequences['prime_frequencies'])
        note_durations = sequences.get('note_durations', [default_duration] * len(sequences['prime_frequencies']))

        # Prime track (using timbre 12 - natural)
        print("   Generating prime distribution track (timing aligned with prime gaps)...")
        prime_audio = self.generate_audio_sequence(sequences['prime_frequencies'], note_durations, timbre=12)

        # Zeta track (using timbre 15 - stretched/round)
        print("   Generating zeta distribution track (same timing for contrary motion)...")
        zeta_audio = self.generate_audio_sequence(sequences['zeta_frequencies'], note_durations, timbre=15)

        results['audio_tracks'] = {
            'prime_audio_length': len(prime_audio),
            'zeta_audio_length': len(zeta_audio),
            'duration_seconds': len(prime_audio) / self.sample_rate
        }

        # 3. Mix tracks in contrary motion
        print("3. Mixing tracks in contrary motion...")
        mixed_audio = self.mix_audio_tracks(prime_audio, zeta_audio)

        # 4. Save audio files
        print("4. Saving audio files...")
        self.save_wav_file(prime_audio, 'prime_distribution_track.wav')
        self.save_wav_file(zeta_audio, 'zeta_distribution_track.wav')
        self.save_wav_file(mixed_audio, 'prime_zeta_contrary_motion.wav')

        results['files_generated'] = [
            'prime_distribution_track.wav',
            'zeta_distribution_track.wav',
            'prime_zeta_contrary_motion.wav'
        ]

        # 5. Analyze the results
        print("5. Analyzing results...")
        results['analysis'] = self.analyze_contrary_motion(sequences)

        return results

    def analyze_contrary_motion(self, sequences: Dict[str, List]) -> Dict[str, Any]:
        """Analyze the contrary motion patterns"""

        analysis = {}

        prime_freqs = sequences['prime_frequencies']
        zeta_freqs = sequences['zeta_frequencies']

        # Frequency range analysis
        analysis['frequency_ranges'] = {
            'prime_min': min(prime_freqs),
            'prime_max': max(prime_freqs),
            'zeta_min': min(zeta_freqs),
            'zeta_max': max(zeta_freqs)
        }

        # Motion analysis (are they truly contrary?)
        prime_directions = []
        zeta_directions = []

        for i in range(1, len(prime_freqs)):
            prime_dir = 1 if prime_freqs[i] > prime_freqs[i-1] else -1
            zeta_dir = 1 if zeta_freqs[i] > zeta_freqs[i-1] else -1
            prime_directions.append(prime_dir)
            zeta_directions.append(zeta_dir)

        contrary_motion_count = sum(1 for p, z in zip(prime_directions, zeta_directions) if p != z)
        contrary_percentage = contrary_motion_count / len(prime_directions) * 100

        analysis['contrary_motion_analysis'] = {
            'total_intervals': len(prime_directions),
            'contrary_intervals': contrary_motion_count,
            'contrary_percentage': contrary_percentage,
            'parallel_intervals': len(prime_directions) - contrary_motion_count
        }

        # Mathematical correlations
        analysis['mathematical_correlations'] = {
            'prime_sequence_growth': 'Exponential growth in prime gaps',
            'zeta_sequence_decay': 'Rapid convergence to 1 for large k',
            'harmonic_relationships': 'Both sequences relate to fundamental mathematical constants'
        }

        return analysis

    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""

        report = []
        report.append("🎵 PRIME DISTRIBUTION vs ZETA DISTRIBUTION")
        report.append("🎼 IN CONTRARY MOTION ANALYSIS")
        report.append("=" * 50)

        # Sequence Information
        sequences = results.get('sequences', {})
        report.append("\n📊 MATHEMATICAL SEQUENCES:")
        report.append(f"• Prime numbers: First 10 = {sequences.get('prime_data', [])[:10]}")
        report.append(f"• Zeta values: First 10 = {['.3f' for z in sequences.get('zeta_data', [])[:10]]}")
        prime_gaps = sequences.get('prime_gaps', [])
        if prime_gaps:
            report.append(f"• Prime gaps: First 10 = {prime_gaps[:10]}")
            report.append(f"• Gap range: {min(prime_gaps)} - {max(prime_gaps)}")

        # Audio Information
        audio_info = results.get('audio_tracks', {})
        duration = audio_info.get('duration_seconds', 0)
        target_duration = sequences.get('target_duration', 0)
        actual_duration = sequences.get('actual_duration', 0)
        note_durations = sequences.get('note_durations', [])

        report.append("\n🎵 AUDIO CHARACTERISTICS:")
        report.append(f"• Target duration: {target_duration:.1f} seconds (2:22)")
        report.append(f"• Actual duration: {actual_duration:.1f} seconds")
        report.append(f"• Total duration: {duration:.1f} seconds")
        report.append(f"• Notes per sequence: {len(sequences.get('prime_frequencies', []))}")

        if note_durations:
            report.append(f"• Note durations: {min(note_durations):.2f} - {max(note_durations):.2f} seconds")
            report.append(f"• Average note duration: {sum(note_durations)/len(note_durations):.2f} seconds")
            report.append(f"• Timing aligned with prime gap progression")
        else:
            avg_duration = sum(note_durations) / len(note_durations) if note_durations else target_duration / len(sequences.get('prime_frequencies', []))
            report.append(f"• Note duration: {avg_duration:.1f} seconds each")

        # Contrary Motion Analysis
        analysis = results.get('analysis', {})
        contrary = analysis.get('contrary_motion_analysis', {})
        contrary_pct = contrary.get('contrary_percentage', 0)

        report.append("\n🎼 CONTRARY MOTION ANALYSIS:")
        report.append(f"• Contrary motion percentage: {contrary_pct:.1f}%")
        report.append(f"• Total musical intervals: {contrary.get('total_intervals', 0)}")
        report.append(f"• Truly contrary intervals: {contrary.get('contrary_intervals', 0)}")

        # Files Generated
        files = results.get('files_generated', [])
        report.append("\n💾 AUDIO FILES GENERATED:")
        for file in files:
            report.append(f"• {file}")

        # Interpretation
        report.append("\n🎭 MUSICAL INTERPRETATION:")
        report.append("• Prime distribution: Represents the irregular, unpredictable nature of primes")
        report.append("• Zeta function: Shows the smooth, convergent behavior of zeta values")
        report.append("• Contrary motion: Highlights the fundamental differences between these sequences")
        report.append("• Harmonic tension: The opposition creates interesting musical dissonance")

        # Mathematical Insights
        report.append("\n🔢 MATHEMATICAL INSIGHTS:")
        report.append("• Primes grow without bound, zeta values converge to constants")
        report.append("• The Riemann Hypothesis connects these through zeta zeros")
        report.append("• Contrary motion reveals the complementary nature of these functions")

        # Chia Friends Connection
        report.append("\n🌱 CHIA FRIENDS CONNECTION:")
        report.append("• Puzzle coordinates (2156, 892) may relate to prime/zeta distributions")
        report.append("• Audio patterns might reveal hidden claiming mechanisms")
        report.append("• Consciousness mathematics could be encoded in these distributions")

        return "\n".join(report)

def main():
    """Main contrary motion analysis function"""

    print("🎵 Starting Prime vs Zeta Distribution Contrary Motion Analysis...")
    print("This will generate audio where mathematical sequences move in opposite directions...")

    analyzer = PrimeZetaContraryMotion()
    results = analyzer.run_contrary_motion_analysis()

    # Generate and display report
    report = analyzer.generate_analysis_report(results)
    print("\n" + report)

    # Save detailed results
    with open('prime_zeta_contrary_motion_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n📄 Detailed results saved to: prime_zeta_contrary_motion_results.json")
    print("🎵 Listen to the generated audio files to experience the mathematical contrary motion!")

if __name__ == "__main__":
    main()

