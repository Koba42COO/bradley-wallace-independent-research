#!/usr/bin/env python3
"""
MUSIC THEORY & GEOMETRY RESEARCH FRAMEWORK
==========================================

Investigation of diatonic scales, triatonic scale experiments,
and Stonehenge geometric measurements with consciousness mathematics connections.

Author: Research Framework
"""

import numpy as np
import math

class MusicGeometryResearch:
    def __init__(self):
        # Fundamental constants
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.alpha = 1/137.036          # Fine structure constant
        self.consciousness_ratio = 0.79 # 79/21 rule

        # Diatonic scale frequencies (A4 = 440 Hz)
        self.diatonic_scale = {
            'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13,
            'E': 329.63, 'F': 349.23, 'F#': 369.99, 'G': 392.00,
            'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88
        }

        # Just intonation ratios
        self.just_intonation = {
            'unison': 1/1,
            'minor_second': 16/15,
            'major_second': 9/8,
            'minor_third': 6/5,
            'major_third': 5/4,
            'perfect_fourth': 4/3,
            'tritone': 7/5,
            'perfect_fifth': 3/2,
            'minor_sixth': 8/5,
            'major_sixth': 5/3,
            'minor_seventh': 16/9,
            'major_seventh': 15/8,
            'octave': 2/1
        }

        # Stonehenge measurements (feet)
        self.stonehenge = {
            'outer_circle': 79.2,
            'inner_circle': 105.6,
            'calculation': '150.6 × 5 + 528 = 3168',
            'mile_calculation': '528 × 50 = 26400 feet (5 miles)'
        }

    def analyze_diatonic_scale(self):
        print('🎼 DIATONIC SCALE ANALYSIS')
        print('=' * 50)

        # Calculate frequency ratios
        c_freq = self.diatonic_scale['C']
        ratios = {}
        for note, freq in self.diatonic_scale.items():
            ratios[note] = freq / c_freq

        print('C major scale frequency ratios:')
        for note in ['C', 'D', 'E', 'F', 'G', 'A', 'B']:
            ratio = ratios[note]
            print(f'  {note}: {ratio:.6f}')

            # Check for golden ratio relationships
            phi_diff = abs(ratio - self.phi)
            phi_inv_diff = abs(ratio - 1/self.phi)
            if phi_diff < 0.01 or phi_inv_diff < 0.01:
                print(f'    ⭐ Golden ratio relationship detected!')

        # Compare with just intonation
        print('\\nComparison with just intonation:')
        just_scale = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
        just_ratios = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8]

        for note, just_ratio in zip(just_scale, just_ratios):
            equal_temp_ratio = ratios[note]
            diff = abs(equal_temp_ratio - just_ratio)
            print(f'  {note}: Equal temp {equal_temp_ratio:.6f}, Just {just_ratio:.6f}, diff {diff:.6f}')

    def propose_triatonic_experiments(self):
        print('\\n\\n🎵 TRIATONIC SCALE EXPERIMENTS')
        print('=' * 50)

        print('Traditional diatonic scale uses 7 notes per octave.')
        print('Triatonic scales use 3 notes per octave - exploring fundamental harmonics.')
        print()

        # Experiment 1: Consciousness triatonic
        print('EXPERIMENT 1: Consciousness-Based Triatonic Scale')
        print('Based on 79/21 consciousness ratio')
        consciousness_freqs = [261.63, 261.63 * self.consciousness_ratio, 523.25]  # C4, consciousness ratio, C5
        print(f'Frequencies: {consciousness_freqs[0]:.2f}, {consciousness_freqs[1]:.2f}, {consciousness_freqs[2]:.2f} Hz')
        print(f'Ratios: 1.000, {self.consciousness_ratio:.3f}, 2.000')
        print('Musical intervals: Unison, consciousness interval, octave')
        print()

        # Experiment 2: Golden ratio triatonic
        print('EXPERIMENT 2: Golden Ratio Triatonic Scale')
        print('Based on φ ≈ 1.618')
        golden_freqs = [261.63, 261.63 * self.phi, 523.25]  # C4, golden ratio, C5
        print(f'Frequencies: {golden_freqs[0]:.2f}, {golden_freqs[1]:.2f}, {golden_freqs[2]:.2f} Hz')
        print(f'Ratios: 1.000, {self.phi:.3f}, 2.000')
        print('Musical intervals: Unison, golden fifth, octave')
        print()

        # Experiment 3: Fine structure triatonic
        print('EXPERIMENT 3: Fine Structure Triatonic Scale')
        print('Based on α ≈ 0.007297')
        alpha_freqs = [261.63, 261.63 * self.alpha, 261.63 * self.alpha * self.phi]  # Microtonal
        print(f'Frequencies: {alpha_freqs[0]:.2f}, {alpha_freqs[1]:.6f}, {alpha_freqs[2]:.6f} Hz')
        print(f'Ratios: 1.000, {self.alpha:.6f}, {self.alpha * self.phi:.6f}')
        print('Musical intervals: Unison, quantum microtone, quantum-golden microtone')
        print()

        # Experiment 4: Platonic harmonics triatonic
        print('EXPERIMENT 4: Platonic Harmonics Triatonic Scale')
        print('Based on Platonic solid ratios (tetrahedron/octahedron ≈ √2)')
        sqrt2 = np.sqrt(2)
        platonic_freqs = [261.63, 261.63 * sqrt2, 523.25]  # C4, √2 ratio, C5
        print(f'Frequencies: {platonic_freqs[0]:.2f}, {platonic_freqs[1]:.2f}, {platonic_freqs[2]:.2f} Hz')
        print(f'Ratios: 1.000, {sqrt2:.3f}, 2.000')
        print('Musical intervals: Unison, quantum uncertainty interval, octave')
        print()

        # Experiment 5: Spectral harmonics triatonic
        print('EXPERIMENT 5: Emission Spectra Triatonic Scale')
        print('Based on hydrogen Balmer series ratios')
        h_beta_h_alpha = 486.1 / 656.3  # ≈ 0.741
        spectral_freqs = [261.63, 261.63 * h_beta_h_alpha, 523.25]
        print(f'Frequencies: {spectral_freqs[0]:.2f}, {spectral_freqs[1]:.2f}, {spectral_freqs[2]:.2f} Hz')
        print(f'Ratios: 1.000, {h_beta_h_alpha:.3f}, 2.000')
        print('Musical intervals: Unison, spectral ratio interval, octave')

    def analyze_triatonic_mathematics(self):
        print('\\n\\n🔢 TRIATONIC SCALE MATHEMATICS')
        print('=' * 50)

        # Generate triatonic scales and analyze their properties
        experiments = {
            'consciousness': self.consciousness_ratio,
            'golden': self.phi,
            'quantum': np.sqrt(2),
            'spectral': 486.1/656.3
        }

        for name, ratio in experiments.items():
            print(f'{name.upper()} triatonic scale:')
            freqs = [261.63, 261.63 * ratio, 523.25]
            cents = [0, 1200 * np.log2(ratio), 2400]  # Convert to cents

            print(f'  Frequencies: {freqs[0]:.2f}, {freqs[1]:.2f}, {freqs[2]:.2f} Hz')
            print(f'  Cents: {cents[0]:.1f}, {cents[1]:.1f}, {cents[2]:.1f}')

            # Interval analysis
            interval1 = cents[1] - cents[0]
            interval2 = cents[2] - cents[1]
            print(f'  Intervals: {interval1:.1f}¢, {interval2:.1f}¢')

            # Check for musical consonance
            if interval1 < 400:  # Less than major third
                print('  → Microtonal character (experimental)')
            elif 400 <= interval1 <= 700:  # Major third to tritone
                print('  → Traditional interval range')
            else:
                print('  → Extended interval range')

            print()

    def analyze_stonehenge_measurements(self):
        print('🪨 STONEHENGE GEOMETRIC ANALYSIS')
        print('=' * 50)

        sh = self.stonehenge
        print(f'Outer circle: {sh["outer_circle"]} ft')
        print(f'Inner circle: {sh["inner_circle"]} ft')
        print(f'Calculation: {sh["calculation"]}')
        print(f'Mile relationship: {sh["mile_calculation"]}')

        # Calculate the given formula
        calculation_result = 150.6 * 5 + 528
        mile_calculation = 528 * 50
        print(f'\\nActual calculations:')
        print(f'150.6 × 5 + 528 = {150.6 * 5} + 528 = {calculation_result}')
        print(f'528 × 50 = {mile_calculation} feet')
        print(f'528 × 50 in miles = {mile_calculation / 5280:.1f} miles')

        # Geometric relationships
        outer = sh['outer_circle']
        inner = sh['inner_circle']
        ratio = outer / inner
        print(f'\\nGeometric analysis:')
        print(f'Outer/Inner ratio: {ratio:.6f}')

        # Check for mathematical constants
        constants = {
            'π': np.pi,
            'φ (golden)': self.phi,
            '√2 (quantum)': np.sqrt(2),
            '√3 (fifth)': np.sqrt(3),
            'α (fine structure)': self.alpha,
            '79/21 (consciousness)': self.consciousness_ratio
        }

        print('\\nConstant relationships:')
        for name, value in constants.items():
            diff = abs(ratio - value)
            inv_diff = abs(ratio - 1/value)
            if diff < 0.01 or inv_diff < 0.01:
                print(f'  🎯 {name}: {value:.6f} (diff: {min(diff, inv_diff):.6f})')
            else:
                closest = min(diff, inv_diff, key=lambda x: abs(x))
                if closest < 0.1:
                    print(f'  • {name}: {value:.6f} (diff: {closest:.6f})')

        # Stonehenge as musical instrument?
        print('\\nMusical interpretation:')
        # Convert dimensions to frequencies (hypothetical)
        speed_of_sound = 1125  # ft/s at 70°F
        for name, dimension in [('Outer circle', outer), ('Inner circle', inner)]:
            # Treat as string length or pipe length
            frequency = speed_of_sound / (2 * dimension)  # Fundamental frequency
            note = self.frequency_to_note(frequency)
            print(f'  {name} ({dimension} ft): {frequency:.2f} Hz ≈ {note}')

    def frequency_to_note(self, freq):
        """Convert frequency to nearest note name"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        a4_freq = 440.0

        # Calculate semitones from A4
        semitones = 12 * np.log2(freq / a4_freq)
        note_index = round(semitones) % 12
        octave = 4 + (round(semitones) // 12)

        return f'{note_names[note_index]}{octave}'

    def analyze_unified_patterns(self):
        print('\\n\\n🌌 UNIFIED MATHEMATICAL PATTERNS')
        print('=' * 50)

        print('Connecting Platonic solids, emission spectra, music, and Stonehenge:')
        print()

        # Platonic connections to music
        print('1. PLATONIC SOLIDS → MUSIC:')
        tetrahedron_ratio = np.sqrt(2)  # Tetrahedron circumradius ratio
        print(f'   Tetrahedron harmonics: √2 ≈ {tetrahedron_ratio:.6f}')
        print('   Musical tritones and augmented fourths')

        dodecahedron_ratio = self.phi**2  # Dodecahedron/icosahedron relationship
        print(f'   Dodecahedron harmonics: φ² ≈ {dodecahedron_ratio:.6f}')
        print('   Golden ratio musical scales')
        print()

        # Spectral connections to geometry
        print('2. EMISSION SPECTRA → GEOMETRY:')
        h_alpha = 656.3e-9  # meters
        h_beta = 486.1e-9   # meters
        spectral_ratio = h_beta / h_alpha
        print(f'   Hydrogen spectral ratio: {spectral_ratio:.6f}')
        print('   Related to icosahedral/dodecahedral dual ratios')
        print()

        # Musical connections to consciousness
        print('3. MUSIC → CONSCIOUSNESS:')
        print(f'   79/21 consciousness ratio: {self.consciousness_ratio}')
        print('   Triatonic scales based on consciousness harmonics')
        print()

        # Stonehenge as unified system
        print('4. STONEHENGE → UNIFIED SYSTEM:')
        outer = self.stonehenge['outer_circle']
        inner = self.stonehenge['inner_circle']
        ratio = outer / inner
        print(f'   Stonehenge ratio: {ratio:.6f}')
        print('   May encode musical and geometric harmonics')
        print()

        # The unified pattern
        print('UNIFIED PATTERN HYPOTHESIS:')
        print('All these systems may be manifestations of the same underlying')
        print('mathematical structure that connects:')
        print('• Quantum mechanics (fine structure constant)')
        print('• Geometry (Platonic solids, golden ratio)')
        print('• Music (harmonic series, scales)')
        print('• Consciousness (79/21 rule)')
        print('• Ancient monuments (Stonehenge measurements)')

if __name__ == '__main__':
    research = MusicGeometryResearch()
    research.analyze_diatonic_scale()
    research.propose_triatonic_experiments()
    research.analyze_triatonic_mathematics()
    research.analyze_stonehenge_measurements()
    research.analyze_unified_patterns()
