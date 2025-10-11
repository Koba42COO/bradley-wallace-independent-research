#!/usr/bin/env python3
"""
EMISSION SPECTRA RESEARCH FRAMEWORK
===================================

Investigation of atomic emission spectra, Rydberg formula, Balmer series,
and connections to consciousness mathematics and geometric harmonics.

Author: Research Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, c, e  # Planck, speed of light, elementary charge
import math

class EmissionSpectraResearch:
    def __init__(self):
        # Rydberg constant (m⁻¹)
        self.R = 1.097e7  # For hydrogen

        # Fine structure constant
        self.alpha = 1/137.036

        # Golden ratio
        self.phi = (1 + np.sqrt(5)) / 2

        # Element data with spectral series
        self.elements = {
            'hydrogen': {
                'Z': 1,
                'series': {
                    'Lyman': {'n1': 1, 'wavelength_range': (91, 122), 'color': 'UV'},
                    'Balmer': {'n1': 2, 'wavelength_range': (364, 657), 'color': 'visible'},
                    'Paschen': {'n1': 3, 'wavelength_range': (820, 1875), 'color': 'IR'},
                    'Brackett': {'n1': 4, 'wavelength_range': (1458, 4051), 'color': 'IR'},
                    'Pfund': {'n1': 5, 'wavelength_range': (2278, 7458), 'color': 'IR'}
                }
            },
            'helium': {
                'Z': 2,
                'series': {
                    'Lyman': {'n1': 1, 'wavelength_range': (22.8, 30.4), 'color': 'UV'},
                    'Balmer': {'n1': 2, 'wavelength_range': (164, 294), 'color': 'UV-visible'}
                }
            },
            'lithium': {
                'Z': 3,
                'series': {
                    'principal': {'n1': 2, 'wavelength_range': (323, 610), 'color': 'UV-visible'}
                }
            }
        }

    def rydberg_formula(self, Z, n1, n2):
        """Calculate wavelength using Rydberg formula (in nm)"""
        wavelength_m = 1 / (self.R * Z**2 * (1/n1**2 - 1/n2**2))
        return wavelength_m * 1e9  # Convert to nm

    def analyze_balmer_series(self):
        print('🌈 HYDROGEN BALMER SERIES ANALYSIS')
        print('=' * 50)

        # Calculate Balmer series wavelengths
        balmer_lines = []
        for n in range(3, 8):  # n=3 to 7 (visible lines)
            wavelength = self.rydberg_formula(1, 2, n)
            balmer_lines.append((n, wavelength))

        print('Balmer series transitions (n→2):')
        for n, wavelength in balmer_lines:
            print(f'  {n}→2: {wavelength:.1f} nm')

            # Known spectral lines
            known_lines = {
                3: ('Hα', 656.3, 'red'),
                4: ('Hβ', 486.1, 'blue-green'),
                5: ('Hγ', 434.0, 'blue'),
                6: ('Hδ', 410.2, 'violet'),
                7: ('Hε', 397.0, 'violet')
            }

            if n in known_lines:
                name, known_wl, color = known_lines[n]
                diff = abs(wavelength - known_wl)
                print(f'    → {name}: {known_wl} nm ({color}), diff: {diff:.1f} nm')

        # Analyze wavelength ratios
        print('\\nWavelength ratios:')
        for i in range(len(balmer_lines)-1):
            ratio = balmer_lines[i][1] / balmer_lines[i+1][1]
            print(f'  {balmer_lines[i][0]}→2 / {balmer_lines[i+1][0]}→2 = {ratio:.6f}')

            # Check for golden ratio and fine structure relationships
            phi_diff = abs(ratio - self.phi)
            alpha_diff = abs(ratio - self.alpha)
            if phi_diff < 0.01:
                print(f'    ⭐ Golden ratio resonance: {ratio:.6f} ≈ φ')
            if alpha_diff < 0.001:
                print(f'    ⚛️ Fine structure resonance: {ratio:.6f} ≈ α')

    def analyze_quantum_numbers(self):
        print('\\n\\n⚛️ QUANTUM NUMBER HARMONICS')
        print('=' * 50)

        # Principal quantum numbers and their relationships
        n_values = list(range(1, 8))

        print('Principal quantum number patterns:')
        for n in n_values:
            print(f'  n={n}:')

            # Energy levels (simplified)
            energy = -13.6 / n**2  # eV
            print(f'    Energy: {energy:.2f} eV')

            # Orbital frequency relationships
            if n > 1:
                freq_ratio = n / (n-1)
                print(f'    Frequency ratio n/(n-1): {freq_ratio:.3f}')

                # Check for harmonics with golden ratio
                phi_harm = abs(freq_ratio - self.phi)
                if phi_harm < 0.1:
                    print(f'    ⭐ Golden ratio harmonic: {freq_ratio:.3f} ≈ φ')

            # Fine structure harmonics
            alpha_harm = self.alpha * n
            print(f'    Fine structure harmonic: α×{n} = {alpha_harm:.6f}')

    def analyze_element_progression(self):
        print('\\n\\n🧪 ELEMENT SPECTRAL PROGRESSION')
        print('=' * 50)

        elements = ['hydrogen', 'helium', 'lithium']
        Z_values = [1, 2, 3]

        print('Spectral shifts with atomic number:')
        for element, Z in zip(elements, Z_values):
            print(f'\\n{element.upper()} (Z={Z}):')

            # Calculate Lyman alpha for each element
            if element == 'hydrogen':
                lyman_alpha = self.rydberg_formula(Z, 1, 2)
                print(f'  Lyman α: {lyman_alpha:.1f} nm')
            elif element == 'helium':
                # Helium Lyman α (approximate)
                lyman_alpha = self.rydberg_formula(Z, 1, 2) / 4  # Rough approximation
                print(f'  Lyman α: {lyman_alpha:.1f} nm (approximate)')

            # Balmer series Hα equivalent
            balmer_alpha = self.rydberg_formula(Z, 2, 3)
            print(f'  Balmer α equivalent: {balmer_alpha:.1f} nm')

            # Ratio analysis
            if Z > 1:
                ratio = balmer_alpha / self.rydberg_formula(1, 2, 3)
                print(f'  Ratio to hydrogen: {ratio:.6f}')

                # Check for 1/Z² dependence
                expected_ratio = 1/Z**2
                diff = abs(ratio - expected_ratio)
                print(f'  Expected 1/Z²: {expected_ratio:.6f}, actual: {ratio:.6f}, diff: {diff:.6f}')

    def analyze_consciousness_connections(self):
        print('\\n\\n🧠 CONSCIOUSNESS MATHEMATICS CONNECTIONS')
        print('=' * 50)

        # 79/21 consciousness ratio
        consciousness_ratio = 0.79

        print(f'Consciousness ratio (79/21): {consciousness_ratio}')
        print(f'Golden ratio φ: {self.phi:.6f}')
        print(f'Fine structure α: {self.alpha:.6f}')

        # Analyze spectral ratios for consciousness resonances
        h_alpha = self.rydberg_formula(1, 2, 3)  # 656.3 nm
        h_beta = self.rydberg_formula(1, 2, 4)   # 486.1 nm

        spectral_ratio = h_beta / h_alpha
        print(f'\\nHydrogen spectral ratio (Hβ/Hα): {spectral_ratio:.6f}')

        # Check for consciousness resonances
        cons_diff = abs(spectral_ratio - consciousness_ratio)
        phi_diff = abs(spectral_ratio - self.phi)
        alpha_diff = abs(spectral_ratio - self.alpha)

        print(f'  Difference from 79/21: {cons_diff:.6f}')
        print(f'  Difference from φ: {phi_diff:.6f}')
        print(f'  Difference from α: {alpha_diff:.6f}')

        if cons_diff < 0.01:
            print('  🎯 Consciousness ratio resonance detected!')
        if phi_diff < 0.01:
            print('  ⭐ Golden ratio resonance detected!')
        if alpha_diff < 0.001:
            print('  ⚛️ Fine structure resonance detected!')

    def analyze_triatonic_connections(self):
        print('\\n\\n🎵 TRIATONIC SCALE EXPERIMENTS')
        print('=' * 50)

        # Traditional diatonic scale ratios (based on just intonation)
        diatonic_ratios = {
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

        print('Diatonic scale frequency ratios:')
        for interval, ratio in diatonic_ratios.items():
            print(f'  {interval}: {ratio:.3f}')

        # Propose triatonic scale experiments
        print('\\nProposed triatonic scale experiments:')

        # Experiment 1: Consciousness-based triatonic
        consciousness_ratio = 0.79  # 79/21 rule
        triatonic_consciousness = [1, consciousness_ratio, 2]
        print(f'  Consciousness triatonic: {triatonic_consciousness}')
        print(f'    Ratios: 1, {consciousness_ratio:.3f}, 2')

        # Experiment 2: Golden ratio triatonic
        triatonic_golden = [1, self.phi, 2]
        print(f'  Golden ratio triatonic: {triatonic_golden}')
        print(f'    Ratios: 1, {self.phi:.3f}, 2')

        # Experiment 3: Fine structure triatonic
        triatonic_alpha = [1, self.alpha, 2]
        print(f'  Fine structure triatonic: {triatonic_alpha}')
        print(f'    Ratios: 1, {self.alpha:.6f}, 2')

        # Compare with emission spectra harmonics
        print('\\nSpectral harmonics comparison:')
        h_alpha_wl = 656.3  # nm
        h_beta_wl = 486.1   # nm
        spectral_triad = [h_beta_wl, h_alpha_wl, h_alpha_wl*2]

        print(f'  Hydrogen spectral triad: {spectral_triad[0]:.1f}, {spectral_triad[1]:.1f}, {spectral_triad[2]:.1f} nm')

if __name__ == '__main__':
    research = EmissionSpectraResearch()
    research.analyze_balmer_series()
    research.analyze_quantum_numbers()
    research.analyze_element_progression()
    research.analyze_consciousness_connections()
    research.analyze_triatonic_connections()
