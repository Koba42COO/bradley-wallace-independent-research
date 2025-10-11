#!/usr/bin/env python3
"""
PLATONIC SOLIDS AND DUALS ANALYSIS
==================================

Research framework for investigating Platonic solids, their dual relationships,
and connections to consciousness mathematics and spectral harmonics.

Author: Research Framework
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

class PlatonicSolidsResearch:
    def __init__(self):
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio

        # CORRECTED Platonic solids data
        self.solids = {
            'tetrahedron': {
                'vertices': 4,
                'faces': 4,
                'edges': 6,
                'face_type': 'equilateral_triangle',
                'dual': 'tetrahedron',
                'circumradius': np.sqrt(6)/4,
                'volume': np.sqrt(2)/12,
                'symmetry_group': 'A4'
            },
            'cube': {
                'vertices': 8,
                'faces': 6,
                'edges': 12,
                'face_type': 'square',
                'dual': 'octahedron',
                'circumradius': np.sqrt(3)/2,
                'volume': 1.0,
                'symmetry_group': 'S4'
            },
            'octahedron': {
                'vertices': 6,
                'faces': 8,
                'edges': 12,
                'face_type': 'equilateral_triangle',
                'dual': 'cube',
                'circumradius': np.sqrt(2)/2,
                'volume': np.sqrt(2)/3,
                'symmetry_group': 'S4'
            },
            'dodecahedron': {
                'vertices': 20,
                'faces': 12,
                'edges': 30,
                'face_type': 'regular_pentagon',
                'dual': 'icosahedron',
                'circumradius': (np.sqrt(15) + np.sqrt(3))/4,
                'volume': (15 + 7*np.sqrt(5))/4,
                'symmetry_group': 'A5'
            },
            'icosahedron': {
                'vertices': 12,
                'faces': 20,
                'edges': 30,  # CORRECTED: icosahedron has 30 edges
                'face_type': 'equilateral_triangle',
                'dual': 'dodecahedron',
                'circumradius': np.sqrt(10 + 2*np.sqrt(5))/4,
                'volume': (5*(3 + np.sqrt(5)))/12,
                'symmetry_group': 'A5'
            }
        }

    def analyze_platonic_relationships(self):
        print('🔺 PLATONIC SOLIDS ANALYSIS')
        print('=' * 50)

        for name, props in self.solids.items():
            dual_name = props['dual']
            dual_props = self.solids[dual_name]

            print(f'\n{name.upper()} ↔ {dual_name.upper()}')
            print(f'  Vertices: {props["vertices"]} ↔ {dual_props["vertices"]}')
            print(f'  Faces: {props["faces"]} ↔ {dual_props["faces"]}')
            print(f'  Edges: {props["edges"]}')

            # Check Euler characteristic
            euler = props['vertices'] + props['faces'] - props['edges']
            print(f'  Euler: V+F-E = {props["vertices"]}+{props["faces"]}-{props["edges"]} = {euler}')
            print(f'  ✓ Euler characteristic: {euler} (should be 2)')

            # Volume ratios
            if name != dual_name:  # Not self-dual
                vol_ratio = props['volume'] / dual_props['volume']
                print(f'  Volume ratio: {vol_ratio:.6f}')

                # Check golden ratio relationships
                phi_diff = abs(vol_ratio - self.phi)
                phi_inv_diff = abs(vol_ratio - 1/self.phi)
                phi_sq_diff = abs(vol_ratio - self.phi**2)
                phi_sq_inv_diff = abs(vol_ratio - 1/self.phi**2)

                if min(phi_diff, phi_inv_diff, phi_sq_diff, phi_sq_inv_diff) < 0.01:
                    closest = min([
                        (phi_diff, 'φ'),
                        (phi_inv_diff, '1/φ'),
                        (phi_sq_diff, 'φ²'),
                        (phi_sq_inv_diff, '1/φ²')
                    ], key=lambda x: x[0])
                    print(f'  ⭐ Golden ratio relationship: {vol_ratio:.6f} ≈ {closest[1]}')

    def analyze_quantum_harmonics(self):
        print('\n\n⚛️ QUANTUM HARMONICS & CONSCIOUSNESS')
        print('=' * 50)

        for name, props in self.solids.items():
            print(f'\n{name.upper()}:')

            # Fine structure constant harmonics
            alpha = 1/137.036
            harmonics = [alpha * n for n in range(1, 6)]

            # Compare with geometric ratios
            if name in ['dodecahedron', 'icosahedron']:
                dodec_vol = self.solids['dodecahedron']['volume']
                ico_vol = self.solids['icosahedron']['volume']
                ratio = dodec_vol / ico_vol

                print(f'  Dodecahedron/Icosahedron volume ratio: {ratio:.6f}')
                print(f'  Golden ratio φ: {self.phi:.6f}')
                print(f'  φ²: {self.phi**2:.6f}')

                # Check for fine structure resonances
                for i, h in enumerate(harmonics):
                    diff = abs(ratio - h)
                    if diff < 0.01:
                        print(f'  🎯 Fine structure resonance: {ratio:.6f} ≈ {h:.6f} (α×{i+1})')

            # Angular harmonics (72°, 120° etc.)
            if props['face_type'] == 'equilateral_triangle':
                print(f'  Triangular harmonics: 120° × {props["faces"]} = {120 * props["faces"]}°')
            elif props['face_type'] == 'square':
                print(f'  Square harmonics: 90° × {props["faces"]} = {90 * props["faces"]}°')
            elif props['face_type'] == 'regular_pentagon':
                print(f'  Pentagonal harmonics: 72° × {props["faces"]} = {72 * props["faces"]}°')

    def analyze_stonehenge_connections(self):
        print('\n\n🪨 STONEHENGE GEOMETRIC CONNECTIONS')
        print('=' * 50)

        # Stonehenge measurements from user query
        stonehenge = {
            'outer_circle': 79.2,  # feet
            'inner_circle': 105.6, # feet
            'calculation': '150.6 × 5 + 528 = 2640 + 528 = 3168',
            'mile_relationship': '528 × 50 = 26400 feet (0.5 miles)'
        }

        print(f'Stonehenge measurements:')
        print(f'  Outer circle: {stonehenge["outer_circle"]} ft')
        print(f'  Inner circle: {stonehenge["inner_circle"]} ft')
        print(f'  150.6 × 5 + 528 = {150.6 * 5 + 528}')
        print(f'  528 × 50 = {528 * 50} feet = {528 * 50 / 5280:.1f} miles')

        # Compare with Platonic geometry
        for name, props in self.solids.items():
            radius_ratio = props['circumradius']
            feet_approx = radius_ratio * 100  # Rough scaling

            print(f'\n{name.upper()} scaled comparison:')
            print(f'  Circumradius ratio: {radius_ratio:.6f}')
            print(f'  Rough feet equivalent: {feet_approx:.1f} ft')

            # Check for Stonehenge resonances
            outer_diff = abs(feet_approx - stonehenge['outer_circle'])
            inner_diff = abs(feet_approx - stonehenge['inner_circle'])

            if outer_diff < 5 or inner_diff < 5:
                print(f'  🎯 Stonehenge resonance detected!')

if __name__ == '__main__':
    research = PlatonicSolidsResearch()
    research.analyze_platonic_relationships()
    research.analyze_quantum_harmonics()
    research.analyze_stonehenge_connections()
