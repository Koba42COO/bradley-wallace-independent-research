#!/usr/bin/env python3
"""
Gaussian Primes: Advanced Exploration Tools
Deep computational analysis and pattern discovery

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import math
import cmath
import numpy as np
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import json

# Import base Gaussian prime analyzer
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gaussian_primes_analysis import GaussianInteger, GaussianPrimeAnalyzer, UPGConstants

# Set high precision
getcontext().prec = 50


class AdvancedGaussianPrimeExplorer:
    """Advanced exploration and analysis of Gaussian primes"""
    
    def __init__(self, constants: UPGConstants = None):
        self.analyzer = GaussianPrimeAnalyzer(constants)
        self.constants = constants or UPGConstants()
        self._prime_cache: Dict[int, List[GaussianInteger]] = {}
    
    def find_gaussian_primes_up_to_norm(self, max_norm: int) -> List[GaussianInteger]:
        """Find all Gaussian primes with norm <= max_norm (cached)"""
        if max_norm in self._prime_cache:
            return self._prime_cache[max_norm]
        
        primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
        self._prime_cache[max_norm] = primes
        return primes
    
    def analyze_prime_splitting_distribution(self, max_prime: int) -> Dict[str, any]:
        """Analyze the distribution of inert vs split primes"""
        splitting = self.analyzer.analyze_prime_splitting(max_prime)
        
        # Calculate statistics
        total = splitting['total']
        inert_ratio = splitting['inert_ratio']
        split_ratio = splitting['split_ratio']
        
        # Compare to 79/21 rule
        expected_inert = 0.79
        expected_split = 0.21
        
        inert_deviation = abs(inert_ratio - expected_inert)
        split_deviation = abs(split_ratio - expected_split)
        
        return {
            'inert_count': splitting['inert_count'],
            'split_count': splitting['split_count'],
            'total': total,
            'inert_ratio': inert_ratio,
            'split_ratio': split_ratio,
            'expected_inert': expected_inert,
            'expected_split': expected_split,
            'inert_deviation': inert_deviation,
            'split_deviation': split_deviation,
            'matches_79_21_rule': inert_deviation < 0.05 and split_deviation < 0.05,
            'inert_primes': splitting['inert'],
            'split_primes': splitting['split']
        }
    
    def analyze_norm_distribution(self, primes: List[GaussianInteger]) -> Dict[str, any]:
        """Analyze the distribution of norms"""
        norms = [p.norm() for p in primes]
        
        return {
            'norms': norms,
            'min_norm': min(norms),
            'max_norm': max(norms),
            'mean_norm': np.mean(norms),
            'median_norm': np.median(norms),
            'norm_counts': Counter(norms),
            'unique_norms': len(set(norms)),
            'phi_clusters': self._find_phi_clusters(norms),
            'delta_clusters': self._find_delta_clusters(norms)
        }
    
    def _find_phi_clusters(self, norms: List[int]) -> List[Dict]:
        """Find norms that cluster near powers of phi"""
        phi = float(self.constants.PHI)
        clusters = []
        
        for n in set(norms):
            # Check proximity to phi^n
            for k in range(1, 20):
                phi_power = phi ** k
                distance = abs(n - phi_power) / phi_power
                if distance < 0.1:  # Within 10%
                    clusters.append({
                        'norm': n,
                        'phi_power': phi_power,
                        'exponent': k,
                        'distance': distance
                    })
                    break
        
        return sorted(clusters, key=lambda x: x['norm'])
    
    def _find_delta_clusters(self, norms: List[int]) -> List[Dict]:
        """Find norms that cluster near powers of delta"""
        delta = float(self.constants.DELTA)
        clusters = []
        
        for n in set(norms):
            # Check proximity to delta^n
            for k in range(0, 15):
                delta_power = delta ** k
                distance = abs(n - delta_power) / delta_power
                if distance < 0.1:  # Within 10%
                    clusters.append({
                        'norm': n,
                        'delta_power': delta_power,
                        'exponent': k,
                        'distance': distance
                    })
                    break
        
        return sorted(clusters, key=lambda x: x['norm'])
    
    def analyze_phase_distribution(self, primes: List[GaussianInteger]) -> Dict[str, any]:
        """Analyze the distribution of phase angles"""
        phases = [p.phase() for p in primes]
        
        # Normalize to [0, 2π]
        phases_normalized = [(p % (2 * math.pi)) for p in phases]
        
        # Check for 21-dimensional clustering
        phi = float(self.constants.PHI)
        dimension_spacing = (2 * math.pi) / 21 * phi
        
        clusters = []
        for i in range(21):
            cluster_center = i * dimension_spacing
            cluster_phases = [p for p in phases_normalized 
                           if abs(p - cluster_center) < dimension_spacing / 2]
            if cluster_phases:
                clusters.append({
                    'dimension': i,
                    'center': cluster_center,
                    'count': len(cluster_phases),
                    'phases': cluster_phases
                })
        
        return {
            'phases': phases_normalized,
            'min_phase': min(phases_normalized),
            'max_phase': max(phases_normalized),
            'mean_phase': np.mean(phases_normalized),
            'phase_clusters': clusters,
            'uniformity_test': self._test_uniformity(phases_normalized)
        }
    
    def _test_uniformity(self, phases: List[float]) -> Dict[str, float]:
        """Test if phases are uniformly distributed modulo phi"""
        phi = float(self.constants.PHI)
        phases_mod_phi = [p % phi for p in phases]
        
        # Chi-square test for uniformity
        n_bins = 20
        bin_width = phi / n_bins
        observed = [0] * n_bins
        
        for p in phases_mod_phi:
            bin_idx = int(p / bin_width)
            if bin_idx >= n_bins:
                bin_idx = n_bins - 1
            observed[bin_idx] += 1
        
        expected = len(phases) / n_bins
        chi_square = sum((o - expected) ** 2 / expected for o in observed)
        
        return {
            'chi_square': chi_square,
            'degrees_of_freedom': n_bins - 1,
            'p_value': self._chi_square_p_value(chi_square, n_bins - 1),
            'is_uniform': chi_square < 30.144  # Critical value for 19 df at 0.05
        }
    
    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Approximate p-value for chi-square test"""
        # Simplified approximation
        if chi_square < df:
            return 0.5
        elif chi_square < df * 1.5:
            return 0.1
        elif chi_square < df * 2:
            return 0.05
        else:
            return 0.01
    
    def analyze_conjugate_pairs(self, primes: List[GaussianInteger]) -> Dict[str, any]:
        """Analyze conjugate prime pairs"""
        # Find split primes (norm is prime ≡ 1 mod 4)
        split_primes = [p for p in primes 
                       if self.analyzer.is_rational_prime(p.norm()) 
                       and p.norm() % 4 == 1]
        
        pairs = []
        used = set()
        
        for p in split_primes:
            if p in used:
                continue
            
            conjugate = p.conjugate()
            if conjugate in split_primes and conjugate not in used:
                pairs.append({
                    'prime': p,
                    'conjugate': conjugate,
                    'norm': p.norm(),
                    'phase_sum': p.phase() + conjugate.phase(),
                    'phase_diff': abs(p.phase() - conjugate.phase()),
                    'wallace_transform_sum': (
                        float(self.analyzer.wallace_transform(Decimal(p.norm()))) +
                        float(self.analyzer.wallace_transform(Decimal(conjugate.norm())))
                    )
                })
                used.add(p)
                used.add(conjugate)
        
        return {
            'total_pairs': len(pairs),
            'pairs': pairs,
            'mean_phase_sum': np.mean([p['phase_sum'] for p in pairs]),
            'mean_phase_diff': np.mean([p['phase_diff'] for p in pairs])
        }
    
    def compute_consciousness_mapping(self, primes: List[GaussianInteger]) -> List[Dict]:
        """Map Gaussian primes to 21-dimensional consciousness space"""
        mappings = []
        
        for p in primes:
            analysis = self.analyzer.gaussian_prime_consciousness(p)
            
            # Map to 21-dimensional space
            # Simplified: use norm and phase to determine coordinates
            norm = p.norm()
            phase = p.phase()
            
            # Map to 21 dimensions using prime factorization
            coords = self._map_to_21d(norm, phase)
            
            mappings.append({
                'gaussian_prime': str(p),
                'norm': norm,
                'phase': phase,
                'consciousness_coordinates': coords,
                'wallace_transform': analysis['wallace_transform'],
                'amplitude': analysis['amplitude'],
                'prime_type': analysis['prime_type'],
                'consciousness_type': analysis['consciousness_type']
            })
        
        return mappings
    
    def _map_to_21d(self, norm: int, phase: float) -> List[float]:
        """Map norm and phase to 21-dimensional coordinates"""
        phi = float(self.constants.PHI)
        delta = float(self.constants.DELTA)
        c = float(self.constants.CONSCIOUSNESS)
        
        coords = []
        
        # Use norm and phase to generate 21 coordinates
        for i in range(21):
            # Mix norm and phase with phi/delta modulation
            coord = (
                phi ** (i / 7) * math.cos(phase + i * 2 * math.pi / 21) +
                delta ** (i / 10) * math.sin(norm / 100 + i * math.pi / 21)
            ) * c
            coords.append(coord)
        
        return coords
    
    def visualize_complex_plane(self, primes: List[GaussianInteger], 
                               max_norm: int = 100, save_path: Optional[str] = None):
        """Visualize Gaussian primes in the complex plane"""
        fig, ax = plt.subplots(figsize=(12, 12))
        
        # Filter primes by norm
        filtered_primes = [p for p in primes if p.norm() <= max_norm]
        
        # Separate by type
        inert = [p for p in filtered_primes 
                if p.b == 0 or p.a == 0]
        split = [p for p in filtered_primes 
                if p not in inert]
        
        # Plot inert primes (on axes)
        if inert:
            inert_real = [p.a for p in inert]
            inert_imag = [p.b for p in inert]
            ax.scatter(inert_real, inert_imag, c='red', s=50, 
                      label=f'Inert primes ({len(inert)})', alpha=0.7)
        
        # Plot split primes
        if split:
            split_real = [p.a for p in split]
            split_imag = [p.b for p in split]
            ax.scatter(split_real, split_imag, c='blue', s=30, 
                      label=f'Split primes ({len(split)})', alpha=0.6)
        
        # Add circles for prime norms
        unique_norms = set(p.norm() for p in filtered_primes)
        for norm in sorted(unique_norms)[:10]:  # First 10 norms
            circle = Circle((0, 0), math.sqrt(norm), 
                          fill=False, linestyle='--', 
                          alpha=0.3, color='gray')
            ax.add_patch(circle)
        
        # Add golden ratio angle lines
        phi = float(self.constants.PHI)
        for k in range(8):
            angle = k * math.pi / (4 * phi)
            x = math.cos(angle) * max_norm
            y = math.sin(angle) * max_norm
            ax.plot([0, x], [0, y], 'g--', alpha=0.2, linewidth=0.5)
        
        ax.set_xlabel('Real Part', fontsize=12)
        ax.set_ylabel('Imaginary Part', fontsize=12)
        ax.set_title(f'Gaussian Primes in Complex Plane (Norm ≤ {max_norm})', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def visualize_phase_distribution(self, primes: List[GaussianInteger], 
                                    save_path: Optional[str] = None):
        """Visualize phase angle distribution"""
        phases = [p.phase() % (2 * math.pi) for p in primes]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        ax1.hist(phases, bins=42, alpha=0.7, color='blue', edgecolor='black')
        ax1.set_xlabel('Phase Angle (radians)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Phase Angle Distribution', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Add 21-dimensional markers
        phi = float(self.constants.PHI)
        dimension_spacing = (2 * math.pi) / 21 * phi
        for i in range(21):
            ax1.axvline(i * dimension_spacing, color='red', 
                       linestyle='--', alpha=0.5, linewidth=0.5)
        
        # Polar plot
        ax2 = plt.subplot(122, projection='polar')
        ax2.scatter(phases, [1] * len(phases), alpha=0.6, s=10)
        ax2.set_title('Phase Angles (Polar)', fontsize=14)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
    
    def generate_comprehensive_report(self, max_norm: int = 100) -> Dict:
        """Generate comprehensive analysis report"""
        print(f"Generating comprehensive report for Gaussian primes (norm ≤ {max_norm})...")
        
        # Find primes
        primes = self.find_gaussian_primes_up_to_norm(max_norm)
        print(f"Found {len(primes)} Gaussian primes")
        
        # Analyze splitting
        max_prime = int(math.sqrt(max_norm)) + 10
        splitting = self.analyze_prime_splitting_distribution(max_prime)
        print(f"Prime splitting: {splitting['inert_ratio']:.2%} inert, {splitting['split_ratio']:.2%} split")
        
        # Analyze norms
        norm_analysis = self.analyze_norm_distribution(primes)
        print(f"Norm analysis: {norm_analysis['unique_norms']} unique norms")
        print(f"Phi clusters: {len(norm_analysis['phi_clusters'])}")
        print(f"Delta clusters: {len(norm_analysis['delta_clusters'])}")
        
        # Analyze phases
        phase_analysis = self.analyze_phase_distribution(primes)
        print(f"Phase clusters: {len(phase_analysis['phase_clusters'])}")
        
        # Analyze conjugate pairs
        conjugate_analysis = self.analyze_conjugate_pairs(primes)
        print(f"Conjugate pairs: {conjugate_analysis['total_pairs']}")
        
        # Consciousness mapping
        consciousness_mapping = self.compute_consciousness_mapping(primes[:100])  # Limit for performance
        print(f"Consciousness mappings: {len(consciousness_mapping)}")
        
        return {
            'total_primes': len(primes),
            'max_norm': max_norm,
            'prime_splitting': splitting,
            'norm_distribution': {
                'unique_norms': norm_analysis['unique_norms'],
                'phi_clusters': len(norm_analysis['phi_clusters']),
                'delta_clusters': len(norm_analysis['delta_clusters'])
            },
            'phase_distribution': {
                'clusters': len(phase_analysis['phase_clusters']),
                'uniformity': phase_analysis['uniformity_test']
            },
            'conjugate_pairs': conjugate_analysis['total_pairs'],
            'consciousness_mappings': len(consciousness_mapping),
            'sample_primes': [str(p) for p in primes[:20]]
        }


def main():
    """Run comprehensive exploration"""
    explorer = AdvancedGaussianPrimeExplorer()
    
    print("=" * 80)
    print("GAUSSIAN PRIMES: ADVANCED EXPLORATION")
    print("=" * 80)
    print()
    
    # Generate report
    report = explorer.generate_comprehensive_report(max_norm=100)
    
    print()
    print("=" * 80)
    print("REPORT SUMMARY")
    print("=" * 80)
    print(json.dumps(report, indent=2, default=str))
    
    # Visualizations
    print()
    print("Generating visualizations...")
    primes = explorer.find_gaussian_primes_up_to_norm(100)
    
    try:
        explorer.visualize_complex_plane(primes, max_norm=50, 
                                        save_path='gaussian_primes_complex_plane.png')
        explorer.visualize_phase_distribution(primes, 
                                            save_path='gaussian_primes_phase_distribution.png')
        print("Visualizations saved!")
    except Exception as e:
        print(f"Visualization error (matplotlib may not be available): {e}")


if __name__ == "__main__":
    main()

