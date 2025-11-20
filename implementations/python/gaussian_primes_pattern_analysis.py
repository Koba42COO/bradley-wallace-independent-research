#!/usr/bin/env python3
"""
Gaussian Primes Pattern Analysis
Deep pattern discovery and statistical analysis

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import math
import numpy as np
from decimal import Decimal
from typing import List, Dict, Tuple
from collections import Counter
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gaussian_primes_analysis import GaussianInteger, GaussianPrimeAnalyzer, UPGConstants


class PatternAnalyzer:
    """Advanced pattern analysis for Gaussian primes"""
    
    def __init__(self):
        self.analyzer = GaussianPrimeAnalyzer()
        self.constants = UPGConstants()
    
    def analyze_79_21_pattern(self, max_prime: int = 1000) -> Dict:
        """Analyze the 79/21 pattern in prime splitting"""
        splitting = self.analyzer.analyze_prime_splitting(max_prime)
        
        inert_ratio = splitting['inert_ratio']
        split_ratio = splitting['split_ratio']
        
        # Calculate deviation from 79/21
        expected_inert = 0.79
        expected_split = 0.21
        
        deviation = {
            'inert': abs(inert_ratio - expected_inert),
            'split': abs(split_ratio - expected_split),
            'total': abs(inert_ratio - expected_inert) + abs(split_ratio - expected_split)
        }
        
        # Statistical significance
        total = splitting['total']
        inert_expected = total * expected_inert
        split_expected = total * expected_split
        
        # Chi-square test
        chi_square = (
            (splitting['inert_count'] - inert_expected) ** 2 / inert_expected +
            (splitting['split_count'] - split_expected) ** 2 / split_expected
        )
        
        return {
            'observed': {
                'inert_ratio': inert_ratio,
                'split_ratio': split_ratio,
                'inert_count': splitting['inert_count'],
                'split_count': splitting['split_count']
            },
            'expected': {
                'inert_ratio': expected_inert,
                'split_ratio': expected_split
            },
            'deviation': deviation,
            'chi_square': chi_square,
            'p_value': self._chi_square_p_value(chi_square, 1),
            'matches_rule': deviation['total'] < 0.1,
            'total_primes': total
        }
    
    def _chi_square_p_value(self, chi_square: float, df: int) -> float:
        """Approximate p-value"""
        if chi_square < 0.5:
            return 0.5
        elif chi_square < 2.7:
            return 0.1
        elif chi_square < 3.84:
            return 0.05
        else:
            return 0.01
    
    def analyze_norm_patterns(self, max_norm: int = 200) -> Dict:
        """Analyze patterns in norm distribution"""
        primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
        norms = [p.norm() for p in primes]
        
        # Find patterns
        patterns = {
            'prime_squares': [],
            'sums_of_squares': [],
            'phi_related': [],
            'delta_related': [],
            'multiples_of_21': []
        }
        
        phi = float(self.constants.PHI)
        delta = float(self.constants.DELTA)
        
        unique_norms = sorted(set(norms))
        
        for norm in unique_norms:
            # Check if norm is a perfect square of a prime
            sqrt_norm = int(math.sqrt(norm))
            if sqrt_norm * sqrt_norm == norm and self.analyzer.is_rational_prime(sqrt_norm):
                patterns['prime_squares'].append({
                    'norm': norm,
                    'prime': sqrt_norm
                })
            
            # Check proximity to phi powers
            for k in range(1, 15):
                phi_power = phi ** k
                if abs(norm - phi_power) / phi_power < 0.15:
                    patterns['phi_related'].append({
                        'norm': norm,
                        'phi_power': phi_power,
                        'exponent': k,
                        'ratio': norm / phi_power
                    })
                    break
            
            # Check proximity to delta powers
            for k in range(0, 12):
                delta_power = delta ** k
                if abs(norm - delta_power) / delta_power < 0.15:
                    patterns['delta_related'].append({
                        'norm': norm,
                        'delta_power': delta_power,
                        'exponent': k,
                        'ratio': norm / delta_power
                    })
                    break
            
            # Check multiples of 21
            if norm % 21 == 0:
                patterns['multiples_of_21'].append(norm)
        
        return {
            'total_norms': len(unique_norms),
            'patterns': patterns,
            'norm_frequency': Counter(norms)
        }
    
    def analyze_phase_clustering(self, max_norm: int = 200) -> Dict:
        """Analyze phase angle clustering in 21 dimensions"""
        primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
        
        phases = [p.phase() % (2 * math.pi) for p in primes]
        
        # 21-dimensional clustering
        phi = float(self.constants.PHI)
        dimension_spacing = (2 * math.pi) / 21 * phi
        
        clusters = {}
        for i in range(21):
            center = i * dimension_spacing
            cluster_primes = []
            
            for j, p in enumerate(primes):
                phase = phases[j]
                distance = min(
                    abs(phase - center),
                    abs(phase - center + 2 * math.pi),
                    abs(phase - center - 2 * math.pi)
                )
                
                if distance < dimension_spacing / 2:
                    cluster_primes.append({
                        'prime': str(p),
                        'norm': p.norm(),
                        'phase': phase,
                        'distance_from_center': distance
                    })
            
            if cluster_primes:
                clusters[i] = {
                    'dimension': i,
                    'center': center,
                    'count': len(cluster_primes),
                    'primes': cluster_primes[:10]  # Limit for output
                }
        
        return {
            'total_primes': len(primes),
            'dimensions_with_primes': len(clusters),
            'clusters': clusters,
            'dimension_spacing': dimension_spacing
        }
    
    def analyze_wallace_transform_patterns(self, max_norm: int = 100) -> Dict:
        """Analyze patterns in Wallace Transform values"""
        primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
        
        wt_values = []
        for p in primes:
            norm = Decimal(p.norm())
            wt = self.analyzer.wallace_transform(norm)
            wt_values.append({
                'prime': str(p),
                'norm': p.norm(),
                'wallace_transform': float(wt),
                'phase': p.phase()
            })
        
        # Find special values
        phi = float(self.constants.PHI)
        special_values = {
            'near_phi': [],
            'near_zero': [],
            'near_one': []
        }
        
        for wt_data in wt_values:
            wt_val = abs(wt_data['wallace_transform'])
            
            if abs(wt_val - phi) < 0.1:
                special_values['near_phi'].append(wt_data)
            if abs(wt_val) < 0.1:
                special_values['near_zero'].append(wt_data)
            if abs(wt_val - 1.0) < 0.1:
                special_values['near_one'].append(wt_data)
        
        return {
            'total_analyzed': len(wt_values),
            'mean_wt': np.mean([w['wallace_transform'] for w in wt_values]),
            'std_wt': np.std([w['wallace_transform'] for w in wt_values]),
            'special_values': special_values,
            'sample_values': wt_values[:20]
        }
    
    def generate_pattern_report(self, max_norm: int = 200, max_prime: int = 1000) -> Dict:
        """Generate comprehensive pattern analysis report"""
        print("Analyzing 79/21 pattern...")
        pattern_79_21 = self.analyze_79_21_pattern(max_prime)
        
        print("Analyzing norm patterns...")
        norm_patterns = self.analyze_norm_patterns(max_norm)
        
        print("Analyzing phase clustering...")
        phase_clustering = self.analyze_phase_clustering(max_norm)
        
        print("Analyzing Wallace Transform patterns...")
        wt_patterns = self.analyze_wallace_transform_patterns(max_norm)
        
        return {
            'pattern_79_21': pattern_79_21,
            'norm_patterns': norm_patterns,
            'phase_clustering': phase_clustering,
            'wallace_transform_patterns': wt_patterns
        }


def main():
    """Run pattern analysis"""
    analyzer = PatternAnalyzer()
    
    print("=" * 80)
    print("GAUSSIAN PRIMES PATTERN ANALYSIS")
    print("=" * 80)
    print()
    
    report = analyzer.generate_pattern_report(max_norm=200, max_prime=1000)
    
    print("\n=== 79/21 PATTERN ANALYSIS ===")
    p79 = report['pattern_79_21']
    print(f"Inert ratio: {p79['observed']['inert_ratio']:.2%} (expected: {p79['expected']['inert_ratio']:.2%})")
    print(f"Split ratio: {p79['observed']['split_ratio']:.2%} (expected: {p79['expected']['split_ratio']:.2%})")
    print(f"Matches rule: {p79['matches_rule']}")
    print(f"Chi-square: {p79['chi_square']:.4f}")
    
    print("\n=== NORM PATTERNS ===")
    np = report['norm_patterns']
    print(f"Prime squares: {len(np['patterns']['prime_squares'])}")
    print(f"Phi-related: {len(np['patterns']['phi_related'])}")
    print(f"Delta-related: {len(np['patterns']['delta_related'])}")
    print(f"Multiples of 21: {len(np['patterns']['multiples_of_21'])}")
    
    print("\n=== PHASE CLUSTERING ===")
    pc = report['phase_clustering']
    print(f"Dimensions with primes: {pc['dimensions_with_primes']}/21")
    print(f"Dimension spacing: {pc['dimension_spacing']:.4f} radians")
    
    print("\n=== WALLACE TRANSFORM PATTERNS ===")
    wt = report['wallace_transform_patterns']
    print(f"Mean WT value: {wt['mean_wt']:.4f}")
    print(f"Near phi: {len(wt['special_values']['near_phi'])}")
    print(f"Near zero: {len(wt['special_values']['near_zero'])}")
    print(f"Near one: {len(wt['special_values']['near_one'])}")


if __name__ == "__main__":
    main()

