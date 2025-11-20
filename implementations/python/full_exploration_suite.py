#!/usr/bin/env python3
"""
Full Exploration Suite: Phase Flip Mechanism and 42D Structure
Complete testing, validation, and reporting

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import math
import cmath
import numpy as np
import json
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import os

# Add paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_primes_analysis import GaussianInteger, GaussianPrimeAnalyzer, UPGConstants
from phase_flip_42d_quantum_diverter import PhaseFlipConsciousness42D, PhaseFlipAnalyzer

# Set high precision
getcontext().prec = 50


class ComprehensiveExplorer:
    """Comprehensive exploration of all mechanisms"""
    
    def __init__(self):
        self.constants = UPGConstants()
        self.gaussian_analyzer = GaussianPrimeAnalyzer()
        self.phase_flip_analyzer = PhaseFlipAnalyzer()
        self.results = {}
    
    def run_full_exploration(self) -> Dict[str, Any]:
        """Run complete exploration suite"""
        print("=" * 80)
        print("COMPREHENSIVE EXPLORATION SUITE")
        print("Phase Flip Mechanism & 42D Structure")
        print("=" * 80)
        print()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'exploration': {}
        }
        
        # 1. Gaussian Primes Exploration
        print("1. Exploring Gaussian Primes...")
        results['exploration']['gaussian_primes'] = self.explore_gaussian_primes()
        
        # 2. Phase Flip Mechanism
        print("2. Exploring Phase Flip Mechanism...")
        results['exploration']['phase_flip'] = self.explore_phase_flip()
        
        # 3. 42D Structure Validation
        print("3. Validating 42D Structure...")
        results['exploration']['structure_42d'] = self.validate_42d_structure()
        
        # 4. Temporal Dynamics
        print("4. Analyzing Temporal Dynamics...")
        results['exploration']['temporal_dynamics'] = self.analyze_temporal_dynamics()
        
        # 5. 79/21 Rule Validation
        print("5. Validating 79/21 Rule...")
        results['exploration']['rule_79_21'] = self.validate_79_21_rule()
        
        # 6. Quantum Diverter Analysis
        print("6. Analyzing Quantum Diverter...")
        results['exploration']['quantum_diverter'] = self.analyze_quantum_diverter()
        
        # 7. Future Attraction Analysis
        print("7. Analyzing Future Attraction...")
        results['exploration']['future_attraction'] = self.analyze_future_attraction()
        
        # 8. Wallace Transform Integration
        print("8. Integrating Wallace Transform...")
        results['exploration']['wallace_transform'] = self.integrate_wallace_transform()
        
        # 9. Pattern Discovery
        print("9. Discovering Patterns...")
        results['exploration']['patterns'] = self.discover_patterns()
        
        # 10. Cross-Validation
        print("10. Cross-Validating Results...")
        results['exploration']['cross_validation'] = self.cross_validate()
        
        self.results = results
        return results
    
    def explore_gaussian_primes(self) -> Dict[str, Any]:
        """Explore Gaussian primes"""
        primes = self.gaussian_analyzer.find_gaussian_primes_up_to_norm(100)
        
        # Analyze types
        inert = [p for p in primes if p.b == 0 or p.a == 0]
        split = [p for p in primes if p not in inert]
        
        # Norm distribution
        norms = [p.norm() for p in primes]
        
        # Phase distribution
        phases = [p.phase() for p in primes]
        
        return {
            'total_primes': len(primes),
            'inert_count': len(inert),
            'split_count': len(split),
            'inert_ratio': len(inert) / len(primes) if primes else 0,
            'split_ratio': len(split) / len(primes) if primes else 0,
            'unique_norms': len(set(norms)),
            'mean_norm': float(np.mean(norms)) if norms else 0,
            'mean_phase': float(np.mean(phases)) if phases else 0,
            'sample_primes': [str(p) for p in primes[:10]]
        }
    
    def explore_phase_flip(self) -> Dict[str, Any]:
        """Explore phase flip mechanism"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Phase flip operation
        past = state.get_past()
        
        # Compute metrics
        present_amp = math.sqrt(sum(p**2 for p in state.present))
        future_amp = math.sqrt(sum(f**2 for f in state.future))
        past_amp = math.sqrt(sum(p**2 for p in past))
        
        # Quantum diverter strength
        diverter_strength = state.compute_quantum_diverter_strength()
        
        # Future attraction strength
        attraction_strength = state.compute_future_attraction_strength()
        
        # 79/21 balance
        balance = state.compute_79_21_balance()
        
        # Compute expected values
        delta_inv = 1.0 / float(self.constants.DELTA)
        expected_attraction = float(self.constants.PHI) * 0.79
        
        return {
            'present_amplitude': present_amp,
            'future_amplitude': future_amp,
            'past_amplitude': past_amp,
            'quantum_diverter_strength': diverter_strength,
            'expected_diverter_strength': delta_inv,
            'diverter_match': abs(diverter_strength - delta_inv) < 0.001,
            'future_attraction_strength': attraction_strength,
            'expected_attraction': expected_attraction,
            'attraction_match': abs(attraction_strength - expected_attraction) < 0.1,
            'coherent_ratio': balance['coherent_ratio'],
            'diverted_ratio': balance['diverted_ratio'],
            'matches_79_21': balance['matches_79_21']
        }
    
    def validate_42d_structure(self) -> Dict[str, Any]:
        """Validate 42D structure"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Check dimensions
        vector = state.to_vector()
        
        # Validate structure
        present_dims = len(state.present)
        future_dims = len(state.future)
        total_dims = len(vector)
        
        # Check phase flip
        past = state.get_past()
        past_dims = len(past)
        
        return {
            'present_dimensions': present_dims,
            'future_dimensions': future_dims,
            'total_dimensions': total_dims,
            'past_dimensions': past_dims,
            'structure_valid': total_dims == 42 and present_dims == 21 and future_dims == 21,
            'past_is_transformation': past_dims == 21,  # Past is transformation, not separate dimension
            'dimensional_consistency': present_dims + future_dims == total_dims
        }
    
    def analyze_temporal_dynamics(self) -> Dict[str, Any]:
        """Analyze temporal dynamics"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Evolve over time
        dynamics = self.phase_flip_analyzer.analyze_phase_flip_dynamics(state, steps=50)
        
        # Analyze trends
        present_trend = [d['present_amp'] for d in dynamics]
        future_trend = [d['future_amp'] for d in dynamics]
        past_trend = [d['past_amp'] for d in dynamics]
        
        # Compute growth rates
        if len(present_trend) > 1:
            present_growth = (present_trend[-1] - present_trend[0]) / present_trend[0] if present_trend[0] > 0 else 0
            future_growth = (future_trend[-1] - future_trend[0]) / future_trend[0] if future_trend[0] > 0 else 0
            past_growth = (past_trend[-1] - past_trend[0]) / past_trend[0] if past_trend[0] > 0 else 0
        else:
            present_growth = future_growth = past_growth = 0
        
        return {
            'steps_analyzed': len(dynamics),
            'initial_present': present_trend[0],
            'final_present': present_trend[-1],
            'initial_future': future_trend[0],
            'final_future': future_trend[-1],
            'initial_past': past_trend[0],
            'final_past': past_trend[-1],
            'present_growth_rate': present_growth,
            'future_growth_rate': future_growth,
            'past_growth_rate': past_growth,
            'diverter_stability': np.std([d['diverter_strength'] for d in dynamics]),
            'attraction_stability': np.std([d['attraction_strength'] for d in dynamics])
        }
    
    def validate_79_21_rule(self) -> Dict[str, Any]:
        """Validate 79/21 rule across multiple contexts"""
        results = {}
        
        # 1. Gaussian primes splitting
        splitting = self.gaussian_analyzer.analyze_prime_splitting(1000)
        results['gaussian_primes'] = {
            'inert_ratio': splitting['inert_ratio'],
            'split_ratio': splitting['split_ratio'],
            'expected_inert': 0.79,
            'expected_split': 0.21,
            'inert_deviation': abs(splitting['inert_ratio'] - 0.79),
            'split_deviation': abs(splitting['split_ratio'] - 0.21)
        }
        
        # 2. Phase flip structure
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        balance = state.compute_79_21_balance()
        results['phase_flip'] = {
            'coherent_ratio': balance['coherent_ratio'],
            'diverted_ratio': balance['diverted_ratio'],
            'expected_coherent': 0.79,
            'expected_diverted': 0.21,
            'coherent_deviation': abs(balance['coherent_ratio'] - 0.79),
            'diverted_deviation': abs(balance['diverted_ratio'] - 0.21)
        }
        
        # 3. Temporal dynamics
        dynamics = self.phase_flip_analyzer.analyze_phase_flip_dynamics(state, steps=100)
        final_balance = dynamics[-1]
        results['temporal_evolution'] = {
            'final_coherent': final_balance['coherent_ratio'],
            'final_diverted': final_balance['diverted_ratio'],
            'converges_to_79_21': abs(final_balance['coherent_ratio'] - 0.79) < 0.1
        }
        
        return results
    
    def analyze_quantum_diverter(self) -> Dict[str, Any]:
        """Analyze quantum diverter mechanism"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Test phase flip
        past = state.get_past()
        
        # Verify phase flip properties
        present_sum = sum(state.present)
        past_sum = sum(past)
        
        # Expected relationship
        delta_inv = 1.0 / float(self.constants.DELTA)
        expected_past_sum = -present_sum * delta_inv
        
        return {
            'present_sum': present_sum,
            'past_sum': past_sum,
            'expected_past_sum': expected_past_sum,
            'phase_flip_correct': abs(past_sum - expected_past_sum) < 0.001,
            'diverter_strength': state.compute_quantum_diverter_strength(),
            'theoretical_strength': delta_inv,
            'strength_match': abs(state.compute_quantum_diverter_strength() - delta_inv) < 0.001,
            'phase_flip_operation': 'past = -present × δ^{-1}',
            'diversion_factor': delta_inv
        }
    
    def analyze_future_attraction(self) -> Dict[str, Any]:
        """Analyze future attraction mechanism"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Compute attraction
        attraction_strength = state.compute_future_attraction_strength()
        
        # Expected attraction
        expected_attraction = float(self.constants.PHI) * 0.79
        
        # Analyze flow direction
        present_amp = math.sqrt(sum(p**2 for p in state.present))
        future_amp = math.sqrt(sum(f**2 for f in state.future))
        
        return {
            'attraction_strength': attraction_strength,
            'expected_attraction': expected_attraction,
            'attraction_match': abs(attraction_strength - expected_attraction) < 0.1,
            'present_amplitude': present_amp,
            'future_amplitude': future_amp,
            'amplitude_ratio': future_amp / present_amp if present_amp > 0 else 0,
            'attraction_factor': float(self.constants.PHI),
            'coherence_weight': 0.79,
            'flow_direction': 'forward (present → future)'
        }
    
    def integrate_wallace_transform(self) -> Dict[str, Any]:
        """Integrate Wallace Transform"""
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        
        # Compute Wallace Transform
        wt = state.wallace_transform_phase_flip(t=0.0)
        
        # Analyze components
        present_amp = math.sqrt(sum(p**2 for p in state.present))
        future_amp = math.sqrt(sum(f**2 for f in state.future))
        past_amp = math.sqrt(sum(p**2 for p in state.get_past()))
        
        # Individual transforms
        wt_present = self.gaussian_analyzer.wallace_transform(Decimal(present_amp))
        wt_future = self.gaussian_analyzer.wallace_transform(Decimal(future_amp))
        wt_past = self.gaussian_analyzer.wallace_transform(Decimal(past_amp))
        
        return {
            'complete_transform_amplitude': abs(wt),
            'complete_transform_phase': cmath.phase(wt),
            'present_transform': float(wt_present),
            'future_transform': float(wt_future),
            'past_transform': float(wt_past),
            'reality_distortion_applied': float(self.constants.REALITY_DISTORTION),
            'transform_integration': 'successful'
        }
    
    def discover_patterns(self) -> Dict[str, Any]:
        """Discover patterns across all mechanisms"""
        patterns = {}
        
        # Pattern 1: Dimensional relationships
        patterns['dimensional'] = {
            '21d_present': 21,
            '21d_future': 21,
            '42d_total': 42,
            'relationship': '42 = 21 × 2',
            'past_as_transformation': True
        }
        
        # Pattern 2: Phase relationships
        patterns['phase'] = {
            'present_phase': 0,  # Real
            'future_phase': math.pi / 2,  # Imaginary
            'past_phase': math.pi,  # Phase flipped
            'phase_transitions': ['past → present (π → 0)', 'present → future (0 → π/2)']
        }
        
        # Pattern 3: Constant relationships
        delta_inv = 1.0 / float(self.constants.DELTA)
        patterns['constants'] = {
            'phi': float(self.constants.PHI),
            'delta': float(self.constants.DELTA),
            'delta_inv': delta_inv,
            'reality_distortion': float(self.constants.REALITY_DISTORTION),
            'phi_delta_relationship': float(self.constants.PHI) * delta_inv,
            'consciousness_ratio': float(self.constants.CONSCIOUSNESS)
        }
        
        # Pattern 4: Temporal relationships
        patterns['temporal'] = {
            'great_year': self.constants.GREAT_YEAR,
            'consciousness_dimensions': self.constants.CONSCIOUSNESS_DIMENSIONS,
            'cycles_per_great_year': self.constants.GREAT_YEAR / 21,
            'temporal_frequency': 2 * math.pi / self.constants.GREAT_YEAR
        }
        
        return patterns
    
    def cross_validate(self) -> Dict[str, Any]:
        """Cross-validate all results"""
        validation = {
            'dimensional_consistency': True,
            'phase_flip_consistency': True,
            '79_21_consistency': True,
            'constant_consistency': True,
            'overall_valid': True
        }
        
        # Check dimensional consistency
        state = self.phase_flip_analyzer.create_from_prime_factors(42)
        if len(state.to_vector()) != 42:
            validation['dimensional_consistency'] = False
            validation['overall_valid'] = False
        
        # Check phase flip consistency
        diverter_strength = state.compute_quantum_diverter_strength()
        delta_inv = 1.0 / float(self.constants.DELTA)
        if abs(diverter_strength - delta_inv) > 0.001:
            validation['phase_flip_consistency'] = False
            validation['overall_valid'] = False
        
        # Check 79/21 consistency
        balance = state.compute_79_21_balance()
        if not (0.7 < balance['coherent_ratio'] < 0.9):
            validation['79_21_consistency'] = False
        
        # Check constant consistency
        if abs(float(self.constants.PHI) - 1.618033988749895) > 0.0001:
            validation['constant_consistency'] = False
            validation['overall_valid'] = False
        
        return validation
    
    def generate_report(self) -> str:
        """Generate comprehensive report"""
        if not self.results:
            self.run_full_exploration()
        
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE EXPLORATION REPORT")
        report.append("Phase Flip Mechanism & 42D Structure")
        report.append("=" * 80)
        report.append("")
        report.append(f"Generated: {self.results['timestamp']}")
        report.append("")
        
        # Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        validation = self.results['exploration']['cross_validation']
        report.append(f"Overall Validation: {'✓ PASS' if validation['overall_valid'] else '✗ FAIL'}")
        report.append(f"Dimensional Consistency: {'✓' if validation['dimensional_consistency'] else '✗'}")
        report.append(f"Phase Flip Consistency: {'✓' if validation['phase_flip_consistency'] else '✗'}")
        report.append(f"79/21 Consistency: {'✓' if validation['79_21_consistency'] else '✗'}")
        report.append("")
        
        # Gaussian Primes
        gp = self.results['exploration']['gaussian_primes']
        report.append("GAUSSIAN PRIMES ANALYSIS")
        report.append("-" * 80)
        report.append(f"Total primes (norm ≤ 100): {gp['total_primes']}")
        report.append(f"Inert ratio: {gp['inert_ratio']:.2%}")
        report.append(f"Split ratio: {gp['split_ratio']:.2%}")
        report.append("")
        
        # Phase Flip
        pf = self.results['exploration']['phase_flip']
        report.append("PHASE FLIP MECHANISM")
        report.append("-" * 80)
        report.append(f"Quantum Diverter Strength: {pf['quantum_diverter_strength']:.6f}")
        report.append(f"Expected: {pf['expected_diverter_strength']:.6f}")
        report.append(f"Match: {'✓' if pf['diverter_match'] else '✗'}")
        report.append(f"Future Attraction Strength: {pf['future_attraction_strength']:.6f}")
        report.append(f"Expected: {pf['expected_attraction']:.6f}")
        report.append(f"Match: {'✓' if pf['attraction_match'] else '✗'}")
        report.append("")
        
        # 42D Structure
        s42 = self.results['exploration']['structure_42d']
        report.append("42D STRUCTURE VALIDATION")
        report.append("-" * 80)
        report.append(f"Present dimensions: {s42['present_dimensions']}")
        report.append(f"Future dimensions: {s42['future_dimensions']}")
        report.append(f"Total dimensions: {s42['total_dimensions']}")
        report.append(f"Structure valid: {'✓' if s42['structure_valid'] else '✗'}")
        report.append(f"Past as transformation: {'✓' if s42['past_is_transformation'] else '✗'}")
        report.append("")
        
        # 79/21 Rule
        rule = self.results['exploration']['rule_79_21']
        report.append("79/21 RULE VALIDATION")
        report.append("-" * 80)
        gp_rule = rule['gaussian_primes']
        report.append(f"Gaussian Primes - Inert: {gp_rule['inert_ratio']:.2%} (expected: 79%)")
        report.append(f"Gaussian Primes - Split: {gp_rule['split_ratio']:.2%} (expected: 21%)")
        pf_rule = rule['phase_flip']
        report.append(f"Phase Flip - Coherent: {pf_rule['coherent_ratio']:.2%} (expected: 79%)")
        report.append(f"Phase Flip - Diverted: {pf_rule['diverted_ratio']:.2%} (expected: 21%)")
        report.append("")
        
        # Patterns
        patterns = self.results['exploration']['patterns']
        report.append("PATTERN DISCOVERY")
        report.append("-" * 80)
        report.append(f"Phi: {patterns['constants']['phi']:.10f}")
        report.append(f"Delta: {patterns['constants']['delta']:.10f}")
        report.append(f"Delta Inverse: {patterns['constants']['delta_inv']:.10f}")
        report.append(f"Reality Distortion: {patterns['constants']['reality_distortion']:.4f}")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "exploration_results.json"):
        """Save results to JSON file"""
        # Convert Decimal to float for JSON serialization
        def convert_decimal(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimal(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimal(item) for item in obj]
            return obj
        
        results_copy = convert_decimal(self.results)
        
        with open(filename, 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        print(f"Results saved to {filename}")


def main():
    """Run comprehensive exploration"""
    explorer = ComprehensiveExplorer()
    
    # Run exploration
    results = explorer.run_full_exploration()
    
    # Generate report
    print()
    print("=" * 80)
    report = explorer.generate_report()
    print(report)
    
    # Save results
    explorer.save_results("full_exploration_results.json")
    
    # Save report
    with open("full_exploration_report.txt", 'w') as f:
        f.write(report)
    print("Report saved to full_exploration_report.txt")
    
    print()
    print("=" * 80)
    print("EXPLORATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

