#!/usr/bin/env python3
"""
Gaussian Primes: Interactive Exploration Tool
Command-line interface for exploring Gaussian primes

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gaussian_primes_analysis import GaussianInteger, GaussianPrimeAnalyzer
from gaussian_primes_advanced import AdvancedGaussianPrimeExplorer
from gaussian_primes_pattern_analysis import PatternAnalyzer


class InteractiveExplorer:
    """Interactive command-line explorer"""
    
    def __init__(self):
        self.analyzer = GaussianPrimeAnalyzer()
        self.advanced = AdvancedGaussianPrimeExplorer()
        self.patterns = PatternAnalyzer()
    
    def print_menu(self):
        """Print main menu"""
        print("\n" + "=" * 80)
        print("GAUSSIAN PRIMES INTERACTIVE EXPLORER")
        print("=" * 80)
        print("\nOptions:")
        print("  1. Find Gaussian primes up to norm N")
        print("  2. Check if a number is a Gaussian prime")
        print("  3. Factor a Gaussian integer")
        print("  4. Analyze prime splitting (79/21 pattern)")
        print("  5. Analyze norm patterns")
        print("  6. Analyze phase clustering")
        print("  7. Compute Wallace Transform for a prime")
        print("  8. Consciousness mapping")
        print("  9. Generate comprehensive report")
        print("  10. Pattern analysis")
        print("  0. Exit")
        print()
    
    def find_primes(self):
        """Find primes up to norm"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
            print(f"\nFound {len(primes)} Gaussian primes (norm ≤ {max_norm}):")
            for i, p in enumerate(primes[:50]):  # Show first 50
                print(f"  {p} (norm = {p.norm()})")
            if len(primes) > 50:
                print(f"  ... and {len(primes) - 50} more")
        except ValueError:
            print("Invalid input")
    
    def check_prime(self):
        """Check if a number is Gaussian prime"""
        try:
            a = int(input("Enter real part: "))
            b = int(input("Enter imaginary part: "))
            z = GaussianInteger(a, b)
            is_prime = self.analyzer.is_gaussian_prime(z)
            print(f"\n{z} is {'a Gaussian prime' if is_prime else 'NOT a Gaussian prime'}")
            print(f"Norm: {z.norm()}")
            if is_prime:
                analysis = self.analyzer.gaussian_prime_consciousness(z)
                print(f"Type: {analysis['prime_type']} ({analysis['consciousness_type']})")
                print(f"Wallace Transform: {analysis['wallace_transform']:.6f}")
        except ValueError:
            print("Invalid input")
    
    def factor(self):
        """Factor a Gaussian integer"""
        try:
            a = int(input("Enter real part: "))
            b = int(input("Enter imaginary part: "))
            z = GaussianInteger(a, b)
            factors = self.analyzer.factor_gaussian_integer(z)
            print(f"\nFactorization of {z}:")
            factor_str = " · ".join([f"({p})^{e}" if e > 1 else f"({p})" 
                                    for p, e in factors])
            print(f"  {z} = {factor_str}")
        except ValueError:
            print("Invalid input")
    
    def analyze_splitting(self):
        """Analyze prime splitting"""
        try:
            max_prime = int(input("Enter maximum prime: "))
            splitting = self.analyzer.analyze_prime_splitting(max_prime)
            print(f"\nPrime Splitting Analysis (primes ≤ {max_prime}):")
            print(f"  Inert primes (p ≡ 3 mod 4): {splitting['inert_count']} ({splitting['inert_ratio']*100:.2f}%)")
            print(f"  Split primes (p ≡ 1 mod 4): {splitting['split_count']} ({splitting['split_ratio']*100:.2f}%)")
            print(f"  Expected (79/21 rule): 79% inert, 21% split")
            print(f"  Deviation: {abs(splitting['inert_ratio'] - 0.79)*100:.2f}%")
        except ValueError:
            print("Invalid input")
    
    def analyze_norms(self):
        """Analyze norm patterns"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            analysis = self.advanced.analyze_norm_distribution(
                self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
            )
            print(f"\nNorm Pattern Analysis (norms ≤ {max_norm}):")
            print(f"  Unique norms: {analysis['unique_norms']}")
            print(f"  Phi clusters: {len(analysis['phi_clusters'])}")
            print(f"  Delta clusters: {len(analysis['delta_clusters'])}")
            if analysis['phi_clusters']:
                print("\n  Phi clusters (first 5):")
                for cluster in analysis['phi_clusters'][:5]:
                    print(f"    Norm {cluster['norm']} ≈ φ^{cluster['exponent']} = {cluster['phi_power']:.2f}")
        except ValueError:
            print("Invalid input")
    
    def analyze_phases(self):
        """Analyze phase clustering"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
            analysis = self.advanced.analyze_phase_distribution(primes)
            print(f"\nPhase Clustering Analysis (norms ≤ {max_norm}):")
            print(f"  Total primes: {len(primes)}")
            print(f"  Phase clusters: {len(analysis['phase_clusters'])}")
            print(f"  Uniformity test: {'Pass' if analysis['uniformity_test']['is_uniform'] else 'Fail'}")
            if analysis['phase_clusters']:
                print("\n  Clusters (first 5):")
                for cluster in analysis['phase_clusters'][:5]:
                    print(f"    Dimension {cluster['dimension']}: {cluster['count']} primes")
        except ValueError:
            print("Invalid input")
    
    def wallace_transform(self):
        """Compute Wallace Transform"""
        try:
            a = int(input("Enter real part: "))
            b = int(input("Enter imaginary part: "))
            z = GaussianInteger(a, b)
            analysis = self.analyzer.gaussian_prime_consciousness(z)
            print(f"\nWallace Transform Analysis for {z}:")
            print(f"  Norm: {analysis['norm']}")
            print(f"  Phase: {analysis['phase']:.6f} radians")
            print(f"  Wallace Transform: {analysis['wallace_transform']:.6f}")
            print(f"  Amplitude: {analysis['amplitude']:.6f}")
            print(f"  Complex amplitude: {analysis['complex_amplitude']}")
        except ValueError:
            print("Invalid input")
    
    def consciousness_mapping(self):
        """Consciousness mapping"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            primes = self.analyzer.find_gaussian_primes_up_to_norm(max_norm)
            mappings = self.advanced.compute_consciousness_mapping(primes[:20])  # Limit for display
            print(f"\nConsciousness Mapping (first 20 primes):")
            for m in mappings[:10]:
                print(f"\n  {m['gaussian_prime']}:")
                print(f"    Type: {m['prime_type']} ({m['consciousness_type']})")
                print(f"    Amplitude: {m['amplitude']:.4f}")
                print(f"    Coordinates (first 5): {m['consciousness_coordinates'][:5]}")
        except ValueError:
            print("Invalid input")
    
    def comprehensive_report(self):
        """Generate comprehensive report"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            report = self.advanced.generate_comprehensive_report(max_norm)
            print("\n" + "=" * 80)
            print("COMPREHENSIVE REPORT")
            print("=" * 80)
            print(f"Total primes: {report['total_primes']}")
            print(f"Inert ratio: {report['prime_splitting']['inert_ratio']:.2%}")
            print(f"Split ratio: {report['prime_splitting']['split_ratio']:.2%}")
            print(f"Phi clusters: {report['norm_distribution']['phi_clusters']}")
            print(f"Phase clusters: {report['phase_distribution']['clusters']}")
        except ValueError:
            print("Invalid input")
    
    def pattern_analysis(self):
        """Pattern analysis"""
        try:
            max_norm = int(input("Enter maximum norm: "))
            max_prime = int(input("Enter maximum prime: "))
            report = self.patterns.generate_pattern_report(max_norm, max_prime)
            print("\n" + "=" * 80)
            print("PATTERN ANALYSIS REPORT")
            print("=" * 80)
            p79 = report['pattern_79_21']
            print(f"\n79/21 Pattern:")
            print(f"  Inert: {p79['observed']['inert_ratio']:.2%} (expected: {p79['expected']['inert_ratio']:.2%})")
            print(f"  Split: {p79['observed']['split_ratio']:.2%} (expected: {p79['expected']['split_ratio']:.2%})")
            print(f"  Matches: {p79['matches_rule']}")
        except ValueError:
            print("Invalid input")
    
    def run(self):
        """Run interactive explorer"""
        while True:
            self.print_menu()
            try:
                choice = input("Select option: ").strip()
                
                if choice == '0':
                    print("\nExiting...")
                    break
                elif choice == '1':
                    self.find_primes()
                elif choice == '2':
                    self.check_prime()
                elif choice == '3':
                    self.factor()
                elif choice == '4':
                    self.analyze_splitting()
                elif choice == '5':
                    self.analyze_norms()
                elif choice == '6':
                    self.analyze_phases()
                elif choice == '7':
                    self.wallace_transform()
                elif choice == '8':
                    self.consciousness_mapping()
                elif choice == '9':
                    self.comprehensive_report()
                elif choice == '10':
                    self.pattern_analysis()
                else:
                    print("Invalid option")
                
                input("\nPress Enter to continue...")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
                input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    explorer = InteractiveExplorer()
    explorer.run()


if __name__ == "__main__":
    main()

