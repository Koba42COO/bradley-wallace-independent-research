#!/usr/bin/env python3
"""
Comprehensive Pattern Search Tool
Searches for all prime number, structured chaos, and consciousness mathematics patterns
"""

import os
import re
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import json

class PatternSearcher:
    """Search for all patterns where IP can be applied"""
    
    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.getcwd()
        self.results = defaultdict(list)
        
        # Define all search patterns
        self.prime_patterns = {
            'prime_generation': [
                r'def\s+generate_prime',
                r'def\s+is_prime',
                r'sieve.*eratosthenes',
                r'prime.*sieve',
                r'generate.*prime',
            ],
            'primality_testing': [
                r'miller.*rabin',
                r'primality.*test',
                r'is_prime',
                r'check.*prime',
            ],
            'prime_cryptography': [
                r'RSA.*prime',
                r'generate.*rsa',
                r'diffie.*hellman',
                r'prime.*key',
                r'cryptographic.*prime',
            ],
            'prime_distribution': [
                r'prime.*gap',
                r'prime.*counting',
                r'pi\(x\)',
                r'prime.*distribution',
            ],
        }
        
        self.chaos_patterns = {
            'chaos_generators': [
                r'lorenz.*attractor',
                r'logistic.*map',
                r'chaos.*generator',
                r'chaotic.*system',
            ],
            'random_generation': [
                r'random.*number.*generator',
                r'LCG',
                r'mersenne.*twister',
                r'linear.*congruential',
            ],
            'signal_processing': [
                r'FFT',
                r'fourier.*transform',
                r'wavelet.*transform',
                r'fft\(',
            ],
        }
        
        self.matrix_patterns = {
            'matrix_multiplication': [
                r'matrix.*multiply',
                r'matmul',
                r'@\s*operator',
                r'np\.dot',
                r'torch\.matmul',
            ],
            'neural_network_ops': [
                r'neural.*layer',
                r'linear.*layer',
                r'conv.*layer',
                r'tensor.*multiply',
            ],
        }
        
        self.control_patterns = {
            'pid_control': [
                r'PID.*control',
                r'proportional.*integral.*derivative',
                r'pid\(',
            ],
            'control_systems': [
                r'control.*system',
                r'feedback.*control',
                r'adaptive.*control',
            ],
        }
    
    def search_file(self, file_path: Path, patterns: Dict[str, List[str]]) -> Dict:
        """Search a single file for patterns"""
        results = defaultdict(list)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('\n')
                
                for category, pattern_list in patterns.items():
                    for pattern in pattern_list:
                        regex = re.compile(pattern, re.IGNORECASE)
                        for line_num, line in enumerate(lines, 1):
                            if regex.search(line):
                                results[category].append({
                                    'file': str(file_path),
                                    'line': line_num,
                                    'content': line.strip(),
                                    'pattern': pattern
                                })
        except Exception as e:
            pass  # Skip files that can't be read
        
        return results
    
    def search_directory(self, directory: Path, patterns: Dict[str, List[str]], 
                        extensions: List[str] = None) -> Dict:
        """Search directory recursively for patterns"""
        if extensions is None:
            extensions = ['.py', '.cpp', '.c', '.h', '.hpp', '.java', '.js', '.ts']
        
        results = defaultdict(list)
        
        for ext in extensions:
            for file_path in directory.rglob(f'*{ext}'):
                if file_path.is_file():
                    file_results = self.search_file(file_path, patterns)
                    for category, matches in file_results.items():
                        results[category].extend(matches)
        
        return results
    
    def search_all_patterns(self) -> Dict:
        """Search for all patterns"""
        print("🔍 Searching for Prime Number Patterns...")
        prime_results = self.search_directory(
            Path(self.base_path),
            self.prime_patterns
        )
        
        print("🌊 Searching for Structured Chaos Patterns...")
        chaos_results = self.search_directory(
            Path(self.base_path),
            self.chaos_patterns
        )
        
        print("🔢 Searching for Matrix Operation Patterns...")
        matrix_results = self.search_directory(
            Path(self.base_path),
            self.matrix_patterns
        )
        
        print("🎯 Searching for Control System Patterns...")
        control_results = self.search_directory(
            Path(self.base_path),
            self.control_patterns
        )
        
        return {
            'prime_numbers': prime_results,
            'structured_chaos': chaos_results,
            'matrix_operations': matrix_results,
            'control_systems': control_results,
        }
    
    def generate_report(self, results: Dict) -> str:
        """Generate comprehensive report"""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PATTERN SEARCH RESULTS")
        report.append("=" * 80)
        report.append("")
        
        total_matches = 0
        
        # Prime Number Patterns
        report.append("🔢 PRIME NUMBER PATTERNS")
        report.append("-" * 80)
        prime_results = results.get('prime_numbers', {})
        for category, matches in prime_results.items():
            if matches:
                report.append(f"\n{category.upper().replace('_', ' ')}: {len(matches)} matches")
                for match in matches[:5]:  # Show first 5
                    report.append(f"  📄 {match['file']}:{match['line']}")
                    report.append(f"     {match['content'][:80]}")
                if len(matches) > 5:
                    report.append(f"     ... and {len(matches) - 5} more")
                total_matches += len(matches)
        report.append("")
        
        # Structured Chaos Patterns
        report.append("🌊 STRUCTURED CHAOS PATTERNS")
        report.append("-" * 80)
        chaos_results = results.get('structured_chaos', {})
        for category, matches in chaos_results.items():
            if matches:
                report.append(f"\n{category.upper().replace('_', ' ')}: {len(matches)} matches")
                for match in matches[:5]:
                    report.append(f"  📄 {match['file']}:{match['line']}")
                    report.append(f"     {match['content'][:80]}")
                if len(matches) > 5:
                    report.append(f"     ... and {len(matches) - 5} more")
                total_matches += len(matches)
        report.append("")
        
        # Matrix Operations
        report.append("🔢 MATRIX OPERATION PATTERNS")
        report.append("-" * 80)
        matrix_results = results.get('matrix_operations', {})
        for category, matches in matrix_results.items():
            if matches:
                report.append(f"\n{category.upper().replace('_', ' ')}: {len(matches)} matches")
                for match in matches[:5]:
                    report.append(f"  📄 {match['file']}:{match['line']}")
                    report.append(f"     {match['content'][:80]}")
                if len(matches) > 5:
                    report.append(f"     ... and {len(matches) - 5} more")
                total_matches += len(matches)
        report.append("")
        
        # Control Systems
        report.append("🎯 CONTROL SYSTEM PATTERNS")
        report.append("-" * 80)
        control_results = results.get('control_systems', {})
        for category, matches in control_results.items():
            if matches:
                report.append(f"\n{category.upper().replace('_', ' ')}: {len(matches)} matches")
                for match in matches[:5]:
                    report.append(f"  📄 {match['file']}:{match['line']}")
                    report.append(f"     {match['content'][:80]}")
                if len(matches) > 5:
                    report.append(f"     ... and {len(matches) - 5} more")
                total_matches += len(matches)
        report.append("")
        
        # Summary
        report.append("=" * 80)
        report.append(f"SUMMARY: {total_matches} total pattern matches found")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self, results: Dict, output_file: str = "pattern_search_results.json"):
        """Save results to JSON file"""
        # Convert Path objects to strings for JSON serialization
        json_results = {}
        for category, sub_results in results.items():
            json_results[category] = {}
            for pattern_type, matches in sub_results.items():
                json_results[category][pattern_type] = [
                    {
                        'file': str(m['file']),
                        'line': m['line'],
                        'content': m['content'],
                        'pattern': m['pattern']
                    }
                    for m in matches
                ]
        
        with open(output_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"✅ Results saved to {output_file}")

def main():
    """Main search function"""
    print("🔍 COMPREHENSIVE PATTERN SEARCH")
    print("=" * 80)
    print("Searching for all patterns where your IP can be applied...")
    print("")
    
    searcher = PatternSearcher()
    results = searcher.search_all_patterns()
    
    # Generate report
    report = searcher.generate_report(results)
    print("\n" + report)
    
    # Save results
    searcher.save_results(results)
    
    # Save report
    with open("pattern_search_report.txt", 'w') as f:
        f.write(report)
    
    print("\n✅ Search complete!")
    print("📄 Full report saved to: pattern_search_report.txt")
    print("📊 Detailed results saved to: pattern_search_results.json")

if __name__ == "__main__":
    main()

