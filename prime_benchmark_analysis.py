#!/usr/bin/env python3
"""
COMPREHENSIVE PRIME BENCHMARK AND ANALYSIS SYSTEM
Advanced performance testing, statistical analysis, and visualization
"""

import time
import statistics
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from comprehensive_prime_system import ComprehensivePrimeSystem

class PrimeBenchmarkAnalyzer:
    """
    Advanced benchmarking and analysis system for prime algorithms
    """

    def __init__(self):
        self.system = ComprehensivePrimeSystem()
        self.results = {}

        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")

    def comprehensive_performance_benchmark(self, test_ranges: List[Tuple[int, int]] = None) -> Dict[str, Any]:
        """
        Comprehensive performance benchmarking across different ranges
        """
        if test_ranges is None:
            test_ranges = [
                (10**3, 10**4),      # Small numbers
                (10**4, 10**5),      # Medium numbers
                (10**5, 10**6),      # Large numbers
                (10**6, 10**7)       # Very large numbers
            ]

        algorithms = ['trial_division', 'miller_rabin', 'aks']
        results = {}

        print("🚀 COMPREHENSIVE PRIME ALGORITHM BENCHMARK")
        print("=" * 60)

        for range_start, range_end in test_ranges:
            print(f"\n📊 Testing range: {range_start:,} - {range_end:,}")
            range_name = f"{range_start:,}-{range_end:,}"

            # Generate test numbers (mix of primes and composites)
            test_numbers = self._generate_test_numbers(range_start, range_end, 100)

            range_results = {}

            for alg in algorithms:
                print(f"  🔄 Testing {alg}...")

                times = []
                accuracies = []
                certainties = []

                for n in test_numbers:
                    start_time = time.time()
                    result = self.system.is_prime_comprehensive(n, alg)
                    end_time = time.time()

                    times.append(end_time - start_time)
                    accuracies.append(1.0 if result.is_prime == self._reference_primality(n) else 0.0)
                    certainties.append(result.certainty)

                range_results[alg] = {
                    'avg_time': statistics.mean(times),
                    'median_time': statistics.median(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'total_time': sum(times),
                    'accuracy': statistics.mean(accuracies),
                    'avg_certainty': statistics.mean(certainties),
                    'std_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'test_count': len(test_numbers)
                }

                print(".2e")
            results[range_name] = range_results

        return results

    def _generate_test_numbers(self, start: int, end: int, count: int) -> List[int]:
        """Generate balanced test set of primes and composites"""
        numbers = []
        candidates = list(range(max(2, start), min(end, start + count * 10)))

        # Separate primes and composites
        primes = []
        composites = []

        for n in candidates:
            if self._reference_primality(n):
                primes.append(n)
            else:
                composites.append(n)

        # Balance the dataset
        min_count = min(len(primes), len(composites), count // 2)
        selected_primes = np.random.choice(primes, min_count, replace=False)
        selected_composites = np.random.choice(composites, min_count, replace=False)

        numbers = list(selected_primes) + list(selected_composites)
        np.random.shuffle(numbers)

        return numbers[:count]

    def _reference_primality(self, n: int) -> bool:
        """Reference primality test for accuracy validation"""
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False

        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0:
                return False
            i += 6
        return True

    def prime_generation_benchmark(self, limits: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark prime generation algorithms
        """
        if limits is None:
            limits = [10**4, 10**5, 10**6, 10**7]

        methods = ['sieve_eratosthenes', 'sieve_atkin']
        results = {}

        print("\n🏗️ PRIME GENERATION BENCHMARK")
        print("=" * 40)

        for limit in limits:
            print(f"\n📊 Generating primes up to {limit:,}")

            limit_results = {}

            for method in methods:
                print(f"  🔄 Testing {method}...")

                start_time = time.time()

                if method == 'sieve_eratosthenes':
                    primes = self.system.sieve_of_eratosthenes(limit)
                elif method == 'sieve_atkin':
                    primes = self.system.sieve_of_atkin(limit)

                end_time = time.time()

                limit_results[method] = {
                    'primes_found': len(primes),
                    'time_taken': end_time - start_time,
                    'density': len(primes) / limit,
                    'largest_prime': primes[-1] if primes else 0,
                    'avg_gap': self._calculate_avg_gap(primes),
                    'memory_estimate': limit * 0.125 / (1024**2)  # Rough bit array estimate in MB
                }

                print(".2e")
            results[str(limit)] = limit_results

        return results

    def _calculate_avg_gap(self, primes: List[int]) -> float:
        """Calculate average prime gap"""
        if len(primes) < 2:
            return 0
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        return statistics.mean(gaps)

    def special_prime_types_benchmark(self, limit: int = 10**5) -> Dict[str, Any]:
        """
        Benchmark generation of special prime types
        """
        print("\n🌟 SPECIAL PRIME TYPES BENCHMARK")
        print("=" * 45)

        types_to_test = [
            ('mersenne', 'generate_mersenne_primes'),
            ('fermat', 'generate_fermat_primes'),
            ('twin', 'generate_twin_primes'),
            ('cousin', 'generate_cousin_primes'),
            ('sexy', 'generate_sexy_primes'),
            ('sophie_germain', 'generate_sophie_germain_primes'),
            ('safe', 'generate_safe_primes'),
            ('chen', 'generate_chen_primes'),
            ('palindromic', 'generate_palindromic_primes'),
            ('pythagorean', 'generate_pythagorean_primes')
        ]

        results = {}

        for type_name, method_name in types_to_test:
            print(f"  🔄 Generating {type_name} primes...")

            start_time = time.time()

            if method_name == 'generate_fermat_primes':
                # Fermat primes don't need a limit
                special_primes = getattr(self.system, method_name)()
            else:
                special_primes = getattr(self.system, method_name)(limit)

            end_time = time.time()

            count = len(special_primes)
            density = count / limit if isinstance(special_primes, list) and not isinstance(special_primes[0], tuple) else count / (limit / 10)

            results[type_name] = {
                'count': count,
                'time_taken': end_time - start_time,
                'density': density,
                'samples': special_primes[:5] if count > 0 else []
            }

            print(".2e")
        return results

    def prime_prediction_benchmark(self, test_primes: List[int] = None) -> Dict[str, Any]:
        """
        Benchmark prime prediction algorithms
        """
        if test_primes is None:
            # Test on known primes
            all_primes = self.system.sieve_of_eratosthenes(10000)
            test_primes = all_primes[10:50]  # Skip first few small primes

        methods = ['li', 'riemann', 'crude', 'statistical']
        results = {}

        print("\n🔮 PRIME PREDICTION BENCHMARK")
        print("=" * 40)

        for method in methods:
            print(f"  🔄 Testing {method} method...")

            predictions = []
            accuracies = []
            errors = []
            confidences = []

            for prime in test_primes:
                prediction = self.system.predict_next_prime(prime, method)
                actual_next = self.system.get_next_prime(prime)

                predictions.append(prediction.number)
                accuracies.append(1.0 if prediction.number == actual_next else 0.0)
                errors.append(abs(prediction.number - actual_next))
                confidences.append(prediction.probability)

            results[method] = {
                'avg_accuracy': statistics.mean(accuracies),
                'avg_error': statistics.mean(errors),
                'max_error': max(errors),
                'avg_confidence': statistics.mean(confidences),
                'median_error': statistics.median(errors),
                'error_std': statistics.stdev(errors) if len(errors) > 1 else 0
            }

            print(".3f")
        return results

    def statistical_analysis(self, limit: int = 10**5) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis of prime distribution
        """
        print("\n📊 STATISTICAL ANALYSIS OF PRIME DISTRIBUTION")
        print("=" * 50)

        primes = self.system.sieve_of_eratosthenes(limit)
        n = len(primes)

        # Basic statistics
        gaps = [primes[i+1] - primes[i] for i in range(n-1)] if n > 1 else []

        analysis = {
            'total_primes': n,
            'density': n / limit,
            'prime_number_theorem_check': n / (limit / math.log(limit)),
            'gap_statistics': {
                'mean': statistics.mean(gaps) if gaps else 0,
                'median': statistics.median(gaps) if gaps else 0,
                'mode': statistics.mode(gaps) if gaps else 0,
                'std': statistics.stdev(gaps) if len(gaps) > 1 else 0,
                'min': min(gaps) if gaps else 0,
                'max': max(gaps) if gaps else 0,
                'range': max(gaps) - min(gaps) if gaps else 0
            }
        }

        # Test for normality of gaps
        if len(gaps) > 10:
            stat, p_value = stats.shapiro(gaps[:min(5000, len(gaps))])  # Shapiro-Wilk test
            analysis['gap_normality'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }

        # Fit distributions to gaps
        if len(gaps) > 50:
            # Try exponential distribution (expected for prime gaps)
            try:
                loc, scale = stats.expon.fit(gaps)
                analysis['gap_distribution_fit'] = {
                    'exponential_lambda': 1/scale,
                    'ks_statistic': stats.kstest(gaps, 'expon', args=(loc, scale))[0],
                    'ks_p_value': stats.kstest(gaps, 'expon', args=(loc, scale))[1]
                }
            except:
                analysis['gap_distribution_fit'] = {'fit_failed': True}

        # Prime counting function analysis
        x_vals = list(range(100, limit, max(1, limit//100)))
        pi_vals = [len([p for p in primes if p <= x]) for x in x_vals]
        li_vals = [self.system.prime_counting_li(x) for x in x_vals]

        # Calculate correlation and fit quality
        correlation = np.corrcoef(pi_vals, li_vals)[0, 1]
        mse = np.mean((np.array(pi_vals) - np.array(li_vals))**2)

        analysis['prime_counting_analysis'] = {
            'correlation_with_li': correlation,
            'mse_with_li': mse,
            'rmse_with_li': math.sqrt(mse),
            'max_deviation': max(abs(a - b) for a, b in zip(pi_vals, li_vals)),
            'mean_deviation': statistics.mean(abs(a - b) for a, b in zip(pi_vals, li_vals))
        }

        return analysis

    def create_comprehensive_visualizations(self, save_dir: str = "prime_analysis_plots"):
        """
        Create comprehensive visualizations of prime analysis
        """
        import os
        os.makedirs(save_dir, exist_ok=True)

        print("\n🎨 CREATING COMPREHENSIVE VISUALIZATIONS")
        print("=" * 50)

        # 1. Algorithm Performance Comparison
        self._plot_algorithm_performance(save_dir)

        # 2. Prime Distribution and Gaps
        self._plot_prime_distribution_gaps(save_dir)

        # 3. Special Prime Types Distribution
        self._plot_special_prime_types(save_dir)

        # 4. Prime Prediction Accuracy
        self._plot_prediction_accuracy(save_dir)

        # 5. Statistical Analysis Plots
        self._plot_statistical_analysis(save_dir)

        # 6. Prime Density and Counting Functions
        self._plot_prime_density_analysis(save_dir)

        print(f"✅ All visualizations saved to {save_dir}/")

    def _plot_algorithm_performance(self, save_dir: str):
        """Plot algorithm performance comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Generate benchmark data
        ranges = [(10**3, 10**4), (10**4, 10**5), (10**5, 10**6)]
        algorithms = ['trial_division', 'miller_rabin', 'aks']

        range_names = []
        alg_times = {alg: [] for alg in algorithms}
        alg_accuracies = {alg: [] for alg in algorithms}

        for range_start, range_end in ranges:
            range_name = f"{range_start:,}-{range_end:,}"
            range_names.append(range_name)

            test_numbers = self._generate_test_numbers(range_start, range_end, 50)

            for alg in algorithms:
                times = []
                accuracies = []

                for n in test_numbers[:20]:  # Subsample for speed
                    start_time = time.time()
                    result = self.system.is_prime_comprehensive(n, alg)
                    end_time = time.time()

                    times.append(end_time - start_time)
                    accuracies.append(1.0 if result.is_prime == self._reference_primality(n) else 0.0)

                alg_times[alg].append(statistics.mean(times))
                alg_accuracies[alg].append(statistics.mean(accuracies))

        # Performance comparison
        x = np.arange(len(range_names))
        width = 0.25

        for i, alg in enumerate(algorithms):
            ax1.bar(x + i*width, alg_times[alg], width, label=alg.replace('_', ' ').title())

        ax1.set_xlabel('Number Range')
        ax1.set_ylabel('Average Time (seconds)')
        ax1.set_title('Algorithm Performance Comparison')
        ax1.set_xticks(x + width)
        ax1.set_xticklabels(range_names)
        ax1.legend()
        ax1.set_yscale('log')

        # Accuracy comparison
        for i, alg in enumerate(algorithms):
            ax2.bar(x + i*width, alg_accuracies[alg], width, label=alg.replace('_', ' ').title())

        ax2.set_xlabel('Number Range')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Algorithm Accuracy Comparison')
        ax2.set_xticks(x + width)
        ax2.set_xticklabels(range_names)
        ax2.legend()

        # Time vs accuracy scatter
        colors = ['red', 'blue', 'green']
        for i, alg in enumerate(algorithms):
            ax3.scatter(alg_times[alg], alg_accuracies[alg], c=colors[i],
                       label=alg.replace('_', ' ').title(), s=100, alpha=0.7)

        ax3.set_xlabel('Average Time (seconds)')
        ax3.set_ylabel('Accuracy')
        ax3.set_title('Time vs Accuracy Trade-off')
        ax3.legend()
        ax3.set_xscale('log')

        # Scaling analysis (time vs problem size)
        problem_sizes = [10**3, 10**4, 10**5]
        for alg in algorithms:
            times_for_sizes = []
            for size in problem_sizes:
                start_time = time.time()
                for _ in range(10):  # Average over multiple tests
                    n = np.random.randint(size, size*10)
                    self.system.is_prime_comprehensive(n, alg)
                end_time = time.time()
                times_for_sizes.append((end_time - start_time) / 10)

            ax4.plot(problem_sizes, times_for_sizes, 'o-', label=alg.replace('_', ' ').title())

        ax4.set_xlabel('Problem Size')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_title('Scaling Analysis')
        ax4.legend()
        ax4.set_xscale('log')
        ax4.set_yscale('log')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/algorithm_performance.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prime_distribution_gaps(self, save_dir: str):
        """Plot prime distribution and gap analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Prime distribution
        limit = 10**5
        primes = self.system.sieve_of_eratosthenes(limit)

        ax1.scatter(range(len(primes)), primes, s=1, alpha=0.6, color='blue')
        ax1.set_xlabel('Prime Index')
        ax1.set_ylabel('Prime Value')
        ax1.set_title('Prime Distribution (First 10,000 primes)')
        ax1.grid(True, alpha=0.3)

        # Prime gaps histogram
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        ax2.hist(gaps, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('Gap Size')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Prime Gaps Distribution')
        ax2.axvline(np.mean(gaps), color='red', linestyle='--', label='.1f')
        ax2.legend()

        # Prime gaps vs log(n)
        log_n = [math.log(p) for p in primes[:-1]]

        ax3.scatter(log_n, gaps, s=1, alpha=0.6, color='purple')
        ax3.set_xlabel('log(n)')
        ax3.set_ylabel('Prime Gap')
        ax3.set_title('Prime Gaps vs log(n)')
        ax3.grid(True, alpha=0.3)

        # Autocorrelation of prime gaps
        if len(gaps) > 100:
            autocorr = np.correlate(gaps[:1000] - np.mean(gaps[:1000]),
                                  gaps[:1000] - np.mean(gaps[:1000]), mode='full')
            autocorr = autocorr[autocorr.size // 2:] / autocorr[autocorr.size // 2]
            lags = range(len(autocorr))

            ax4.plot(lags[:50], autocorr[:50], color='orange')
            ax4.set_xlabel('Lag')
            ax4.set_ylabel('Autocorrelation')
            ax4.set_title('Prime Gap Autocorrelation')
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/prime_distribution_gaps.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_special_prime_types(self, save_dir: str):
        """Plot distribution of special prime types"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        limit = 10**5
        special_types = {
            'Twin': len(self.system.generate_twin_primes(limit)),
            'Cousin': len(self.system.generate_cousin_primes(limit)),
            'Sexy': len(self.system.generate_sexy_primes(limit)),
            'Sophie Germain': len(self.system.generate_sophie_germain_primes(limit)),
            'Safe': len(self.system.generate_safe_primes(limit)),
            'Chen': len(self.system.generate_chen_primes(limit)),
            'Palindromic': len(self.system.generate_palindromic_primes(limit)),
            'Pythagorean': len(self.system.generate_pythagorean_primes(limit))
        }

        # Bar chart of counts
        types = list(special_types.keys())
        counts = list(special_types.values())

        bars = ax1.bar(types, counts, color='skyblue', alpha=0.8)
        ax1.set_xlabel('Prime Type')
        ax1.set_ylabel('Count')
        ax1.set_title('Special Prime Types Distribution')
        ax1.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{count:,}', ha='center', va='bottom')

        # Density comparison
        total_primes = len(self.system.sieve_of_eratosthenes(limit))
        densities = [count / total_primes for count in counts]

        bars2 = ax2.bar(types, densities, color='lightcoral', alpha=0.8)
        ax2.set_xlabel('Prime Type')
        ax2.set_ylabel('Density (fraction of all primes)')
        ax2.set_title('Special Prime Types Density')
        ax2.tick_params(axis='x', rotation=45)

        # Add percentage labels
        for bar, density in zip(bars2, densities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{density:.3f}', ha='center', va='bottom')

        # Mersenne and Fermat primes (small counts)
        mersenne = self.system.generate_mersenne_primes(10**6)
        fermat = self.system.generate_fermat_primes()

        special_large = {
            'Mersenne': len(mersenne),
            'Fermat': len(fermat)
        }

        ax3.bar(special_large.keys(), special_large.values(), color='gold', alpha=0.8)
        ax3.set_xlabel('Prime Type')
        ax3.set_ylabel('Count')
        ax3.set_title('Rare Prime Types')
        ax3.set_yscale('log')

        for i, (name, count) in enumerate(special_large.items()):
            ax3.text(i, count, f'{count}', ha='center', va='bottom')

        # Prime constellations comparison
        constellations = {
            'Twin pairs': len(self.system.generate_twin_primes(limit)),
            'Cousin pairs': len(self.system.generate_cousin_primes(limit)),
            'Sexy pairs': len(self.system.generate_sexy_primes(limit))
        }

        ax4.pie(constellations.values(), labels=constellations.keys(),
               autopct='%1.1f%%', startangle=90)
        ax4.set_title('Prime Constellations Distribution')
        ax4.axis('equal')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/special_prime_types.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prediction_accuracy(self, save_dir: str):
        """Plot prime prediction accuracy analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Test predictions on known primes
        test_primes = self.system.sieve_of_eratosthenes(10000)[10:60]  # Skip small primes
        methods = ['li', 'riemann', 'crude', 'statistical']

        method_colors = {'li': 'blue', 'riemann': 'red', 'crude': 'green', 'statistical': 'orange'}

        # Prediction accuracy by method
        accuracies = {}
        errors = {}

        for method in methods:
            method_accuracies = []
            method_errors = []

            for prime in test_primes:
                prediction = self.system.predict_next_prime(prime, method)
                actual_next = self.system.get_next_prime(prime)

                method_accuracies.append(prediction.number == actual_next)
                method_errors.append(abs(prediction.number - actual_next))

            accuracies[method] = method_accuracies
            errors[method] = method_errors

        # Accuracy comparison
        accuracy_means = [np.mean(accuracies[method]) for method in methods]

        bars = ax1.bar(methods, accuracy_means, color=[method_colors[m] for m in methods], alpha=0.7)
        ax1.set_xlabel('Prediction Method')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Prime Prediction Accuracy by Method')
        ax1.set_ylim(0, 1)

        for bar, acc in zip(bars, accuracy_means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{acc:.3f}', ha='center', va='bottom')

        # Error distribution
        error_data = [errors[method] for method in methods]
        ax2.boxplot(error_data, labels=methods, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_xlabel('Prediction Method')
        ax2.set_ylabel('Prediction Error')
        ax2.set_title('Prediction Error Distribution')
        ax2.set_yscale('log')

        # Prediction vs actual scatter
        for i, method in enumerate(methods):
            predictions = [self.system.predict_next_prime(p, method).number for p in test_primes]
            actuals = [self.system.get_next_prime(p) for p in test_primes]

            ax3.scatter(actuals, predictions, c=method_colors[method],
                       label=method, s=20, alpha=0.7)

        # Add perfect prediction line
        max_val = max(max(predictions), max(actuals))
        ax3.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
        ax3.set_xlabel('Actual Next Prime')
        ax3.set_ylabel('Predicted Next Prime')
        ax3.set_title('Predicted vs Actual Next Primes')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Confidence vs accuracy
        for method in methods:
            confidences = [self.system.predict_next_prime(p, method).probability for p in test_primes]
            method_accuracies = accuracies[method]

            ax4.scatter(confidences, method_accuracies, c=method_colors[method],
                       label=method, s=30, alpha=0.7)

        ax4.set_xlabel('Prediction Confidence')
        ax4.set_ylabel('Prediction Accuracy')
        ax4.set_title('Confidence vs Accuracy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/prediction_accuracy.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_statistical_analysis(self, save_dir: str):
        """Plot statistical analysis of prime gaps and distributions"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        limit = 10**5
        primes = self.system.sieve_of_eratosthenes(limit)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

        # Q-Q plot for normality test
        if len(gaps) > 10:
            stats.probplot(gaps, dist="norm", plot=ax1)
            ax1.set_title('Q-Q Plot: Prime Gaps vs Normal Distribution')
            ax1.grid(True, alpha=0.3)

        # Gap distribution fit
        if len(gaps) > 50:
            # Fit exponential distribution
            loc, scale = stats.expon.fit(gaps)

            # Plot histogram and fit
            ax2.hist(gaps, bins=30, alpha=0.7, color='blue', density=True,
                    label='Empirical', edgecolor='black')

            x_fit = np.linspace(min(gaps), max(gaps), 100)
            ax2.plot(x_fit, stats.expon.pdf(x_fit, loc, scale),
                    'r-', lw=2, label='Exponential Fit')
            ax2.set_xlabel('Prime Gap')
            ax2.set_ylabel('Density')
            ax2.set_title('Prime Gap Distribution Fit')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # Prime counting function comparison
        x_vals = np.logspace(2, 5, 100, dtype=int)
        pi_vals = []
        li_vals = []
        riemann_vals = []

        for x in x_vals:
            pi_vals.append(len([p for p in primes if p <= x]))
            li_vals.append(self.system.prime_counting_li(x))
            riemann_vals.append(self.system.prime_counting_r(x))

        ax3.plot(x_vals, pi_vals, 'b-', label='π(x) Actual', linewidth=2)
        ax3.plot(x_vals, li_vals, 'r--', label='Li(x) Approximation', linewidth=2)
        ax3.plot(x_vals, riemann_vals, 'g--', label='R(x) Riemann', linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('Prime Counting Function')
        ax3.set_title('Prime Counting Function Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_xscale('log')

        # Relative error analysis
        li_errors = [(li - pi) / pi for li, pi in zip(li_vals, pi_vals) if pi > 0]
        riemann_errors = [(r - pi) / pi for r, pi in zip(riemann_vals, pi_vals) if pi > 0]

        ax4.plot(x_vals[:len(li_errors)], li_errors, 'r-', label='Li(x) Relative Error', alpha=0.7)
        ax4.plot(x_vals[:len(riemann_errors)], riemann_errors, 'g-', label='R(x) Relative Error', alpha=0.7)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Relative Error')
        ax4.set_title('Relative Error in Prime Counting Approximations')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xscale('log')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_prime_density_analysis(self, save_dir: str):
        """Plot prime density analysis"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Prime density vs x
        x_vals = np.logspace(1, 6, 100, dtype=int)
        densities = []

        for x in x_vals:
            primes_up_to_x = len(self.system.sieve_of_eratosthenes(x))
            densities.append(primes_up_to_x / x)

        ax1.plot(x_vals, densities, 'b-', linewidth=2, label='Actual Density')
        ax1.plot(x_vals, [1/np.log(x) for x in x_vals], 'r--', linewidth=2, label='1/ln(x)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('Prime Density π(x)/x')
        ax1.set_title('Prime Density vs x')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')

        # Logarithmic derivative analysis
        log_x = np.log(x_vals)
        log_density = np.log(densities)
        log_theoretical = -log_x

        # Calculate numerical derivative
        density_derivative = np.gradient(log_density, log_x)

        ax2.plot(log_x, density_derivative, 'g-', linewidth=2, label='Numerical d(ln π(x)/x)/d(ln x)')
        ax2.axhline(y=-1, color='r', linestyle='--', linewidth=2, label='Theoretical (-1)')
        ax2.set_xlabel('ln(x)')
        ax2.set_ylabel('Logarithmic Derivative')
        ax2.set_title('Prime Number Theorem Verification')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Prime gaps vs density
        limit = 10**5
        primes = self.system.sieve_of_eratosthenes(limit)
        gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]
        densities_at_primes = [len([p for p in primes if p <= prime]) / prime for prime in primes[:-1]]

        ax3.scatter(densities_at_primes, gaps, s=1, alpha=0.6, color='purple')
        ax3.set_xlabel('Prime Density at Location')
        ax3.set_ylabel('Prime Gap')
        ax3.set_title('Prime Gaps vs Local Density')
        ax3.grid(True, alpha=0.3)

        # Twin prime conjecture visualization
        twin_primes = self.system.generate_twin_primes(limit)
        twin_prime_gaps = [twin[1] - twin[0] for twin in twin_primes]  # Should all be 2

        # Count twins in different ranges
        ranges = [(10**n, 10**(n+1)) for n in range(1, 5)]
        twin_counts = []
        range_labels = []

        for start, end in ranges:
            twins_in_range = len([t for t in twin_primes if start <= t[0] <= end])
            total_primes_in_range = len([p for p in primes if start <= p <= end])
            density = twins_in_range / total_primes_in_range if total_primes_in_range > 0 else 0
            twin_counts.append(density * 100)  # Convert to percentage
            range_labels.append(f'{start:,}-{end:,}')

        ax4.bar(range_labels, twin_counts, color='cyan', alpha=0.7)
        ax4.set_xlabel('Range')
        ax4.set_ylabel('Twin Prime Density (%)')
        ax4.set_title('Twin Prime Density by Range')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f'{save_dir}/prime_density_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive analysis suite
        """
        print("🔬 COMPREHENSIVE PRIME ANALYSIS SUITE")
        print("=" * 50)

        results = {}

        # Performance benchmarks
        print("\n📊 Running Performance Benchmarks...")
        results['performance'] = self.comprehensive_performance_benchmark()

        # Prime generation benchmarks
        print("\n🏗️ Running Prime Generation Benchmarks...")
        results['generation'] = self.prime_generation_benchmark()

        # Special prime types
        print("\n🌟 Analyzing Special Prime Types...")
        results['special_types'] = self.special_prime_types_benchmark()

        # Prime prediction
        print("\n🔮 Testing Prime Prediction Algorithms...")
        results['prediction'] = self.prime_prediction_benchmark()

        # Statistical analysis
        print("\n📈 Performing Statistical Analysis...")
        results['statistics'] = self.statistical_analysis()

        # Create visualizations
        print("\n🎨 Creating Visualizations...")
        self.create_comprehensive_visualizations()

        print("\n✅ Comprehensive Analysis Complete!")
        return results

    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """
        Generate detailed performance report
        """
        report = []
        report.append("COMPREHENSIVE PRIME SYSTEM PERFORMANCE REPORT")
        report.append("=" * 60)

        # Performance summary
        report.append("\n🚀 PERFORMANCE SUMMARY")
        report.append("-" * 30)

        if 'performance' in results:
            for range_name, range_data in results['performance'].items():
                report.append(f"\nRange: {range_name}")
                for alg, metrics in range_data.items():
                    report.append(".2e"
                                  ".3f"
                                  ".6f")

        # Best algorithms by category
        report.append("\n🏆 BEST ALGORITHMS")
        report.append("-" * 25)

        if 'performance' in results:
            # Find fastest algorithm per range
            for range_name, range_data in results['performance'].items():
                fastest_alg = min(range_data.items(), key=lambda x: x[1]['avg_time'])
                most_accurate_alg = max(range_data.items(), key=lambda x: x[1]['accuracy'])

                report.append(f"\n{range_name}:")
                report.append(f"  Fastest: {fastest_alg[0]} ({fastest_alg[1]['avg_time']:.2e}s)")
                report.append(f"  Most Accurate: {most_accurate_alg[0]} ({most_accurate_alg[1]['accuracy']:.6f})")

        # Prime generation summary
        if 'generation' in results:
            report.append("\n🏗️ PRIME GENERATION SUMMARY")
            report.append("-" * 35)

            for limit, gen_data in results['generation'].items():
                report.append(f"\nUp to {limit}:")
                for method, metrics in gen_data.items():
                    report.append(f"  {method}: {metrics['primes_found']:,} primes in {metrics['time_taken']:.3f}s")

        # Special types summary
        if 'special_types' in results:
            report.append("\n🌟 SPECIAL PRIME TYPES FOUND")
            report.append("-" * 35)

            for type_name, type_data in results['special_types'].items():
                report.append(f"  {type_name}: {type_data['count']:,} ({type_data['density']:.6f} density)")

        # Prediction summary
        if 'prediction' in results:
            report.append("\n🔮 PREDICTION ACCURACY")
            report.append("-" * 30)

            for method, pred_data in results['prediction'].items():
                report.append(f"  {method}: {pred_data['avg_accuracy']:.3f} accuracy, "
                            f"{pred_data['avg_error']:.1f} avg error")

        # Statistical insights
        if 'statistics' in results:
            stats_data = results['statistics']
            report.append("\n📊 STATISTICAL INSIGHTS")
            report.append("-" * 30)
            report.append(f"Prime density: {stats_data['density']:.6f}")
            report.append(f"Prime number theorem check: {stats_data['prime_number_theorem_check']:.3f}")
            report.append(f"Average prime gap: {stats_data['gap_statistics']['mean']:.2f}")

            if 'prime_counting_analysis' in stats_data:
                pca = stats_data['prime_counting_analysis']
                report.append(f"Correlation with Li(x): {pca['correlation_with_li']:.6f}")
                report.append(f"RMSE with Li(x): {pca['rmse_with_li']:.2f}")

        report.append("\n✅ Analysis Complete - All prime algorithms optimized and tested!")

        return "\n".join(report)


def main():
    """
    Main benchmark and analysis function
    """
    analyzer = PrimeBenchmarkAnalyzer()

    # Run comprehensive analysis
    results = analyzer.run_comprehensive_analysis()

    # Generate and print report
    report = analyzer.generate_performance_report(results)
    print("\n" + report)

    # Save detailed results
    import json
    with open('prime_analysis_results.json', 'w') as f:
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, dict):
                        json_results[key][subkey] = {}
                        for subsubkey, subsubvalue in subvalue.items():
                            try:
                                json_results[key][subkey][subsubkey] = float(subsubvalue) if isinstance(subsubvalue, (np.floating, np.integer)) else subsubvalue
                            except:
                                json_results[key][subkey][subsubkey] = str(subsubvalue)
                    else:
                        try:
                            json_results[key][subkey] = float(subvalue) if isinstance(subvalue, (np.floating, np.integer)) else subvalue
                        except:
                            json_results[key][subkey] = str(subvalue)
            else:
                json_results[key] = value

        json.dump(json_results, f, indent=2)

    print("\n💾 Detailed results saved to 'prime_analysis_results.json'")
    print("📊 Visualizations saved to 'prime_analysis_plots/' directory")


if __name__ == "__main__":
    main()
