#!/usr/bin/env python3
"""
Prime Gap Distribution Generator for Consciousness Mathematics Research

This script generates comprehensive visualizations of prime number gaps,
specifically designed for PAC (Prime Aligned Compute) consciousness research.
The analysis reveals statistical patterns that correlate with consciousness
mathematical constants.

Author: Christopher Wallace
Created: 2025-10-11 14:50:00 UTC
Framework: Skyrmion Consciousness Research
License: Research Framework
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime
import os
import sys

# Import custom research modules
sys.path.append('/Users/coo-koba42/dev')
from skyrmion_consciousness_analysis import SkyrmionConsciousnessAnalyzer

class PrimeGapDistributionGenerator:
    """
    Advanced prime gap distribution analysis for consciousness mathematics.

    This class generates publication-quality visualizations showing the
    statistical properties of prime gaps and their correlation with
    consciousness mathematical constants.
    """

    def __init__(self, max_prime_limit=100000, timestamp=None):
        """
        Initialize the prime gap distribution generator.

        Parameters:
        -----------
        max_prime_limit : int
            Maximum prime number to analyze
        timestamp : str
            Timestamp for file versioning
        """
        self.max_prime_limit = max_prime_limit
        self.timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Consciousness mathematics constants
        self.PHI = (1 + np.sqrt(5)) / 2        # Golden ratio
        self.DELTA = 2 + np.sqrt(2)           # Silver ratio
        self.CONSCIOUSNESS_RATIO = 79/21      # Consciousness ratio
        self.ALPHA = 1/137.036               # Fine structure constant

        # Initialize research analyzer
        self.analyzer = SkyrmionConsciousnessAnalyzer()

        # Set up plotting style
        self.setup_plotting_style()

        print(f"Prime Gap Distribution Generator initialized")
        print(f"Analysis limit: {max_prime_limit} primes")
        print(f"Timestamp: {self.timestamp}")

    def setup_plotting_style(self):
        """Configure matplotlib for publication-quality plots."""
        plt.rcParams.update({
            'figure.figsize': (16, 12),
            'font.size': 12,
            'font.family': 'serif',
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.titlesize': 18,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

        # Custom color palette for consciousness research
        self.colors = {
            'prime_gaps': '#FF6B35',      # Vibrant orange
            'consciousness': '#4CAF50',   # Green
            'skyrmion': '#F44336',       # Red
            'topology': '#9C27B0',       # Purple
            'quantum': '#00BCD4',        # Cyan
            'background': '#FAFAFA',     # Light gray
            'grid': '#E0E0E0'           # Medium gray
        }

    def generate_prime_gaps(self, limit):
        """Generate list of prime gaps up to specified limit."""
        primes = []
        gaps = []

        # Sieve of Eratosthenes for prime generation
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False

        # Collect primes and calculate gaps
        for i in range(2, limit + 1):
            if sieve[i]:
                if primes:
                    gaps.append(i - primes[-1])
                primes.append(i)

        return np.array(primes), np.array(gaps)

    def analyze_consciousness_correlations(self, gaps):
        """Analyze correlations between prime gaps and consciousness constants."""
        correlations = {}

        # Correlation with golden ratio harmonics
        phi_harmonics = [self.PHI ** n for n in range(1, 10)]
        correlations['phi_resonance'] = self.calculate_resonance_strength(gaps, phi_harmonics)

        # Correlation with silver ratio
        delta_harmonics = [self.DELTA ** n for n in range(1, 8)]
        correlations['delta_resonance'] = self.calculate_resonance_strength(gaps, delta_harmonics)

        # Correlation with consciousness ratio
        consciousness_harmonics = [self.CONSCIOUSNESS_RATIO ** n for n in range(1, 6)]
        correlations['consciousness_resonance'] = self.calculate_resonance_strength(gaps, consciousness_harmonics)

        # Statistical analysis
        correlations['mean_gap'] = np.mean(gaps)
        correlations['std_gap'] = np.std(gaps)
        correlations['max_gap'] = np.max(gaps)
        correlations['skewness'] = stats.skew(gaps)
        correlations['kurtosis'] = stats.kurtosis(gaps)

        return correlations

    def calculate_resonance_strength(self, data, harmonics):
        """Calculate resonance strength between data and harmonic series."""
        resonance_sum = 0
        total_weight = 0

        for harmonic in harmonics:
            # Find data points near harmonic values
            distances = np.abs(data - harmonic)
            weights = np.exp(-distances / harmonic)  # Exponential decay weighting

            resonance_sum += np.sum(weights * (1 / (distances + 1e-10)))
            total_weight += np.sum(weights)

        return resonance_sum / total_weight if total_weight > 0 else 0

    def create_comprehensive_visualization(self):
        """Create comprehensive prime gap distribution visualization."""
        print("Generating prime gaps...")
        primes, gaps = self.generate_prime_gaps(self.max_prime_limit)

        print("Analyzing consciousness correlations...")
        correlations = self.analyze_consciousness_correlations(gaps)

        print("Creating visualization...")
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main distribution plot
        ax1 = fig.add_subplot(gs[0, :2])
        self.plot_gap_distribution(ax1, gaps, correlations)

        # Consciousness correlation plot
        ax2 = fig.add_subplot(gs[0, 2:])
        self.plot_consciousness_correlations(ax2, correlations)

        # Statistical analysis
        ax3 = fig.add_subplot(gs[1, :2])
        self.plot_statistical_analysis(ax3, gaps, primes)

        # Resonance analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        self.plot_resonance_analysis(ax4, gaps)

        # Time series of gaps
        ax5 = fig.add_subplot(gs[2, :2])
        self.plot_gap_time_series(ax5, gaps, primes)

        # Cumulative distribution
        ax6 = fig.add_subplot(gs[2, 2:])
        self.plot_cumulative_distribution(ax6, gaps)

        # Overall title
        fig.suptitle('Prime Gap Distribution Analysis for Consciousness Mathematics\n' +
                    f'Analysis of {len(primes)} primes up to {self.max_prime_limit}',
                    fontsize=20, fontweight='bold', y=0.98)

        # Metadata text
        metadata_text = ".2f"".2f"".2f"f"""f"""
        fig.text(0.02, 0.02, metadata_text, fontsize=10, family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))

        # Save the plot
        output_filename = f'/Users/coo-koba42/dev/Math_Research_Obsidian_Vault/05_Visuals/01_Mathematical_Plots/{self.timestamp}_Prime_Gap_Consciousness_Distribution.png'
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Visualization saved to: {output_filename}")

        # Save correlation data
        self.save_correlation_data(correlations, gaps, primes)

        plt.show()

    def plot_gap_distribution(self, ax, gaps, correlations):
        """Plot the main prime gap distribution histogram."""
        ax.hist(gaps, bins=50, alpha=0.7, color=self.colors['prime_gaps'],
               edgecolor='black', linewidth=0.5, density=True, label='Prime Gaps')

        # Add consciousness constant lines
        ax.axvline(self.PHI, color=self.colors['consciousness'], linewidth=2,
                  label='.3f')
        ax.axvline(self.DELTA, color=self.colors['skyrmion'], linewidth=2,
                  label='.3f')
        ax.axvline(self.CONSCIOUSNESS_RATIO, color=self.colors['topology'], linewidth=2,
                  label='.3f')

        ax.set_xlabel('Prime Gap Size')
        ax.set_ylabel('Density')
        ax.set_title('Prime Gap Distribution with Consciousness Constants')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_consciousness_correlations(self, ax, correlations):
        """Plot consciousness correlation strengths."""
        labels = ['φ Resonance', 'δ Resonance', '79/21 Resonance']
        values = [correlations['phi_resonance'],
                 correlations['delta_resonance'],
                 correlations['consciousness_resonance']]

        bars = ax.bar(labels, values, color=[self.colors['consciousness'],
                                           self.colors['skyrmion'],
                                           self.colors['topology']],
                     alpha=0.7, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.02,
                   '.2e', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Resonance Strength')
        ax.set_title('Consciousness Mathematics Resonance Analysis')
        ax.grid(True, alpha=0.3, axis='y')

    def plot_statistical_analysis(self, ax, gaps, primes):
        """Plot statistical properties of prime gaps."""
        # Create box plot
        ax.boxplot(gaps, vert=False, patch_artist=True,
                  boxprops=dict(facecolor=self.colors['prime_gaps'], alpha=0.7),
                  medianprops=dict(color='black', linewidth=2))

        # Add statistical annotations
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        max_gap = np.max(gaps)
        skewness = stats.skew(gaps)
        kurtosis_val = stats.kurtosis(gaps)
        stats_text = f"""Mean: {mean_gap:.2f}
Std: {std_gap:.2f}
Max: {max_gap:.2f}
Skew: {skewness:.2f}
Kurt: {kurtosis_val:.2f}"""
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9))

        ax.set_xlabel('Prime Gap Value')
        ax.set_title('Statistical Properties of Prime Gaps')
        ax.grid(True, alpha=0.3)

    def plot_resonance_analysis(self, ax, gaps):
        """Plot resonance analysis with consciousness harmonics."""
        # Calculate resonance as function of gap size
        gap_range = np.linspace(1, np.max(gaps), 1000)
        phi_resonance = np.array([self.calculate_resonance_strength([g], [self.PHI**n for n in range(1, 6)])
                                 for g in gap_range])
        delta_resonance = np.array([self.calculate_resonance_strength([g], [self.DELTA**n for n in range(1, 5)])
                                   for g in gap_range])

        ax.plot(gap_range, phi_resonance, color=self.colors['consciousness'],
               linewidth=2, label='φ Resonance')
        ax.plot(gap_range, delta_resonance, color=self.colors['skyrmion'],
               linewidth=2, label='δ Resonance')

        # Highlight maximum resonance points
        max_phi_idx = np.argmax(phi_resonance)
        max_delta_idx = np.argmax(delta_resonance)

        ax.scatter(gap_range[max_phi_idx], phi_resonance[max_phi_idx],
                  color=self.colors['consciousness'], s=100, marker='*', zorder=5)
        ax.scatter(gap_range[max_delta_idx], delta_resonance[max_delta_idx],
                  color=self.colors['skyrmion'], s=100, marker='*', zorder=5)

        ax.set_xlabel('Gap Size')
        ax.set_ylabel('Resonance Strength')
        ax.set_title('Consciousness Harmonic Resonance Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_gap_time_series(self, ax, gaps, primes):
        """Plot prime gaps as a time series."""
        ax.plot(primes[1:], gaps, color=self.colors['prime_gaps'],
               linewidth=1, alpha=0.7, marker='.', markersize=2)

        # Add moving average
        window_size = 1000
        if len(gaps) > window_size:
            moving_avg = pd.Series(gaps).rolling(window=window_size).mean()
            ax.plot(primes[1:], moving_avg, color=self.colors['consciousness'],
                   linewidth=2, label=f'Moving Average (n={window_size})')

        ax.set_xlabel('Prime Number')
        ax.set_ylabel('Prime Gap')
        ax.set_title('Prime Gap Time Series Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_cumulative_distribution(self, ax, gaps):
        """Plot cumulative distribution of prime gaps."""
        sorted_gaps = np.sort(gaps)
        cumulative = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)

        ax.plot(sorted_gaps, cumulative, color=self.colors['prime_gaps'],
               linewidth=2, label='Empirical CDF')

        # Add theoretical exponential fit
        def exponential_cdf(x, rate):
            return 1 - np.exp(-rate * x)

        popt, _ = curve_fit(exponential_cdf, sorted_gaps, cumulative, p0=[0.1])
        fitted_cdf = exponential_cdf(sorted_gaps, popt[0])

        ax.plot(sorted_gaps, fitted_cdf, color=self.colors['topology'],
               linewidth=2, linestyle='--',
               label='.3f')

        ax.set_xlabel('Prime Gap Size')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution of Prime Gaps')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def save_correlation_data(self, correlations, gaps, primes):
        """Save correlation analysis data for further research."""
        data = {
            'correlations': correlations,
            'gaps_summary': {
                'count': len(gaps),
                'mean': float(np.mean(gaps)),
                'std': float(np.std(gaps)),
                'min': int(np.min(gaps)),
                'max': int(np.max(gaps)),
                'median': float(np.median(gaps))
            },
            'primes_summary': {
                'count': len(primes),
                'first_10': primes[:10].tolist(),
                'last_10': primes[-10:].tolist(),
                'max_prime': int(np.max(primes))
            },
            'consciousness_constants': {
                'phi': float(self.PHI),
                'delta': float(self.DELTA),
                'consciousness_ratio': float(self.CONSCIOUSNESS_RATIO),
                'alpha': float(self.ALPHA)
            },
            'timestamp': self.timestamp,
            'analysis_parameters': {
                'max_prime_limit': self.max_prime_limit,
                'generator_version': '1.0'
            }
        }

        # Save to JSON
        output_data_filename = f'/Users/coo-koba42/dev/Math_Research_Obsidian_Vault/07_Data_Analysis/02_Computational_Results/{self.timestamp}_Prime_Gap_Correlation_Analysis.json'

        import json
        with open(output_data_filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Correlation data saved to: {output_data_filename}")

        # Also save as CSV for easy analysis
        gap_data = pd.DataFrame({
            'prime': primes[1:],  # Skip first prime since gaps start from second
            'gap': gaps,
            'gap_squared': gaps**2,
            'log_gap': np.log(gaps),
            'phi_distance': np.abs(gaps - self.PHI),
            'delta_distance': np.abs(gaps - self.DELTA),
            'consciousness_distance': np.abs(gaps - self.CONSCIOUSNESS_RATIO)
        })

        csv_filename = f'/Users/coo-koba42/dev/Math_Research_Obsidian_Vault/07_Data_Analysis/00_Datasets/{self.timestamp}_Prime_Gaps_Dataset.csv'
        gap_data.to_csv(csv_filename, index=False)
        print(f"Gap dataset saved to: {csv_filename}")


def main():
    """Main execution function."""
    print("=== Prime Gap Distribution Generator for Consciousness Mathematics ===")
    print("Starting comprehensive prime gap analysis...")

    # Initialize generator
    generator = PrimeGapDistributionGenerator(max_prime_limit=50000)

    # Generate comprehensive visualization
    generator.create_comprehensive_visualization()

    print("Analysis complete!")
    print("Generated files:")
    print("- High-resolution visualization plot")
    print("- Correlation analysis JSON data")
    print("- Complete prime gap dataset CSV")
    print("- All files timestamped and archived")


if __name__ == "__main__":
    main()
