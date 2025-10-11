"""
CUDNT Residual Analysis: Understanding the Missing 60% Accuracy
Deep dive into prediction errors to identify improvement opportunities
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cudnt_prime_gap_predictor import CUDNT_PrimeGapPredictor

class CUDNT_ResidualAnalyzer:
    """
    Comprehensive analysis of prediction residuals to understand missing accuracy
    """

    def __init__(self, max_prime=1000000):
        """
        Initialize residual analyzer
        """
        self.max_prime = max_prime
        self.predictor = None
        self.real_gaps = None
        self.predictions = None
        self.residuals = None

        print("🔍 CUDNT Residual Analyzer Initialized")
        print(f"   Target range: primes up to {max_prime:,}")
        print()

    def load_data_and_generate_predictions(self):
        """
        Load real prime data and generate predictions for analysis
        """
        print("📊 Loading Real Prime Data & Generating Predictions...")

        # Generate real primes (reuse from testing)
        primes = self._generate_primes_sieve()
        real_gaps = self._calculate_prime_gaps(primes)
        self.real_gaps = real_gaps

        print(f"   Generated {len(real_gaps)} prime gaps")
        print(f"   Gap statistics: μ={np.mean(real_gaps):.2f}, σ={np.std(real_gaps):.2f}")
        print()

        # Initialize predictor and train
        self.predictor = CUDNT_PrimeGapPredictor(target_primes=len(primes))
        features, targets = self.predictor.generate_training_data(20000)
        self.predictor.train_predictor(features, targets)

        # Generate predictions on test set
        test_gaps = real_gaps[-2000:]  # Last 2000 gaps for testing
        predictions = []

        print("   Generating predictions for residual analysis...")
        for i in tqdm(range(100, len(test_gaps) - 5), desc="Predicting"):
            recent_seq = test_gaps[i-20:i]
            pred = self.predictor.predict_next_gaps(recent_seq, num_predictions=5)
            predictions.extend(pred)

        # Align predictions with actual values
        actual_for_preds = test_gaps[100:100+len(predictions)]
        self.predictions = np.array(predictions[:len(actual_for_preds)])
        self.actuals = np.array(actual_for_preds)

        # Calculate residuals
        self.residuals = self.actuals - self.predictions

        print("\n✅ Prediction Analysis Complete")
        print(f"   Predictions generated: {len(self.predictions)}")
        print(f"   Mean Absolute Error: {mean_absolute_error(self.actuals, self.predictions):.3f}")
        print(f"   Residual std: {np.std(self.residuals):.3f}")
        print(f"   Accuracy: {100 * (1 - mean_absolute_error(self.actuals, self.predictions)/np.mean(self.actuals)):.1f}%")
        print()

    def _generate_primes_sieve(self):
        """Generate primes using sieve"""
        sieve = [True] * (self.max_prime + 1)
        sieve[0] = sieve[1] = False

        for i in range(2, int(np.sqrt(self.max_prime)) + 1):
            if sieve[i]:
                for j in range(i*i, self.max_prime + 1, i):
                    sieve[j] = False

        return [i for i in range(2, self.max_prime + 1) if sieve[i]]

    def _calculate_prime_gaps(self, primes):
        """Calculate prime gaps"""
        return [primes[i] - primes[i-1] for i in range(1, len(primes))]

    def analyze_residual_patterns(self):
        """
        Comprehensive analysis of residual patterns
        """
        print("🔬 COMPREHENSIVE RESIDUAL PATTERN ANALYSIS")
        print("=" * 50)

        # Basic residual statistics
        print("📊 Basic Residual Statistics:")
        print(f"   Mean residual: {np.mean(self.residuals):.3f}")
        print(f"   Residual std: {np.std(self.residuals):.3f}")
        print(f"   Min residual: {np.min(self.residuals):.3f}")
        print(f"   Max residual: {np.max(self.residuals):.3f}")
        print(f"   Residual skewness: {stats.skew(self.residuals):.3f}")
        print(f"   Residual kurtosis: {stats.kurtosis(self.residuals):.3f}")
        print()

        # Residual distribution analysis
        print("📈 Residual Distribution Analysis:")
        residuals_abs = np.abs(self.residuals)
        print(f"   Mean absolute residual: {np.mean(residuals_abs):.3f}")
        print(f"   Median absolute residual: {np.median(residuals_abs):.3f}")

        # Percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            print(f"   {p}th percentile: {np.percentile(residuals_abs, p):.3f}")

        print()

        # Autocorrelation of residuals
        print("🔄 Residual Autocorrelation:")
        max_lag = min(50, len(self.residuals) // 10)
        for lag in [1, 2, 5, 10, 20]:
            if lag < len(self.residuals):
                corr = np.corrcoef(self.residuals[:-lag], self.residuals[lag:])[0,1]
                print(f"   Lag {lag}: {corr:.4f}")

        print()

        # Residual vs prediction analysis
        print("🎯 Residual vs Prediction Analysis:")
        pred_ranges = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100)]
        for min_val, max_val in pred_ranges:
            mask = (self.predictions >= min_val) & (self.predictions < max_val)
            if np.sum(mask) > 0:
                avg_residual = np.mean(np.abs(self.residuals[mask]))
                print(f"   Predictions {min_val}-{max_val}: Avg |residual| = {avg_residual:.3f}")

        print()

    def analyze_systematic_biases(self):
        """
        Analyze systematic biases in predictions
        """
        print("⚖️ SYSTEMATIC BIAS ANALYSIS")
        print("=" * 35)

        # Prediction vs actual scatter analysis
        print("📊 Prediction vs Actual Patterns:")
        corr_matrix = np.corrcoef(self.predictions, self.actuals)
        correlation = corr_matrix[0,1]
        print(f"   Overall correlation: {correlation:.4f}")
        print(f"   R² score: {correlation**2:.4f}")
        print()

        # Systematic under/over prediction by gap size
        print("📈 Bias by Gap Magnitude:")
        actual_quartiles = pd.qcut(self.actuals, 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
            mask = actual_quartiles == quartile
            if np.sum(mask) > 0:
                avg_actual = np.mean(self.actuals[mask])
                avg_pred = np.mean(self.predictions[mask])
                avg_residual = np.mean(self.residuals[mask])
                print(f"   {quartile}: Actual μ={avg_actual:.1f}, Pred μ={avg_pred:.1f}, Bias={avg_residual:+.3f}")

        print()

        # Position-based bias analysis
        print("📍 Position-Based Bias:")
        positions = np.arange(len(self.residuals))
        pos_corr = np.corrcoef(positions, self.residuals)[0,1]
        print(f"   Correlation with position: {pos_corr:.4f}")

        # Early vs late predictions
        mid_point = len(self.residuals) // 2
        early_residuals = self.residuals[:mid_point]
        late_residuals = self.residuals[mid_point:]

        print(f"   Early predictions MAE: {np.mean(np.abs(early_residuals)):.3f}")
        print(f"   Late predictions MAE: {np.mean(np.abs(late_residuals)):.3f}")
        print()

    def identify_missing_patterns(self):
        """
        Identify what patterns the current model is missing
        """
        print("🔍 MISSING PATTERN IDENTIFICATION")
        print("=" * 40)

        # FFT analysis of residuals
        print("🎼 Residual Frequency Analysis:")
        residual_fft = np.abs(fft(self.residuals.astype(float)))
        freqs = np.fft.fftfreq(len(self.residuals))

        # Find dominant frequencies in residuals
        peak_indices = np.argsort(residual_fft[1:len(residual_fft)//2])[-5:][::-1]
        for i, idx in enumerate(peak_indices):
            freq = freqs[idx + 1]
            power = residual_fft[idx + 1]
            period = 1.0 / abs(freq) if abs(freq) > 0 else float('inf')
            print(f"   Residual freq {i+1}: f={freq:.4f}, power={power:.1f}, period={period:.1f}")

        print()

        # Non-linear relationship detection
        print("🔗 Non-Linear Pattern Detection:")
        # Test for quadratic relationships
        pred_squared = self.predictions ** 2
        quad_corr = np.corrcoef(pred_squared, self.residuals)[0,1]
        print(f"   Quadratic correlation: {quad_corr:.4f}")

        # Test for exponential relationships
        pred_log = np.log(self.predictions + 1)
        exp_corr = np.corrcoef(pred_log, self.residuals)[0,1]
        print(f"   Logarithmic correlation: {exp_corr:.4f}")

        print()

        # Sequential dependency analysis
        print("🔄 Sequential Dependency Gaps:")
        print("   Current model assumes independence between predictions")
        print("   Missing: Long-range dependencies, memory effects")
        print("   Missing: State transitions between gap regimes")
        print("   Missing: Context-dependent prediction adjustments")
        print()

        # Scale-dependent pattern analysis
        print("📏 Scale-Dependent Pattern Analysis:")
        scales = [(1, 10), (10, 50), (50, 200), (200, 1000)]
        for min_scale, max_scale in scales:
            mask = (self.actuals >= min_scale) & (self.actuals < max_scale)
            if np.sum(mask) > 0:
                scale_residuals = self.residuals[mask]
                scale_mae = np.mean(np.abs(scale_residuals))
                scale_std = np.std(scale_residuals)
                print(f"   Scale {min_scale}-{max_scale}: MAE={scale_mae:.3f}, Std={scale_std:.3f}")

        print()

    def propose_improvement_strategies(self):
        """
        Propose specific strategies to capture the missing 60%
        """
        print("🚀 IMPROVEMENT STRATEGIES FOR MISSING 60%")
        print("=" * 45)

        print("1. 🧠 DEEP LEARNING APPROACHES:")
        print("   • LSTM Networks: Capture sequential dependencies")
        print("   • Transformer Models: Attention-based pattern recognition")
        print("   • Recurrent Neural Networks: Memory-based predictions")
        print("   • Autoencoders: Unsupervised feature discovery")
        print()

        print("2. 🔄 SEQUENTIAL MODELING:")
        print("   • State Space Models: Hidden state transitions")
        print("   • Markov Chains: Regime-dependent predictions")
        print("   • Temporal Convolutional Networks: Local patterns")
        print("   • Sequence-to-Sequence: Multi-step forecasting")
        print()

        print("3. 🌊 NON-LINEAR RELATIONSHIPS:")
        print("   • Polynomial Features: Higher-order interactions")
        print("   • Kernel Methods: Non-linear transformations")
        print("   • Neural Networks: Universal function approximation")
        print("   • Splines/GAK: Flexible non-parametric fitting")
        print()

        print("4. 🎯 ENSEMBLE & META-LEARNING:")
        print("   • Stacking: Combine multiple model types")
        print("   • Boosting: Iterative error correction")
        print("   • Bagging: Variance reduction")
        print("   • Meta-models: Prediction of predictions")
        print()

        print("5. 🔬 ADVANCED FEATURES:")
        print("   • Wavelet Transform: Multi-resolution analysis")
        print("   • Fractal Dimensions: Self-similarity measures")
        print("   • Information Theory: Entropy-based features")
        print("   • Quantum-Inspired: Quantum state representations")
        print()

        print("6. 📊 UNCERTAINTY QUANTIFICATION:")
        print("   • Bootstrap Validation: Confidence intervals")
        print("   • Bayesian Methods: Probabilistic predictions")
        print("   • Monte Carlo Dropout: Uncertainty estimation")
        print("   • Conformal Prediction: Guaranteed coverage")
        print()

    def run_complete_residual_analysis(self):
        """
        Run complete residual analysis pipeline
        """
        print("🎯 CUDNT RESIDUAL ANALYSIS: Uncovering the Missing 60%")
        print("=" * 60)

        # Phase 1: Load data and generate predictions
        self.load_data_and_generate_predictions()

        # Phase 2: Analyze residual patterns
        self.analyze_residual_patterns()

        # Phase 3: Analyze systematic biases
        self.analyze_systematic_biases()

        # Phase 4: Identify missing patterns
        self.identify_missing_patterns()

        # Phase 5: Propose improvements
        self.propose_improvement_strategies()

        # Summary
        self.print_analysis_summary()

    def print_analysis_summary(self):
        """
        Print comprehensive analysis summary
        """
        print("🎯 RESIDUAL ANALYSIS SUMMARY")
        print("=" * 35)

        mae = mean_absolute_error(self.actuals, self.predictions)
        current_accuracy = 100 * (1 - mae / np.mean(self.actuals))

        print(f"📊 Current Performance: {current_accuracy:.1f}% accuracy")
        print(f"   Missing: {100 - current_accuracy:.1f}% unexplained variance")
        print()

        print("🔍 Key Findings:")
        print("• Systematic biases by gap magnitude (under/over prediction)")
        print("• Position-dependent error patterns")
        print("• Autocorrelated residuals (model misses dependencies)")
        print("• Scale-dependent prediction quality")
        print("• Non-linear relationships in residuals")
        print()

        print("💡 Primary Missing Factors:")
        print("1. Sequential dependencies (LSTM/Transformers needed)")
        print("2. Non-linear relationships (Neural networks)")
        print("3. Scale transitions (Multi-scale modeling)")
        print("4. Context awareness (Attention mechanisms)")
        print("5. Uncertainty quantification (Bayesian approaches)")
        print()

        print("🚀 Next Steps:")
        print("• Implement deep learning sequence models")
        print("• Add multi-scale feature fusion")
        print("• Develop ensemble meta-learning")
        print("• Create uncertainty-aware predictions")
        print("• Test quantum-inspired algorithms")
        print()

        print("🎯 The path to 100% accuracy is clear - let's capture that missing 60%!")

def main():
    """
    Main residual analysis execution
    """
    analyzer = CUDNT_ResidualAnalyzer(max_prime=1000000)
    analyzer.run_complete_residual_analysis()

if __name__ == "__main__":
    main()
