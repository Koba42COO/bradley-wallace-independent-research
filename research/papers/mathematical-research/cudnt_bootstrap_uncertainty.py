"""
CUDNT Bootstrap Uncertainty Quantification
Phase 5: Bootstrap validation and uncertainty quantification
"""

import sys
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cudnt_prime_gap_predictor import CUDNT_PrimeGapPredictor

class CUDNT_BootstrapUncertainty:
    """
    Bootstrap-based uncertainty quantification for prime gap predictions
    """

    def __init__(self, n_bootstrap=50):
        self.n_bootstrap = n_bootstrap
        self.bootstrap_models = []
        self.scaler = RobustScaler()
        self.base_predictor = CUDNT_PrimeGapPredictor()

        print("🔄 CUDNT Bootstrap Uncertainty Quantification")
        print(f"   Bootstrap samples: {n_bootstrap}")
        print("   Providing prediction confidence intervals")
        print()

    def create_bootstrap_samples(self, features, targets):
        """Create bootstrap samples for uncertainty estimation"""
        print("📊 Creating bootstrap samples...")

        bootstrap_samples = []
        n_samples = len(features)

        for i in range(self.n_bootstrap):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_features = features[indices]
            bootstrap_targets = targets[indices]

            bootstrap_samples.append((bootstrap_features, bootstrap_targets))

        print(f"   Generated {len(bootstrap_samples)} bootstrap samples")
        return bootstrap_samples

    def train_bootstrap_models(self, features, targets):
        """Train models on bootstrap samples"""
        print("🎯 Training bootstrap models...")

        bootstrap_samples = self.create_bootstrap_samples(features, targets)

        for i, (boot_features, boot_targets) in enumerate(bootstrap_samples):
            # Scale features
            boot_features_scaled = self.scaler.fit_transform(boot_features)

            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i, n_jobs=-1)
            model.fit(boot_features_scaled, boot_targets)

            self.bootstrap_models.append(model)

            if (i + 1) % 10 == 0:
                print(f"   Trained {i+1}/{self.n_bootstrap} models")

        print(f"   Bootstrap ensemble ready with {len(self.bootstrap_models)} models")
        print()

    def predict_with_uncertainty(self, input_features, confidence_level=0.95):
        """Make predictions with uncertainty quantification"""
        if not self.bootstrap_models:
            raise ValueError("Bootstrap models not trained")

        # Scale input features
        input_scaled = self.scaler.transform(input_features)

        # Get predictions from all bootstrap models
        predictions = []
        for model in self.bootstrap_models:
            pred = model.predict(input_scaled)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Calculate statistics
        mean_prediction = np.mean(predictions, axis=0)
        std_prediction = np.std(predictions, axis=0)

        # Confidence intervals
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = mean_prediction - z_score * std_prediction
        ci_upper = mean_prediction + z_score * std_prediction

        # Prediction intervals (wider than confidence intervals)
        prediction_interval = z_score * std_prediction * np.sqrt(1 + 1/len(self.bootstrap_models))

        return {
            'mean': mean_prediction,
            'std': std_prediction,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'prediction_interval': prediction_interval,
            'all_predictions': predictions
        }

    def evaluate_uncertainty_quality(self, test_features, test_targets):
        """Evaluate the quality of uncertainty estimates"""
        print("📊 Evaluating uncertainty quantification...")

        uncertainty_results = self.predict_with_uncertainty(test_features)

        # Coverage analysis
        actuals = test_targets
        means = uncertainty_results['mean']
        ci_lower = uncertainty_results['ci_lower']
        ci_upper = uncertainty_results['ci_upper']

        # Calculate coverage
        coverage = np.mean((actuals >= ci_lower) & (actuals <= ci_upper))
        mean_interval_width = np.mean(ci_upper - ci_lower)

        # Error analysis within confidence intervals
        errors = np.abs(actuals - means)
        avg_error_in_ci = np.mean(errors[(actuals >= ci_lower) & (actuals <= ci_upper)])
        avg_error_out_ci = np.mean(errors[(actuals < ci_lower) | (actuals > ci_upper)])

        print(f"   Confidence interval coverage: {coverage:.3f} (target: 0.95)")
        print(f"   Mean interval width: {mean_interval_width:.3f}")
        print(f"   Average error in CI: {avg_error_in_ci:.3f}")
        print(f"   Average error outside CI: {avg_error_out_ci:.3f}")
        print()

        return {
            'coverage': coverage,
            'interval_width': mean_interval_width,
            'error_in_ci': avg_error_in_ci,
            'error_out_ci': avg_error_out_ci
        }

def run_bootstrap_uncertainty_demo():
    """Demonstrate bootstrap uncertainty quantification"""
    print("🔄 CUDNT BOOTSTRAP UNCERTAINTY DEMO")
    print("=" * 40)

    # Initialize uncertainty system
    uncertainty_system = CUDNT_BootstrapUncertainty(n_bootstrap=20)  # Smaller for demo

    # Generate training data
    print("📚 Generating training data...")
    features, targets = uncertainty_system.base_predictor.generate_training_data(5000)

    # Split data
    train_features, test_features, train_targets, test_targets = train_test_split(
        features, targets, test_size=0.3, random_state=42
    )

    # Train bootstrap models
    print("🎯 Training bootstrap ensemble...")
    uncertainty_system.train_bootstrap_models(train_features, train_targets)

    # Test uncertainty quantification
    print("🧪 Testing uncertainty quantification...")
    uncertainty_metrics = uncertainty_system.evaluate_uncertainty_quality(
        test_features[:100], test_targets[:100]  # Test on subset
    )

    # Make sample predictions with uncertainty
    sample_predictions = uncertainty_system.predict_with_uncertainty(test_features[:5])

    print("🎯 UNCERTAINTY DEMO RESULTS:")
    print(f"   Sample prediction 1: {sample_predictions['mean'][0]:.1f}")
    print(f"   95% CI: [{sample_predictions['ci_lower'][0]:.1f}, {sample_predictions['ci_upper'][0]:.1f}]")
    print(f"   Standard deviation: {sample_predictions['std'][0]:.3f}")
    print()

    print("💡 Uncertainty Quantification Benefits:")
    print("  • Provides confidence intervals for predictions")
    print("  • Quantifies prediction reliability")
    print("  • Enables risk-aware decision making")
    print("  • Improves model interpretability")
    print("  • Could capture 5-10% of missing accuracy through better calibration")

    print("
🎼 Mathematical Impact:"    print("  • Addresses uncertainty in prime gap predictions")
    print("  • Enables probabilistic forecasting")
    print("  • Improves model trustworthiness")
    print("  • Foundation for Bayesian approaches")

    print("
🚀 Uncertainty quantification enhances prediction reliability!"    print()

if __name__ == "__main__":
    run_bootstrap_uncertainty_demo()
