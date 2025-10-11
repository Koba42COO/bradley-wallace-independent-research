"""
CUDNT Accuracy Boost: Capturing the Missing 60%
Enhanced ensemble with temporal features and advanced ML techniques
"""

import sys
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cudnt_prime_gap_predictor import CUDNT_PrimeGapPredictor

class CUDNT_AccuracyBoost:
    """
    Accuracy boost system targeting the missing 60% through advanced techniques
    """

    def __init__(self, target_primes=1000000):
        self.target_primes = target_primes
        self.base_predictor = CUDNT_PrimeGapPredictor(target_primes=target_primes)

        # Enhanced components
        self.temporal_scaler = RobustScaler()
        self.models = {}
        self.ensemble_weights = {}

        print("🚀 CUDNT Accuracy Boost System Initialized")
        print("   Targeting the missing 60% accuracy gap")
        print("   Techniques: Temporal features + Ensemble + Advanced ML")

    def create_temporal_features(self, gaps, window_size=20):
        """Create temporal features to capture sequential dependencies"""
        features = []

        for i in range(window_size, len(gaps)):
            window = gaps[i-window_size:i]

            # Basic temporal features
            feat_dict = {
                # Statistical moments
                'mean': np.mean(window),
                'std': np.std(window),
                'skew': stats.skew(window),
                'kurt': stats.kurtosis(window),

                # Trend features
                'trend': np.polyfit(range(window_size), window, 1)[0],
                'accel': np.polyfit(range(window_size), window, 2)[0] if window_size >= 3 else 0,

                # Autocorrelation
                'autocorr_1': np.corrcoef(window[:-1], window[1:])[0,1],
                'autocorr_2': np.corrcoef(window[:-2], window[2:])[0,1] if window_size > 2 else 0,

                # Recent patterns
                'last_gap': window[-1],
                'second_last': window[-2],
                'third_last': window[-3] if window_size >= 3 else window[-1],

                # Volatility measures
                'range': np.max(window) - np.min(window),
                'iqr': np.percentile(window, 75) - np.percentile(window, 25),

                # Position features
                'position': i / len(gaps),
                'log_position': np.log(i + 1),
            }

            # Pattern recognition
            if window_size >= 3:
                # Increasing/decreasing patterns
                feat_dict['pattern_up'] = 1 if window[-1] > window[-2] > window[-3] else 0
                feat_dict['pattern_down'] = 1 if window[-1] < window[-2] < window[-3] else 0
                feat_dict['pattern_volatile'] = 1 if np.std(window[-3:]) > np.std(window[:-3]) else 0

            features.append(list(feat_dict.values()))

        return np.array(features)

    def train_boosted_ensemble(self, gaps, n_models=3):
        """Train boosted ensemble with temporal features"""
        print("Training boosted ensemble with temporal features...")

        # Create temporal features
        X = self.create_temporal_features(gaps)
        y = gaps[len(gaps) - len(X):]  # Align targets

        # Scale features
        X_scaled = self.temporal_scaler.fit_transform(X)

        # Define models
        base_models = [
            ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ]

        # Train models and calculate weights
        model_scores = {}

        for name, model in base_models:
            print(f"  Training {name}...")

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                      scoring='neg_mean_absolute_error')

            mean_score = -cv_scores.mean()
            model_scores[name] = mean_score

            # Train final model
            model.fit(X_scaled, y)
            self.models[name] = model

            print(f"    CV MAE: {mean_score:.3f}")

        # Calculate ensemble weights (inverse of MAE)
        total_weight = sum(1.0 / score for score in model_scores.values())
        self.ensemble_weights = {name: (1.0 / score) / total_weight
                               for name, score in model_scores.items()}

        print("Ensemble weights:", self.ensemble_weights)
        return self.models

    def predict_boosted(self, recent_gaps, num_predictions=5):
        """Make boosted predictions"""
        predictions = []

        current_sequence = np.array(recent_gaps.copy(), dtype=float)

        for _ in range(num_predictions):
            # Extract features
            features = self.create_temporal_features(current_sequence)

            if len(features) == 0:
                break

            # Scale features
            features_scaled = self.temporal_scaler.transform(features)

            # Get predictions from all models
            model_predictions = []
            for name, model in self.models.items():
                pred = model.predict(features_scaled[-1:])[0]
                model_predictions.append(pred)

            # Ensemble prediction
            weights = [self.ensemble_weights.get(name, 1.0) for name in self.models.keys()]
            final_pred = np.average(model_predictions, weights=weights)

            # Constrain and round
            final_pred = max(1, min(50, int(np.round(final_pred))))
            predictions.append(final_pred)

            # Update sequence
            current_sequence = np.append(current_sequence, final_pred)

        return predictions

    def benchmark_accuracy_gain(self, test_gaps):
        """Benchmark accuracy gain vs baseline"""
        print("Benchmarking accuracy improvements...")

        # Generate predictions
        test_predictions = []
        for i in range(20, len(test_gaps) - 5):
            recent_seq = test_gaps[i-20:i]
            pred = self.predict_boosted(recent_seq, num_predictions=5)
            test_predictions.extend(pred)

        # Align with actual
        actual = test_gaps[20:20+len(test_predictions)]

        boosted_mae = mean_absolute_error(actual, test_predictions)
        boosted_accuracy = 100 * (1 - boosted_mae / np.mean(actual))

        # Baseline comparison
        baseline_predictions = []
        for i in range(20, len(test_gaps) - 5):
            recent_seq = test_gaps[i-20:i]
            pred = self.base_predictor.predict_next_gaps(recent_seq, num_predictions=5)
            baseline_predictions.extend(pred)

        baseline_actual = test_gaps[20:20+len(baseline_predictions)]
        baseline_mae = mean_absolute_error(baseline_actual, baseline_predictions)
        baseline_accuracy = 100 * (1 - baseline_mae / np.mean(baseline_actual))

        improvement = boosted_accuracy - baseline_accuracy

        print("\nACCURACY BOOST RESULTS:")
        print(f"Boosted MAE: {boosted_mae:.3f}")
        print(f"Boosted Accuracy: {boosted_accuracy:.1f}%")
        print(f"Baseline MAE: {baseline_mae:.3f}")
        print(f"Baseline Accuracy: {baseline_accuracy:.1f}%")
        print(f"Improvement: {improvement:.1f}%")
        return boosted_accuracy, baseline_accuracy, improvement

def run_accuracy_boost_demo():
    """Demonstrate accuracy boost system"""
    print("🚀 CUDNT ACCURACY BOOST: Capturing the Missing 60%")
    print("=" * 55)

    # Initialize boost system
    booster = CUDNT_AccuracyBoost()

    # Generate training data
    print("📚 Generating training data...")
    features, gaps = booster.base_predictor.generate_training_data(15000)

    # Train boosted ensemble
    print("🎯 Training boosted ensemble...")
    booster.train_boosted_ensemble(gaps[:12000])

    # Test on held-out data
    print("🧪 Testing accuracy improvements...")
    test_gaps = gaps[12000:13000]  # 1000 test gaps
    boosted_acc, baseline_acc, improvement = booster.benchmark_accuracy_gain(test_gaps)

    # Results summary
    print("\n🎯 ACCURACY BOOST SUMMARY")
    print("=" * 30)

    print(f"Baseline: {baseline_acc:.1f}%")
    print(f"Boosted: {boosted_acc:.1f}%")
    print(f"Improvement: {improvement:.1f}%")
    if improvement > 0:
        captured_percent = (improvement / 60) * 100
        print(f"   ✅ Captured {captured_percent:.1f}% of missing 60%")
    else:
        print("   ⚠️ Further optimization needed")

    print("\n💡 Key Improvements:")
    print("  • Temporal feature engineering")
    print("  • Ensemble model weighting")
    print("  • Sequential dependency capture")
    print("  • Advanced statistical features")

    print("\n🎼 Impact on Missing 60%:")
    print("  • Addresses sequential dependencies")
    print("  • Models temporal patterns")
    print("  • Captures non-linear relationships")
    print("  • Improves ensemble stability")

    print("\n🚀 Progress toward 100% accuracy!")
    print()

if __name__ == "__main__":
    run_accuracy_boost_demo()
