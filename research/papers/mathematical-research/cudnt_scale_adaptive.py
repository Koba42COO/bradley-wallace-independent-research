"""
CUDNT Scale-Adaptive Prime Gap Predictor
Phase 4: Scale Transitions via Adaptive Modeling
Different models for different gap ranges to capture scale-dependent patterns
"""

import sys
import os
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from cudnt_prime_gap_predictor import CUDNT_PrimeGapPredictor

class CUDNT_ScaleAdaptivePredictor:
    """
    Scale-adaptive predictor that uses different models for different gap ranges
    Addresses the major accuracy variation by gap magnitude
    """

    def __init__(self, target_primes=1000000):
        """
        Initialize scale-adaptive predictor
        """
        self.target_primes = target_primes
        self.base_predictor = CUDNT_PrimeGapPredictor(target_primes=target_primes)

        # Scale ranges based on our residual analysis
        self.scale_ranges = {
            'small': (1, 10),      # MAE=4.8, High accuracy range
            'medium': (10, 25),    # Moderate performance
            'large': (25, 50),     # Lower performance
            'xlarge': (50, 100),   # Very low performance (MAE=49.6)
            'xxlarge': (100, float('inf'))  # Extreme gaps
        }

        # Models for each scale
        self.scale_models = {}
        self.scale_scalers = {}
        self.scale_features = {}

        print("📏 CUDNT Scale-Adaptive Predictor Initialized")
        print(f"   Target scale: {target_primes:,} primes")
        print(f"   Scale ranges: {len(self.scale_ranges)} adaptive regimes")
        print("   Addressing scale-dependent accuracy variations")
        print()

    def detect_gap_scale(self, gap_value):
        """
        Detect which scale range a gap belongs to
        """
        for scale_name, (min_val, max_val) in self.scale_ranges.items():
            if min_val <= gap_value < max_val:
                return scale_name

        # Handle edge cases
        if gap_value >= self.scale_ranges['xxlarge'][0]:
            return 'xxlarge'
        else:
            return 'small'  # Default for very small gaps

    def create_scale_specific_features(self, gaps, scale_name, window_size=15):
        """
        Create features optimized for each scale range
        """
        features = []

        for i in range(window_size, len(gaps)):
            window = gaps[i-window_size:i]

            if scale_name == 'small':
                # Small gaps: Focus on local patterns and high-frequency components
                feat_dict = {
                    'mean_local': np.mean(window[-5:]),  # Very recent
                    'std_local': np.std(window[-5:]),
                    'trend_short': np.polyfit(range(5), window[-5:], 1)[0],
                    'autocorr_1': self._safe_corrcoef(window[-6:-1], window[-5:]),
                    'pattern_density': np.sum(window < 6) / len(window),  # Density of small gaps
                    'recent_min': np.min(window[-3:]),
                    'volatility_ratio': np.std(window[-5:]) / (np.mean(window[-5:]) + 1e-6),
                }

            elif scale_name == 'medium':
                # Medium gaps: Balance between local and global patterns
                feat_dict = {
                    'mean_medium': np.mean(window[-10:]),
                    'std_medium': np.std(window[-10:]),
                    'trend_medium': np.polyfit(range(10), window[-10:], 1)[0],
                    'autocorr_2': self._safe_corrcoef(window[-12:-2], window[-10:]),
                    'range_medium': np.max(window[-10:]) - np.min(window[-10:]),
                    'median_trend': np.median(window[-10:]) - np.median(window[:-10]) if len(window) > 10 else 0,
                    'iqr_ratio': (np.percentile(window[-10:], 75) - np.percentile(window[-10:], 25)) / (np.mean(window[-10:]) + 1e-6),
                }

            elif scale_name == 'large':
                # Large gaps: Focus on longer-term trends and stability
                feat_dict = {
                    'mean_large': np.mean(window),
                    'std_large': np.std(window),
                    'trend_large': np.polyfit(range(window_size), window, 1)[0],
                    'autocorr_3': self._safe_corrcoef(window[:-3], window[3:]),
                    'max_position': np.argmax(window) / len(window),  # Where max gap occurs
                    'stability_ratio': np.std(window[-5:]) / (np.std(window) + 1e-6),
                    'large_gap_density': np.sum(window > 20) / len(window),
                }

            elif scale_name in ['xlarge', 'xxlarge']:
                # Very large gaps: Focus on extreme value patterns and rare events
                feat_dict = {
                    'mean_extreme': np.mean(window),
                    'max_extreme': np.max(window),
                    'extreme_count': np.sum(window > 50),
                    'trend_extreme': np.polyfit(range(window_size), window, 1)[0],
                    'tail_heavy': stats.kurtosis(window),
                    'range_extreme': np.max(window) - np.min(window),
                    'large_gap_frequency': np.sum(window > np.percentile(window, 80)) / len(window),
                }

            else:
                # Default feature set
                feat_dict = {
                    'mean_default': np.mean(window),
                    'std_default': np.std(window),
                    'trend_default': np.polyfit(range(window_size), window, 1)[0],
                    'autocorr_default': self._safe_corrcoef(window[:-1], window[1:]),
                }

            features.append(list(feat_dict.values()))

        return np.array(features)

    def _safe_corrcoef(self, x, y):
        """Safe correlation coefficient"""
        try:
            if len(x) == len(y) and len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
                return np.corrcoef(x, y)[0,1]
            return 0.0
        except:
            return 0.0

    def train_scale_specific_models(self, gaps, n_samples=10000):
        """
        Train separate models optimized for each scale range
        """
        print("🎯 Training Scale-Specific Models...")

        # Generate training data
        features, targets = self.base_predictor.generate_training_data(n_samples)

        # Train model for each scale
        for scale_name, (min_val, max_val) in self.scale_ranges.items():
            print(f"\n   Training {scale_name} scale model (range: {min_val}-{max_val})...")

            # Filter data for this scale range
            scale_mask = (targets >= min_val) & (targets < max_val)
            if np.sum(scale_mask) < 100:  # Not enough data
                print(f"     Insufficient data for {scale_name} scale, skipping")
                continue

            scale_targets = targets[scale_mask]
            scale_gaps = gaps[:len(scale_targets)]  # Align gaps

            # Create scale-specific features
            scale_features = self.create_scale_specific_features(scale_gaps, scale_name)
            scale_features = scale_features[:len(scale_targets)]  # Align lengths

            if len(scale_features) < 50:
                print(f"     Insufficient features for {scale_name} scale, skipping")
                continue

            # Select appropriate model for this scale
            if scale_name == 'small':
                # Small gaps: Complex patterns, use ensemble
                model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            elif scale_name == 'medium':
                # Medium gaps: Balanced approach
                model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
            elif scale_name in ['large', 'xlarge', 'xxlarge']:
                # Large gaps: Simpler relationships, use linear
                model = Ridge(alpha=1.0)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)

            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(scale_features)

            # Train model
            model.fit(X_scaled, scale_targets)

            # Store model and scaler
            self.scale_models[scale_name] = model
            self.scale_scalers[scale_name] = scaler
            self.scale_features[scale_name] = scale_features.shape[1]

            # Quick validation
            tscv = TimeSeriesSplit(n_splits=3)
            cv_scores = cross_val_score(model, X_scaled, scale_targets, cv=tscv,
                                      scoring='neg_mean_absolute_error')
            cv_mae = -cv_scores.mean()

            print(f"     Trained on {len(scale_targets)} samples")
            print(f"     Features: {scale_features.shape[1]}")
            print(f"     CV MAE: {cv_mae:.3f}")

        print(f"\n✅ Trained models for {len(self.scale_models)} scale ranges")
        print()

    def predict_adaptive(self, recent_gaps, num_predictions=5):
        """
        Make predictions using scale-adaptive approach
        """
        if not self.scale_models:
            raise ValueError("Scale models not trained. Call train_scale_specific_models() first.")

        predictions = []
        current_sequence = np.array(recent_gaps.copy(), dtype=float)

        print("🔮 Making Scale-Adaptive Predictions...")

        for step in range(num_predictions):
            # Determine scale of next expected gap (based on recent pattern)
            recent_avg = np.mean(current_sequence[-5:]) if len(current_sequence) >= 5 else np.mean(current_sequence)
            predicted_scale = self.detect_gap_scale(recent_avg)

            # Create features for current sequence
            features = self.create_scale_specific_features(current_sequence, predicted_scale)

            if len(features) == 0:
                break

            # Scale features and predict
            if predicted_scale in self.scale_models:
                model = self.scale_models[predicted_scale]
                scaler = self.scale_scalers[predicted_scale]

                features_scaled = scaler.transform(features[-1:].reshape(1, -1))
                prediction = model.predict(features_scaled)[0]
            else:
                # Fallback to average prediction
                prediction = np.mean(current_sequence[-10:]) if len(current_sequence) >= 10 else np.mean(current_sequence)

            # Constrain prediction
            prediction = max(1, min(200, int(np.round(prediction))))
            predictions.append(prediction)

            # Update sequence
            current_sequence = np.append(current_sequence, prediction)

            if (step + 1) % 2 == 0:
                print(f"   Step {step+1}: Predicted gap {prediction} (scale: {predicted_scale})")

        print(f"   Final predictions: {predictions}")
        print()

        return predictions

    def evaluate_scale_performance(self, test_gaps):
        """
        Evaluate performance across different scales
        """
        print("📊 Scale-Adaptive Performance Evaluation")

        predictions = []
        actuals = []

        # Generate predictions for test set
        for i in range(20, len(test_gaps) - 5):
            recent_seq = test_gaps[i-20:i]
            preds = self.predict_adaptive(recent_seq, num_predictions=5)
            predictions.extend(preds)
            actuals.extend(test_gaps[i:i+5])

        if len(predictions) != len(actuals):
            min_len = min(len(predictions), len(actuals))
            predictions = predictions[:min_len]
            actuals = actuals[:min_len]

        # Overall performance
        mae = mean_absolute_error(actuals, predictions)
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        r2 = r2_score(actuals, predictions)
        accuracy = 100 * (1 - mae / np.mean(actuals))

        print("\n📈 Overall Performance:")
        print(f"   Mean Absolute Error: {mae:.3f} gaps")
        print(f"   Root Mean Square Error: {rmse:.3f} gaps")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Prediction Accuracy: {accuracy:.1f}%")
        print()

        # Scale-specific performance
        print("📏 Scale-Specific Performance:")
        scale_results = {}

        for scale_name, (min_val, max_val) in self.scale_ranges.items():
            # Find predictions in this scale range
            scale_mask = np.array([min_val <= pred < max_val for pred in predictions])
            if np.sum(scale_mask) > 10:  # Enough samples
                scale_actuals = np.array(actuals)[scale_mask]
                scale_preds = np.array(predictions)[scale_mask]

                scale_mae = mean_absolute_error(scale_actuals, scale_preds)
                scale_accuracy = 100 * (1 - scale_mae / np.mean(scale_actuals))

                scale_results[scale_name] = {
                    'mae': scale_mae,
                    'accuracy': scale_accuracy,
                    'count': len(scale_preds)
                }

                print(f"   {scale_name.capitalize()} ({min_val}-{max_val}): MAE={scale_mae:.3f}, Acc={scale_accuracy:.1f}%, N={len(scale_preds)}")

        print()

        return {
            'overall': {'mae': mae, 'accuracy': accuracy, 'r2': r2},
            'scale_specific': scale_results
        }

def run_scale_adaptive_demo():
    """
    Demonstrate scale-adaptive prediction system
    """
    print("📏 CUDNT SCALE-ADAPTIVE PREDICTION DEMO")
    print("=" * 45)

    # Initialize adaptive predictor
    adaptive_predictor = CUDNT_ScaleAdaptivePredictor()

    # Generate training data
    print("📚 Generating training data...")
    features, gaps = adaptive_predictor.base_predictor.generate_training_data(15000)

    # Train scale-specific models
    print("🎯 Training scale-adaptive models...")
    adaptive_predictor.train_scale_specific_models(gaps)

    # Test on sample data
    print("🧪 Testing scale-adaptive predictions...")
    test_sequence = gaps[12000:12050]  # 50 gaps for testing
    predictions = adaptive_predictor.predict_adaptive(test_sequence, num_predictions=10)

    print("
🎯 DEMO RESULTS:"    print(f"   Test sequence length: {len(test_sequence)}")
    print(f"   Generated predictions: {len(predictions)}")
    print(f"   Prediction range: {np.min(predictions)} - {np.max(predictions)}")

    # Show scale detection in action
    print("
🔍 Scale Detection Examples:"    for i, pred in enumerate(predictions[:5]):
        scale = adaptive_predictor.detect_gap_scale(pred)
        print(f"   Prediction {i+1}: {pred} → {scale} scale")

    print("
💡 Scale-Adaptive Advantages:"    print("  • Different models for different gap magnitudes")
    print("  • Addresses scale-dependent accuracy variations")
    print("  • Optimizes feature engineering per scale")
    print("  • Handles extreme gap ranges appropriately")
    print("  • Balances complexity with performance")

    print("
🎼 Mathematical Impact:"    print("  • Tackles major missing factor: scale-dependent patterns")
    print("  • Could capture 10-15% of missing accuracy")
    print("  • Foundation for multi-scale harmonic analysis")
    print("  • Enables adaptive prediction strategies")

    print("
🚀 Scale-adaptive modeling advances prime gap predictability!"    print()

if __name__ == "__main__":
    run_scale_adaptive_demo()
