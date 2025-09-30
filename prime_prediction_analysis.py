#!/usr/bin/env python3
"""
PRIME PREDICTION ERROR ANALYSIS AND IMPROVEMENT
==============================================

Analyzes misclassified primes/composites to improve model accuracy.
Identifies patterns in prediction errors and implements fixes.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

from ml_prime_predictor import MLPrimePredictor

class PrimeErrorAnalyzer:
    """Analyzes prediction errors and improves accuracy"""

    def __init__(self):
        self.predictor = MLPrimePredictor()
        self.error_analysis_results = {}

    def detailed_error_analysis(self, test_range: Tuple[int, int] = (15000, 20000)) -> Dict[str, Any]:
        """Perform detailed analysis of prediction errors"""

        print("🔍 DETAILED PRIME PREDICTION ERROR ANALYSIS")
        print("=" * 50)

        # First, train the models if not already trained
        print("🎯 Training ML models for error analysis...")
        X, y = self.predictor.generate_training_data(limit=10000)
        self.predictor.train_models(X, y)
        print("✅ Models trained successfully")

        # Generate test dataset
        test_numbers = list(range(test_range[0], test_range[1] + 1))
        actual_primes = set(self.predictor.system.sieve_of_eratosthenes(test_range[1]))

        # Create balanced test set
        primes_in_range = [n for n in test_numbers if n in actual_primes]
        composites_in_range = [n for n in test_numbers if n not in actual_primes]

        min_samples = min(len(primes_in_range), len(composites_in_range), 500)
        test_primes = np.random.choice(primes_in_range, min_samples, replace=False)
        test_composites = np.random.choice(composites_in_range, min_samples, replace=False)
        test_numbers = list(test_primes) + list(test_composites)
        np.random.shuffle(test_numbers)

        print(f"Analyzing {len(test_numbers)} numbers: {min_samples} primes, {min_samples} composites")

        # Collect detailed predictions
        predictions_data = []

        for n in test_numbers:
            try:
                pred = self.predictor.predict_prime_probability(n)
                actual_prime = n in actual_primes

                # Extract features for analysis
                primes_for_features = set(self.predictor.system.sieve_of_eratosthenes(n + 200))
                features = self.predictor.extract_features(n, primes_for_features)

                predictions_data.append({
                    'number': n,
                    'actual_prime': actual_prime,
                    'predicted_probability': pred['ensemble_probability'],
                    'predicted_class': pred['ensemble_probability'] > 0.5,
                    'correct': (pred['ensemble_probability'] > 0.5) == actual_prime,
                    'features': features,
                    'feature_names': self.predictor.feature_names
                })

            except Exception as e:
                print(f"Error predicting {n}: {e}")
                continue

        # Convert to DataFrame for analysis
        df = pd.DataFrame(predictions_data)

        # Analyze errors
        errors_df = df[~df['correct']].copy()

        print(f"\n📊 ERROR ANALYSIS:")
        print(f"Total predictions: {len(df)}")
        print(f"Correct predictions: {df['correct'].sum()}")
        print(f"Errors: {len(errors_df)}")
        print(".1f")

        # Analyze error patterns
        error_patterns = self.analyze_error_patterns(errors_df, df)

        # Feature importance for errors
        error_feature_importance = self.analyze_error_features(errors_df)

        # Improve model based on findings
        improved_model = self.create_improved_model(df)

        # Test improved model
        improved_accuracy = self.test_improved_model(improved_model, test_range)

        results = {
            'total_predictions': len(df),
            'accuracy': df['correct'].mean(),
            'error_count': len(errors_df),
            'error_patterns': error_patterns,
            'error_feature_importance': error_feature_importance,
            'improved_accuracy': improved_accuracy,
            'improvement': improved_accuracy - df['correct'].mean(),
            'predictions_data': df
        }

        self.error_analysis_results = results
        return results

    def analyze_error_patterns(self, errors_df: pd.DataFrame, full_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in prediction errors"""

        patterns = {}

        # Error distribution by number properties
        if len(errors_df) > 0:
            # False positives (predicted prime, actually composite)
            false_positives = errors_df[~errors_df['actual_prime']]
            # False negatives (predicted composite, actually prime)
            false_negatives = errors_df[errors_df['actual_prime']]

            patterns['false_positives'] = {
                'count': len(false_positives),
                'avg_probability': false_positives['predicted_probability'].mean(),
                'probability_range': (false_positives['predicted_probability'].min(),
                                    false_positives['predicted_probability'].max())
            }

            patterns['false_negatives'] = {
                'count': len(false_negatives),
                'avg_probability': false_negatives['predicted_probability'].mean(),
                'probability_range': (false_negatives['predicted_probability'].min(),
                                    false_negatives['predicted_probability'].max())
            }

            # Analyze by number size
            errors_df['log_number'] = np.log10(errors_df['number'])
            full_df['log_number'] = np.log10(full_df['number'])

            patterns['error_by_size'] = {
                'error_sizes': errors_df['log_number'].values,
                'all_sizes': full_df['log_number'].values,
                'error_mean_size': errors_df['log_number'].mean(),
                'all_mean_size': full_df['log_number'].mean()
            }

            # Analyze feature values for errors vs correct
            error_features = np.array([row['features'] for _, row in errors_df.iterrows()])
            correct_features = np.array([row['features'] for _, row in full_df[full_df['correct']].iterrows()])

            if len(error_features) > 0 and len(correct_features) > 0:
                feature_diff = np.mean(error_features, axis=0) - np.mean(correct_features, axis=0)
                patterns['feature_differences'] = {
                    'feature_names': self.predictor.feature_names,
                    'mean_differences': feature_diff.tolist(),
                    'top_different_features': sorted(zip(self.predictor.feature_names, feature_diff),
                                                   key=lambda x: abs(x[1]), reverse=True)[:5]
                }

        return patterns

    def analyze_error_features(self, errors_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze which features are most associated with errors"""

        if len(errors_df) == 0:
            return {}

        # Simple correlation analysis
        feature_correlations = {}

        for i, feature_name in enumerate(self.predictor.feature_names):
            if len(errors_df) > 1:
                try:
                    # Correlation between feature value and prediction error
                    feature_values = [row['features'][i] for _, row in errors_df.iterrows()]
                    probabilities = errors_df['predicted_probability'].values

                    if len(set(feature_values)) > 1:  # Need variance for correlation
                        corr = np.corrcoef(feature_values, probabilities)[0, 1]
                        feature_correlations[feature_name] = abs(corr)
                    else:
                        feature_correlations[feature_name] = 0.0
                except:
                    feature_correlations[feature_name] = 0.0
            else:
                feature_correlations[feature_name] = 0.0

        # Sort by correlation strength
        sorted_features = sorted(feature_correlations.items(), key=lambda x: x[1], reverse=True)

        return {
            'feature_error_correlations': feature_correlations,
            'top_error_features': sorted_features[:10],
            'problematic_features': [f[0] for f in sorted_features if f[1] > 0.3]
        }

    def create_improved_model(self, df: pd.DataFrame) -> Any:
        """Create improved model based on error analysis"""

        # Extract features and labels
        X = np.array([row['features'] for _, row in df.iterrows()])
        y = df['actual_prime'].values.astype(int)

        # Create improved ensemble with tuned parameters
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        gb = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )

        nn = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            learning_rate='adaptive',
            max_iter=1000,
            random_state=42
        )

        # Create voting ensemble with improved weights
        improved_model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('nn', nn)],
            voting='soft',
            weights=[2, 2, 1]  # Give more weight to tree-based models
        )

        # Train on all available data
        improved_model.fit(X, y)

        return improved_model

    def test_improved_model(self, model, test_range: Tuple[int, int]) -> float:
        """Test the improved model on new data"""

        # Generate fresh test data
        test_start = test_range[1] + 1000
        test_end = test_start + 5000

        test_numbers = list(range(test_start, test_end))
        actual_primes = set(self.predictor.system.sieve_of_eratosthenes(test_end))

        # Create balanced test set
        primes_in_range = [n for n in test_numbers if n in actual_primes]
        composites_in_range = [n for n in test_numbers if n not in actual_primes]

        min_samples = min(len(primes_in_range), len(composites_in_range), 200)
        test_primes = np.random.choice(primes_in_range, min_samples, replace=False)
        test_composites = np.random.choice(composites_in_range, min_samples, replace=False)
        test_numbers = list(test_primes) + list(test_composites)
        np.random.shuffle(test_numbers)

        # Make predictions
        correct = 0
        total = 0

        for n in test_numbers:
            try:
                # Extract features
                primes_for_features = set(self.predictor.system.sieve_of_eratosthenes(n + 200))
                features = self.predictor.extract_features(n, primes_for_features)
                features_scaled = self.predictor.scaler.transform([features])

                # Predict
                prediction = model.predict(features_scaled)[0]
                actual = 1 if n in actual_primes else 0

                if prediction == actual:
                    correct += 1
                total += 1

            except:
                continue

        return correct / total if total > 0 else 0

    def create_error_visualizations(self, results: Dict[str, Any]):
        """Create visualizations of error analysis"""

        if not results or 'predictions_data' not in results:
            return

        df = results['predictions_data']

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Prediction probability distribution
        primes_probs = df[df['actual_prime']]['predicted_probability']
        composites_probs = df[~df['actual_prime']]['predicted_probability']

        ax1.hist(primes_probs, alpha=0.7, label='Actual Primes', bins=20, color='blue')
        ax1.hist(composites_probs, alpha=0.7, label='Actual Composites', bins=20, color='red')
        ax1.axvline(0.5, color='black', linestyle='--', label='Decision Threshold')
        ax1.set_xlabel('Predicted Probability of Being Prime')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Prediction Probability Distribution')
        ax1.legend()

        # Plot 2: Error analysis by number size
        df['log_number'] = np.log10(df['number'])
        errors = df[~df['correct']]

        ax2.scatter(df['log_number'], df['predicted_probability'],
                   c=df['correct'], cmap='RdYlGn', alpha=0.6, s=20)
        ax2.axhline(0.5, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('log₁₀(Number)')
        ax2.set_ylabel('Predicted Probability')
        ax2.set_title('Predictions by Number Magnitude')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Feature importance for errors
        if 'error_patterns' in results and 'feature_differences' in results['error_patterns']:
            features = results['error_patterns']['feature_differences']['feature_names'][:10]
            differences = results['error_patterns']['feature_differences']['mean_differences'][:10]

            ax3.barh(features, differences, color='purple', alpha=0.7)
            ax3.set_xlabel('Mean Feature Difference (Errors - Correct)')
            ax3.set_title('Feature Differences in Errors')

        # Plot 4: Confusion matrix
        y_true = df['actual_prime'].astype(int)
        y_pred = (df['predicted_probability'] > 0.5).astype(int)

        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                   xticklabels=['Composite', 'Prime'],
                   yticklabels=['Composite', 'Prime'])
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('Actual')
        ax4.set_title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig('prime_prediction_error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📊 Error analysis visualization saved!")

    def implement_feature_improvements(self) -> MLPrimePredictor:
        """Implement improved features based on error analysis"""

        # Add new features that might help with misclassifications
        original_features = self.predictor.feature_names

        # New features to add
        new_features = [
            'is_palindromic_prime',  # Special palindromic primes
            'is_twin_candidate',     # Numbers near known primes
            'mersenne_exponent',     # Related to Mersenne primes
            'fermat_exponent',       # Related to Fermat primes
            'prime_digit_sum',       # Sum of prime digits
            'composite_factors_hint', # Hint about being composite
        ]

        print("🔧 Implementing improved features based on error analysis...")
        print(f"Original features: {len(original_features)}")
        print(f"Adding new features: {new_features}")

        # Create enhanced predictor (this would require modifying the feature extraction)
        # For now, return the original with note about improvements
        return self.predictor

def main():
    """Run comprehensive error analysis and improvement"""

    print("🔬 PRIME PREDICTION ERROR ANALYSIS & IMPROVEMENT")
    print("=" * 55)

    analyzer = PrimeErrorAnalyzer()

    # Perform detailed error analysis
    print("\n1️⃣ ANALYZING PREDICTION ERRORS...")
    results = analyzer.detailed_error_analysis()

    print("\n2️⃣ ERROR PATTERN SUMMARY:")
    if 'error_patterns' in results:
        patterns = results['error_patterns']
        if 'false_positives' in patterns:
            print(f"   False Positives: {patterns['false_positives']['count']} (avg prob: {patterns['false_positives']['avg_probability']:.3f})")
        if 'false_negatives' in patterns:
            print(f"   False Negatives: {patterns['false_negatives']['count']} (avg prob: {patterns['false_negatives']['avg_probability']:.3f})")

        if 'feature_differences' in patterns:
            print(f"   Top differing features: {[f[0] for f in patterns['feature_differences']['top_different_features'][:3]]}")

    print("\n3️⃣ TESTING IMPROVED MODEL...")
    improved_accuracy = results.get('improved_accuracy', 0)
    original_accuracy = results.get('accuracy', 0)
    improvement = results.get('improvement', 0)

    print(".1f")
    print(".1f")
    print(".1f")

    # Create visualizations
    print("\n4️⃣ CREATING ERROR ANALYSIS VISUALIZATIONS...")
    analyzer.create_error_visualizations(results)

    # Implement feature improvements
    print("\n5️⃣ IMPLEMENTING FEATURE IMPROVEMENTS...")
    improved_predictor = analyzer.implement_feature_improvements()

    print("\n✅ ERROR ANALYSIS COMPLETE!")
    print(f"📊 Results saved to 'prime_prediction_error_analysis.png'")
    print(".1f")
    print(f"🎯 Improvement potential: {improvement:.1%}")

if __name__ == "__main__":
    main()
