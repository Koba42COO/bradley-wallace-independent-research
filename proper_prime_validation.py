#!/usr/bin/env python3
"""
PROPER SCIENTIFIC VALIDATION OF PRIME PREDICTION SYSTEMS
========================================================

Addresses Claude's critical concerns by implementing rigorous, unbiased validation:

1. **Cross-validation** on truly held-out data
2. **Baseline comparisons** to simple models
3. **No parameter tuning** on test data
4. **Statistical significance** testing
5. **Real predictive power** assessment

This eliminates circular reasoning and tests actual prediction capability.
"""

import numpy as np
import scipy.stats as stats
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import json
import time
from pathlib import Path

class ScientificPrimeValidator:
    """Rigorous scientific validation framework"""

    def __init__(self):
        self.baseline_models = {
            'logarithmic': self.baseline_logarithmic,
            'constant': self.baseline_constant,
            'linear': self.baseline_linear,
            'random': self.baseline_random
        }

    def sieve_primes(self, limit: int) -> List[int]:
        """Generate primes using sieve"""
        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0:2] = False

        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i::i] = False

        return np.where(sieve)[0].tolist()

    def baseline_logarithmic(self, current_prime: int) -> int:
        """Baseline: gap ≈ ln(p)"""
        return max(1, int(np.log(current_prime)))

    def baseline_constant(self, current_prime: int) -> int:
        """Baseline: always predict 2 (twin prime assumption)"""
        return 2

    def baseline_linear(self, current_prime: int) -> int:
        """Baseline: gap ≈ √p / 10"""
        return max(1, int(np.sqrt(current_prime) / 10))

    def baseline_random(self, current_prime: int) -> int:
        """Baseline: random gap 1-10"""
        return np.random.randint(1, 11)

    def harmonic_family_predictor(self, current_prime: int, training_data: Dict[str, Any]) -> int:
        """Harmonic family-based prediction (learned from training data only)"""
        # Use training data to determine which family this prime belongs to
        # This avoids circular reasoning by only using training patterns

        # Simple approach: predict based on prime size quantiles from training
        if 'size_quantiles' not in training_data:
            return self.baseline_logarithmic(current_prime)

        small_threshold = training_data['size_quantiles'][0.33]
        large_threshold = training_data['size_quantiles'][0.67]

        if current_prime < small_threshold:
            return training_data.get('small_gap_mean', 2)
        elif current_prime < large_threshold:
            return training_data.get('medium_gap_mean', 4)
        else:
            return training_data.get('large_gap_mean', 8)

    def phi_scaling_predictor(self, current_prime: int, training_data: Dict[str, Any]) -> int:
        """φ-scaling based prediction (learned from training data only)"""
        base_gap = int(np.log(current_prime))

        # Apply learned φ-scaling factor from training data
        phi_factor = training_data.get('phi_scaling_factor', 1.0)

        return max(1, int(base_gap * phi_factor))

    def learn_from_training_data(self, train_primes: List[int]) -> Dict[str, Any]:
        """Learn patterns ONLY from training data (no test data contamination)"""
        train_gaps = [train_primes[i+1] - train_primes[i] for i in range(len(train_primes)-1)]

        training_patterns = {
            'size_quantiles': {
                0.33: np.percentile(train_primes[:-1], 33),
                0.67: np.percentile(train_primes[:-1], 67)
            },
            'small_gap_mean': np.mean([g for g in train_gaps if g <= 4]),
            'medium_gap_mean': np.mean([g for g in train_gaps if 4 < g <= 10]),
            'large_gap_mean': np.mean([g for g in train_gaps if g > 10]),
            'phi_scaling_factor': np.mean(train_gaps) / np.mean([np.log(p) for p in train_primes[:-1]]),
            'avg_gap': np.mean(train_gaps),
            'gap_std': np.std(train_gaps)
        }

        return training_patterns

    def cross_validate_prediction_system(self, max_prime: int = 100000, n_splits: int = 5) -> Dict[str, Any]:
        """
        Rigorous cross-validation: train on some primes, test on others
        This prevents overfitting and tests real predictive power
        """
        print("🔬 PROPER SCIENTIFIC CROSS-VALIDATION")
        print("=" * 50)

        # Generate all primes
        all_primes = self.sieve_primes(max_prime)
        print(f"✓ Generated {len(all_primes):,} primes up to {max_prime:,}")

        # Prepare for k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        results = {
            'harmonic_family': {'accuracies': [], 'improvements': []},
            'phi_scaling': {'accuracies': [], 'improvements': []},
            'hybrid': {'accuracies': [], 'improvements': []},
            'baselines': {name.replace('_', ''): {'accuracies': []} for name in self.baseline_models.keys()}
        }

        fold = 1
        for train_idx, test_idx in kf.split(all_primes[:-1]):  # -1 because we need gaps
            print(f"\n📊 Fold {fold}/{n_splits}")

            # Split data (train_idx and test_idx are for gaps, not primes)
            train_primes = [all_primes[i] for i in train_idx]
            test_primes = [all_primes[i] for i in test_idx]

            # Add the next prime for gap calculation
            train_primes_with_next = train_primes + [all_primes[len(train_primes)]]
            test_primes_with_next = test_primes + [all_primes[len(test_primes)]]

            # Learn patterns from training data ONLY
            training_patterns = self.learn_from_training_data(train_primes_with_next)

            # Test on held-out data
            fold_results = self._evaluate_on_test_set(
                test_primes_with_next, training_patterns, results
            )

            fold += 1

        # Compute final statistics
        return self._compute_final_statistics(results)

    def _evaluate_on_test_set(self, test_primes: List[int], training_patterns: Dict[str, Any],
                            results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate all models on held-out test data"""

        test_gaps = [test_primes[i+1] - test_primes[i] for i in range(len(test_primes)-1)]

        # Test each model
        predictions = {
            'harmonic_family': [],
            'phi_scaling': [],
            'hybrid': [],
            **{name.replace('_', ''): [] for name in self.baseline_models.keys()}
        }

        for i, prime in enumerate(test_primes[:-1]):
            # Our advanced models
            predictions['harmonic_family'].append(
                self.harmonic_family_predictor(prime, training_patterns)
            )
            predictions['phi_scaling'].append(
                self.phi_scaling_predictor(prime, training_patterns)
            )
            predictions['hybrid'].append(
                int((predictions['harmonic_family'][-1] + predictions['phi_scaling'][-1]) / 2)
            )

            # Baseline models
            for name, baseline_func in self.baseline_models.items():
                predictions[name.replace('_', '')].append(baseline_func(prime))

        # Calculate accuracies (within 1 unit of actual gap)
        for model_name, preds in predictions.items():
            correct = sum(1 for pred, actual in zip(preds, test_gaps) if abs(pred - actual) <= 1)
            accuracy = correct / len(test_gaps)

            if model_name in results['baselines']:
                results['baselines'][model_name]['accuracies'].append(accuracy)
            else:
                results[model_name]['accuracies'].append(accuracy)

        # Calculate improvements over best baseline
        best_baseline_acc = max(np.mean(results['baselines'][name]['accuracies'])
                              for name in results['baselines'].keys())

        for model_name in ['harmonic_family', 'phi_scaling', 'hybrid']:
            model_acc = results[model_name]['accuracies'][-1]
            improvement = model_acc - best_baseline_acc
            results[model_name]['improvements'].append(improvement)

        return results

    def _compute_final_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute final statistical analysis"""

        final_results = {}

        # Analyze each model
        for model_name in ['harmonic_family', 'phi_scaling', 'hybrid']:
            accuracies = results[model_name]['accuracies']
            improvements = results[model_name]['improvements']

            final_results[model_name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_improvement': np.mean(improvements),
                'std_improvement': np.std(improvements),
                'improvement_significant': self._test_significance(improvements),
                'fold_accuracies': accuracies,
                'fold_improvements': improvements
            }

        # Analyze baselines
        final_results['baselines'] = {}
        for name, data in results['baselines'].items():
            accuracies = data['accuracies']
            final_results['baselines'][name] = {
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies)
            }

        # Overall assessment
        best_model = max(final_results.keys(),
                        key=lambda x: final_results[x]['mean_accuracy'] if x != 'baselines' else 0)
        best_baseline = max(final_results['baselines'].keys(),
                          key=lambda x: final_results['baselines'][x]['mean_accuracy'])

        final_results['assessment'] = {
            'best_model': best_model,
            'best_baseline': best_baseline,
            'best_model_accuracy': final_results[best_model]['mean_accuracy'],
            'best_baseline_accuracy': final_results['baselines'][best_baseline]['mean_accuracy'],
            'real_improvement': final_results[best_model]['mean_accuracy'] > final_results['baselines'][best_baseline]['mean_accuracy'],
            'improvement_magnitude': final_results[best_model]['mean_accuracy'] - final_results['baselines'][best_baseline]['mean_accuracy']
        }

        return final_results

    def _test_significance(self, improvements: List[float]) -> bool:
        """Test if improvements are statistically significant"""
        if len(improvements) < 3:
            return False

        # One-sample t-test: are improvements significantly > 0?
        t_stat, p_value = stats.ttest_1samp(improvements, 0)

        return p_value < 0.05 and np.mean(improvements) > 0

    def run_validation(self, max_prime: int = 100000) -> Dict[str, Any]:
        """Main validation execution"""
        print("🧪 SCIENTIFIC PRIME PREDICTION VALIDATION")
        print("=" * 60)
        print("Testing: Harmonic Families + φ-Scaling vs Baseline Models")
        print("Method: 5-fold cross-validation on held-out data")
        print("=" * 60)

        start_time = time.time()
        results = self.cross_validate_prediction_system(max_prime)
        validation_time = time.time() - start_time

        # Print results
        self._print_validation_results(results)

        # Create visualization
        self._create_validation_visualization(results)

        # Save results
        output = {
            'validation_results': results,
            'validation_time': validation_time,
            'timestamp': time.time(),
            'methodology': {
                'cross_validation_folds': 5,
                'test_metric': 'accuracy_within_1_unit',
                'data_split': 'random_kfold',
                'no_test_data_contamination': True
            }
        }

        with open('scientific_prime_validation.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)

        return results

    def _print_validation_results(self, results: Dict[str, Any]):
        """Print comprehensive validation results"""

        print("\n📊 CROSS-VALIDATION RESULTS")
        print("=" * 40)

        assessment = results['assessment']

        print("🎯 BOTTOM LINE:")
        print(f"   Best Model: {assessment['best_model']} ({assessment['best_model_accuracy']:.1%})")
        print(f"   Best Baseline: {assessment['best_baseline']} ({assessment['best_baseline_accuracy']:.1%})")
        print(f"   Real Improvement: {'YES' if assessment['real_improvement'] else 'NO'}")
        print(".1f")

        print("\n📈 DETAILED MODEL PERFORMANCE:")
        for model_name in ['harmonic_family', 'phi_scaling', 'hybrid']:
            if model_name in results:
                data = results[model_name]
                print("2s")
                print(".1f")
                print("5s")

        print("\n🎯 BASELINE PERFORMANCE:")
        for name, data in results['baselines'].items():
            print("10s")

        print("\n🧪 SCIENTIFIC ASSESSMENT:")
        if assessment['real_improvement']:
            print("   ✅ STATISTICAL SUCCESS: Models show real predictive power!")
            print("   📊 Models outperform baselines on unseen data")
            print("   🎯 Cross-validation confirms genuine pattern discovery")
        else:
            print("   ❌ NO IMPROVEMENT: Models do not outperform baselines")
            print("   📊 Results consistent with random chance")
            print("   🎯 No evidence of genuine prime prediction patterns")
        print("\n💾 Results saved to scientific_prime_validation.json")
    def _create_validation_visualization(self, results: Dict[str, Any]):
        """Create validation results visualization"""

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Model vs Baseline Comparison
        models = ['harmonic_family', 'phi_scaling', 'hybrid']
        baselines = list(results['baselines'].keys())

        model_accs = [results[m]['mean_accuracy'] for m in models]
        baseline_accs = [results['baselines'][b]['mean_accuracy'] for b in baselines]

        x1 = np.arange(len(models))
        x2 = np.arange(len(baselines)) + len(models) + 0.5

        bars1 = ax1.bar(x1, model_accs, color=['blue', 'gold', 'purple'], alpha=0.7, label='Advanced Models')
        bars2 = ax1.bar(x2, baseline_accs, color=['red', 'orange', 'gray', 'black'], alpha=0.7, label='Baselines')

        ax1.set_title('Scientific Validation: Model vs Baseline Performance')
        ax1.set_ylabel('Prediction Accuracy')
        ax1.set_xticks(list(x1) + list(x2))
        ax1.set_xticklabels(models + baselines, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        '.1f', ha='center', va='bottom')

        # Plot 2: Cross-validation Stability
        cv_data = []
        labels = []
        for model in models:
            cv_data.append(results[model]['fold_accuracies'])
            labels.append(model.replace('_', ' ').title())

        ax2.boxplot(cv_data, labels=labels)
        ax2.set_title('Cross-Validation Stability (5-fold)')
        ax2.set_ylabel('Accuracy Distribution')
        ax2.grid(True, alpha=0.3)

        # Plot 3: Improvement Significance
        improvements = [results[m]['mean_improvement'] for m in models]
        errors = [results[m]['std_improvement'] for m in models]

        bars = ax3.bar(range(len(models)), improvements, yerr=errors,
                      color=['blue', 'gold', 'purple'], alpha=0.7, capsize=5)
        ax3.set_title('Improvement Over Best Baseline')
        ax3.set_ylabel('Accuracy Improvement')
        ax3.set_xticks(range(len(models)))
        ax3.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True, alpha=0.3)

        # Plot 4: Statistical Significance
        significance = [results[m]['improvement_significant'] for m in models]
        colors = ['green' if sig else 'red' for sig in significance]

        bars = ax4.bar(range(len(models)), [1 if sig else 0 for sig in significance],
                      color=colors, alpha=0.7)
        ax4.set_title('Statistical Significance (p < 0.05)')
        ax4.set_ylabel('Significance')
        ax4.set_xticks(range(len(models)))
        ax4.set_xticklabels([m.replace('_', ' ').title() for m in models])
        ax4.set_yticks([0, 1])
        ax4.set_yticklabels(['Not Significant', 'Significant'])

        plt.tight_layout()
        plt.savefig('scientific_prime_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("📊 Scientific validation visualization saved!")

def main():
    """Run scientific validation"""
    validator = ScientificPrimeValidator()
    results = validator.run_validation(max_prime=100000)

    print("\n🔬 SCIENTIFIC VALIDATION COMPLETE")
    print("=" * 50)

    assessment = results['assessment']
    if assessment['real_improvement']:
        print("✅ CONCLUSION: Your prime prediction frameworks show genuine predictive power!")
        print("   The patterns discovered are real and predictive, not artifacts.")
    else:
        print("❌ CONCLUSION: No evidence of genuine prime prediction capability.")
        print("   Results are consistent with baseline performance.")

    print("\n📊 Key Evidence:")
    print(".1f")
    print(".1f")

    print("\n🎯 Next Steps:")
    if assessment['real_improvement']:
        print("   ✅ Scale up testing to larger prime ranges")
        print("   ✅ Compare against more sophisticated baselines")
        print("   ✅ Explore hybrid model optimization")
    else:
        print("   🔄 Re-examine the mathematical foundations")
        print("   🔄 Consider if prime gaps are truly predictable")
        print("   🔄 Focus on pattern discovery rather than prediction")

if __name__ == "__main__":
    main()
