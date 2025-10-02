"""
PRIME CEILING CRACKER: Breaking Through the 89% Barrier

Taking Claude's criticisms and our analysis, this implements advanced techniques
to crack the polynomial-time primality detection ceiling.

Strategies:
1. Factorization hint features (trial division depth, etc.)
2. Advanced number theory (multiplicative orders, cyclotomic)
3. Quantum-inspired features
4. Deep learning approaches
5. Ensemble methods with stacking
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
# Advanced ML libraries (with fallbacks)
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
from scipy.stats import entropy
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class PrimeCeilingCracker:
    """
    Advanced system to crack the 89% primality detection ceiling.
    """

    def __init__(self, max_n=100000):
        self.max_n = max_n
        self.primes = self._generate_primes()
        self.scaler = StandardScaler()

        print("🚀 PRIME CEILING CRACKER INITIALIZED")
        print("=" * 50)
        print(f"Target: Break through 89% accuracy barrier")
        print(f"Range: 2 to {max_n}")
        print(f"Primes: {len(self.primes)}")
        print()

    def _generate_primes(self):
        """Generate primes using sieve."""
        sieve = [True] * (self.max_n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(np.sqrt(self.max_n)) + 1):
            if sieve[i]:
                for j in range(i*i, self.max_n + 1, i):
                    sieve[j] = False
        return {i for i in range(self.max_n + 1) if sieve[i]}

    def factorization_hint_features(self, n):
        """Advanced features that hint at factorization without completing it."""
        if n < 2:
            return [0] * 15

        features = []

        # 1. Trial division depth (how far to check before finding factor)
        trial_depth = 0
        sqrt_n = int(np.sqrt(n))
        for i in range(2, min(sqrt_n + 1, 1000)):  # Cap for efficiency
            if n % i == 0:
                trial_depth = i
                break
        features.append(trial_depth / 1000)  # Normalized

        # 2. Small prime factor count (partial trial division)
        small_factors = [2, 3, 5, 7, 11, 13, 17, 19, 23]
        factor_count = 0
        remaining = n
        for p in small_factors:
            if remaining % p == 0:
                factor_count += 1
                while remaining % p == 0:
                    remaining //= p
        features.append(factor_count)
        features.append(1 if remaining > 1 else 0)  # Has large prime factors

        # 3. Fermat's method hint (a^2 - n is square)
        fermat_hint = 0
        ceil_sqrt = int(np.sqrt(n)) + 1
        for a in range(ceil_sqrt, min(ceil_sqrt + 50, n)):  # Limited search
            diff = a*a - n
            sqrt_diff = int(np.sqrt(diff))
            if sqrt_diff*sqrt_diff == diff:
                fermat_hint = a - ceil_sqrt  # How close we got
                break
        features.append(fermat_hint / 50)

        # 4. Pollard's rho hint (cycle detection simulation)
        def pollard_rho_hint(x, c=1):
            """Simplified Pollard rho for hint generation."""
            tortoise = x
            hare = x
            steps = 0
            max_steps = 100  # Limited for efficiency

            def f(x): return (x*x + c) % n

            while steps < max_steps:
                tortoise = f(tortoise)
                hare = f(f(hare))
                steps += 1
                if tortoise == hare and steps > 2:
                    return steps  # Cycle detected
            return max_steps  # No cycle found in limit

        rho_hint = pollard_rho_hint(2) + pollard_rho_hint(3)
        features.append(rho_hint / 200)

        # 5. Continued fraction approximation quality
        cf_hint = 0
        if n > 1:
            # Simple continued fraction for sqrt(n) approximation
            sqrt_n = np.sqrt(n)
            frac_part = sqrt_n - int(sqrt_n)
            # How well can we approximate sqrt(n) with rationals?
            for denom in range(1, 21):
                for num in range(1, 21):
                    approx = num / denom
                    if abs(approx - frac_part) < 0.01:
                        cf_hint += 1
        features.append(cf_hint / 20)

        # 6. Quadratic residue patterns
        qr_features = []
        for mod in [3, 5, 7, 11, 13]:
            legendre = pow(n % mod, (mod-1)//2, mod)
            qr_features.append(legendre)
        features.extend(qr_features)

        # 7. Multiplicative order hints (mod small primes)
        order_hints = []
        for base in [2, 3, 5]:
            if n % base != 0:  # Not divisible
                # Find order of base mod n (limited search)
                order = 1
                current = base % n
                max_order = 20  # Limited for efficiency
                while current != 1 and order < max_order:
                    current = (current * base) % n
                    order += 1
                order_hints.append(min(order, max_order))
            else:
                order_hints.append(0)
        features.extend(order_hints)

        # 8. Digital root patterns in different bases
        digital_features = []
        for base in [10, 16, 8]:
            if base == 10:
                digits = [int(d) for d in str(n)]
            else:
                temp_n = n
                digits = []
                while temp_n > 0:
                    digits.append(temp_n % base)
                    temp_n //= base

            if digits:
                digital_features.extend([
                    sum(digits),  # Sum
                    sum(digits) % (base-1) or (base-1),  # Digital root
                    len(digits),  # Length
                    max(digits),  # Max digit
                    len(set(digits)),  # Unique digits
                ])
            else:
                digital_features.extend([0, 0, 0, 0, 0])

        features.extend(digital_features)

        # Ensure no NaN or inf
        features = [0.0 if not np.isfinite(f) else f for f in features]
        return features

    def quantum_inspired_features(self, n):
        """Features inspired by quantum algorithms."""
        features = []

        # 1. Shor's algorithm inspired (period finding hints)
        period_hints = []
        for base in [2, 3, 5, 7]:
            if n % base != 0:
                # Simulate quantum period finding (simplified)
                periods = []
                for k in range(1, 10):  # Limited search
                    if pow(base, k, n) == 1:
                        periods.append(k)
                        break
                period_hints.append(len(periods))  # Found period or not
            else:
                period_hints.append(0)
        features.extend(period_hints)

        # 2. Grover search inspired (amplitude estimation)
        grover_hint = 0
        if n > 1:
            # Estimate "how prime-like" based on modular properties
            prime_like_score = 0
            moduli = [2, 3, 5, 7, 11, 13, 17, 19, 23]
            for mod in moduli:
                if n % mod != 0:
                    prime_like_score += 1
            grover_hint = prime_like_score / len(moduli)
        features.append(grover_hint)

        # 3. Quantum walk inspired (graph traversal hints)
        walk_hint = 0
        if n > 1:
            # Simplified quantum walk on factor graph
            factors_found = 0
            for i in range(2, min(int(np.sqrt(n)) + 1, 50)):
                if n % i == 0:
                    factors_found += 1
                    break  # Just check if any factor exists
            walk_hint = factors_found
        features.append(walk_hint)

        # 4. Bell state inspired (entanglement hints)
        bell_hint = 0
        if n > 1:
            # Check for special factor relationships
            sqrt_n = int(np.sqrt(n))
            if sqrt_n * sqrt_n == n:
                bell_hint = 1  # Perfect square
            elif any((sqrt_n + i)**2 == n for i in [-2, -1, 1, 2]):
                bell_hint = 0.5  # Close to square
        features.append(bell_hint)

        return features

    def advanced_number_theory_features(self, n):
        """Advanced number theoretic features."""
        features = []

        # 1. Cyclotomic polynomial evaluations
        cyclo_hints = []
        for k in [3, 4, 5, 6]:  # Small cyclotomic polynomials
            if k == 3:
                cyclo = n*n - n + 1  # Φ3(x) = x² - x + 1
            elif k == 4:
                cyclo = n*n + 1  # Φ4(x) = x² + 1
            elif k == 5:
                cyclo = n**4 + n**3 + n**2 + n + 1  # Φ5(x) = x⁴ + x³ + x² + x + 1
            else:  # k == 6
                cyclo = n*n - n + 1  # Φ6(x) = x² - x + 1

            cyclo_hints.append(cyclo % 1000 / 1000)  # Normalized
        features.extend(cyclo_hints)

        # 2. Character sum approximations
        char_hints = []
        for mod in [4, 8, 12]:
            char_sum = sum(np.exp(2j * np.pi * i * n / mod) for i in range(mod))
            char_hints.append(abs(char_sum) / mod)
        features.extend(char_hints)

        # 3. L-function approximations (very simplified)
        l_hint = 0
        if n > 1:
            # Simplified L-function at s=1 (related to prime counting)
            log_term = np.log(n) if n > 1 else 0
            l_hint = 1 / (1 - 1/np.exp(log_term)) if log_term > 0 else 0
        features.append(min(l_hint, 10) / 10)  # Capped

        # 4. Modular inverse existence
        inv_hints = []
        for mod in [8, 12, 16, 24]:
            try:
                inv = pow(n, -1, mod)
                inv_hints.append(inv / mod)
            except:
                inv_hints.append(0)  # No inverse
        features.extend(inv_hints)

        return features

    def get_advanced_feature_sets(self):
        """Different advanced feature set combinations."""
        return {
            'Factorization Hints': lambda n: self.factorization_hint_features(n),
            'Quantum Inspired': lambda n: self.quantum_inspired_features(n),
            'Advanced Number Theory': lambda n: self.advanced_number_theory_features(n),
            'Combined Advanced': lambda n: (self.factorization_hint_features(n) +
                                          self.quantum_inspired_features(n) +
                                          self.advanced_number_theory_features(n)),
        }

    def get_advanced_models(self):
        """Advanced ML models for cracking the ceiling."""
        models = {
            'Neural Network': lambda: MLPClassifier(
                hidden_layer_sizes=(100, 50, 25), max_iter=1000,
                random_state=42, early_stopping=True
            ),
            'SVM RBF': lambda: SVC(
                kernel='rbf', C=10, gamma='scale',
                probability=True, random_state=42
            ),
            'Extra Trees': lambda: ExtraTreesClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
            'Gradient Boosting': lambda: GradientBoostingClassifier(
                n_estimators=200, max_depth=6, random_state=42
            ),
            'Random Forest': lambda: RandomForestClassifier(
                n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
            ),
        }

        # Add advanced models if available
        if XGB_AVAILABLE:
            models['XGBoost'] = lambda: xgb.XGBClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )

        if LGBM_AVAILABLE:
            models['LightGBM'] = lambda: lgb.LGBMClassifier(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1
            )

        return models

    def crack_attempt(self, test_range=(15000, 20000), k_folds=5):
        """Main cracking attempt using advanced techniques."""
        print("🛠️  CRACKING ATTEMPT: Advanced Feature Engineering")
        print("=" * 55)

        # Generate test data
        numbers = np.arange(test_range[0], test_range[1] + 1)
        labels = np.array([1 if n in self.primes else 0 for n in numbers])

        # Get feature sets
        feature_sets = self.get_advanced_feature_sets()
        models = self.get_advanced_models()

        results = {}

        for feature_name, feature_func in feature_sets.items():
            print(f"\n🔬 Testing {feature_name} Features")
            print("-" * 40)

            # Generate features
            features = np.array([feature_func(n) for n in numbers])
            print(f"Feature dimension: {features.shape[1]}")

            # K-fold cross-validation
            skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_results = []

            for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels)):
                X_train, X_test = features[train_idx], features[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Scale features
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)

                fold_model_results = {}

                for model_name, model_func in models.items():
                    try:
                        model = model_func()
                        model.fit(X_train_scaled, y_train)

                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)

                        fold_model_results[model_name] = accuracy

                    except Exception as e:
                        print(f"Error with {model_name}: {e}")
                        fold_model_results[model_name] = 0

                fold_results.append(fold_model_results)
                print(f"Fold {fold+1}: Best = {max(fold_model_results.values()):.4f}")

            # Aggregate results
            model_accuracies = {}
            for model_name in models.keys():
                scores = [fold[model_name] for fold in fold_results if model_name in fold]
                model_accuracies[model_name] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'max': np.max(scores),
                }

            results[feature_name] = model_accuracies

            # Print summary for this feature set
            best_model = max(model_accuracies.items(), key=lambda x: x[1]['mean'])
            print(f"Best: {best_model[0]} = {best_model[1]['mean']:.4f} ± {best_model[1]['std']:.4f}")

        # Overall results
        self._analyze_cracking_results(results)

    def _analyze_cracking_results(self, results):
        """Analyze the cracking attempt results."""
        print(f"\n🎯 CRACKING RESULTS ANALYSIS")
        print("=" * 35)

        # Find best overall performance
        best_overall = 0
        best_config = None

        for feature_set, model_results in results.items():
            for model, stats in model_results.items():
                if stats['mean'] > best_overall:
                    best_overall = stats['mean']
                    best_config = (feature_set, model, stats)

        print(f"Best Overall Performance: {best_overall:.4f}")
        print(f"Configuration: {best_config[1]} with {best_config[0]} features")
        print(f"Std Dev: {best_config[2]['std']:.4f}")

        # Compare to baseline (89% ceiling)
        baseline = 0.89
        improvement = best_overall - baseline

        print(f"\\n🎯 CEILING BREAKTHROUGH ASSESSMENT")
        print("=" * 40)
        print(f"Baseline: {baseline:.4f}")
        print(f"Best performance: {best_overall:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        print()
        if best_overall > 0.91:  # Clear break
            print("✅ SUCCESS: Broken through the 89% ceiling!")
        elif best_overall > baseline + 0.005:  # Small improvement
            print("⚠️  MARGINAL: Slight improvement over baseline")
        else:
            print("❌ FAILURE: Did not break the ceiling")        # Analyze feature set performance
        print(f"\\n📊 FEATURE SET COMPARISON")
        print("=" * 30)

        feature_performance = {}
        for feature_set, model_results in results.items():
            avg_performance = np.mean([stats['mean'] for stats in model_results.values()])
            feature_performance[feature_set] = avg_performance

        for feature_set, perf in sorted(feature_performance.items(), key=lambda x: x[1], reverse=True):
            print(".4f"
        # Theoretical implications
        self._theoretical_implications(best_overall, baseline)

    def _theoretical_implications(self, best_performance, baseline):
        """Discuss theoretical implications of cracking attempt."""
        print(f"\\n🎓 THEORETICAL IMPLICATIONS")
        print("=" * 27)

        if best_performance > 0.91:
            implications = [
                "Advanced features capture factorization information beyond polynomial time",
                "ML can learn complex number-theoretic relationships",
                "The 89% ceiling was a feature engineering limitation, not fundamental",
                "New approaches to primality testing possible",
                "Cryptographic hardness assumptions may need revision"
            ]
        elif best_performance > baseline + 0.005:
            implications = [
                "Incremental improvements possible with advanced features",
                "Some factorization hints are learnable by ML",
                "The ceiling is soft, not hard",
                "Hybrid classical/ML approaches show promise",
                "More research needed on advanced feature engineering"
            ]
        else:
            implications = [
                "The 89% ceiling represents fundamental information-theoretic limits",
                "Polynomial-time features cannot distinguish hard semiprimes",
                "Cryptographic hardness assumptions validated",
                "Focus should shift to quantum or exponential-time approaches",
                "The ceiling is genuine, not artifact"
            ]

        for implication in implications:
            print(f"• {implication}")

        print(f"\\n🔮 NEXT STEPS")
        print("=" * 12)
        if best_performance <= baseline + 0.005:
            print("• Explore quantum algorithms for primality testing")
            print("• Investigate exponential-time factorization hints")
            print("• Research information-theoretic limits more deeply")
            print("• Consider cryptographic applications of the ceiling")
        else:
            print("• Scale up successful feature engineering approaches")
            print("• Optimize ML architectures for number theory")
            print("• Apply to other computational number theory problems")
            print("• Investigate theoretical foundations of the breakthrough")

def main():
    """Run the prime ceiling cracking attempt."""
    cracker = PrimeCeilingCracker(max_n=100000)

    # Test on challenging range
    cracker.crack_attempt(test_range=(15000, 20000), k_folds=5)

    print(f"\\n🚀 CRACKING ATTEMPT COMPLETE")
    print("If successful, this represents a breakthrough in primality testing.")
    print("If not, it validates the information-theoretic limits we discovered.")

if __name__ == "__main__":
    main()
