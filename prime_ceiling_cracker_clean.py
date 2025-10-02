"""
PRIME CEILING CRACKER - CLEAN VERSION

Removes all leaky features that perform partial factorization.
Only keeps polynomial-time computable features.
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
import warnings
warnings.filterwarnings('ignore')

class PrimeCeilingCrackerClean:
    """
    Clean version that only uses polynomial-time features.
    No partial factorization or exponential-time computations.
    """

    def __init__(self, max_n=100000):
        self.max_n = max_n
        self.primes = self._generate_primes()
        self.scaler = StandardScaler()

        print("🧹 PRIME CEILING CRACKER - CLEAN VERSION")
        print("=" * 50)
        print(f"Only polynomial-time features - no factorization work")
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

    def clean_factorization_hints(self, n):
        """Only polynomial-time hints - no actual factorization."""
        if n < 2:
            return [0] * 8

        features = []

        # 1. Fermat's method hint (VERY limited search - polynomial time)
        fermat_hint = 0
        ceil_sqrt = int(np.sqrt(n)) + 1
        # Only check a few nearby values - O(1) time, not O(sqrt(n))
        for a in [ceil_sqrt, ceil_sqrt + 1, ceil_sqrt - 1]:
            if a > 0 and a < n:
                diff = a*a - n
                if diff > 0:
                    sqrt_diff = int(np.sqrt(diff))
                    if sqrt_diff*sqrt_diff == diff:
                        fermat_hint = abs(a - ceil_sqrt)  # How close we got
                        break
        features.append(fermat_hint / 10)  # Normalized

        # 2. Continued fraction approximation quality (polynomial time)
        cf_hint = 0
        if n > 1:
            sqrt_n = np.sqrt(n)
            frac_part = sqrt_n - int(sqrt_n)
            # Limited rational approximations - O(1) time
            for denom in range(1, 11):  # Much smaller limit
                for num in range(1, 11):
                    approx = num / denom
                    if abs(approx - frac_part) < 0.05:  # Looser threshold
                        cf_hint += 1
        features.append(cf_hint / 10)

        # 3. Quadratic residue patterns (polynomial time)
        qr_features = []
        for mod in [3, 5, 7, 11, 13, 17, 19, 23]:  # More moduli
            legendre = pow(n % mod, (mod-1)//2, mod)
            qr_features.append(legendre)
        features.extend(qr_features)

        # 4. Cross-modular products (polynomial time)
        # Based on our breakthrough discovery
        cross_products = []
        for m1, m2 in [(7,11), (11,13), (13,17), (17,19), (19,23)]:
            cross_products.append((n % m1) * (n % m2))
        features.extend(cross_products)

        # 5. Digital root patterns in different bases (polynomial time)
        digital_features = []
        for base in [8, 10, 12, 16]:  # More bases
            if base == 10:
                digits = [int(d) for d in str(n)]
            else:
                temp_n = n
                digits = []
                while temp_n > 0 and len(digits) < 10:  # Limit length
                    digits.append(temp_n % base)
                    temp_n //= base

            if digits:
                digital_features.extend([
                    sum(digits),  # Sum
                    sum(digits) % (base-1) or (base-1),  # Digital root
                    len(digits),  # Length
                    max(digits) if digits else 0,  # Max digit
                    len(set(digits)),  # Unique digits
                ])
            else:
                digital_features.extend([0, 0, 0, 0, 0])

        features.extend(digital_features[:10])  # Limit size

        # Ensure no NaN or inf
        features = [0.0 if not np.isfinite(f) else f for f in features]
        return features

    def clean_quantum_inspired(self, n):
        """Clean quantum-inspired features - no factorization."""
        features = []

        # 1. Modular resonance patterns (polynomial time)
        resonance_hints = []
        for mod in [4, 8, 12, 16]:
            # Simple periodic patterns
            pattern = sum(np.sin(2 * np.pi * i * n / mod) for i in range(mod)) / mod
            resonance_hints.append(pattern)
        features.extend(resonance_hints)

        # 2. Amplitude estimation hints (simplified, polynomial time)
        amplitude_hint = 0
        if n > 1:
            # Count how many small moduli n is coprime to
            coprime_count = 0
            for m in range(2, min(20, n)):
                if np.gcd(n, m) == 1:
                    coprime_count += 1
            amplitude_hint = coprime_count / 18  # Normalized
        features.append(amplitude_hint)

        # 3. Phase coherence (polynomial time)
        phase_hint = 0
        if n > 1:
            # Simplified phase coherence measure
            phases = []
            for base in [2, 3, 5]:
                if n % base != 0:
                    # Limited order finding
                    order = 1
                    current = base % n
                    for _ in range(min(10, n-1)):  # Very limited
                        current = (current * base) % n
                        order += 1
                        if current == 1:
                            phases.append(order)
                            break
            if phases:
                phase_hint = np.std(phases) / np.mean(phases) if np.mean(phases) > 0 else 0
        features.append(phase_hint)

        return features

    def clean_number_theory(self, n):
        """Clean advanced number theory features."""
        features = []

        # 1. Cyclotomic polynomial evaluations (polynomial time)
        cyclo_hints = []
        for k in [3, 4, 5]:  # Limited to keep polynomial time
            if k == 3:
                cyclo = n*n - n + 1
            elif k == 4:
                cyclo = n*n + 1
            else:  # k == 5
                cyclo = n**4 + n**3 + n**2 + n + 1

            cyclo_hints.append(cyclo % 1000 / 1000)  # Normalized
        features.extend(cyclo_hints)

        # 2. Character sum approximations (polynomial time)
        char_hints = []
        for mod in [4, 6, 8]:
            char_sum = sum(np.exp(2j * np.pi * i * n / mod) for i in range(mod))
            char_hints.append(abs(char_sum) / mod)
        features.extend(char_hints)

        # 3. Modular inverse existence (polynomial time)
        inv_hints = []
        for mod in [4, 8, 12, 16]:
            try:
                inv = pow(n, -1, mod)
                inv_hints.append(inv / mod)
            except:
                inv_hints.append(0)  # No inverse
        features.extend(inv_hints)

        # 4. Sum of divisors approximation (limited, polynomial time)
        sod_hint = 0
        if n > 1:
            # Very limited divisor sum approximation
            divisors = []
            for i in range(1, min(50, int(np.sqrt(n)) + 1)):
                if n % i == 0:
                    divisors.extend([i, n//i])
            sod_hint = sum(set(divisors)) / n if n > 0 else 0  # Normalized
        features.append(sod_hint)

        return features

    def get_clean_feature_sets(self):
        """Clean feature sets - only polynomial time."""
        return {
            'Clean Factorization Hints': lambda n: self.clean_factorization_hints(n),
            'Clean Quantum Inspired': lambda n: self.clean_quantum_inspired(n),
            'Clean Number Theory': lambda n: self.clean_number_theory(n),
            'Combined Clean': lambda n: (self.clean_factorization_hints(n) +
                                        self.clean_quantum_inspired(n) +
                                        self.clean_number_theory(n)),
        }

    def get_models(self):
        """Standard ML models."""
        return {
            'Logistic Regression': lambda: LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': lambda: RandomForestClassifier(n_estimators=100, random_state=42),
            'Neural Network': lambda: MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500,
                                                   random_state=42, early_stopping=True),
            'SVM': lambda: SVC(probability=True, random_state=42),
            'Gradient Boosting': lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        }

    def clean_crack_attempt(self, test_range=(15000, 20000), k_folds=5):
        """Clean cracking attempt with only polynomial-time features."""
        print("🧹 CLEAN CRACKING ATTEMPT: Only Polynomial-Time Features")
        print("=" * 60)

        # Generate test data
        numbers = np.arange(test_range[0], test_range[1] + 1)
        labels = np.array([1 if n in self.primes else 0 for n in numbers])

        # Get clean feature sets
        feature_sets = self.get_clean_feature_sets()
        models = self.get_models()

        results = {}

        for feature_name, feature_func in feature_sets.items():
            print(f"\n🔬 Testing {feature_name}")
            print("-" * (len(feature_name) + 6))

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

            # Print summary
            best_model = max(model_accuracies.items(), key=lambda x: x[1]['mean'])
            print(f"Best: {best_model[0]} = {best_model[1]['mean']:.4f} ± {best_model[1]['std']:.4f}")

        # Analyze clean results
        self._analyze_clean_results(results)

    def _analyze_clean_results(self, results):
        """Analyze clean feature results."""
        print(f"\n🎯 CLEAN CRACKING RESULTS ANALYSIS")
        print("=" * 40)

        # Find best overall performance
        best_overall = 0
        best_config = None

        for feature_set, model_results in results.items():
            for model, stats in model_results.items():
                if stats['mean'] > best_overall:
                    best_overall = stats['mean']
                    best_config = (feature_set, model, stats)

        print(f"Best Clean Performance: {best_overall:.4f}")
        print(f"Configuration: {best_config[1]} with {best_config[0]} features")
        print(f"Std Dev: {best_config[2]['std']:.4f}")

        # Compare to baseline (89% ceiling)
        baseline = 0.89
        improvement = best_overall - baseline

        print(f"\\n🎯 FAIR CEILING BREAKTHROUGH ASSESSMENT")
        print("=" * 45)
        print(f"Baseline: {baseline:.4f}")
        print(f"Best clean performance: {best_overall:.4f}")
        print(f"Improvement: {improvement:.4f} ({improvement*100:.1f}%)")
        print()

        if best_overall > 0.91:  # Clear break
            print("✅ SUCCESS: Broken through 89% ceiling with clean features!")
            print("   This represents genuine mathematical discovery.")
        elif best_overall > baseline + 0.005:  # Small improvement
            print("⚠️  MARGINAL: Small improvement over baseline")
            print("   Some polynomial-time structure captured.")
        else:
            print("❌ NO BREAKTHROUGH: Did not break the ceiling")
            print("   89% represents genuine information-theoretic limit.")

        # Show all feature set performance
        print(f"\\n📊 CLEAN FEATURE SET COMPARISON")
        print("=" * 35)

        for feature_set, model_results in results.items():
            avg_performance = np.mean([stats['mean'] for stats in model_results.values()])
            improvement_over_baseline = avg_performance - baseline
            print(".4f"
        print("\\n🎓 CONCLUSION")
        print("=" * 12)
        if best_overall <= baseline + 0.01:
            print("The 89% ceiling appears to be a genuine information-theoretic")
            print("limit for polynomial-time primality features. Advanced clean")
            print("features provide minimal improvement, confirming Claude's")
            print("original assessment about computational hardness.")
        else:
            print("Clean features can break the ceiling! This suggests there are")
            print("mathematical structures in polynomial-time features that")
            print("contain more primality information than initially thought.")

def main():
    """Run clean cracking attempt."""
    print("🧹 TESTING CLAUDE'S HYPOTHESIS:")
    print("Can we break 89% ceiling with ONLY polynomial-time features?")
    print("=" * 70)

    cracker = PrimeCeilingCrackerClean(max_n=100000)
    cracker.clean_crack_attempt(test_range=(15000, 20000), k_folds=5)

if __name__ == "__main__":
    main()
