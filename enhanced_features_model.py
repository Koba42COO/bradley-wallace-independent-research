"""
ENHANCED FEATURES MODEL: Adding Primes 13-100 Detection

Based on error analysis showing errors are unbalanced composites (small prime × large prime),
this adds explicit divisibility features for primes 13-100 to catch these systematic errors.

Expected improvement: 93.8% → 95%+
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

class EnhancedFeaturesModel:
    """
    Enhanced model with explicit divisibility checks for primes 13-100
    to catch unbalanced composite errors.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []

    def enhanced_features(self, n):
        """Enhanced feature set with primes 13-100 divisibility checks"""
        if n < 2:
            return [0] * 131  # Will be set properly after first call

        features = []

        # Original features
        n_int = int(n)
        features.extend([n % m for m in [2, 3, 5, 7, 11, 13, 17, 19, 23]])

        # Cross-modular products (original breakthrough)
        cross_products = []
        for m1, m2 in [(7,11), (11,13), (13,17), (17,19), (19,23), (23,29)]:
            cross_products.append((n_int % m1) * (n_int % m2))
        features.extend(cross_products)

        # Quadratic residues (original)
        qr_features = []
        for mod in [3, 5, 7, 11, 13, 17, 19, 23]:
            n_mod = int(n) % mod
            legendre = pow(n_mod, (mod-1)//2, mod)
            qr_features.append(legendre)
        features.extend(qr_features)

        # Digital properties (original)
        digits = [int(d) for d in str(n)]
        if digits:
            features.extend([sum(digits), sum(digits) % 9 or 9, len(digits), max(digits), len(set(digits))])

        # Character sums (original)
        char_features = []
        n_int = int(n)
        for mod in [4, 6, 8]:
            char_sum = sum(np.exp(2j * np.pi * i * n_int / mod) for i in range(mod))
            char_features.append(abs(char_sum) / mod)
        features.extend(char_features)

        # NEW: Extended prime divisibility features (13-100)
        extended_primes = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

        for p in extended_primes:
            features.append(n_int % p)  # Modular residue
            features.append(1 if n_int % p == 0 else 0)  # Binary divisibility flag

        return [0.0 if not np.isfinite(f) else f for f in features]

    def train_enhanced_model(self):
        """Train model with enhanced features"""
        print("🏆 TRAINING ENHANCED MODEL WITH PRIMES 13-100 FEATURES")
        print("=" * 60)

        # Load data (same as before)
        test_range = np.arange(15000, 20000)
        numbers = test_range

        # Generate primes
        primes = set()
        sieve = [True] * 20001
        sieve[0] = sieve[1] = False
        for i in range(2, int(np.sqrt(20000)) + 1):
            if sieve[i]:
                for j in range(i*i, 20001, i):
                    sieve[j] = False
        for i in range(2, 20001):
            if sieve[i]:
                primes.add(i)

        labels = np.array([1 if n in primes else 0 for n in numbers])

        # Generate enhanced features
        print("Generating enhanced features (includes primes 13-100)...")
        features = np.array([self.enhanced_features(n) for n in numbers])

        feature_count = features.shape[1]
        print(f"Enhanced feature count: {feature_count} (original 31 + 40 new = 71)")

        # Build feature names
        self.feature_names = (
            [f'mod_{m}' for m in [2, 3, 5, 7, 11, 13, 17, 19, 23]] +
            [f'xmod_{m1}_{m2}' for m1, m2 in [(7,11), (11,13), (13,17), (17,19), (19,23), (23,29)]] +
            [f'qr_mod_{m}' for m in [3, 5, 7, 11, 13, 17, 19, 23]] +
            ['sum_digits', 'digit_root', 'num_digits', 'max_digit', 'unique_digits'] +
            [f'char_sum_mod_{m}' for m in [4, 6, 8]] +
            [f'mod_{p}' for p in [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]] +
            [f'div_{p}' for p in [13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]]
        )

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("\\nTraining Random Forest with enhanced features...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train_scaled, y_train)

        # Evaluate
        train_acc = accuracy_score(y_train, self.model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, self.model.predict(X_test_scaled))

        print(f"\\nRESULTS:")
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Test accuracy: {test_acc:.4f} (+{test_acc - 0.938:.1f} vs original 93.8%)")

        # Confusion matrix
        y_pred = self.model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\\nConfusion Matrix:")
        print(f"True Positives (correct primes): {tp}")
        print(f"True Negatives (correct composites): {tn}")
        print(f"False Positives (composites → primes): {fp} (was 78)")
        print(f"False Negatives (primes → composites): {fn} (was 15)")

        improvement = test_acc - 0.938
        if improvement > 0.01:
            print(f"\\n✅ SUCCESS: Significant improvement! {improvement:.1f} accuracy gain")
            print("The primes 13-100 features caught the unbalanced composite errors!")
        elif improvement > 0:
            print(f"\\n⚠️  Marginal improvement: {improvement:.3f} accuracy gain")
        else:
            print("\\n❌ No improvement - different approach needed")

        # Save model
        joblib.dump(self.model, '/Users/coo-koba42/dev/enhanced_rf_model.pkl')
        joblib.dump(self.scaler, '/Users/coo-koba42/dev/enhanced_scaler.pkl')
        print("\\n💾 Model saved as enhanced_rf_model.pkl")

        return test_acc

    def analyze_enhanced_features(self):
        """Analyze which new features are most important"""
        if self.model is None:
            print("Model not trained yet")
            return

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]

        print("\\n🎯 TOP 10 FEATURES IN ENHANCED MODEL")
        print("=" * 45)

        # Focus on the new features (primes 13-100)
        new_feature_start = 31  # After original 31 features
        new_importances = importances[new_feature_start:]
        new_names = self.feature_names[new_feature_start:]

        new_indices = np.argsort(new_importances)[::-1]

        print("New features added (primes 13-100):")
        for i in range(min(10, len(new_indices))):
            idx = new_indices[i]
            name = new_names[idx]
            imp = new_importances[idx]
            print(f"  {i+1}. {name}: {imp:.4f}")

        # Check if any new features made it to global top 10
        global_top_10 = indices[:10]
        new_features_in_top_10 = [i for i in global_top_10 if i >= new_feature_start]

        if new_features_in_top_10:
            print(f"\\n🚀 BREAKTHROUGH: {len(new_features_in_top_10)} new features in global top 10!")
            for idx in new_features_in_top_10:
                name = self.feature_names[idx]
                imp = importances[idx]
                rank = list(indices).index(idx) + 1
                print(f"  Rank {rank}: {name} ({imp:.4f})")
        else:
            print("\\n📊 New features not yet in global top 10, but still contributing")

def main():
    """Run enhanced model training and analysis"""
    print("🔬 ENHANCED FEATURES MODEL: Adding Primes 13-100 Detection")
    print("=" * 65)
    print("Target: Catch unbalanced composites (13×1531, 17×997, etc.)")
    print("Expected improvement: 93.8% → 95%+")
    print()

    model = EnhancedFeaturesModel()
    accuracy = model.train_enhanced_model()
    model.analyze_enhanced_features()

    print("\\n" + "="*65)
    if accuracy > 0.95:
        print("🎉 SPECTACULAR SUCCESS! Broke 95% barrier!")
    elif accuracy > 0.948:
        print("✅ MAJOR SUCCESS! Significant improvement achieved!")
    elif accuracy > 0.938:
        print("⚠️  MODERATE SUCCESS: Some improvement")
    else:
        print("❌ NO IMPROVEMENT: Different strategy needed")

    print(".1f")
    print("\\nNext: Test on unseen ranges and deploy to API")

if __name__ == "__main__":
    main()
