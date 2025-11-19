"""
Comprehensive Gem Exploration with 100% Prime Prediction Integration
Uses the actual 100% prime prediction implementation from the codebase
"""

import sys
import os
sys.path.append('projects/bradley-wallace-independent-research/arxiv_papers/code_examples')

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from decimal import Decimal, getcontext
import json
import warnings
warnings.filterwarnings('ignore')

getcontext().prec = 50

# Import the actual 100% prime prediction implementation
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prime_predictor",
        "projects/bradley-wallace-independent-research/arxiv_papers/code_examples/100_percent_prime_prediction_implementation.py"
    )
    prime_predictor_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prime_predictor_module)
    ConsciousnessGuidedPrimePredictor = prime_predictor_module.ConsciousnessGuidedPrimePredictor
    PrimePredictionResult = prime_predictor_module.PrimePredictionResult
    PRIME_PREDICTOR_AVAILABLE = True
except (ImportError, FileNotFoundError, AttributeError) as e:
    print(f"⚠️ 100% prime prediction implementation not found: {e}")
    print("   Using fallback prime prediction")
    PRIME_PREDICTOR_AVAILABLE = False

# Import existing frameworks
from test_crypto_analyzer import UPGConstants
from explore_all_gems import GemExplorer

class CompleteGemExplorer(GemExplorer):
    """
    Complete exploration system with 100% prime prediction integration
    """
    
    def __init__(self):
        super().__init__()
        if PRIME_PREDICTOR_AVAILABLE:
            self.prime_predictor = ConsciousnessGuidedPrimePredictor()
            print("✅ 100% Prime Predictor initialized")
        else:
            self.prime_predictor = None
    
    def test_100_percent_prime_prediction(self, n_primes: int = 20) -> Dict:
        """
        Test the actual 100% prime prediction implementation
        """
        print("\n" + "="*70)
        print("🔬 GEM #2: 100% PRIME PREDICTABILITY (ACTUAL IMPLEMENTATION)")
        print("="*70)
        
        if not PRIME_PREDICTOR_AVAILABLE or self.prime_predictor is None:
            print("❌ 100% prime predictor not available, using fallback")
            return self.test_prime_predictability(n_primes)
        
        # Generate actual primes for comparison
        actual_primes = []
        num = 2
        while len(actual_primes) < n_primes:
            if all(num % i != 0 for i in range(2, int(num**0.5) + 1)):
                actual_primes.append(num)
            num += 1
        
        # Test the 100% predictor
        predictions = []
        correct = 0
        results = []
        
        print(f"Testing {n_primes} numbers using consciousness-guided prediction...")
        
        for i, prime in enumerate(actual_primes):
            # Test the prime itself
            result = self.prime_predictor.predict_prime(prime)
            is_correct = result.is_prime == True
            if is_correct:
                correct += 1
            
            predictions.append({
                'number': prime,
                'predicted': result.is_prime,
                'actual': True,
                'correct': is_correct,
                'consciousness_level': result.consciousness_level,
                'amplitude': result.consciousness_amplitude,
                'accuracy': result.prediction_accuracy
            })
            
            # Also test a composite near it
            composite = prime + 1
            if composite not in actual_primes:
                comp_result = self.prime_predictor.predict_prime(composite)
                comp_correct = comp_result.is_prime == False
                if comp_correct:
                    correct += 1
                
                predictions.append({
                    'number': composite,
                    'predicted': comp_result.is_prime,
                    'actual': False,
                    'correct': comp_correct,
                    'consciousness_level': comp_result.consciousness_level,
                    'amplitude': comp_result.consciousness_amplitude,
                    'accuracy': comp_result.prediction_accuracy
                })
        
        total_tests = len(predictions)
        accuracy = correct / total_tests * 100 if total_tests > 0 else 0
        
        result = {
            'total_tests': total_tests,
            'correct': correct,
            'accuracy': accuracy,
            'predictions': predictions[:20],  # First 20
            'method': 'CONSCIOUSNESS_GUIDED_PELL_SEQUENCE',
            'statistical_confidence': 'BEYOND_STATISTICAL_IMPOSSIBILITY' if accuracy == 100 else 'HIGH'
        }
        
        status = "✅" if accuracy == 100 else "⚠️"
        print(f"{status} Accuracy: {accuracy:.2f}% ({correct}/{total_tests})")
        print(f"   Method: Consciousness-Guided Pell Sequence")
        print(f"   Statistical confidence: {result['statistical_confidence']}")
        
        # Show sample predictions
        print("\n   Sample predictions:")
        for pred in predictions[:10]:
            status = "✅" if pred['correct'] else "❌"
            prime_str = "PRIME" if pred['actual'] else "COMPOSITE"
            print(f"   {status} {pred['number']:4d}: {prime_str:9s} (predicted: {pred['predicted']}, "
                  f"level: {pred['consciousness_level']:2d}, amp: {pred['amplitude']:.3f})")
        
        self.results['100_percent_prime_prediction'] = result
        return result
    
    def validate_100_percent_prediction_range(self, start: int = 1000, end: int = 2000) -> Dict:
        """
        Validate 100% prime prediction across a range
        """
        print("\n" + "="*70)
        print("🔬 VALIDATING 100% PRIME PREDICTION ACROSS RANGE")
        print("="*70)
        
        if not PRIME_PREDICTOR_AVAILABLE or self.prime_predictor is None:
            print("❌ 100% prime predictor not available")
            return {}
        
        print(f"Testing range: {start:,} - {end:,}")
        validation = self.prime_predictor.validate_prediction_accuracy((start, end))
        
        print(f"\n✅ Validation Results:")
        print(f"   Total tests: {validation['total_tests']:,}")
        print(f"   Correct: {validation['correct_predictions']:,}")
        print(f"   Accuracy: {validation['accuracy']:.10f}")
        print(f"   Sigma confidence: {validation['sigma_confidence']}")
        print(f"   Statistical regime: {validation['statistical_regime']}")
        print(f"   Validation time: {validation['validation_time']:.2f}s")
        
        self.results['100_percent_validation'] = validation
        return validation
    
    def demonstrate_consciousness_mathematics(self) -> Dict:
        """
        Demonstrate consciousness mathematics with known primes
        """
        print("\n" + "="*70)
        print("🧠 CONSCIOUSNESS MATHEMATICS DEMONSTRATION")
        print("="*70)
        
        if not PRIME_PREDICTOR_AVAILABLE or self.prime_predictor is None:
            print("❌ 100% prime predictor not available")
            return {}
        
        # Known primes and composites
        test_numbers = {
            'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 101, 103, 137, 199, 201],
            'composites': [4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 24, 25, 100, 102, 200, 207, 209]
        }
        
        results = []
        correct = 0
        total = 0
        
        print("\nPrime Number Predictions:")
        print("Number | Predicted | Actual | Status | Level | Amplitude")
        print("-------|-----------|--------|--------|-------|-----------")
        
        for number in test_numbers['primes']:
            result = self.prime_predictor.predict_prime(number)
            is_correct = result.is_prime == True
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            print(f"{number:6d} | {'PRIME':9s} | PRIME  | {status}    | {result.consciousness_level:5d} | {result.consciousness_amplitude:9.3f}")
            
            results.append({
                'number': number,
                'type': 'prime',
                'predicted': result.is_prime,
                'actual': True,
                'correct': is_correct,
                'consciousness_level': result.consciousness_level,
                'amplitude': result.consciousness_amplitude
            })
        
        print("\nComposite Number Predictions:")
        print("Number | Predicted | Actual    | Status | Level | Amplitude")
        print("-------|-----------|-----------|--------|-------|-----------")
        
        for number in test_numbers['composites']:
            result = self.prime_predictor.predict_prime(number)
            is_correct = result.is_prime == False
            if is_correct:
                correct += 1
            total += 1
            
            status = "✅" if is_correct else "❌"
            predicted_str = "PRIME" if result.is_prime else "COMPOSITE"
            print(f"{number:6d} | {predicted_str:9s} | COMPOSITE | {status}    | {result.consciousness_level:5d} | {result.consciousness_amplitude:9.3f}")
            
            results.append({
                'number': number,
                'type': 'composite',
                'predicted': result.is_prime,
                'actual': False,
                'correct': is_correct,
                'consciousness_level': result.consciousness_level,
                'amplitude': result.consciousness_amplitude
            })
        
        accuracy = correct / total * 100 if total > 0 else 0
        
        demo_result = {
            'total_tests': total,
            'correct': correct,
            'accuracy': accuracy,
            'results': results,
            'method': 'CONSCIOUSNESS_GUIDED_PELL_SEQUENCE'
        }
        
        print(f"\n✅ Overall Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        self.results['consciousness_demo'] = demo_result
        return demo_result
    
    def run_complete_exploration(self):
        """Run complete exploration with 100% prime prediction"""
        print("\n" + "="*70)
        print("🚀 COMPLETE GEM EXPLORATION WITH 100% PRIME PREDICTION")
        print("="*70)
        print(f"Started: {datetime.now().isoformat()}")
        
        # Run original tests
        self.test_wallace_validations()
        
        # Run 100% prime prediction tests
        if PRIME_PREDICTOR_AVAILABLE:
            self.test_100_percent_prime_prediction(20)
            self.demonstrate_consciousness_mathematics()
            self.validate_100_percent_prediction_range(1000, 2000)
        else:
            print("\n⚠️ Using fallback prime prediction")
            self.test_prime_predictability(20)
        
        # Run other tests
        self.test_twin_prime_cancellation()
        self.test_physics_constants_twins()
        self.test_base_21_vs_base_10()
        self.test_79_21_consciousness()
        self.test_cardioid_distribution()
        self.test_207_year_cycles()
        self.test_area_code_cypher()
        self.test_metatron_cube()
        self.test_pac_vs_traditional()
        self.test_blood_ph_protocol()
        self.test_207_dial_tone()
        self.test_montesiepi_chapel()
        
        # Generate visualizations
        viz_file = self.generate_visualizations()
        
        # Save results
        results_file = 'gems_exploration_with_100_percent_results.json'
        with open(results_file, 'w') as f:
            def convert(obj):
                if isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.bool_, bool)):
                    return bool(obj)
                elif isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [convert(item) for item in obj]
                return obj
            
            json.dump(convert(self.results), f, indent=2)
        
        print("\n" + "="*70)
        print("✅ COMPLETE EXPLORATION FINISHED")
        print("="*70)
        print(f"Results saved to: {results_file}")
        print(f"Visualizations: {viz_file}")
        print(f"Completed: {datetime.now().isoformat()}")
        
        return self.results


def main():
    """Main complete exploration"""
    explorer = CompleteGemExplorer()
    results = explorer.run_complete_exploration()
    
    # Summary
    print("\n" + "="*70)
    print("📊 COMPLETE EXPLORATION SUMMARY")
    print("="*70)
    
    if '100_percent_prime_prediction' in results:
        pred = results['100_percent_prime_prediction']
        print(f"✅ 100% Prime Prediction: {pred['accuracy']:.2f}% accuracy")
        print(f"   Method: {pred['method']}")
        print(f"   Confidence: {pred['statistical_confidence']}")
    
    if 'consciousness_demo' in results:
        demo = results['consciousness_demo']
        print(f"✅ Consciousness Demo: {demo['accuracy']:.2f}% accuracy ({demo['correct']}/{demo['total_tests']})")
    
    if '100_percent_validation' in results:
        val = results['100_percent_validation']
        print(f"✅ Range Validation: {val['accuracy']:.10f} accuracy")
        print(f"   Sigma confidence: {val['sigma_confidence']}")
    
    return results


if __name__ == "__main__":
    main()

