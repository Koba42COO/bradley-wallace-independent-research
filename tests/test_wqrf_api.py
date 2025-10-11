#!/usr/bin/env python3
"""
WQRF API Testing Suite
Comprehensive tests for Wallace Quantum Resonance Framework API
"""

import unittest
import requests
import json
import time
from unittest.mock import Mock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestWQRFApi(unittest.TestCase):
    """Test cases for WQRF primality testing API"""

    def setUp(self):
        """Set up test fixtures"""
        self.base_url = "http://localhost:5001"
        self.test_timeout = 10

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = requests.get(f"{self.base_url}/health", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('status', data)
        self.assertIn('models_available', data)
        self.assertIn('timestamp', data)

        # Should have at least one model available
        self.assertGreater(len(data['models_available']), 0)

    def test_single_prediction_prime(self):
        """Test single prediction with a prime number"""
        response = requests.get(f"{self.base_url}/predict/17", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data['number'], 17)
        self.assertEqual(data['prediction'], 'prime')
        self.assertGreater(data['confidence'], 0.5)  # Should be confident
        self.assertIn('timestamp', data)

    def test_single_prediction_composite(self):
        """Test single prediction with a composite number"""
        response = requests.get(f"{self.base_url}/predict/15", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(data['number'], 15)
        self.assertEqual(data['prediction'], 'composite')
        self.assertIn('timestamp', data)

    def test_single_prediction_invalid(self):
        """Test single prediction with invalid input"""
        response = requests.get(f"{self.base_url}/predict/1", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 400)

        data = response.json()
        self.assertIn('error', data)
        self.assertIn('Number must be >= 2', data['error'])

    def test_batch_prediction(self):
        """Test batch prediction endpoint"""
        test_data = {
            'numbers': [13, 15, 17, 19, 20],
            'model': 'clean_ml'
        }

        response = requests.post(
            f"{self.base_url}/predict",
            json=test_data,
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 5)
        self.assertIn('batch_size', data)
        self.assertEqual(data['batch_size'], 5)

        # Check specific results
        results = data['results']
        self.assertEqual(results[0]['prediction'], 'prime')   # 13
        self.assertEqual(results[1]['prediction'], 'composite') # 15
        self.assertEqual(results[2]['prediction'], 'prime')   # 17

    def test_batch_prediction_large(self):
        """Test batch prediction with larger dataset"""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        composites = [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

        test_data = {
            'numbers': primes + composites,
            'model': 'clean_ml'
        }

        response = requests.post(
            f"{self.base_url}/predict",
            json=test_data,
            timeout=30  # Longer timeout for larger batch
        )
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertEqual(len(data['results']), 20)

        # Verify accuracy on known primes
        results = data['results']
        prime_predictions = sum(1 for r in results[:10] if r['prediction'] == 'prime')
        composite_predictions = sum(1 for r in results[10:] if r['prediction'] == 'composite')

        # Should have high accuracy
        self.assertGreaterEqual(prime_predictions, 8)  # At least 80% accuracy on primes
        self.assertGreaterEqual(composite_predictions, 8)  # At least 80% accuracy on composites

    def test_info_endpoint(self):
        """Test API information endpoint"""
        response = requests.get(f"{self.base_url}/info", timeout=self.test_timeout)
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn('service', data)
        self.assertIn('version', data)
        self.assertIn('models', data)
        self.assertIn('endpoints', data)
        self.assertIn('limitations', data)

    def test_model_selection(self):
        """Test different model selection"""
        models_to_test = ['clean_ml', 'hybrid_ml']

        for model in models_to_test:
            with self.subTest(model=model):
                test_data = {
                    'numbers': [17, 19],
                    'model': model
                }

                response = requests.post(
                    f"{self.base_url}/predict",
                    json=test_data,
                    timeout=self.test_timeout
                )
                self.assertEqual(response.status_code, 200)

                data = response.json()
                self.assertEqual(data['model'], model)

    def test_error_handling(self):
        """Test error handling scenarios"""
        # Test with invalid JSON
        response = requests.post(
            f"{self.base_url}/predict",
            data="invalid json",
            headers={'Content-Type': 'application/json'},
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 400)

        # Test with empty batch
        test_data = {'numbers': []}
        response = requests.post(
            f"{self.base_url}/predict",
            json=test_data,
            timeout=self.test_timeout
        )
        self.assertEqual(response.status_code, 400)

        # Test with too large batch
        test_data = {'numbers': list(range(2, 200))}  # 198 numbers
        response = requests.post(
            f"{self.base_url}/predict",
            json=test_data,
            timeout=self.test_timeout
        )
        # Should either succeed or return appropriate error
        self.assertIn(response.status_code, [200, 413])

class TestWQRFApiLoad(unittest.TestCase):
    """Load testing for WQRF API"""

    def setUp(self):
        self.base_url = "http://localhost:5001"

    def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        import concurrent.futures
        import threading

        results = []
        errors = []

        def make_request(number):
            try:
                response = requests.get(f"{self.base_url}/predict/{number}", timeout=5)
                results.append((number, response.status_code))
            except Exception as e:
                errors.append((number, str(e)))

        # Test with 10 concurrent requests
        numbers = [13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(make_request, numbers)

        # Should have 10 successful results
        self.assertEqual(len(results), 10)
        successful = sum(1 for _, status in results if status == 200)
        self.assertGreaterEqual(successful, 9)  # At least 90% success rate

        # Minimal errors
        self.assertLessEqual(len(errors), 1)

if __name__ == '__main__':
    # Check if API is running
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code != 200:
            print("❌ WQRF API is not running. Please start it first:")
            print("   python3 deployment_api.py")
            sys.exit(1)
    except:
        print("❌ WQRF API is not running. Please start it first:")
        print("   python3 deployment_api.py")
        sys.exit(1)

    print("🧪 Running WQRF API Tests...")
    unittest.main(verbosity=2)
