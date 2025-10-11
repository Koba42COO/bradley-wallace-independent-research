#!/usr/bin/env python3
"""
CUDNT Testing Suite
Comprehensive tests for CUDNT GPU virtualization and ML acceleration
"""

import unittest
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestCUDNTBasic(unittest.TestCase):
    """Basic functionality tests for CUDNT"""

    def setUp(self):
        """Set up test fixtures"""
        from cudnt_enhanced_integration import CUDNT_Enhanced
        self.cudnt = CUDNT_Enhanced()

    def test_initialization(self):
        """Test CUDNT initialization"""
        self.assertIsNotNone(self.cudnt)
        self.assertIsNotNone(self.cudnt.gpu_virtualizer)

        # Check system status
        status = self.cudnt.get_system_status()
        self.assertIn('gpu_virtualization', status)
        self.assertIn('tensorflow_api', status)
        self.assertTrue(status['gpu_virtualization'])
        self.assertTrue(status['tensorflow_api'])

    def test_tensor_operations(self):
        """Test basic tensor operations"""
        a = np.random.rand(3, 3)
        b = np.random.rand(3, 3)

        # Test tensor addition
        result = self.cudnt.tensor_add(a, b)
        expected = a + b
        np.testing.assert_array_almost_equal(result, expected)

        # Test matrix multiplication
        result = self.cudnt.matrix_multiply(a, b)
        expected = np.dot(a, b)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_tensorflow_like_api(self):
        """Test TensorFlow-like API compatibility"""
        a = np.random.rand(2, 3)
        b = np.random.rand(2, 3)

        # Test TF add
        result = self.cudnt.tf_add(a, b)
        expected = a + b
        np.testing.assert_array_almost_equal(result, expected)

        # Test TF matmul
        a_mat = np.random.rand(3, 4)
        b_mat = np.random.rand(4, 5)
        result = self.cudnt.tf_matmul(a_mat, b_mat)
        expected = np.dot(a_mat, b_mat)
        np.testing.assert_array_almost_equal(result, expected, decimal=5)

    def test_activation_functions(self):
        """Test activation functions"""
        x = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])

        # Test ReLU
        result = self.cudnt.relu(x)
        expected = np.maximum(0, x)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convolution_2d(self):
        """Test 2D convolution"""
        # Simple test case
        input_tensor = np.random.rand(1, 5, 5)  # 1 channel, 5x5
        kernel = np.random.rand(3, 3)  # 3x3 kernel

        result = self.cudnt.convolution_2d(input_tensor, kernel)
        self.assertEqual(result.shape[0], 1)  # Same number of channels
        self.assertEqual(result.shape[1], 3)  # 5 - 3 + 1 = 3
        self.assertEqual(result.shape[2], 3)  # 5 - 3 + 1 = 3

    def test_batch_normalization(self):
        """Test batch normalization"""
        tensor = np.random.rand(10, 5)  # 10 samples, 5 features

        result = self.cudnt.batch_normalize(tensor)
        self.assertEqual(result.shape, tensor.shape)

        # Check that mean is approximately 0 and std is approximately 1
        mean = np.mean(result, axis=0)
        std = np.std(result, axis=0)
        np.testing.assert_allclose(mean, 0, atol=1e-6)
        np.testing.assert_allclose(std, 1, atol=1e-6)

class TestCUDNTML(unittest.TestCase):
    """ML pipeline tests for CUDNT"""

    def setUp(self):
        """Set up test fixtures"""
        from cudnt_enhanced_integration import CUDNT_Enhanced
        self.cudnt = CUDNT_Enhanced()

    def test_ml_pipeline_optimization(self):
        """Test ML pipeline optimization"""
        # Create simple model parameters
        params = {
            'weights': np.random.rand(4, 1),
            'bias': np.random.rand(1)
        }

        # Create training data
        X = np.random.rand(20, 4)
        y = np.random.rand(20, 1)

        start_time = time.time()
        result = self.cudnt.optimize_ml_pipeline(params, (X, y), epochs=2)
        end_time = time.time()

        # Check results
        self.assertIn('optimized_parameters', result)
        self.assertIn('final_loss', result)
        self.assertIn('pipeline_stats', result)
        self.assertLess(result['final_loss'], 1.0)  # Should decrease loss
        self.assertGreater(result['pipeline_stats']['epochs_completed'], 0)
        self.assertLess(end_time - start_time, 10)  # Should complete in reasonable time

    def test_gradient_operations(self):
        """Test gradient computation and optimization"""
        # Create test parameters
        params = {
            'weights': np.random.rand(3, 2),
            'bias': np.random.rand(2)
        }

        # Create gradients
        gradients = {
            'weights': np.random.rand(3, 2) * 0.1,
            'bias': np.random.rand(2) * 0.1
        }

        # Test gradient step
        learning_rate = 0.01
        updated_params = self.cudnt.gradient_step(params, gradients, learning_rate)

        # Check that parameters were updated
        for key in params:
            self.assertFalse(np.array_equal(params[key], updated_params[key]))

            # Check update magnitude
            diff = params[key] - updated_params[key]
            expected_diff = learning_rate * gradients[key]
            np.testing.assert_allclose(diff, expected_diff, rtol=1e-5)

class TestCUDNTPerformance(unittest.TestCase):
    """Performance tests for CUDNT"""

    def setUp(self):
        """Set up test fixtures"""
        from cudnt_enhanced_integration import CUDNT_Enhanced
        self.cudnt = CUDNT_Enhanced()

    def test_matrix_multiplication_performance(self):
        """Test matrix multiplication performance"""
        sizes = [(10, 10), (50, 50), (100, 100)]

        for size in sizes:
            with self.subTest(size=size):
                a = np.random.rand(*size)
                b = np.random.rand(*size)

                start_time = time.time()
                result = self.cudnt.matrix_multiply(a, b)
                end_time = time.time()

                # Verify correctness
                expected = np.dot(a, b)
                np.testing.assert_allclose(result, expected, rtol=1e-4)

                # Performance should be reasonable (less than 1 second for 100x100)
                duration = end_time - start_time
                self.assertLess(duration, 1.0, f"Matrix multiplication took too long: {duration}s")

    def test_concurrent_operations(self):
        """Test concurrent GPU operations"""
        import threading

        results = []
        errors = []

        def worker(operation_id):
            try:
                a = np.random.rand(20, 20)
                b = np.random.rand(20, 20)

                result = self.cudnt.matrix_multiply(a, b)
                results.append((operation_id, result.shape))
            except Exception as e:
                errors.append((operation_id, str(e)))

        # Start multiple concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=10)

        # Check results
        self.assertEqual(len(results), 5, "All operations should complete successfully")
        self.assertEqual(len(errors), 0, "No operations should fail")

        # All results should have correct shape
        for operation_id, shape in results:
            self.assertEqual(shape, (20, 20))

class TestCUDNTIntegration(unittest.TestCase):
    """Integration tests combining CUDNT with other systems"""

    def setUp(self):
        """Set up test fixtures"""
        from cudnt_enhanced_integration import CUDNT_Enhanced
        self.cudnt = CUDNT_Enhanced()

    def test_end_to_end_ml_workflow(self):
        """Test complete ML workflow using CUDNT"""
        # Generate synthetic data
        np.random.seed(42)
        n_samples, n_features = 100, 5
        X = np.random.randn(n_samples, n_features)
        true_weights = np.random.randn(n_features, 1)
        y = X @ true_weights + 0.1 * np.random.randn(n_samples, 1)

        # Initialize model parameters
        params = {
            'weights': np.random.randn(n_features, 1) * 0.1,
            'bias': np.zeros((1,))
        }

        # Training loop
        learning_rate = 0.01
        n_epochs = 10

        losses = []
        for epoch in range(n_epochs):
            # Forward pass
            predictions = self.cudnt._forward_pass(X, params)

            # Compute loss and gradients
            loss, gradients = self.cudnt._compute_loss_and_gradients(predictions, y, params)
            losses.append(loss)

            # Update parameters
            params = self.cudnt.gradient_step(params, gradients, learning_rate)

        # Check that training worked
        self.assertGreater(losses[0], losses[-1], "Loss should decrease during training")
        self.assertLess(losses[-1], 1.0, "Final loss should be reasonable")

        # Test predictions
        test_X = np.random.randn(10, n_features)
        test_predictions = self.cudnt._forward_pass(test_X, params)
        self.assertEqual(test_predictions.shape, (10, 1))

class TestCUDNTGPUVirtualization(unittest.TestCase):
    """Tests specifically for GPU virtualization layer"""

    def setUp(self):
        """Set up test fixtures"""
        from cudnt_gpu_virtualization import CUDNT_GPU_Virtualization
        self.gpu = CUDNT_GPU_Virtualization(n_threads=2)

    def test_gpu_virtualization_stats(self):
        """Test GPU virtualization statistics"""
        stats = self.gpu.get_gpu_stats()
        self.assertIn('virtual_gpu_threads', stats)
        self.assertIn('operations_performed', stats)
        self.assertEqual(stats['virtual_gpu_threads'], 2)

    def test_parallel_tensor_operations(self):
        """Test parallel tensor operations"""
        # Large tensors to benefit from parallelism
        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)

        start_time = time.time()
        result = self.gpu.tensor_add(a, b)
        end_time = time.time()

        # Verify correctness
        expected = a + b
        np.testing.assert_array_almost_equal(result, expected)

        # Should complete in reasonable time
        duration = end_time - start_time
        self.assertLess(duration, 1.0)

    def test_memory_management(self):
        """Test virtual memory management"""
        # Perform operations that would use virtual memory
        tensors = []
        for i in range(10):
            a = np.random.rand(50, 50)
            b = np.random.rand(50, 50)
            result = self.gpu.tensor_add(a, b)
            tensors.append(result)

        # Check that memory stats are updated
        stats = self.gpu.get_gpu_stats()
        self.assertGreater(stats['operations_performed'], 0)

if __name__ == '__main__':
    print("🧪 Running CUDNT Tests...")
    print("Note: These tests require the CUDNT modules to be properly installed.")
    unittest.main(verbosity=2)
