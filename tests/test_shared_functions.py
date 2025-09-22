#!/usr/bin/env python3
"""
Tests for shared_functions module
"""

import pytest
import numpy as np
import math
from shared_functions import (
    wallace_transform, wallace_transform_vectorized,
    consciousness_multiplier, phi_optimization_factor,
    calculate_energy_efficiency, fractal_energy_reduction,
    prime_distribution_energy_factor, optimal_gpu_batch_size,
    gpu_memory_efficiency_score, safe_divide, normalize_array,
    format_bytes, validate_numeric_range
)


class TestWallaceTransform:
    """Test Wallace Transform functions"""

    def test_wallace_transform_basic(self):
        """Test basic Wallace transform"""
        result = wallace_transform(2.0)
        assert isinstance(result, float)
        assert result > 0

    def test_wallace_transform_with_params(self):
        """Test Wallace transform with custom parameters"""
        result = wallace_transform(2.0, alpha=1.5, beta=0.5)
        assert isinstance(result, float)

    def test_wallace_transform_edge_cases(self):
        """Test Wallace transform edge cases"""
        # Very small input
        result = wallace_transform(1e-10)
        assert isinstance(result, float)

        # Large input
        result = wallace_transform(1000.0)
        assert isinstance(result, float)

    def test_wallace_transform_vectorized(self):
        """Test vectorized Wallace transform"""
        x_array = np.array([1.0, 2.0, 3.0, 4.0])
        result = wallace_transform_vectorized(x_array)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x_array)
        assert all(r > 0 for r in result)


class TestConsciousnessFunctions:
    """Test prime aligned compute-related functions"""

    def test_consciousness_multiplier(self):
        """Test prime aligned compute multiplier"""
        result = consciousness_multiplier(5.5)
        expected = 5.5 * ((1 + math.sqrt(5)) / 2)
        assert abs(result - expected) < 1e-10

    def test_phi_optimization_factor(self):
        """Test PHI optimization factor"""
        result = phi_optimization_factor(2.0, 5.5)
        assert isinstance(result, float)
        assert result > 0


class TestEnergyOptimization:
    """Test energy optimization functions"""

    def test_calculate_energy_efficiency(self):
        """Test energy efficiency calculation"""
        result = calculate_energy_efficiency(50.0, 0.8)
        assert 0 <= result <= 1

    def test_fractal_energy_reduction(self):
        """Test fractal energy reduction"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = fractal_energy_reduction(data)
        assert 0 < result <= 1

    def test_fractal_energy_reduction_empty(self):
        """Test fractal energy reduction with empty data"""
        data = np.array([])
        result = fractal_energy_reduction(data)
        assert result == 1.0

    def test_prime_distribution_energy_factor(self):
        """Test prime distribution energy factor"""
        result = prime_distribution_energy_factor(12)  # 12 = 2^2 * 3
        assert 0 < result <= 1


class TestGPUOptimization:
    """Test GPU optimization functions"""

    def test_optimal_gpu_batch_size(self):
        """Test optimal GPU batch size calculation"""
        result = optimal_gpu_batch_size(8_000_000, 1000)  # 8MB memory, 1KB per item
        assert isinstance(result, int)
        assert result > 0

    def test_optimal_gpu_batch_size_zero_item_size(self):
        """Test optimal GPU batch size with zero item size"""
        result = optimal_gpu_batch_size(8_000_000, 0)
        assert result == 1

    def test_gpu_memory_efficiency_score(self):
        """Test GPU memory efficiency score"""
        result = gpu_memory_efficiency_score(6_000_000, 8_000_000)  # 75% utilization
        assert 0 <= result <= 1

    def test_gpu_memory_efficiency_score_zero_total(self):
        """Test GPU memory efficiency with zero total memory"""
        result = gpu_memory_efficiency_score(1000, 0)
        assert result == 0


class TestDataProcessing:
    """Test data processing functions"""

    def test_safe_divide(self):
        """Test safe divide function"""
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0
        assert safe_divide(10, 0, 42) == 42

    def test_normalize_array_minmax(self):
        """Test array normalization with minmax method"""
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr, 'minmax')

        assert len(result) == len(arr)
        assert result.min() == 0.0
        assert result.max() == 1.0

    def test_normalize_array_zscore(self):
        """Test array normalization with zscore method"""
        arr = np.array([1, 2, 3, 4, 5])
        result = normalize_array(arr, 'zscore')

        assert len(result) == len(arr)
        assert abs(result.mean()) < 1e-10  # Should be approximately 0

    def test_normalize_array_single_value(self):
        """Test array normalization with single value"""
        arr = np.array([5, 5, 5])
        result = normalize_array(arr, 'minmax')

        assert len(result) == len(arr)
        assert all(r == 0.5 for r in result)  # All values become 0.5

    def test_normalize_array_invalid_method(self):
        """Test array normalization with invalid method"""
        arr = np.array([1, 2, 3])
        with pytest.raises(ValueError):
            normalize_array(arr, 'invalid')


class TestUtilityFunctions:
    """Test utility functions"""

    @pytest.mark.parametrize("size,expected", [
        (0, "0B"),
        (1023, "1023B"),
        (1024, "1KB"),
        (1536, "1.5KB"),
        (1048576, "1MB"),
        (1073741824, "1GB"),
    ])
    def test_format_bytes(self, size, expected):
        """Test byte formatting"""
        result = format_bytes(size)
        assert result == expected

    @pytest.mark.parametrize("value,min_val,max_val,expected", [
        (5, 0, 10, True),
        (0, 0, 10, True),
        (10, 0, 10, True),
        (-1, 0, 10, False),
        (11, 0, 10, False),
    ])
    def test_validate_numeric_range(self, value, min_val, max_val, expected):
        """Test numeric range validation"""
        assert validate_numeric_range(value, min_val, max_val) == expected


if __name__ == "__main__":
    pytest.main([__file__])
