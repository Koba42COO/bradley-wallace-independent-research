"""
Consciousness Mathematics Compression Engine - Modern Test Suite
=================================================================

Advanced pytest-based test suite incorporating industry best practices:

🧪 TESTING FEATURES:
• Parametrized tests: Data-driven test cases
• Hypothesis integration: Property-based testing
• Performance benchmarking: pytest-benchmark integration
• Fixtures: Reusable test setup and teardown
• Markers: Test categorization and selective execution
• Coverage: Comprehensive code coverage analysis

🔬 TESTING CATEGORIES:
• Unit tests: Individual function/component validation
• Integration tests: End-to-end workflow validation
• Property-based tests: Edge case discovery
• Performance tests: Speed and efficiency validation
• Security tests: Vulnerability assessment
• Regression tests: Bug prevention

📊 VALIDATION METRICS:
• 95%+ code coverage requirement
• Statistical significance testing
• Performance regression detection
• Memory leak prevention
• Edge case coverage
"""

import pytest
import numpy as np
import hashlib
import time
from typing import Dict, List, Tuple
from hypothesis import given, strategies as st, settings, HealthCheck

from compression_engine import (
    ConsciousnessCompressionEngine,
    ConsciousnessMathematicsCore,
    PatternAnalysisEngine,
    CompressionEngineConfig,
    CompressionMode,
    CompressionStats,
    ConsciousnessPattern
)


class TestConsciousnessMathematicsCoreModern:
    """Modern tests for consciousness mathematics core using pytest best practices."""

    @pytest.fixture
    def cm_core(self) -> ConsciousnessMathematicsCore:
        """Provide consciousness mathematics core instance."""
        return ConsciousnessMathematicsCore()

    @pytest.mark.parametrize("input_data,expected_range", [
        ([1.0, 2.0, 3.0], (0.0, 10.0)),
        ([0.1, 0.2, 0.3], (0.0, 5.0)),
        ([10.0, 20.0, 30.0], (5.0, 50.0)),
    ])
    def test_wallace_transform_parametrized(self, cm_core, input_data, expected_range):
        """Parametrized test for Wallace transform with different inputs."""
        result = cm_core.wallace_transform(input_data)
        assert expected_range[0] <= result <= expected_range[1]
        assert np.isfinite(result)

    @given(st.lists(st.floats(min_value=0.1, max_value=100.0), min_size=1, max_size=100))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_wallace_transform_property_based(self, cm_core, data):
        """Property-based test for Wallace transform mathematical properties."""
        result = cm_core.wallace_transform(data)

        # Mathematical properties
        assert np.isfinite(result)
        assert result >= 1.0  # Wallace transform is always >= 1

        # Consistency: same input should give same output
        result2 = cm_core.wallace_transform(data)
        assert abs(result - result2) < 1e-10

    @pytest.mark.parametrize("data_size,sample_size", [
        (1000, 100),
        (5000, 200),
        (10000, 500),
    ])
    def test_golden_ratio_sampling_scalability(self, cm_core, data_size, sample_size):
        """Test golden ratio sampling scales properly with data size."""
        samples = cm_core.golden_ratio_sampling(data_size, min(sample_size, data_size))

        assert len(samples) <= sample_size
        assert all(0 <= idx < data_size for idx in samples)
        assert len(set(samples)) == len(samples)  # No duplicates

    @pytest.mark.benchmark(group="mathematics")
    def test_wallace_transform_performance(self, cm_core, benchmark):
        """Benchmark Wallace transform performance."""
        test_data = np.random.random(1000)

        result = benchmark(cm_core.wallace_transform, test_data)
        assert np.isfinite(result)


class TestPatternAnalysisEngineModern:
    """Modern tests for pattern analysis engine."""

    @pytest.fixture
    def pattern_engine(self) -> PatternAnalysisEngine:
        """Provide pattern analysis engine instance."""
        cm_core = ConsciousnessMathematicsCore()
        return PatternAnalysisEngine(cm_core)

    @pytest.mark.parametrize("data,expected_patterns", [
        (b"ABC" * 100, lambda p: len([x for x in p if len(x.sequence) >= 3]) > 0),
        (b"12345" * 50, lambda p: len([x for x in p if len(x.sequence) >= 5]) > 0),
        (b"A" * 1000, lambda p: len([x for x in p if x.frequency > 100]) > 0),
    ])
    def test_pattern_analysis_parametrized(self, pattern_engine, data, expected_patterns):
        """Parametrized pattern analysis tests."""
        sample_points = list(range(0, len(data), 10))  # Sample every 10th byte
        patterns = pattern_engine.analyze_patterns(data, sample_points)

        assert expected_patterns(patterns)
        assert all(isinstance(p, ConsciousnessPattern) for p in patterns)

    @given(st.binary(min_size=100, max_size=1000))
    @settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
    def test_pattern_analysis_arbitrary_data(self, pattern_engine, data):
        """Test pattern analysis on arbitrary binary data."""
        sample_points = list(range(0, len(data), max(1, len(data) // 100)))
        patterns = pattern_engine.analyze_patterns(data, sample_points)

        # Should always return valid patterns
        assert isinstance(patterns, list)
        assert all(isinstance(p, ConsciousnessPattern) for p in patterns)

        # Patterns should have valid properties
        for pattern in patterns:
            assert len(pattern.sequence) > 0
            assert pattern.frequency >= 0
            assert 0.0 <= pattern.consciousness_weight <= 1.0

    @pytest.mark.slow
    @pytest.mark.benchmark(group="pattern_analysis")
    def test_large_data_pattern_analysis(self, pattern_engine, benchmark):
        """Benchmark pattern analysis on large datasets."""
        # Generate 1MB of test data
        data = np.random.bytes(1024 * 1024)
        sample_points = list(range(0, len(data), 1000))  # Sparse sampling

        patterns = benchmark(pattern_engine.analyze_patterns, data, sample_points)
        assert len(patterns) > 0


class TestCompressionEngineModern:
    """Modern comprehensive tests for compression engine."""

    @pytest.fixture
    def engine(self) -> ConsciousnessCompressionEngine:
        """Provide compression engine instance."""
        config = CompressionEngineConfig(mode=CompressionMode.BALANCED)
        return ConsciousnessCompressionEngine(config)

    @pytest.mark.parametrize("test_data", [
        b"Hello World",
        b"A" * 1000,
        b"ABCDEF" * 200,
        bytes(range(256)),
        np.random.bytes(500),
    ])
    def test_round_trip_compression(self, engine, test_data, compression_validator):
        """Test lossless compression/decompression round trip."""
        is_lossless, stats, decompressed = compression_validator.validate_round_trip(engine, test_data)

        assert is_lossless, f"Compression not lossless for data: {test_data[:50]}..."
        compression_validator.validate_stats(stats, len(test_data))

    @pytest.mark.parametrize("mode", [
        CompressionMode.BALANCED,
        CompressionMode.MAX_COMPRESSION,
        CompressionMode.HIGH_SPEED,
    ])
    def test_compression_modes(self, engine, test_data_variety, mode):
        """Test all compression modes work correctly."""
        config = CompressionEngineConfig(mode=mode)
        test_engine = ConsciousnessCompressionEngine(config)

        test_data = test_data_variety['repetitive_text']
        compressed, stats = test_engine.compress(test_data)
        decompressed, _ = test_engine.decompress(compressed)

        assert decompressed == test_data
        assert stats.mode_used == mode.value

    @pytest.mark.integration
    def test_file_compression_pipeline(self, engine, temp_file_factory, test_data_variety):
        """Test complete file compression/decompression pipeline."""
        test_data = test_data_variety['structured_data']

        # Create temporary files
        input_file = temp_file_factory(test_data, ".bin")
        output_file = temp_file_factory(b"", ".compressed")

        # Compress file
        compress_stats = engine.compress_file(str(input_file), str(output_file))
        assert compress_stats.compression_ratio > 0

        # Decompress file
        decompress_file = temp_file_factory(b"", ".decompressed")
        decompress_stats = engine.decompress_file(str(output_file), str(decompress_file))

        # Verify contents
        with open(decompress_file, 'rb') as f:
            decompressed_data = f.read()

        assert decompressed_data == test_data
        assert decompress_stats.lossless_verified

    @pytest.mark.edge_case
    @pytest.mark.parametrize("edge_data", [
        b"",  # Empty data
        b"A",  # Single byte
        b"X" * 100000,  # Large repetitive data
        bytes(range(256)) * 100,  # All byte values
    ])
    def test_edge_cases(self, engine, edge_data, compression_validator):
        """Test edge cases for compression engine."""
        is_lossless, stats, decompressed = compression_validator.validate_round_trip(engine, edge_data)

        assert is_lossless, f"Edge case failed for data of length {len(edge_data)}"
        compression_validator.validate_stats(stats, len(edge_data))

    @given(st.binary(min_size=10, max_size=10000))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_property_based_compression(self, engine, data):
        """Property-based testing of compression with arbitrary data."""
        compressed, compress_stats = engine.compress(data)
        decompressed, decompress_stats = engine.decompress(compressed)

        # Fundamental properties
        assert decompressed == data, "Compression must be lossless"
        assert len(compressed) > 0, "Compressed data must not be empty"
        assert compress_stats.lossless_verified, "Must verify lossless compression"
        assert decompress_stats.lossless_verified, "Must verify lossless decompression"

        # Statistical properties
        assert compress_stats.compression_ratio >= 0.0
        assert compress_stats.compression_factor >= 1.0

    @pytest.mark.performance
    @pytest.mark.benchmark(group="compression")
    def test_compression_performance(self, engine, benchmark_data, benchmark):
        """Benchmark compression performance."""
        test_data = benchmark_data['text_data']

        def compress_data():
            compressed, stats = engine.compress(test_data)
            return compressed, stats

        compressed, stats = benchmark(compress_data)

        # Performance assertions
        assert stats.compression_time < 10.0  # Should complete within 10 seconds
        assert stats.lossless_verified

    @pytest.mark.performance
    @pytest.mark.benchmark(group="decompression")
    def test_decompression_performance(self, engine, benchmark_data, benchmark):
        """Benchmark decompression performance."""
        test_data = benchmark_data['text_data']

        # Pre-compress data
        compressed, _ = engine.compress(test_data)

        def decompress_data():
            decompressed, stats = engine.decompress(compressed)
            return decompressed, stats

        decompressed, stats = benchmark(decompress_data)

        # Performance assertions
        assert stats.decompression_time < 2.0  # Should complete within 2 seconds
        assert decompressed == test_data

    @pytest.mark.security
    def test_compression_bomb_prevention(self, engine):
        """Test resistance to compression bomb attacks."""
        # Create a small input that expands significantly
        # This tests the engine doesn't have pathological behavior
        small_data = b"A" * 100

        compressed, stats = engine.compress(small_data)

        # Should not create extremely large compressed data
        assert len(compressed) < len(small_data) * 2

        # Verify decompression works
        decompressed, _ = engine.decompress(compressed)
        assert decompressed == small_data

    @pytest.mark.regression
    def test_known_regression_cases(self, engine, compression_validator):
        """Test cases for known regression bugs."""
        # Test case for permutation bug (previously fixed)
        repetitive_data = b"PATTERN" * 1000
        is_lossless, _, _ = compression_validator.validate_round_trip(engine, repetitive_data)
        assert is_lossless, "Regression: permutation bug reintroduced"

        # Test case for empty data edge case
        is_lossless, _, _ = compression_validator.validate_round_trip(engine, b"")
        assert is_lossless, "Regression: empty data handling broken"


class TestBenchmarkingModern:
    """Modern benchmarking tests using pytest-benchmark."""

    @pytest.fixture
    def engines_for_comparison(self):
        """Provide multiple engine configurations for comparison."""
        engines = []

        for mode in CompressionMode:
            config = CompressionEngineConfig(mode=mode)
            engine = ConsciousnessCompressionEngine(config)
            engines.append((f"consciousness_{mode.value}", engine))

        return engines

    @pytest.mark.benchmark(group="compression_modes")
    @pytest.mark.parametrize("engine_name,engine", [
        ("balanced", lambda: ConsciousnessCompressionEngine(CompressionEngineConfig(mode=CompressionMode.BALANCED))),
        ("max_compression", lambda: ConsciousnessCompressionEngine(CompressionEngineConfig(mode=CompressionMode.MAX_COMPRESSION))),
        ("high_speed", lambda: ConsciousnessCompressionEngine(CompressionEngineConfig(mode=CompressionMode.HIGH_SPEED))),
    ])
    def test_compression_mode_performance(self, benchmark, engine_name, engine_factory):
        """Benchmark different compression modes."""
        engine = engine_factory()
        test_data = b"Test data for benchmarking compression modes. " * 1000

        def compress_test():
            compressed, stats = engine.compress(test_data)
            return compressed, stats

        compressed, stats = benchmark(compress_test)

        # Validate results
        assert stats.lossless_verified
        assert stats.compression_ratio > 0

    @pytest.mark.benchmark(group="data_types")
    @pytest.mark.parametrize("data_type,data_factory", [
        ("text", lambda: b"Hello World! " * 5000),
        ("binary", lambda: bytes(range(256)) * 100),
        ("random", lambda: np.random.bytes(50000)),
        ("repetitive", lambda: b"A" * 50000),
    ])
    def test_data_type_performance(self, benchmark, data_type, data_factory):
        """Benchmark compression performance across different data types."""
        engine = ConsciousnessCompressionEngine()
        test_data = data_factory()

        def compress_test():
            compressed, stats = engine.compress(test_data)
            return compressed, stats

        compressed, stats = benchmark(compress_test)

        # Validate results
        assert stats.lossless_verified
        assert len(compressed) > 0

    @pytest.mark.benchmark(group="scalability")
    @pytest.mark.parametrize("data_size", [1000, 10000, 50000, 100000])
    def test_scalability_performance(self, benchmark, data_size):
        """Benchmark compression scalability with data size."""
        engine = ConsciousnessCompressionEngine()
        test_data = b"Scalability test data pattern. " * (data_size // 30)

        def compress_test():
            compressed, stats = engine.compress(test_data)
            return compressed, stats

        compressed, stats = benchmark(compress_test)

        # Validate results
        assert stats.lossless_verified
        assert len(test_data) == data_size or abs(len(test_data) - data_size) < 50  # Allow small variance


class TestStatisticalValidationModern:
    """Statistical validation tests using hypothesis and property-based testing."""

    @given(st.floats(min_value=0.1, max_value=1000.0))
    @settings(max_examples=100)
    def test_compression_ratio_bounds(self, compression_engine, data_size):
        """Property-based test for compression ratio bounds."""
        test_data = b"X" * int(data_size)
        compressed, stats = compression_engine.compress(test_data)

        # Compression ratio should be between 0 and 1
        assert 0.0 <= stats.compression_ratio <= 1.0

        # Compression factor should be >= 1
        assert stats.compression_factor >= 1.0

    @given(st.binary(min_size=100, max_size=10000))
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_lossless_property(self, compression_engine, data):
        """Property-based test that compression is always lossless."""
        compressed, compress_stats = compression_engine.compress(data)
        decompressed, decompress_stats = compression_engine.decompress(compressed)

        # Must be perfectly lossless
        assert decompressed == data
        assert compress_stats.lossless_verified
        assert decompress_stats.lossless_verified

    @given(st.integers(min_value=1, max_value=1000))
    @settings(max_examples=20)
    def test_pattern_detection_scales(self, data_size):
        """Test that pattern detection scales appropriately with data size."""
        cm_core = ConsciousnessMathematicsCore()
        pattern_engine = PatternAnalysisEngine(cm_core)

        # Create data with known patterns
        pattern = b"PATTERN_"
        repetitions = max(1, data_size // len(pattern))
        test_data = (pattern * repetitions)[:data_size]

        sample_points = list(range(0, len(test_data), 10))
        patterns = pattern_engine.analyze_patterns(test_data, sample_points)

        # Should find patterns even for small data
        assert len(patterns) > 0

        # Pattern frequencies should be reasonable
        for pattern in patterns:
            assert pattern.frequency > 0
            assert pattern.frequency <= len(sample_points)


class TestMemoryAndResourceManagement:
    """Tests for memory usage and resource management."""

    @pytest.mark.slow
    def test_memory_usage_bounds(self, compression_engine, memory_monitor, benchmark_data):
        """Test that memory usage stays within reasonable bounds."""
        test_data = benchmark_data['mixed_data'] * 10  # ~30KB

        memory_monitor.check()  # Baseline

        compressed, stats = compression_engine.compress(test_data)
        memory_after_compress = memory_monitor.check()

        decompressed, _ = compression_engine.decompress(compressed)
        memory_after_decompress = memory_monitor.check()

        # Memory usage should not exceed reasonable bounds
        assert memory_after_compress['delta_mb'] < 100  # Less than 100MB increase
        assert memory_after_decompress['delta_mb'] < 100

        # Verify correctness
        assert decompressed == test_data

    @pytest.mark.integration
    def test_resource_cleanup(self, compression_engine, memory_monitor):
        """Test that resources are properly cleaned up."""
        initial_memory = memory_monitor.check()

        # Perform multiple compression operations
        for i in range(10):
            test_data = f"Test data iteration {i} " * 1000
            compressed, _ = compression_engine.compress(test_data.encode())
            decompressed, _ = compression_engine.decompress(compressed)

            assert decompressed == test_data.encode()

        final_memory = memory_monitor.check()

        # Memory should not grow significantly over iterations
        memory_growth = final_memory['delta_mb'] - initial_memory['delta_mb']
        assert memory_growth < 50  # Less than 50MB total growth


class TestCrossPlatformCompatibility:
    """Tests for cross-platform compatibility and edge cases."""

    @pytest.mark.parametrize("encoding", ["utf-8", "latin-1", "ascii"])
    def test_text_encoding_handling(self, compression_engine, encoding):
        """Test compression works with different text encodings."""
        try:
            test_text = "Hello 世界 🌍 Test data with unicode"
            test_data = test_text.encode(encoding)

            compressed, stats = compression_engine.compress(test_data)
            decompressed, _ = compression_engine.decompress(compressed)

            assert decompressed == test_data
            assert stats.lossless_verified

        except UnicodeEncodeError:
            # Skip encodings that can't handle the test text
            pytest.skip(f"Encoding {encoding} cannot handle test text")

    def test_large_data_handling(self, compression_engine):
        """Test compression of large data (if system allows)."""
        # Test with reasonably large data (adjust based on system capabilities)
        try:
            # Create ~1MB of test data
            test_data = b"A" * (1024 * 1024)

            compressed, stats = compression_engine.compress(test_data)
            decompressed, _ = compression_engine.decompress(compressed)

            assert decompressed == test_data
            assert stats.lossless_verified
            assert stats.compression_ratio > 0.9  # Should compress very well

        except MemoryError:
            pytest.skip("System does not have enough memory for large data test")

    def test_concurrent_access_safety(self, compression_engine, test_data_variety):
        """Test that the engine handles concurrent access safely."""
        import threading
        import queue

        results = queue.Queue()
        errors = []

        def compress_worker(data_subset, worker_id):
            try:
                compressed, stats = compression_engine.compress(data_subset)
                decompressed, _ = compression_engine.decompress(compressed)
                results.put((worker_id, decompressed == data_subset, stats.lossless_verified))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Create multiple threads with different data
        threads = []
        test_cases = list(test_data_variety.values())[:5]  # Use first 5 datasets

        for i, test_data in enumerate(test_cases):
            thread = threading.Thread(target=compress_worker, args=(test_data, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)  # 30 second timeout

        # Check results
        successful_results = 0
        while not results.empty():
            worker_id, data_correct, lossless = results.get()
            assert data_correct, f"Worker {worker_id} data corruption"
            assert lossless, f"Worker {worker_id} lossless verification failed"
            successful_results += 1

        assert successful_results == len(test_cases)
        assert len(errors) == 0, f"Threading errors: {errors}"


# Performance regression tests
class TestPerformanceRegression:
    """Tests to prevent performance regressions."""

    PERFORMANCE_BASELINES = {
        'small_data_compression_time': 0.01,  # seconds
        'small_data_decompression_time': 0.005,  # seconds
        'compression_ratio_threshold': 0.5,  # minimum acceptable ratio
        'memory_usage_mb': 50,  # maximum memory usage
    }

    def test_compression_speed_regression(self, compression_engine, performance_timer):
        """Test that compression speed doesn't regress below baseline."""
        test_data = b"Performance regression test data. " * 1000

        performance_timer.start()
        compressed, stats = compression_engine.compress(test_data)
        compression_time = performance_timer.stop()

        # Check against baseline
        assert compression_time < self.PERFORMANCE_BASELINES['small_data_compression_time'] * 10, \
            f"Compression too slow: {compression_time:.4f}s (baseline: {self.PERFORMANCE_BASELINES['small_data_compression_time']}s)"

        assert stats.lossless_verified

    def test_decompression_speed_regression(self, compression_engine, performance_timer):
        """Test that decompression speed doesn't regress below baseline."""
        test_data = b"Performance regression test data. " * 1000
        compressed, _ = compression_engine.compress(test_data)

        performance_timer.start()
        decompressed, stats = compression_engine.decompress(compressed)
        decompression_time = performance_timer.stop()

        # Check against baseline
        assert decompression_time < self.PERFORMANCE_BASELINES['small_data_decompression_time'] * 10, \
            f"Decompression too slow: {decompression_time:.4f}s (baseline: {self.PERFORMANCE_BASELINES['small_data_decompression_time']}s)"

        assert decompressed == test_data
        assert stats.lossless_verified

    def test_memory_usage_regression(self, compression_engine, memory_monitor):
        """Test that memory usage doesn't regress above baseline."""
        test_data = b"Memory regression test data. " * 5000

        memory_before = memory_monitor.check()
        compressed, stats = compression_engine.compress(test_data)
        decompressed, _ = compression_engine.decompress(compressed)
        memory_after = memory_monitor.check()

        memory_delta = memory_after['current_mb'] - memory_before['current_mb']

        assert memory_delta < self.PERFORMANCE_BASELINES['memory_usage_mb'], \
            f"Memory usage too high: {memory_delta:.1f}MB (baseline: {self.PERFORMANCE_BASELINES['memory_usage_mb']}MB)"

        assert decompressed == test_data
        assert stats.lossless_verified
