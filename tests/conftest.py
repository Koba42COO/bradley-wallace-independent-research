"""
Consciousness Compression Engine - pytest Configuration
=======================================================

Shared fixtures, configuration, and test utilities for comprehensive testing.

🛠️ FIXTURES PROVIDED:
• compression_engine: Fresh engine instance for each test
• test_data_variety: Collection of diverse test datasets
• benchmark_data: Standardized benchmark datasets
• memory_monitor: Memory usage tracking fixture
• performance_timer: High-precision timing fixture
• temp_file_factory: Temporary file management

🔧 CONFIGURATION:
• Custom markers for test categorization
• Warning filters for clean test output
• Random seed control for reproducible tests
• Performance thresholds and validation

📊 TEST UTILITIES:
• Data generators for various compression scenarios
• Statistical validation helpers
• Performance comparison utilities
• Memory leak detection tools
"""

import pytest
import numpy as np
import tempfile
import hashlib
import psutil
import os
from pathlib import Path
from typing import Dict, List, Generator, Any, Tuple
import time
import gc

# Import the compression engine
from compression_engine import (
    ConsciousnessCompressionEngine,
    CompressionEngineConfig,
    CompressionMode,
    CompressionStats
)


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers for test categorization."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "benchmark: marks tests as benchmarks")
    config.addinivalue_line("markers", "security: marks tests as security tests")
    config.addinivalue_line("markers", "performance: marks tests as performance tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "edge_case: marks tests as edge case tests")
    config.addinivalue_line("markers", "regression: marks tests as regression tests")


@pytest.fixture(scope="session")
def random_seed():
    """Set random seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture(scope="function")
def compression_engine():
    """Provide a fresh compression engine instance for each test."""
    config = CompressionEngineConfig(mode=CompressionMode.BALANCED)
    engine = ConsciousnessCompressionEngine(config)
    yield engine
    # Cleanup
    del engine
    gc.collect()


@pytest.fixture(scope="session")
def test_data_variety(random_seed) -> Dict[str, bytes]:
    """Provide a variety of test datasets for comprehensive testing."""
    datasets = {}

    # 1. Repetitive text data
    datasets['repetitive_text'] = b"Hello World! " * 1000

    # 2. Structured data (JSON-like)
    structured_data = b""
    for i in range(500):
        structured_data += f'{{"id": {i}, "name": "item_{i}", "value": {i*1.5}}}\n'.encode()
    datasets['structured_data'] = structured_data

    # 3. Random data (low compressibility)
    datasets['random_data'] = np.random.bytes(5000)

    # 4. Highly compressible data
    datasets['compressible_data'] = b"A" * 5000

    # 5. Mixed content
    mixed_data = b""
    patterns = [b"pattern_A", b"pattern_B", b"pattern_C", np.random.bytes(10)]
    for i in range(500):
        mixed_data += patterns[i % len(patterns)]
    datasets['mixed_content'] = mixed_data

    # 6. Binary data
    datasets['binary_data'] = bytes(range(256)) * 20

    # 7. Empty data (edge case)
    datasets['empty_data'] = b""

    # 8. Single byte (edge case)
    datasets['single_byte'] = b"X"

    # 9. Large repetitive data
    datasets['large_repetitive'] = b"VERY_LONG_PATTERN_THAT_REPEATS_MANY_TIMES " * 1000

    # 10. Unicode text
    unicode_text = "Hello 世界 🌍 This is unicode text with emojis 🎉 and symbols ∑∆∞"
    datasets['unicode_text'] = (unicode_text * 200).encode('utf-8')

    return datasets


@pytest.fixture(scope="session")
def benchmark_data() -> Dict[str, bytes]:
    """Provide standardized benchmark datasets similar to Silesia corpus."""
    datasets = {}

    # Create Silesia-corpus-like data
    # 1. Text data (dickens)
    datasets['text_data'] = b"To be or not to be, that is the question. " * 2000

    # 2. Binary data (mozilla)
    binary_content = bytes(range(256)) * 50 + np.random.bytes(2000)
    datasets['binary_data'] = binary_content

    # 3. Structured data (samba)
    structured = b""
    for i in range(1000):
        structured += f"[section_{i}]\nkey{i}=value{i}\n".encode()
    datasets['structured_data'] = structured

    # 4. Repetitive data (webster)
    datasets['repetitive_data'] = b"compression algorithm mathematics consciousness " * 1000

    # 5. Mixed data (mixed)
    mixed = np.random.bytes(3000) + b"PATTERN" * 500 + bytes(range(128)) * 10
    datasets['mixed_data'] = mixed

    return datasets


@pytest.fixture(scope="function")
def memory_monitor():
    """Monitor memory usage during test execution."""
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    class MemoryMonitor:
        def __init__(self, initial_mb):
            self.initial_mb = initial_mb
            self.peak_mb = initial_mb

        def check(self) -> Dict[str, float]:
            current_mb = process.memory_info().rss / 1024 / 1024
            self.peak_mb = max(self.peak_mb, current_mb)
            return {
                'current_mb': current_mb,
                'peak_mb': self.peak_mb,
                'delta_mb': current_mb - self.initial_mb
            }

    monitor = MemoryMonitor(initial_memory)
    yield monitor


@pytest.fixture(scope="function")
def performance_timer():
    """High-precision performance timing fixture."""
    class PerformanceTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = time.perf_counter()

        def stop(self) -> float:
            self.end_time = time.perf_counter()
            return self.end_time - self.start_time

        def elapsed_ns(self) -> int:
            """Return elapsed time in nanoseconds."""
            if self.start_time and self.end_time:
                return int((self.end_time - self.start_time) * 1e9)
            return 0

    timer = PerformanceTimer()
    yield timer


@pytest.fixture(scope="function")
def temp_file_factory():
    """Factory for creating temporary files that are automatically cleaned up."""
    created_files = []

    def create_temp_file(content: bytes, suffix: str = ".bin") -> Path:
        fd, path = tempfile.mkstemp(suffix=suffix)
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(content)
            file_path = Path(path)
            created_files.append(file_path)
            return file_path
        except:
            os.close(fd)
            raise

    yield create_temp_file

    # Cleanup
    for file_path in created_files:
        try:
            if file_path.exists():
                file_path.unlink()
        except:
            pass


@pytest.fixture(scope="function")
def compression_validator():
    """Fixture for validating compression results."""
    class CompressionValidator:
        def validate_round_trip(self, engine, data: bytes) -> Tuple[bool, CompressionStats, bytes]:
            """Validate that compression/decompression is lossless."""
            # Compress
            compressed, compress_stats = engine.compress(data)

            # Decompress
            decompressed, decompress_stats = engine.decompress(compressed)

            # Validate
            is_lossless = (decompressed == data)
            original_hash = hashlib.sha256(data).hexdigest()
            decompressed_hash = hashlib.sha256(decompressed).hexdigest()

            if not is_lossless:
                print(f"Original hash: {original_hash}")
                print(f"Decompressed hash: {decompressed_hash}")
                print(f"Data lengths: {len(data)} -> {len(decompressed)}")

            return is_lossless, compress_stats, decompressed

        def validate_stats(self, stats: CompressionStats, original_size: int):
            """Validate compression statistics."""
            assert stats.original_size == original_size
            assert stats.compressed_size >= 0
            assert 0.0 <= stats.compression_ratio <= 1.0
            assert stats.compression_factor >= 1.0
            assert stats.compression_time >= 0.0
            assert stats.patterns_found >= 0
            assert 0.0 <= stats.consciousness_level <= 12.0

    validator = CompressionValidator()
    yield validator


# Performance thresholds for regression testing
PERFORMANCE_THRESHOLDS = {
    'compression_speed_mb_s': 10.0,  # Minimum 10 MB/s compression
    'decompression_speed_mb_s': 50.0,  # Minimum 50 MB/s decompression
    'max_memory_mb': 500.0,  # Maximum 500MB memory usage
    'min_compression_ratio': 0.0,  # Any compression is acceptable
    'max_compression_time_s': 30.0,  # Maximum 30s for reasonable data
}


@pytest.fixture(scope="session", autouse=True)
def configure_test_environment():
    """Configure the test environment for optimal testing."""
    # Set numpy random seed for reproducible tests
    np.random.seed(42)

    # Configure warnings
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

    # Set environment variables for testing
    os.environ['PYTHONHASHSEED'] = '42'  # Reproducible hash randomization
    os.environ['COVERAGE_PROCESS_START'] = '1'  # Enable coverage subprocess tracking

    yield

    # Cleanup environment
    if 'PYTHONHASHSEED' in os.environ:
        del os.environ['PYTHONHASHSEED']
    if 'COVERAGE_PROCESS_START' in os.environ:
        del os.environ['COVERAGE_PROCESS_START']


# Utility functions for test data generation
def generate_compressible_data(size: int, pattern_length: int = 100) -> bytes:
    """Generate highly compressible test data."""
    pattern = b"A" * pattern_length
    repetitions = size // pattern_length
    return (pattern * repetitions)[:size]


def generate_random_data(size: int, seed: int = 42) -> bytes:
    """Generate random test data with fixed seed."""
    rng = np.random.RandomState(seed)
    return rng.bytes(size)


def generate_structured_data(num_records: int) -> bytes:
    """Generate structured test data (JSON-like)."""
    data = b""
    for i in range(num_records):
        record = f'{{"id": {i}, "name": "record_{i}", "value": {i*3.14}}}\n'
        data += record.encode()
    return data


def benchmark_compression_engines(engines: List, data: bytes) -> Dict[str, Dict]:
    """Benchmark multiple compression engines on the same data."""
    results = {}

    for engine_name, engine in engines:
        # Time compression
        start_time = time.perf_counter()
        compressed, stats = engine.compress(data)
        compression_time = time.perf_counter() - start_time

        # Time decompression
        start_time = time.perf_counter()
        decompressed, _ = engine.decompress(compressed)
        decompression_time = time.perf_counter() - start_time

        # Calculate metrics
        compression_speed = len(data) / compression_time / 1024 / 1024  # MB/s
        decompression_speed = len(data) / decompression_time / 1024 / 1024  # MB/s

        results[engine_name] = {
            'compression_ratio': stats.compression_ratio,
            'compression_speed_mb_s': compression_speed,
            'decompression_speed_mb_s': decompression_speed,
            'lossless': decompressed == data,
            'compressed_size': len(compressed),
            'original_size': len(data)
        }

    return results