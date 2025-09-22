import pytest

def test_sample_function(sample_data):
    """Test basic functionality"""
    assert len(sample_data["measurements"]) == 5
    assert all(isinstance(x, float) for x in sample_data["measurements"])

def test_baseline_calculation(sample_data):
    """Test baseline calculation"""
    measurements = sample_data["measurements"]
    avg = sum(measurements) / len(measurements)
    assert avg > sample_data["baseline"]
