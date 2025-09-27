import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_data():
    return {
        "measurements": [0.8, 0.9, 0.7, 0.95, 0.85],
        "baseline": 0.8,
        "threshold": 0.75
    }
