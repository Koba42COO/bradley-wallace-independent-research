"""Central configuration for PAC System.

Reads environment variables with safe defaults.
"""

import os


def getenv_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def getenv_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default


def getenv_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except Exception:
        return default


# Core
PRIME_SCALE = getenv_int("PAC_PRIME_SCALE", 50000)
CONSCIOUSNESS_WEIGHT = getenv_float("PAC_WEIGHT", 0.79)

# Storage
DATA_DIR = getenv_str("PAC_DATA_DIR", "data")
ARTIFACTS_DIR = getenv_str("PAC_ARTIFACTS_DIR", "artifacts")
REPORTS_DIR = getenv_str("PAC_REPORTS_DIR", "reports")
LOGS_DIR = getenv_str("PAC_LOGS_DIR", "logs")


