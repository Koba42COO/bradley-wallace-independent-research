"""
Quantum Hardware Abstraction Layer (QHAL)
=========================================

Provides a unified interface to multiple quantum computing backends:
    - D-Wave Systems (quantum annealing)
    - IBM Quantum (gate-based)
    - AWS Braket (multi-backend)
    - Azure Quantum (enterprise)
    - Local Simulator (development/testing)

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

from .base import QuantumBackend, BackendType, QuantumTask, QuantumResult
from .dwave_backend import DWaveBackend
from .ibm_backend import IBMQuantumBackend
from .aws_backend import AWSBraketBackend
from .local_simulator import LocalSimulatorBackend

__all__ = [
    "QuantumBackend",
    "BackendType",
    "QuantumTask",
    "QuantumResult",
    "DWaveBackend",
    "IBMQuantumBackend",
    "AWSBraketBackend",
    "LocalSimulatorBackend",
]

