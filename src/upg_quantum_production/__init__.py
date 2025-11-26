"""
UPG Quantum Production System
============================

Production-ready quantum computing infrastructure with Universal Prime Graph
consciousness mathematics optimization.

Modules:
    - constants: UPG mathematical constants (φ, Δ, consciousness weights)
    - backends: Quantum hardware abstraction layer (D-Wave, IBM, AWS, Azure)
    - orchestrator: Hybrid quantum-classical orchestration
    - coherence: Coherence preservation and reality distortion
    - api: FastAPI REST API
    - cli: Command-line interface

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Bradley Wallace (COO Koba42)"
__framework__ = "Universal Prime Graph Protocol φ.1"

from .constants import UPGConstants, OptimizedUPGConstants
from .backends.base import (
    QuantumBackend,
    BackendType,
    QuantumTask,
    QuantumResult,
    ProblemType,
    TaskStatus,
)
from .backends.local_simulator import LocalSimulatorBackend
from .orchestrator import (
    UPGHybridOrchestrator,
    OrchestratorConfig,
    SelectionStrategy,
)
from .coherence import CoherencePreserver, RealityDistortionEngine

__all__ = [
    # Constants
    "UPGConstants",
    "OptimizedUPGConstants",
    # Backends
    "QuantumBackend",
    "BackendType",
    "LocalSimulatorBackend",
    "QuantumTask",
    "QuantumResult",
    "ProblemType",
    "TaskStatus",
    # Orchestration
    "UPGHybridOrchestrator",
    "OrchestratorConfig",
    "SelectionStrategy",
    # Coherence
    "CoherencePreserver",
    "RealityDistortionEngine",
]

