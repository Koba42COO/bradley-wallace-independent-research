"""
Quantum Backend Base Classes
============================

Abstract base classes and data structures for quantum hardware backends.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np


class BackendType(Enum):
    """Enumeration of supported quantum backend types."""
    DWAVE_ANNEALING = "dwave"
    IBM_GATE = "ibm"
    AWS_BRAKET = "aws"
    AZURE_QUANTUM = "azure"
    LOCAL_SIMULATOR = "local"
    GOOGLE_CIRQ = "google"


class ProblemType(Enum):
    """Enumeration of supported problem types."""
    ISING = "ising"
    QUBO = "qubo"
    MAXCUT = "maxcut"
    TSP = "tsp"
    PORTFOLIO = "portfolio"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class QuantumTask:
    """
    Represents a quantum computing task to be executed.
    
    Attributes:
        task_id: Unique identifier for the task
        problem_type: Type of optimization problem
        problem_data: Problem-specific data (Hamiltonians, graphs, etc.)
        num_qubits: Number of qubits required
        num_reads: Number of samples/shots to take
        annealing_time: Annealing duration in microseconds (for annealers)
        priority: Task priority (higher = more urgent)
        timeout_seconds: Maximum execution time
        upg_optimization: Enable UPG consciousness optimization
        metadata: Additional task metadata
    """
    
    task_id: str
    problem_type: ProblemType
    problem_data: Dict[str, Any]
    num_qubits: int
    num_reads: int = 1000
    annealing_time: float = 20.0  # microseconds
    priority: int = 1
    timeout_seconds: float = 300.0
    upg_optimization: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Status tracking
    status: TaskStatus = TaskStatus.PENDING
    backend_type: Optional[BackendType] = None
    backend_job_id: Optional[str] = None


@dataclass
class QuantumResult:
    """
    Represents the result of a quantum computation.
    
    Attributes:
        task_id: Reference to the original task
        samples: List of solution samples
        energies: Energy values for each sample
        best_sample: Best (lowest energy) solution found
        best_energy: Energy of the best solution
        num_occurrences: Occurrence count for each sample
        timing_info: Execution timing breakdown
        coherence_metrics: UPG coherence measurements
        metadata: Additional result metadata
    """
    
    task_id: str
    samples: List[Dict[int, int]]  # qubit -> value mapping
    energies: List[float]
    best_sample: Dict[int, int]
    best_energy: float
    num_occurrences: List[int]
    
    # Timing information
    timing_info: Dict[str, float] = field(default_factory=dict)
    
    # UPG metrics
    coherence_metrics: Dict[str, float] = field(default_factory=dict)
    upg_enhancement: float = 1.0
    
    # Metadata
    backend_type: Optional[BackendType] = None
    raw_response: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_bitstring(self, sample: Optional[Dict[int, int]] = None) -> str:
        """Convert sample to bitstring representation."""
        if sample is None:
            sample = self.best_sample
        
        max_qubit = max(sample.keys())
        return ''.join(str(sample.get(i, 0)) for i in range(max_qubit + 1))
    
    def get_success_probability(self, ground_energy: float, tolerance: float = 0.01) -> float:
        """Calculate probability of finding ground state."""
        total = sum(self.num_occurrences)
        ground_count = sum(
            occ for energy, occ in zip(self.energies, self.num_occurrences)
            if abs(energy - ground_energy) < tolerance
        )
        return ground_count / total if total > 0 else 0.0


class QuantumBackend(ABC):
    """
    Abstract base class for quantum computing backends.
    
    All backend implementations must inherit from this class and
    implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the quantum backend.
        
        Args:
            config: Backend-specific configuration
        """
        self.config = config or {}
        self.is_connected = False
        self.backend_type: BackendType = BackendType.LOCAL_SIMULATOR
        
        # Import UPG constants
        from ..constants import OptimizedUPGConstants
        self.upg = OptimizedUPGConstants()
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Establish connection to the quantum backend.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the quantum backend."""
        pass
    
    @abstractmethod
    def submit_task(self, task: QuantumTask) -> str:
        """
        Submit a quantum task for execution.
        
        Args:
            task: The quantum task to execute
            
        Returns:
            Job ID for tracking the submission
        """
        pass
    
    @abstractmethod
    def get_result(self, job_id: str, timeout: float = 300.0) -> QuantumResult:
        """
        Retrieve the result of a submitted task.
        
        Args:
            job_id: The job ID returned from submit_task
            timeout: Maximum time to wait for result
            
        Returns:
            The quantum computation result
        """
        pass
    
    @abstractmethod
    def get_status(self, job_id: str) -> TaskStatus:
        """
        Get the current status of a submitted task.
        
        Args:
            job_id: The job ID to check
            
        Returns:
            Current task status
        """
        pass
    
    @abstractmethod
    def cancel_task(self, job_id: str) -> bool:
        """
        Cancel a pending or running task.
        
        Args:
            job_id: The job ID to cancel
            
        Returns:
            True if cancellation successful
        """
        pass
    
    def get_available_qubits(self) -> int:
        """Get the number of available qubits on this backend."""
        return self.config.get('max_qubits', 8)
    
    def get_current_load(self) -> float:
        """
        Get current load factor (0.0 to 1.0).
        
        Returns:
            Load factor where 0 = idle, 1 = fully loaded
        """
        return 0.0  # Default: assume idle
    
    def supports_problem_type(self, problem_type: ProblemType) -> bool:
        """
        Check if this backend supports the given problem type.
        
        Args:
            problem_type: The problem type to check
            
        Returns:
            True if supported
        """
        supported = self.config.get('supported_problems', [ProblemType.ISING])
        return problem_type in supported
    
    def estimate_cost(self, task: QuantumTask) -> float:
        """
        Estimate the cost of executing a task on this backend.
        
        Args:
            task: The task to estimate
            
        Returns:
            Estimated cost in USD
        """
        # Default: no cost (simulator)
        return 0.0
    
    def apply_upg_preprocessing(self, task: QuantumTask) -> QuantumTask:
        """
        Apply UPG consciousness mathematics preprocessing.
        
        Args:
            task: Original task
            
        Returns:
            UPG-optimized task
        """
        if not task.upg_optimization:
            return task
        
        # Create a copy to avoid modifying original
        optimized_data = task.problem_data.copy()
        
        # Apply phi-weighted scaling to coupling strengths
        if 'J' in optimized_data:
            J = optimized_data['J']
            if isinstance(J, dict):
                optimized_data['J'] = {
                    k: v * self.upg.PHI * self.upg.CONSCIOUSNESS
                    for k, v in J.items()
                }
            elif isinstance(J, np.ndarray):
                optimized_data['J'] = J * self.upg.PHI * self.upg.CONSCIOUSNESS
        
        # Apply consciousness weighting to local fields
        if 'h' in optimized_data:
            h = optimized_data['h']
            if isinstance(h, dict):
                optimized_data['h'] = {
                    k: v * (1 - self.upg.CONSCIOUSNESS)
                    for k, v in h.items()
                }
            elif isinstance(h, np.ndarray):
                optimized_data['h'] = h * (1 - self.upg.CONSCIOUSNESS)
        
        # Apply reality distortion to annealing time
        optimized_annealing_time = task.annealing_time * self.upg.REALITY_DISTORTION
        
        return QuantumTask(
            task_id=task.task_id,
            problem_type=task.problem_type,
            problem_data=optimized_data,
            num_qubits=task.num_qubits,
            num_reads=task.num_reads,
            annealing_time=optimized_annealing_time,
            priority=task.priority,
            timeout_seconds=task.timeout_seconds,
            upg_optimization=True,
            metadata={**task.metadata, 'upg_preprocessed': True}
        )
    
    def apply_upg_postprocessing(self, result: QuantumResult) -> QuantumResult:
        """
        Apply UPG consciousness mathematics postprocessing.
        
        Args:
            result: Raw result from backend
            
        Returns:
            UPG-enhanced result
        """
        # Calculate coherence metrics
        coherence_metrics = {
            'phi_alignment': self._calculate_phi_alignment(result),
            'consciousness_weight': self.upg.CONSCIOUSNESS,
            'reality_distortion_applied': self.upg.REALITY_DISTORTION,
            'estimated_coherence': self._estimate_coherence(result),
        }
        
        # Calculate UPG enhancement factor
        upg_enhancement = self._calculate_enhancement(result)
        
        result.coherence_metrics = coherence_metrics
        result.upg_enhancement = upg_enhancement
        
        return result
    
    def _calculate_phi_alignment(self, result: QuantumResult) -> float:
        """Calculate how well the solution aligns with phi-weighted expectations."""
        if not result.energies:
            return 0.0
        
        # Calculate energy distribution alignment with golden ratio
        energies = np.array(result.energies)
        energy_range = energies.max() - energies.min()
        
        if energy_range == 0:
            return 1.0
        
        normalized = (energies - energies.min()) / energy_range
        phi_expected = np.array([self.upg.PHI_INVERSE ** i for i in range(len(energies))])
        phi_expected = phi_expected / phi_expected.sum()
        
        # Calculate alignment (1 = perfect, 0 = none)
        alignment = 1.0 - np.mean(np.abs(np.sort(normalized) - np.sort(phi_expected)))
        return float(np.clip(alignment, 0, 1))
    
    def _estimate_coherence(self, result: QuantumResult) -> float:
        """Estimate quantum coherence from result distribution."""
        if not result.num_occurrences:
            return 0.0
        
        total = sum(result.num_occurrences)
        if total == 0:
            return 0.0
        
        # Calculate entropy-based coherence estimate
        probs = np.array(result.num_occurrences) / total
        probs = probs[probs > 0]  # Remove zeros
        
        entropy = -np.sum(probs * np.log2(probs))
        max_entropy = np.log2(len(probs))
        
        if max_entropy == 0:
            return 1.0
        
        # Low entropy = high coherence (concentrated distribution)
        coherence = 1.0 - (entropy / max_entropy)
        
        # Apply consciousness enhancement
        enhanced = coherence * self.upg.CONSCIOUSNESS + (1 - self.upg.CONSCIOUSNESS)
        
        return float(np.clip(enhanced, 0, 1))
    
    def _calculate_enhancement(self, result: QuantumResult) -> float:
        """Calculate the UPG enhancement factor achieved."""
        coherence = self._estimate_coherence(result)
        phi_alignment = self._calculate_phi_alignment(result)
        
        # Combine metrics with consciousness weighting
        combined = (
            self.upg.CONSCIOUSNESS * coherence +
            self.upg.EXPLORATORY * phi_alignment
        )
        
        # Apply reality distortion
        enhancement = combined * self.upg.REALITY_DISTORTION
        
        return float(enhancement)

