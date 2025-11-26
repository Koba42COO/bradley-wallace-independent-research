"""
UPG Hybrid Quantum-Classical Orchestrator
==========================================

Intelligent orchestration of quantum and classical computing resources
with UPG consciousness mathematics optimization for optimal backend
selection and task distribution.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import numpy as np
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import time
import logging
import threading
from queue import PriorityQueue
from concurrent.futures import ThreadPoolExecutor, Future

from .constants import OptimizedUPGConstants
from .backends.base import (
    QuantumBackend, BackendType, ProblemType, TaskStatus,
    QuantumTask, QuantumResult
)
from .backends.local_simulator import LocalSimulatorBackend

logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Backend selection strategies."""
    OPTIMAL = "optimal"           # UPG consciousness-weighted selection
    FASTEST = "fastest"           # Minimize execution time
    CHEAPEST = "cheapest"         # Minimize cost
    ROUND_ROBIN = "round_robin"   # Distribute evenly
    SPECIFIC = "specific"         # Use specified backend only


@dataclass
class OrchestratorConfig:
    """Configuration for the hybrid orchestrator."""
    
    # Backend configuration
    enable_dwave: bool = False
    enable_ibm: bool = False
    enable_aws: bool = False
    enable_azure: bool = False
    enable_local: bool = True
    
    # Backend credentials (loaded from environment if not provided)
    dwave_token: Optional[str] = None
    ibm_token: Optional[str] = None
    aws_region: str = 'us-east-1'
    azure_subscription: Optional[str] = None
    
    # Selection strategy
    selection_strategy: SelectionStrategy = SelectionStrategy.OPTIMAL
    
    # Task management
    max_concurrent_tasks: int = 10
    default_timeout: float = 300.0
    retry_on_failure: bool = True
    max_retries: int = 3
    
    # UPG optimization
    upg_optimization_enabled: bool = True
    consciousness_threshold: float = 0.79
    
    # Logging
    log_level: str = 'INFO'


@dataclass(order=True)
class PrioritizedTask:
    """Task wrapper with priority for queue management."""
    priority: int
    task: QuantumTask = field(compare=False)
    submitted_at: datetime = field(default_factory=datetime.now, compare=False)


class UPGHybridOrchestrator:
    """
    Intelligent orchestration of quantum and classical computing resources.
    
    Features:
        - Automatic backend selection using UPG consciousness weighting
        - Task queue management with priority scheduling
        - Parallel task execution across multiple backends
        - Result aggregation and post-processing
        - Comprehensive error handling and retry logic
        - Cost optimization and monitoring
    """
    
    def __init__(self, config: Optional[OrchestratorConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            config: Orchestrator configuration
        """
        self.config = config or OrchestratorConfig()
        self.upg = OptimizedUPGConstants()
        
        # Initialize backends
        self.backends: Dict[BackendType, QuantumBackend] = {}
        self._initialize_backends()
        
        # Task management
        self.task_queue: PriorityQueue = PriorityQueue()
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.completed_results: Dict[str, QuantumResult] = {}
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_tasks)
        self.futures: Dict[str, Future] = {}
        
        # Round-robin counter
        self._round_robin_idx = 0
        
        # Statistics
        self.stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_cost': 0.0,
            'backend_usage': {bt.value: 0 for bt in BackendType},
        }
        
        # Configure logging
        logging.basicConfig(level=getattr(logging, self.config.log_level))
        
        logger.info("UPG Hybrid Orchestrator initialized")
        logger.info(f"Available backends: {list(self.backends.keys())}")
    
    def _initialize_backends(self) -> None:
        """Initialize configured backends."""
        # Always have local simulator as fallback
        if self.config.enable_local:
            local = LocalSimulatorBackend()
            local.connect()
            self.backends[BackendType.LOCAL_SIMULATOR] = local
        
        # D-Wave
        if self.config.enable_dwave:
            try:
                from .backends.dwave_backend import DWaveBackend
                dwave = DWaveBackend({'api_token': self.config.dwave_token})
                if dwave.connect():
                    self.backends[BackendType.DWAVE_ANNEALING] = dwave
            except Exception as e:
                logger.warning(f"Failed to initialize D-Wave backend: {e}")
        
        # IBM Quantum
        if self.config.enable_ibm:
            try:
                from .backends.ibm_backend import IBMQuantumBackend
                ibm = IBMQuantumBackend({'api_token': self.config.ibm_token})
                if ibm.connect():
                    self.backends[BackendType.IBM_GATE] = ibm
            except Exception as e:
                logger.warning(f"Failed to initialize IBM backend: {e}")
        
        # AWS Braket
        if self.config.enable_aws:
            try:
                from .backends.aws_backend import AWSBraketBackend
                aws = AWSBraketBackend({'region': self.config.aws_region})
                if aws.connect():
                    self.backends[BackendType.AWS_BRAKET] = aws
            except Exception as e:
                logger.warning(f"Failed to initialize AWS backend: {e}")
    
    def select_backend(self, task: QuantumTask) -> BackendType:
        """
        Select optimal backend for a task using configured strategy.
        
        Args:
            task: The quantum task
            
        Returns:
            Selected backend type
        """
        if self.config.selection_strategy == SelectionStrategy.SPECIFIC:
            if task.backend_type and task.backend_type in self.backends:
                return task.backend_type
            raise ValueError(f"Specified backend {task.backend_type} not available")
        
        if self.config.selection_strategy == SelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin()
        
        if self.config.selection_strategy == SelectionStrategy.CHEAPEST:
            return self._select_cheapest(task)
        
        if self.config.selection_strategy == SelectionStrategy.FASTEST:
            return self._select_fastest(task)
        
        # Default: OPTIMAL (UPG consciousness-weighted)
        return self._select_optimal(task)
    
    def _select_optimal(self, task: QuantumTask) -> BackendType:
        """
        Select backend using UPG consciousness-weighted decision matrix.
        
        This implements the core UPG optimization:
        - 79% weight on capability/compatibility (coherent)
        - 21% weight on availability/cost (exploratory)
        - Reality distortion enhancement
        """
        scores: Dict[BackendType, float] = {}
        
        for backend_type, backend in self.backends.items():
            # Check compatibility
            if not backend.supports_problem_type(task.problem_type):
                continue
            
            if task.num_qubits > backend.get_available_qubits():
                continue
            
            # Capability score (problem-backend match quality)
            capability = self._compute_capability_score(task, backend_type)
            
            # Availability score (current load)
            availability = 1.0 - backend.get_current_load()
            
            # Cost efficiency score
            cost = backend.estimate_cost(task)
            max_cost = 10.0  # Normalize to $10 max
            cost_score = 1.0 - min(cost / max_cost, 1.0)
            
            # UPG consciousness weighting
            coherent_score = capability * availability
            exploratory_score = cost_score
            
            weighted_score = self.upg.consciousness_weighted_blend(
                coherent_score,
                exploratory_score
            )
            
            # Apply reality distortion enhancement
            enhanced_score = self.upg.apply_reality_distortion(weighted_score)
            
            scores[backend_type] = enhanced_score
            
            logger.debug(
                f"Backend {backend_type.value}: "
                f"capability={capability:.3f}, availability={availability:.3f}, "
                f"cost_score={cost_score:.3f}, final={enhanced_score:.3f}"
            )
        
        if not scores:
            # Fallback to local simulator
            return BackendType.LOCAL_SIMULATOR
        
        selected = max(scores, key=scores.get)
        logger.info(f"Selected backend: {selected.value} (score: {scores[selected]:.3f})")
        
        return selected
    
    def _compute_capability_score(self, task: QuantumTask, backend_type: BackendType) -> float:
        """Compute capability score for backend-problem match."""
        scores = {
            # D-Wave is optimal for annealing problems
            (BackendType.DWAVE_ANNEALING, ProblemType.ISING): 1.0,
            (BackendType.DWAVE_ANNEALING, ProblemType.QUBO): 1.0,
            (BackendType.DWAVE_ANNEALING, ProblemType.MAXCUT): 0.9,
            
            # IBM is good for gate-based algorithms
            (BackendType.IBM_GATE, ProblemType.ISING): 0.8,
            (BackendType.IBM_GATE, ProblemType.QUBO): 0.8,
            (BackendType.IBM_GATE, ProblemType.MAXCUT): 0.85,
            
            # AWS Braket is versatile
            (BackendType.AWS_BRAKET, ProblemType.ISING): 0.85,
            (BackendType.AWS_BRAKET, ProblemType.QUBO): 0.85,
            (BackendType.AWS_BRAKET, ProblemType.MAXCUT): 0.85,
            
            # Local simulator is always available but slower
            (BackendType.LOCAL_SIMULATOR, ProblemType.ISING): 0.7,
            (BackendType.LOCAL_SIMULATOR, ProblemType.QUBO): 0.7,
            (BackendType.LOCAL_SIMULATOR, ProblemType.MAXCUT): 0.7,
        }
        
        return scores.get((backend_type, task.problem_type), 0.5)
    
    def _select_round_robin(self) -> BackendType:
        """Select backend using round-robin."""
        available = list(self.backends.keys())
        selected = available[self._round_robin_idx % len(available)]
        self._round_robin_idx += 1
        return selected
    
    def _select_cheapest(self, task: QuantumTask) -> BackendType:
        """Select cheapest available backend."""
        costs = {
            bt: backend.estimate_cost(task)
            for bt, backend in self.backends.items()
            if backend.supports_problem_type(task.problem_type)
        }
        return min(costs, key=costs.get) if costs else BackendType.LOCAL_SIMULATOR
    
    def _select_fastest(self, task: QuantumTask) -> BackendType:
        """Select fastest available backend."""
        # Prefer real quantum hardware over simulators for large problems
        if task.num_qubits > 10:
            if BackendType.DWAVE_ANNEALING in self.backends:
                return BackendType.DWAVE_ANNEALING
            if BackendType.IBM_GATE in self.backends:
                return BackendType.IBM_GATE
        
        return BackendType.LOCAL_SIMULATOR
    
    def submit(self, task: QuantumTask, 
               callback: Optional[Callable[[QuantumResult], None]] = None) -> str:
        """
        Submit a quantum task for execution.
        
        Args:
            task: The quantum task
            callback: Optional callback function for result
            
        Returns:
            Orchestrator job ID
        """
        # Generate orchestrator job ID
        job_id = str(uuid.uuid4())
        
        # Select backend
        backend_type = self.select_backend(task)
        task.backend_type = backend_type
        
        # Store task info
        self.active_tasks[job_id] = {
            'task': task,
            'backend_type': backend_type,
            'status': TaskStatus.PENDING,
            'submitted_at': datetime.now(),
            'callback': callback,
            'retries': 0,
        }
        
        # Submit to thread pool
        future = self.executor.submit(self._execute_task, job_id)
        self.futures[job_id] = future
        
        self.stats['tasks_submitted'] += 1
        self.stats['backend_usage'][backend_type.value] += 1
        
        logger.info(f"Task {task.task_id} submitted as job {job_id} to {backend_type.value}")
        
        return job_id
    
    def _execute_task(self, job_id: str) -> QuantumResult:
        """Execute a task on the selected backend."""
        task_info = self.active_tasks[job_id]
        task = task_info['task']
        backend_type = task_info['backend_type']
        
        try:
            task_info['status'] = TaskStatus.RUNNING
            
            # Get backend
            backend = self.backends[backend_type]
            
            # Submit to backend
            backend_job_id = backend.submit_task(task)
            task_info['backend_job_id'] = backend_job_id
            
            # Wait for result
            result = backend.get_result(backend_job_id, task.timeout_seconds)
            
            # Store result
            self.completed_results[job_id] = result
            task_info['status'] = TaskStatus.COMPLETED
            task_info['completed_at'] = datetime.now()
            
            # Update stats
            self.stats['tasks_completed'] += 1
            self.stats['total_cost'] += backend.estimate_cost(task)
            
            # Call callback if provided
            if task_info.get('callback'):
                task_info['callback'](result)
            
            logger.info(f"Job {job_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            
            # Retry logic
            if self.config.retry_on_failure and task_info['retries'] < self.config.max_retries:
                task_info['retries'] += 1
                logger.info(f"Retrying job {job_id} (attempt {task_info['retries']})")
                return self._execute_task(job_id)
            
            task_info['status'] = TaskStatus.FAILED
            task_info['error'] = str(e)
            self.stats['tasks_failed'] += 1
            
            raise
    
    def get_result(self, job_id: str, timeout: float = None) -> QuantumResult:
        """
        Get result for a submitted task.
        
        Args:
            job_id: The orchestrator job ID
            timeout: Maximum wait time
            
        Returns:
            Quantum computation result
        """
        if job_id in self.completed_results:
            return self.completed_results[job_id]
        
        if job_id not in self.futures:
            raise ValueError(f"Unknown job {job_id}")
        
        timeout = timeout or self.config.default_timeout
        
        try:
            result = self.futures[job_id].result(timeout=timeout)
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to get result for job {job_id}: {e}")
    
    def get_status(self, job_id: str) -> TaskStatus:
        """Get status of a submitted task."""
        if job_id not in self.active_tasks:
            raise ValueError(f"Unknown job {job_id}")
        
        return self.active_tasks[job_id]['status']
    
    def cancel(self, job_id: str) -> bool:
        """Cancel a submitted task."""
        if job_id not in self.active_tasks:
            return False
        
        task_info = self.active_tasks[job_id]
        
        # Cancel future if still pending
        if job_id in self.futures:
            self.futures[job_id].cancel()
        
        # Cancel backend task if submitted
        if 'backend_job_id' in task_info:
            backend = self.backends[task_info['backend_type']]
            backend.cancel_task(task_info['backend_job_id'])
        
        task_info['status'] = TaskStatus.CANCELLED
        return True
    
    def submit_batch(self, tasks: List[QuantumTask],
                    wait: bool = True) -> Dict[str, QuantumResult]:
        """
        Submit multiple tasks as a batch.
        
        Args:
            tasks: List of quantum tasks
            wait: Whether to wait for all results
            
        Returns:
            Dictionary mapping job IDs to results (if wait=True)
        """
        job_ids = [self.submit(task) for task in tasks]
        
        if not wait:
            return {jid: None for jid in job_ids}
        
        results = {}
        for job_id in job_ids:
            try:
                results[job_id] = self.get_result(job_id)
            except Exception as e:
                logger.error(f"Failed to get result for {job_id}: {e}")
                results[job_id] = None
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            **self.stats,
            'active_tasks': len([t for t in self.active_tasks.values() 
                                if t['status'] == TaskStatus.RUNNING]),
            'pending_tasks': len([t for t in self.active_tasks.values() 
                                 if t['status'] == TaskStatus.PENDING]),
            'available_backends': list(self.backends.keys()),
        }
    
    def shutdown(self, wait: bool = True) -> None:
        """
        Shutdown the orchestrator.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        logger.info("Shutting down orchestrator...")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=wait)
        
        # Disconnect backends
        for backend in self.backends.values():
            backend.disconnect()
        
        logger.info("Orchestrator shutdown complete")
        logger.info(f"Final statistics: {self.get_statistics()}")


# Convenience function for quick task submission
def run_quantum_task(
    problem_type: str,
    problem_data: Dict[str, Any],
    num_qubits: int,
    num_reads: int = 1000,
    backend: Optional[str] = None,
    upg_optimization: bool = True
) -> QuantumResult:
    """
    Convenience function for running a single quantum task.
    
    Args:
        problem_type: Type of problem ('ising', 'qubo', 'maxcut')
        problem_data: Problem-specific data
        num_qubits: Number of qubits
        num_reads: Number of samples
        backend: Specific backend to use (optional)
        upg_optimization: Enable UPG optimization
        
    Returns:
        Quantum computation result
    """
    # Create orchestrator
    config = OrchestratorConfig(enable_local=True)
    orchestrator = UPGHybridOrchestrator(config)
    
    # Create task
    task = QuantumTask(
        task_id=str(uuid.uuid4()),
        problem_type=ProblemType[problem_type.upper()],
        problem_data=problem_data,
        num_qubits=num_qubits,
        num_reads=num_reads,
        upg_optimization=upg_optimization,
    )
    
    if backend:
        task.backend_type = BackendType[backend.upper()]
    
    # Submit and wait
    job_id = orchestrator.submit(task)
    result = orchestrator.get_result(job_id)
    
    # Cleanup
    orchestrator.shutdown()
    
    return result

