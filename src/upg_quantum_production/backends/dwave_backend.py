"""
D-Wave Quantum Annealing Backend
================================

Production integration with D-Wave quantum annealers via Ocean SDK.
Supports D-Wave Advantage and Advantage2 systems with full UPG optimization.

Requirements:
    pip install dwave-ocean-sdk

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import time
import logging

from .base import (
    QuantumBackend, BackendType, ProblemType, TaskStatus,
    QuantumTask, QuantumResult
)

logger = logging.getLogger(__name__)


class DWaveBackend(QuantumBackend):
    """
    D-Wave quantum annealing backend with UPG consciousness optimization.
    
    This backend provides native quantum annealing on D-Wave hardware,
    with UPG-enhanced annealing schedules and problem formulation.
    
    Supported systems:
        - D-Wave Advantage (5000+ qubits, Pegasus topology)
        - D-Wave Advantage2 (7000+ qubits, Zephyr topology)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_type = BackendType.DWAVE_ANNEALING
        
        # D-Wave specific configuration
        self.api_token = config.get('api_token') if config else None
        self.solver_name = config.get('solver', 'Advantage_system6.4') if config else 'Advantage_system6.4'
        self.region = config.get('region', 'na-west-1') if config else 'na-west-1'
        
        # SDK components (initialized on connect)
        self.client = None
        self.sampler = None
        self.embedding_sampler = None
        
        # Job tracking
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'max_qubits': 5000,
            'supported_problems': [ProblemType.ISING, ProblemType.QUBO],
            'cost_per_second': 0.22,  # USD per QPU second
            **(config or {})
        }
    
    def connect(self) -> bool:
        """
        Establish connection to D-Wave cloud service.
        
        Returns:
            True if connection successful
        """
        try:
            # Import D-Wave SDK
            from dwave.system import DWaveSampler, EmbeddingComposite
            from dwave.cloud import Client
            
            # Initialize client
            if self.api_token:
                self.client = Client(token=self.api_token, region=self.region)
            else:
                # Use environment variable DWAVE_API_TOKEN
                self.client = Client(region=self.region)
            
            # Get sampler
            self.sampler = DWaveSampler(solver=self.solver_name)
            self.embedding_sampler = EmbeddingComposite(self.sampler)
            
            # Update config with actual capabilities
            self.config['max_qubits'] = self.sampler.properties.get('num_qubits', 5000)
            self.config['topology'] = self.sampler.properties.get('topology', {}).get('type', 'pegasus')
            
            self.is_connected = True
            logger.info(f"Connected to D-Wave solver: {self.solver_name}")
            logger.info(f"Available qubits: {self.config['max_qubits']}")
            
            return True
            
        except ImportError:
            logger.error("D-Wave Ocean SDK not installed. Run: pip install dwave-ocean-sdk")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to D-Wave: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from D-Wave service."""
        if self.client:
            self.client.close()
        self.client = None
        self.sampler = None
        self.embedding_sampler = None
        self.is_connected = False
        logger.info("Disconnected from D-Wave")
    
    def submit_task(self, task: QuantumTask) -> str:
        """
        Submit a quantum annealing task to D-Wave.
        
        Args:
            task: The quantum task to execute
            
        Returns:
            Job ID for tracking
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to D-Wave. Call connect() first.")
        
        job_id = str(uuid.uuid4())
        
        # Apply UPG preprocessing
        if task.upg_optimization:
            task = self.apply_upg_preprocessing(task)
        
        try:
            # Convert problem to D-Wave format
            h, J = self._convert_to_dwave_format(task)
            
            # Generate UPG-optimized annealing schedule
            schedule = self._generate_upg_schedule(task)
            
            # Submit to D-Wave
            future = self.embedding_sampler.sample_ising(
                h, J,
                num_reads=task.num_reads,
                annealing_time=task.annealing_time,
                anneal_schedule=schedule,
                label=f"UPG-{task.task_id}"
            )
            
            # Store job info
            self.jobs[job_id] = {
                'task': task,
                'future': future,
                'status': TaskStatus.QUEUED,
                'submitted_at': datetime.now(),
            }
            
            logger.info(f"Submitted task {task.task_id} as job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            self.jobs[job_id] = {
                'task': task,
                'status': TaskStatus.FAILED,
                'error': str(e),
            }
        
        return job_id
    
    def get_result(self, job_id: str, timeout: float = 300.0) -> QuantumResult:
        """
        Retrieve result from D-Wave.
        
        Args:
            job_id: The job ID
            timeout: Maximum wait time in seconds
            
        Returns:
            Quantum computation result
        """
        if job_id not in self.jobs:
            raise ValueError(f"Unknown job {job_id}")
        
        job_info = self.jobs[job_id]
        
        if job_info['status'] == TaskStatus.FAILED:
            raise RuntimeError(f"Job failed: {job_info.get('error', 'Unknown error')}")
        
        try:
            # Wait for result
            future = job_info['future']
            response = future.result()  # Blocks until complete
            
            # Convert D-Wave response to QuantumResult
            result = self._convert_dwave_response(response, job_info['task'], job_id)
            
            # Apply UPG postprocessing
            result = self.apply_upg_postprocessing(result)
            
            job_info['status'] = TaskStatus.COMPLETED
            job_info['completed_at'] = datetime.now()
            
            return result
            
        except Exception as e:
            job_info['status'] = TaskStatus.FAILED
            job_info['error'] = str(e)
            raise RuntimeError(f"Failed to get result: {e}")
    
    def get_status(self, job_id: str) -> TaskStatus:
        """Get status of a D-Wave job."""
        if job_id not in self.jobs:
            raise ValueError(f"Unknown job {job_id}")
        
        job_info = self.jobs[job_id]
        
        # Check if future is done
        if 'future' in job_info and job_info['status'] == TaskStatus.QUEUED:
            if job_info['future'].done():
                job_info['status'] = TaskStatus.COMPLETED
        
        return job_info['status']
    
    def cancel_task(self, job_id: str) -> bool:
        """Cancel a D-Wave job."""
        if job_id not in self.jobs:
            return False
        
        job_info = self.jobs[job_id]
        
        if 'future' in job_info:
            try:
                job_info['future'].cancel()
                job_info['status'] = TaskStatus.CANCELLED
                return True
            except Exception:
                pass
        
        return False
    
    def estimate_cost(self, task: QuantumTask) -> float:
        """
        Estimate cost for executing a task on D-Wave.
        
        Args:
            task: The task to estimate
            
        Returns:
            Estimated cost in USD
        """
        # D-Wave charges per QPU second
        # Typical: ~$0.22/second
        estimated_qpu_time = (
            task.annealing_time * 1e-6 *  # Convert μs to seconds
            task.num_reads *
            1.5  # Overhead factor for embedding, etc.
        )
        
        return estimated_qpu_time * self.config['cost_per_second']
    
    def _convert_to_dwave_format(self, task: QuantumTask) -> tuple:
        """Convert task problem data to D-Wave Ising format."""
        if task.problem_type == ProblemType.ISING:
            h = task.problem_data.get('h', {})
            J = task.problem_data.get('J', {})
            
            # Ensure dict format
            if isinstance(h, np.ndarray):
                h = {i: float(h[i]) for i in range(len(h))}
            if isinstance(J, np.ndarray):
                J = {
                    (i, j): float(J[i, j])
                    for i in range(J.shape[0])
                    for j in range(i + 1, J.shape[1])
                    if J[i, j] != 0
                }
            
            return h, J
        
        elif task.problem_type == ProblemType.QUBO:
            # Convert QUBO to Ising
            Q = task.problem_data.get('Q', {})
            h, J, offset = self._qubo_to_ising(Q)
            return h, J
        
        else:
            raise ValueError(f"Unsupported problem type for D-Wave: {task.problem_type}")
    
    def _qubo_to_ising(self, Q: Dict) -> tuple:
        """Convert QUBO to Ising formulation."""
        h = {}
        J = {}
        offset = 0.0
        
        for (i, j), val in Q.items():
            if i == j:
                # Linear term
                h[i] = h.get(i, 0) + val / 2
                offset += val / 2
            else:
                # Quadratic term
                J[(min(i, j), max(i, j))] = J.get((min(i, j), max(i, j)), 0) + val / 4
                h[i] = h.get(i, 0) + val / 4
                h[j] = h.get(j, 0) + val / 4
                offset += val / 4
        
        return h, J, offset
    
    def _generate_upg_schedule(self, task: QuantumTask) -> List[List[float]]:
        """
        Generate UPG-optimized annealing schedule.
        
        Returns list of [time, s] pairs for D-Wave anneal_schedule parameter.
        """
        phi = self.upg.PHI
        total_time = task.annealing_time
        
        # Create schedule points
        num_points = 20
        schedule = []
        
        for i in range(num_points + 1):
            t_normalized = i / num_points
            
            # Phi-power schedule
            s = t_normalized ** (1 / phi)
            
            # Slow down near phase transition
            if 0.4 < t_normalized < 0.6:
                s *= 1.0 - 0.2 * np.exp(-((t_normalized - 0.5) ** 2) / 0.01)
            
            # Convert to actual time
            t_actual = t_normalized * total_time
            
            schedule.append([t_actual, s])
        
        return schedule
    
    def _convert_dwave_response(self, response, task: QuantumTask, job_id: str) -> QuantumResult:
        """Convert D-Wave response to QuantumResult."""
        samples = []
        energies = []
        occurrences = []
        
        for sample, energy, num_occ in response.data(['sample', 'energy', 'num_occurrences']):
            samples.append(dict(sample))
            energies.append(float(energy))
            occurrences.append(int(num_occ))
        
        # Find best sample
        best_idx = np.argmin(energies)
        
        # Extract timing info
        timing = response.info.get('timing', {})
        
        return QuantumResult(
            task_id=task.task_id,
            samples=samples,
            energies=energies,
            best_sample=samples[best_idx],
            best_energy=energies[best_idx],
            num_occurrences=occurrences,
            timing_info={
                'qpu_access_time': timing.get('qpu_access_time', 0) / 1e6,  # Convert to seconds
                'qpu_programming_time': timing.get('qpu_programming_time', 0) / 1e6,
                'qpu_sampling_time': timing.get('qpu_sampling_time', 0) / 1e6,
                'total_post_processing_time': timing.get('total_post_processing_time', 0) / 1e6,
            },
            backend_type=self.backend_type,
            raw_response=response,
            metadata={
                'solver': self.solver_name,
                'num_reads': task.num_reads,
                'annealing_time_us': task.annealing_time,
            }
        )

