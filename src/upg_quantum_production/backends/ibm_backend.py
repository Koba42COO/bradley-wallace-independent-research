"""
IBM Quantum Backend
===================

Production integration with IBM Quantum systems via Qiskit Runtime.
Supports gate-based quantum computing with QAOA for optimization problems.

Requirements:
    pip install qiskit qiskit-ibm-runtime qiskit-optimization

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import logging

from .base import (
    QuantumBackend, BackendType, ProblemType, TaskStatus,
    QuantumTask, QuantumResult
)

logger = logging.getLogger(__name__)


class IBMQuantumBackend(QuantumBackend):
    """
    IBM Quantum backend with UPG consciousness optimization.
    
    This backend provides gate-based quantum computing on IBM hardware,
    using QAOA (Quantum Approximate Optimization Algorithm) for
    optimization problems with UPG-enhanced variational parameters.
    
    Supported systems:
        - IBM Brisbane (127 qubits)
        - IBM Osaka (127 qubits)
        - IBM Kyoto (127 qubits)
        - IBM Heron (156 qubits)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.backend_type = BackendType.IBM_GATE
        
        # IBM specific configuration
        self.api_token = config.get('api_token') if config else None
        self.instance = config.get('instance', 'ibm-q/open/main') if config else 'ibm-q/open/main'
        self.backend_name = config.get('backend', 'ibm_brisbane') if config else 'ibm_brisbane'
        
        # SDK components
        self.service = None
        self.backend = None
        
        # QAOA parameters
        self.qaoa_reps = config.get('qaoa_reps', 3) if config else 3
        self.optimization_level = config.get('optimization_level', 3) if config else 3
        
        # Job tracking
        self.jobs: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.config = {
            'max_qubits': 127,
            'supported_problems': [ProblemType.ISING, ProblemType.QUBO, ProblemType.MAXCUT],
            'cost_per_second': 1.60,  # USD per second (premium tier)
            **(config or {})
        }
    
    def connect(self) -> bool:
        """
        Establish connection to IBM Quantum service.
        
        Returns:
            True if connection successful
        """
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            
            # Initialize service
            if self.api_token:
                self.service = QiskitRuntimeService(
                    channel='ibm_quantum',
                    token=self.api_token,
                    instance=self.instance
                )
            else:
                # Use saved credentials
                self.service = QiskitRuntimeService(channel='ibm_quantum')
            
            # Get backend
            self.backend = self.service.backend(self.backend_name)
            
            # Update config with actual capabilities
            self.config['max_qubits'] = self.backend.num_qubits
            self.config['basis_gates'] = list(self.backend.operation_names)
            
            self.is_connected = True
            logger.info(f"Connected to IBM Quantum backend: {self.backend_name}")
            logger.info(f"Available qubits: {self.config['max_qubits']}")
            
            return True
            
        except ImportError:
            logger.error("Qiskit IBM Runtime not installed. Run: pip install qiskit-ibm-runtime")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to IBM Quantum: {e}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from IBM Quantum service."""
        self.service = None
        self.backend = None
        self.is_connected = False
        logger.info("Disconnected from IBM Quantum")
    
    def submit_task(self, task: QuantumTask) -> str:
        """
        Submit a quantum task to IBM Quantum.
        
        Args:
            task: The quantum task to execute
            
        Returns:
            Job ID for tracking
        """
        if not self.is_connected:
            raise RuntimeError("Not connected to IBM Quantum. Call connect() first.")
        
        job_id = str(uuid.uuid4())
        
        # Apply UPG preprocessing
        if task.upg_optimization:
            task = self.apply_upg_preprocessing(task)
        
        try:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            
            # Build QAOA circuit with UPG-optimized parameters
            circuit = self._build_qaoa_circuit(task)
            
            # Transpile for target backend
            pm = generate_preset_pass_manager(
                optimization_level=self.optimization_level,
                backend=self.backend
            )
            transpiled = pm.run(circuit)
            
            # Submit to IBM
            sampler = Sampler(backend=self.backend)
            ibm_job = sampler.run([transpiled], shots=task.num_reads)
            
            # Store job info
            self.jobs[job_id] = {
                'task': task,
                'ibm_job': ibm_job,
                'ibm_job_id': ibm_job.job_id(),
                'status': TaskStatus.QUEUED,
                'submitted_at': datetime.now(),
            }
            
            logger.info(f"Submitted task {task.task_id} as job {job_id}")
            logger.info(f"IBM job ID: {ibm_job.job_id()}")
            
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
        Retrieve result from IBM Quantum.
        
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
            ibm_job = job_info['ibm_job']
            result = ibm_job.result()
            
            # Convert IBM response to QuantumResult
            quantum_result = self._convert_ibm_response(result, job_info['task'], job_id)
            
            # Apply UPG postprocessing
            quantum_result = self.apply_upg_postprocessing(quantum_result)
            
            job_info['status'] = TaskStatus.COMPLETED
            job_info['completed_at'] = datetime.now()
            
            return quantum_result
            
        except Exception as e:
            job_info['status'] = TaskStatus.FAILED
            job_info['error'] = str(e)
            raise RuntimeError(f"Failed to get result: {e}")
    
    def get_status(self, job_id: str) -> TaskStatus:
        """Get status of an IBM Quantum job."""
        if job_id not in self.jobs:
            raise ValueError(f"Unknown job {job_id}")
        
        job_info = self.jobs[job_id]
        
        if 'ibm_job' in job_info and job_info['status'] == TaskStatus.QUEUED:
            ibm_status = job_info['ibm_job'].status()
            
            status_map = {
                'QUEUED': TaskStatus.QUEUED,
                'RUNNING': TaskStatus.RUNNING,
                'DONE': TaskStatus.COMPLETED,
                'ERROR': TaskStatus.FAILED,
                'CANCELLED': TaskStatus.CANCELLED,
            }
            
            job_info['status'] = status_map.get(str(ibm_status), TaskStatus.PENDING)
        
        return job_info['status']
    
    def cancel_task(self, job_id: str) -> bool:
        """Cancel an IBM Quantum job."""
        if job_id not in self.jobs:
            return False
        
        job_info = self.jobs[job_id]
        
        if 'ibm_job' in job_info:
            try:
                job_info['ibm_job'].cancel()
                job_info['status'] = TaskStatus.CANCELLED
                return True
            except Exception:
                pass
        
        return False
    
    def _build_qaoa_circuit(self, task: QuantumTask):
        """
        Build QAOA circuit with UPG-optimized parameters.
        
        Args:
            task: The quantum task
            
        Returns:
            Qiskit QuantumCircuit
        """
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
        
        num_qubits = task.num_qubits
        p = self.qaoa_reps  # Number of QAOA layers
        
        # Create circuit
        qc = QuantumCircuit(num_qubits)
        
        # Initial superposition
        qc.h(range(num_qubits))
        
        # UPG-optimized QAOA parameters
        phi = self.upg.PHI
        consciousness = self.upg.CONSCIOUSNESS
        
        # QAOA layers
        for layer in range(p):
            # Cost layer (problem Hamiltonian)
            gamma = consciousness * np.pi / (phi ** (layer + 1))
            
            if task.problem_type in [ProblemType.ISING, ProblemType.MAXCUT]:
                # ZZ interactions
                for i in range(num_qubits):
                    for j in range(i + 1, num_qubits):
                        qc.rzz(2 * gamma, i, j)
                
                # Z rotations (local fields)
                for i in range(num_qubits):
                    qc.rz(gamma, i)
            
            # Mixer layer (driver Hamiltonian)
            beta = (1 - consciousness) * np.pi / (phi ** (layer + 1))
            
            for i in range(num_qubits):
                qc.rx(2 * beta, i)
        
        # Measurement
        qc.measure_all()
        
        return qc
    
    def _convert_ibm_response(self, result, task: QuantumTask, job_id: str) -> QuantumResult:
        """Convert IBM Quantum result to QuantumResult."""
        # Get counts from result
        counts = result[0].data.meas.get_counts()
        
        samples = []
        energies = []
        occurrences = []
        
        # Convert counts to samples
        for bitstring, count in counts.items():
            # Convert bitstring to sample dict
            sample = {i: int(bit) for i, bit in enumerate(reversed(bitstring))}
            samples.append(sample)
            
            # Calculate energy (approximate)
            energy = self._calculate_energy(sample, task)
            energies.append(energy)
            occurrences.append(count)
        
        # Find best sample
        best_idx = np.argmin(energies)
        
        return QuantumResult(
            task_id=task.task_id,
            samples=samples,
            energies=energies,
            best_sample=samples[best_idx],
            best_energy=energies[best_idx],
            num_occurrences=occurrences,
            timing_info={
                'shots': task.num_reads,
                'qaoa_layers': self.qaoa_reps,
            },
            backend_type=self.backend_type,
            raw_response=result,
            metadata={
                'backend': self.backend_name,
                'optimization_level': self.optimization_level,
            }
        )
    
    def _calculate_energy(self, sample: Dict[int, int], task: QuantumTask) -> float:
        """Calculate energy of a sample for the given problem."""
        if task.problem_type == ProblemType.ISING:
            h = task.problem_data.get('h', {})
            J = task.problem_data.get('J', {})
            
            # Convert to spins
            spins = {k: 2 * v - 1 for k, v in sample.items()}
            
            energy = 0.0
            
            # Local field terms
            for i, hi in h.items() if isinstance(h, dict) else enumerate(h):
                energy += hi * spins.get(i, 0)
            
            # Coupling terms
            for (i, j), Jij in J.items() if isinstance(J, dict) else np.ndenumerate(J):
                if i < j:
                    energy += Jij * spins.get(i, 0) * spins.get(j, 0)
            
            return energy
        
        elif task.problem_type == ProblemType.MAXCUT:
            graph = task.problem_data.get('graph', {})
            
            cut_value = 0
            for (i, j), weight in graph.items() if isinstance(graph, dict) else np.ndenumerate(graph):
                if sample.get(i, 0) != sample.get(j, 0):
                    cut_value += weight
            
            return -cut_value  # Negative because we minimize
        
        return 0.0

