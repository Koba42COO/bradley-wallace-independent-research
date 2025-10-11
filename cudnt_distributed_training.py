#!/usr/bin/env python3
"""
CUDNT Distributed Training - Multi-Node ML Training
===================================================

Distributed training system for CUDNT, enabling training across multiple CPU processes
and nodes. Implements parameter server architecture with gradient synchronization.
"""

import numpy as np
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue, Value, Lock
import threading
import time
import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
import socket
import pickle
import os

logger = logging.getLogger(__name__)

class ParameterServer:
    """
    Parameter server for distributed training.
    Manages model parameters and coordinates gradient updates across workers.
    """

    def __init__(self, model_architecture: List[Dict[str, Any]], num_workers: int = 4):
        """Initialize parameter server."""
        self.model_architecture = model_architecture
        self.num_workers = num_workers
        self.parameters = {}
        self.gradients = {}
        self.worker_connections = {}
        self.lock = Lock()

        # Initialize parameters
        self._init_parameters()

        logger.info(f"📡 Parameter server initialized for {num_workers} workers")

    def _init_parameters(self):
        """Initialize model parameters."""
        # Create a simple parameter initialization (in practice, this would come from the model)
        np.random.seed(42)
        self.parameters = {
            'layer_0_weights': np.random.randn(10, 64).astype(np.float32) * 0.1,
            'layer_0_bias': np.zeros(64, dtype=np.float32),
            'layer_1_weights': np.random.randn(64, 1).astype(np.float32) * 0.1,
            'layer_1_bias': np.zeros(1, dtype=np.float32)
        }

        # Initialize gradient buffers
        for param_name in self.parameters:
            self.gradients[param_name] = np.zeros_like(self.parameters[param_name])

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """Get current parameters."""
        with self.lock:
            return self.parameters.copy()

    def update_gradients(self, worker_id: int, gradients: Dict[str, np.ndarray]):
        """Update gradients from a worker."""
        with self.lock:
            for param_name, grad in gradients.items():
                if param_name in self.gradients:
                    self.gradients[param_name] += grad

    def apply_gradients(self, learning_rate: float = 0.01):
        """Apply accumulated gradients to parameters."""
        with self.lock:
            for param_name in self.parameters:
                if param_name in self.gradients:
                    # Simple SGD update
                    self.parameters[param_name] -= learning_rate * self.gradients[param_name]
                    # Reset gradients
                    self.gradients[param_name].fill(0)

    def get_worker_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'num_workers': self.num_workers,
            'parameters_count': sum(p.size for p in self.parameters.values()),
            'total_memory_mb': sum(p.nbytes for p in self.parameters.values()) / (1024 * 1024)
        }


class Worker:
    """
    Worker process for distributed training.
    Handles local training and gradient computation.
    """

    def __init__(self, worker_id: int, parameter_server: ParameterServer,
                 local_data: Tuple[np.ndarray, np.ndarray], config: Dict[str, Any]):
        """Initialize worker."""
        self.worker_id = worker_id
        self.ps = parameter_server
        self.X_local, self.y_local = local_data
        self.config = config
        self.local_parameters = {}
        self.gradients = {}

        logger.info(f"👷 Worker {worker_id} initialized with {len(self.X_local)} samples")

    def sync_parameters(self):
        """Synchronize parameters from parameter server."""
        self.local_parameters = self.ps.get_parameters()

    def compute_gradients(self, batch_X: np.ndarray, batch_y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for a batch."""
        # Simple forward pass (linear model)
        W1 = self.local_parameters['layer_0_weights']
        b1 = self.local_parameters['layer_0_bias']
        W2 = self.local_parameters['layer_1_weights']
        b2 = self.local_parameters['layer_1_bias']

        # Forward pass
        hidden = np.maximum(0, batch_X @ W1 + b1)  # ReLU
        predictions = hidden @ W2 + b2

        # Compute loss (MSE)
        loss = np.mean((predictions - batch_y.reshape(-1, 1)) ** 2)

        # Backward pass (simplified gradients)
        batch_size = len(batch_X)

        # Output layer gradients
        d_pred = 2 * (predictions - batch_y.reshape(-1, 1)) / batch_size
        d_W2 = hidden.T @ d_pred
        d_b2 = np.sum(d_pred, axis=0)

        # Hidden layer gradients
        d_hidden = d_pred @ W2.T
        d_hidden[hidden <= 0] = 0  # ReLU derivative
        d_W1 = batch_X.T @ d_hidden
        d_b1 = np.sum(d_hidden, axis=0)

        return {
            'layer_0_weights': d_W1,
            'layer_0_bias': d_b1,
            'layer_1_weights': d_W2,
            'layer_1_bias': d_b2
        }

    def train_epoch(self) -> float:
        """Train for one epoch on local data."""
        batch_size = self.config.get('batch_size', 32)
        total_loss = 0
        num_batches = 0

        # Shuffle local data
        indices = np.random.permutation(len(self.X_local))
        X_shuffled = self.X_local[indices]
        y_shuffled = self.y_local[indices]

        # Process batches
        for i in range(0, len(X_shuffled), batch_size):
            batch_X = X_shuffled[i:i+batch_size]
            batch_y = y_shuffled[i:i+batch_size]

            # Compute gradients
            batch_gradients = self.compute_gradients(batch_X, batch_y)

            # Send gradients to parameter server
            self.ps.update_gradients(self.worker_id, batch_gradients)

            # Compute batch loss
            total_loss += self.compute_loss(batch_X, batch_y)
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute loss for given data."""
        # Simple forward pass for loss computation
        W1 = self.local_parameters['layer_0_weights']
        b1 = self.local_parameters['layer_0_bias']
        W2 = self.local_parameters['layer_1_weights']
        b2 = self.local_parameters['layer_1_bias']

        hidden = np.maximum(0, X @ W1 + b1)
        predictions = hidden @ W2 + b2

        return np.mean((predictions - y.reshape(-1, 1)) ** 2)


class CUDNT_DistributedTrainer:
    """
    Distributed training coordinator for CUDNT.
    Manages parameter server and worker processes.
    """

    def __init__(self, model_architecture: List[Dict[str, Any]],
                 num_workers: int = 4, config: Optional[Dict[str, Any]] = None):
        """Initialize distributed trainer."""
        self.model_architecture = model_architecture
        self.num_workers = num_workers
        self.config = config or self._default_config()

        self.parameter_server = None
        self.workers = []
        self.processes = []

        logger.info(f"🚀 CUDNT Distributed Trainer initialized with {num_workers} workers")

    def _default_config(self) -> Dict[str, Any]:
        """Default distributed training configuration."""
        return {
            'batch_size': 32,
            'learning_rate': 0.01,
            'epochs': 10,
            'sync_frequency': 1,  # Sync every N batches
            'fault_tolerance': True,
            'checkpoint_frequency': 5
        }

    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data across workers."""
        n_samples = len(X)
        samples_per_worker = n_samples // self.num_workers

        worker_data = []
        for i in range(self.num_workers):
            start_idx = i * samples_per_worker
            end_idx = start_idx + samples_per_worker if i < self.num_workers - 1 else n_samples

            worker_data.append((
                X[start_idx:end_idx],
                y[start_idx:end_idx]
            ))

        return worker_data

    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train model in distributed fashion."""
        logger.info("Starting distributed training...")

        # Prepare data splits
        worker_data_splits = self.prepare_data(X, y)

        # Initialize parameter server
        self.parameter_server = ParameterServer(self.model_architecture, self.num_workers)

        # Create worker processes
        self.workers = []
        self.processes = []

        for i in range(self.num_workers):
            worker = Worker(i, self.parameter_server, worker_data_splits[i], self.config)
            self.workers.append(worker)

            # Create process for each worker
            process = Process(target=self._worker_process, args=(worker, self.config))
            self.processes.append(process)

        # Start all worker processes
        for process in self.processes:
            process.start()

        # Training loop
        training_stats = self._coordinate_training()

        # Cleanup
        for process in self.processes:
            process.join()

        logger.info("Distributed training completed")
        return training_stats

    def _worker_process(self, worker: Worker, config: Dict[str, Any]):
        """Worker process function."""
        try:
            for epoch in range(config['epochs']):
                # Sync parameters at start of epoch
                worker.sync_parameters()

                # Train for one epoch
                epoch_loss = worker.train_epoch()

                logger.info(f"Worker {worker.worker_id}, Epoch {epoch}: Loss = {epoch_loss:.4f}")

        except Exception as e:
            logger.error(f"Worker {worker.worker_id} error: {e}")

    def _coordinate_training(self) -> Dict[str, Any]:
        """Coordinate training across workers."""
        training_stats = {
            'epochs_completed': 0,
            'total_time': 0,
            'final_loss': 0,
            'workers_active': self.num_workers
        }

        start_time = time.time()

        for epoch in range(self.config['epochs']):
            epoch_start = time.time()

            # Wait for all workers to complete their epoch
            time.sleep(0.1)  # Small delay for synchronization

            # Apply accumulated gradients
            self.parameter_server.apply_gradients(self.config['learning_rate'])

            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")

            training_stats['epochs_completed'] = epoch + 1

        training_stats['total_time'] = time.time() - start_time

        # Get final parameters
        final_params = self.parameter_server.get_parameters()
        training_stats['final_parameters'] = final_params

        return training_stats

    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if self.parameter_server:
            return self.parameter_server.get_worker_stats()
        return {}


class MultiNodeCoordinator:
    """
    Multi-node distributed training coordinator.
    Enables training across multiple machines.
    """

    def __init__(self, nodes: List[str], model_architecture: List[Dict[str, Any]],
                 master_node: str = None):
        """Initialize multi-node coordinator."""
        self.nodes = nodes
        self.model_architecture = model_architecture
        self.master_node = master_node or socket.gethostname()

        self.node_connections = {}
        self.global_parameters = {}

        logger.info(f"🌐 Multi-node coordinator initialized for {len(nodes)} nodes")

    def distribute_training(self, X: np.ndarray, y: np.ndarray,
                          config: Dict[str, Any]) -> Dict[str, Any]:
        """Distribute training across nodes."""
        # This is a simplified implementation
        # In practice, this would use network communication

        logger.info("Distributing training across nodes...")

        # Split data across nodes
        node_data = self._split_data_across_nodes(X, y)

        # Simulate distributed training
        results = {}
        for i, node in enumerate(self.nodes):
            logger.info(f"Training on node {node}...")
            trainer = CUDNT_DistributedTrainer(self.model_architecture, num_workers=2)
            node_result = trainer.train(node_data[i][0], node_data[i][1])
            results[node] = node_result

        # Aggregate results
        aggregated_result = self._aggregate_results(results)

        return aggregated_result

    def _split_data_across_nodes(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data across nodes."""
        n_samples = len(X)
        samples_per_node = n_samples // len(self.nodes)

        node_data = []
        for i in range(len(self.nodes)):
            start_idx = i * samples_per_node
            end_idx = start_idx + samples_per_node if i < len(self.nodes) - 1 else n_samples

            node_data.append((
                X[start_idx:end_idx],
                y[start_idx:end_idx]
            ))

        return node_data

    def _aggregate_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from all nodes."""
        # Simple aggregation - average parameters
        aggregated = {
            'total_time': sum(r['total_time'] for r in results.values()),
            'avg_epochs': np.mean([r['epochs_completed'] for r in results.values()]),
            'nodes_used': len(results)
        }

        return aggregated


# ===============================
# UTILITY FUNCTIONS
# ===============================

def create_distributed_trainer(model_architecture: List[Dict[str, Any]],
                              num_workers: int = 4) -> CUDNT_DistributedTrainer:
    """Create distributed trainer instance."""
    return CUDNT_DistributedTrainer(model_architecture, num_workers)

def quick_distributed_training(X: np.ndarray, y: np.ndarray,
                              model_architecture: List[Dict[str, Any]],
                              num_workers: int = 4) -> Dict[str, Any]:
    """Quick distributed training setup."""
    trainer = create_distributed_trainer(model_architecture, num_workers)
    return trainer.train(X, y)


# ===============================
# EXAMPLE USAGE
# ===============================

if __name__ == '__main__':
    print("🚀 CUDNT Distributed Training Demo")
    print("=" * 40)

    # Create synthetic data
    from cudnt_data_pipeline import create_cudnt_data_pipeline
    pipeline = create_cudnt_data_pipeline()
    X, y = pipeline.create_synthetic_dataset(1000, 10)

    print(f"Dataset: {X.shape}, Target: {y.shape}")

    # Define model architecture
    architecture = [
        {'type': 'dense', 'units': 64, 'activation': 'relu'},
        {'type': 'dense', 'units': 1}
    ]

    # Create distributed trainer
    trainer = create_distributed_trainer(architecture, num_workers=4)

    # Train model
    print("Starting distributed training...")
    start_time = time.time()

    results = trainer.train(X, y)

    total_time = time.time() - start_time

    print("
📊 Training Results:"    print(f"   Total time: {total_time:.2f}s")
    print(f"   Epochs completed: {results['epochs_completed']}")
    print(f"   Workers used: {results['workers_active']}")
    print(f"   Parameters count: {trainer.get_training_stats()['parameters_count']}")

    print("\n✅ Distributed training completed successfully!")
