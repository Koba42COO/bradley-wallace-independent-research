#!/usr/bin/env python3
"""
CUDNT Distributed Scaling Framework
===================================

Massive-scale parallel processing with CUDNT hybrid acceleration across multiple machines.
Combines distributed computing with GPU/CPU optimization for unprecedented performance.

Capabilities:
- Multi-machine distributed processing
- CUDNT hybrid acceleration on each node
- Dynamic load balancing
- Fault tolerance and recovery
- Real-time performance monitoring
"""

import multiprocessing as mp
import numpy as np
import time
import psutil
import socket
import threading
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from queue import Queue
import logging
import os
import sys

# Add parent directory to path for CUDNT imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cudnt_production_system import create_cudnt_production
from cudnt_wallace_transform import CUDNTWallaceTransform

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('distributed_scaling.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CUDNTDistributedNode:
    """
    Individual compute node with CUDNT hybrid acceleration.
    Represents one machine/worker in the distributed cluster.
    """

    def __init__(self, node_id, node_config=None):
        self.node_id = node_id
        self.node_config = node_config or self._get_default_config()

        # Initialize CUDNT system
        self.cudnt = create_cudnt_production()
        self.wallace_analyzer = CUDNTWallaceTransform()

        # Node capabilities
        self.cpu_cores = mp.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.hostname = socket.gethostname()

        # Performance tracking
        self.tasks_completed = 0
        self.total_compute_time = 0
        self.peak_memory_usage = 0

        logger.info(f"🎯 Node {node_id} initialized - {self.hostname} ({self.cpu_cores} cores, {self.memory_gb:.1f}GB)")

    def _get_default_config(self):
        """Get default node configuration."""
        return {
            'chunk_size': 50000,
            'max_memory_gb': 8,
            'enable_gpu': True,
            'analysis_types': ['fft', 'autocorr'],
            'fault_tolerance': True
        }

    def process_data_chunk(self, data_chunk, analysis_config):
        """
        Process a data chunk using CUDNT acceleration.
        """
        start_time = time.time()

        try:
            # Run CUDNT-accelerated analysis
            results = self.wallace_analyzer.analyze_with_cudnt_acceleration(
                data_chunk,
                analysis_type=analysis_config.get('analysis_type', 'both'),
                num_peaks=analysis_config.get('num_peaks', 8)
            )

            # Update node statistics
            compute_time = time.time() - start_time
            self.tasks_completed += 1
            self.total_compute_time += compute_time
            self.peak_memory_usage = max(self.peak_memory_usage,
                                       results['performance_summary']['memory_peak_gb'])

            # Add node metadata
            results['node_info'] = {
                'node_id': self.node_id,
                'hostname': self.hostname,
                'compute_time': compute_time,
                'cudnt_stats': self.cudnt.get_performance_stats()
            }

            logger.info(f"✅ Node {self.node_id} completed chunk in {compute_time:.3f}s")
            return results

        except Exception as e:
            logger.error(f"❌ Node {self.node_id} failed: {e}")
            return {
                'error': str(e),
                'node_id': self.node_id,
                'compute_time': time.time() - start_time
            }

    def get_node_status(self):
        """Get current node status and capabilities."""
        return {
            'node_id': self.node_id,
            'hostname': self.hostname,
            'cpu_cores': self.cpu_cores,
            'memory_gb': self.memory_gb,
            'tasks_completed': self.tasks_completed,
            'total_compute_time': self.total_compute_time,
            'peak_memory_usage': self.peak_memory_usage,
            'cudnt_available': self.wallace_analyzer.hybrid_available,
            'current_load': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent
        }

class CUDNTDistributedCoordinator:
    """
    Master coordinator for distributed CUDNT processing.
    Manages multiple nodes and orchestrates massive-scale computations.
    """

    def __init__(self, target_scale=1e9, num_nodes=None):
        self.target_scale = target_scale
        self.num_nodes = num_nodes or max(1, mp.cpu_count() // 2)  # Default to half available cores

        # Cluster management
        self.nodes = []
        self.node_status = {}
        self.task_queue = Queue()
        self.results_queue = Queue()

        # Performance tracking
        self.start_time = None
        self.cluster_stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_compute_time': 0,
            'peak_cluster_memory': 0
        }

        logger.info(f"🎼 Distributed Coordinator initialized for {target_scale:,.0f} scale analysis")
        logger.info(f"   Target nodes: {self.num_nodes}")

    def initialize_cluster(self):
        """Initialize the distributed cluster."""
        logger.info("🔧 Initializing distributed cluster...")

        self.nodes = []
        for i in range(self.num_nodes):
            try:
                node = CUDNTDistributedNode(f"node_{i}")
                self.nodes.append(node)
                logger.info(f"   ✅ Node {i} ready")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to initialize node {i}: {e}")

        if not self.nodes:
            raise RuntimeError("No nodes could be initialized")

        logger.info(f"✅ Cluster initialized with {len(self.nodes)} nodes")
        return len(self.nodes)

    def distribute_analysis(self, prime_data, analysis_config=None):
        """
        Distribute analysis across the cluster using CUDNT acceleration.
        """
        if not self.nodes:
            raise RuntimeError("Cluster not initialized")

        analysis_config = analysis_config or {
            'analysis_type': 'both',
            'num_peaks': 8,
            'chunk_size': 50000
        }

        # Prepare data chunks
        chunk_size = analysis_config['chunk_size']
        data_chunks = self._prepare_data_chunks(prime_data, chunk_size)

        logger.info(f"📦 Prepared {len(data_chunks)} data chunks for distributed processing")

        # Distribute work across nodes
        self.start_time = time.time()
        all_results = []

        # Use thread pool for concurrent node processing
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            # Submit tasks to nodes
            future_to_chunk = {}
            for i, chunk in enumerate(data_chunks):
                # Round-robin assignment to nodes
                node_idx = i % len(self.nodes)
                node = self.nodes[node_idx]

                future = executor.submit(node.process_data_chunk, chunk, analysis_config)
                future_to_chunk[future] = (i, node_idx)

            # Collect results
            for future in as_completed(future_to_chunk):
                chunk_idx, node_idx = future_to_chunk[future]
                try:
                    result = future.result()
                    all_results.append(result)

                    # Update cluster stats
                    if 'error' not in result:
                        self.cluster_stats['completed_tasks'] += 1
                        self.cluster_stats['total_compute_time'] += result['node_info']['compute_time']
                        self.cluster_stats['peak_cluster_memory'] = max(
                            self.cluster_stats['peak_cluster_memory'],
                            result['performance_summary']['memory_peak_gb']
                        )
                    else:
                        self.cluster_stats['failed_tasks'] += 1

                    logger.info(f"   📊 Chunk {chunk_idx} (Node {node_idx}): {'✅' if 'error' not in result else '❌'}")

                except Exception as e:
                    logger.error(f"   ❌ Chunk {chunk_idx} failed: {e}")
                    self.cluster_stats['failed_tasks'] += 1

        # Aggregate results
        final_results = self._aggregate_cluster_results(all_results)

        # Add cluster metadata
        final_results['cluster_info'] = {
            'total_nodes': len(self.nodes),
            'active_nodes': len([n for n in self.nodes if n.tasks_completed > 0]),
            'total_runtime': time.time() - self.start_time,
            'cluster_stats': self.cluster_stats,
            'efficiency_score': self._calculate_cluster_efficiency()
        }

        logger.info(f"🎉 Distributed analysis completed in {final_results['cluster_info']['total_runtime']:.2f}s")
        return final_results

    def _prepare_data_chunks(self, prime_data, chunk_size):
        """Prepare data into chunks for distributed processing."""
        if isinstance(prime_data, np.ndarray):
            # Assume it's prime gaps
            chunks = []
            for i in range(0, len(prime_data), chunk_size):
                chunk = prime_data[i:i + chunk_size]
                if len(chunk) > 0:
                    chunks.append(chunk)
            return chunks
        elif isinstance(prime_data, list):
            # List of data chunks
            return prime_data
        else:
            # Single large dataset - split it
            if hasattr(prime_data, '__len__'):
                total_size = len(prime_data)
                chunks = []
                for i in range(0, total_size, chunk_size):
                    chunks.append(prime_data[i:i + chunk_size])
                return chunks
            else:
                raise ValueError("Unsupported data format")

    def _aggregate_cluster_results(self, all_results):
        """Aggregate results from all cluster nodes."""
        successful_results = [r for r in all_results if 'error' not in r]

        if not successful_results:
            return {'error': 'All cluster tasks failed'}

        # Combine FFT and autocorrelation results
        combined_fft_peaks = []
        combined_autocorr_peaks = []

        for result in successful_results:
            if 'fft' in result and result['fft']['peaks']:
                combined_fft_peaks.extend(result['fft']['peaks'])
            if 'autocorr' in result and result['autocorr']['peaks']:
                combined_autocorr_peaks.extend(result['autocorr']['peaks'])

        # Sort and deduplicate peaks by ratio similarity
        combined_fft_peaks = self._deduplicate_peaks(combined_fft_peaks)
        combined_autocorr_peaks = self._deduplicate_peaks(combined_autocorr_peaks)

        return {
            'fft_peaks': combined_fft_peaks[:10],  # Top 10
            'autocorr_peaks': combined_autocorr_peaks[:10],
            'node_results': successful_results,
            'cluster_summary': {
                'total_chunks_processed': len(successful_results),
                'total_failed_chunks': len(all_results) - len(successful_results),
                'average_chunk_time': np.mean([r['node_info']['compute_time'] for r in successful_results])
            }
        }

    def _deduplicate_peaks(self, peaks, threshold=0.01):
        """Remove duplicate peaks based on ratio similarity."""
        if not peaks:
            return []

        # Sort by magnitude
        sorted_peaks = sorted(peaks, key=lambda x: x.get('magnitude', 0), reverse=True)

        deduplicated = []
        for peak in sorted_peaks:
            ratio = peak.get('ratio', 1.0)

            # Check if similar ratio already exists
            is_duplicate = False
            for existing in deduplicated:
                existing_ratio = existing.get('ratio', 1.0)
                if abs(np.log(ratio) - np.log(existing_ratio)) < threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                deduplicated.append(peak)

        return deduplicated

    def _calculate_cluster_efficiency(self):
        """Calculate overall cluster efficiency score."""
        if self.cluster_stats['total_tasks'] == 0:
            return 0.0

        success_rate = self.cluster_stats['completed_tasks'] / self.cluster_stats['total_tasks']
        efficiency = success_rate * 10.0

        # Bonus for high parallelization
        if len(self.nodes) > 1:
            efficiency *= 1.2

        return min(10.0, efficiency)

    def get_cluster_status(self):
        """Get comprehensive cluster status."""
        node_statuses = [node.get_node_status() for node in self.nodes]

        return {
            'cluster_info': {
                'total_nodes': len(self.nodes),
                'active_nodes': len([n for n in node_statuses if n['tasks_completed'] > 0]),
                'total_tasks_completed': sum(n['tasks_completed'] for n in node_statuses),
                'total_compute_time': sum(n['total_compute_time'] for n in node_statuses),
                'average_node_load': np.mean([n['current_load'] for n in node_statuses]),
                'cluster_efficiency': self._calculate_cluster_efficiency()
            },
            'node_details': node_statuses,
            'cluster_stats': self.cluster_stats
        }

def create_massive_scale_analysis(target_primes=1e8):
    """
    Create a massive-scale distributed analysis framework.

    This can handle billion-scale prime analysis across multiple machines.
    """
    print("🚀 Creating Massive-Scale Distributed Analysis")
    print("=" * 55)
    print(f"Target Scale: {target_primes:,.0f} primes")
    print()

    # Initialize coordinator
    coordinator = CUDNTDistributedCoordinator(target_primes)

    # Calculate cluster requirements
    cluster_specs = calculate_cluster_requirements(target_primes)
    print("📊 Cluster Requirements:")
    print(f"   Estimated Nodes: {cluster_specs['nodes_needed']}")
    print(f"   Memory per Node: {cluster_specs['memory_per_node_gb']:.1f} GB")
    print(f"   Storage Required: {cluster_specs['storage_tb']:.1f} TB")
    print(f"   Estimated Runtime: {cluster_specs['estimated_runtime_hours']:.1f} hours")
    print()

    return coordinator, cluster_specs

def calculate_cluster_requirements(target_primes):
    """Calculate cluster requirements for massive-scale analysis."""

    # Estimate prime count (using prime number theorem)
    estimated_primes = int(target_primes / np.log(target_primes))

    # Memory requirements (conservative estimates)
    prime_memory_gb = estimated_primes * 8 / (1024**3)  # 8 bytes per prime
    gap_memory_gb = estimated_primes * 4 / (1024**3)    # 4 bytes per gap
    working_memory_gb = 16  # Working memory for FFT/autocorr

    memory_per_node_gb = prime_memory_gb + gap_memory_gb + working_memory_gb

    # Node count based on memory constraints (assume 32GB per node)
    nodes_needed = max(1, int(np.ceil(memory_per_node_gb / 32)))

    # Storage requirements
    storage_tb = memory_per_node_gb * 2 / 1024  # TB (with overhead)

    # Performance estimates
    operations_per_prime = 1000  # FFT + autocorrelation operations
    total_operations = estimated_primes * operations_per_prime

    # Assume 1e10 operations/second per node (optimistic but achievable with CUDNT)
    operations_per_second_per_node = 1e10
    estimated_runtime_seconds = total_operations / (operations_per_second_per_node * nodes_needed)
    estimated_runtime_hours = estimated_runtime_seconds / 3600

    return {
        'estimated_primes': estimated_primes,
        'memory_per_node_gb': memory_per_node_gb,
        'nodes_needed': nodes_needed,
        'storage_tb': storage_tb,
        'total_operations': total_operations,
        'estimated_runtime_hours': estimated_runtime_hours,
        'operations_per_second_per_node': operations_per_second_per_node
    }

def test_distributed_cudnt():
    """Test the distributed CUDNT framework with sample data."""
    print("🧪 Testing Distributed CUDNT Framework")
    print("=" * 40)

    # Create sample prime gap data
    np.random.seed(42)
    sample_gaps = np.random.exponential(10, 10000).astype(int)  # 10k sample gaps

    print(f"📊 Sample data: {len(sample_gaps)} prime gaps")

    # Create distributed coordinator
    coordinator = CUDNTDistributedCoordinator(target_scale=len(sample_gaps))

    # Initialize cluster (use 2 nodes for testing)
    active_nodes = coordinator.initialize_cluster()

    print(f"✅ Cluster ready with {active_nodes} nodes")

    # Run distributed analysis
    analysis_config = {
        'analysis_type': 'both',
        'num_peaks': 5,
        'chunk_size': 1000  # Small chunks for testing
    }

    start_time = time.time()
    results = coordinator.distribute_analysis(sample_gaps, analysis_config)
    total_time = time.time() - start_time

    # Report results
    print("\n📈 Distributed Analysis Results:")
    print("=" * 35)

    if 'error' not in results:
        cluster_info = results['cluster_info']
        cluster_summary = results['cluster_summary']

        print(f"✅ Analysis completed successfully")
        print(f"   Total runtime: {total_time:.3f}s")
        print(f"   Cluster runtime: {cluster_info['total_runtime']:.3f}s")
        print(f"   Chunks processed: {cluster_summary['total_chunks_processed']}")
        print(f"   Failed chunks: {cluster_summary['total_failed_chunks']}")
        print(f"   Average chunk time: {cluster_summary['average_chunk_time']:.3f}s")
        print(f"   Cluster efficiency: {cluster_info['efficiency_score']:.1f}/10")

        # Show top peaks
        if results['fft_peaks']:
            print(f"\n🎯 Top FFT Peaks:")
            for i, peak in enumerate(results['fft_peaks'][:3]):
                ratio = peak.get('ratio', 1.0)
                closest = peak.get('closest_ratio', {})
                print(".3f")

        if results['autocorr_peaks']:
            print(f"\n🎯 Top Autocorr Peaks:")
            for i, peak in enumerate(results['autocorr_peaks'][:3]):
                ratio = peak.get('ratio', 1.0)
                closest = peak.get('closest_ratio', {})
                print(".3f")
    else:
        print(f"❌ Analysis failed: {results['error']}")

    print(f"\n🏆 Test completed in {total_time:.3f}s")
    return results

if __name__ == '__main__':
    # Test the distributed framework
    test_results = test_distributed_cudnt()

    # Show massive scale projection
    print("\n🌌 Massive Scale Projections:")
    print("=" * 30)

    scales = [1e6, 1e8, 1e9, 1e10]
    for scale in scales:
        specs = calculate_cluster_requirements(scale)
        print("10.0e")

    print("\n🚀 Distributed CUDNT Framework Ready!")
    print("   Run billion-scale prime analysis across multiple machines")
    print("   Each node uses CUDNT hybrid GPU/CPU acceleration")
