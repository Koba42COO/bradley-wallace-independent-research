#!/usr/bin/env python3
"""
F2 GPU Optimizer - Advanced Chia Plotting Optimization System
===========================================================

High-performance F2 optimization with GPU acceleration for Chia blockchain plotting.
Supports three optimization profiles: Speed, Cost, and Middle (balanced).

Features:
- F2 optimization algorithm with GPU acceleration
- Performance profiling (Speed/Cost/Middle)
- Real-time GPU utilization monitoring
- Intelligent resource allocation
- Plotting queue management
- Cost-benefit analysis

Author: Bradley Wallace (COO, Koba42 Corp)
Contact: user@domain.com
License: MIT License
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import warnings

# GPU abstraction layer
from gpu_abstraction_layer import GPUAbstractionLayer, GPUBackend

# Legacy GPU dependencies (fallback)
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('f2_optimizer')

class PerformanceProfile(Enum):
    """Performance optimization profiles"""
    SPEED = "speed"
    COST = "cost"
    MIDDLE = "middle"

@dataclass
class F2Config:
    """F2 optimization configuration"""
    profile: PerformanceProfile
    gpu_acceleration: bool
    max_threads: int
    memory_limit_gb: float
    batch_size: int
    priority_level: str
    cost_weight: float
    speed_weight: float

@dataclass
class PlottingJob:
    """Plotting job information"""
    job_id: str
    start_time: datetime
    estimated_completion: datetime
    progress: float
    status: str
    temp_dir: str
    final_dir: str
    size_gb: float
    resource_usage: Dict[str, float]

@dataclass
class OptimizationMetrics:
    """Performance optimization metrics"""
    plotting_speed_plots_per_hour: float
    resource_efficiency: float
    cost_per_plot_usd: float
    gpu_utilization_percent: float
    cpu_utilization_percent: float
    memory_utilization_percent: float
    power_consumption_watts: float
    cost_benefit_ratio: float

class F2GPUOptimizer:
    """Advanced F2 optimization system with GPU acceleration"""

    def __init__(self, chia_root: str = "~/chia-blockchain",
                 temp_dirs: List[str] = None,
                 final_dirs: List[str] = None,
                 profile: PerformanceProfile = PerformanceProfile.MIDDLE):
        """
        Initialize F2 GPU Optimizer

        Args:
            chia_root: Path to Chia blockchain installation
            temp_dirs: Temporary directories for plotting
            final_dirs: Final directories for completed plots
            profile: Performance optimization profile
        """
        self.chia_root = os.path.expanduser(chia_root)
        self.temp_dirs = temp_dirs or []
        self.final_dirs = final_dirs or []
        self.profile = profile

        # Initialize configuration
        self.config = self._create_config(profile)

        # Initialize components
        self.gpu_layer = GPUAbstractionLayer()
        self.gpu_manager = GPUManager() if GPU_AVAILABLE else None
        self.resource_monitor = ResourceMonitor()
        self.plot_queue = PlotQueue()
        self.cost_analyzer = CostAnalyzer()

        # Performance tracking
        self.metrics = OptimizationMetrics(0, 0, 0, 0, 0, 0, 0, 0)
        self.active_jobs: List[PlottingJob] = []

        # GPU memory pool (using abstraction layer)
        self.gpu_memory_pool = None
        if self.gpu_layer.current_device:
            self._initialize_gpu_memory_pool()

        logger.info(f"F2 GPU Optimizer initialized with {profile.value} profile on {self.gpu_layer.current_device.name if self.gpu_layer.current_device else 'CPU'}")

    def _create_config(self, profile: PerformanceProfile) -> F2Config:
        """Create optimization configuration based on profile"""
        base_config = {
            'profile': profile,
            'gpu_acceleration': GPU_AVAILABLE,
            'max_threads': psutil.cpu_count(),
            'memory_limit_gb': psutil.virtual_memory().total / (1024**3),
            'batch_size': 2,
            'priority_level': 'normal',
            'cost_weight': 0.5,
            'speed_weight': 0.5
        }

        if profile == PerformanceProfile.SPEED:
            base_config.update({
                'max_threads': psutil.cpu_count(),
                'memory_limit_gb': psutil.virtual_memory().total * 0.8 / (1024**3),
                'batch_size': 4,
                'priority_level': 'high',
                'cost_weight': 0.2,
                'speed_weight': 0.8
            })
        elif profile == PerformanceProfile.COST:
            base_config.update({
                'max_threads': max(1, psutil.cpu_count() // 4),
                'memory_limit_gb': psutil.virtual_memory().total * 0.3 / (1024**3),
                'batch_size': 1,
                'priority_level': 'low',
                'cost_weight': 0.8,
                'speed_weight': 0.2
            })
        else:  # MIDDLE
            base_config.update({
                'max_threads': max(2, psutil.cpu_count() // 2),
                'memory_limit_gb': psutil.virtual_memory().total * 0.5 / (1024**3),
                'batch_size': 2,
                'priority_level': 'normal',
                'cost_weight': 0.5,
                'speed_weight': 0.5
            })

        return F2Config(**base_config)

    def _initialize_gpu_memory_pool(self):
        """Initialize GPU memory pool for optimization using abstraction layer"""
        try:
            # Pre-allocate GPU memory for F2 operations using abstraction layer
            memory_limit = int(self.config.memory_limit_gb * 1024**3)
            pool_size = memory_limit // 4  # Use 1/4 for GPU ops

            if self.gpu_layer.current_device:
                self.gpu_memory_pool = self.gpu_layer.allocate_memory(pool_size)
                logger.info(f"GPU memory pool initialized ({pool_size / (1024**3):.1f} GB) on {self.gpu_layer.current_device.backend.value}")
            else:
                logger.warning("No GPU device available for memory pool initialization")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU memory pool: {e}")

    def optimize_f2_plotting(self, num_plots: int,
                           farmer_key: str,
                           pool_key: str) -> Dict[str, Any]:
        """
        Optimize F2 plotting with GPU acceleration

        Args:
            num_plots: Number of plots to create
            farmer_key: Chia farmer public key
            pool_key: Chia pool public key

        Returns:
            Optimization plan and results
        """
        logger.info(f"Starting F2 optimization for {num_plots} plots")

        # Analyze current system state
        system_analysis = self._analyze_system_state()

        # Create optimization plan
        optimization_plan = self._create_optimization_plan(num_plots, system_analysis)

        # Execute plotting jobs
        results = self._execute_plotting_jobs(optimization_plan, farmer_key, pool_key)

        # Analyze results
        final_analysis = self._analyze_results(results)

        return {
            'optimization_plan': optimization_plan,
            'execution_results': results,
            'final_analysis': final_analysis,
            'performance_metrics': asdict(self.metrics)
        }

    def _analyze_system_state(self) -> Dict[str, Any]:
        """Analyze current system state for optimization"""
        analysis = {
            'cpu_cores': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'gpu_available': GPU_AVAILABLE,
            'temp_space_available': 0,
            'final_space_available': 0
        }

        # Check GPU using abstraction layer
        if self.gpu_layer.current_device:
            try:
                device = self.gpu_layer.current_device
                mem_info = self.gpu_layer.get_memory_info()
                analysis.update({
                    'gpu_name': device.name,
                    'gpu_backend': device.backend.value,
                    'gpu_memory_gb': device.memory_total / (1024**3),
                    'gpu_memory_free_gb': mem_info['free_gb'],
                    'gpu_utilization': 0  # Placeholder - would need backend-specific monitoring
                })
            except Exception as e:
                logger.warning(f"GPU analysis failed: {e}")

        # Check storage space
        for temp_dir in self.temp_dirs:
            if os.path.exists(temp_dir):
                stat = os.statvfs(temp_dir)
                analysis['temp_space_available'] += (stat.f_bavail * stat.f_frsize) / (1024**3)

        for final_dir in self.final_dirs:
            if os.path.exists(final_dir):
                stat = os.statvfs(final_dir)
                analysis['final_space_available'] += (stat.f_bavail * stat.f_frsize) / (1024**3)

        return analysis

    def _create_optimization_plan(self, num_plots: int,
                                system_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive optimization plan"""
        plan = {
            'num_plots': num_plots,
            'profile': self.config.profile.value,
            'gpu_acceleration': self.config.gpu_acceleration,
            'parallel_jobs': min(num_plots, self.config.batch_size),
            'thread_allocation': self.config.max_threads,
            'memory_allocation_gb': self.config.memory_limit_gb,
            'estimated_completion_hours': self._estimate_completion_time(num_plots),
            'cost_estimate_usd': self._estimate_cost(num_plots),
            'resource_utilization': self._calculate_resource_utilization(system_analysis)
        }

        return plan

    def _estimate_completion_time(self, num_plots: int) -> float:
        """Estimate completion time based on profile"""
        base_time_per_plot = 4.0  # Base hours per plot

        if self.config.profile == PerformanceProfile.SPEED:
            multiplier = 0.6  # Faster with GPU
        elif self.config.profile == PerformanceProfile.COST:
            multiplier = 2.2  # Slower with limited resources
        else:
            multiplier = 1.0  # Balanced

        gpu_bonus = 0.7 if self.config.gpu_acceleration and GPU_AVAILABLE else 1.0

        return num_plots * base_time_per_plot * multiplier * gpu_bonus

    def _estimate_cost(self, num_plots: int) -> float:
        """Estimate cost in USD"""
        # Base costs (approximate)
        electricity_cost_per_kwh = 0.12
        avg_power_consumption_kw = 0.3  # 300W average

        hours_per_plot = self._estimate_completion_time(1)
        total_hours = num_plots * hours_per_plot
        electricity_cost = total_hours * avg_power_consumption_kw * electricity_cost_per_kwh

        # Storage cost (minimal)
        storage_cost = num_plots * 100 * 0.02 / 1000  # $0.02 per GB per month

        return electricity_cost + storage_cost

    def _calculate_resource_utilization(self, system_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal resource utilization"""
        utilization = {
            'cpu_utilization_target': 70.0,
            'memory_utilization_target': 60.0,
            'gpu_utilization_target': 80.0 if self.config.gpu_acceleration else 0.0
        }

        if self.config.profile == PerformanceProfile.SPEED:
            utilization.update({
                'cpu_utilization_target': 90.0,
                'memory_utilization_target': 80.0,
                'gpu_utilization_target': 95.0
            })
        elif self.config.profile == PerformanceProfile.COST:
            utilization.update({
                'cpu_utilization_target': 40.0,
                'memory_utilization_target': 30.0,
                'gpu_utilization_target': 0.0
            })

        return utilization

    def _execute_plotting_jobs(self, plan: Dict[str, Any],
                             farmer_key: str, pool_key: str) -> List[Dict[str, Any]]:
        """Execute plotting jobs with F2 optimization"""
        results = []
        completed_jobs = 0

        logger.info(f"Starting {plan['parallel_jobs']} parallel plotting jobs")

        # Create plotting jobs
        for i in range(plan['num_plots']):
            job = self._create_plotting_job(i, plan, farmer_key, pool_key)
            self.plot_queue.add_job(job)

        # Execute jobs with resource optimization
        while completed_jobs < plan['num_plots']:
            available_slots = self._get_available_slots()

            for _ in range(min(available_slots, plan['parallel_jobs'])):
                if self.plot_queue.has_pending_jobs():
                    job = self.plot_queue.get_next_job()
                    result = self._execute_single_job(job)
                    results.append(result)
                    completed_jobs += 1

                    # Update metrics
                    self._update_performance_metrics(result)

            time.sleep(30)  # Check every 30 seconds

        return results

    def _create_plotting_job(self, job_id: int, plan: Dict[str, Any],
                           farmer_key: str, pool_key: str) -> PlottingJob:
        """Create a plotting job configuration"""
        job = PlottingJob(
            job_id=f"plot_{job_id}",
            start_time=datetime.now(),
            estimated_completion=datetime.now() + timedelta(hours=plan['estimated_completion_hours']),
            progress=0.0,
            status="pending",
            temp_dir=self.temp_dirs[job_id % len(self.temp_dirs)] if self.temp_dirs else "/tmp",
            final_dir=self.final_dirs[job_id % len(self.final_dirs)] if self.final_dirs else "~/plots",
            size_gb=100.0,  # Standard plot size
            resource_usage={'cpu': 0, 'memory': 0, 'gpu': 0}
        )

        return job

    def _get_available_slots(self) -> int:
        """Get number of available plotting slots"""
        resources = self.resource_monitor.get_resources()

        # Check resource availability
        cpu_available = resources['cpu_usage'] < 80
        memory_available = resources['memory_usage'] < self.config.memory_limit_gb * 100 / psutil.virtual_memory().total * (1024**3)

        gpu_available = True
        if self.config.gpu_acceleration and GPU_AVAILABLE:
            gpu_available = resources.get('gpu_usage', 0) < 90

        return int(cpu_available and memory_available and gpu_available)

    def _execute_single_job(self, job: PlottingJob) -> Dict[str, Any]:
        """Execute a single plotting job with F2 optimization"""
        logger.info(f"Executing plotting job {job.job_id}")

        # F2 optimization: Optimize memory usage and I/O patterns
        optimized_params = self._apply_f2_optimization(job)

        # Execute plotting command
        result = self._run_chia_plotting_command(job, optimized_params)

        # Update job status
        job.status = "completed" if result['success'] else "failed"
        job.progress = 100.0 if result['success'] else 0.0

        return result

    def _apply_f2_optimization(self, job: PlottingJob) -> Dict[str, Any]:
        """Apply F2 optimization algorithm"""
        params = {
            'threads': self.config.max_threads,
            'memory_buffer': int(self.config.memory_limit_gb * 1024**3),
            'gpu_acceleration': self.config.gpu_acceleration,
            'batch_size': self.config.batch_size,
            'f2_optimization': True
        }

        # Profile-specific optimizations
        if self.config.profile == PerformanceProfile.SPEED:
            params.update({
                'aggressive_memory': True,
                'gpu_priority': 'high',
                'io_optimization': 'speed'
            })
        elif self.config.profile == PerformanceProfile.COST:
            params.update({
                'memory_conservative': True,
                'gpu_disabled': True,
                'io_optimization': 'efficiency'
            })

        return params

    def _run_chia_plotting_command(self, job: PlottingJob,
                                 params: Dict[str, Any]) -> Dict[str, Any]:
        """Run Chia plotting command with optimizations"""
        try:
            # Build command with F2 optimizations
            cmd = [
                os.path.join(self.chia_root, 'chia'),
                'plots', 'create',
                '-k', '32',  # k-size for standard plots
                '-n', '1',   # Number of plots
                '-t', job.temp_dir,
                '-d', job.final_dir,
                '-r', str(params['threads']),
                '-b', str(params['memory_buffer'])
            ]

            # Add GPU acceleration if available and enabled (using abstraction layer)
            if params.get('gpu_acceleration') and self.gpu_layer.current_device:
                backend = self.gpu_layer.current_device.backend
                if backend == GPUBackend.CUDA:
                    cmd.extend(['--gpu', '--gpu-index', '0'])
                elif backend == GPUBackend.METAL:
                    # Chia may not support Metal directly, use CPU fallback
                    logger.info("Apple Silicon detected - using CPU plotting (Metal not supported by Chia)")
                else:
                    logger.info(f"GPU backend {backend.value} not supported by Chia - using CPU")

            # Execute command
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600*12)  # 12 hour timeout
            execution_time = time.time() - start_time

            success = result.returncode == 0
            if success:
                logger.info(f"Plot {job.job_id} completed in {execution_time/3600:.2f} hours")
            else:
                logger.error(f"Plot {job.job_id} failed: {result.stderr}")

            return {
                'job_id': job.job_id,
                'success': success,
                'execution_time_hours': execution_time / 3600,
                'output': result.stdout,
                'error': result.stderr,
                'resource_usage': self.resource_monitor.get_resources()
            }

        except subprocess.TimeoutExpired:
            logger.error(f"Plot {job.job_id} timed out")
            return {
                'job_id': job.job_id,
                'success': False,
                'execution_time_hours': 12,
                'error': 'Timeout after 12 hours'
            }
        except Exception as e:
            logger.error(f"Plot {job.job_id} failed with error: {e}")
            return {
                'job_id': job.job_id,
                'success': False,
                'execution_time_hours': 0,
                'error': str(e)
            }

    def _update_performance_metrics(self, result: Dict[str, Any]):
        """Update performance metrics based on job results"""
        if result['success']:
            execution_time = result['execution_time_hours']
            plots_per_hour = 1 / execution_time if execution_time > 0 else 0

            # Update metrics
            self.metrics.plotting_speed_plots_per_hour = (
                self.metrics.plotting_speed_plots_per_hour + plots_per_hour
            ) / 2  # Running average

            # Update resource utilization
            resources = result.get('resource_usage', {})
            self.metrics.cpu_utilization_percent = resources.get('cpu_usage', 0)
            self.metrics.memory_utilization_percent = resources.get('memory_usage', 0)
            self.metrics.gpu_utilization_percent = resources.get('gpu_usage', 0)

    def _analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze plotting results"""
        successful_plots = sum(1 for r in results if r['success'])
        total_plots = len(results)

        avg_time = np.mean([r['execution_time_hours'] for r in results if r['success']]) if successful_plots > 0 else 0

        analysis = {
            'total_plots': total_plots,
            'successful_plots': successful_plots,
            'success_rate': successful_plots / total_plots if total_plots > 0 else 0,
            'average_plotting_time_hours': avg_time,
            'total_execution_time_hours': sum(r['execution_time_hours'] for r in results),
            'cost_analysis': self.cost_analyzer.analyze_costs(results),
            'performance_score': self._calculate_performance_score(results)
        }

        return analysis

    def _calculate_performance_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall performance score"""
        if not results:
            return 0.0

        success_rate = sum(1 for r in results if r['success']) / len(results)
        avg_time = np.mean([r['execution_time_hours'] for r in results if r['success']])

        # Normalize time (lower is better)
        time_score = max(0, 1 - (avg_time - 2) / 8)  # Optimal around 2-4 hours

        # Weighted score based on profile
        if self.config.profile == PerformanceProfile.SPEED:
            return 0.3 * success_rate + 0.7 * time_score
        elif self.config.profile == PerformanceProfile.COST:
            return 0.7 * success_rate + 0.3 * (1 - time_score)  # Penalize speed
        else:  # MIDDLE
            return 0.5 * success_rate + 0.5 * time_score

class GPUManager:
    """GPU resource management for plotting optimization"""

    def __init__(self):
        self.gpu_info = self._get_gpu_info()

    def _get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        if not GPU_AVAILABLE:
            return {}

        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return {
                    'name': gpu.name,
                    'memory_total': gpu.memoryTotal,
                    'memory_free': gpu.memoryFree,
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                }
        except Exception as e:
            logger.warning(f"Failed to get GPU info: {e}")

        return {}

    def optimize_gpu_allocation(self, num_jobs: int) -> Dict[str, Any]:
        """Optimize GPU memory allocation for multiple jobs"""
        if not self.gpu_info:
            return {'gpu_acceleration': False}

        memory_per_job = self.gpu_info.get('memory_total', 0) / max(num_jobs, 1)
        recommended_jobs = min(num_jobs, int(self.gpu_info.get('memory_total', 0) / 2048))  # 2GB per job

        return {
            'gpu_acceleration': True,
            'memory_per_job_mb': memory_per_job,
            'recommended_parallel_jobs': recommended_jobs,
            'gpu_utilization_target': 85
        }

class ResourceMonitor:
    """Monitor system resources during plotting"""

    def __init__(self):
        self.last_net_io = psutil.net_io_counters()

    def get_resources(self) -> Dict[str, float]:
        """Get current resource utilization"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'gpu_usage': self._get_gpu_usage()
        }

    def _get_gpu_usage(self) -> float:
        """Get GPU utilization"""
        if not GPU_AVAILABLE:
            return 0.0

        try:
            gpus = GPUtil.getGPUs()
            return gpus[0].load * 100 if gpus else 0.0
        except:
            return 0.0

class PlotQueue:
    """Manage plotting job queue"""

    def __init__(self):
        self.queue: List[PlottingJob] = []
        self.completed: List[PlottingJob] = []

    def add_job(self, job: PlottingJob):
        """Add job to queue"""
        self.queue.append(job)

    def get_next_job(self) -> Optional[PlottingJob]:
        """Get next job from queue"""
        return self.queue.pop(0) if self.queue else None

    def has_pending_jobs(self) -> bool:
        """Check if there are pending jobs"""
        return len(self.queue) > 0

    def get_queue_status(self) -> Dict[str, Any]:
        """Get queue status"""
        return {
            'pending_jobs': len(self.queue),
            'completed_jobs': len(self.completed),
            'total_jobs': len(self.queue) + len(self.completed)
        }

class CostAnalyzer:
    """Analyze plotting costs and efficiency"""

    def __init__(self):
        self.electricity_cost_per_kwh = 0.12
        self.avg_power_consumption_kw = 0.3

    def analyze_costs(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze costs for plotting results"""
        successful_jobs = [r for r in results if r['success']]

        if not successful_jobs:
            return {'total_cost': 0, 'cost_per_plot': 0, 'cost_efficiency': 0}

        total_hours = sum(r['execution_time_hours'] for r in successful_jobs)
        electricity_cost = total_hours * self.avg_power_consumption_kw * self.electricity_cost_per_kwh

        num_plots = len(successful_jobs)
        cost_per_plot = electricity_cost / num_plots if num_plots > 0 else 0

        # Cost efficiency (lower is better)
        cost_efficiency = cost_per_plot / 10  # Normalize against $10 baseline

        return {
            'total_cost': electricity_cost,
            'cost_per_plot': cost_per_plot,
            'cost_efficiency': cost_efficiency,
            'plots_completed': num_plots
        }

def main():
    """Main F2 GPU Optimizer application"""
    import argparse

    parser = argparse.ArgumentParser(description='F2 GPU Optimizer - Chia Plotting Optimization')
    parser.add_argument('--chia-root', default='~/chia-blockchain',
                       help='Path to Chia blockchain installation')
    parser.add_argument('--temp-dirs', nargs='+', required=True,
                       help='Temporary directories for plotting')
    parser.add_argument('--final-dirs', nargs='+', required=True,
                       help='Final directories for completed plots')
    parser.add_argument('--profile', choices=['speed', 'cost', 'middle'],
                       default='middle', help='Performance profile')
    parser.add_argument('--num-plots', type=int, default=1,
                       help='Number of plots to create')
    parser.add_argument('--farmer-key', required=True,
                       help='Chia farmer public key')
    parser.add_argument('--pool-key', required=True,
                       help='Chia pool public key')
    parser.add_argument('--output', help='Output file for results')

    args = parser.parse_args()

    # Initialize optimizer
    profile = PerformanceProfile(args.profile)
    optimizer = F2GPUOptimizer(
        chia_root=args.chia_root,
        temp_dirs=args.temp_dirs,
        final_dirs=args.final_dirs,
        profile=profile
    )

    try:
        print("🚀 F2 GPU Optimizer - Chia Plotting Optimization System")
        print(f"Profile: {profile.value.upper()}")
        print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")
        print(f"Target Plots: {args.num_plots}")
        print("=" * 60)

        # Run optimization
        results = optimizer.optimize_f2_plotting(
            num_plots=args.num_plots,
            farmer_key=args.farmer_key,
            pool_key=args.pool_key
        )

        # Print results
        analysis = results['final_analysis']
        print("""
📊 RESULTS:""")
        print(f"Successful Plots: {analysis['successful_plots']}/{analysis['total_plots']}")
        print(f"Total Time: {analysis['total_time']:.1f} hours")
        print(f"Average Time per Plot: {analysis['avg_time_per_plot']:.2f} hours")
        print(f"GPU Utilization: {analysis['avg_gpu_utilization']:.2f}%")
        print(f"Estimated Cost: ${analysis['estimated_cost']:.3f}")

        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to {args.output}")

    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
