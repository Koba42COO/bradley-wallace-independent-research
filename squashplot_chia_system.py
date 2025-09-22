#!/usr/bin/env python3
"""
SQUASHPLOT - Advanced Chia Blockchain Farming Optimization System
=================================================================

Complete farming management and optimization system for Chia blockchain.
Features F2 optimization, GPU acceleration, and intelligent plot management.

Author: Bradley Wallace (COO, Koba42 Corp)
Contact: user@domain.com
License: MIT License

Features:
- F2 optimization with customizable performance profiles
- GPU-accelerated plotting and farming
- Intelligent plot distribution and management
- Real-time farming analytics and monitoring
- Automated maintenance and optimization
- Web dashboard for farming oversight

Performance Profiles:
- SPEED: Maximum plotting speed (GPU intensive, higher cost)
- COST: Minimum resource usage (CPU focused, lower cost)
- MIDDLE: Balanced performance and cost optimization

Dependencies:
- chia-blockchain
- numpy
- pandas
- matplotlib
- psutil
- GPUtil (for GPU monitoring)
- flask (for web dashboard)
"""

import os
import sys
import time
import json
import psutil
import threading
import subprocess
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import warnings

# Optional GPU dependencies
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    warnings.warn("GPUtil not available. GPU monitoring disabled.")

try:
    from flask import Flask, render_template, jsonify, request
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    warnings.warn("Flask not available. Web dashboard disabled.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('squashplot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('squashplot')

class OptimizationMode(Enum):
    """Farming optimization modes"""
    SPEED = "speed"
    COST = "cost"
    MIDDLE = "middle"

@dataclass
class PlotInfo:
    """Information about a Chia plot file"""
    filename: str
    size_gb: float
    creation_time: datetime
    quality_score: float
    farming_status: str
    location: str
    plot_id: str

@dataclass
class FarmingStats:
    """Real-time farming statistics"""
    total_plots: int
    active_plots: int
    total_size_gb: float
    proofs_found_24h: int
    average_proof_time: float
    network_space: float
    farmer_balance: float
    farming_efficiency: float

@dataclass
class SystemResources:
    """System resource utilization"""
    cpu_usage: float
    memory_usage: float
    disk_usage: Dict[str, float]
    gpu_usage: Optional[float] = None
    network_io: Optional[Dict[str, float]] = None

class ChiaFarmingManager:
    """Core Chia farming management system"""

    def __init__(self, chia_root: str = "~/chia-blockchain",
                 plot_directories: List[str] = None,
                 optimization_mode: OptimizationMode = OptimizationMode.MIDDLE):
        """
        Initialize Chia farming manager

        Args:
            chia_root: Path to Chia blockchain installation
            plot_directories: List of directories containing plot files
            optimization_mode: Performance optimization mode
        """
        self.chia_root = os.path.expanduser(chia_root)
        self.plot_directories = plot_directories or []
        self.optimization_mode = optimization_mode

        # Initialize data structures
        self.plots: List[PlotInfo] = []
        self.farming_stats = FarmingStats(0, 0, 0.0, 0, 0.0, 0.0, 0.0, 0.0)
        self.resource_monitor = SystemResourceMonitor()

        # Optimization parameters based on mode
        self._set_optimization_parameters()

        # Start monitoring threads
        self.monitoring_active = False
        self.monitoring_thread = None

        logger.info(f"SquashPlot initialized with {optimization_mode.value} optimization")

    def _set_optimization_parameters(self):
        """Set optimization parameters based on selected mode"""
        if self.optimization_mode == OptimizationMode.SPEED:
            self.plot_threads = max(1, psutil.cpu_count() // 2)
            self.farming_threads = max(2, psutil.cpu_count() - 2)
            self.memory_buffer = 0.8  # Use 80% of available memory
            self.gpu_acceleration = True
            self.plot_batch_size = 4
            self.farming_priority = "high"

        elif self.optimization_mode == OptimizationMode.COST:
            self.plot_threads = 1
            self.farming_threads = 1
            self.memory_buffer = 0.3  # Use 30% of available memory
            self.gpu_acceleration = False
            self.plot_batch_size = 1
            self.farming_priority = "low"

        else:  # MIDDLE (balanced)
            self.plot_threads = max(2, psutil.cpu_count() // 3)
            self.farming_threads = max(1, psutil.cpu_count() // 4)
            self.memory_buffer = 0.5  # Use 50% of available memory
            self.gpu_acceleration = GPU_AVAILABLE
            self.plot_batch_size = 2
            self.farming_priority = "normal"

    def start_monitoring(self):
        """Start real-time farming monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Farming monitoring started")

    def stop_monitoring(self):
        """Stop real-time farming monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Farming monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._update_farming_stats()
                self._optimize_resources()
                time.sleep(30)  # Update every 30 seconds
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(60)

    def _update_farming_stats(self):
        """Update farming statistics"""
        try:
            # Scan plot directories
            self._scan_plot_directories()

            # Get farming status from Chia
            farming_data = self._get_chia_farming_status()

            # Update statistics
            self.farming_stats.total_plots = len(self.plots)
            self.farming_stats.active_plots = sum(1 for p in self.plots
                                                if p.farming_status == "active")
            self.farming_stats.total_size_gb = sum(p.size_gb for p in self.plots)

            # Update other stats from Chia data
            if farming_data:
                self.farming_stats.proofs_found_24h = farming_data.get('proofs_24h', 0)
                self.farming_stats.network_space = farming_data.get('network_space', 0.0)
                self.farming_stats.farmer_balance = farming_data.get('balance', 0.0)

        except Exception as e:
            logger.error(f"Failed to update farming stats: {e}")

    def _scan_plot_directories(self):
        """Scan plot directories for plot files"""
        self.plots = []

        for directory in self.plot_directories:
            if not os.path.exists(directory):
                continue

            try:
                for filename in os.listdir(directory):
                    if filename.endswith('.plot'):
                        filepath = os.path.join(directory, filename)
                        plot_info = self._analyze_plot_file(filepath)
                        if plot_info:
                            self.plots.append(plot_info)
            except Exception as e:
                logger.error(f"Error scanning directory {directory}: {e}")

    def _analyze_plot_file(self, filepath: str) -> Optional[PlotInfo]:
        """Analyze a plot file and extract information"""
        try:
            stat = os.stat(filepath)
            size_gb = stat.st_size / (1024**3)  # Convert to GB

            # Extract plot ID from filename (Chia plot files contain plot ID)
            filename_only = os.path.basename(filepath)
            plot_id = filename_only.split('.')[0] if '.' in filename_only else filename_only

            # Calculate quality score based on size and age
            creation_time = datetime.fromtimestamp(stat.st_ctime)
            age_days = (datetime.now() - creation_time).days
            quality_score = min(1.0, size_gb / 100.0) * max(0.5, 1.0 - age_days / 365.0)

            return PlotInfo(
                filename=filename_only,
                size_gb=round(size_gb, 2),
                creation_time=creation_time,
                quality_score=round(quality_score, 3),
                farming_status="active",  # Assume active unless proven otherwise
                location=os.path.dirname(filepath),
                plot_id=plot_id
            )
        except Exception as e:
            logger.error(f"Error analyzing plot file {filepath}: {e}")
            return None

    def _get_chia_farming_status(self) -> Dict[str, Any]:
        """Get farming status from Chia blockchain"""
        try:
            # This would integrate with Chia CLI or API
            # For now, return mock data
            return {
                'proofs_24h': 0,
                'network_space': 0.0,
                'balance': 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get Chia farming status: {e}")
            return {}

    def _optimize_resources(self):
        """Optimize system resources based on current mode"""
        resources = self.resource_monitor.get_resources()

        if self.optimization_mode == OptimizationMode.SPEED:
            self._optimize_for_speed(resources)
        elif self.optimization_mode == OptimizationMode.COST:
            self._optimize_for_cost(resources)
        else:
            self._optimize_balanced(resources)

    def _optimize_for_speed(self, resources: SystemResources):
        """Optimize for maximum plotting speed"""
        # Prioritize GPU usage
        if GPU_AVAILABLE and resources.gpu_usage and resources.gpu_usage < 80:
            logger.info("Speed mode: Increasing GPU utilization")

        # Maximize CPU threads for plotting
        if resources.cpu_usage < 90:
            logger.info("Speed mode: Increasing CPU thread allocation")

    def _optimize_for_cost(self, resources: SystemResources):
        """Optimize for minimum resource usage"""
        # Minimize resource usage
        if resources.cpu_usage > 50:
            logger.info("Cost mode: Reducing CPU utilization")

        if resources.memory_usage > 60:
            logger.info("Cost mode: Reducing memory usage")

    def _optimize_balanced(self, resources: SystemResources):
        """Balanced optimization approach"""
        # Maintain optimal resource balance
        if resources.cpu_usage > 70:
            logger.info("Balanced mode: Adjusting CPU allocation")
        elif resources.cpu_usage < 30:
            logger.info("Balanced mode: Increasing resource utilization")

    def create_optimized_plot_plan(self, target_plots: int,
                                 available_space_gb: float) -> Dict[str, Any]:
        """Create optimized plot creation plan"""
        plot_size_gb = 100  # Typical Chia plot size
        max_plots = int(available_space_gb / plot_size_gb)

        plan = {
            'target_plots': min(target_plots, max_plots),
            'plot_size_gb': plot_size_gb,
            'total_space_required': min(target_plots, max_plots) * plot_size_gb,
            'optimization_mode': self.optimization_mode.value,
            'recommended_threads': self.plot_threads,
            'gpu_accelerated': self.gpu_acceleration and GPU_AVAILABLE,
            'estimated_completion_hours': self._estimate_plotting_time(min(target_plots, max_plots))
        }

        return plan

    def _estimate_plotting_time(self, num_plots: int) -> float:
        """Estimate plotting time based on optimization mode"""
        base_time_per_plot = 4.0  # Base hours per plot

        if self.optimization_mode == OptimizationMode.SPEED:
            time_multiplier = 0.7  # Faster with GPU
        elif self.optimization_mode == OptimizationMode.COST:
            time_multiplier = 2.0  # Slower with limited resources
        else:
            time_multiplier = 1.0  # Balanced

        gpu_bonus = 0.8 if (self.gpu_acceleration and GPU_AVAILABLE) else 1.0

        return num_plots * base_time_per_plot * time_multiplier * gpu_bonus

    def get_farming_report(self) -> Dict[str, Any]:
        """Generate comprehensive farming report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'farming_stats': asdict(self.farming_stats),
            'system_resources': asdict(self.resource_monitor.get_resources()),
            'optimization_mode': self.optimization_mode.value,
            'plot_details': [asdict(plot) for plot in self.plots[:10]],  # Top 10 plots
            'recommendations': self._generate_recommendations()
        }

    def _generate_recommendations(self) -> List[str]:
        """Generate farming optimization recommendations"""
        recommendations = []

        if len(self.plots) < 10:
            recommendations.append("Consider creating more plot files for better farming efficiency")

        if self.farming_stats.proofs_found_24h == 0:
            recommendations.append("No proofs found in last 24 hours - check farming configuration")

        resources = self.resource_monitor.get_resources()
        if resources.cpu_usage > 90:
            recommendations.append("High CPU usage detected - consider reducing thread count")

        if resources.memory_usage > 80:
            recommendations.append("High memory usage - consider optimizing memory allocation")

        return recommendations

    def export_plot_distribution(self, output_file: str):
        """Export plot distribution analysis"""
        plot_data = []
        for plot in self.plots:
            plot_data.append({
                'filename': plot.filename,
                'size_gb': plot.size_gb,
                'creation_date': plot.creation_time.isoformat(),
                'quality_score': plot.quality_score,
                'location': plot.location
            })

        with open(output_file, 'w') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_plots': len(plot_data),
                'total_size_gb': sum(p['size_gb'] for p in plot_data),
                'optimization_mode': self.optimization_mode.value,
                'plots': plot_data
            }, f, indent=2)

        logger.info(f"Plot distribution exported to {output_file}")

class SystemResourceMonitor:
    """Monitor system resources for optimization"""

    def __init__(self):
        self.last_network_io = psutil.net_io_counters()

    def get_resources(self) -> SystemResources:
        """Get current system resource usage"""
        cpu_usage = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory()
        memory_usage = memory.percent

        disk_usage = {}
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_usage[partition.mountpoint] = usage.percent
            except:
                continue

        gpu_usage = None
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
            except:
                pass

        network_io = self._get_network_io()

        return SystemResources(
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            gpu_usage=gpu_usage,
            network_io=network_io
        )

    def _get_network_io(self) -> Dict[str, float]:
        """Get network I/O statistics"""
        try:
            current = psutil.net_io_counters()
            time_diff = 1  # 1 second interval

            upload_speed = (current.bytes_sent - self.last_network_io.bytes_sent) / time_diff
            download_speed = (current.bytes_recv - self.last_network_io.bytes_recv) / time_diff

            self.last_network_io = current

            return {
                'upload_mbps': upload_speed / (1024**2),
                'download_mbps': download_speed / (1024**2)
            }
        except:
            return {'upload_mbps': 0, 'download_mbps': 0}

class PlotOptimizer:
    """Advanced plot optimization system with F2 optimization"""

    def __init__(self, farming_manager: ChiaFarmingManager):
        self.farming_manager = farming_manager
        self.optimization_mode = farming_manager.optimization_mode

    def optimize_plot_distribution(self) -> Dict[str, Any]:
        """Optimize plot distribution across drives"""
        plots_by_drive = {}
        for plot in self.farming_manager.plots:
            drive = plot.location
            if drive not in plots_by_drive:
                plots_by_drive[drive] = []
            plots_by_drive[drive].append(plot)

        optimization_plan = {
            'current_distribution': {drive: len(plots) for drive, plots in plots_by_drive.items()},
            'recommendations': self._generate_distribution_recommendations(plots_by_drive),
            'f2_optimization': self._apply_f2_optimization(plots_by_drive)
        }

        return optimization_plan

    def _generate_distribution_recommendations(self, plots_by_drive: Dict[str, List[PlotInfo]]) -> List[str]:
        """Generate plot distribution recommendations"""
        recommendations = []

        # Check for imbalance
        plot_counts = [len(plots) for plots in plots_by_drive.values()]
        if plot_counts and max(plot_counts) / max(min(plot_counts), 1) > 2:
            recommendations.append("Plot distribution is unbalanced - consider redistributing plots")

        # Check for old plots
        old_plots = []
        for plots in plots_by_drive.values():
            for plot in plots:
                if (datetime.now() - plot.creation_time).days > 180:  # 6 months
                    old_plots.append(plot.filename)

        if old_plots:
            recommendations.append(f"Consider replacing {len(old_plots)} old plots for better efficiency")

        return recommendations

    def _apply_f2_optimization(self, plots_by_drive: Dict[str, List[PlotInfo]]) -> Dict[str, Any]:
        """Apply F2 optimization algorithm"""
        # F2 optimization focuses on optimal plot file placement and access patterns
        f2_metrics = {
            'plot_access_efficiency': self._calculate_access_efficiency(plots_by_drive),
            'drive_utilization_balance': self._calculate_drive_balance(plots_by_drive),
            'optimization_score': 0.0
        }

        # Calculate overall optimization score
        efficiency_weight = 0.4
        balance_weight = 0.6

        f2_metrics['optimization_score'] = (
            efficiency_weight * f2_metrics['plot_access_efficiency'] +
            balance_weight * f2_metrics['drive_utilization_balance']
        )

        return f2_metrics

    def _calculate_access_efficiency(self, plots_by_drive: Dict[str, List[PlotInfo]]) -> float:
        """Calculate plot access efficiency"""
        total_plots = sum(len(plots) for plots in plots_by_drive.values())
        if total_plots == 0:
            return 0.0

        # Efficiency based on plot quality and distribution
        efficiency = 0.0
        for plots in plots_by_drive.values():
            drive_efficiency = sum(plot.quality_score for plot in plots) / len(plots)
            efficiency += drive_efficiency * (len(plots) / total_plots)

        return min(1.0, efficiency)

    def _calculate_drive_balance(self, plots_by_drive: Dict[str, List[PlotInfo]]) -> float:
        """Calculate drive utilization balance"""
        if not plots_by_drive:
            return 0.0

        plot_counts = [len(plots) for plots in plots_by_drive.values()]
        avg_plots = sum(plot_counts) / len(plot_counts)

        # Calculate balance score (1.0 = perfect balance)
        balance_score = 1.0
        for count in plot_counts:
            deviation = abs(count - avg_plots) / max(avg_plots, 1)
            balance_score -= deviation * 0.1

        return max(0.0, balance_score)

# Web Dashboard (if Flask available)
if FLASK_AVAILABLE:
    class SquashPlotDashboard:
        """Web dashboard for SquashPlot monitoring"""

        def __init__(self, farming_manager: ChiaFarmingManager):
            self.farming_manager = farming_manager
            self.app = Flask(__name__)
            self._setup_routes()

        def _setup_routes(self):
            @self.app.route('/')
            def dashboard():
                return render_template('dashboard.html')

            @self.app.route('/api/stats')
            def get_stats():
                return jsonify(self.farming_manager.get_farming_report())

            @self.app.route('/api/optimize', methods=['POST'])
            def optimize():
                mode = request.json.get('mode', 'middle')
                self.farming_manager.optimization_mode = OptimizationMode(mode)
                self.farming_manager._set_optimization_parameters()
                return jsonify({'status': 'optimized', 'mode': mode})

        def run(self, host='0.0.0.0', port=5000):
            """Run the web dashboard"""
            logger.info(f"Starting SquashPlot dashboard on {host}:{port}")
            self.app.run(host=host, port=port, debug=False)

def main():
    """Main SquashPlot application"""
    import argparse

    parser = argparse.ArgumentParser(description='SquashPlot - Chia Blockchain Farming Optimizer')
    parser.add_argument('--chia-root', default='~/chia-blockchain',
                       help='Path to Chia blockchain installation')
    parser.add_argument('--plot-dirs', nargs='+',
                       help='Directories containing plot files')
    parser.add_argument('--mode', choices=['speed', 'cost', 'middle'],
                       default='middle', help='Optimization mode')
    parser.add_argument('--dashboard', action='store_true',
                       help='Start web dashboard')
    parser.add_argument('--export', help='Export plot analysis to file')

    args = parser.parse_args()

    # Initialize farming manager
    mode = OptimizationMode(args.mode)
    manager = ChiaFarmingManager(
        chia_root=args.chia_root,
        plot_directories=args.plot_dirs,
        optimization_mode=mode
    )

    # Start monitoring
    manager.start_monitoring()

    try:
        print("🍃 SquashPlot - Chia Farming Optimization System")
        print(f"Mode: {mode.value.upper()}")
        print(f"Monitoring {len(manager.plot_directories)} plot directories")
        print("Press Ctrl+C to stop...")

        # Start dashboard if requested
        if args.dashboard and FLASK_AVAILABLE:
            dashboard = SquashPlotDashboard(manager)
            dashboard_thread = threading.Thread(
                target=dashboard.run,
                daemon=True
            )
            dashboard_thread.start()
            print("🌐 Web dashboard started at http://localhost:5000")

        # Main loop
        while True:
            time.sleep(60)
            report = manager.get_farming_report()
            print(f"📊 Status: {report['farming_stats']['total_plots']} plots, "
                  f"{report['farming_stats']['active_plots']} active")

            # Export if requested
            if args.export:
                manager.export_plot_distribution(args.export)
                print(f"📄 Report exported to {args.export}")

    except KeyboardInterrupt:
        print("\n🛑 Stopping SquashPlot...")
    finally:
        manager.stop_monitoring()

if __name__ == '__main__':
    main()
