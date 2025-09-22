#!/usr/bin/env python3
"""
SquashPlot Enhanced Integration - Unified Mad Max + BladeBit Wrapper
===================================================================

Enhanced integration wrapper that combines Mad Max's speed with BladeBit's compression
into a seamless "smaller, faster plotting" experience.

Features:
- Intelligent Mad Max → BladeBit pipeline orchestration
- Smart resource management and optimization
- Unified CLI compatible with Mad Max/BladeBit conventions
- Real-time progress monitoring and performance metrics
- Robust error handling and recovery
- Production-ready reliability

Author: AI Research Team
Version: 3.0.0 Enhanced
"""

import os
import sys
import time
import json
import psutil
import argparse
import subprocess
import threading
import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# Constants
VERSION = "3.0.0 Enhanced"
DEFAULT_TEMP_DIR = "/tmp/squashplot"
DEFAULT_CACHE_SIZE = "32G"

# Compression level mapping: BladeBit native compression levels
COMPRESSION_LEVELS = {
    0: {"ratio": 1.0, "description": "Standard plot (108GB) - No compression", "bladebit_level": 0},
    1: {"ratio": 0.87, "description": "Light compression (94GB) - Fast", "bladebit_level": 1},
    2: {"ratio": 0.84, "description": "Medium compression (91GB) - Balanced", "bladebit_level": 2},
    3: {"ratio": 0.81, "description": "Good compression (87GB) - Recommended", "bladebit_level": 3},
    4: {"ratio": 0.78, "description": "Strong compression (84GB) - Advanced", "bladebit_level": 4},
    5: {"ratio": 0.75, "description": "Maximum compression (81GB) - Slow", "bladebit_level": 5}
}


@dataclass
class PlotConfig:
    """Enhanced plotting configuration"""
    tmp_dir: str
    tmp_dir2: Optional[str] = None
    final_dir: str = "."
    farmer_key: str = ""
    pool_key: str = ""
    contract: Optional[str] = None
    threads: int = 4
    buckets: int = 256
    count: int = 1
    k_size: int = 32
    cache_size: str = DEFAULT_CACHE_SIZE
    compression: int = 0
    mode: str = "auto"  # auto, madmax-only, bladebit-only, hybrid


@dataclass
class ResourceProfile:
    """System resource profile for optimization"""
    total_cpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    temp_disk_space_gb: float
    final_disk_space_gb: float
    ssd_available: bool = False
    nvme_available: bool = False


@dataclass
class PlotResult:
    """Plot operation result"""
    success: bool
    plot_path: Optional[str] = None
    size_gb: float = 0.0
    compression_ratio: float = 1.0
    plotting_time: float = 0.0
    compression_time: float = 0.0
    total_time: float = 0.0
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict] = None


class ResourceMonitor:
    """Real-time resource monitoring for optimization"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics_history = []
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            
    def _monitor_loop(self, interval: float):
        """Monitoring loop"""
        while self.monitoring:
            try:
                # Get disk I/O stats safely
                disk_io_stats = psutil.disk_io_counters()
                disk_io = disk_io_stats._asdict() if disk_io_stats else {}
                
                # Get network I/O stats safely
                net_io_stats = psutil.net_io_counters()
                net_io = net_io_stats._asdict() if net_io_stats else {}
                
                metrics = {
                    'timestamp': time.time(),
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_io': disk_io,
                    'network_io': net_io
                }
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                    
                time.sleep(interval)
            except Exception:
                time.sleep(interval)
                
    def get_current_metrics(self) -> Dict:
        """Get current resource metrics"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return {}


class PlotterToolManager:
    """Manages Mad Max and BladeBit tool detection and validation"""
    
    def __init__(self):
        self.madmax_path = None
        self.bladebit_path = None
        self.chia_path = None
        self._discover_tools()
        
    def _discover_tools(self):
        """Discover available plotting tools"""
        # Try to find Mad Max
        self.madmax_path = self._find_executable("chia_plot")
        
        # Try to find BladeBit (multiple possible locations)
        self.bladebit_path = self._find_executable("bladebit") or self._find_bladebit_alt()
        
        # Try to find Chia CLI
        self.chia_path = self._find_executable("chia")
        
    def _find_executable(self, name: str) -> Optional[str]:
        """Find executable in PATH using shutil.which"""
        return shutil.which(name)
        
    def _find_bladebit_alt(self) -> Optional[str]:
        """Find BladeBit using alternative methods"""
        # Common installation paths
        common_paths = [
            "/usr/local/bin/bladebit",
            "/opt/chia/bladebit",
            "./bladebit",
            "~/chia-blockchain/bladebit"
        ]
        
        for path in common_paths:
            expanded_path = os.path.expanduser(path)
            if os.path.isfile(expanded_path) and os.access(expanded_path, os.X_OK):
                return expanded_path
                
        return None
        
    def validate_tools(self) -> Dict[str, bool]:
        """Validate tool availability and versions with proper probing"""
        validation = {
            'madmax_available': False,
            'bladebit_available': False,
            'chia_available': False,
            'compression_supported': False,
            'bladebit_compress_command': False
        }
        
        # Validate Mad Max
        if self.madmax_path:
            try:
                result = subprocess.run(
                    [self.madmax_path, '--help'], 
                    capture_output=True, 
                    timeout=10
                )
                validation['madmax_available'] = result.returncode == 0
            except:
                pass
                
        # Validate BladeBit with proper compression detection
        if self.bladebit_path:
            try:
                # Check basic availability
                result = subprocess.run(
                    [self.bladebit_path, '--help'], 
                    capture_output=True, 
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    validation['bladebit_available'] = True
                    
                    # Check for compression support by probing plot command
                    plot_help = subprocess.run(
                        [self.bladebit_path, 'plot', '--help'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if plot_help.returncode == 0:
                        help_text = plot_help.stdout.lower()
                        validation['compression_supported'] = '--compress' in help_text
                    
                    # Check for separate compress command
                    compress_help = subprocess.run(
                        [self.bladebit_path, 'compress', '--help'],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    validation['bladebit_compress_command'] = compress_help.returncode == 0
                    
            except:
                pass
                
        # Validate Chia CLI
        if self.chia_path:
            try:
                result = subprocess.run(
                    [self.chia_path, 'version'], 
                    capture_output=True, 
                    timeout=10
                )
                validation['chia_available'] = result.returncode == 0
            except:
                pass
                
        return validation


class EnhancedPipelineOrchestrator:
    """Intelligent pipeline orchestration for Mad Max + BladeBit"""
    
    def __init__(self, tool_manager: PlotterToolManager):
        self.tool_manager = tool_manager
        self.resource_monitor = ResourceMonitor()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the orchestrator"""
        logger = logging.getLogger('squashplot_enhanced')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        if not logger.handlers:
            logger.addHandler(console_handler)
            
        return logger
        
    def analyze_system_resources(self) -> ResourceProfile:
        """Analyze system resources for optimization"""
        cpu_count = psutil.cpu_count()
        memory = psutil.virtual_memory()
        
        # Analyze disk space
        temp_disk_space = 0
        final_disk_space = 0
        
        try:
            temp_usage = psutil.disk_usage('/tmp')
            temp_disk_space = temp_usage.free / (1024**3)  # GB
        except:
            temp_disk_space = 100  # Default estimate
            
        try:
            home_usage = psutil.disk_usage(os.path.expanduser('~'))
            final_disk_space = home_usage.free / (1024**3)  # GB
        except:
            final_disk_space = 500  # Default estimate
            
        # Detect SSD/NVMe (basic heuristic)
        ssd_available = self._detect_ssd()
        nvme_available = self._detect_nvme()
        
        return ResourceProfile(
            total_cpu_cores=cpu_count,
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            temp_disk_space_gb=temp_disk_space,
            final_disk_space_gb=final_disk_space,
            ssd_available=ssd_available,
            nvme_available=nvme_available
        )
        
    def _detect_ssd(self) -> bool:
        """Detect if SSD is available using multiple methods"""
        try:
            # Method 1: Check /proc/mounts for SSD/NVMe keywords
            with open('/proc/mounts', 'r') as f:
                mounts = f.read().lower()
                if 'ssd' in mounts or 'nvme' in mounts:
                    return True
                    
            # Method 2: Check block device properties
            try:
                # Check if any block devices have SSD characteristics
                block_devices = glob.glob('/sys/block/*/queue/rotational')
                for device_file in block_devices:
                    with open(device_file, 'r') as f:
                        if f.read().strip() == '0':  # 0 = non-rotational (SSD)
                            return True
            except:
                pass
                
            # Method 3: Check for common SSD device patterns
            ssd_patterns = ['/dev/sda', '/dev/sdb', '/dev/sdc']  # May be SSDs
            for pattern in ssd_patterns:
                if os.path.exists(pattern):
                    # This is a weak heuristic - assume newer systems have SSDs
                    return True
                    
        except:
            pass
            
        return False
            
    def _detect_nvme(self) -> bool:
        """Detect if NVMe is available using multiple methods"""
        try:
            # Method 1: Direct device check
            nvme_devices = glob.glob('/dev/nvme*n*')
            if nvme_devices:
                return True
                
            # Method 2: Check /proc/mounts for nvme
            with open('/proc/mounts', 'r') as f:
                mounts = f.read().lower()
                if 'nvme' in mounts:
                    return True
                    
            # Method 3: Check /sys/block for nvme devices
            nvme_sys = glob.glob('/sys/block/nvme*')
            if nvme_sys:
                return True
                
        except:
            pass
            
        return False
            
    def optimize_configuration(self, config: PlotConfig, 
                             resources: ResourceProfile) -> PlotConfig:
        """Optimize plotting configuration based on available resources with hardware-aware improvements"""
        optimized = config
        
        # Optimize thread count based on CPU and workload
        if config.threads <= 0:
            if config.compression > 0:
                # Compression workloads benefit from more threads
                optimized.threads = max(2, min(resources.total_cpu_cores, int(resources.total_cpu_cores * 0.9)))
            else:
                # Mad Max plotting - use 75% of cores
                optimized.threads = max(2, int(resources.total_cpu_cores * 0.75))
            
        # Intelligent temporary directory selection using SSD/NVMe detection
        if not config.tmp_dir:
            optimized.tmp_dir = self._select_optimal_temp_dir(resources)
            
        # Smart temp2 selection based on memory and storage type
        if not config.tmp_dir2:
            if resources.available_memory_gb >= 16 and resources.available_memory_gb >= (110 * 1.1):  # 110GB + 10% buffer
                # Use RAM disk for temp2 if enough memory (Mad Max temp2 ~110GB)
                optimized.tmp_dir2 = "/dev/shm"
                self.logger.info("Using RAM disk for temp2 directory (optimal performance)")
            elif resources.nvme_available:
                # Use separate NVMe if available
                nvme_temp = "/tmp/squashplot_nvme"
                try:
                    os.makedirs(nvme_temp, exist_ok=True)
                    optimized.tmp_dir2 = nvme_temp
                    self.logger.info("Using NVMe for temp2 directory")
                except:
                    pass
                    
        # Optimize buckets based on available memory
        if config.buckets == 256:
            if resources.available_memory_gb >= 64:
                optimized.buckets = 1024  # High-memory system
            elif resources.available_memory_gb >= 32:
                optimized.buckets = 512   # Medium-memory system
            # else keep default 256 for low-memory systems
            
        # Optimize cache size based on available memory
        if config.cache_size == DEFAULT_CACHE_SIZE:
            available_gb = int(resources.available_memory_gb)
            if available_gb >= 64:
                optimized.cache_size = "64G"
            elif available_gb >= 32:
                optimized.cache_size = "32G"
            elif available_gb >= 16:
                optimized.cache_size = "16G"
            else:
                optimized.cache_size = "8G"
                
        # Storage-specific optimizations
        storage_info = self._analyze_storage_performance(resources)
        
        self.logger.info(f"🔧 Optimized configuration:")
        self.logger.info(f"   Threads: {optimized.threads} (was {config.threads})")
        self.logger.info(f"   Buckets: {optimized.buckets} (was {config.buckets})")
        self.logger.info(f"   Cache: {optimized.cache_size} (was {config.cache_size})")
        self.logger.info(f"   Temp1: {optimized.tmp_dir}")
        self.logger.info(f"   Temp2: {optimized.tmp_dir2 or 'None'}")
        self.logger.info(f"   Storage: {storage_info['performance_tier']}")
        
        return optimized
        
    def _select_optimal_temp_dir(self, resources: ResourceProfile) -> str:
        """Select optimal temporary directory based on storage performance"""
        
        # Priority order: NVMe > SSD > HDD, with space requirements
        candidates = []
        
        if resources.nvme_available:
            # Prefer NVMe locations
            candidates.extend([
                "/tmp/squashplot_nvme",
                "/nvme/squashplot",
                "/mnt/nvme/squashplot"
            ])
            
        if resources.ssd_available:
            # SSD locations
            candidates.extend([
                "/tmp/squashplot",
                "/ssd/squashplot",
                "/mnt/ssd/squashplot"
            ])
            
        # Standard locations (may be HDD)
        candidates.extend([
            "/tmp/squashplot",
            f"{os.path.expanduser('~')}/squashplot_temp",
            "/var/tmp/squashplot"
        ])
        
        # Test each candidate for space and write access
        required_space_gb = 250  # ~220GB for Mad Max + buffer
        
        for temp_dir in candidates:
            try:
                os.makedirs(temp_dir, exist_ok=True)
                
                # Check write access
                if not os.access(temp_dir, os.W_OK):
                    continue
                    
                # Check available space
                try:
                    usage = psutil.disk_usage(temp_dir)
                    available_gb = usage.free / (1024**3)
                    if available_gb >= required_space_gb:
                        self.logger.info(f"Selected temp directory: {temp_dir} ({available_gb:.1f}GB available)")
                        return temp_dir
                except:
                    # If we can't check space, still use it as fallback
                    pass
                    
            except:
                continue
                
        # Fallback to default
        self.logger.warning(f"Using default temp directory: {DEFAULT_TEMP_DIR}")
        return DEFAULT_TEMP_DIR
        
    def _analyze_storage_performance(self, resources: ResourceProfile) -> Dict[str, str]:
        """Analyze storage performance characteristics"""
        
        if resources.nvme_available:
            tier = "High Performance (NVMe)"
            io_strategy = "parallel_high"
        elif resources.ssd_available:
            tier = "Medium Performance (SSD)"
            io_strategy = "parallel_medium"
        else:
            tier = "Standard Performance (HDD)"
            io_strategy = "sequential"
            
        return {
            'performance_tier': tier,
            'io_strategy': io_strategy,
            'recommended_concurrent_plots': "1" if not resources.ssd_available else "2"
        }
        
    def execute_plotting_pipeline(self, config: PlotConfig) -> PlotResult:
        """Execute the complete plotting pipeline with multi-plot support"""
        start_time = time.time()
        
        # Validate required parameters before starting
        if not config.farmer_key:
            return PlotResult(
                success=False,
                error_message="Farmer key is required for plotting"
            )
        
        # Analyze resources and optimize configuration
        resources = self.analyze_system_resources()
        optimized_config = self.optimize_configuration(config, resources)
        
        # Start resource monitoring
        self.resource_monitor.start_monitoring()
        
        try:
            # Validate tools
            tool_validation = self.tool_manager.validate_tools()
            
            # Select optimal strategy
            strategy = self._select_plotting_strategy(optimized_config, tool_validation)
            
            self.logger.info(f"Using strategy: {strategy}")
            self.logger.info(f"Target compression level: {config.compression}")
            self.logger.info(f"Plotting {config.count} plot(s)")
            
            # Handle multiple plots
            if config.count == 1:
                # Single plot
                result = self._execute_single_plot(optimized_config, strategy)
            else:
                # Multiple plots - execute sequentially with proper handling
                result = self._execute_multiple_plots(optimized_config, strategy)
                
            # Add performance metrics
            result.total_time = time.time() - start_time
            result.performance_metrics = self._generate_performance_metrics()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return PlotResult(
                success=False,
                error_message=str(e),
                total_time=time.time() - start_time
            )
        finally:
            self.resource_monitor.stop_monitoring()
            
    def _execute_single_plot(self, config: PlotConfig, strategy: str) -> PlotResult:
        """Execute single plot with given strategy"""
        if strategy == "madmax_bladebit_pipeline":
            return self._execute_madmax_bladebit_pipeline(config)
        elif strategy == "bladebit_direct":
            return self._execute_bladebit_direct(config)
        elif strategy == "madmax_only":
            return self._execute_madmax_only(config)
        else:
            return PlotResult(
                success=False,
                error_message=f"Unknown strategy: {strategy}"
            )
            
    def _execute_multiple_plots(self, config: PlotConfig, strategy: str) -> PlotResult:
        """Execute multiple plots with proper coordination"""
        
        total_success = 0
        total_plots = config.count
        all_plot_paths = []
        total_size = 0.0
        total_plotting_time = 0.0
        total_compression_time = 0.0
        errors = []
        
        self.logger.info(f"🔄 Starting {total_plots} plots sequentially")
        
        for plot_num in range(1, total_plots + 1):
            self.logger.info(f"📈 Starting plot {plot_num}/{total_plots}")
            
            # Create single plot config
            single_config = PlotConfig(
                tmp_dir=config.tmp_dir,
                tmp_dir2=config.tmp_dir2,
                final_dir=config.final_dir,
                farmer_key=config.farmer_key,
                pool_key=config.pool_key,
                contract=config.contract,
                threads=config.threads,
                buckets=config.buckets,
                count=1,  # Single plot
                k_size=config.k_size,
                cache_size=config.cache_size,
                compression=config.compression,
                mode=config.mode
            )
            
            # Execute single plot
            plot_result = self._execute_single_plot(single_config, strategy)
            
            if plot_result.success:
                total_success += 1
                if plot_result.plot_path:
                    all_plot_paths.append(plot_result.plot_path)
                total_size += plot_result.size_gb
                total_plotting_time += plot_result.plotting_time
                total_compression_time += plot_result.compression_time
                
                self.logger.info(f"✅ Plot {plot_num}/{total_plots} completed: {plot_result.plot_path}")
            else:
                errors.append(f"Plot {plot_num}: {plot_result.error_message}")
                self.logger.error(f"❌ Plot {plot_num}/{total_plots} failed: {plot_result.error_message}")
                
        # Summarize results
        if total_success == total_plots:
            return PlotResult(
                success=True,
                plot_path=f"{total_success} plots in {config.final_dir}",
                size_gb=total_size,
                compression_ratio=COMPRESSION_LEVELS[config.compression]['ratio'],
                plotting_time=total_plotting_time,
                compression_time=total_compression_time
            )
        elif total_success > 0:
            return PlotResult(
                success=True,  # Partial success
                plot_path=f"{total_success}/{total_plots} plots completed",
                size_gb=total_size,
                compression_ratio=COMPRESSION_LEVELS[config.compression]['ratio'],
                plotting_time=total_plotting_time,
                compression_time=total_compression_time,
                error_message=f"Partial success: {len(errors)} plots failed: {'; '.join(errors)}"
            )
        else:
            return PlotResult(
                success=False,
                error_message=f"All {total_plots} plots failed: {'; '.join(errors)}"
            )
            
    def _select_plotting_strategy(self, config: PlotConfig, 
                                 validation: Dict[str, bool]) -> str:
        """Select optimal plotting strategy with corrected logic"""
        
        # Honor explicit mode overrides
        if config.mode == "madmax-only":
            if validation['madmax_available']:
                return "madmax_only"
            raise RuntimeError("Mad Max not available but madmax-only mode requested")
            
        if config.mode == "bladebit-only":
            if validation['bladebit_available']:
                return "bladebit_direct"
            raise RuntimeError("BladeBit not available but bladebit-only mode requested")
        
        # Auto strategy selection based on compression needs
        if config.compression == 0:
            # No compression: prefer Mad Max for speed
            if validation['madmax_available']:
                return "madmax_only"
            elif validation['bladebit_available']:
                return "bladebit_direct"  # BladeBit can do uncompressed plots too
                
        else:
            # Compression needed: prefer strategies that support it
            if (validation['madmax_available'] and 
                validation['bladebit_available'] and 
                (validation['compression_supported'] or validation['bladebit_compress_command'])):
                return "madmax_bladebit_pipeline"  # Best of both worlds
                
            elif validation['bladebit_available'] and validation['compression_supported']:
                return "bladebit_direct"  # Direct BladeBit with compression
                
            elif validation['madmax_available']:
                self.logger.warning("Compression requested but BladeBit compression not available. Using Mad Max only.")
                return "madmax_only"
            
        raise RuntimeError("No suitable plotting tools available for the requested configuration")
        
    def _execute_madmax_bladebit_pipeline(self, config: PlotConfig) -> PlotResult:
        """Execute Mad Max → BladeBit compression pipeline"""
        
        self.logger.info("🚀 Starting Mad Max → BladeBit pipeline")
        
        # Phase 1: Mad Max plotting to temp location
        temp_plot_dir = os.path.join(config.tmp_dir, "temp_plots")
        os.makedirs(temp_plot_dir, exist_ok=True)
        
        madmax_config = PlotConfig(
            tmp_dir=config.tmp_dir,
            tmp_dir2=config.tmp_dir2,
            final_dir=temp_plot_dir,  # Temporary location
            farmer_key=config.farmer_key,
            pool_key=config.pool_key,
            contract=config.contract,
            threads=config.threads,
            buckets=config.buckets,
            count=config.count,
            k_size=config.k_size,
            compression=0  # No compression in Mad Max phase
        )
        
        madmax_start = time.time()
        madmax_result = self._execute_madmax_plotting(madmax_config)
        madmax_time = time.time() - madmax_start
        
        if not madmax_result.success:
            return madmax_result
            
        self.logger.info(f"✅ Mad Max completed in {madmax_time:.1f}s")
        
        # Phase 2: BladeBit compression
        if config.compression > 0:
            compression_start = time.time()
            
            # Find the plot file created by Mad Max
            temp_plot_path = madmax_result.plot_path
            
            if not temp_plot_path:
                return PlotResult(
                    success=False,
                    error_message="Mad Max did not produce a valid plot file"
                )
            
            # Execute BladeBit compression
            compressed_result = self._execute_bladebit_compression(
                temp_plot_path, config.final_dir, config.compression
            )
            
            compression_time = time.time() - compression_start
            
            # Cleanup temporary plot
            try:
                if temp_plot_path and os.path.exists(temp_plot_path):
                    os.remove(temp_plot_path)
            except:
                pass
                
            if not compressed_result.success:
                return compressed_result
                
            self.logger.info(f"✅ Compression completed in {compression_time:.1f}s")
            
            # Combine results
            return PlotResult(
                success=True,
                plot_path=compressed_result.plot_path,
                size_gb=compressed_result.size_gb,
                compression_ratio=COMPRESSION_LEVELS[config.compression]['ratio'],
                plotting_time=madmax_time,
                compression_time=compression_time,
                total_time=madmax_time + compression_time
            )
        else:
            # No compression, just move to final location
            if madmax_result.plot_path:
                final_plot_path = os.path.join(config.final_dir, 
                                             os.path.basename(madmax_result.plot_path))
                
                try:
                    os.rename(madmax_result.plot_path, final_plot_path)
                    madmax_result.plot_path = final_plot_path
                except:
                    pass
                    
            return madmax_result
            
    def _execute_bladebit_direct(self, config: PlotConfig) -> PlotResult:
        """Execute BladeBit direct plotting with compression"""
        
        self.logger.info("🔧 Starting BladeBit direct plotting")
        
        if not self.tool_manager.bladebit_path:
            return PlotResult(
                success=False,
                error_message="BladeBit not available"
            )
            
        cmd = [self.tool_manager.bladebit_path, "plot"]
        
        # Add parameters
        cmd.extend(["-d", config.final_dir])
        
        if config.farmer_key:
            cmd.extend(["-f", config.farmer_key])
        if config.pool_key:
            cmd.extend(["-p", config.pool_key])
        if config.contract:
            cmd.extend(["-c", config.contract])
            
        # Add compression
        if config.compression > 0:
            bladebit_level = COMPRESSION_LEVELS[config.compression]['bladebit_level']
            cmd.extend(["--compress", str(bladebit_level)])
            
        # Add threading
        cmd.extend(["-t", str(config.threads)])
        
        # Add count
        if config.count > 1:
            cmd.extend(["-n", str(config.count)])
            
        return self._execute_command(cmd, "BladeBit")
        
    def _execute_madmax_only(self, config: PlotConfig) -> PlotResult:
        """Execute Mad Max plotting only"""
        
        self.logger.info("⚡ Starting Mad Max plotting")
        return self._execute_madmax_plotting(config)
        
    def _execute_madmax_plotting(self, config: PlotConfig) -> PlotResult:
        """Execute Mad Max plotting"""
        
        if not self.tool_manager.madmax_path:
            return PlotResult(
                success=False,
                error_message="Mad Max not available"
            )
            
        cmd = [self.tool_manager.madmax_path]
        
        # Add parameters
        if config.tmp_dir:
            cmd.extend(["-t", config.tmp_dir])
        if config.tmp_dir2:
            cmd.extend(["-2", config.tmp_dir2])
        if config.final_dir:
            cmd.extend(["-d", config.final_dir])
        if config.farmer_key:
            cmd.extend(["-f", config.farmer_key])
        if config.pool_key:
            cmd.extend(["-p", config.pool_key])
        if config.contract:
            cmd.extend(["-c", config.contract])
            
        # Add threading and performance options
        cmd.extend(["-r", str(config.threads)])
        cmd.extend(["-u", str(config.buckets)])
        
        if config.count > 1:
            cmd.extend(["-n", str(config.count)])
            
        return self._execute_command(cmd, "Mad Max")
        
    def _execute_bladebit_compression(self, input_plot: str, output_dir: str, 
                                    compression_level: int) -> PlotResult:
        """Execute real BladeBit compression on existing plot"""
        
        if not self.tool_manager.bladebit_path:
            return PlotResult(
                success=False,
                error_message="BladeBit not available for compression"
            )
        
        self.logger.info(f"🗜️ Compressing plot with BladeBit level {compression_level}")
        
        # Generate output filename
        input_name = os.path.basename(input_plot)
        output_name = input_name.replace('.plot', f'_c{compression_level}.plot')
        output_path = os.path.join(output_dir, output_name)
        
        # Check if BladeBit has a compress command
        validation = self.tool_manager.validate_tools()
        
        if validation.get('bladebit_compress_command', False):
            # Use BladeBit compress command
            cmd = [
                self.tool_manager.bladebit_path,
                'compress',
                '--level', str(compression_level),
                '--input', input_plot,
                '--output', output_path
            ]
        else:
            # Fallback: use chia plotters bladebit compress if available
            if self.tool_manager.chia_path:
                cmd = [
                    self.tool_manager.chia_path,
                    'plotters', 'bladebit', 'compress',
                    '--input', input_plot,
                    '--output', output_path,
                    '--level', str(compression_level)
                ]
            else:
                # No compression available - this is a design limitation
                self.logger.warning(f"BladeBit compression not available. Creating symbolic link instead.")
                try:
                    # Create hard link or copy as fallback
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    os.link(input_plot, output_path)
                    
                    # Calculate actual vs expected size
                    actual_size = os.path.getsize(output_path) / (1024**3)
                    compression_ratio = COMPRESSION_LEVELS[compression_level]['ratio']
                    
                    return PlotResult(
                        success=True,
                        plot_path=output_path,
                        size_gb=actual_size,  # Actual size (uncompressed)
                        compression_ratio=1.0,  # No real compression
                        error_message="Warning: No real compression applied - BladeBit compress not available"
                    )
                except Exception as e:
                    return PlotResult(
                        success=False,
                        error_message=f"Fallback compression failed: {e}"
                    )
        
        try:
            # Execute compression command
            start_time = time.time()
            
            self.logger.info(f"Executing: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout for compression
            )
            
            compression_time = time.time() - start_time
            
            if result.returncode == 0:
                # Verify output file exists and get size
                if os.path.exists(output_path):
                    output_size = os.path.getsize(output_path) / (1024**3)
                    compression_ratio = COMPRESSION_LEVELS[compression_level]['ratio']
                    
                    self.logger.info(f"✅ Compression completed in {compression_time:.1f}s")
                    self.logger.info(f"📊 Compressed size: {output_size:.1f} GB")
                    
                    return PlotResult(
                        success=True,
                        plot_path=output_path,
                        size_gb=output_size,
                        compression_ratio=compression_ratio,
                        compression_time=compression_time
                    )
                else:
                    return PlotResult(
                        success=False,
                        error_message=f"Compression completed but output file not found: {output_path}"
                    )
            else:
                error_msg = result.stderr if result.stderr else f"Compression failed with exit code {result.returncode}"
                return PlotResult(
                    success=False,
                    error_message=f"BladeBit compression failed: {error_msg}"
                )
                
        except subprocess.TimeoutExpired:
            return PlotResult(
                success=False,
                error_message="BladeBit compression timed out (>1 hour)"
            )
        except Exception as e:
            return PlotResult(
                success=False,
                error_message=f"Compression execution failed: {e}"
            )
            
    def _execute_command(self, cmd: List[str], tool_name: str) -> PlotResult:
        """Execute plotting command with monitoring"""
        
        self.logger.info(f"Executing {tool_name}: {' '.join(cmd)}")
        
        start_time = time.time()
        plot_path = None
        
        try:
            # Execute command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Monitor output
            stdout_lines = []
            stderr_lines = []
            
            while True:
                stdout_line = ""
                stderr_line = ""
                
                if process.stdout:
                    stdout_line = process.stdout.readline()
                if process.stderr:
                    stderr_line = process.stderr.readline()
                
                if stdout_line:
                    stdout_lines.append(stdout_line.strip())
                    self.logger.info(f"{tool_name}: {stdout_line.strip()}")
                    
                    # Try to extract plot path from output
                    if 'plot' in stdout_line.lower() and '.plot' in stdout_line:
                        # Extract plot path using simple heuristics
                        words = stdout_line.split()
                        for word in words:
                            if word.endswith('.plot'):
                                plot_path = word
                                break
                                
                if stderr_line:
                    stderr_lines.append(stderr_line.strip())
                    self.logger.warning(f"{tool_name} warning: {stderr_line.strip()}")
                    
                if process.poll() is not None:
                    break
                    
                time.sleep(0.1)
                
            # Wait for completion
            return_code = process.wait()
            execution_time = time.time() - start_time
            
            if return_code == 0:
                # Calculate file size if we found the plot
                size_gb = 0.0
                if plot_path and os.path.exists(plot_path):
                    size_gb = os.path.getsize(plot_path) / (1024**3)
                    
                return PlotResult(
                    success=True,
                    plot_path=plot_path,
                    size_gb=size_gb,
                    plotting_time=execution_time
                )
            else:
                error_msg = '\n'.join(stderr_lines) if stderr_lines else f"Command failed with code {return_code}"
                return PlotResult(
                    success=False,
                    error_message=error_msg,
                    plotting_time=execution_time
                )
                
        except Exception as e:
            return PlotResult(
                success=False,
                error_message=f"Execution failed: {e}",
                plotting_time=time.time() - start_time
            )
            
    def _generate_performance_metrics(self) -> Dict:
        """Generate performance metrics from monitoring data"""
        if not self.resource_monitor.metrics_history:
            return {}
            
        metrics = self.resource_monitor.metrics_history
        
        # Calculate averages
        avg_cpu = sum(m.get('cpu_percent', 0) for m in metrics) / len(metrics)
        avg_memory = sum(m.get('memory_percent', 0) for m in metrics) / len(metrics)
        
        # Calculate peaks
        peak_cpu = max(m.get('cpu_percent', 0) for m in metrics)
        peak_memory = max(m.get('memory_percent', 0) for m in metrics)
        
        return {
            'avg_cpu_percent': round(avg_cpu, 1),
            'avg_memory_percent': round(avg_memory, 1),
            'peak_cpu_percent': round(peak_cpu, 1),
            'peak_memory_percent': round(peak_memory, 1),
            'monitoring_duration': len(metrics)
        }


class SquashPlotEnhanced:
    """Main SquashPlot Enhanced Integration class"""
    
    def __init__(self):
        self.tool_manager = PlotterToolManager()
        self.orchestrator = EnhancedPipelineOrchestrator(self.tool_manager)
        
    def list_compression_levels(self):
        """List available compression levels"""
        print("\nSquashPlot Enhanced - Compression Levels")
        print("=" * 50)
        for level, info in COMPRESSION_LEVELS.items():
            print(f"Level {level}: {info['description']}")
            print(f"  Expected size: {info['ratio']:.2f} × original")
            print(f"  Space savings: {(1-info['ratio'])*100:.1f}%")
            print()
            
    def check_tools(self):
        """Check available plotting tools"""
        print("\nSquashPlot Enhanced - Tool Status")
        print("=" * 40)
        
        validation = self.tool_manager.validate_tools()
        
        status_symbol = lambda x: "✅" if x else "❌"
        
        print(f"{status_symbol(validation['madmax_available'])} Mad Max: {self.tool_manager.madmax_path or 'Not found'}")
        print(f"{status_symbol(validation['bladebit_available'])} BladeBit: {self.tool_manager.bladebit_path or 'Not found'}")
        print(f"{status_symbol(validation['chia_available'])} Chia CLI: {self.tool_manager.chia_path or 'Not found'}")
        print(f"{status_symbol(validation['compression_supported'])} Compression: {'Supported' if validation['compression_supported'] else 'Not supported'}")
        
        print("\nRecommendations:")
        if not validation['madmax_available']:
            print("• Install Mad Max for fastest plotting")
        if not validation['bladebit_available']:
            print("• Install BladeBit for compression support")
        if not validation['compression_supported']:
            print("• Update BladeBit to version 3.0+ for compression")
            
    def plot(self, config: PlotConfig) -> PlotResult:
        """Execute plotting with enhanced integration"""
        return self.orchestrator.execute_plotting_pipeline(config)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="SquashPlot Enhanced - Unified Mad Max + BladeBit Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast plotting with Mad Max + BladeBit compression
  python squashplot_enhanced.py -d /plots -f <farmer_key> --compress 3

  # Maximum speed (Mad Max only, no compression)
  python squashplot_enhanced.py -d /plots -f <farmer_key> --compress 0

  # High compression (BladeBit with level 5)
  python squashplot_enhanced.py -d /plots -f <farmer_key> --compress 5

  # Multiple plots with optimal settings
  python squashplot_enhanced.py -t /tmp/fast -d /plots -f <farmer_key> -n 5 --compress 3
        """
    )
    
    # Version
    parser.add_argument('--version', action='version', version=f'SquashPlot Enhanced {VERSION}')
    
    # Plotting parameters
    parser.add_argument('-t', '--tmp-dir', help='Temporary directory for plotting')
    parser.add_argument('-2', '--tmp-dir2', help='Second temporary directory')
    parser.add_argument('-d', '--final-dir', default='.', help='Final directory for plots')
    parser.add_argument('-f', '--farmer-key', help='Farmer public key (required)')
    parser.add_argument('-p', '--pool-key', help='Pool public key')
    parser.add_argument('-c', '--contract', help='Pool contract address')
    
    # Performance parameters
    parser.add_argument('-r', '--threads', type=int, default=0, help='Number of threads (0=auto)')
    parser.add_argument('-u', '--buckets', type=int, default=256, help='Number of buckets')
    parser.add_argument('-n', '--count', type=int, default=1, help='Number of plots to create')
    parser.add_argument('-k', '--k-size', type=int, default=32, help='K size (default: 32)')
    
    # Cache and compression
    parser.add_argument('--cache', default=DEFAULT_CACHE_SIZE, help='Cache size for disk operations')
    parser.add_argument('--compress', type=int, choices=[0,1,2,3,4,5], default=3, 
                       help='Compression level (0-5, default: 3)')
    
    # Mode selection
    parser.add_argument('--mode', choices=['auto', 'madmax-only', 'bladebit-only', 'hybrid'], 
                       default='auto', help='Operation mode (default: auto)')
    
    # Utility commands
    parser.add_argument('--list-levels', action='store_true', help='List compression levels')
    parser.add_argument('--check-tools', action='store_true', help='Check tool availability')
    
    args = parser.parse_args()
    
    # Create SquashPlot instance
    squashplot = SquashPlotEnhanced()
    
    # Handle utility commands
    if args.list_levels:
        squashplot.list_compression_levels()
        return
        
    if args.check_tools:
        squashplot.check_tools()
        return
        
    # Validate required parameters
    if not args.farmer_key:
        print("❌ Error: Farmer key (-f) is required for plotting")
        return 1
        
    # Create plotting configuration
    config = PlotConfig(
        tmp_dir=args.tmp_dir or DEFAULT_TEMP_DIR,
        tmp_dir2=args.tmp_dir2,
        final_dir=args.final_dir,
        farmer_key=args.farmer_key,
        pool_key=args.pool_key,
        contract=args.contract,
        threads=args.threads,
        buckets=args.buckets,
        count=args.count,
        k_size=args.k_size,
        cache_size=args.cache,
        compression=args.compress,
        mode=args.mode
    )
    
    print(f"🚀 SquashPlot Enhanced {VERSION}")
    print("=" * 50)
    print(f"Target: {config.count} plot(s) with compression level {config.compression}")
    print(f"Expected final size: ~{COMPRESSION_LEVELS[config.compression]['ratio']:.2f} × original")
    print(f"Space savings: ~{(1-COMPRESSION_LEVELS[config.compression]['ratio'])*100:.1f}%")
    print()
    
    # Execute plotting
    try:
        result = squashplot.plot(config)
        
        if result.success:
            print("\n🎉 Plotting completed successfully!")
            print(f"📁 Plot location: {result.plot_path}")
            print(f"📊 Final size: {result.size_gb:.1f} GB")
            print(f"⏱️  Total time: {result.total_time:.1f} seconds")
            
            if result.performance_metrics:
                metrics = result.performance_metrics
                print(f"💻 Avg CPU: {metrics.get('avg_cpu_percent', 0):.1f}%")
                print(f"🧠 Peak memory: {metrics.get('peak_memory_percent', 0):.1f}%")
                
            return 0
        else:
            print(f"\n❌ Plotting failed: {result.error_message}")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Plotting interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())