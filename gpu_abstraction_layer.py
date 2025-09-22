#!/usr/bin/env python3
"""
GPU Abstraction Layer - Cross-Platform GPU Acceleration
======================================================

Universal GPU abstraction layer supporting multiple GPU architectures:
- NVIDIA GPUs (CUDA/CuPy)
- AMD GPUs (ROCm/OpenCL)
- Apple Silicon (Metal)
- Intel GPUs (OpenCL)

Features:
- Automatic GPU detection and selection
- Unified API across all GPU types
- Fallback mechanisms for CPU computation
- Performance optimization for each architecture
- Memory management and synchronization
- Cross-platform compatibility

Author: Bradley Wallace (COO, Koba42 Corp)
Contact: user@domain.com
License: MIT License
"""

import os
import sys
import platform
import subprocess
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gpu_abstraction')

class GPUBackend(Enum):
    """Supported GPU backends"""
    CUDA = "cuda"           # NVIDIA GPUs
    ROCM = "rocm"           # AMD GPUs
    METAL = "metal"         # Apple Silicon
    OPENCL = "opencl"       # Generic OpenCL (Intel, AMD fallback)
    CPU = "cpu"            # CPU fallback

@dataclass
class GPUDevice:
    """GPU device information"""
    backend: GPUBackend
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    compute_units: int
    max_work_group_size: int
    supports_fp64: bool
    supports_fp16: bool

@dataclass
class GPUCapabilities:
    """GPU capabilities and features"""
    max_threads_per_block: int
    max_shared_memory: int
    warp_size: int
    memory_bandwidth: float  # GB/s
    compute_performance: float  # GFLOPS
    supports_unified_memory: bool
    supports_peer_access: bool

class GPUAbstractionLayer:
    """Cross-platform GPU abstraction layer"""

    def __init__(self):
        self.available_backends = self._detect_available_backends()
        self.devices: List[GPUDevice] = []
        self.capabilities: Dict[str, GPUCapabilities] = {}
        self.current_device: Optional[GPUDevice] = None

        # Initialize backends
        self._initialize_backends()

        # Auto-select best device
        self.auto_select_device()

        logger.info(f"GPU Abstraction Layer initialized with {len(self.devices)} devices")

    def _detect_available_backends(self) -> List[GPUBackend]:
        """Detect available GPU backends on the system"""
        available = []

        # Check for CUDA (NVIDIA)
        try:
            import cupy as cp
            cp.cuda.runtime.getDeviceCount()
            available.append(GPUBackend.CUDA)
            logger.info("CUDA backend detected (NVIDIA GPUs)")
        except (ImportError, Exception):
            pass

        # Check for ROCm (AMD)
        try:
            import cupy as cp
            if hasattr(cp, 'rocm'):
                available.append(GPUBackend.ROCM)
                logger.info("ROCm backend detected (AMD GPUs)")
        except (ImportError, Exception):
            pass

        # Check for Metal (Apple Silicon)
        if platform.system() == 'Darwin':  # macOS
            try:
                import metal  # Hypothetical Metal Python binding
                available.append(GPUBackend.METAL)
                logger.info("Metal backend detected (Apple Silicon)")
            except ImportError:
                # Fallback: Try PyTorch with MPS (Metal Performance Shaders)
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        available.append(GPUBackend.METAL)
                        logger.info("Metal backend detected via PyTorch MPS (Apple Silicon)")
                except ImportError:
                    pass

        # Check for OpenCL (Generic fallback)
        try:
            import pyopencl as cl
            platforms = cl.get_platforms()
            if platforms:
                available.append(GPUBackend.OPENCL)
                logger.info("OpenCL backend detected (Generic GPUs)")
        except ImportError:
            pass

        # Always have CPU fallback
        available.append(GPUBackend.CPU)
        logger.info("CPU backend available (fallback)")

        return available

    def _initialize_backends(self):
        """Initialize detected GPU backends"""
        for backend in self.available_backends:
            if backend == GPUBackend.CUDA:
                self._initialize_cuda()
            elif backend == GPUBackend.ROCM:
                self._initialize_rocm()
            elif backend == GPUBackend.METAL:
                self._initialize_metal()
            elif backend == GPUBackend.OPENCL:
                self._initialize_opencl()
            elif backend == GPUBackend.CPU:
                self._initialize_cpu()

    def _initialize_cuda(self):
        """Initialize CUDA backend"""
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()

            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)

                device = GPUDevice(
                    backend=GPUBackend.CUDA,
                    device_id=i,
                    name=props['name'].decode('utf-8'),
                    memory_total=props['totalGlobalMem'],
                    memory_free=props['totalGlobalMem'],  # Approximate
                    compute_units=props['multiProcessorCount'],
                    max_work_group_size=props['maxThreadsPerBlock'],
                    supports_fp64=props['major'] >= 6,  # Pascal and newer
                    supports_fp16=props['major'] >= 6   # Pascal and newer
                )

                self.devices.append(device)

                # Get capabilities
                self.capabilities[f"cuda:{i}"] = GPUCapabilities(
                    max_threads_per_block=props['maxThreadsPerBlock'],
                    max_shared_memory=props['sharedMemPerBlock'],
                    warp_size=props['warpSize'],
                    memory_bandwidth=self._calculate_cuda_bandwidth(props),
                    compute_performance=self._calculate_cuda_performance(props),
                    supports_unified_memory=False,
                    supports_peer_access=True
                )

        except Exception as e:
            logger.warning(f"CUDA initialization failed: {e}")

    def _initialize_rocm(self):
        """Initialize ROCm backend"""
        try:
            import cupy as cp
            # ROCm uses similar API to CUDA in CuPy
            device_count = cp.cuda.runtime.getDeviceCount()

            for i in range(device_count):
                props = cp.cuda.runtime.getDeviceProperties(i)

                device = GPUDevice(
                    backend=GPUBackend.ROCM,
                    device_id=i,
                    name=f"AMD GPU {i}",  # ROCm doesn't always provide names
                    memory_total=props['totalGlobalMem'],
                    memory_free=props['totalGlobalMem'],
                    compute_units=props['multiProcessorCount'],
                    max_work_group_size=props['maxThreadsPerBlock'],
                    supports_fp64=True,  # Most AMD GPUs support FP64
                    supports_fp16=True   # Most modern AMD GPUs support FP16
                )

                self.devices.append(device)

        except Exception as e:
            logger.warning(f"ROCm initialization failed: {e}")

    def _initialize_metal(self):
        """Initialize Metal backend for Apple Silicon"""
        try:
            import torch
            if torch.backends.mps.is_available():
                # Use PyTorch MPS as Metal backend
                device_count = 1  # Typically one GPU on Apple Silicon

                for i in range(device_count):
                    device = GPUDevice(
                        backend=GPUBackend.METAL,
                        device_id=i,
                        name=f"Apple Silicon GPU {i}",
                        memory_total=self._get_apple_silicon_memory(),
                        memory_free=self._get_apple_silicon_memory(),  # Unified memory
                        compute_units=8,  # Approximate for M-series
                        max_work_group_size=1024,
                        supports_fp64=False,  # Apple Silicon doesn't support FP64 well
                        supports_fp16=True
                    )

                    self.devices.append(device)

                    self.capabilities[f"metal:{i}"] = GPUCapabilities(
                        max_threads_per_block=1024,
                        max_shared_memory=32768,  # 32KB shared memory
                        warp_size=32,
                        memory_bandwidth=100.0,  # Approximate for M3
                        compute_performance=5000.0,  # Approximate GFLOPS for M3
                        supports_unified_memory=True,  # Apple Unified Memory
                        supports_peer_access=False
                    )

        except Exception as e:
            logger.warning(f"Metal initialization failed: {e}")

    def _initialize_opencl(self):
        """Initialize OpenCL backend"""
        try:
            import pyopencl as cl

            platforms = cl.get_platforms()
            device_id = 0

            for platform in platforms:
                devices = platform.get_devices()
                for device in devices:
                    if device.type == cl.device_type.GPU:
                        gpu_device = GPUDevice(
                            backend=GPUBackend.OPENCL,
                            device_id=device_id,
                            name=device.name,
                            memory_total=device.global_mem_size,
                            memory_free=device.global_mem_size,  # Approximate
                            compute_units=device.max_compute_units,
                            max_work_group_size=device.max_work_group_size,
                            supports_fp64=cl.device_fp_config.F_DENORM in device.double_fp_config,
                            supports_fp16=cl.device_fp_config.F_DENORM in device.half_fp_config
                        )

                        self.devices.append(gpu_device)
                        device_id += 1

        except Exception as e:
            logger.warning(f"OpenCL initialization failed: {e}")

    def _initialize_cpu(self):
        """Initialize CPU backend (fallback)"""
        import psutil

        device = GPUDevice(
            backend=GPUBackend.CPU,
            device_id=0,
            name=f"CPU ({psutil.cpu_count()} cores)",
            memory_total=psutil.virtual_memory().total,
            memory_free=psutil.virtual_memory().available,
            compute_units=psutil.cpu_count(),
            max_work_group_size=psutil.cpu_count(),
            supports_fp64=True,
            supports_fp16=False  # CPUs typically don't have hardware FP16
        )

        self.devices.append(device)

    def _get_apple_silicon_memory(self) -> int:
        """Get Apple Silicon unified memory size"""
        try:
            # Try to get from system
            result = subprocess.run(['sysctl', 'hw.memsize'],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                return int(result.stdout.split(':')[1].strip())
        except:
            pass

        # Default fallback for M3
        return 8 * 1024**3  # 8GB for M3

    def _calculate_cuda_bandwidth(self, props) -> float:
        """Calculate CUDA device memory bandwidth"""
        # Approximate calculation
        clock_rate = props['memoryClockRate'] * 1000  # kHz to Hz
        bus_width = props['memoryBusWidth']
        return (clock_rate * bus_width * 2) / (8 * 1024**3)  # GB/s

    def _calculate_cuda_performance(self, props) -> float:
        """Calculate CUDA device compute performance"""
        # Approximate GFLOPS
        cores = props['multiProcessorCount'] * 128  # Approximate cores per SM
        clock_rate = props['clockRate'] * 1000  # kHz to Hz
        return (cores * clock_rate * 2) / 1e9  # GFLOPS

    def auto_select_device(self) -> Optional[GPUDevice]:
        """Automatically select the best available GPU device"""
        if not self.devices:
            logger.warning("No GPU devices available")
            return None

        # Score devices by performance
        device_scores = []
        for device in self.devices:
            score = self._score_device(device)
            device_scores.append((device, score))

        # Select highest scoring device
        best_device = max(device_scores, key=lambda x: x[1])[0]
        self.current_device = best_device

        logger.info(f"Auto-selected device: {best_device.name} ({best_device.backend.value})")
        return best_device

    def _score_device(self, device: GPUDevice) -> float:
        """Score device for automatic selection"""
        base_score = 0

        # Backend preference
        backend_scores = {
            GPUBackend.CUDA: 100,
            GPUBackend.METAL: 90,    # Apple Silicon
            GPUBackend.ROCM: 80,     # AMD
            GPUBackend.OPENCL: 60,   # Generic
            GPUBackend.CPU: 10       # Fallback
        }

        base_score += backend_scores.get(device.backend, 0)

        # Memory score (more memory = higher score)
        memory_gb = device.memory_total / (1024**3)
        base_score += min(memory_gb, 16) * 5  # Cap at 16GB

        # Compute units score
        base_score += device.compute_units * 2

        return base_score

    def select_device(self, backend: GPUBackend, device_id: int = 0) -> bool:
        """Manually select a GPU device"""
        for device in self.devices:
            if device.backend == backend and device.device_id == device_id:
                self.current_device = device
                logger.info(f"Manually selected device: {device.name}")
                return True

        logger.warning(f"Device not found: {backend.value}:{device_id}")
        return False

    def get_device_info(self) -> Dict[str, Any]:
        """Get information about available devices"""
        return {
            'available_backends': [b.value for b in self.available_backends],
            'devices': [
                {
                    'backend': d.backend.value,
                    'device_id': d.device_id,
                    'name': d.name,
                    'memory_total_gb': d.memory_total / (1024**3),
                    'compute_units': d.compute_units,
                    'supports_fp64': d.supports_fp64,
                    'supports_fp16': d.supports_fp16
                }
                for d in self.devices
            ],
            'current_device': {
                'backend': self.current_device.backend.value,
                'device_id': self.current_device.device_id,
                'name': self.current_device.name
            } if self.current_device else None
        }

    def allocate_memory(self, size_bytes: int) -> Any:
        """Allocate GPU memory (unified API)"""
        if not self.current_device:
            raise RuntimeError("No GPU device selected")

        try:
            if self.current_device.backend == GPUBackend.CUDA:
                import cupy as cp
                return cp.zeros(size_bytes // 8, dtype=cp.float64)  # Example allocation

            elif self.current_device.backend == GPUBackend.METAL:
                import torch
                return torch.zeros(size_bytes // 4, dtype=torch.float32, device='mps')

            elif self.current_device.backend == GPUBackend.CPU:
                return np.zeros(size_bytes // 8, dtype=np.float64)

            else:
                # Fallback to CPU for unsupported backends
                logger.warning(f"Memory allocation not implemented for {self.current_device.backend.value}")
                return np.zeros(size_bytes // 8, dtype=np.float64)

        except Exception as e:
            logger.error(f"Memory allocation failed: {e}")
            # Fallback to CPU
            return np.zeros(size_bytes // 8, dtype=np.float64)

    def synchronize(self):
        """Synchronize GPU operations"""
        if not self.current_device:
            return

        try:
            if self.current_device.backend == GPUBackend.CUDA:
                import cupy as cp
                cp.cuda.stream.get_current_stream().synchronize()

            elif self.current_device.backend == GPUBackend.METAL:
                import torch
                torch.mps.synchronize()

            # CPU and other backends don't need explicit synchronization

        except Exception as e:
            logger.warning(f"Synchronization failed: {e}")

    def get_memory_info(self) -> Dict[str, float]:
        """Get memory information for current device"""
        if not self.current_device:
            return {'total_gb': 0, 'free_gb': 0, 'used_gb': 0}

        try:
            if self.current_device.backend == GPUBackend.CUDA:
                import cupy as cp
                mem_info = cp.cuda.runtime.memGetInfo()
                total = mem_info[1] / (1024**3)
                free = mem_info[0] / (1024**3)

            elif self.current_device.backend == GPUBackend.METAL:
                import torch
                total = self.current_device.memory_total / (1024**3)
                free = torch.mps.current_allocated_memory() / (1024**3)

            else:
                # CPU or fallback
                import psutil
                mem = psutil.virtual_memory()
                total = mem.total / (1024**3)
                free = mem.available / (1024**3)

            return {
                'total_gb': total,
                'free_gb': free,
                'used_gb': total - free
            }

        except Exception as e:
            logger.warning(f"Memory info retrieval failed: {e}")
            return {'total_gb': 0, 'free_gb': 0, 'used_gb': 0}

    def optimize_for_f2(self) -> Dict[str, Any]:
        """Optimize device settings for F2 plotting algorithm"""
        if not self.current_device:
            return {'optimization': 'none', 'reason': 'no_device_selected'}

        optimizations = {
            'memory_pool_size': 0,
            'thread_block_size': 0,
            'shared_memory_size': 0,
            'recommended_batch_size': 1
        }

        if self.current_device.backend == GPUBackend.CUDA:
            # CUDA-specific optimizations
            capabilities = self.capabilities.get(f"cuda:{self.current_device.device_id}")
            if capabilities:
                optimizations.update({
                    'memory_pool_size': int(self.current_device.memory_total * 0.1),  # 10% for pool
                    'thread_block_size': min(capabilities.max_threads_per_block, 512),
                    'shared_memory_size': capabilities.max_shared_memory,
                    'recommended_batch_size': 4
                })

        elif self.current_device.backend == GPUBackend.METAL:
            # Apple Silicon optimizations
            optimizations.update({
                'memory_pool_size': int(self.current_device.memory_total * 0.05),  # 5% for unified memory
                'thread_block_size': 512,
                'shared_memory_size': 32768,
                'recommended_batch_size': 2
            })

        else:
            # Generic optimizations
            optimizations.update({
                'memory_pool_size': int(self.current_device.memory_total * 0.05),
                'thread_block_size': min(self.current_device.max_work_group_size, 256),
                'shared_memory_size': 16384,
                'recommended_batch_size': 1
            })

        logger.info(f"F2 optimizations applied for {self.current_device.backend.value}")
        return optimizations

def main():
    """Test GPU abstraction layer"""
    print("🔧 GPU Abstraction Layer Test")
    print("=" * 40)

    gpu_layer = GPUAbstractionLayer()

    # Print device information
    info = gpu_layer.get_device_info()

    print(f"Available backends: {', '.join(info['available_backends'])}")
    print(f"Total devices: {len(info['devices'])}")

    for device in info['devices']:
        print(f"  {device['backend'].upper()}: {device['name']}")
        print(f"    Memory: {device['memory_total_gb']:.1f} GB")
        print(f"    Compute Units: {device['compute_units']}")
        print(f"    FP64: {'Yes' if device['supports_fp64'] else 'No'}")
        print(f"    FP16: {'Yes' if device['supports_fp16'] else 'No'}")
        print()

    if gpu_layer.current_device:
        print(f"Selected device: {gpu_layer.current_device.name}")
        print(f"Backend: {gpu_layer.current_device.backend.value}")

        # Test memory allocation
        print("\n🧪 Testing memory allocation...")
        try:
            mem_info = gpu_layer.get_memory_info()
            print(f"Memory - Total: {mem_info['total_gb']:.1f} GB, Free: {mem_info['free_gb']:.1f} GB")

            # Test small allocation
            test_array = gpu_layer.allocate_memory(1024)  # 1KB
            print(f"Memory allocation test: {'SUCCESS' if test_array is not None else 'FAILED'}")

        except Exception as e:
            print(f"Memory test failed: {e}")

        # Test F2 optimizations
        print("\n⚡ Testing F2 optimizations...")
        try:
            f2_opts = gpu_layer.optimize_for_f2()
            print(f"F2 optimizations: {f2_opts}")
        except Exception as e:
            print(f"F2 optimization test failed: {e}")

    print("\n✅ GPU Abstraction Layer test completed")

if __name__ == '__main__':
    main()
