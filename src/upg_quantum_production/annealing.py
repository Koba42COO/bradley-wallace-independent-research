"""
Optimized Quantum Annealing Module
==================================

Production-ready quantum annealing with full UPG consciousness
mathematics optimization. Re-exports the optimized annealer.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

# Re-export from the main implementation
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from quantum_annealing_optimized_upg import OptimizedQuantumAnnealer, OptimizedBenchmarkSuite

__all__ = ['OptimizedQuantumAnnealer', 'OptimizedBenchmarkSuite']

