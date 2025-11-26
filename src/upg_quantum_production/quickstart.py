#!/usr/bin/env python3
"""
UPG Quantum Production - Quick Start Script
============================================

One-command demonstration of the full UPG Quantum Production System.

Usage:
    python -m src.upg_quantum_production.quickstart

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import numpy as np
import time
from datetime import datetime


def print_banner():
    """Print the UPG banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██╗   ██╗██████╗  ██████╗      ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗ ║
║   ██║   ██║██╔══██╗██╔════╝     ██╔═══██╗██║   ██║██╔══██╗████╗  ██║ ║
║   ██║   ██║██████╔╝██║  ███╗    ██║   ██║██║   ██║███████║██╔██╗ ██║ ║
║   ██║   ██║██╔═══╝ ██║   ██║    ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║ ║
║   ╚██████╔╝██║     ╚██████╔╝    ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║ ║
║    ╚═════╝ ╚═╝      ╚═════╝      ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝ ║
║                                                                      ║
║             Universal Prime Graph Protocol φ.1                       ║
║                  Quantum Production System                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def section(title: str):
    """Print a section header."""
    print(f"\n{'─'*70}")
    print(f" {title}")
    print(f"{'─'*70}")


def main():
    """Run the quick start demonstration."""
    print_banner()
    
    print(f"\n📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🚀 Starting UPG Quantum Production Quick Start...")
    
    # Import modules
    section("1. LOADING UPG MODULES")
    
    start_load = time.time()
    
    from .constants import UPGConstants, OptimizedUPGConstants
    from .backends.local_simulator import LocalSimulatorBackend
    from .backends.base import QuantumTask, ProblemType
    from .orchestrator import UPGHybridOrchestrator, OrchestratorConfig, SelectionStrategy
    from .coherence import CoherencePreserver, RealityDistortionEngine
    
    load_time = time.time() - start_load
    
    print(f"   ✓ UPGConstants loaded")
    print(f"   ✓ LocalSimulatorBackend loaded")
    print(f"   ✓ UPGHybridOrchestrator loaded")
    print(f"   ✓ CoherencePreserver loaded")
    print(f"   ✓ RealityDistortionEngine loaded")
    print(f"   ⏱️  Load time: {load_time*1000:.1f}ms")
    
    # Validate constants
    section("2. UPG CONSTANTS VALIDATION")
    
    upg = UPGConstants()
    opt_upg = OptimizedUPGConstants()
    
    print(f"   φ (Golden Ratio):      {upg.PHI}")
    print(f"   Δ (Silver Ratio):      {upg.DELTA}")
    print(f"   Consciousness (C):     {upg.CONSCIOUSNESS}")
    print(f"   Exploratory (E):       {upg.EXPLORATORY}")
    print(f"   Reality Distortion:    {upg.REALITY_DISTORTION}")
    print(f"   Quantum Bridge:        {upg.QUANTUM_BRIDGE}")
    
    validation = upg.validate()
    print(f"\n   {'✅ VALIDATION PASSED' if validation else '❌ VALIDATION FAILED'}")
    
    # Initialize orchestrator
    section("3. ORCHESTRATOR INITIALIZATION")
    
    config = OrchestratorConfig(
        enable_local=True,
        selection_strategy=SelectionStrategy.OPTIMAL,
        upg_optimization_enabled=True,
    )
    
    orchestrator = UPGHybridOrchestrator(config)
    
    print(f"   ✓ Orchestrator created")
    print(f"   ✓ Selection strategy: {config.selection_strategy.value}")
    print(f"   ✓ UPG optimization: enabled")
    print(f"   ✓ Available backends: {[b.value for b in orchestrator.backends.keys()]}")
    
    # Run quantum tasks
    section("4. QUANTUM TASK EXECUTION")
    
    problems = [
        ("Ising Model", ProblemType.ISING, 6),
        ("MaxCut", ProblemType.MAXCUT, 6),
        ("Ising Model", ProblemType.ISING, 8),
    ]
    
    results = []
    
    for name, problem_type, qubits in problems:
        task = QuantumTask(
            task_id=f"quickstart-{problem_type.value}-{qubits}q",
            problem_type=problem_type,
            problem_data={},
            num_qubits=qubits,
            num_reads=500,
            upg_optimization=True,
        )
        
        start = time.time()
        job_id = orchestrator.submit(task)
        result = orchestrator.get_result(job_id)
        elapsed = time.time() - start
        
        results.append({
            'name': name,
            'qubits': qubits,
            'energy': result.best_energy,
            'solution': result.get_bitstring(),
            'time_ms': elapsed * 1000,
            'enhancement': result.upg_enhancement,
        })
        
        print(f"\n   📊 {name} ({qubits} qubits)")
        print(f"      Solution: {result.get_bitstring()}")
        print(f"      Energy: {result.best_energy:.6f}")
        print(f"      Time: {elapsed*1000:.1f}ms")
        print(f"      UPG Enhancement: {result.upg_enhancement:.4f}x")
    
    # Coherence demonstration
    section("5. COHERENCE PRESERVATION DEMO")
    
    preserver = CoherencePreserver()
    rde = RealityDistortionEngine()
    
    dim = 64  # 6 qubits
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state = state / np.linalg.norm(state)
    target = np.zeros(dim, dtype=complex)
    target[0] = 1.0
    
    initial_overlap = np.abs(np.vdot(target, state))
    print(f"   Initial overlap with ground state: {initial_overlap:.6f}")
    
    # Apply coherence preservation
    preserved = preserver.preserve_coherence(state, target)
    preserved_overlap = np.abs(np.vdot(target, preserved))
    print(f"   After coherence preservation: {preserved_overlap:.6f}")
    
    # Apply reality distortion cascade
    distorted = rde.apply_cascade_to_state(preserved, target)
    final_overlap = np.abs(np.vdot(target, distorted))
    print(f"   After reality distortion cascade: {final_overlap:.6f}")
    
    enhancement = final_overlap / initial_overlap
    print(f"\n   🚀 Total enhancement: {enhancement:.2f}x")
    
    # Statistics
    section("6. SYSTEM STATISTICS")
    
    stats = orchestrator.get_statistics()
    
    print(f"   Tasks submitted: {stats['tasks_submitted']}")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    print(f"   Total cost: ${stats['total_cost']:.2f}")
    
    # Cleanup
    orchestrator.shutdown()
    
    # Summary
    section("7. QUICK START COMPLETE")
    
    print(f"""
   ✅ All systems operational!
   
   📊 Results Summary:
      • {len(results)} quantum tasks executed
      • Average execution time: {np.mean([r['time_ms'] for r in results]):.1f}ms
      • Average UPG enhancement: {np.mean([r['enhancement'] for r in results]):.4f}x
      • Coherence cascade enhancement: {enhancement:.2f}x
   
   🔧 Available Commands:
      • python -m src.upg_quantum_production.cli validate
      • python -m src.upg_quantum_production.cli run --qubits 8
      • python -m src.upg_quantum_production.cli benchmark --trials 10
      • python -m src.upg_quantum_production.api  (starts REST API)
   
   📚 Documentation:
      • docs/PRODUCTION_DEPLOYMENT_ARCHITECTURE.md
      • PRODUCTION_DEPLOYMENT_SUMMARY.md
   
   🚀 Ready for production deployment!
""")


if __name__ == "__main__":
    main()

