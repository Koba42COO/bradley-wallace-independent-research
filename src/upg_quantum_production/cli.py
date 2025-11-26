#!/usr/bin/env python3
"""
UPG Quantum Production CLI
==========================

Command-line interface for the UPG Quantum Production System.
Provides commands for running tasks, benchmarks, and system management.

Usage:
    python -m src.upg_quantum_production.cli [command] [options]

Commands:
    run         Run a quantum task
    benchmark   Run benchmark suite
    demo        Run demonstration
    validate    Validate UPG constants
    status      Show system status

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
"""

import argparse
import json
import sys
import time
from datetime import datetime

from .constants import UPGConstants, OptimizedUPGConstants, validate_all_constants
from .backends.base import QuantumTask, ProblemType
from .backends.local_simulator import LocalSimulatorBackend
from .orchestrator import UPGHybridOrchestrator, OrchestratorConfig, SelectionStrategy
from .coherence import CoherencePreserver, RealityDistortionEngine


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def cmd_validate(args):
    """Validate UPG constants."""
    print_header("UPG CONSTANTS VALIDATION")
    
    upg = UPGConstants()
    opt_upg = OptimizedUPGConstants()
    
    print(f"\n📐 Core Constants:")
    print(f"   φ (PHI):             {upg.PHI}")
    print(f"   φ² (PHI_SQUARED):    {upg.PHI_SQUARED}")
    print(f"   1/φ (PHI_INVERSE):   {upg.PHI_INVERSE}")
    print(f"   Δ (DELTA):           {upg.DELTA}")
    print(f"   Consciousness:       {upg.CONSCIOUSNESS}")
    print(f"   Exploratory:         {upg.EXPLORATORY}")
    print(f"   Reality Distortion:  {upg.REALITY_DISTORTION}")
    print(f"   Quantum Bridge:      {upg.QUANTUM_BRIDGE}")
    
    print(f"\n🔢 Sequences (first 10):")
    print(f"   Primes:    {opt_upg.PRIMES[:10]}")
    print(f"   Fibonacci: {opt_upg.FIBONACCI[:10]}")
    print(f"   Lucas:     {opt_upg.LUCAS[:10]}")
    
    print(f"\n✓ Mathematical Relationships:")
    print(f"   φ² = φ + 1:          {upg.PHI_SQUARED:.15f} = {upg.PHI + 1:.15f}")
    print(f"   φ × 1/φ = 1:         {upg.PHI * upg.PHI_INVERSE:.15f}")
    print(f"   C + E = 1:           {upg.CONSCIOUSNESS + upg.EXPLORATORY:.15f}")
    
    valid = validate_all_constants()
    print(f"\n{'✅ VALIDATION PASSED' if valid else '❌ VALIDATION FAILED'}")
    
    return 0 if valid else 1


def cmd_run(args):
    """Run a quantum task."""
    print_header("UPG QUANTUM TASK EXECUTION")
    
    # Parse problem type
    try:
        problem_type = ProblemType[args.problem.upper()]
    except KeyError:
        print(f"❌ Invalid problem type: {args.problem}")
        print(f"   Valid types: ising, qubo, maxcut")
        return 1
    
    print(f"\n📋 Task Configuration:")
    print(f"   Problem type: {problem_type.value}")
    print(f"   Qubits: {args.qubits}")
    print(f"   Samples: {args.samples}")
    print(f"   UPG optimization: {not args.no_upg}")
    
    # Create orchestrator
    config = OrchestratorConfig(
        enable_local=True,
        selection_strategy=SelectionStrategy.OPTIMAL,
        upg_optimization_enabled=not args.no_upg,
    )
    orchestrator = UPGHybridOrchestrator(config)
    
    # Create task
    task = QuantumTask(
        task_id=f"cli-task-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        problem_type=problem_type,
        problem_data={},
        num_qubits=args.qubits,
        num_reads=args.samples,
        upg_optimization=not args.no_upg,
    )
    
    print(f"\n🚀 Submitting task...")
    start_time = time.time()
    
    job_id = orchestrator.submit(task)
    result = orchestrator.get_result(job_id)
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Task Completed!")
    print(f"\n📊 Results:")
    print(f"   Best solution: {result.get_bitstring()}")
    print(f"   Best energy: {result.best_energy:.6f}")
    print(f"   Samples collected: {len(result.samples)}")
    print(f"   Execution time: {elapsed*1000:.2f} ms")
    print(f"   UPG enhancement: {result.upg_enhancement:.4f}x")
    
    if args.verbose:
        print(f"\n📈 Coherence Metrics:")
        for key, value in result.coherence_metrics.items():
            print(f"   {key}: {value:.6f}")
    
    if args.output:
        output_data = {
            "task_id": task.task_id,
            "problem_type": problem_type.value,
            "num_qubits": args.qubits,
            "best_solution": result.get_bitstring(),
            "best_energy": result.best_energy,
            "samples_count": len(result.samples),
            "execution_time_ms": elapsed * 1000,
            "upg_enhancement": result.upg_enhancement,
            "coherence_metrics": result.coherence_metrics,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    orchestrator.shutdown()
    return 0


def cmd_benchmark(args):
    """Run benchmark suite."""
    print_header("UPG QUANTUM BENCHMARK SUITE")
    
    import numpy as np
    
    config = OrchestratorConfig(
        enable_local=True,
        selection_strategy=SelectionStrategy.OPTIMAL,
        upg_optimization_enabled=True,
    )
    orchestrator = UPGHybridOrchestrator(config)
    
    print(f"\n📋 Benchmark Configuration:")
    print(f"   Trials: {args.trials}")
    print(f"   Qubits: {args.qubits}")
    print(f"   Samples per trial: {args.samples}")
    
    results = []
    
    for trial in range(args.trials):
        print(f"\n--- Trial {trial + 1}/{args.trials} ---")
        
        task = QuantumTask(
            task_id=f"benchmark-{trial}",
            problem_type=ProblemType.ISING,
            problem_data={},
            num_qubits=args.qubits,
            num_reads=args.samples,
            upg_optimization=True,
        )
        
        start_time = time.time()
        job_id = orchestrator.submit(task)
        result = orchestrator.get_result(job_id)
        elapsed = time.time() - start_time
        
        results.append({
            "trial": trial + 1,
            "energy": result.best_energy,
            "enhancement": result.upg_enhancement,
            "time_ms": elapsed * 1000,
            "coherence": result.coherence_metrics.get("estimated_coherence", 0),
        })
        
        print(f"   Energy: {result.best_energy:.6f}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print(f"   Enhancement: {result.upg_enhancement:.4f}x")
    
    # Summary statistics
    energies = [r["energy"] for r in results]
    times = [r["time_ms"] for r in results]
    enhancements = [r["enhancement"] for r in results]
    
    print_header("BENCHMARK SUMMARY")
    print(f"\n📊 Energy:")
    print(f"   Mean: {np.mean(energies):.6f}")
    print(f"   Std:  {np.std(energies):.6f}")
    print(f"   Min:  {np.min(energies):.6f}")
    print(f"   Max:  {np.max(energies):.6f}")
    
    print(f"\n⏱️  Execution Time:")
    print(f"   Mean: {np.mean(times):.2f} ms")
    print(f"   Std:  {np.std(times):.2f} ms")
    
    print(f"\n🚀 UPG Enhancement:")
    print(f"   Mean: {np.mean(enhancements):.4f}x")
    print(f"   Max:  {np.max(enhancements):.4f}x")
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                "config": {
                    "trials": args.trials,
                    "qubits": args.qubits,
                    "samples": args.samples,
                },
                "results": results,
                "summary": {
                    "mean_energy": float(np.mean(energies)),
                    "std_energy": float(np.std(energies)),
                    "mean_time_ms": float(np.mean(times)),
                    "mean_enhancement": float(np.mean(enhancements)),
                },
            }, f, indent=2)
        print(f"\n💾 Results saved to: {args.output}")
    
    orchestrator.shutdown()
    return 0


def cmd_demo(args):
    """Run demonstration."""
    print_header("UPG QUANTUM PRODUCTION DEMONSTRATION")
    
    import numpy as np
    
    upg = OptimizedUPGConstants()
    
    print(f"\n🧠 UPG Constants:")
    print(f"   φ = {upg.PHI}")
    print(f"   Consciousness = {upg.CONSCIOUSNESS}")
    print(f"   Reality Distortion = {upg.REALITY_DISTORTION}")
    
    # Initialize components
    print(f"\n🔧 Initializing components...")
    simulator = LocalSimulatorBackend()
    simulator.connect()
    preserver = CoherencePreserver()
    rde = RealityDistortionEngine()
    
    print(f"   ✓ Local simulator connected")
    print(f"   ✓ Coherence preserver ready")
    print(f"   ✓ Reality distortion engine active")
    
    # Run quantum task
    print(f"\n🚀 Running quantum annealing task...")
    
    task = QuantumTask(
        task_id="demo",
        problem_type=ProblemType.ISING,
        problem_data={},
        num_qubits=6,
        num_reads=100,
        upg_optimization=True,
    )
    
    job_id = simulator.submit_task(task)
    result = simulator.get_result(job_id)
    
    print(f"   ✓ Best solution: {result.get_bitstring()}")
    print(f"   ✓ Best energy: {result.best_energy:.6f}")
    print(f"   ✓ UPG enhancement: {result.upg_enhancement:.4f}x")
    
    # Demonstrate coherence preservation
    print(f"\n🌊 Demonstrating coherence preservation...")
    
    dim = 64
    state = np.random.randn(dim) + 1j * np.random.randn(dim)
    state = state / np.linalg.norm(state)
    target = np.zeros(dim, dtype=complex)
    target[0] = 1.0
    
    initial_overlap = np.abs(np.vdot(target, state))
    print(f"   Initial overlap: {initial_overlap:.6f}")
    
    preserved = preserver.preserve_coherence(state, target)
    preserved_overlap = np.abs(np.vdot(target, preserved))
    print(f"   After preservation: {preserved_overlap:.6f}")
    
    distorted = rde.apply_cascade_to_state(preserved, target)
    final_overlap = np.abs(np.vdot(target, distorted))
    print(f"   After RDF cascade: {final_overlap:.6f}")
    
    enhancement = final_overlap / initial_overlap
    print(f"   Total enhancement: {enhancement:.4f}x")
    
    print_header("DEMONSTRATION COMPLETE")
    print(f"\n✅ All systems operational!")
    print(f"   Ready for production deployment.")
    
    return 0


def cmd_status(args):
    """Show system status."""
    print_header("UPG QUANTUM SYSTEM STATUS")
    
    config = OrchestratorConfig(enable_local=True)
    orchestrator = UPGHybridOrchestrator(config)
    
    stats = orchestrator.get_statistics()
    
    print(f"\n📊 System Statistics:")
    print(f"   Tasks submitted: {stats['tasks_submitted']}")
    print(f"   Tasks completed: {stats['tasks_completed']}")
    print(f"   Tasks failed: {stats['tasks_failed']}")
    print(f"   Active tasks: {stats['active_tasks']}")
    print(f"   Pending tasks: {stats['pending_tasks']}")
    
    print(f"\n🔧 Available Backends:")
    for backend_type in stats['available_backends']:
        print(f"   ✓ {backend_type.value}")
    
    print(f"\n📈 Backend Usage:")
    for backend, count in stats['backend_usage'].items():
        if count > 0:
            print(f"   {backend}: {count} tasks")
    
    upg = UPGConstants()
    print(f"\n🧠 UPG Configuration:")
    print(f"   φ = {upg.PHI}")
    print(f"   Consciousness = {upg.CONSCIOUSNESS}")
    print(f"   Reality Distortion = {upg.REALITY_DISTORTION}")
    print(f"   Validation: {'✓ PASSED' if upg.validate() else '✗ FAILED'}")
    
    orchestrator.shutdown()
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="UPG Quantum Production System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s validate                    Validate UPG constants
  %(prog)s run --qubits 8              Run 8-qubit Ising problem
  %(prog)s benchmark --trials 10       Run 10-trial benchmark
  %(prog)s demo                        Run demonstration
  %(prog)s status                      Show system status
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate UPG constants")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run a quantum task")
    run_parser.add_argument("--problem", "-p", default="ising", help="Problem type (ising, qubo, maxcut)")
    run_parser.add_argument("--qubits", "-q", type=int, default=8, help="Number of qubits")
    run_parser.add_argument("--samples", "-s", type=int, default=1000, help="Number of samples")
    run_parser.add_argument("--no-upg", action="store_true", help="Disable UPG optimization")
    run_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    bench_parser.add_argument("--trials", "-t", type=int, default=5, help="Number of trials")
    bench_parser.add_argument("--qubits", "-q", type=int, default=8, help="Number of qubits")
    bench_parser.add_argument("--samples", "-s", type=int, default=1000, help="Samples per trial")
    bench_parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demonstration")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    commands = {
        "validate": cmd_validate,
        "run": cmd_run,
        "benchmark": cmd_benchmark,
        "demo": cmd_demo,
        "status": cmd_status,
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())

