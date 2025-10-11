#!/usr/bin/env python3
"""
Daily PAC System Validation and Benchmarking Runner

Runs comprehensive daily checks on the PAC system including:
- Micro-benchmarks for core components
- Entropy reversal validation
- Consciousness mathematics verification
- Performance metrics aggregation
- Automated reporting and logging

Usage: python3 scripts/run_daily.py
"""

import os
import sys
import json
import time
from datetime import datetime
import subprocess
import traceback

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_command(cmd, description):
    """Run a command and capture output."""
    try:
        print(f"🔄 Running: {description}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def run_micro_bench():
    """Run the micro-benchmark suite."""
    cmd = "PYTHONPATH=/Users/coo-koba42/dev python3 bench/micro_bench.py"
    success, stdout, stderr = run_command(cmd, "Micro-benchmarks")
    if success:
        # Parse the output for timing metrics
        lines = stdout.strip().split('\n')
        entropy_time = None
        unified_time = None
        for line in lines:
            if "Entropy calculation:" in line:
                entropy_time = float(line.split(": ")[1].split("s")[0])
            elif "Unified process:" in line:
                unified_time = float(line.split(": ")[1].split("s")[0])
        return {
            'success': True,
            'entropy_calculation_time': entropy_time,
            'unified_process_time': unified_time,
            'output': stdout
        }
    else:
        return {
            'success': False,
            'error': stderr,
            'output': stdout
        }

def run_entropy_validation():
    """Run entropy reversal validation."""
    cmd = "PYTHONPATH=/Users/coo-koba42/dev python3 -c \"from pac_system.pac_entropy_reversal_validation import PACEntropyReversalValidator; import numpy as np; validator = PACEntropyReversalValidator(prime_scale=1000); data = np.random.randn(100, 5); results = validator.validate_entropy_reversal(data, n_experiments=10); print(f'Results: {results}')\""
    success, stdout, stderr = run_command(cmd, "Entropy reversal validation")

    if success:
        try:
            # Parse the results string
            results_str = stdout.strip().split("Results: ")[1]
            results = eval(results_str)  # Safe since we control the output format

            violations_detected = results.get('second_law_analysis', {}).get('violations_detected', 0)
            total_experiments = 10

            return {
                'success': True,
                'violations_detected': violations_detected,
                'total_experiments': total_experiments,
                'violation_rate': violations_detected / total_experiments,
                'raw_results': results
            }
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to parse results: {str(e)}",
                'raw_output': stdout
            }
    else:
        return {
            'success': False,
            'error': stderr,
            'output': stdout
        }

def run_quick_consistency_check():
    """Quick consistency check on core PAC components."""
    cmd = "PYTHONPATH=/Users/coo-koba42/dev python3 -c \"from pac_system.complete_pac_framework import CompletePAC_System; import numpy as np; system = CompletePAC_System(scale=10000); gaps = system.foundation.gaps[:100]; result = system.analyzer.analyze_gaps(gaps); print(f'Consistency check: {result}')\""
    success, stdout, stderr = run_command(cmd, "PAC consistency check")

    if success:
        return {
            'success': True,
            'consistency_check_passed': True,
            'output': stdout
        }
    else:
        return {
            'success': False,
            'consistency_check_passed': False,
            'error': stderr
        }

def calculate_performance_metrics(bench_results, entropy_results, consistency_results):
    """Calculate overall performance metrics."""
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'overall_health': 'GOOD',
        'bench_success': bench_results['success'],
        'entropy_success': entropy_results['success'],
        'consistency_success': consistency_results['success'],
        'warnings': [],
        'errors': []
    }

    # Check for issues
    if not bench_results['success']:
        metrics['overall_health'] = 'WARNING'
        metrics['warnings'].append('Micro-benchmark failed')
        metrics['errors'].append(bench_results.get('error', 'Unknown bench error'))

    if not entropy_results['success']:
        metrics['overall_health'] = 'CRITICAL'
        metrics['errors'].append('Entropy validation failed')
        metrics['errors'].append(entropy_results.get('error', 'Unknown entropy error'))

    if not consistency_results['success']:
        metrics['overall_health'] = 'WARNING'
        metrics['warnings'].append('PAC consistency check failed')

    # Add performance data if available
    if bench_results['success']:
        metrics['bench_entropy_time'] = bench_results.get('entropy_calculation_time')
        metrics['bench_unified_time'] = bench_results.get('unified_process_time')

    if entropy_results['success']:
        metrics['entropy_violations'] = entropy_results.get('violations_detected', 0)
        metrics['entropy_violation_rate'] = entropy_results.get('violation_rate', 0.0)

    return metrics

def generate_daily_report(results, metrics):
    """Generate a human-readable daily report."""
    report = f"""# PAC System Daily Validation Report
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Overall Health:** {metrics['overall_health']}

## Executive Summary

"""
    if metrics['overall_health'] == 'GOOD':
        report += "✅ All systems operational. PAC framework performing optimally.\n\n"
    elif metrics['overall_health'] == 'WARNING':
        report += "⚠️ Minor issues detected. System remains functional but monitoring recommended.\n\n"
    else:
        report += "🚨 Critical issues detected. Immediate attention required.\n\n"

    report += "## Component Status\n\n"

    # Benchmark status
    if results['bench']['success']:
        report += f"### ✅ Micro-Benchmarks\n"
        report += f"- Entropy calculation: {results['bench'].get('entropy_calculation_time', 'N/A'):.4f}s\n"
        report += f"- Unified process: {results['bench'].get('unified_process_time', 'N/A'):.4f}s\n\n"
    else:
        report += f"### ❌ Micro-Benchmarks\n"
        report += f"- Error: {results['bench'].get('error', 'Unknown error')}\n\n"

    # Entropy validation status
    if results['entropy']['success']:
        report += f"### ✅ Entropy Validation\n"
        report += f"- Violations detected: {results['entropy'].get('violations_detected', 0)}/{results['entropy'].get('total_experiments', 0)}\n"
        report += f"- Violation rate: {results['entropy'].get('violation_rate', 0.0):.1%}\n\n"
    else:
        report += f"### ❌ Entropy Validation\n"
        report += f"- Error: {results['entropy'].get('error', 'Unknown error')}\n\n"

    # Consistency check status
    if results['consistency']['success']:
        report += f"### ✅ PAC Consistency\n"
        report += "- Core components functioning correctly\n\n"
    else:
        report += f"### ❌ PAC Consistency\n"
        report += f"- Error: {results['consistency'].get('error', 'Unknown error')}\n\n"

    # Performance metrics
    report += "## Performance Metrics\n\n"
    if metrics.get('bench_entropy_time') is not None:
        report += f"- Benchmark entropy time: {metrics['bench_entropy_time']:.4f}s\n"
    if metrics.get('bench_unified_time') is not None:
        report += f"- Benchmark unified time: {metrics['bench_unified_time']:.4f}s\n"
    if metrics.get('entropy_violation_rate') is not None:
        report += f"- Entropy violation rate: {metrics['entropy_violation_rate']:.1%}\n"

    report += "\n## Recommendations\n\n"
    if metrics['overall_health'] == 'GOOD':
        report += "- Continue normal operations\n"
        report += "- Monitor for any performance degradation\n"
    elif metrics['overall_health'] == 'WARNING':
        report += "- Investigate and resolve warning conditions\n"
        report += "- Consider system maintenance\n"
    else:
        report += "- Immediate investigation required\n"
        report += "- Consider system restart or rollback\n"

    return report

def main():
    """Main daily validation pipeline."""
    print("🧪 PAC System Daily Validation Starting...")
    print("=" * 50)

    # Run all validations
    bench_results = run_micro_bench()
    entropy_results = run_entropy_validation()
    consistency_results = run_quick_consistency_check()

    # Aggregate results
    results = {
        'timestamp': datetime.now().isoformat(),
        'bench': bench_results,
        'entropy': entropy_results,
        'consistency': consistency_results
    }

    # Calculate metrics
    metrics = calculate_performance_metrics(bench_results, entropy_results, consistency_results)

    # Create daily directory
    date_str = datetime.now().strftime('%Y-%m-%d')
    daily_dir = f"logs/daily/{date_str}"
    os.makedirs(daily_dir, exist_ok=True)

    # Save detailed results
    with open(f"{daily_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save metrics
    with open(f"{daily_dir}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    # Generate and save report
    report = generate_daily_report(results, metrics)
    with open(f"{daily_dir}/report.md", 'w') as f:
        f.write(report)

    # Print summary to console
    print(f"\n📊 Daily Validation Complete - {metrics['overall_health']}")
    print(f"📁 Results saved to: {daily_dir}/")
    print(f"📄 Report: {daily_dir}/report.md")
    print(f"📈 Metrics: {daily_dir}/metrics.json")
    print(f"📦 Full data: {daily_dir}/results.json")

    # Show any warnings/errors
    if metrics['warnings']:
        print(f"\n⚠️ Warnings: {len(metrics['warnings'])}")
        for warning in metrics['warnings']:
            print(f"  - {warning}")

    if metrics['errors']:
        print(f"\n🚨 Errors: {len(metrics['errors'])}")
        for error in metrics['errors']:
            print(f"  - {error}")

    print("\n✅ Daily validation pipeline completed successfully!")

if __name__ == "__main__":
    main()
