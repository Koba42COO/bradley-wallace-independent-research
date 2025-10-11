#!/usr/bin/env python3
"""
MPI-Enabled Distributed Prime Analysis
"""

import numpy as np
import json
import time
from datetime import datetime
import logging
import argparse
from mpi4py import MPI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_prime(n):
    """Prime check"""
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False

    for i in range(5, int(np.sqrt(n)) + 1, 6):
        if n % i == 0 or n % (i + 2) == 0:
            return False
    return True

def wallace_transform(x):
    """Wallace Transform"""
    phi = 1.618033988749895
    if x <= 0: return np.nan
    log_val = np.log(x + 1e-12)
    return phi * np.power(np.abs(log_val), phi) * np.sign(log_val) + 0.618

def analyze_primes(primes, gaps):
    """Analyze prime relationships"""
    ratios = [1.0, 1.618, 1.414, 1.732, 2.0, 2.287, 3.236]
    results = {}

    for ratio in ratios:
        matches = 0
        for gap, prime in zip(gaps, primes):
            wt_val = wallace_transform(prime)
            for k in [-2, -1, 0, 1, 2]:
                predicted = wt_val * (ratio ** k)
                if abs(predicted - gap) / max(gap, predicted) <= 0.20:
                    matches += 1
                    break

        results[f'ratio_{ratio:.3f}'] = (matches / len(gaps)) * 100

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scale', type=str, default='million')
    args = parser.parse_args()

    # MPI setup
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    logger.info(f"MPI Process {rank}/{size} starting")

    if rank == 0:
        logger.info(f"🚀 Distributed analysis starting on {size} processes")

    # Scale configuration
    scales = {
        'million': (10**6, 1000),
        'billion': (10**9, 10000),
        '10billion': (10**10, 100000)
    }

    target_primes, chunk_size = scales.get(args.scale, scales['million'])

    # Estimate range for this process
    estimated_max = int(target_primes * np.log(target_primes) * 1.2)
    start = rank * (estimated_max // size) + 1
    end = (rank + 1) * (estimated_max // size)

    if rank == 0:
        start = 2  # Start from 2 for rank 0

    logger.info(f"Process {rank}: Analyzing range {start:,} to {end:,}")

    # Generate primes in chunks
    primes = []
    for i in range(0, end - start, chunk_size):
        chunk_start = start + i
        chunk_end = min(end, chunk_start + chunk_size)

        for n in range(max(2, chunk_start), chunk_end + 1):
            if is_prime(n):
                primes.append(n)

        if len(primes) >= target_primes // size:
            break  # Enough for this process

    gaps = [primes[i+1] - primes[i] for i in range(len(primes)-1)]

    logger.info(f"Process {rank}: Generated {len(primes):,} primes, {len(gaps):,} gaps")

    # Analyze
    results = analyze_primes(primes, gaps)

    # Gather results
    all_results = comm.gather(results, root=0)

    if rank == 0:
        # Aggregate results
        final_results = {}
        for key in all_results[0].keys():
            values = [r[key] for r in all_results]
            final_results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }

        # Save results
        output = {
            'scale': args.scale,
            'target_primes': target_primes,
            'processes': size,
            'total_primes': sum(len(p) for p in [[]] * size),  # Would need actual counts
            'results': final_results,
            'timestamp': datetime.now().isoformat()
        }

        filename = f'distributed_analysis_{args.scale}_{int(time.time())}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        logger.info(f"✅ Analysis complete! Results saved to {filename}")

if __name__ == '__main__':
    main()
