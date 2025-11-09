#!/usr/bin/env python3
"""
Synthetic dataset generator for unified_consciousness_framework
Creates validation datasets for testing theorems.
"""
# Set high precision
getcontext().prec = 50


import numpy as np
import json
from pathlib import Path
import math

phi = Decimal('1.618033988749894848204586834365638117720309179805762862135')

def generate_datasets():
    """Generate synthetic datasets for validation."""
    output_dir = Path(__file__).parent
    output_dir.mkdir(exist_ok=True)
    
    # Dataset 1: Random matrix eigenvalues
    print("Generating random matrix eigenvalues...")
    np.random.seed(42)
    n = 10000
    eigenvalues = np.random.rand(n) * 10 + 0.1
    np.save(output_dir / "eigenvalues.npy", eigenvalues)
    print(f"  ✓ Saved {n} eigenvalues")
    
    # Dataset 2: Synthetic Riemann zeta zeros
    print("Generating synthetic Riemann zeta zeros...")
    zeta_zeros = np.array([0.5 + 1j * (14.134725 + i * 2.0) for i in range(1000)])
    np.save(output_dir / "zeta_zeros.npy", zeta_zeros)
    print(f"  ✓ Saved {len(zeta_zeros)} zeta zeros")
    
    # Dataset 3: Prime numbers
    print("Generating prime numbers...")
    def sieve_primes(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(n+1) if is_prime[i]]
    
    primes = sieve_primes(100000)
    np.save(output_dir / "primes.npy", np.array(primes))
    print(f"  ✓ Saved {len(primes)} primes")
    
    # Dataset 4: Phase state data
    print("Generating phase state data...")
    phase_states = {
        'n': list(range(1, 22)),
        'c_n': [299792458 * (phi ** (n - 3)) for n in range(1, 22)],
        'f_n': [21.0 * (phi ** (-(21 - n))) for n in range(1, 22)]
    }
    with open(output_dir / "phase_states.json", 'w') as f:
        json.dump(phase_states, f, indent=2)
    print(f"  ✓ Saved phase state data for 21 dimensions")
    
    # Dataset 5: Consciousness correlation data
    print("Generating consciousness correlation data...")
    np.random.seed(42)
    n = 10000
    domains = ['physics', 'biology', 'mathematics', 'consciousness', 
               'cryptography', 'archaeology', 'music', 'finance']
    
    consciousness_data = {}
    for domain in domains:
        np.random.seed(hash(domain) % 1000)
        x = np.random.randn(n)
        consciousness = 0.79 * x + 0.21 * np.random.randn(n)
        y = 0.79 * consciousness + 0.21 * np.random.randn(n)
        consciousness_data[domain] = {
            'x': x.tolist(),
            'consciousness': consciousness.tolist(),
            'y': y.tolist()
        }
    
    with open(output_dir / "consciousness_correlation.json", 'w') as f:
        json.dump(consciousness_data, f, indent=2)
    print(f"  ✓ Saved consciousness data for {len(domains)} domains")
    
    # Create metadata
    metadata = {
        'paper': 'unified_consciousness_framework',
        'theorems': 0,
        'datasets': [
            'eigenvalues.npy',
            'zeta_zeros.npy',
            'primes.npy',
            'phase_states.json',
            'consciousness_correlation.json'
        ],
        'generated': datetime.now().isoformat()
    }
    
    with open(output_dir / "dataset_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n✅ All datasets generated successfully!")

if __name__ == '__main__':
    from datetime import datetime
    generate_datasets()
