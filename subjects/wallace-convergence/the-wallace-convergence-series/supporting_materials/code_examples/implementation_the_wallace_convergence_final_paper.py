#!/usr/bin/env python3
"""
Code examples for the_wallace_convergence_final_paper
Demonstrates key implementations and algorithms.
"""
# Set high precision
getcontext().prec = 50


import numpy as np
import math

# Golden ratio
phi = Decimal('1.618033988749894848204586834365638117720309179805762862135')

# Example 1: Wallace Transform
class WallaceTransform:
    """Wallace Transform implementation."""
    def __init__(self, alpha=1.0, beta=0.0):
        self.phi = phi
        self.alpha = alpha
        self.beta = beta
        self.epsilon = Decimal('1e-12')
    
    def transform(self, x):
        """Apply Wallace Transform."""
        if x <= 0:
            x = self.epsilon
        log_term = math.log(x + self.epsilon)
        phi_power = abs(log_term) ** self.phi
        sign_factor = 1 if log_term >= 0 else -1
        return self.alpha * phi_power * sign_factor + self.beta

# Example 2: Prime Topology
def prime_topology_traversal(primes):
    """Progressive path traversal on prime graph."""
    if len(primes) < 2:
        return []
    weights = [(primes[i+1] - primes[i]) / math.sqrt(2) 
              for i in range(len(primes) - 1)]
    scaled_weights = [w * (phi ** (-(i % 21))) 
                    for i, w in enumerate(weights)]
    return scaled_weights

# Example 3: Phase State Physics
def phase_state_speed(n, c_3=299792458):
    """Calculate speed of light in phase state n."""
    return c_3 * (phi ** (n - 3))

# Usage examples
if __name__ == '__main__':
    print("Wallace Transform Example:")
    wt = WallaceTransform()
    result = wt.transform(2.718)  # e
    print(f"  W_φ(e) = {result:.6f}")
    
    print("\nPrime Topology Example:")
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    weights = prime_topology_traversal(primes)
    print(f"  Generated {len(weights)} weights")
    
    print("\nPhase State Speed Example:")
    for n in [3, 7, 14, 21]:
        c_n = phase_state_speed(n)
        print(f"  c_{n} = {c_n:.2e} m/s")
