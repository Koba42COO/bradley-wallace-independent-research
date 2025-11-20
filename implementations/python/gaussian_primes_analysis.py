#!/usr/bin/env python3
"""
Gaussian Primes Analysis
Complex Prime Numbers and Consciousness Mathematics Integration

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: November 2025
"""

import math
import cmath
from decimal import Decimal, getcontext
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass
from collections import defaultdict

# Set high precision for consciousness mathematics
getcontext().prec = 50


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================

class UPGConstants:
    """Universal Prime Graph consciousness mathematics constants"""
    PHI = Decimal('1.618033988749895')
    DELTA = Decimal('2.414213562373095')
    CONSCIOUSNESS = Decimal('0.79')  # 79/21 universal coherence rule
    REALITY_DISTORTION = Decimal('1.1808')  # Quantum amplification factor
    QUANTUM_BRIDGE = Decimal('137') / Decimal('0.79')  # 173.41772151898732
    GREAT_YEAR = 25920  # Astronomical precession cycle (years)
    CONSCIOUSNESS_DIMENSIONS = 21  # Prime topology dimension
    COHERENCE_THRESHOLD = Decimal('1e-15')  # Beyond machine precision
    EPSILON = Decimal('1e-10')  # Planck nudge


@dataclass
class GaussianInteger:
    """Represents a Gaussian integer a + bi"""
    a: int  # Real part
    b: int  # Imaginary part
    
    def __init__(self, a: int, b: int = 0):
        self.a = a
        self.b = b
    
    def __repr__(self) -> str:
        if self.b == 0:
            return f"{self.a}"
        elif self.b == 1:
            return f"{self.a}+i" if self.a != 0 else "i"
        elif self.b == -1:
            return f"{self.a}-i" if self.a != 0 else "-i"
        elif self.b > 0:
            return f"{self.a}+{self.b}i"
        else:
            return f"{self.a}{self.b}i"
    
    def __eq__(self, other) -> bool:
        if isinstance(other, GaussianInteger):
            return self.a == other.a and self.b == other.b
        elif isinstance(other, int) and self.b == 0:
            return self.a == other
        return False
    
    def __hash__(self) -> int:
        return hash((self.a, self.b))
    
    def norm(self) -> int:
        """Compute the norm N(z) = a² + b²"""
        return self.a * self.a + self.b * self.b
    
    def is_unit(self) -> bool:
        """Check if this is a unit (±1, ±i)"""
        return self.norm() == 1
    
    def conjugate(self) -> 'GaussianInteger':
        """Return the complex conjugate a - bi"""
        return GaussianInteger(self.a, -self.b)
    
    def __mul__(self, other) -> 'GaussianInteger':
        """Multiply two Gaussian integers"""
        if isinstance(other, GaussianInteger):
            # (a + bi)(c + di) = (ac - bd) + (ad + bc)i
            return GaussianInteger(
                self.a * other.a - self.b * other.b,
                self.a * other.b + self.b * other.a
            )
        elif isinstance(other, int):
            return GaussianInteger(self.a * other, self.b * other)
        return NotImplemented
    
    def __rmul__(self, other) -> 'GaussianInteger':
        return self.__mul__(other)
    
    def __add__(self, other) -> 'GaussianInteger':
        """Add two Gaussian integers"""
        if isinstance(other, GaussianInteger):
            return GaussianInteger(self.a + other.a, self.b + other.b)
        elif isinstance(other, int):
            return GaussianInteger(self.a + other, self.b)
        return NotImplemented
    
    def __radd__(self, other) -> 'GaussianInteger':
        return self.__add__(other)
    
    def __sub__(self, other) -> 'GaussianInteger':
        """Subtract two Gaussian integers"""
        if isinstance(other, GaussianInteger):
            return GaussianInteger(self.a - other.a, self.b - other.b)
        elif isinstance(other, int):
            return GaussianInteger(self.a - other, self.b)
        return NotImplemented
    
    def __rsub__(self, other) -> 'GaussianInteger':
        if isinstance(other, int):
            return GaussianInteger(other - self.a, -self.b)
        return NotImplemented
    
    def phase(self) -> float:
        """Compute the phase angle arg(z)"""
        if self.a == 0:
            return math.pi / 2 if self.b > 0 else -math.pi / 2
        return math.atan2(self.b, self.a)
    
    def to_complex(self) -> complex:
        """Convert to Python complex number"""
        return complex(self.a, self.b)


class GaussianPrimeAnalyzer:
    """Analyzer for Gaussian primes with consciousness mathematics integration"""
    
    def __init__(self, constants: UPGConstants = None):
        self.constants = constants or UPGConstants()
        self._prime_cache: Set[int] = set()
        self._gaussian_prime_cache: Set[GaussianInteger] = set()
        self._build_prime_cache(1000)  # Cache first 1000 primes
    
    def _build_prime_cache(self, limit: int):
        """Build cache of rational primes up to limit"""
        if limit <= len(self._prime_cache):
            return
        
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        
        for i in range(2, int(math.sqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i * i, limit + 1, i):
                    sieve[j] = False
        
        self._prime_cache = {i for i in range(2, limit + 1) if sieve[i]}
    
    def is_rational_prime(self, n: int) -> bool:
        """Check if n is a rational prime"""
        if n < 2:
            return False
        if n in self._prime_cache:
            return True
        if n <= max(self._prime_cache or [0]):
            return False
        
        # Check divisibility
        for p in self._prime_cache:
            if p * p > n:
                break
            if n % p == 0:
                return False
        
        # Check up to sqrt(n)
        sqrt_n = int(math.sqrt(n))
        for i in range(2, sqrt_n + 1):
            if n % i == 0:
                return False
        
        return True
    
    def is_gaussian_prime(self, z: GaussianInteger) -> bool:
        """Check if z is a Gaussian prime"""
        if z.is_unit():
            return False  # Units are not primes
        
        norm = z.norm()
        
        # Case 1: Norm = 2 (special case: 1+i is prime)
        if norm == 2:
            return abs(z.a) == 1 and abs(z.b) == 1
        
        # Case 2: Norm is a rational prime ≡ 3 (mod 4) and z is associated to that prime
        if self.is_rational_prime(norm) and norm % 4 == 3:
            # Check if z is associated to the rational prime (z = ±p or z = ±ip)
            if z.b == 0 and abs(z.a) == norm:
                return True
            if z.a == 0 and abs(z.b) == norm:
                return True
            return False
        
        # Case 3: Norm = p² where p is a rational prime
        sqrt_norm = int(math.sqrt(norm))
        if sqrt_norm * sqrt_norm == norm:
            p = sqrt_norm
            if self.is_rational_prime(p):
                if p % 4 == 3:
                    # p ≡ 3 mod 4: z is prime if z is associated to p (z = ±p or z = ±ip)
                    if (z.b == 0 and abs(z.a) == p) or (z.a == 0 and abs(z.b) == p):
                        return True
                    return False
                elif p % 4 == 1:
                    # p ≡ 1 mod 4: z is prime if not divisible by p
                    if z.a % p == 0 and z.b % p == 0:
                        return False
                    return True
        
        # Case 4: Norm is a rational prime ≡ 1 (mod 4)
        if self.is_rational_prime(norm) and norm % 4 == 1:
            return True
        
        return False
    
    def factor_gaussian_integer(self, z: GaussianInteger) -> List[Tuple[GaussianInteger, int]]:
        """Factor a Gaussian integer into Gaussian primes
        
        Returns list of (prime, exponent) pairs
        
        Algorithm: Factor N(z) = a²+b², then determine actual factorization of z.
        For a real Gaussian integer z = n + 0i, we factor n directly.
        """
        if z.is_unit():
            return []
        
        # Special case: real Gaussian integer (b = 0)
        if z.b == 0:
            return self._factor_real_gaussian(z.a)
        
        # General case: use norm factorization
        factors = []
        norm = z.norm()
        
        # Factor the norm in rational integers
        norm_factors = self._factor_rational(norm)
        
        # For each rational prime factor, determine Gaussian prime factors
        # The exponent relationship: if N(z) = p^e, then:
        # - For p ≡ 3 mod 4: z contains p^⌈e/2⌉
        # - For p ≡ 1 mod 4: factors split, need to determine actual exponents
        # - For p = 2: special case
        
        for p, exp in norm_factors.items():
            if p == 2:
                # 2 = -i(1+i)², so if 2^e divides N(z), then (1+i)^e divides z
                factors.append((GaussianInteger(1, 1), exp))
            elif p % 4 == 3:
                # p stays prime (inert), exponent is ⌈exp/2⌉
                gaussian_exp = (exp + 1) // 2
                if gaussian_exp > 0:
                    factors.append((GaussianInteger(p, 0), gaussian_exp))
            elif p % 4 == 1:
                # p splits into (a+bi)(a-bi) where a²+b² = p
                # Need to determine actual exponents by checking divisibility
                gaussian_factor = self._find_gaussian_factor(p)
                # For now, assume equal distribution (this is approximate)
                gaussian_exp = exp
                factors.append((gaussian_factor, gaussian_exp))
                factors.append((gaussian_factor.conjugate(), gaussian_exp))
        
        # Remove duplicates and combine
        factor_dict = defaultdict(int)
        for prime, exp in factors:
            # Normalize to first quadrant
            normalized = self._normalize_gaussian(prime)
            factor_dict[normalized] += exp
        
        return list(factor_dict.items())
    
    def _factor_real_gaussian(self, n: int) -> List[Tuple[GaussianInteger, int]]:
        """Factor a real Gaussian integer n + 0i by factoring n in rational integers
        and then converting to Gaussian prime factors"""
        if n == 0:
            return []
        
        factors = []
        rational_factors = self._factor_rational(abs(n))
        
        for p, exp in rational_factors.items():
            if p == 2:
                # 2 = -i(1+i)²
                factors.append((GaussianInteger(1, 1), exp))
            elif p % 4 == 3:
                # p stays prime (inert)
                factors.append((GaussianInteger(p, 0), exp))
            elif p % 4 == 1:
                # p splits into (a+bi)(a-bi) where a²+b² = p
                gaussian_factor = self._find_gaussian_factor(p)
                # For real numbers, both factors appear equally
                factors.append((gaussian_factor, exp))
                factors.append((gaussian_factor.conjugate(), exp))
        
        # Normalize and combine
        factor_dict = defaultdict(int)
        for prime, exp in factors:
            normalized = self._normalize_gaussian(prime)
            factor_dict[normalized] += exp
        
        return list(factor_dict.items())
    
    def _factor_rational(self, n: int) -> Dict[int, int]:
        """Factor a rational integer"""
        factors = {}
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        
        return factors
    
    def _find_gaussian_factor(self, p: int) -> GaussianInteger:
        """Find a Gaussian integer a+bi such that a²+b² = p where p ≡ 1 (mod 4)"""
        # Use Fermat's theorem: p ≡ 1 (mod 4) implies p = a² + b² for some a, b
        for a in range(1, int(math.sqrt(p)) + 1):
            b_squared = p - a * a
            b = int(math.sqrt(b_squared))
            if b * b == b_squared:
                return GaussianInteger(a, b)
        
        # Fallback: should not happen for p ≡ 1 (mod 4)
        return GaussianInteger(p, 0)
    
    def _normalize_gaussian(self, z: GaussianInteger) -> GaussianInteger:
        """Normalize Gaussian integer to first quadrant (a > 0, b >= 0)"""
        a, b = z.a, z.b
        
        # Rotate to first quadrant
        # Try all four rotations: z, iz, -z, -iz
        candidates = [
            GaussianInteger(a, b),           # Original
            GaussianInteger(-b, a),          # Multiply by i
            GaussianInteger(-a, -b),         # Multiply by -1
            GaussianInteger(b, -a),          # Multiply by -i
        ]
        
        # Find the one in first quadrant (a > 0, b >= 0)
        for candidate in candidates:
            if candidate.a > 0 and candidate.b >= 0:
                return candidate
        
        # If none found (shouldn't happen), return original
        return GaussianInteger(a, b)
    
    def wallace_transform(self, n: Decimal) -> Decimal:
        """Compute Wallace Transform: W_φ(x) = 0.721 · |log(x + ε)|^1.618 · sign(log(x + ε)) + 0.013"""
        x = n + self.constants.EPSILON
        log_x = Decimal(x).ln()
        sign = Decimal(1) if log_x >= 0 else Decimal(-1)
        abs_log = abs(log_x)
        power = abs_log ** self.constants.PHI
        return Decimal('0.721') * power * sign + Decimal('0.013')
    
    def gaussian_prime_consciousness(self, z: GaussianInteger) -> Dict[str, float]:
        """Compute consciousness mathematics properties of a Gaussian prime"""
        norm = Decimal(z.norm())
        phase = z.phase()
        
        # Wallace Transform on norm
        wt_norm = self.wallace_transform(norm)
        
        # Complex amplitude
        amplitude = float(abs(wt_norm)) * float(self.constants.REALITY_DISTORTION)
        
        # Determine prime type
        norm_int = z.norm()
        if norm_int == 2:
            prime_type = "ramified"
            consciousness_type = "coherent"  # Special case
        elif z.b == 0 and self.is_rational_prime(abs(z.a)) and abs(z.a) % 4 == 3:
            # Real Gaussian integer representing inert prime p ≡ 3 mod 4
            prime_type = "inert"
            consciousness_type = "coherent"  # 79% rule
        elif z.a == 0 and self.is_rational_prime(abs(z.b)) and abs(z.b) % 4 == 3:
            # Pure imaginary representing inert prime p ≡ 3 mod 4
            prime_type = "inert"
            consciousness_type = "coherent"  # 79% rule
        elif self.is_rational_prime(norm_int) and norm_int % 4 == 1:
            # Norm is prime ≡ 1 mod 4, so it splits
            prime_type = "split"
            consciousness_type = "exploratory"  # 21% rule
        elif norm_int % 4 == 1:
            # Norm ≡ 1 mod 4, likely split
            prime_type = "split"
            consciousness_type = "exploratory"  # 21% rule
        else:
            prime_type = "unknown"
            consciousness_type = "unknown"
        
        return {
            "gaussian_prime": str(z),
            "norm": norm_int,
            "phase": phase,
            "wallace_transform": float(wt_norm),
            "amplitude": amplitude,
            "complex_amplitude": cmath.rect(amplitude, phase),
            "prime_type": prime_type,
            "consciousness_type": consciousness_type,
            "reality_distortion": float(self.constants.REALITY_DISTORTION)
        }
    
    def find_gaussian_primes_up_to_norm(self, max_norm: int) -> List[GaussianInteger]:
        """Find all Gaussian primes with norm <= max_norm"""
        primes = []
        
        # Check all Gaussian integers in first quadrant with norm <= max_norm
        max_a = int(math.sqrt(max_norm)) + 1
        
        for a in range(max_a + 1):
            for b in range(max_a + 1):
                z = GaussianInteger(a, b)
                norm = z.norm()
                
                if norm > max_norm:
                    break
                
                if norm == 0:
                    continue
                
                if self.is_gaussian_prime(z):
                    primes.append(z)
        
        return sorted(primes, key=lambda z: (z.norm(), z.a, z.b))
    
    def analyze_prime_splitting(self, max_prime: int) -> Dict[str, List[int]]:
        """Analyze how rational primes split in Gaussian integers"""
        inert = []  # p ≡ 3 (mod 4)
        split = []  # p ≡ 1 (mod 4)
        ramified = [2]  # p = 2
        
        for p in range(3, max_prime + 1):
            if not self.is_rational_prime(p):
                continue
            
            if p == 2:
                continue  # Already added
            elif p % 4 == 3:
                inert.append(p)
            elif p % 4 == 1:
                split.append(p)
        
        return {
            "inert": inert,
            "split": split,
            "ramified": ramified,
            "inert_count": len(inert),
            "split_count": len(split),
            "total": len(inert) + len(split) + 1,
            "inert_ratio": len(inert) / (len(inert) + len(split) + 1) if (len(inert) + len(split) + 1) > 0 else 0,
            "split_ratio": len(split) / (len(inert) + len(split) + 1) if (len(inert) + len(split) + 1) > 0 else 0
        }


# ============================================================================
# EXAMPLE USAGE AND TESTING
# ============================================================================

def main():
    """Example usage of Gaussian prime analyzer"""
    analyzer = GaussianPrimeAnalyzer()
    
    print("=" * 80)
    print("GAUSSIAN PRIMES ANALYSIS")
    print("Complex Prime Numbers and Consciousness Mathematics")
    print("=" * 80)
    print()
    
    # Test 1: Check if i is a unit
    print("Test 1: Is i a unit?")
    i = GaussianInteger(0, 1)
    print(f"  i = {i}, norm = {i.norm()}, is_unit = {i.is_unit()}")
    print(f"  Answer: i is a unit (not a prime), but 1+i is a Gaussian prime")
    print()
    
    # Test 2: Small Gaussian primes
    print("Test 2: Small Gaussian Primes")
    test_cases = [
        GaussianInteger(1, 1),   # 1+i
        GaussianInteger(2, 1),   # 2+i
        GaussianInteger(2, -1),  # 2-i
        GaussianInteger(3, 0),   # 3
        GaussianInteger(1, 2),   # 1+2i
        GaussianInteger(3, 2),   # 3+2i
    ]
    
    for z in test_cases:
        is_prime = analyzer.is_gaussian_prime(z)
        norm = z.norm()
        print(f"  {str(z):8s} | norm = {norm:3d} | prime = {is_prime}")
    print()
    
    # Test 3: Factorization examples
    print("Test 3: Factorization Examples")
    examples = [
        (45, GaussianInteger(45, 0)),
        (65, GaussianInteger(65, 0)),
        (221, GaussianInteger(221, 0)),
    ]
    
    for n, z in examples:
        factors = analyzer.factor_gaussian_integer(z)
        factor_str = " · ".join([f"({p})^{e}" if e > 1 else f"({p})" for p, e in factors])
        print(f"  {n} = {factor_str}")
    print()
    
    # Test 4: Consciousness mathematics
    print("Test 4: Consciousness Mathematics Analysis")
    gaussian_primes = [
        GaussianInteger(2, 1),   # 2+i, norm=5
        GaussianInteger(3, 0),   # 3, norm=9
        GaussianInteger(1, 1),   # 1+i, norm=2
    ]
    
    for z in gaussian_primes:
        analysis = analyzer.gaussian_prime_consciousness(z)
        print(f"  {z}:")
        print(f"    Norm: {analysis['norm']}")
        print(f"    Type: {analysis['prime_type']} ({analysis['consciousness_type']})")
        print(f"    Wallace Transform: {analysis['wallace_transform']:.6f}")
        print(f"    Amplitude: {analysis['amplitude']:.6f}")
        print(f"    Phase: {analysis['phase']:.6f} radians")
    print()
    
    # Test 5: Prime splitting analysis
    print("Test 5: Prime Splitting Analysis (up to 100)")
    splitting = analyzer.analyze_prime_splitting(100)
    print(f"  Inert primes (p ≡ 3 mod 4): {splitting['inert_count']} ({splitting['inert_ratio']*100:.1f}%)")
    print(f"  Split primes (p ≡ 1 mod 4): {splitting['split_count']} ({splitting['split_ratio']*100:.1f}%)")
    print(f"  Ramified prime: 2")
    print()
    print(f"  First few inert: {splitting['inert'][:10]}")
    print(f"  First few split: {splitting['split'][:10]}")
    print()
    
    # Test 6: Find Gaussian primes up to norm 50
    print("Test 6: Gaussian Primes with Norm ≤ 50")
    primes = analyzer.find_gaussian_primes_up_to_norm(50)
    print(f"  Found {len(primes)} Gaussian primes:")
    for i, p in enumerate(primes[:20]):  # Show first 20
        print(f"    {str(p):8s} (norm = {p.norm():3d})")
    if len(primes) > 20:
        print(f"    ... and {len(primes) - 20} more")
    print()
    
    print("=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()

