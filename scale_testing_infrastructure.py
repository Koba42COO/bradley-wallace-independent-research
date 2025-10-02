"""
SCALE TESTING INFRASTRUCTURE FOR 10^6 RANGE
Optimized for memory efficiency and speed
"""

import numpy as np
import time
import psutil
import os
from typing import List, Tuple, Dict
import json

class MemoryEfficientSieve:
    """Segmented sieve for large ranges with memory control"""

    def __init__(self, max_n: int = 10**7, segment_size: int = 10**6):
        self.max_n = max_n
        self.segment_size = segment_size

    def segmented_sieve(self, start: int, end: int) -> List[int]:
        """Generate primes in range [start, end] using segmented sieve"""
        if end > self.max_n:
            raise ValueError(f"End {end} exceeds max_n {self.max_n}")

        primes = []
        segment_start = start
        segment_end = min(start + self.segment_size, end + 1)

        # Get small primes for sieving larger segments
        small_primes = self._small_primes(int(np.sqrt(end)) + 1)

        while segment_start < end:
            # Create boolean array for current segment
            segment_size = segment_end - segment_start
            is_prime = np.ones(segment_size, dtype=bool)

            # Handle number 1
            if segment_start == 0:
                is_prime[1] = False
            elif segment_start == 1:
                is_prime[0] = False

            # Mark multiples of small primes
            for prime in small_primes:
                # Find smallest multiple >= segment_start
                start_idx = ((segment_start + prime - 1) // prime) * prime - segment_start
                if start_idx < 0:
                    start_idx += prime

                # Mark multiples
                for j in range(start_idx, segment_size, prime):
                    is_prime[j] = False

            # Collect primes from this segment
            for i in range(segment_size):
                if is_prime[i]:
                    num = segment_start + i
                    if num >= start and num <= end:
                        primes.append(num)

            # Next segment
            segment_start = segment_end
            segment_end = min(segment_end + self.segment_size, end + 1)

        return primes

    def _small_primes(self, limit: int) -> List[int]:
        """Generate small primes using simple sieve"""
        if limit < 2:
            return []

        sieve = np.ones(limit + 1, dtype=bool)
        sieve[0:2] = False

        for i in range(2, int(np.sqrt(limit)) + 1):
            if sieve[i]:
                sieve[i*i:limit+1:i] = False

        return [i for i in range(2, limit + 1) if sieve[i]]

class ScaleBenchmark:
    """Benchmark ML performance across scales"""

    def __init__(self):
        self.sieve = MemoryEfficientSieve()
        self.baseline_accuracies = {}

    def test_range(self, start: int, end: int) -> Dict:
        """Test performance on a specific range"""
        print(f"Testing range: {start:,} - {end:,}")

        # Memory and time tracking
        start_time = time.time()
        start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB

        # Generate primes
        primes = self.sieve.segmented_sieve(start, end)
        n_primes = len(primes)

        # Calculate baseline accuracy (extended sieving effectiveness)
        total_numbers = end - start + 1
        total_candidates = 0
        correct_sieve_calls = 0

        for n in range(max(2, start), end + 1):  # Start from 2, exclude 1
            total_candidates += 1

            # Check if sieved out (divisible by primes ≤ 11)
            sieved_out = any(n % p == 0 for p in [2, 3, 5, 7, 11])

            # Check if actually composite (not prime)
            is_actual_prime = n in primes
            is_actual_composite = not is_actual_prime

            # Correct if: (sieved_out AND is_composite) OR (not sieved_out AND is_prime)
            if (sieved_out and is_actual_composite) or (not sieved_out and is_actual_prime):
                correct_sieve_calls += 1

        baseline_accuracy = correct_sieve_calls / total_candidates if total_candidates > 0 else 0

        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

        result = {
            'range': f"{start}-{end}",
            'total_numbers': total_numbers,
            'primes_found': n_primes,
            'baseline_accuracy': baseline_accuracy,
            'time_seconds': end_time - start_time,
            'memory_mb': end_memory - start_memory,
            'primes_per_second': n_primes / (end_time - start_time)
        }

        print(".4f")
        print(".1f")
        print(".2f")
        return result

    def run_scale_tests(self) -> Dict:
        """Run tests across multiple scales"""
        test_ranges = [
            (100000, 101000),    # 10^5 scale
            (500000, 501000),    # 5×10^5 scale
            (1000000, 1001000),  # 10^6 scale (target)
        ]

        results = {}
        for start, end in test_ranges:
            results[f"{start}-{end}"] = self.test_range(start, end)

        # Save results
        with open('/Users/coo-koba42/dev/scale_test_results.json', 'w') as f:
            json.dump(results, f, indent=2)

        return results

def main():
    """Run scale testing infrastructure"""
    print("🚀 SCALE TESTING INFRASTRUCTURE")
    print("=" * 40)

    benchmark = ScaleBenchmark()
    results = benchmark.run_scale_tests()

    print("\\n📊 SCALE TEST SUMMARY")
    print("=" * 25)

    for range_name, data in results.items():
        print(f"{range_name}: {data['baseline_accuracy']:.1%} accuracy, "
              f"{data['time_seconds']:.2f}s, {data['memory_mb']:.1f}MB")

    # Check if baseline degrades with scale
    accuracies = [data['baseline_accuracy'] for data in results.values()]
    if len(accuracies) > 1:
        trend = np.polyfit(range(len(accuracies)), accuracies, 1)[0]
        if trend < 0:
            print("\\n⚠️  Baseline accuracy degrades with scale (as expected)")
            print(".6f")
        else:
            print("\\n✅ Baseline accuracy stable across scales")

if __name__ == "__main__":
    main()
