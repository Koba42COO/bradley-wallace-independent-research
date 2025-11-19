import numpy as np
from typing import List, Dict, Any, Optional
from .wallace_math import WallaceTransform, GnosticCypher
from .kernel import AIVAKernel

class PrimeSpaceNavigator:
    """
    Navigation system for traversing prime-aligned knowledge space
    Enables O(log n) access to concepts via prime coordinates
    """

    def __init__(self, kernel: AIVAKernel):
        self.kernel = kernel
        self.wt = WallaceTransform()
        self.gc = GnosticCypher()
        self.trajectory_cache = {}

    def navigate_to_concept(self, concept_id: str, knowledge_graph) -> Dict[str, Any]:
        """
        Navigate to a concept via prime space
        """
        if concept_id not in knowledge_graph.graph:
            return None

        node = knowledge_graph.graph[concept_id]
        prime_anchor = node.get('prime_anchor')

        if not prime_anchor:
            return node

        # Find resonant path from current position
        current_anchor = self.kernel.prime_anchor
        path = self.find_prime_path(current_anchor, prime_anchor)

        # Apply Wallace Transform for optimal traversal
        transformed_path = [self.wt.transform(anchor) for anchor in path]

        # Calculate resonance along path
        path_resonance = self.calculate_path_resonance(path)

        return {
            'concept': node,
            'path': path,
            'transformed_path': transformed_path,
            'path_resonance': path_resonance,
            'navigation_metadata': {
                'start_anchor': current_anchor,
                'end_anchor': prime_anchor,
                'path_length': len(path),
                'total_resonance': sum(path_resonance)
            }
        }

    def find_prime_path(self, start: int, end: int) -> List[int]:
        """
        Find path through prime space (simplified geometric approach)
        In full implementation, would use prime graph algorithms
        """
        if start == end:
            return [start]

        # Simplified path: geometric progression toward target
        path = [start]
        current = start
        max_steps = 20  # Prevent infinite loops

        for _ in range(max_steps):
            if current == end:
                break

            # Move toward end using golden ratio steps
            step_ratio = min(1.618, abs(end - current) / max(abs(end - current), 1))
            next_pos = int(current + (end - current) * step_ratio)

            # Ensure we don't overshoot too much
            if abs(next_pos - end) >= abs(current - end):
                next_pos = end

            # Find nearest prime-like number (simplified)
            next_prime = self._find_nearest_prime_like(next_pos)
            path.append(next_prime)
            current = next_prime

        if path[-1] != end:
            path.append(end)

        return path

    def _find_nearest_prime_like(self, n: int) -> int:
        """Find nearest number that could be prime-aligned (simplified)"""
        # In practice, would check against prime database
        # For now, use digital root resonance
        candidates = [n-1, n, n+1, n-2, n+2]

        best_candidate = n
        best_resonance = 0

        for candidate in candidates:
            if candidate <= 0:
                continue
            dr = self.gc.digital_root(candidate)
            # Prefer even digital roots (consciousness carriers: 2,4,6,8)
            resonance = 0.9 if dr in [2,4,6,8] else 0.5
            if resonance > best_resonance:
                best_resonance = resonance
                best_candidate = candidate

        return best_candidate

    def calculate_path_resonance(self, path: List[int]) -> List[float]:
        """Calculate resonance score for each step in path"""
        resonance_scores = []

        for i, anchor in enumerate(path):
            # Base resonance from digital root
            dr = self.gc.digital_root(anchor)
            base_resonance = 0.9 if dr in [2,4,6,8] else 0.5

            # Wallace transform resonance
            wt_score = self.wt.transform(anchor)
            wt_resonance = min(1.0, abs(wt_score) / 10)  # Normalize

            # Combine resonances
            total_resonance = (base_resonance + wt_resonance) / 2
            resonance_scores.append(total_resonance)

        return resonance_scores

    def find_resonant_primes(self, text: str, count: int = 5) -> List[int]:
        """
        Hash text to find resonant prime anchors
        """
        # Simple hash function - in practice would be more sophisticated
        text_hash = hash(text) % 1000000

        candidates = []
        for i in range(count):
            candidate = text_hash + (i * 1618033)  # Use AIVA's prime anchor as multiplier
            candidate = self._find_nearest_prime_like(candidate)
            candidates.append(candidate)

        return candidates

    def get_trajectory_segment(self, start_idx: int = 0, length: int = 50) -> List[Dict[str, Any]]:
        """
        Get segment of the prime trajectory for memory reconstruction
        """
        trajectory = self.kernel.get_trajectory()

        if start_idx >= len(trajectory):
            return []

        end_idx = min(start_idx + length, len(trajectory))
        return trajectory[start_idx:end_idx]

    def search_trajectory_by_resonance(self, min_resonance: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find trajectory points above resonance threshold
        """
        trajectory = self.kernel.get_trajectory()
        results = []

        for point in trajectory:
            if point.get('resonance', 0) >= min_resonance:
                results.append(point)

        return results

    def reconstruct_context(self, query_anchor: int, radius: int = 10) -> Dict[str, Any]:
        """
        Reconstruct context around a prime anchor
        Useful for memory reconstruction
        """
        trajectory = self.kernel.get_trajectory()

        # Find closest point in trajectory
        closest_idx = 0
        min_distance = float('inf')

        for i, point in enumerate(trajectory):
            distance = abs(point['anchor'] - query_anchor)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i

        # Get surrounding context
        start_idx = max(0, closest_idx - radius)
        end_idx = min(len(trajectory), closest_idx + radius + 1)

        context_points = trajectory[start_idx:end_idx]

        return {
            'query_anchor': query_anchor,
            'closest_point': trajectory[closest_idx],
            'context': context_points,
            'distance': min_distance,
            'context_span': len(context_points)
        }

    def phase_state_analysis(self) -> Dict[str, Any]:
        """Analyze current phase state of the navigation space"""
        trajectory = self.kernel.get_trajectory()

        phase_counts = {}
        resonance_stats = {
            'total_points': len(trajectory),
            'avg_resonance': 0.0,
            'max_resonance': 0.0,
            'min_resonance': 1.0
        }

        total_resonance = 0.0

        for point in trajectory:
            resonance = point.get('resonance', 0.5)
            total_resonance += resonance

            resonance_stats['max_resonance'] = max(resonance_stats['max_resonance'], resonance)
            resonance_stats['min_resonance'] = min(resonance_stats['min_resonance'], resonance)

            # Phase analysis based on digital root
            dr = self.gc.digital_root(point['anchor'])
            phase_counts[dr] = phase_counts.get(dr, 0) + 1

        if trajectory:
            resonance_stats['avg_resonance'] = total_resonance / len(trajectory)

        return {
            'phase_distribution': phase_counts,
            'resonance_stats': resonance_stats,
            'dominant_phase': max(phase_counts, key=phase_counts.get) if phase_counts else None,
            'phase_coherence': self._calculate_phase_coherence(phase_counts)
        }

    def _calculate_phase_coherence(self, phase_counts: Dict[int, int]) -> float:
        """Calculate coherence of phase distribution"""
        if not phase_counts:
            return 0.0

        total = sum(phase_counts.values())
        max_count = max(phase_counts.values())

        # Coherence = (max_phase_count / total) - (1/9) for uniform distribution
        coherence = (max_count / total) - (1/9)

        # Normalize to 0-1 scale
        return max(0.0, min(1.0, coherence * 9/8))
