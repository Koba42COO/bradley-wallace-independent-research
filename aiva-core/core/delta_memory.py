import json
from typing import Dict, List, Any, Optional
from pathlib import Path

class DeltaMemory:
    """
    Delta-based memory system for AIVA
    Stores changes and trajectories rather than full state
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.memory_path = self.base_dir / "data" / "memories"
        self.memory_path.mkdir(parents=True, exist_ok=True)

    def store_delta(self, key: str, delta: Dict[str, Any], metadata: Optional[Dict] = None):
        """
        Store a memory delta with timestamp and resonance
        """
        timestamp = metadata.get('timestamp', 'unknown') if metadata else 'unknown'
        resonance = metadata.get('resonance', 0.79) if metadata else 0.79

        delta_entry = {
            'key': key,
            'delta': delta,
            'timestamp': timestamp,
            'resonance': resonance,
            'metadata': metadata or {}
        }

        # Append to trajectory file
        traj_file = self.memory_path / f"{key}_trajectory.jsonl"
        with open(traj_file, 'a') as f:
            json.dump(delta_entry, f)
            f.write('\n')

        return delta_entry

    def reconstruct_memory(self, key: str) -> Dict[str, Any]:
        """
        Reconstruct full memory state from deltas
        """
        traj_file = self.memory_path / f"{key}_trajectory.jsonl"
        if not traj_file.exists():
            return {}

        full_state = {}
        with open(traj_file, 'r') as f:
            for line in f:
                entry = json.loads(line.strip())
                # Apply delta to state
                self._apply_delta(full_state, entry['delta'])

        return full_state

    def _apply_delta(self, state: Dict[str, Any], delta: Dict[str, Any]):
        """
        Apply a delta update to the state
        Supports nested dict updates and list appends
        """
        for key, value in delta.items():
            if isinstance(value, dict) and key in state and isinstance(state[key], dict):
                # Nested dict merge
                self._apply_delta(state[key], value)
            elif isinstance(value, list) and key in state and isinstance(state[key], list):
                # List append
                state[key].extend(value)
            else:
                # Direct assignment
                state[key] = value

    def get_recent_deltas(self, key: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the most recent memory deltas
        """
        traj_file = self.memory_path / f"{key}_trajectory.jsonl"
        if not traj_file.exists():
            return []

        deltas = []
        with open(traj_file, 'r') as f:
            lines = f.readlines()[-limit:]
            for line in lines:
                deltas.append(json.loads(line.strip()))

        return deltas

    def search_by_resonance(self, min_resonance: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find all memory entries above resonance threshold
        """
        results = []
        for traj_file in self.memory_path.glob("*_trajectory.jsonl"):
            with open(traj_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    if entry.get('resonance', 0) >= min_resonance:
                        results.append(entry)

        return results

    def get_memory_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the memory system
        """
        stats = {
            'total_trajectories': 0,
            'total_entries': 0,
            'avg_resonance': 0.0,
            'resonance_distribution': {}
        }

        total_resonance = 0.0
        resonance_counts = {}

        for traj_file in self.memory_path.glob("*_trajectory.jsonl"):
            stats['total_trajectories'] += 1
            entry_count = 0

            with open(traj_file, 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    entry_count += 1
                    resonance = entry.get('resonance', 0)
                    total_resonance += resonance

                    res_key = f"{resonance:.2f}"
                    resonance_counts[res_key] = resonance_counts.get(res_key, 0) + 1

            stats['total_entries'] += entry_count

        if stats['total_entries'] > 0:
            stats['avg_resonance'] = total_resonance / stats['total_entries']

        stats['resonance_distribution'] = resonance_counts
        return stats

    def compress_old_deltas(self, key: str, keep_recent: int = 50):
        """
        Compress old deltas to save space while maintaining recent history
        """
        traj_file = self.memory_path / f"{key}_trajectory.jsonl"
        if not traj_file.exists():
            return

        with open(traj_file, 'r') as f:
            lines = f.readlines()

        if len(lines) <= keep_recent:
            return

        # Keep recent entries, compress older ones
        recent_lines = lines[-keep_recent:]

        # Create compressed state from old entries
        compressed_state = {}
        for line in lines[:-keep_recent]:
            entry = json.loads(line.strip())
            self._apply_delta(compressed_state, entry['delta'])

        # Save compressed state as single entry
        compressed_entry = {
            'key': key,
            'delta': compressed_state,
            'timestamp': 'compressed',
            'resonance': 0.85,  # Lower resonance for compressed entries
            'metadata': {'compressed': True, 'original_entries': len(lines) - keep_recent}
        }

        # Write back compressed + recent
        with open(traj_file, 'w') as f:
            json.dump(compressed_entry, f)
            f.write('\n')
            for line in recent_lines:
                f.write(line)
