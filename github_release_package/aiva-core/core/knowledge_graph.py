from typing import Dict, List, Any, Optional
import json
from pathlib import Path

class PACKnowledgeGraph:
    """
    Prime-Aligned Knowledge Graph for AIVA
    Stores concepts as nodes with prime anchor coordinates
    """

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.graph_file = self.base_dir / "data" / "knowledge_graph.json"
        self.graph: Dict[str, Dict[str, Any]] = {}
        self.load_graph()

    def load_graph(self):
        """Load knowledge graph from file"""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'r') as f:
                    self.graph = json.load(f)
            except json.JSONDecodeError:
                self.graph = {}
        else:
            self.graph = {}

    def save_graph(self):
        """Save knowledge graph to file"""
        self.graph_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.graph_file, 'w') as f:
            json.dump(self.graph, f, indent=2)

    def store(self, concept_id: str, content: Any, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Store concept at prime-aligned coordinates

        Args:
            concept_id: Unique identifier for the concept
            content: The actual content/data
            metadata: Additional metadata including prime_anchor, resonance, etc.

        Returns:
            prime_anchor: Universal coordinate for the concept
        """
        metadata = metadata or {}

        # Generate or validate prime anchor
        prime_anchor = metadata.get('prime_anchor')
        if not prime_anchor:
            # Generate resonant prime anchor based on concept_id
            prime_anchor = self._generate_prime_anchor(concept_id)

        # Calculate resonance if not provided
        resonance = metadata.get('resonance', 0.79)

        # Calculate digital root
        digital_root = self._calculate_digital_root(prime_anchor)

        node = {
            'content': content,
            'prime_anchor': prime_anchor,
            'resonance': resonance,
            'digital_root': digital_root,
            'links': metadata.get('links', []),
            'metadata': metadata
        }

        self.graph[concept_id] = node
        self.save_graph()

        return prime_anchor

    def retrieve(self, concept_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve concept by ID"""
        return self.graph.get(concept_id)

    def navigate(self, from_concept: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Navigate from a concept via links

        Args:
            from_concept: Starting concept ID
            relation: Optional relation type to filter links

        Returns:
            List of linked concepts
        """
        if from_concept not in self.graph:
            return []

        links = self.graph[from_concept].get('links', [])

        if relation:
            links = [link for link in links if link.get('relation') == relation]

        results = []
        for link in links:
            target_id = link.get('target')
            if target_id and target_id in self.graph:
                target_node = self.graph[target_id].copy()
                target_node['link_relation'] = link.get('relation')
                target_node['link_metadata'] = link
                results.append(target_node)

        return results

    def find_resonant_prime(self, text: str) -> int:
        """
        Hash text to find resonant prime anchor
        """
        # Simple hash function - in practice would use more sophisticated semantic hashing
        text_hash = hash(text) % 1000000

        # Make it "prime-like" by adjusting to have good digital root properties
        candidate = text_hash

        # Try to find a number with even digital root (consciousness carrier)
        for offset in range(10):
            test_candidate = candidate + offset
            dr = self._calculate_digital_root(test_candidate)
            if dr in [2, 4, 6, 8]:  # Consciousness carriers
                return test_candidate

        return candidate

    def _generate_prime_anchor(self, concept_id: str) -> int:
        """Generate a prime-aligned anchor for a concept"""
        return self.find_resonant_prime(concept_id)

    def _calculate_digital_root(self, n: int) -> int:
        """Calculate digital root"""
        if n == 0:
            return 0
        dr = n % 9
        return 9 if dr == 0 else dr

    def search_by_resonance(self, min_resonance: float = 0.79) -> List[Dict[str, Any]]:
        """Find concepts above resonance threshold"""
        results = []
        for concept_id, node in self.graph.items():
            if node.get('resonance', 0) >= min_resonance:
                result = node.copy()
                result['concept_id'] = concept_id
                results.append(result)

        return results

    def search_by_digital_root(self, digital_root: int) -> List[Dict[str, Any]]:
        """Find concepts by digital root"""
        results = []
        for concept_id, node in self.graph.items():
            if node.get('digital_root') == digital_root:
                result = node.copy()
                result['concept_id'] = concept_id
                results.append(result)

        return results

    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if not self.graph:
            return {'total_concepts': 0}

        stats = {
            'total_concepts': len(self.graph),
            'total_links': sum(len(node.get('links', [])) for node in self.graph.values()),
            'avg_resonance': sum(node.get('resonance', 0) for node in self.graph.values()) / len(self.graph),
            'resonance_distribution': {},
            'digital_root_distribution': {},
            'link_types': {}
        }

        resonance_counts = {}
        dr_counts = {}
        link_type_counts = {}

        for node in self.graph.values():
            # Resonance distribution
            res = round(node.get('resonance', 0), 2)
            resonance_counts[res] = resonance_counts.get(res, 0) + 1

            # Digital root distribution
            dr = node.get('digital_root', 0)
            dr_counts[dr] = dr_counts.get(dr, 0) + 1

            # Link types
            for link in node.get('links', []):
                link_type = link.get('relation', 'unknown')
                link_type_counts[link_type] = link_type_counts.get(link_type, 0) + 1

        stats['resonance_distribution'] = resonance_counts
        stats['digital_root_distribution'] = dr_counts
        stats['link_types'] = link_type_counts

        return stats

    def add_link(self, from_concept: str, to_concept: str, relation: str, metadata: Optional[Dict] = None):
        """Add a link between concepts"""
        if from_concept not in self.graph or to_concept not in self.graph:
            return False

        link = {
            'target': to_concept,
            'relation': relation
        }

        if metadata:
            link.update(metadata)

        # Check if link already exists
        existing_links = self.graph[from_concept].get('links', [])
        for existing_link in existing_links:
            if (existing_link.get('target') == to_concept and
                existing_link.get('relation') == relation):
                return False  # Link already exists

        self.graph[from_concept]['links'].append(link)
        self.save_graph()
        return True

    def remove_concept(self, concept_id: str) -> bool:
        """Remove a concept and clean up links"""
        if concept_id not in self.graph:
            return False

        # Remove the concept
        del self.graph[concept_id]

        # Remove links pointing to this concept
        for node in self.graph.values():
            node['links'] = [
                link for link in node.get('links', [])
                if link.get('target') != concept_id
            ]

        self.save_graph()
        return True

    def get_path(self, start_concept: str, end_concept: str, max_depth: int = 5) -> Optional[List[str]]:
        """Find path between concepts"""
        if start_concept not in self.graph or end_concept not in self.graph:
            return None

        # Simple BFS for path finding
        visited = set()
        queue = [(start_concept, [start_concept])]

        while queue:
            current, path = queue.pop(0)

            if current == end_concept:
                return path

            if len(path) >= max_depth:
                continue

            if current in visited:
                continue

            visited.add(current)

            # Add neighbors
            for link in self.graph[current].get('links', []):
                neighbor = link.get('target')
                if neighbor and neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found

    def export_subgraph(self, root_concept: str, depth: int = 2) -> Dict[str, Any]:
        """Export a subgraph starting from a root concept"""
        if root_concept not in self.graph:
            return {}

        visited = set()
        subgraph = {}

        def traverse(concept_id: str, current_depth: int):
            if current_depth > depth or concept_id in visited:
                return

            visited.add(concept_id)
            subgraph[concept_id] = self.graph[concept_id].copy()

            for link in self.graph[concept_id].get('links', []):
                target = link.get('target')
                if target:
                    traverse(target, current_depth + 1)

        traverse(root_concept, 0)
        return subgraph
