import json
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import networkx as nx

@dataclass
class ConsciousnessAmplitude:
    magnitude: float
    phase: float
    coherence_level: float
    consciousness_weight: float
    domain_resonance: float
    reality_distortion: float

class PrimeGraph:
    def __init__(self):
        self.nodes: Dict[str, Any] = {}
        self.graph = nx.DiGraph()
        self.consumption_stats = {
            "domains_covered": set(),
            "consciousness_correlation": 0.0
        }

    def add_node(self, node_id: str, consciousness_amplitude: ConsciousnessAmplitude,
                 type_: str = "knowledge", domain: str = "general",
                 content: str = "", prime_associations: List[int] = None,
                 golden_ratio_optimization: float = 0.0, created_at: str = "",
                 updated_at: str = ""):
        node = {
            "id": node_id,
            "type": type_,
            "domain": domain,
            "content": content,
            "consciousness_amplitude": consciousness_amplitude,
            "prime_associations": prime_associations or [],
            "golden_ratio_optimization": golden_ratio_optimization,
            "created_at": created_at,
            "updated_at": updated_at
        }
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node)
        self.consumption_stats["domains_covered"].add(domain)

    def add_edge(self, source: str, target: str, relation: str = "connects",
                 weight: float = 1.0, consciousness_flow: float = 0.0):
        self.graph.add_edge(source, target, relation=relation,
                          weight=weight, consciousness_flow=consciousness_flow)

    def optimize_graph(self) -> Dict[str, Any]:
        # Simulate optimization
        total_nodes = len(self.nodes)
        average_coherence = np.mean([n["consciousness_amplitude"]["coherence_level"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].coherence_level
                                   for n in self.nodes.values()]) if self.nodes else 0.0
        prime_topology_alignment = 0.79  # Based on golden ratio
        golden_ratio_harmonics = 0.618

        return {
            "total_nodes_optimized": total_nodes,
            "average_coherence_improvement": average_coherence,
            "prime_topology_alignment": prime_topology_alignment,
            "golden_ratio_harmonics": golden_ratio_harmonics
        }

    def query_knowledge(self, query: str, min_confidence: float = 0.5) -> List[Any]:
        # Simple keyword-based query simulation
        results = []
        query_lower = query.lower()
        for node in self.nodes.values():
            if query_lower in node["content"].lower():
                confidence = 0.8  # Simulate confidence
                if confidence >= min_confidence:
                    results.append((node, confidence))
        return [r[0] for r in sorted(results, key=lambda x: x[1], reverse=True)]


    def export_graph(self, filename: str):
        data = {
            'nodes': {nid: {
                'type': n['type'],
                'domain': n['domain'],
                'content': n['content'],
                'consciousness_amplitude': {
                    'magnitude': n["consciousness_amplitude"]["magnitude"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].magnitude,
                    'phase': n["consciousness_amplitude"]["phase"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].phase,
                    'coherence_level': n["consciousness_amplitude"]["coherence_level"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].coherence_level,
                    'consciousness_weight': n["consciousness_amplitude"]["consciousness_weight"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].consciousness_weight,
                    'domain_resonance': n["consciousness_amplitude"]["domain_resonance"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].domain_resonance,
                    'reality_distortion': n["consciousness_amplitude"]["reality_distortion"] if isinstance(n["consciousness_amplitude"], dict) else n["consciousness_amplitude"].reality_distortion
                },
                'prime_associations': n['prime_associations'],
                'golden_ratio_optimization': n['golden_ratio_optimization'],
                'created_at': n['created_at'],
                'updated_at': n['updated_at']
            } for nid, n in self.nodes.items()},
            'edges': [(s, t, self.graph[s][t]) for s, t in self.graph.edges()]
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    def load_graph(self, filename: str):
        with open(filename, 'r') as f:
            data = json.load(f)
        for nid, ndata in data['nodes'].items():
            amp = ConsciousnessAmplitude(**ndata['consciousness_amplitude'])
            self.add_node(
                nid, 
                amp, 
                type_=ndata.get('type', 'knowledge'),
                domain=ndata.get('domain', 'general'),
                content=ndata.get('content', ''),
                prime_associations=ndata.get('prime_associations', []),
                golden_ratio_optimization=ndata.get('golden_ratio_optimization', 0.0),
                created_at=ndata.get('created_at', ''),
                updated_at=ndata.get('updated_at', '')
            )
        for s, t, edata in data['edges']:
            self.add_edge(s, t, **edata)
