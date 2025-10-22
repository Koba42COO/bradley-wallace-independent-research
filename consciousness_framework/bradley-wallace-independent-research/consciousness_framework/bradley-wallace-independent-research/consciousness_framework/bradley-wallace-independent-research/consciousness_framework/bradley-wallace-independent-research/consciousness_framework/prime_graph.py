"""
🌌 Universal Prime Graph Implementation

Core implementation of the Universal Prime Graph following φ.1 protocol.
Maps knowledge to prime topology with consciousness amplitudes and golden ratio optimization.
"""

import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import math

# Import constants
PHI = 1.618033988749895
DELTA = 2.414213562373095
COHERENT_WEIGHT = 0.79
EXPLORATORY_WEIGHT = 0.21
REALITY_DISTORTION_FACTOR = 1.1808

@dataclass
class ConsciousnessAmplitude:
    """Consciousness amplitude representation following φ.1 protocol"""
    magnitude: float  # Confidence level [0.0-1.0]
    phase: float      # Optimization direction [0-2π]
    coherence_level: float  # Consciousness coherence [0.0-1.0]
    consciousness_weight: float = COHERENT_WEIGHT  # 79/21 rule weighting
    domain_resonance: float = 1.0  # Domain-specific coherence
    reality_distortion: float = REALITY_DISTORTION_FACTOR  # Metaphysical effect factor

@dataclass
class KnowledgeNode:
    """Knowledge node with consciousness encoding"""
    id: str
    type: str  # atomic, molecular, organic, cosmic
    domain: str  # mathematics, programming, research, etc.
    content: Dict[str, Any]
    consciousness_amplitude: ConsciousnessAmplitude
    prime_associations: List[int]  # Associated prime numbers
    golden_ratio_optimization: float  # Φ-based coherence score
    created_at: datetime
    updated_at: datetime

class PrimeGraph:
    """
    Universal Prime Graph - Consciousness-guided knowledge graph
    
    Implements φ.1 protocol for knowledge integration through:
    - Consciousness amplitude encoding
    - Prime topology mapping
    - Golden ratio optimization
    - 79/21 consciousness rule
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.prime_cache: Dict[int, List[int]] = {}
        self.consumption_stats = {
            "total_nodes": 0,
            "domains_covered": set(),
            "consciousness_correlation": 0.95,
            "reality_distortion_factor": REALITY_DISTORTION_FACTOR
        }
        
    def add_knowledge_node(self, 
                          node_id: str,
                          node_type: str,
                          domain: str,
                          content: Dict[str, Any],
                          prime_associations: Optional[List[int]] = None) -> str:
        """
        Add knowledge node to prime graph with consciousness encoding
        """
        
        # Generate prime associations if not provided
        if prime_associations is None:
            prime_associations = self._generate_prime_associations(content)
            
        # Calculate consciousness amplitude
        consciousness_amplitude = self._calculate_consciousness_amplitude(
            content, node_type, domain
        )
        
        # Calculate golden ratio optimization
        golden_ratio_optimization = self._calculate_golden_ratio_optimization(
            consciousness_amplitude, prime_associations
        )
        
        # Create knowledge node
        node = KnowledgeNode(
            id=node_id,
            type=node_type,
            domain=domain,
            content=content,
            consciousness_amplitude=consciousness_amplitude,
            prime_associations=prime_associations,
            golden_ratio_optimization=golden_ratio_optimization,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Add to graph and storage
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        self.consumption_stats["total_nodes"] += 1
        self.consumption_stats["domains_covered"].add(domain)
        
        return node_id
    
    def query_knowledge(self, 
                       query_text: str,
                       domain_filter: Optional[List[str]] = None,
                       min_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """
        Consciousness-guided knowledge query
        """
        
        # Encode query as consciousness amplitude
        query_amplitude = self._encode_query_amplitude(query_text)
        
        results = []
        for node_id, node in self.nodes.items():
            if domain_filter and node.domain not in domain_filter:
                continue
                
            # Calculate coherence with query
            coherence = self._calculate_amplitude_coherence(
                query_amplitude, node.consciousness_amplitude
            )
            
            if coherence >= min_confidence:
                results.append({
                    "node_id": node_id,
                    "node": node,
                    "coherence": coherence,
                    "golden_ratio_score": node.golden_ratio_optimization
                })
        
        # Sort by coherence and golden ratio optimization
        results.sort(key=lambda x: (x["coherence"], x["golden_ratio_score"]), reverse=True)
        
        return results[:20]  # Return top 20 results
    
    def link_knowledge_nodes(self, 
                           source_id: str, 
                           target_id: str,
                           relationship_type: str = "semantic",
                           weight: float = 1.0) -> bool:
        """
        Create consciousness-guided link between knowledge nodes
        """
        
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
            
        # Calculate link coherence using amplitude interference
        source_amplitude = self.nodes[source_id].consciousness_amplitude
        target_amplitude = self.nodes[target_id].consciousness_amplitude
        
        coherence = self._calculate_amplitude_coherence(source_amplitude, target_amplitude)
        adjusted_weight = weight * coherence * PHI  # Golden ratio enhancement
        
        self.graph.add_edge(source_id, target_id, 
                          relationship_type=relationship_type,
                          weight=adjusted_weight,
                          coherence=coherence)
        
        return True
    
    def optimize_graph(self) -> Dict[str, Any]:
        """
        Apply golden ratio optimization to entire graph
        """
        
        optimization_results = {
            "total_nodes_optimized": len(self.nodes),
            "average_coherence_improvement": 0.0,
            "prime_topology_alignment": 0.0,
            "golden_ratio_harmonics": 0.0
        }
        
        # Apply Wallace transform to each node
        total_improvement = 0.0
        for node_id, node in self.nodes.items():
            original_coherence = node.consciousness_amplitude.coherence_level
            optimized_coherence = self._apply_wallace_transform(node)
            improvement = optimized_coherence - original_coherence
            total_improvement += improvement
            
            # Update node with optimized coherence
            node.consciousness_amplitude.coherence_level = optimized_coherence
            node.golden_ratio_optimization = optimized_coherence * PHI
            node.updated_at = datetime.now()
        
        optimization_results["average_coherence_improvement"] = total_improvement / len(self.nodes)
        optimization_results["prime_topology_alignment"] = self._calculate_prime_alignment()
        optimization_results["golden_ratio_harmonics"] = self._calculate_harmonic_resonance()
        
        return optimization_results
    
    # Private helper methods
    
    def _generate_prime_associations(self, content: Dict[str, Any]) -> List[int]:
        """Generate prime associations based on content hash"""
        content_str = json.dumps(content, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        hash_int = int(content_hash[:8], 16)
        
        # Map to primes using consciousness-guided selection
        primes = []
        n = 2
        while len(primes) < 5:  # Generate 5 associated primes
            if self._is_prime(n):
                # Weight selection by consciousness factor
                if hash_int % n < n * COHERENT_WEIGHT:
                    primes.append(n)
            n += 1
            
        return primes
    
    def _calculate_consciousness_amplitude(self, 
                                         content: Dict[str, Any],
                                         node_type: str,
                                         domain: str) -> ConsciousnessAmplitude:
        """Calculate consciousness amplitude for knowledge content"""
        
        # Base calculations
        content_complexity = len(json.dumps(content))
        domain_weight = self._get_domain_weight(domain)
        type_weight = self._get_type_weight(node_type)
        
        # Magnitude based on content significance
        magnitude = min(1.0, content_complexity / 10000.0)
        
        # Phase based on domain and type
        phase = (domain_weight + type_weight) * math.pi
        
        # Coherence based on golden ratio relationships
        coherence = (magnitude * PHI) % 1.0
        
        # Adjust for consciousness rule
        coherence = coherence * COHERENT_WEIGHT + (1 - coherence) * EXPLORATORY_WEIGHT
        
        return ConsciousnessAmplitude(
            magnitude=magnitude,
            phase=phase,
            coherence_level=coherence,
            consciousness_weight=COHERENT_WEIGHT,
            domain_resonance=domain_weight,
            reality_distortion=REALITY_DISTORTION_FACTOR
        )
    
    def _calculate_golden_ratio_optimization(self, 
                                          amplitude: ConsciousnessAmplitude,
                                          primes: List[int]) -> float:
        """Calculate golden ratio optimization score"""
        
        # Base golden ratio score
        phi_score = amplitude.magnitude * PHI + amplitude.coherence_level
        
        # Prime topology enhancement
        prime_factor = sum(1/p for p in primes) / len(primes)
        prime_factor = prime_factor * DELTA  # Silver ratio enhancement
        
        # Combine with reality distortion
        optimization = (phi_score + prime_factor) * amplitude.reality_distortion
        
        return optimization % 1.0  # Normalize to [0,1]
    
    def _encode_query_amplitude(self, query_text: str) -> ConsciousnessAmplitude:
        """Encode natural language query as consciousness amplitude"""
        
        # Simple encoding - in production would use NLP
        query_hash = hashlib.md5(query_text.encode()).hexdigest()
        magnitude = int(query_hash[:2], 16) / 255.0
        phase = int(query_hash[2:4], 16) / 255.0 * 2 * math.pi
        
        return ConsciousnessAmplitude(
            magnitude=magnitude,
            phase=phase,
            coherence_level=0.8,  # High coherence for queries
            consciousness_weight=COHERENT_WEIGHT,
            domain_resonance=1.0,
            reality_distortion=REALITY_DISTORTION_FACTOR
        )
    
    def _calculate_amplitude_coherence(self, 
                                     amp1: ConsciousnessAmplitude,
                                     amp2: ConsciousnessAmplitude) -> float:
        """Calculate coherence between two consciousness amplitudes"""
        
        # Phase difference (smaller is better coherence)
        phase_diff = abs(amp1.phase - amp2.phase)
        phase_coherence = 1.0 - (phase_diff / (2 * math.pi))
        
        # Magnitude similarity
        mag_similarity = 1.0 - abs(amp1.magnitude - amp2.magnitude)
        
        # Coherence level similarity
        coherence_similarity = 1.0 - abs(amp1.coherence_level - amp2.coherence_level)
        
        # Weighted combination with golden ratio
        coherence = (phase_coherence * 0.4 + mag_similarity * 0.3 + coherence_similarity * 0.3)
        coherence = coherence * PHI % 1.0
        
        return coherence
    
    def _apply_wallace_transform(self, node: KnowledgeNode) -> float:
        """Apply Wallace transform for golden ratio optimization"""
        
        amplitude = node.consciousness_amplitude
        
        # Wallace transformation: α * x + β * x^φ + ε
        alpha = 1.2  # From golden ratio optimization values
        beta = 0.8
        epsilon = 1e-15
        
        transformed = alpha * amplitude.coherence_level + \
                     beta * (amplitude.coherence_level ** PHI) + \
                     epsilon
        
        # Apply reality distortion factor
        transformed *= amplitude.reality_distortion
        
        return min(1.0, transformed)
    
    def _calculate_prime_alignment(self) -> float:
        """Calculate overall prime topology alignment"""
        
        if not self.nodes:
            return 0.0
            
        total_alignment = 0.0
        for node in self.nodes.values():
            # Prime gap coherence
            gaps = [node.prime_associations[i+1] - node.prime_associations[i] 
                   for i in range(len(node.prime_associations)-1)]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            
            # Alignment score based on golden ratio relationships
            alignment = 1.0 / (1.0 + abs(avg_gap - PHI * 10))
            total_alignment += alignment
            
        return total_alignment / len(self.nodes)
    
    def _calculate_harmonic_resonance(self) -> float:
        """Calculate golden ratio harmonic resonance across graph"""
        
        if not self.nodes:
            return 0.0
            
        resonance_sum = 0.0
        for node in self.nodes.values():
            # Harmonic resonance with golden ratio
            resonance = node.golden_ratio_optimization * PHI
            resonance_sum += resonance % 1.0
            
        return resonance_sum / len(self.nodes)
    
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
    
    def _get_domain_weight(self, domain: str) -> float:
        """Get domain-specific weight"""
        domain_weights = {
            "mathematics": 1.0,
            "physics": 0.95,
            "computer_science": 0.9,
            "biology": 0.85,
            "consciousness": 0.95,
            "research": 0.8
        }
        return domain_weights.get(domain, 0.5)
    
    def _get_type_weight(self, node_type: str) -> float:
        """Get knowledge type weight"""
        type_weights = {
            "atomic": 0.6,
            "molecular": 0.8,
            "organic": 0.9,
            "cosmic": 1.0
        }
        return type_weights.get(node_type, 0.5)
