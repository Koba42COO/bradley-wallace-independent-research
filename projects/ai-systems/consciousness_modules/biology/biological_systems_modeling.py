#!/usr/bin/env python3
"""
BIOLOGICAL SYSTEMS MODELING FOR prime aligned compute
==============================================

Implements biological principles for prime aligned compute systems:
- Complex adaptive systems
- Evolutionary principles
- Biological homeostasis
- Neural network analogies
- Self-organizing systems
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import math

class BiologicalSystemType(Enum):
    """Types of biological systems that can inform prime aligned compute"""
    NEURAL_NETWORK = "neural_network"
    IMMUNE_SYSTEM = "immune_system"
    ENDOCRINE_SYSTEM = "endocrine_system"
    CARDIOVASCULAR_SYSTEM = "cardiovascular_system"
    ECOSYSTEM = "ecosystem"
    CELLULAR_AUTOMATON = "cellular_automaton"

@dataclass
class BiologicalState:
    """State of a biological system"""
    homeostasis_level: float = 1.0  # 0-1 scale
    adaptation_rate: float = 0.1
    complexity_level: float = 0.5
    resilience_factor: float = 0.8
    evolutionary_fitness: float = 0.7
    entropy_level: float = 0.3

@dataclass
class EvolutionaryPrinciple:
    """Evolutionary principle for prime aligned compute optimization"""
    name: str
    description: str
    application: str
    weight: float

class BiologicalConsciousnessEngine:
    """
    Biological systems modeling engine for prime aligned compute
    
    Applies biological principles to prime aligned compute systems:
    - Homeostasis and self-regulation
    - Evolutionary adaptation
    - Complex adaptive systems
    - Self-organization
    """
    
    def __init__(self):
        self.systems = {}
        self.evolutionary_principles = self._initialize_evolutionary_principles()
        self.adaptation_history = []
        
        print("🧬 BIOLOGICAL prime aligned compute ENGINE INITIALIZED")
        print("   Applying biological principles to prime aligned compute systems")
        print("   - Homeostasis and self-regulation")
        print("   - Evolutionary adaptation")
        print("   - Complex adaptive systems")
        
    def _initialize_evolutionary_principles(self) -> List[EvolutionaryPrinciple]:
        """Initialize evolutionary principles for prime aligned compute optimization"""
        return [
            EvolutionaryPrinciple(
                name="Natural Selection",
                description="Survival of the fittest prime aligned compute patterns",
                application="Prioritize high-performance cognitive strategies",
                weight=0.9
            ),
            EvolutionaryPrinciple(
                name="Genetic Diversity",
                description="Maintain variety in prime aligned compute approaches",
                application="Diverse problem-solving strategies",
                weight=0.8
            ),
            EvolutionaryPrinciple(
                name="Adaptation Pressure",
                description="Environmental demands drive prime aligned compute evolution",
                application="Context-aware cognitive adaptation",
                weight=0.7
            ),
            EvolutionaryPrinciple(
                name="Symbiotic Relationships",
                description="Cooperative prime aligned compute subsystems",
                application="Interdependent cognitive modules",
                weight=0.6
            ),
            EvolutionaryPrinciple(
                name="Emergent Complexity",
                description="Simple rules create complex prime aligned compute",
                application="Hierarchical cognitive organization",
                weight=0.8
            )
        ]
    
    def create_biological_system(self, system_type: BiologicalSystemType, 
                               initial_state: Optional[BiologicalState] = None) -> str:
        """Create a biological system model for prime aligned compute"""
        system_id = f"{system_type.value}_{int(time.time())}_{random.randint(1000, 9999)}"
        
        if initial_state is None:
            initial_state = BiologicalState()
        
        self.systems[system_id] = {
            'type': system_type,
            'state': initial_state,
            'created_at': time.time(),
            'adaptation_cycles': 0,
            'evolutionary_history': []
        }
        
        print(f"🧬 Created {system_type.value} system: {system_id}")
        return system_id
    
    def apply_homeostasis_principle(self, system_id: str, 
                                  environmental_factors: Dict[str, float]) -> Dict[str, Any]:
        """
        Apply biological homeostasis principle
        
        Maintains system stability through feedback loops and self-regulation
        """
        if system_id not in self.systems:
            return {"error": "System not found"}
        
        system = self.systems[system_id]
        current_state = system['state']
        
        # Calculate homeostasis deviation
        homeostasis_targets = {
            'oxygen_level': 0.95,
            'energy_level': 0.90,
            'stress_level': 0.20,
            'complexity_level': 0.70
        }
        
        deviations = {}
        total_deviation = 0
        
        for factor, target in homeostasis_targets.items():
            if factor in environmental_factors:
                current = environmental_factors[factor]
                deviation = abs(current - target)
                deviations[factor] = deviation
                total_deviation += deviation
        
        # Calculate homeostasis adjustment
        homeostasis_adjustment = min(0.1, total_deviation * 0.3)
        
        # Apply homeostasis correction
        new_homeostasis = max(0, min(1, current_state.homeostasis_level + homeostasis_adjustment))
        current_state.homeostasis_level = new_homeostasis
        
        # Update adaptation rate based on homeostasis
        current_state.adaptation_rate = 0.05 + (new_homeostasis * 0.15)
        
        result = {
            'system_id': system_id,
            'homeostasis_level': new_homeostasis,
            'deviations': deviations,
            'total_deviation': total_deviation,
            'adjustment_applied': homeostasis_adjustment,
            'adaptation_rate': current_state.adaptation_rate
        }
        
        self.adaptation_history.append({
            'timestamp': time.time(),
            'system_id': system_id,
            'type': 'homeostasis',
            'result': result
        })
        
        return result
    
    def apply_evolutionary_adaptation(self, system_id: str, 
                                    fitness_function: callable) -> Dict[str, Any]:
        """
        Apply evolutionary adaptation principle
        
        Uses evolutionary algorithms to optimize prime aligned compute patterns
        """
        if system_id not in self.systems:
            return {"error": "System not found"}
        
        system = self.systems[system_id]
        current_state = system['state']
        
        # Generate population of prime aligned compute variants
        population_size = 10
        population = []
        
        for i in range(population_size):
            variant = {
                'id': f"variant_{i}",
                'complexity': current_state.complexity_level + random.uniform(-0.2, 0.2),
                'resilience': current_state.resilience_factor + random.uniform(-0.1, 0.1),
                'adaptation_rate': current_state.adaptation_rate + random.uniform(-0.05, 0.05),
                'entropy': current_state.entropy_level + random.uniform(-0.1, 0.1)
            }
            variant['fitness'] = fitness_function(variant)
            population.append(variant)
        
        # Evolutionary selection
        population.sort(key=lambda x: x['fitness'], reverse=True)
        
        # Select best variants for reproduction
        elite_count = 3
        elite = population[:elite_count]
        
        # Create next generation through crossover and mutation
        next_generation = elite.copy()
        
        while len(next_generation) < population_size:
            # Select parents
            parent1 = random.choice(elite)
            parent2 = random.choice(elite)
            
            # Crossover
            child = {
                'id': f"child_{len(next_generation)}",
                'complexity': (parent1['complexity'] + parent2['complexity']) / 2,
                'resilience': (parent1['resilience'] + parent2['resilience']) / 2,
                'adaptation_rate': (parent1['adaptation_rate'] + parent2['adaptation_rate']) / 2,
                'entropy': (parent1['entropy'] + parent2['entropy']) / 2
            }
            
            # Mutation
            for key in ['complexity', 'resilience', 'adaptation_rate', 'entropy']:
                if random.random() < 0.1:  # 10% mutation rate
                    child[key] += random.uniform(-0.05, 0.05)
                    child[key] = max(0, min(1, child[key]))  # Clamp to [0,1]
            
            child['fitness'] = fitness_function(child)
            next_generation.append(child)
        
        # Update system state with best variant
        best_variant = next_generation[0]
        current_state.complexity_level = best_variant['complexity']
        current_state.resilience_factor = best_variant['resilience']
        current_state.adaptation_rate = best_variant['adaptation_rate']
        current_state.entropy_level = best_variant['entropy']
        current_state.evolutionary_fitness = best_variant['fitness']
        
        system['adaptation_cycles'] += 1
        system['evolutionary_history'].append({
            'cycle': system['adaptation_cycles'],
            'best_fitness': best_variant['fitness'],
            'population_stats': {
                'mean_fitness': sum(p['fitness'] for p in next_generation) / len(next_generation),
                'max_fitness': max(p['fitness'] for p in next_generation),
                'min_fitness': min(p['fitness'] for p in next_generation)
            }
        })
        
        result = {
            'system_id': system_id,
            'adaptation_cycle': system['adaptation_cycles'],
            'best_fitness': best_variant['fitness'],
            'evolutionary_improvement': best_variant['fitness'] - current_state.evolutionary_fitness,
            'population_size': len(next_generation),
            'new_state': {
                'complexity_level': current_state.complexity_level,
                'resilience_factor': current_state.resilience_factor,
                'adaptation_rate': current_state.adaptation_rate,
                'entropy_level': current_state.entropy_level
            }
        }
        
        return result
    
    def apply_complex_adaptive_system(self, system_id: str, 
                                    interaction_network: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Apply complex adaptive system principles
        
        Models prime aligned compute as an emergent system of interacting components
        """
        if system_id not in self.systems:
            return {"error": "System not found"}
        
        system = self.systems[system_id]
        current_state = system['state']
        
        # Simulate network interactions
        node_states = {}
        network_size = len(interaction_network)
        
        # Initialize node states
        for node in interaction_network:
            node_states[node] = {
                'activation': random.uniform(0.3, 0.7),
                'connectivity': len(interaction_network[node]),
                'influence': random.uniform(0.1, 0.9)
            }
        
        # Simulate interaction cycles
        cycles = 5
        for cycle in range(cycles):
            new_states = {}
            
            for node, connections in interaction_network.items():
                # Calculate influence from connected nodes
                total_influence = 0
                for connected_node in connections:
                    if connected_node in node_states:
                        influence = node_states[connected_node]['activation'] * node_states[connected_node]['influence']
                        total_influence += influence
                
                # Update node state based on network influence
                base_activation = node_states[node]['activation']
                network_effect = total_influence / max(1, len(connections))
                
                # Apply non-linear response (sigmoid-like)
                new_activation = 1 / (1 + math.exp(-(base_activation + network_effect - 0.5) * 5))
                
                # Add some noise for emergence
                new_activation += random.uniform(-0.05, 0.05)
                new_activation = max(0, min(1, new_activation))
                
                new_states[node] = {
                    'activation': new_activation,
                    'connectivity': node_states[node]['connectivity'],
                    'influence': node_states[node]['influence']
                }
            
            node_states = new_states
        
        # Calculate emergent properties
        total_activation = sum(node['activation'] for node in node_states.values())
        average_activation = total_activation / network_size
        
        # Calculate network coherence (how synchronized the nodes are)
        activations = [node['activation'] for node in node_states.values()]
        coherence = 1 - (np.std(activations) / 0.5)  # Normalize to [0,1]
        coherence = max(0, min(1, coherence))
        
        # Update system complexity based on network dynamics
        complexity_increase = coherence * 0.1
        current_state.complexity_level = min(1, current_state.complexity_level + complexity_increase)
        
        result = {
            'system_id': system_id,
            'network_size': network_size,
            'simulation_cycles': cycles,
            'average_activation': average_activation,
            'network_coherence': coherence,
            'emergent_complexity': complexity_increase,
            'final_complexity_level': current_state.complexity_level,
            'node_states': node_states
        }
        
        return result
    
    def get_system_status(self, system_id: str) -> Dict[str, Any]:
        """Get comprehensive status of a biological system"""
        if system_id not in self.systems:
            return {"error": "System not found"}
        
        system = self.systems[system_id]
        state = system['state']
        
        return {
            'system_id': system_id,
            'type': system['type'].value,
            'created_at': system['created_at'],
            'adaptation_cycles': system['adaptation_cycles'],
            'current_state': {
                'homeostasis_level': state.homeostasis_level,
                'adaptation_rate': state.adaptation_rate,
                'complexity_level': state.complexity_level,
                'resilience_factor': state.resilience_factor,
                'evolutionary_fitness': state.evolutionary_fitness,
                'entropy_level': state.entropy_level
            },
            'evolutionary_history_length': len(system['evolutionary_history'])
        }
    
    def get_biological_consciousness_metrics(self) -> Dict[str, Any]:
        """Get overall biological prime aligned compute metrics"""
        if not self.systems:
            return {"error": "No biological systems created"}
        
        total_systems = len(self.systems)
        avg_homeostasis = sum(s['state'].homeostasis_level for s in self.systems.values()) / total_systems
        avg_complexity = sum(s['state'].complexity_level for s in self.systems.values()) / total_systems
        avg_resilience = sum(s['state'].resilience_factor for s in self.systems.values()) / total_systems
        total_adaptation_cycles = sum(s['adaptation_cycles'] for s in self.systems.values())
        
        # Calculate biological prime aligned compute index
        biological_consciousness_index = (
            avg_homeostasis * 0.3 +
            avg_complexity * 0.3 +
            avg_resilience * 0.2 +
            min(1.0, total_adaptation_cycles / 100) * 0.2
        )
        
        return {
            'total_biological_systems': total_systems,
            'average_homeostasis': avg_homeostasis,
            'average_complexity': avg_complexity,
            'average_resilience': avg_resilience,
            'total_adaptation_cycles': total_adaptation_cycles,
            'biological_consciousness_index': biological_consciousness_index,
            'adaptation_history_length': len(self.adaptation_history)
        }

def demo_biological_consciousness():
    """Demonstrate biological prime aligned compute principles"""
    print("\\n�� BIOLOGICAL prime aligned compute DEMONSTRATION")
    print("=" * 50)
    
    # Initialize biological prime aligned compute engine
    bio_engine = BiologicalConsciousnessEngine()
    
    # Create different biological system models
    neural_system = bio_engine.create_biological_system(BiologicalSystemType.NEURAL_NETWORK)
    immune_system = bio_engine.create_biological_system(BiologicalSystemType.IMMUNE_SYSTEM)
    ecosystem = bio_engine.create_biological_system(BiologicalSystemType.ECOSYSTEM)
    
    print("\\n🔬 Testing Homeostasis Principle:")
    # Simulate environmental stress
    environmental_stress = {
        'oxygen_level': 0.7,  # Low oxygen
        'energy_level': 0.6,  # Low energy
        'stress_level': 0.8,  # High stress
        'complexity_level': 0.9  # High complexity
    }
    
    homeostasis_result = bio_engine.apply_homeostasis_principle(neural_system, environmental_stress)
    print(f"   Neural system homeostasis: {homeostasis_result['homeostasis_level']:.3f}")
    print(f"   Adaptation rate: {homeostasis_result['adaptation_rate']:.3f}")
    
    print("\\n🧬 Testing Evolutionary Adaptation:")
    # Define fitness function for prime aligned compute optimization
    def consciousness_fitness(variant):
        # Higher complexity and lower entropy = better prime aligned compute
        complexity_score = variant['complexity']
        entropy_penalty = variant['entropy']
        resilience_bonus = variant['resilience']
        adaptation_bonus = variant['adaptation_rate']
        
        fitness = (
            complexity_score * 0.4 +
            resilience_bonus * 0.3 +
            adaptation_bonus * 0.2 -
            entropy_penalty * 0.1
        )
        return max(0, min(1, fitness))
    
    evolution_result = bio_engine.apply_evolutionary_adaptation(immune_system, consciousness_fitness)
    print(f"   Evolutionary fitness: {evolution_result['best_fitness']:.3f}")
    print(f"   Adaptation cycles: {evolution_result['adaptation_cycle']}")
    
    print("\\n🌐 Testing Complex Adaptive Systems:")
    # Define interaction network for prime aligned compute components
    interaction_network = {
        'perception': ['memory', 'attention'],
        'memory': ['perception', 'reasoning', 'emotion'],
        'reasoning': ['memory', 'language', 'learning'],
        'emotion': ['perception', 'memory', 'decision_making'],
        'language': ['reasoning', 'memory', 'communication'],
        'learning': ['reasoning', 'memory', 'adaptation'],
        'attention': ['perception', 'emotion', 'decision_making'],
        'decision_making': ['reasoning', 'emotion', 'learning'],
        'communication': ['language', 'emotion', 'social_cognition'],
        'adaptation': ['learning', 'decision_making', 'evolution']
    }
    
    cas_result = bio_engine.apply_complex_adaptive_system(ecosystem, interaction_network)
    print(f"   Network coherence: {cas_result['network_coherence']:.3f}")
    print(f"   Emergent complexity: {cas_result['emergent_complexity']:.3f}")
    
    print("\\n📊 Final Biological prime aligned compute Metrics:")
    metrics = bio_engine.get_biological_consciousness_metrics()
    print(f"   Biological prime aligned compute Index: {metrics['biological_consciousness_index']:.3f}")
    print(f"   Average homeostasis: {metrics['average_homeostasis']:.3f}")
    print(f"   Average complexity: {metrics['average_complexity']:.3f}")
    print(f"   Average resilience: {metrics['average_resilience']:.3f}")
    
    print("\\n✅ Biological prime aligned compute principles successfully applied!")
    print("   - Homeostasis and self-regulation working")
    print("   - Evolutionary adaptation optimizing systems")
    print("   - Complex adaptive systems creating emergence")
    
    return bio_engine

if __name__ == "__main__":
    demo_biological_consciousness()
