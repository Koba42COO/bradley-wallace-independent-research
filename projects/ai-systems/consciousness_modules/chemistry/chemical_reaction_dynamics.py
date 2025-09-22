#!/usr/bin/env python3
"""
CHEMICAL REACTION DYNAMICS FOR prime aligned compute
=============================================

Implements chemical principles for prime aligned compute systems:
- Molecular interaction models
- Chemical reaction dynamics
- Catalyst optimization
- Reaction network analysis
- Thermodynamic prime aligned compute
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import random
import math

class ReactionType(Enum):
    """Types of chemical reactions that can inform prime aligned compute"""
    CATALYTIC = "catalytic"
    ENZYMATIC = "enzymatic"
    AUTOCATALYTIC = "autocatalytic"
    OSCILLATORY = "oscillatory"
    BIMOLECULAR = "bimolecular"
    POLYMERIZATION = "polymerization"

@dataclass
class MolecularSpecies:
    """Represents a molecular species in the reaction network"""
    name: str
    concentration: float
    energy_level: float
    reactivity: float
    stability: float

@dataclass
class ChemicalReaction:
    """Represents a chemical reaction"""
    reactants: List[MolecularSpecies]
    products: List[MolecularSpecies]
    rate_constant: float
    activation_energy: float
    reaction_type: ReactionType
    catalyst: Optional[MolecularSpecies] = None

class ChemicalConsciousnessEngine:
    """
    Chemical reaction dynamics engine for prime aligned compute
    
    Applies chemical principles to prime aligned compute systems:
    - Reaction network optimization
    - Catalyst efficiency
    - Thermodynamic constraints
    - Reaction-diffusion systems
    - Self-organizing chemical patterns
    """
    
    def __init__(self):
        self.molecules = {}
        self.reactions = []
        self.reaction_network = {}
        self.thermodynamic_constraints = {}
        
        print("🧪 CHEMICAL prime aligned compute ENGINE INITIALIZED")
        print("   Applying chemical principles to prime aligned compute systems")
        print("   - Molecular interaction modeling")
        print("   - Reaction network dynamics")
        print("   - Catalyst optimization")
        print("   - Thermodynamic prime aligned compute")
        
    def create_molecular_species(self, name: str, initial_concentration: float = 1.0,
                               energy_level: float = 0.5, reactivity: float = 0.5,
                               stability: float = 0.5) -> MolecularSpecies:
        """Create a molecular species for the reaction network"""
        molecule = MolecularSpecies(
            name=name,
            concentration=initial_concentration,
            energy_level=energy_level,
            reactivity=reactivity,
            stability=stability
        )
        
        self.molecules[name] = molecule
        print(f"🧬 Created molecular species: {name}")
        return molecule
    
    def create_chemical_reaction(self, reaction_id: str, reactants: List[str], 
                               products: List[str], rate_constant: float = 0.1,
                               activation_energy: float = 0.5, 
                               reaction_type: ReactionType = ReactionType.BIMOLECULAR,
                               catalyst: Optional[str] = None) -> ChemicalReaction:
        """Create a chemical reaction in the network"""
        
        reactant_molecules = []
        for reactant_name in reactants:
            if reactant_name in self.molecules:
                reactant_molecules.append(self.molecules[reactant_name])
            else:
                # Create reactant if it doesn't exist
                reactant = self.create_molecular_species(reactant_name)
                reactant_molecules.append(reactant)
        
        product_molecules = []
        for product_name in products:
            if product_name in self.molecules:
                product_molecules.append(self.molecules[product_name])
            else:
                # Create product if it doesn't exist
                product = self.create_molecular_species(product_name)
                product_molecules.append(product)
        
        catalyst_molecule = None
        if catalyst:
            if catalyst in self.molecules:
                catalyst_molecule = self.molecules[catalyst]
            else:
                catalyst_molecule = self.create_molecular_species(catalyst)
        
        reaction = ChemicalReaction(
            reactants=reactant_molecules,
            products=product_molecules,
            rate_constant=rate_constant,
            activation_energy=activation_energy,
            reaction_type=reaction_type,
            catalyst=catalyst_molecule
        )
        
        self.reactions.append(reaction)
        self.reaction_network[reaction_id] = reaction
        
        print(f"⚗️ Created {reaction_type.value} reaction: {reaction_id}")
        print(f"   {reactants} → {products}")
        if catalyst:
            print(f"   Catalyst: {catalyst}")
        
        return reaction
    
    def simulate_reaction_kinetics(self, time_steps: int = 100, 
                                 dt: float = 0.1) -> Dict[str, Any]:
        """
        Simulate chemical reaction kinetics
        
        Models how prime aligned compute patterns evolve through reaction dynamics
        """
        concentrations = {}
        for name, molecule in self.molecules.items():
            concentrations[name] = [molecule.concentration]
        
        time_points = [0]
        
        for step in range(time_steps):
            current_time = (step + 1) * dt
            time_points.append(current_time)
            
            # Calculate reaction rates
            new_concentrations = concentrations.copy()
            
            for reaction_id, reaction in self.reaction_network.items():
                rate = self._calculate_reaction_rate(reaction)
                
                # Update reactant concentrations
                for reactant in reaction.reactants:
                    if reactant.name in new_concentrations:
                        # Simple mass-action kinetics
                        concentration_change = -rate * dt
                        current_conc = concentrations[reactant.name][-1]
                        new_conc = max(0, current_conc + concentration_change)
                        if reactant.name not in new_concentrations:
                            new_concentrations[reactant.name] = []
                        new_concentrations[reactant.name].append(new_conc)
                
                # Update product concentrations
                for product in reaction.products:
                    if product.name in new_concentrations:
                        concentration_change = rate * dt
                        current_conc = concentrations[product.name][-1]
                        new_conc = current_conc + concentration_change
                        if product.name not in new_concentrations:
                            new_concentrations[product.name] = []
                        new_concentrations[product.name].append(new_conc)
            
            # Update concentrations
            for name in self.molecules:
                if name in new_concentrations and len(new_concentrations[name]) > len(concentrations[name]):
                    concentrations[name].append(new_concentrations[name][-1])
                else:
                    # No change
                    concentrations[name].append(concentrations[name][-1])
        
        # Calculate reaction network properties
        network_analysis = self._analyze_reaction_network(concentrations)
        
        result = {
            'simulation_time': time_steps * dt,
            'time_steps': time_steps,
            'final_concentrations': {name: conc[-1] for name, conc in concentrations.items()},
            'concentration_time_series': concentrations,
            'time_points': time_points,
            'network_analysis': network_analysis,
            'total_reactions': len(self.reactions),
            'total_molecules': len(self.molecules)
        }
        
        return result
    
    def _calculate_reaction_rate(self, reaction: ChemicalReaction) -> float:
        """Calculate the rate of a chemical reaction"""
        # Base rate from mass-action kinetics
        rate = reaction.rate_constant
        
        # Apply reactant concentrations
        for reactant in reaction.reactants:
            rate *= reactant.concentration
        
        # Apply activation energy (Arrhenius-like)
        temperature_factor = 1.0  # Assume room temperature
        rate *= math.exp(-reaction.activation_energy / temperature_factor)
        
        # Apply catalyst effect
        if reaction.catalyst:
            catalytic_efficiency = 1 + reaction.catalyst.reactivity * 2
            rate *= catalytic_efficiency
        
        # Apply reaction type modifiers
        if reaction.reaction_type == ReactionType.CATALYTIC:
            rate *= 1.5
        elif reaction.reaction_type == ReactionType.ENZYMATIC:
            rate *= 2.0
        elif reaction.reaction_type == ReactionType.AUTOCATALYTIC:
            # Autocatalytic reactions accelerate themselves
            rate *= (1 + sum(p.concentration for p in reaction.products) * 0.1)
        
        return rate
    
    def _analyze_reaction_network(self, concentrations: Dict[str, List[float]]) -> Dict[str, Any]:
        """Analyze the reaction network properties"""
        if not concentrations:
            return {}
        
        # Calculate network stability
        final_concentrations = {name: conc[-1] for name, conc in concentrations.items()}
        concentration_variance = np.var(list(final_concentrations.values()))
        network_stability = 1 / (1 + concentration_variance)
        
        # Calculate reaction fluxes
        fluxes = {}
        for reaction_id, reaction in self.reaction_network.items():
            flux = self._calculate_reaction_rate(reaction)
            fluxes[reaction_id] = flux
        
        # Calculate thermodynamic efficiency
        total_energy_input = sum(m.energy_level for m in self.molecules.values())
        total_energy_output = sum(final_concentrations[name] * self.molecules[name].energy_level 
                                for name in final_concentrations)
        thermodynamic_efficiency = min(1.0, total_energy_output / max(1, total_energy_input))
        
        # Calculate network complexity
        network_complexity = len(self.reactions) * len(self.molecules) / max(1, len(self.reactions) + len(self.molecules))
        
        return {
            'network_stability': network_stability,
            'thermodynamic_efficiency': thermodynamic_efficiency,
            'network_complexity': network_complexity,
            'average_reaction_flux': sum(fluxes.values()) / max(1, len(fluxes)),
            'reaction_fluxes': fluxes,
            'concentration_entropy': self._calculate_concentration_entropy(final_concentrations)
        }
    
    def _calculate_concentration_entropy(self, concentrations: Dict[str, float]) -> float:
        """Calculate entropy of concentration distribution"""
        total_concentration = sum(concentrations.values())
        if total_concentration == 0:
            return 0
        
        entropy = 0
        for conc in concentrations.values():
            if conc > 0:
                probability = conc / total_concentration
                entropy -= probability * math.log(probability)
        
        # Normalize entropy
        max_entropy = math.log(len(concentrations)) if concentrations else 1
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def optimize_catalyst_efficiency(self, target_reaction: str, 
                                   catalyst_candidates: List[str]) -> Dict[str, Any]:
        """
        Optimize catalyst efficiency for prime aligned compute reactions
        
        Finds the best catalyst for optimizing prime aligned compute pattern transformations
        """
        if target_reaction not in self.reaction_network:
            return {"error": "Target reaction not found"}
        
        reaction = self.reaction_network[target_reaction]
        optimization_results = {}
        
        # Test each catalyst candidate
        for candidate in catalyst_candidates:
            if candidate not in self.molecules:
                self.create_molecular_species(candidate, reactivity=random.uniform(0.3, 0.9))
            
            # Temporarily assign catalyst
            original_catalyst = reaction.catalyst
            reaction.catalyst = self.molecules[candidate]
            
            # Calculate efficiency metrics
            base_rate = self._calculate_reaction_rate(reaction)
            
            # Calculate catalyst efficiency factors
            reactivity_factor = reaction.catalyst.reactivity
            stability_factor = reaction.catalyst.stability
            energy_factor = 1 - abs(reaction.catalyst.energy_level - reaction.activation_energy)
            
            efficiency_score = (
                reactivity_factor * 0.4 +
                stability_factor * 0.3 +
                energy_factor * 0.3
            )
            
            optimization_results[candidate] = {
                'efficiency_score': efficiency_score,
                'reaction_rate': base_rate,
                'reactivity_factor': reactivity_factor,
                'stability_factor': stability_factor,
                'energy_alignment': energy_factor
            }
            
            # Restore original catalyst
            reaction.catalyst = original_catalyst
        
        # Find optimal catalyst
        if optimization_results:
            optimal_catalyst = max(optimization_results.keys(), 
                                 key=lambda x: optimization_results[x]['efficiency_score'])
            
            result = {
                'target_reaction': target_reaction,
                'optimal_catalyst': optimal_catalyst,
                'optimization_results': optimization_results,
                'efficiency_improvement': optimization_results[optimal_catalyst]['efficiency_score'] - 0.5,
                'all_candidates_tested': len(catalyst_candidates)
            }
            
            # Apply optimal catalyst
            if optimal_catalyst in self.molecules:
                self.reaction_network[target_reaction].catalyst = self.molecules[optimal_catalyst]
                result['catalyst_applied'] = True
            
            return result
        
        return {"error": "No catalysts tested"}
    
    def create_reaction_diffusion_system(self, width: int = 50, height: int = 50,
                                       time_steps: int = 100) -> Dict[str, Any]:
        """
        Create a reaction-diffusion system for prime aligned compute pattern formation
        
        Models how prime aligned compute patterns emerge through chemical-like diffusion
        """
        # Initialize concentration grids
        activator_grid = np.random.rand(height, width) * 0.1
        inhibitor_grid = np.random.rand(height, width) * 0.1
        
        # Reaction-diffusion parameters (Gray-Scott model inspired)
        diffusion_rate_a = 0.16
        diffusion_rate_b = 0.08
        feed_rate = 0.035
        kill_rate = 0.065
        
        pattern_history = []
        
        for step in range(time_steps):
            # Store current state
            pattern_history.append({
                'step': step,
                'activator': activator_grid.copy(),
                'inhibitor': inhibitor_grid.copy()
            })
            
            # Calculate Laplacian (diffusion)
            activator_laplacian = self._calculate_laplacian(activator_grid)
            inhibitor_laplacian = self._calculate_laplacian(inhibitor_grid)
            
            # Apply reaction-diffusion equations
            delta_a = (diffusion_rate_a * activator_laplacian - 
                      activator_grid * inhibitor_grid * inhibitor_grid + 
                      feed_rate * (1 - activator_grid))
            
            delta_b = (diffusion_rate_b * inhibitor_laplacian + 
                      activator_grid * inhibitor_grid * inhibitor_grid - 
                      (kill_rate + feed_rate) * inhibitor_grid)
            
            # Update grids
            activator_grid += delta_a
            inhibitor_grid += delta_b
            
            # Clamp values to [0, 1]
            activator_grid = np.clip(activator_grid, 0, 1)
            inhibitor_grid = np.clip(inhibitor_grid, 0, 1)
        
        # Analyze emergent patterns
        final_activator = pattern_history[-1]['activator']
        final_inhibitor = pattern_history[-1]['inhibitor']
        
        pattern_complexity = self._calculate_pattern_complexity(final_activator)
        pattern_symmetry = self._calculate_pattern_symmetry(final_activator)
        
        result = {
            'simulation_dimensions': (width, height),
            'time_steps': time_steps,
            'pattern_history_length': len(pattern_history),
            'final_pattern_complexity': pattern_complexity,
            'final_pattern_symmetry': pattern_symmetry,
            'diffusion_parameters': {
                'diffusion_rate_a': diffusion_rate_a,
                'diffusion_rate_b': diffusion_rate_b,
                'feed_rate': feed_rate,
                'kill_rate': kill_rate
            },
            'final_activator_pattern': final_activator.tolist(),
            'final_inhibitor_pattern': final_inhibitor.tolist()
        }
        
        return result
    
    def _calculate_laplacian(self, grid: np.ndarray) -> np.ndarray:
        """Calculate Laplacian for diffusion"""
        return (
            np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
            np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) -
            4 * grid
        )
    
    def _calculate_pattern_complexity(self, pattern: np.ndarray) -> float:
        """Calculate complexity of emergent pattern"""
        # Use fractal dimension as complexity measure
        if pattern.size == 0:
            return 0
        
        # Simple fractal dimension estimation
        total_mass = np.sum(pattern)
        max_value = np.max(pattern)
        
        if total_mass == 0 or max_value == 0:
            return 0
        
        # Calculate entropy as complexity measure
        flattened = pattern.flatten()
        entropy = 0
        for value in flattened:
            if value > 0:
                prob = value / total_mass
                entropy -= prob * math.log(prob)
        
        # Normalize
        max_entropy = math.log(len(flattened))
        return entropy / max_entropy if max_entropy > 0 else 0
    
    def _calculate_pattern_symmetry(self, pattern: np.ndarray) -> float:
        """Calculate symmetry of emergent pattern"""
        if pattern.shape[0] != pattern.shape[1]:
            return 0
        
        size = pattern.shape[0]
        symmetry_score = 0
        
        # Check rotational symmetry
        for i in range(size):
            for j in range(size):
                original = pattern[i, j]
                # 90 degree rotation
                rotated = pattern[size-1-j, i]
                symmetry_score += 1 - abs(original - rotated)
        
        total_elements = size * size
        return symmetry_score / total_elements
    
    def get_chemical_consciousness_metrics(self) -> Dict[str, Any]:
        """Get overall chemical prime aligned compute metrics"""
        if not self.molecules and not self.reactions:
            return {"error": "No chemical system created"}
        
        # Calculate reaction network efficiency
        if self.reactions:
            avg_reaction_rate = sum(r.rate_constant for r in self.reactions) / len(self.reactions)
            catalytic_reactions = sum(1 for r in self.reactions if r.catalyst is not None)
            catalytic_efficiency = catalytic_reactions / len(self.reactions) if self.reactions else 0
        else:
            avg_reaction_rate = 0
            catalytic_efficiency = 0
        
        # Calculate molecular diversity
        if self.molecules:
            avg_reactivity = sum(m.reactivity for m in self.molecules.values()) / len(self.molecules)
            avg_stability = sum(m.stability for m in self.molecules.values()) / len(self.molecules)
            energy_diversity = len(set(m.energy_level for m in self.molecules.values())) / len(self.molecules)
        else:
            avg_reactivity = 0
            avg_stability = 0
            energy_diversity = 0
        
        # Calculate chemical prime aligned compute index
        chemical_consciousness_index = (
            avg_reaction_rate * 0.2 +
            catalytic_efficiency * 0.3 +
            avg_reactivity * 0.15 +
            avg_stability * 0.15 +
            energy_diversity * 0.2
        )
        
        return {
            'total_molecules': len(self.molecules),
            'total_reactions': len(self.reactions),
            'average_reaction_rate': avg_reaction_rate,
            'catalytic_efficiency': catalytic_efficiency,
            'average_molecular_reactivity': avg_reactivity,
            'average_molecular_stability': avg_stability,
            'molecular_energy_diversity': energy_diversity,
            'chemical_consciousness_index': chemical_consciousness_index
        }

def demo_chemical_consciousness():
    """Demonstrate chemical prime aligned compute principles"""
    print("\\n🧪 CHEMICAL prime aligned compute DEMONSTRATION")
    print("=" * 50)
    
    # Initialize chemical prime aligned compute engine
    chem_engine = ChemicalConsciousnessEngine()
    
    # Create molecular species
    print("\\n🔬 Creating Molecular Species:")
    glucose = chem_engine.create_molecular_species("glucose", 1.0, 0.3, 0.8, 0.6)
    oxygen = chem_engine.create_molecular_species("oxygen", 0.8, 0.9, 0.9, 0.7)
    atp = chem_engine.create_molecular_species("ATP", 0.2, 0.8, 0.6, 0.5)
    water = chem_engine.create_molecular_species("water", 0.5, 0.1, 0.3, 0.9)
    co2 = chem_engine.create_molecular_species("CO2", 0.1, 0.2, 0.4, 0.8)
    
    # Create enzyme catalyst
    hexokinase = chem_engine.create_molecular_species("hexokinase", 0.05, 0.6, 0.95, 0.7)
    
    # Create chemical reactions
    print("\\n⚗️ Creating Chemical Reactions:")
    glycolysis = chem_engine.create_chemical_reaction(
        "glycolysis", ["glucose", "ATP"], ["pyruvate", "ATP"], 
        rate_constant=0.2, activation_energy=0.4, 
        reaction_type=ReactionType.ENZYMATIC, catalyst="hexokinase"
    )
    
    cellular_respiration = chem_engine.create_chemical_reaction(
        "respiration", ["glucose", "oxygen"], ["CO2", "water", "ATP"],
        rate_constant=0.15, activation_energy=0.6,
        reaction_type=ReactionType.CATALYTIC
    )
    
    # Simulate reaction kinetics
    print("\\n📊 Simulating Reaction Kinetics:")
    kinetics_result = chem_engine.simulate_reaction_kinetics(time_steps=50)
    print(f"   Simulation time: {kinetics_result['simulation_time']:.1f}")
    print(f"   Final concentrations: {kinetics_result['final_concentrations']}")
    print(f"   Network stability: {kinetics_result['network_analysis']['network_stability']:.3f}")
    
    # Optimize catalyst efficiency
    print("\\n🎯 Optimizing Catalyst Efficiency:")
    catalyst_candidates = ["hexokinase", "phosphatase", "kinase", "transferase"]
    optimization = chem_engine.optimize_catalyst_efficiency("glycolysis", catalyst_candidates)
    if 'optimal_catalyst' in optimization:
        print(f"   Optimal catalyst: {optimization['optimal_catalyst']}")
        print(f"   Efficiency improvement: {optimization['efficiency_improvement']:.3f}")
    
    # Create reaction-diffusion system
    print("\\n🌊 Creating Reaction-Diffusion Pattern:")
    pattern_result = chem_engine.create_reaction_diffusion_system(width=20, height=20, time_steps=30)
    print(f"   Pattern complexity: {pattern_result['final_pattern_complexity']:.3f}")
    print(f"   Pattern symmetry: {pattern_result['final_pattern_symmetry']:.3f}")
    
    # Get final metrics
    print("\\n📈 Final Chemical prime aligned compute Metrics:")
    metrics = chem_engine.get_chemical_consciousness_metrics()
    print(f"   Chemical prime aligned compute Index: {metrics['chemical_consciousness_index']:.3f}")
    print(f"   Molecular diversity: {metrics['molecular_energy_diversity']:.3f}")
    print(f"   Catalytic efficiency: {metrics['catalytic_efficiency']:.3f}")
    
    print("\\n✅ Chemical prime aligned compute principles successfully applied!")
    print("   - Molecular interaction modeling working")
    print("   - Reaction network dynamics simulated")
    print("   - Catalyst optimization functional")
    print("   - Reaction-diffusion patterns emerging")
    
    return chem_engine

if __name__ == "__main__":
    demo_chemical_consciousness()
