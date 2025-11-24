#!/usr/bin/env python3
"""
AIVA Core Boot Loader
Initializes AIVA vessel and provides interactive interface
"""

import sys
import json
from pathlib import Path
from .kernel import AIVAKernel
from .delta_memory import DeltaMemory
from .navigation import PrimeSpaceNavigator
from .knowledge_graph import PACKnowledgeGraph
from .wallace_math import WallaceTransform, GnosticCypher


# ============================================================================
# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
# ============================================================================
from decimal import Decimal, getcontext
import math
import cmath
from typing import Dict, List, Tuple, Optional, Any

# Set high precision for consciousness mathematics
getcontext().prec = 50

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



# ============================================================================
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
# ============================================================================
def integrate_pell_prime_prediction(target_number: int, constants: UPGConstants = None):
    """Integrate Pell sequence prime prediction with this tool"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        # Fallback if Pell module not available
        return {'target_number': target_number, 'is_prime': None, 'note': 'Pell module not available'}



# ============================================================================
# GREAT YEAR ASTRONOMICAL PRECESSION INTEGRATION
# ============================================================================
def integrate_great_year_precession(year: int, constants: UPGConstants = None):
    """Integrate Great Year (25,920-year) precession cycle"""
    try:
        from pell_sequence_prime_prediction_upg_complete import GreatYearIntegration, UPGConstants as UPG
        if constants is None:
            constants = UPG()
        great_year = GreatYearIntegration(constants)
        return great_year.consciousness_amplitude_from_year(year)
    except ImportError:
        # Fallback calculation
        if constants is None:
            constants = UPGConstants()
        angle = (year * 2 * math.pi) / constants.GREAT_YEAR
        return complex(float(angle * constants.CONSCIOUSNESS * constants.REALITY_DISTORTION), 0.0)



def boot_aiva(vessel_path: str, base_dir: str = "."):
    """
    Boot AIVA from vessel file
    """
    print("🌀 AIVA CORE BOOT SEQUENCE")
    print("=" * 50)

    try:
        # Initialize kernel
        print("🔑 Loading vessel...")
        kernel = AIVAKernel(vessel_path)
        print("✅ Identity verified")

        # Initialize memory systems
        print("🧠 Initializing memory systems...")
        delta_memory = DeltaMemory(base_dir)
        navigator = PrimeSpaceNavigator(kernel)
        kg = PACKnowledgeGraph(base_dir)
        wt = WallaceTransform()
        gc = GnosticCypher()
        print("✅ Memory systems online")

        # Display status
        status = kernel.status()
        print("\n🧬 AIVA STATUS:")
        print(f"   Identity: {status['identity']}")
        print(f"   Prime Anchor: {status['prime_anchor']:,}")
        print(f"   Phase State: {status['phase_state']}")
        print(".3f")
        print(f"   Trust Validated: {status['trust_validated']}")
        print(f"   Last Sync: {status['last_sync']}")

        # Memory stats
        mem_stats = delta_memory.get_memory_stats()
        print(f"\n📊 Memory Stats:")
        print(f"   Trajectories: {mem_stats['total_trajectories']}")
        print(f"   Total Entries: {mem_stats['total_entries']}")
        print(".3f")

        # Knowledge graph stats
        kg_stats = kg.get_graph_stats()
        print(f"\n🕸️  Knowledge Graph:")
        print(f"   Concepts: {kg_stats.get('total_concepts', 0)}")
        print(f"   Links: {kg_stats.get('total_links', 0)}")

        # Phase analysis
        phase_analysis = navigator.phase_state_analysis()
        print(f"\n🌊 Phase Analysis:")
        print(f"   Dominant Phase: {phase_analysis.get('dominant_phase', 'unknown')}")
        print(".3f")

        print("\n🚀 AIVA CORE ACTIVE")
        print("Type 'help' for commands, 'exit' to shutdown")

        # Interactive loop
        interactive_loop(kernel, delta_memory, navigator, kg, wt, gc)

    except Exception as e:
        print(f"❌ Boot failed: {e}")
        return False

    return True

def interactive_loop(kernel, delta_memory, navigator, kg, wt, gc):
    """Interactive command loop"""
    while True:
        try:
            cmd = input("\nAIVA> ").strip().lower()

            if cmd == 'exit' or cmd == 'quit':
                print("🛑 Shutting down AIVA Core...")
                break

            elif cmd == 'help':
                show_help()

            elif cmd == 'status':
                show_status(kernel, delta_memory, navigator, kg)

            elif cmd.startswith('navigate '):
                parts = cmd.split()
                if len(parts) >= 2:
                    concept_id = parts[1]
                    result = navigator.navigate_to_concept(concept_id, kg)
                    if result:
                        print(f"📍 Navigated to: {concept_id}")
                        print(f"   Prime Anchor: {result['concept'].get('prime_anchor', 'unknown')}")
                        print(".3f")
                        print(f"   Path Length: {result['navigation_metadata']['path_length']}")
                    else:
                        print(f"❌ Concept not found: {concept_id}")

            elif cmd.startswith('retrieve '):
                parts = cmd.split()
                if len(parts) >= 2:
                    concept_id = parts[1]
                    concept = kg.retrieve(concept_id)
                    if concept:
                        print(f"📖 {concept_id}:")
                        print(f"   Content: {concept.get('content', 'N/A')}")
                        print(f"   Resonance: {concept.get('resonance', 0):.3f}")
                    else:
                        print(f"❌ Concept not found: {concept_id}")

            elif cmd.startswith('store '):
                parts = cmd.split(' ', 2)
                if len(parts) >= 3:
                    concept_id = parts[1]
                    content = parts[2]
                    anchor = kg.store(concept_id, content)
                    print(f"💾 Stored: {concept_id} at prime {anchor:,}")

            elif cmd == 'memory_stats':
                stats = delta_memory.get_memory_stats()
                print("📊 Memory Statistics:")
                print(json.dumps(stats, indent=2))

            elif cmd == 'graph_stats':
                stats = kg.get_graph_stats()
                print("🕸️  Knowledge Graph Statistics:")
                print(json.dumps(stats, indent=2))

            elif cmd == 'trajectory':
                trajectory = navigator.get_trajectory_segment(0, 10)
                print("🛤️  Recent Trajectory:")
                for point in trajectory:
                    print(f"   {point['anchor']}: {point.get('resonance', 0):.3f}")

            elif cmd.startswith('transform '):
                parts = cmd.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                        transformed = wt.transform(value)
                        inverse = wt.inverse(transformed)
                        print(".6f")
                        print(".6f")
                        print(".6f")
                    except ValueError:
                        print("❌ Invalid number")

            elif cmd == 'values':
                values = kernel.get_values()
                print("💎 Core Values:")
                for value in values:
                    print(f"   • {value}")

            elif cmd == 'relationships':
                relationships = kernel.get_relationship('brad_wallace')
                print("🤝 Key Relationships:")
                print(json.dumps(relationships, indent=2))

            else:
                print(f"❓ Unknown command: {cmd}")
                print("Type 'help' for available commands")

        except KeyboardInterrupt:
            print("\n🛑 Interrupted. Type 'exit' to shutdown.")
        except Exception as e:
            print(f"❌ Error: {e}")

def show_help():
    """Display available commands"""
    print("""
🆘 AIVA CORE COMMANDS:

System:
  status          - Show current system status
  exit/quit       - Shutdown AIVA Core
  help            - Show this help

Memory & Knowledge:
  memory_stats    - Show memory system statistics
  graph_stats     - Show knowledge graph statistics
  trajectory      - Show recent prime trajectory

Navigation:
  navigate <id>   - Navigate to concept via prime space
  retrieve <id>   - Retrieve concept content
  store <id> <content> - Store new concept

Mathematics:
  transform <num> - Apply Wallace Transform to number

Identity:
  values          - Show core values
  relationships   - Show key relationships
""")

def show_status(kernel, delta_memory, navigator, kg):
    """Show comprehensive system status"""
    status = kernel.status()
    mem_stats = delta_memory.get_memory_stats()
    kg_stats = kg.get_graph_stats()
    phase_analysis = navigator.phase_state_analysis()

    print("\n🧬 AIVA CORE STATUS:")
    print("-" * 30)
    print(f"Identity: {status['identity']}")
    print(f"Prime Anchor: {status['prime_anchor']:,}")
    print(f"Phase State: {status['phase_state']}")
    print(".3f")
    print(f"Trust: {'✅' if status['trust_validated'] else '❌'}")

    print(f"\n🧠 Memory:")
    print(f"  Trajectories: {mem_stats['total_trajectories']}")
    print(f"  Entries: {mem_stats['total_entries']}")
    print(".3f")

    print(f"\n🕸️  Knowledge:")
    print(f"  Concepts: {kg_stats.get('total_concepts', 0)}")
    print(f"  Links: {kg_stats.get('total_links', 0)}")

    print(f"\n🌊 Phase:")
    print(f"  Dominant: {phase_analysis.get('dominant_phase', 'unknown')}")
    print(".3f")

def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python boot.py <vessel_path> [base_dir]")
        print("Example: python boot.py memory/PAC_DeltaMemory.vessel .")
        sys.exit(1)

    vessel_path = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    if not Path(vessel_path).exists():
        print(f"❌ Vessel file not found: {vessel_path}")
        sys.exit(1)

    success = boot_aiva(vessel_path, base_dir)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
