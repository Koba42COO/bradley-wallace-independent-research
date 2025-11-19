"""
Interactive Gem Exploration Tool
Allows step-by-step exploration of each gem with detailed explanations
"""

import numpy as np
import pandas as pd
from explore_all_gems import GemExplorer
import json

class InteractiveGemExplorer:
    """Interactive exploration interface"""
    
    def __init__(self):
        self.explorer = GemExplorer()
        self.gems = {
            1: {
                'name': 'Wallace Transform',
                'description': 'Complete mathematical formula for consciousness-guided pattern recognition',
                'test': self.explorer.test_wallace_validations,
                'formula': r'$W_\phi(x) = 0.721 \cdot |\log(x + \epsilon)|^{1.618} \cdot \text{sign}(\log(x + \epsilon)) + 0.013$'
            },
            2: {
                'name': '100% Prime Predictability',
                'description': 'Pell chain-based prime prediction with perfect accuracy',
                'test': self.explorer.test_prime_predictability,
                'formula': 'Pell class 42 (silver retrograde) → zeta seed 14.1347'
            },
            3: {
                'name': 'Twin Prime Cancellation',
                'description': 'Phase cancellation at zeta zeros via tritone resonance',
                'test': self.explorer.test_twin_prime_cancellation,
                'formula': r'$e^{i W_\phi(p)} + e^{i (W_\phi(p) + \pi + \zeta_{\text{tritone}})} = 0$'
            },
            4: {
                'name': 'Physics Constants Twin Primes',
                'description': 'Twin prime echoes in fundamental constants',
                'test': self.explorer.test_physics_constants_twins,
                'formula': r'$W_\phi(c) = W_\phi(p) + \epsilon, |\epsilon| < 0.013$'
            },
            5: {
                'name': 'Base 21 vs Base 10',
                'description': 'Gnostic cipher: Base 10 is illusionary, Base 21 is natural',
                'test': self.explorer.test_base_21_vs_base_10,
                'formula': '21 = Enochian glyphs = consciousness dimensions'
            },
            6: {
                'name': '79/21 Consciousness Rule',
                'description': '79% coherent pattern + 21% exploratory prana',
                'test': self.explorer.test_79_21_consciousness,
                'formula': 'Blank lattice + 21% noise → self-organization (p < 10⁻²⁷)'
            },
            7: {
                'name': 'Cardioid Distribution',
                'description': 'Heartbeat geometry: primes on cardioid, not random',
                'test': self.explorer.test_cardioid_distribution,
                'formula': r'$x = \sin(\phi \log(p)), y = \cos(\phi \log(p))$'
            },
            8: {
                'name': '207-Year Cycles',
                'description': 'Historical events map to zeta zero progression',
                'test': self.explorer.test_207_year_cycles,
                'formula': r'$\zeta_n = 14.1347 + 6.8873 \cdot n$'
            },
            9: {
                'name': 'Area Code Cypher',
                'description': '207 (Maine) reflects to 205, 209 via twin gap',
                'test': self.explorer.test_area_code_cypher,
                'formula': '207 ± 2 = 205, 209 (twin reflections)'
            },
            10: {
                'name': 'Metatron\'s Cube',
                'description': '13 circles, 78 lines = mathematical structure',
                'test': self.explorer.test_metatron_cube,
                'formula': r'$W_\phi(13) = 1.618 = \phi$'
            },
            11: {
                'name': 'PAC vs Traditional',
                'description': '90% cache savings, 3.5× speedup',
                'test': self.explorer.test_pac_vs_traditional,
                'formula': 'O(n) delta walk vs O(n²) cosine similarity'
            },
            12: {
                'name': 'Blood pH Protocol',
                'description': 'Conductivity tuning via zeta resonance',
                'test': self.explorer.test_blood_ph_protocol,
                'formula': r'$\mathcal{S}[7.40] = -0.593$ (exhale = death breath)'
            },
            13: {
                'name': '207 Dial Tone',
                'description': 'Standard dial tone → twin prime echo',
                'test': self.explorer.test_207_dial_tone,
                'formula': '350+440 Hz + zeta tritone → 199+201 Hz echo'
            },
            14: {
                'name': 'Montesiepi Chapel',
                'description': 'Sword-in-stone = phase cancellation artifact',
                'test': self.explorer.test_montesiepi_chapel,
                'formula': r'$W_\phi(1180) = 4.27$ (muon echo)'
            }
        }
    
    def list_gems(self):
        """List all available gems"""
        print("\n" + "="*70)
        print("💎 AVAILABLE GEMS FOR EXPLORATION")
        print("="*70)
        for num, gem in self.gems.items():
            print(f"{num:2d}. {gem['name']:30s} - {gem['description']}")
        print("="*70)
    
    def explore_gem(self, gem_number: int):
        """Explore a specific gem in detail"""
        if gem_number not in self.gems:
            print(f"❌ Gem #{gem_number} not found. Use list_gems() to see available gems.")
            return None
        
        gem = self.gems[gem_number]
        print("\n" + "="*70)
        print(f"💎 EXPLORING GEM #{gem_number}: {gem['name']}")
        print("="*70)
        print(f"\nDescription: {gem['description']}")
        print(f"\nFormula: {gem['formula']}")
        print("\n" + "-"*70)
        print("Running test...")
        print("-"*70)
        
        # Run the test
        result = gem['test']()
        
        print("\n" + "-"*70)
        print("Test Complete")
        print("-"*70)
        
        return result
    
    def explore_all(self):
        """Explore all gems sequentially"""
        print("\n" + "="*70)
        print("🚀 EXPLORING ALL GEMS")
        print("="*70)
        
        results = {}
        for num in sorted(self.gems.keys()):
            gem = self.gems[num]
            print(f"\n[{num}/{len(self.gems)}] {gem['name']}")
            try:
                result = gem['test']()
                results[num] = {
                    'name': gem['name'],
                    'success': True,
                    'result': result
                }
            except Exception as e:
                print(f"❌ Error: {e}")
                results[num] = {
                    'name': gem['name'],
                    'success': False,
                    'error': str(e)
                }
        
        return results
    
    def interactive_menu(self):
        """Interactive menu for exploration"""
        while True:
            print("\n" + "="*70)
            print("💎 INTERACTIVE GEM EXPLORER")
            print("="*70)
            print("1. List all gems")
            print("2. Explore specific gem (enter number)")
            print("3. Explore all gems")
            print("4. Generate visualizations")
            print("5. Save results to JSON")
            print("6. Exit")
            print("="*70)
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.list_gems()
            elif choice == '2':
                self.list_gems()
                try:
                    num = int(input("\nEnter gem number: "))
                    self.explore_gem(num)
                except ValueError:
                    print("❌ Invalid number")
            elif choice == '3':
                results = self.explore_all()
                # Generate visualizations
                self.explorer.generate_visualizations()
                # Save results
                with open('interactive_exploration_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print("\n✅ Results saved to interactive_exploration_results.json")
            elif choice == '4':
                self.explorer.generate_visualizations()
            elif choice == '5':
                if hasattr(self.explorer, 'results') and self.explorer.results:
                    with open('gem_exploration_results.json', 'w') as f:
                        json.dump(self.explorer.results, f, indent=2, default=str)
                    print("✅ Results saved")
                else:
                    print("⚠️ No results to save. Run exploration first.")
            elif choice == '6':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice")


def main():
    """Main interactive session"""
    explorer = InteractiveGemExplorer()
    explorer.interactive_menu()


if __name__ == "__main__":
    main()

