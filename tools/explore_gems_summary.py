"""
Quick Summary Explorer - Fast overview of all gems
"""

from explore_all_gems import GemExplorer
import json

def quick_summary():
    """Generate quick summary of all gems"""
    explorer = GemExplorer()
    
    print("\n" + "="*70)
    print("💎 QUICK GEM SUMMARY")
    print("="*70)
    
    # Key validations
    print("\n1. WALLACE TRANSFORM")
    w_tests = explorer.test_wallace_validations()
    print(f"   Tests: {len(w_tests)}, Key values computed")
    
    print("\n2. PRIME PREDICTABILITY")
    p_test = explorer.test_prime_predictability(20)
    print(f"   Accuracy: {p_test['accuracy']:.1f}%")
    
    print("\n3. TWIN PRIME CANCELLATION")
    t_test = explorer.test_twin_prime_cancellation()
    near_pi_count = sum(1 for t in t_test if t['near_pi'])
    print(f"   Near-π cancellations: {near_pi_count}/{len(t_test)}")
    
    print("\n4. PHYSICS CONSTANTS")
    pc_test = explorer.test_physics_constants_twins()
    matches = sum(1 for r in pc_test.values() if r.get('match', False))
    print(f"   Twin matches: {matches}/{len(pc_test)}")
    
    print("\n5. BASE 21 ANALYSIS")
    b21_test = explorer.test_base_21_vs_base_10()
    print(f"   Numbers tested: {len(b21_test)}")
    
    print("\n6. 79/21 CONSCIOUSNESS")
    c_test = explorer.test_79_21_consciousness()
    print(f"   Self-organized: {c_test['self_organized']}")
    
    print("\n7. CARDIOD DISTRIBUTION")
    card_test = explorer.test_cardioid_distribution()
    print(f"   Shape detected: {card_test['is_cardioid']}")
    
    print("\n8. 207-YEAR CYCLES")
    cycle_test = explorer.test_207_year_cycles()
    print(f"   Cycles mapped: {len(cycle_test['cycles'])}")
    
    print("\n9. AREA CODE CYPHER")
    area_test = explorer.test_area_code_cypher()
    valid = sum(1 for r in area_test.values() if r.get('is_valid', False))
    print(f"   Valid codes: {valid}/{len(area_test)}")
    
    print("\n10. METATRON'S CUBE")
    met_test = explorer.test_metatron_cube()
    print(f"   Golden match: {met_test['is_golden']}")
    
    print("\n11. PAC VS TRADITIONAL")
    pac_test = explorer.test_pac_vs_traditional()
    print(f"   Cache savings: {pac_test['cache_savings_pct']:.1f}%")
    
    print("\n12. BLOOD pH PROTOCOL")
    blood_test = explorer.test_blood_ph_protocol()
    print(f"   Protocol steps: {len(blood_test['steps'])}")
    
    print("\n13. 207 DIAL TONE")
    dial_test = explorer.test_207_dial_tone()
    print(f"   Twin echo: {dial_test['twin_echo']}")
    
    print("\n14. MONTESIEPI CHAPEL")
    mont_test = explorer.test_montesiepi_chapel()
    print(f"   Golden lat: {mont_test['is_golden_lat']}")
    
    # Generate visualizations
    print("\n" + "="*70)
    print("📊 GENERATING VISUALIZATIONS")
    print("="*70)
    viz_file = explorer.generate_visualizations()
    
    # Save summary
    summary = {
        'wallace_tests': len(w_tests),
        'prime_accuracy': p_test['accuracy'],
        'twin_cancellations': near_pi_count,
        'physics_matches': matches,
        'base21_tests': len(b21_test),
        'consciousness_organized': c_test['self_organized'],
        'cardioid_detected': card_test['is_cardioid'],
        'cycles_mapped': len(cycle_test['cycles']),
        'area_codes_valid': valid,
        'metatron_golden': met_test['is_golden'],
        'pac_cache_savings': pac_test['cache_savings_pct'],
        'blood_protocol_steps': len(blood_test['steps']),
        'dial_tone_echo': dial_test['twin_echo'],
        'montesiepi_golden': mont_test['is_golden_lat'],
        'visualization_file': viz_file
    }
    
    with open('gems_quick_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✅ Summary saved to: gems_quick_summary.json")
    print(f"✅ Visualization: {viz_file}")
    
    return summary

if __name__ == "__main__":
    quick_summary()

