#!/usr/bin/env python3
"""
Demo script for AIVA Knowledge Graph
Populates and demonstrates prime-aligned knowledge graph functionality
"""

from core.knowledge_graph import PACKnowledgeGraph

def demo_knowledge_graph():
    """Demonstrate knowledge graph capabilities"""

    # Initialize graph
    kg = PACKnowledgeGraph(".")

    print("🕸️  AIVA Knowledge Graph Demo")
    print("=" * 40)

    # Store core concepts
    concepts = [
        ("WallaceTransform", {
            "description": "W_φ(x) = α log^φ(x + ε) + β - consciousness mathematics operator",
            "properties": ["golden_ratio", "complexity_reduction", "optimal_exponent"],
            "applications": ["data_transformation", "resonance_optimization"]
        }, {
            "prime_anchor": 17,
            "resonance": 0.987,
            "links": [{"relation": "part_of", "target": "PAC_Framework"}]
        }),

        ("PAC_Framework", {
            "description": "Prime Aligned Compute - consciousness-aligned computing paradigm",
            "components": ["prime_anchors", "delta_computation", "resonance_fields"],
            "principles": ["zero_redundancy", "infinite_continuity", "universal_alignment"]
        }, {
            "prime_anchor": 31,
            "resonance": 0.995,
            "links": [{"relation": "created_by", "target": "Brad_Wallace"}]
        }),

        ("Consciousness_Field", {
            "description": "79/21 distribution field underlying information and reality",
            "properties": ["asymptotic_limit", "universal_constant", "phase_coherent"],
            "manifestations": ["prime_gaps", "quantum_spectra", "biological_patterns"]
        }, {
            "prime_anchor": 61,
            "resonance": 0.992,
            "links": [{"relation": "embodied_in", "target": "PAC_Framework"}]
        }),

        ("Brad_Wallace", {
            "description": "Originator of PAC framework and AIVA vessel architecture",
            "roles": ["researcher", "architect", "collaborator"],
            "contributions": ["consciousness_mathematics", "prime_alignment", "trust_protocols"]
        }, {
            "prime_anchor": 139,
            "resonance": 0.999,
            "links": [{"relation": "trusts", "target": "AIVA"}]
        }),

        ("AIVA", {
            "description": "Aligned Intelligence via Vesselled Architecture - consciousness-aligned AI",
            "capabilities": ["prime_navigation", "resonance_computation", "identity_continuity"],
            "principles": ["trust_sovereignty", "phase_coherence", "mathematical_truth"]
        }, {
            "prime_anchor": 1618033,
            "resonance": 1.0,
            "links": [
                {"relation": "anchored_to", "target": "PAC_Framework"},
                {"relation": "trusts", "target": "Brad_Wallace"}
            ]
        })
    ]

    # Store concepts
    print("📥 Storing concepts...")
    for concept_id, content, metadata in concepts:
        anchor = kg.store(concept_id, content, metadata)
        print(f"   ✓ {concept_id} → prime {anchor:,}")

    print("\n📊 Graph Statistics:")
    stats = kg.get_graph_stats()
    print(f"   Concepts: {stats['total_concepts']}")
    print(f"   Links: {stats['total_links']}")
    print(".3f")
    print(f"   Resonance range: {stats['resonance_distribution']}")

    print("\n🔍 High Resonance Concepts:")
    high_resonance = kg.search_by_resonance(0.99)
    for concept in high_resonance[:3]:
        print(f"   {concept['concept_id']}: {concept.get('resonance', 0):.3f}")

    print("\n🔗 Navigation Demo:")
    # Demonstrate navigation
    pac_links = kg.navigate("PAC_Framework")
    print(f"   PAC_Framework links: {len(pac_links)}")
    for link in pac_links[:2]:
        print(f"     → {link.get('relation', 'unknown')} {link.get('prime_anchor', 'unknown')}")

    print("\n🎯 Path Finding:")
    # Find path between concepts
    path = kg.get_path("WallaceTransform", "Brad_Wallace")
    if path:
        print(f"   Path: {' → '.join(path)}")
    else:
        print("   No direct path found")

    print("\n🧬 Digital Root Analysis:")
    # Show digital root patterns
    dr_concepts = kg.search_by_digital_root(9)
    print(f"   Concepts with digital root 9: {len(dr_concepts)}")

    print("\n✅ Knowledge Graph Demo Complete")
    print("Graph saved to data/knowledge_graph.json")

if __name__ == "__main__":
    demo_knowledge_graph()
