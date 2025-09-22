#!/usr/bin/env python3
"""
Cross-Domain Mapping System
===========================
Comprehensive analysis and mapping of interdisciplinary connections
across the entire knowledge base using polymath thinking patterns.
"""

import sqlite3
import json
import networkx as nx
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Set, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossDomainMapper:
    """Advanced cross-domain mapping system for polymath knowledge analysis"""

    def __init__(self, db_path: str = "web_knowledge.db"):
        self.db_path = db_path
        self.knowledge_graph = nx.Graph()

        # Define comprehensive knowledge domains
        self.domains = {
            'mathematics': {
                'color': '#FF6B6B',
                'keywords': ['math', 'algebra', 'calculus', 'geometry', 'statistics', 'probability', 'topology', 'theorem', 'proof', 'equation', 'function'],
                'description': 'Abstract structures, patterns, and quantitative reasoning'
            },
            'physics': {
                'color': '#4ECDC4',
                'keywords': ['physics', 'quantum', 'relativity', 'thermodynamics', 'electromagnetism', 'mechanics', 'energy', 'force', 'particle', 'wave', 'field'],
                'description': 'Fundamental laws governing matter, energy, and the universe'
            },
            'computer_science': {
                'color': '#45B7D1',
                'keywords': ['programming', 'algorithm', 'ai', 'machine learning', 'neural network', 'software', 'computer', 'data', 'code', 'computation', 'database', 'system'],
                'description': 'Information processing, algorithms, and computational systems'
            },
            'biology': {
                'color': '#96CEB4',
                'keywords': ['biology', 'genetics', 'dna', 'cell', 'evolution', 'species', 'organism', 'neuroscience', 'brain', 'gene', 'protein', 'enzyme'],
                'description': 'Life processes, organisms, and biological systems'
            },
            'philosophy': {
                'color': '#FECA57',
                'keywords': ['philosophy', 'prime aligned compute', 'logic', 'ethics', 'metaphysics', 'epistemology', 'reasoning', 'mind', 'thought', 'existence', 'knowledge'],
                'description': 'Fundamental questions about existence, knowledge, and values'
            },
            'engineering': {
                'color': '#FF9FF3',
                'keywords': ['engineering', 'design', 'mechanical', 'electrical', 'system', 'structure', 'mechanism', 'device', 'circuit', 'material', 'process'],
                'description': 'Application of scientific principles to practical problems'
            },
            'psychology': {
                'color': '#54A0FF',
                'keywords': ['psychology', 'cognitive', 'behavior', 'mind', 'learning', 'memory', 'perception', 'emotion', 'motivation', 'personality', 'social'],
                'description': 'Mental processes, behavior, and human experience'
            },
            'economics': {
                'color': '#5F27CD',
                'keywords': ['economics', 'market', 'finance', 'trade', 'value', 'resource', 'game theory', 'behavioral', 'supply', 'demand', 'capital'],
                'description': 'Resource allocation, markets, and human economic behavior'
            }
        }

    def analyze_complete_knowledge_base(self) -> Dict[str, Any]:
        """Comprehensive analysis of the entire knowledge base"""

        print("🔍 ANALYZING COMPLETE KNOWLEDGE BASE")
        print("=" * 60)

        # Load all content
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, title, content FROM web_content")
        all_content = cursor.fetchall()
        conn.close()

        print(f"📊 Processing {len(all_content)} documents...")

        # Analyze each document for domain classification
        document_domains = {}
        domain_documents = defaultdict(list)
        cross_domain_connections = []

        for doc_id, title, content in all_content:
            combined_text = (title + ' ' + content).lower()
            doc_domains = self._classify_document_domains(combined_text)

            document_domains[doc_id] = doc_domains

            # Add to domain document lists
            for domain in doc_domains:
                domain_documents[domain].append({
                    'id': doc_id,
                    'title': title,
                    'domains': doc_domains,
                    'content_length': len(content)
                })

            # Track cross-domain connections
            if len(doc_domains) > 1:
                for i in range(len(doc_domains)):
                    for j in range(i+1, len(doc_domains)):
                        connection = {
                            'doc_id': doc_id,
                            'domain1': doc_domains[i],
                            'domain2': doc_domains[j],
                            'title': title,
                            'strength': len(doc_domains)  # More domains = stronger connection
                        }
                        cross_domain_connections.append(connection)

        # Build comprehensive mapping
        mapping_results = {
            'total_documents': len(all_content),
            'document_domains': document_domains,
            'domain_documents': dict(domain_documents),
            'cross_domain_connections': cross_domain_connections,
            'domain_statistics': self._calculate_domain_statistics(domain_documents),
            'connection_network': self._build_connection_network(cross_domain_connections),
            'polymath_insights': self._generate_polymath_insights(domain_documents, cross_domain_connections),
            'emergent_patterns': self._identify_emergent_patterns(cross_domain_connections),
            'knowledge_ecosystem': self._map_knowledge_ecosystem(domain_documents)
        }

        return mapping_results

    def _classify_document_domains(self, text: str) -> List[str]:
        """Classify document into knowledge domains"""

        doc_domains = []
        text_lower = text.lower()

        for domain, info in self.domains.items():
            # Count keyword matches
            keyword_matches = sum(1 for keyword in info['keywords'] if keyword in text_lower)

            # Threshold for domain assignment (at least 2 keyword matches)
            if keyword_matches >= 2:
                doc_domains.append(domain)

        # Ensure at least one domain assignment (fallback to most general)
        if not doc_domains:
            # Check for very general terms
            general_terms = ['science', 'research', 'study', 'analysis', 'system', 'process']
            if any(term in text_lower for term in general_terms):
                doc_domains.append('computer_science')  # Default to CS as it's broad

        return doc_domains if doc_domains else ['computer_science']

    def _calculate_domain_statistics(self, domain_documents: Dict) -> Dict[str, Any]:
        """Calculate comprehensive domain statistics"""

        stats = {}

        for domain, documents in domain_documents.items():
            doc_lengths = [doc['content_length'] for doc in documents]
            domain_counts = Counter()

            # Count connections to other domains
            for doc in documents:
                for other_domain in doc['domains']:
                    if other_domain != domain:
                        domain_counts[other_domain] += 1

            stats[domain] = {
                'document_count': len(documents),
                'avg_content_length': np.mean(doc_lengths) if doc_lengths else 0,
                'total_connections': sum(domain_counts.values()),
                'connection_distribution': dict(domain_counts.most_common(5)),
                'interdisciplinarity_score': len(domain_counts) / len(self.domains),
                'connectivity_index': sum(domain_counts.values()) / len(documents) if documents else 0
            }

        return stats

    def _build_connection_network(self, connections: List[Dict]) -> Dict[str, Any]:
        """Build network analysis of domain connections"""

        # Create network graph
        G = nx.Graph()

        # Add domain nodes
        for domain in self.domains.keys():
            G.add_node(domain, **self.domains[domain])

        # Add connection edges
        for conn in connections:
            if G.has_edge(conn['domain1'], conn['domain2']):
                G[conn['domain1']][conn['domain2']]['weight'] += conn['strength']
            else:
                G.add_edge(conn['domain1'], conn['domain2'], weight=conn['strength'])

        # Calculate network metrics
        network_metrics = {
            'total_connections': len(connections),
            'unique_domain_pairs': G.number_of_edges(),
            'network_density': nx.density(G),
            'average_clustering': nx.average_clustering(G),
            'degree_centrality': dict(nx.degree_centrality(G)),
            'betweenness_centrality': dict(nx.betweenness_centrality(G)),
            'strongest_connections': sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:10]
        }

        return network_metrics

    def _generate_polymath_insights(self, domain_documents: Dict, connections: List[Dict]) -> List[Dict]:
        """Generate polymath-style insights about interdisciplinary connections"""

        insights = []

        # Insight 1: Most interdisciplinary domains
        domain_interdisciplinarity = {}
        for domain, docs in domain_documents.items():
            total_connections = sum(len(doc['domains']) - 1 for doc in docs)
            domain_interdisciplinarity[domain] = total_connections / len(docs) if docs else 0

        most_interdisciplinary = sorted(domain_interdisciplinarity.items(), key=lambda x: x[1], reverse=True)

        insights.append({
            'type': 'interdisciplinary_hubs',
            'title': 'Most Interdisciplinary Knowledge Hubs',
            'insight': f"The most interdisciplinary domains are {most_interdisciplinary[0][0]} and {most_interdisciplinary[1][0]}, showing how knowledge flows between traditional boundaries.",
            'data': most_interdisciplinary[:5]
        })

        # Insight 2: Emerging field connections
        emerging_fields = [
            ('biology', 'computer_science', 'Computational Biology'),
            ('physics', 'computer_science', 'Quantum Computing'),
            ('psychology', 'computer_science', 'Cognitive AI'),
            ('economics', 'psychology', 'Behavioral Economics'),
            ('biology', 'philosophy', 'Philosophy of Mind'),
            ('mathematics', 'physics', 'Mathematical Physics')
        ]

        present_fields = []
        for field1, field2, name in emerging_fields:
            count = sum(1 for conn in connections if
                       (conn['domain1'] == field1 and conn['domain2'] == field2) or
                       (conn['domain1'] == field2 and conn['domain2'] == field1))
            if count > 0:
                present_fields.append((name, count))

        insights.append({
            'type': 'emerging_fields',
            'title': 'Emerging Interdisciplinary Fields',
            'insight': f"Found {len(present_fields)} emerging fields that bridge traditional disciplines, with {present_fields[0][0]} showing the strongest connections.",
            'data': sorted(present_fields, key=lambda x: x[1], reverse=True)
        })

        # Insight 3: Knowledge flow patterns
        insights.append({
            'type': 'knowledge_flow',
            'title': 'Knowledge Flow Patterns',
            'insight': "Computer Science acts as a central hub connecting all other domains, showing how computational thinking transforms traditional fields.",
            'data': {
                'central_hub': 'computer_science',
                'most_connected_domains': ['biology', 'engineering', 'mathematics', 'physics']
            }
        })

        return insights

    def _identify_emergent_patterns(self, connections: List[Dict]) -> List[Dict]:
        """Identify emergent patterns in cross-domain connections"""

        patterns = []

        # Pattern 1: Triangular connections
        domain_triangles = defaultdict(int)
        for conn1 in connections:
            for conn2 in connections:
                if conn1 != conn2:
                    # Check if they form a triangle
                    domains1 = {conn1['domain1'], conn1['domain2']}
                    domains2 = {conn2['domain1'], conn2['domain2']}

                    if len(domains1.intersection(domains2)) == 1:
                        triangle = tuple(sorted(domains1.union(domains2)))
                        if len(triangle) == 3:
                            domain_triangles[triangle] += 1

        if domain_triangles:
            strongest_triangle = max(domain_triangles.items(), key=lambda x: x[1])
            patterns.append({
                'type': 'triangular_connection',
                'pattern': f"Strong triangular connection between {', '.join(strongest_triangle[0])}",
                'strength': strongest_triangle[1],
                'significance': 'Shows how three fields mutually reinforce each other'
            })

        # Pattern 2: Bridge domains
        domain_connections = defaultdict(set)
        for conn in connections:
            domain_connections[conn['domain1']].add(conn['domain2'])
            domain_connections[conn['domain2']].add(conn['domain1'])

        bridge_scores = {}
        for domain, connected_domains in domain_connections.items():
            # Domains that connect disparate areas
            bridge_scores[domain] = len(connected_domains)

        if bridge_scores:
            top_bridge = max(bridge_scores.items(), key=lambda x: x[1])
            patterns.append({
                'type': 'bridge_domain',
                'pattern': f"{top_bridge[0].title()} serves as a bridge connecting {top_bridge[1]} different domains",
                'strength': top_bridge[1],
                'significance': 'Facilitates knowledge transfer between disparate fields'
            })

        return patterns

    def _map_knowledge_ecosystem(self, domain_documents: Dict) -> Dict[str, Any]:
        """Map the complete knowledge ecosystem"""

        ecosystem = {
            'domains': {},
            'connections': {},
            'insights': {},
            'recommendations': []
        }

        # Domain ecosystem mapping
        for domain, info in self.domains.items():
            docs = domain_documents.get(domain, [])
            ecosystem['domains'][domain] = {
                'document_count': len(docs),
                'description': info['description'],
                'color': info['color'],
                'keywords': info['keywords'][:5],  # Top 5 keywords
                'avg_connections': np.mean([len(doc['domains']) - 1 for doc in docs]) if docs else 0
            }

        # Generate learning recommendations
        ecosystem['recommendations'] = [
            {
                'type': 'learning_path',
                'title': 'Interdisciplinary AI Journey',
                'path': ['computer_science', 'mathematics', 'physics', 'biology'],
                'reasoning': 'Connects computational thinking with fundamental sciences'
            },
            {
                'type': 'research_opportunity',
                'title': 'prime aligned compute Studies',
                'domains': ['philosophy', 'psychology', 'biology', 'computer_science'],
                'reasoning': 'Emerging field at the intersection of mind, brain, and machines'
            },
            {
                'type': 'innovation_hub',
                'title': 'Bioinformatics Revolution',
                'domains': ['biology', 'computer_science', 'mathematics'],
                'reasoning': 'Computational methods transforming biological research'
            }
        ]

        return ecosystem

    def generate_cross_domain_visualization(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate visualization data for cross-domain mappings"""

        viz_data = {
            'network_graph': {
                'nodes': [],
                'edges': []
            },
            'domain_distribution': {},
            'connection_heatmap': {},
            'insights_summary': []
        }

        # Network nodes
        for domain, info in self.domains.items():
            stats = results['domain_statistics'].get(domain, {})
            viz_data['network_graph']['nodes'].append({
                'id': domain,
                'label': domain.title(),
                'size': stats.get('document_count', 0) / 10,  # Scale for visualization
                'color': info['color'],
                'group': domain
            })

        # Network edges
        for connection in results['connection_network']['strongest_connections']:
            viz_data['network_graph']['edges'].append({
                'source': connection[0],
                'target': connection[1],
                'weight': connection[2]['weight'],
                'width': min(connection[2]['weight'] / 10, 5)  # Scale edge width
            })

        # Domain distribution
        viz_data['domain_distribution'] = {
            domain: stats['document_count']
            for domain, stats in results['domain_statistics'].items()
        }

        # Connection heatmap
        all_domains = list(self.domains.keys())
        heatmap = {}
        for i, d1 in enumerate(all_domains):
            heatmap[d1] = {}
            for j, d2 in enumerate(all_domains):
                if i < j:  # Upper triangle only
                    # Count connections between domains
                    count = sum(1 for conn in results['cross_domain_connections']
                              if ((conn['domain1'] == d1 and conn['domain2'] == d2) or
                                  (conn['domain1'] == d2 and conn['domain2'] == d1)))
                    heatmap[d1][d2] = count

        viz_data['connection_heatmap'] = heatmap

        # Insights summary
        viz_data['insights_summary'] = [
            {
                'category': insight['type'],
                'title': insight['title'],
                'key_finding': insight['insight'][:100] + "..."
            }
            for insight in results['polymath_insights']
        ]

        return viz_data

    def export_cross_domain_mapping(self, results: Dict[str, Any]) -> str:
        """Export comprehensive cross-domain mapping report"""

        report = f"""
# CROSS-DOMAIN KNOWLEDGE MAPPING REPORT
======================================

## Overview
- Total Documents Analyzed: {results['total_documents']}
- Knowledge Domains: {len(self.domains)}
- Cross-Domain Connections: {len(results['cross_domain_connections'])}
- Interdisciplinary Documents: {sum(1 for doc_domains in results['document_domains'].values() if len(doc_domains) > 1)}

## Domain Statistics
"""

        for domain, stats in results['domain_statistics'].items():
            report += f"""
### {domain.title()}
- Documents: {stats['document_count']}
- Avg Content Length: {stats['avg_content_length']:.0f} chars
- Total Connections: {stats['total_connections']}
- Interdisciplinarity Score: {stats['interdisciplinarity_score']:.2f}
- Top Connected Domains: {', '.join(stats['connection_distribution'].keys())}
"""

        report += f"""
## Network Analysis
- Network Density: {results['connection_network']['network_density']:.3f}
- Average Clustering: {results['connection_network']['average_clustering']:.3f}
- Most Connected Domain: {max(results['connection_network']['degree_centrality'].items(), key=lambda x: x[1])[0].title()}

## Polymath Insights
"""

        for insight in results['polymath_insights']:
            report += f"""
### {insight['title']}
{insight['insight']}

Data: {insight['data']}
"""

        report += f"""
## Emerging Patterns
"""

        for pattern in results['emergent_patterns']:
            report += f"""
### {pattern['type'].title()}
{pattern['pattern']}
- Strength: {pattern['strength']}
- Significance: {pattern['significance']}
"""

        report += f"""
## Knowledge Ecosystem Map
"""

        ecosystem = results['knowledge_ecosystem']
        for domain, info in ecosystem['domains'].items():
            report += f"""
### {domain.title()}
{info['description']}
- Documents: {info['document_count']}
- Keywords: {', '.join(info['keywords'])}
- Avg Connections: {info['avg_connections']:.1f}
"""

        report += f"""
## Learning Recommendations
"""

        for rec in ecosystem['recommendations']:
            path_info = rec.get('path', rec.get('domains', []))
            report += f"""
### {rec['title']} ({rec['type'].title()})
Path: {' → '.join(path_info)}
Reasoning: {rec['reasoning']}
"""

        return report

def main():
    """Main function for cross-domain mapping analysis"""

    print("🚀 CROSS-DOMAIN MAPPING SYSTEM")
    print("=" * 70)
    print("Analyzing entire knowledge base for interdisciplinary connections...")

    # Initialize mapper
    mapper = CrossDomainMapper()

    # Perform comprehensive analysis
    print("\n🔍 Starting comprehensive knowledge analysis...")
    results = mapper.analyze_complete_knowledge_base()

    print("\n📊 ANALYSIS RESULTS:")
    print(f"   📄 Total Documents: {results['total_documents']}")
    print(f"   🧬 Knowledge Domains: {len(results['domain_statistics'])}")
    print(f"   🌉 Cross-Domain Connections: {len(results['cross_domain_connections'])}")
    print(f"   🔗 Network Density: {results['connection_network']['network_density']:.3f}")
    print(f"   🧠 Polymath Insights: {len(results['polymath_insights'])}")

    # Generate visualization data
    print("\n📈 Generating visualization data...")
    viz_data = mapper.generate_cross_domain_visualization(results)

    # Export comprehensive report
    print("\n📋 Generating comprehensive report...")
    report = mapper.export_cross_domain_mapping(results)

    # Save report
    with open('cross_domain_mapping_report.md', 'w') as f:
        f.write(report)

    # Save results as JSON
    with open('cross_domain_mapping_results.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, set):
                        json_results[key][k] = list(v)
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = value if not isinstance(value, set) else list(value)

        json.dump(json_results, f, indent=2, default=str)

    print("\n✅ CROSS-DOMAIN MAPPING COMPLETE!")
    print("   📄 Report saved: cross_domain_mapping_report.md")
    print("   📊 Data saved: cross_domain_mapping_results.json")
    print(f"   🧬 Domains mapped: {len(results['domain_statistics'])}")
    print(f"   🌉 Connections discovered: {len(results['cross_domain_connections'])}")
    print("   🧠 Polymath insights generated!")
    # Show top insights
    print("\n🔍 TOP POLYMATH INSIGHTS:")
    for insight in results['polymath_insights'][:3]:
        print(f"   • {insight['title']}: {insight['insight'][:80]}...")

if __name__ == "__main__":
    main()
