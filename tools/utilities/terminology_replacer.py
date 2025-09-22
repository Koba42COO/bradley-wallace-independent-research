#!/usr/bin/env python3
import json
import re
import os
import sys

def replace_terminology_in_json(file_path):
    """Replace prime aligned compute terminology with prime aligned terminology in JSON files"""

    # Read the JSON file
    with open(file_path, 'r') as f:
        content = f.read()

    # Define replacements
    replacements = {
        'prime aligned compute': 'prime_aligned_compute',
        'prime_aligned_math': 'prime_aligned_math',
        'prime_aligned_score': 'prime_aligned_score',
        'prime_aligned_evolution': 'prime_aligned_evolution',
        'prime_aligned_framework': 'prime_aligned_framework',
        'prime_aligned_awareness': 'prime_aligned_awareness',
        'prime_aligned_coherence': 'prime_aligned_coherence',
        'prime_aligned_patterns': 'prime_aligned_patterns',
        'prime_aligned_tests': 'prime_aligned_tests',
        'prime_aligned_analysis': 'prime_aligned_analysis',
        'prime_aligned_heatmaps': 'prime_aligned_heatmaps',
        'prime_aligned_resonance': 'prime_aligned_resonance',
        'prime_aligned_wave': 'prime_aligned_wave',
        'prime_aligned_emergence': 'prime_aligned_emergence',
        'prime_aligned_optimized': 'prime_aligned_optimized',
        'prime_aligned_enhanced': 'prime_aligned_enhanced',
        'prime_aligned_breakthrough': 'prime_aligned_breakthrough',
        'prime_aligned_level': 'prime_aligned_level',
        'prime_aligned_technology': 'prime_aligned_technology',
        'prime_aligned_ecosystem': 'prime_aligned_ecosystem',
        'prime_aligned_benchmark': 'prime_aligned_benchmark',
        'prime_aligned_metrics': 'prime_aligned_metrics',
        'prime_aligned_maturity': 'prime_aligned_maturity',
        'prime_aligned_trend': 'prime_aligned_trend',
        'prime_aligned_breakthrough': 'prime_aligned_breakthrough',
        'prime_aligned_aware': 'prime_aligned_aware',
        'prime_aligned_sentiment': 'prime_aligned_sentiment',
        'prime_aligned_counter': 'prime_aligned_counter',
        'prime_aligned_ml_training': 'prime_aligned_ml_training',
        'prime_aligned_math_physics': 'prime_aligned_math_physics',
        'prime_aligned_ecosystem_benchmark': 'prime_aligned_ecosystem_benchmark'
    }

    # Apply replacements
    for old_term, new_term in replacements.items():
        content = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, content, flags=re.IGNORECASE)

    # Write back the file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Updated terminology in {file_path}")

def replace_terminology_in_text(file_path):
    """Replace prime aligned compute terminology in text files"""

    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Define replacements
    replacements = {
        'prime aligned compute': 'prime aligned compute',
        'prime aligned compute mathematics': 'prime aligned math',
        'prime aligned compute score': 'prime aligned score',
        'prime aligned compute evolution': 'prime aligned evolution',
        'prime aligned compute framework': 'prime aligned framework',
        'prime aligned compute awareness': 'prime aligned awareness',
        'prime aligned compute coherence': 'prime aligned coherence',
        'prime aligned compute patterns': 'prime aligned patterns',
        'prime aligned compute tests': 'prime aligned tests',
        'prime aligned compute analysis': 'prime aligned analysis',
        'prime aligned compute heatmaps': 'prime aligned heatmaps',
        'prime aligned compute resonance': 'prime aligned resonance',
        'prime aligned compute wave': 'prime aligned wave',
        'prime aligned compute emergence': 'prime aligned emergence',
        'prime aligned compute optimized': 'prime aligned optimized',
        'prime aligned compute enhanced': 'prime aligned enhanced',
        'prime aligned compute breakthrough': 'prime aligned breakthrough',
        'prime aligned compute level': 'prime aligned level',
        'prime aligned compute technology': 'prime aligned technology',
        'prime aligned compute ecosystem': 'prime aligned ecosystem',
        'prime aligned compute benchmark': 'prime aligned benchmark',
        'prime aligned compute metrics': 'prime aligned metrics',
        'prime aligned compute maturity': 'prime aligned maturity',
        'prime aligned compute trend': 'prime aligned trend',
        'prime aligned compute breakthrough': 'prime aligned breakthrough',
        'prime aligned compute aware': 'prime aligned aware',
        'prime aligned compute sentiment': 'prime aligned sentiment',
        'prime aligned compute counter': 'prime aligned counter',
        'prime aligned compute ml training': 'prime aligned ml training',
        'prime aligned compute mathematics physics': 'prime aligned math physics',
        'prime aligned compute ecosystem benchmark': 'prime aligned ecosystem benchmark'
    }

    # Apply replacements
    for old_term, new_term in replacements.items():
        content = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, content, flags=re.IGNORECASE)

    # Write back the file
    with open(file_path, 'w') as f:
        f.write(content)

    print(f"Updated terminology in {file_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python terminology_replacer.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        sys.exit(1)

    if file_path.endswith('.json'):
        replace_terminology_in_json(file_path)
    else:
        replace_terminology_in_text(file_path)
