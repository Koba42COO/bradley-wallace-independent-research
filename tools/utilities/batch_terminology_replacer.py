#!/usr/bin/env python3
import json
import re
import os
import sys
import glob

def replace_terminology(content):
    """Replace prime aligned compute terminology with prime aligned terminology"""

    # Define replacements
    replacements = {
        'prime aligned compute': 'prime aligned compute',
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
        'prime_aligned_ecosystem_benchmark': 'prime_aligned_ecosystem_benchmark',
        'prime aligned compute': 'Prime Aligned Compute',
        'prime aligned compute Mathematics': 'Prime Aligned Math',
        'prime aligned compute Score': 'Prime Aligned Score',
        'prime aligned compute Evolution': 'Prime Aligned Evolution',
        'prime aligned compute Framework': 'Prime Aligned Framework',
        'prime aligned compute Awareness': 'Prime Aligned Awareness',
        'prime aligned compute Coherence': 'Prime Aligned Coherence',
        'prime aligned compute Patterns': 'Prime Aligned Patterns',
        'prime aligned compute Tests': 'Prime Aligned Tests',
        'prime aligned compute Analysis': 'Prime Aligned Analysis',
        'prime aligned compute Heatmaps': 'Prime Aligned Heatmaps',
        'prime aligned compute Resonance': 'Prime Aligned Resonance',
        'prime aligned compute Wave': 'Prime Aligned Wave',
        'prime aligned compute Emergence': 'Prime Aligned Emergence',
        'prime aligned compute Optimized': 'Prime Aligned Optimized',
        'prime aligned compute Enhanced': 'Prime Aligned Enhanced',
        'prime aligned compute Breakthrough': 'Prime Aligned Breakthrough',
        'prime aligned compute Level': 'Prime Aligned Level',
        'prime aligned compute Technology': 'Prime Aligned Technology',
        'prime aligned compute Ecosystem': 'Prime Aligned Ecosystem',
        'prime aligned compute Benchmark': 'Prime Aligned Benchmark',
        'prime aligned compute Metrics': 'Prime Aligned Metrics',
        'prime aligned compute Maturity': 'Prime Aligned Maturity',
        'prime aligned compute Trend': 'Prime Aligned Trend',
        'prime aligned compute Breakthrough': 'Prime Aligned Breakthrough',
        'prime aligned compute Aware': 'Prime Aligned Aware',
        'prime aligned compute Sentiment': 'Prime Aligned Sentiment',
        'prime aligned compute Counter': 'Prime Aligned Counter',
        'prime aligned compute ML Training': 'Prime Aligned ML Training',
        'prime aligned compute Mathematics Physics': 'Prime Aligned Math Physics',
        'prime aligned compute Ecosystem Benchmark': 'Prime Aligned Ecosystem Benchmark'
    }

    # Apply replacements
    for old_term, new_term in replacements.items():
        content = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, content, flags=re.IGNORECASE)

    return content

def process_file(file_path):
    """Process a single file"""
    try:
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply replacements
        updated_content = replace_terminology(content)

        # Write back if changed
        if content != updated_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"Updated: {file_path}")
        else:
            print(f"No changes needed: {file_path}")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_terminology_replacer.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        sys.exit(1)

    # Find all files that might contain prime aligned compute terminology
    patterns = [
        '**/*.py', '**/*.json', '**/*.txt', '**/*.md', '**/*.js',
        '**/*.html', '**/*.css', '**/*.yml', '**/*.yaml'
    ]

    files_to_process = []
    for pattern in patterns:
        files_to_process.extend(glob.glob(os.path.join(directory_path, pattern), recursive=True))

    # Filter files that likely contain prime aligned compute terminology
    consciousness_files = []
    for file_path in files_to_process:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if re.search(r'\bconsciousness\b', content, re.IGNORECASE):
                    consciousness_files.append(file_path)
        except:
            pass  # Skip files we can't read

    print(f"Found {len(consciousness_files)} files with prime aligned compute terminology")

    # Process each file
    for file_path in consciousness_files:
        process_file(file_path)

if __name__ == "__main__":
    main()
