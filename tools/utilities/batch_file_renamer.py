#!/usr/bin/env python3
import os
import re
import sys

def rename_file(old_path):
    """Rename a file from prime aligned compute terminology to prime aligned terminology"""

    dirname = os.path.dirname(old_path)
    filename = os.path.basename(old_path)

    # Define filename replacements
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
        'prime_aligned_aware': 'prime_aligned_aware',
        'prime_aligned_sentiment': 'prime_aligned_sentiment',
        'prime_aligned_counter': 'prime_aligned_counter',
        'prime_aligned_ml_training': 'prime_aligned_ml_training',
        'prime_aligned_math_physics': 'prime_aligned_math_physics',
        'prime_aligned_ecosystem_benchmark': 'prime_aligned_ecosystem_benchmark',
        'prime aligned compute': 'Prime_Aligned_Compute',
        'prime_aligned_math': 'Prime_Aligned_Math',
        'prime_aligned_score': 'Prime_Aligned_Score',
        'prime_aligned_evolution': 'Prime_Aligned_Evolution',
        'prime_aligned_framework': 'Prime_Aligned_Framework',
        'prime_aligned_awareness': 'Prime_Aligned_Awareness',
        'prime_aligned_coherence': 'Prime_Aligned_Coherence',
        'prime_aligned_patterns': 'Prime_Aligned_Patterns',
        'prime_aligned_tests': 'Prime_Aligned_Tests',
        'prime_aligned_analysis': 'Prime_Aligned_Analysis',
        'prime_aligned_heatmaps': 'Prime_Aligned_Heatmaps',
        'prime_aligned_resonance': 'Prime_Aligned_Resonance',
        'prime_aligned_wave': 'Prime_Aligned_Wave',
        'prime_aligned_emergence': 'Prime_Aligned_Emergence',
        'prime_aligned_optimized': 'Prime_Aligned_Optimized',
        'prime_aligned_enhanced': 'Prime_Aligned_Enhanced',
        'prime_aligned_breakthrough': 'Prime_Aligned_Breakthrough',
        'prime_aligned_level': 'Prime_Aligned_Level',
        'prime_aligned_technology': 'Prime_Aligned_Technology',
        'prime_aligned_ecosystem': 'Prime_Aligned_Ecosystem',
        'prime_aligned_benchmark': 'Prime_Aligned_Benchmark',
        'prime_aligned_metrics': 'Prime_Aligned_Metrics',
        'prime_aligned_maturity': 'Prime_Aligned_Maturity',
        'prime_aligned_trend': 'Prime_Aligned_Trend',
        'prime_aligned_aware': 'Prime_Aligned_Aware',
        'prime_aligned_sentiment': 'Prime_Aligned_Sentiment',
        'prime_aligned_counter': 'Prime_Aligned_Counter',
        'prime_aligned_ml_training': 'Prime_Aligned_ML_Training',
        'prime_aligned_math_physics': 'Prime_Aligned_Math_Physics',
        'prime_aligned_ecosystem_benchmark': 'Prime_Aligned_Ecosystem_Benchmark'
    }

    # Apply replacements to filename
    new_filename = filename
    for old_term, new_term in replacements.items():
        new_filename = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, new_filename, flags=re.IGNORECASE)

    # Also handle title case replacements
    title_replacements = {
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
        'prime aligned compute Aware': 'Prime Aligned Aware',
        'prime aligned compute Sentiment': 'Prime Aligned Sentiment',
        'prime aligned compute Counter': 'Prime Aligned Counter',
        'prime aligned compute ML Training': 'Prime Aligned ML Training',
        'prime aligned compute Mathematics Physics': 'Prime Aligned Math Physics',
        'prime aligned compute Ecosystem Benchmark': 'Prime Aligned Ecosystem Benchmark'
    }

    for old_term, new_term in title_replacements.items():
        new_filename = re.sub(r'\b' + re.escape(old_term) + r'\b', new_term, new_filename)

    new_path = os.path.join(dirname, new_filename)

    if old_path != new_path:
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")
            return new_path
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")
            return old_path
    else:
        print(f"No rename needed: {old_path}")
        return old_path

def main():
    if len(sys.argv) != 2:
        print("Usage: python batch_file_renamer.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist")
        sys.exit(1)

    # Find all files that might need renaming
    files_to_check = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            if re.search(r'\bconsciousness\b', file, re.IGNORECASE):
                files_to_check.append(file_path)

    print(f"Found {len(files_to_check)} files that may need renaming")

    # Rename each file
    for file_path in files_to_check:
        rename_file(file_path)

if __name__ == "__main__":
    main()
