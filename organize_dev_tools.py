#!/usr/bin/env python3
"""
Dev Tools Organization Script
=============================

Automatically organizes the dev folder into modular chunks based on file names
and content analysis. This script categorizes tools into logical modules for
better organization and maintainability.

Author: Bradley Wallace - Development Organization System
License: Proprietary Research - Internal Use Only
"""

import os
import shutil
import re
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict

class DevToolsOrganizer:
    """
    Automated organizer for development tools into modular chunks.
    """

    def __init__(self, base_path: str = "/Users/coo-koba42/dev"):
        self.base_path = Path(base_path)
        self.chunks_path = self.base_path / "modular_chunks"

        # Define categorization rules
        self.categories = {
            'ai_ml_systems': {
                'keywords': ['ai_', 'ml_', 'neural', 'deep_learning', 'machine_learning', 'gpt', 'llm', 'transformer', 'bert'],
                'patterns': [r'ADVANCED_ML_', r'AI_', r'AUDIO_AGENT_', r'COMPREHENSIVE_DEVELOPMENT_MASTERY_']
            },
            'security_cyber_tools': {
                'keywords': ['security', 'cyber', 'pentest', 'bounty', 'vulnerability', 'encryption', 'hacking', 'exploit'],
                'patterns': [r'ADVANCED_PENETRATION_', r'COMPREHENSIVE_BUG_BOUNTY_', r'ADVANCED_BOUNTY_', r'ADVANCED_COUNTERCODE_']
            },
            'mathematical_research': {
                'keywords': ['math', 'fractal', 'wallace', 'riemann', 'algebra', 'geometry', 'topology', 'analysis'],
                'patterns': [r'COMPLETE_MATHEMATICAL_', r'BROAD_FIELD_MATH_', r'FRACTAL_', r'wallace_transform_']
            },
            'blockchain_crypto': {
                'keywords': ['blockchain', 'crypto', 'bitcoin', 'ethereum', 'quantum_email', 'encryption'],
                'patterns': [r'BLOCKCHAIN_', r'quantum_email_', r'crypto']
            },
            'consciousness_neural': {
                'keywords': ['prime aligned compute', 'neural', 'brain', 'cognitive', 'mind', 'awareness'],
                'patterns': [r'consciousness_', r'CHUNKED_256D_', r'neural_']
            },
            'quantum_computing': {
                'keywords': ['quantum', 'qubit', 'superposition', 'entanglement', 'quantum_computing'],
                'patterns': [r'quantum_', r'qubit_']
            },
            'data_processing': {
                'keywords': ['data', 'processing', 'analysis', 'visualization', 'statistics', 'dataset'],
                'patterns': [r'data_', r'processing_', r'analysis_']
            },
            'educational_tools': {
                'keywords': ['education', 'learning', 'teaching', 'curriculum', 'tutorial'],
                'patterns': [r'COMPREHENSIVE_KNOWLEDGE_', r'education_', r'learning_']
            },
            'development_tools': {
                'keywords': ['development', 'coding', 'programming', 'debug', 'test', 'build'],
                'patterns': [r'ADVANCED_DEVELOPMENT_', r'COMPREHENSIVE_TOOL_ANALYSIS_', r'KOBA42_CODING_']
            },
            'integration_systems': {
                'keywords': ['integration', 'system', 'platform', 'framework', 'unified'],
                'patterns': [r'UNIFIED_SYSTEM_', r'COMPLETE_SYSTEM_MAINTENANCE_', r'INTEGRATED_']
            },
            'utility_scripts': {
                'keywords': ['utility', 'helper', 'tool', 'script', 'automation'],
                'patterns': [r'utility_', r'helper_', r'tool_']
            }
        }

        # Create chunk directories
        self._create_chunk_directories()

    def _create_chunk_directories(self):
        """Create all chunk directories."""
        for category in self.categories.keys():
            chunk_path = self.chunks_path / category
            chunk_path.mkdir(parents=True, exist_ok=True)
            print(f"📁 Created chunk directory: {chunk_path}")

    def categorize_file(self, filename: str) -> str:
        """
        Categorize a file based on its name and content.

        Args:
            filename: Name of the file to categorize

        Returns:
            Category name for the file
        """
        filename_lower = filename.lower()

        # Check each category
        for category, rules in self.categories.items():
            # Check keywords
            for keyword in rules['keywords']:
                if keyword in filename_lower:
                    return category

            # Check patterns
            for pattern in rules['patterns']:
                if re.search(pattern, filename, re.IGNORECASE):
                    return category

        # Default category for uncategorized files
        return 'utility_scripts'

    def organize_files(self):
        """Organize all files in the dev directory into chunks."""
        print("🔄 Starting file organization...")

        # Get all Python files in the main directory (excluding chunks and wallace_research_suite)
        main_files = []
        for file_path in self.base_path.glob("*.py"):
            if not str(file_path).startswith(str(self.chunks_path)) and 'wallace_research_suite' not in str(file_path):
                main_files.append(file_path)

        print(f"📊 Found {len(main_files)} files to organize")

        # Categorize and move files
        moved_files = 0
        category_counts = defaultdict(int)

        for file_path in main_files:
            category = self.categorize_file(file_path.name)
            destination = self.chunks_path / category / file_path.name

            try:
                shutil.move(str(file_path), str(destination))
                moved_files += 1
                category_counts[category] += 1
                print(f"✅ Moved {file_path.name} → {category}")
            except Exception as e:
                print(f"❌ Failed to move {file_path.name}: {e}")

        # Print summary
        print("\n📊 Organization Summary:")
        print("=" * 50)
        print(f"Total files moved: {moved_files}")

        for category, count in sorted(category_counts.items()):
            print("30")

        # Handle other file types
        self._organize_other_files()

    def _organize_other_files(self):
        """Organize non-Python files."""
        print("\n🔄 Organizing other file types...")

        other_extensions = ['.json', '.txt', '.md', '.html', '.css', '.js', '.sh']

        for ext in other_extensions:
            for file_path in self.base_path.glob(f"*{ext}"):
                if not str(file_path).startswith(str(self.chunks_path)) and 'wallace_research_suite' not in str(file_path):
                    # Move to utility_scripts for now
                    destination = self.chunks_path / 'utility_scripts' / file_path.name
                    try:
                        shutil.move(str(file_path), str(destination))
                        print(f"✅ Moved {file_path.name} → utility_scripts")
                    except Exception as e:
                        print(f"❌ Failed to move {file_path.name}: {e}")

    def create_chunk_readmes(self):
        """Create README files for each chunk."""
        for category, rules in self.categories.items():
            readme_path = self.chunks_path / category / "README.md"

            readme_content = f"""# {category.replace('_', ' ').title()} Module

## Overview
This module contains development tools and systems related to {category.replace('_', ' ')}.

## Keywords
{', '.join(rules['keywords'])}

## Patterns
{', '.join(rules['patterns'])}

## Contents
This directory contains Python scripts and related files for {category.replace('_', ' ')} functionality.

## Usage
Refer to individual script documentation for usage instructions.

---
*Auto-generated by DevToolsOrganizer*
"""

            try:
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                print(f"📝 Created README for {category}")
            except Exception as e:
                print(f"❌ Failed to create README for {category}: {e}")

    def create_master_readme(self):
        """Create a master README for the modular chunks."""
        master_readme = f"""# Modular Dev Tools Chunks
## Bradley Wallace Development Environment

This directory contains the complete development environment organized into modular chunks for better maintainability and organization.

## Chunk Structure

"""

        for category in sorted(self.categories.keys()):
            chunk_path = self.chunks_path / category
            if chunk_path.exists():
                file_count = len(list(chunk_path.glob("*.py")))
                master_readme += f"### {category.replace('_', ' ').title()}\n"
                master_readme += f"- **Files:** {file_count} Python scripts\n"
                master_readme += f"- **Purpose:** {category.replace('_', ' ')} tools and systems\n\n"

        master_readme += """## Organization Script

This structure was created using the `organize_dev_tools.py` script, which automatically categorizes files based on naming patterns and keywords.

## Usage

Each chunk contains related tools and can be developed, tested, and deployed independently. Refer to individual chunk READMEs for specific usage instructions.

## Maintenance

To reorganize files or add new categories, run:
```bash
python organize_dev_tools.py
```

---
*Generated by DevToolsOrganizer*
*Bradley Wallace - Development Environment Organization*
"""

        master_readme_path = self.chunks_path / "README.md"
        try:
            with open(master_readme_path, 'w') as f:
                f.write(master_readme)
            print("📝 Created master README")
        except Exception as e:
            print(f"❌ Failed to create master README: {e}")

    def run_complete_organization(self):
        """Run the complete organization process."""
        print("🚀 Starting Complete Dev Tools Organization")
        print("=" * 60)

        self.organize_files()
        self.create_chunk_readmes()
        self.create_master_readme()

        print("\n✅ Organization Complete!")
        print("📁 Check the modular_chunks/ directory for organized tools")


def main():
    """Main organization function."""
    organizer = DevToolsOrganizer()
    organizer.run_complete_organization()


if __name__ == "__main__":
    main()
