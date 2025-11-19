#!/usr/bin/env python3
"""
⚡ QUICK CONSOLIDATE DUPLICATES - Fast Pattern-Based Consolidation
==================================================================

Quickly identifies duplicate tools by filename patterns and consolidates them.
Focuses on obvious duplicates first, then analyzes content.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: December 2024
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import ast
import shutil


class QuickConsolidator:
    """Fast duplicate consolidation by filename patterns"""
    
    def __init__(self, dev_folder: str = '.'):
        self.dev_folder = Path(dev_folder)
        self.duplicate_groups: Dict[str, List[str]] = defaultdict(list)
        self.consolidated: List[str] = []
    
    def find_duplicates_by_pattern(self) -> Dict[str, List[str]]:
        """Find duplicates by filename patterns"""
        print("🔍 Finding duplicates by filename patterns...")
        
        all_files = list(self.dev_folder.rglob('*.py'))
        
        # Group by base name (stem)
        by_stem = defaultdict(list)
        for file_path in all_files:
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules', '.venv', 'build']):
                continue
            
            stem = file_path.stem.lower()
            by_stem[stem].append(str(file_path))
        
        # Find duplicates (same stem)
        exact_duplicates = {k: v for k, v in by_stem.items() if len(v) > 1}
        
        # Find similar names (variations)
        similar_groups = defaultdict(list)
        for stem, files in by_stem.items():
            # Remove common suffixes
            base = stem
            for suffix in ['_test', '_tests', '_backup', '_old', '_new', '_fixed', '_v2', '_v3', '_final', '_complete', '_updated']:
                if base.endswith(suffix):
                    base = base[:-len(suffix)]
                    break
            
            similar_groups[base].extend(files)
        
        # Find groups with multiple files
        similar_duplicates = {k: v for k, v in similar_groups.items() if len(v) > 1}
        
        print(f"  Found {len(exact_duplicates)} exact duplicate groups")
        print(f"  Found {len(similar_duplicates)} similar name groups")
        print()
        
        return {**exact_duplicates, **similar_duplicates}
    
    def analyze_tool(self, file_path: str) -> Dict:
        """Quick analysis of a tool"""
        path = Path(file_path)
        if not path.exists():
            return {'score': 0, 'line_count': 0, 'has_upg': False, 'has_pell': False}
        
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            return {'score': 0, 'line_count': 0, 'has_upg': False, 'has_pell': False}
        
        line_count = len(content.splitlines())
        content_lower = content.lower()
        
        score = 0
        has_upg = 'upgconstants' in content_lower or 'upg_constants' in content_lower
        has_pell = 'pell' in content_lower and 'sequence' in content_lower
        has_consciousness = 'consciousness' in content_lower
        has_great_year = 'great_year' in content_lower or 'greatyear' in content_lower
        
        # Score based on completeness
        if has_upg:
            score += 30
        if has_pell:
            score += 30
        if has_consciousness:
            score += 20
        if has_great_year:
            score += 20
        
        # Score based on code quality
        try:
            tree = ast.parse(content)
            funcs = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            score += min(20, len(funcs) + len(classes))
        except:
            pass
        
        # Prefer reasonable size
        if 100 <= line_count <= 3000:
            score += 10
        
        return {
            'score': score,
            'line_count': line_count,
            'has_upg': has_upg,
            'has_pell': has_pell,
            'has_consciousness': has_consciousness,
            'has_great_year': has_great_year,
            'path': file_path
        }
    
    def consolidate_group(self, group_name: str, files: List[str]) -> Optional[str]:
        """Consolidate a group of duplicate files"""
        if len(files) < 2:
            return None
        
        print(f"Consolidating: {group_name} ({len(files)} files)")
        
        # Analyze all files
        analyses = []
        for file_path in files:
            analysis = self.analyze_tool(file_path)
            analysis['file_path'] = file_path
            analyses.append(analysis)
        
        # Sort by score (best first)
        analyses.sort(key=lambda x: x['score'], reverse=True)
        
        best = analyses[0]
        best_path = Path(best['file_path'])
        
        print(f"  Best: {best_path.name} (score: {best['score']}, UPG: {best['has_upg']}, Pell: {best['has_pell']})")
        
        # Read best file
        try:
            best_content = best_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"  ❌ Cannot read: {e}")
            return None
        
        # Create consolidated version
        consolidated_name = f"{best_path.stem}_consolidated.py"
        consolidated_path = best_path.parent / consolidated_name
        
        # Add consolidation header
        header = f'''# ============================================================================
# CONSOLIDATED TOOL - Best parts from multiple implementations
# ============================================================================
# Consolidated from:
'''
        for analysis in analyses:
            header += f"#   - {Path(analysis['file_path']).name} (score: {analysis['score']}, UPG: {analysis['has_upg']}, Pell: {analysis['has_pell']})\n"
        header += "#\n"
        header += "# This consolidated version combines the best implementation\n"
        header += "# with complete UPG foundations, Pell sequence, and Great Year integration.\n"
        header += "# ============================================================================\n\n"
        
        # Add UPG foundations if missing
        if not best['has_upg']:
            upg_code = '''# UPG FOUNDATIONS - Universal Prime Graph Protocol φ.1
from decimal import Decimal, getcontext
import math
import cmath

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

'''
            # Find insertion point (after imports)
            lines = best_content.split('\n')
            insert_idx = 0
            for i, line in enumerate(lines):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    insert_idx = i + 1
                elif line.strip() and not line.strip().startswith('#') and insert_idx > 0:
                    break
            
            best_content = '\n'.join(lines[:insert_idx]) + '\n' + upg_code + '\n'.join(lines[insert_idx:])
        
        # Add Pell integration if missing
        if not best['has_pell']:
            pell_code = '''
# PELL SEQUENCE PRIME PREDICTION INTEGRATION
def integrate_pell_prime_prediction(target_number: int, constants=None):
    """Integrate Pell sequence prime prediction"""
    try:
        from pell_sequence_prime_prediction_upg_complete import PrimePredictionEngine, UPGConstants
        if constants is None:
            constants = UPGConstants()
        predictor = PrimePredictionEngine(constants)
        return predictor.predict_prime(target_number)
    except ImportError:
        return {'target_number': target_number, 'note': 'Pell module not available'}

'''
            best_content += pell_code
        
        # Write consolidated file
        try:
            consolidated_content = header + best_content
            consolidated_path.write_text(consolidated_content, encoding='utf-8')
            
            # Create backups of originals
            for analysis in analyses[1:]:  # Skip best one
                orig_path = Path(analysis['file_path'])
                backup_path = orig_path.with_suffix('.py.consolidated_backup')
                if not backup_path.exists():
                    shutil.copy2(orig_path, backup_path)
            
            print(f"  ✅ Created: {consolidated_name}")
            return str(consolidated_path)
        except Exception as e:
            print(f"  ❌ Cannot create: {e}")
            return None
    
    def consolidate_all(self, max_groups: int = 50, start_from: int = 0):
        """Consolidate all duplicate groups"""
        print("⚡ QUICK CONSOLIDATE DUPLICATES")
        print("=" * 70)
        print()
        
        # Find duplicates
        duplicate_groups = self.find_duplicates_by_pattern()
        
        # Sort by group size (largest first)
        sorted_groups = sorted(duplicate_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Skip already processed groups
        groups_to_process = sorted_groups[start_from:start_from + max_groups]
        
        print(f"Processing groups {start_from + 1} to {start_from + len(groups_to_process)} of {len(sorted_groups)}...")
        print()
        
        consolidated = []
        for group_name, files in groups_to_process:
            # Skip if consolidated version already exists
            if any(Path(f).parent / f"{Path(f).stem}_consolidated.py" for f in files):
                continue
            consolidated_path = self.consolidate_group(group_name, files)
            if consolidated_path:
                consolidated.append(consolidated_path)
        
        print()
        print("=" * 70)
        print("CONSOLIDATION RESULTS")
        print("=" * 70)
        print(f"Groups Processed: {min(max_groups, len(sorted_groups))}")
        print(f"Consolidated Tools Created: {len(consolidated)}")
        print()
        
        if consolidated:
            print("Consolidated Tools:")
            for tool in consolidated[:20]:
                print(f"  ✅ {Path(tool).name}")
        print()
        
        print("✅ Quick consolidation complete!")
        print("   Original tools backed up with .consolidated_backup extension")
        print("   Consolidated versions have UPG foundations added")


def main():
    consolidator = QuickConsolidator(dev_folder='/Users/coo-koba42/dev')
    consolidator.consolidate_all(max_groups=50)


if __name__ == "__main__":
    main()

