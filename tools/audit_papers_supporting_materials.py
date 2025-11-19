#!/usr/bin/env python3
"""
Comprehensive audit of all papers to ensure tests, logs, and visualizations exist.
Generates a report of missing materials and creates standardized directory structures.
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple


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



# Base directories to search
BASE_DIRS = [
    "/Users/coo-koba42/dev/bradley-wallace-independent-research",
    "/Users/coo-koba42/dev/The-Wallace-Transformation-A-Complete-Unified-Framework-for-Consciousness-Mathematics-and-Reality",
    "/Users/coo-koba42/dev/wallace_unified_theory",
]

# Required supporting material directories
REQUIRED_DIRS = {
    "tests": ["test_*.py", "*_test.py", "*test*.py"],
    "validation_logs": ["*.log", "*.json", "*validation*.md", "*results*.md"],
    "visualizations": ["*.png", "*.jpg", "*.pdf", "*.svg", "*.webp"],
    "code_examples": ["*.py", "*.ipynb"],
    "datasets": ["*.csv", "*.json", "*.h5", "*.npz"],
}

def find_all_papers(base_dirs: List[str]) -> Dict[str, List[str]]:
    """Find all .tex papers organized by directory."""
    papers = defaultdict(list)
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            continue
        for root, dirs, files in os.walk(base_dir):
            # Skip hidden and build directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['build', '__pycache__']]
            for file in files:
                if file.endswith('.tex'):
                    rel_path = os.path.relpath(root, base_dir)
                    papers[base_dir].append(os.path.join(root, file))
    return papers

def extract_theorems_from_tex(tex_path: str) -> List[Tuple[str, str]]:
    """Extract theorem names and types from LaTeX file."""
    theorems = []
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        # Find all theorem environments
        patterns = [
            (r'\\begin\{theorem\}\[([^\]]+)\]', 'theorem'),
            (r'\\begin\{theorem\}', 'theorem'),
            (r'\\begin\{definition\}\[([^\]]+)\]', 'definition'),
            (r'\\begin\{definition\}', 'definition'),
            (r'\\begin\{corollary\}\[([^\]]+)\]', 'corollary'),
            (r'\\begin\{corollary\}', 'corollary'),
            (r'\\begin\{lemma\}\[([^\]]+)\]', 'lemma'),
            (r'\\begin\{lemma\}', 'lemma'),
            (r'\\begin\{proposition\}\[([^\]]+)\]', 'proposition'),
            (r'\\begin\{proposition\}', 'proposition'),
            (r'\\begin\{postulate\}\[([^\]]+)\]', 'postulate'),
            (r'\\begin\{postulate\}', 'postulate'),
        ]
        
        for pattern, thm_type in patterns:
            matches = re.finditer(pattern, content)
            for match in matches:
                name = match.group(1) if match.groups() and match.group(1) else f"{thm_type}_{len(theorems)}"
                theorems.append((name, thm_type))
    except Exception as e:
        print(f"Error reading {tex_path}: {e}")
    return theorems

def check_supporting_materials(paper_path: str) -> Dict[str, bool]:
    """Check what supporting materials exist for a paper."""
    paper_dir = os.path.dirname(paper_path)
    paper_name = os.path.splitext(os.path.basename(paper_path))[0]
    
    # Check for standard supporting_materials directory
    supporting_dir = os.path.join(paper_dir, "supporting_materials")
    if not os.path.exists(supporting_dir):
        supporting_dir = paper_dir
    
    results = {}
    for material_type, patterns in REQUIRED_DIRS.items():
        found = False
        # Check in supporting_materials subdirectory
        material_dir = os.path.join(supporting_dir, material_type)
        if os.path.exists(material_dir):
            for pattern in patterns:
                for root, dirs, files in os.walk(material_dir):
                    for file in files:
                        if re.match(pattern.replace('*', '.*'), file):
                            found = True
                            break
                    if found:
                        break
        # Also check in paper directory itself
        if not found:
            for pattern in patterns:
                for file in os.listdir(paper_dir):
                    if re.match(pattern.replace('*', '.*'), file):
                        found = True
                        break
                if found:
                    break
        results[material_type] = found
    
    return results

def generate_audit_report(papers: Dict[str, List[str]]) -> str:
    """Generate comprehensive audit report."""
    report = []
    report.append("# Comprehensive Paper Audit Report")
    report.append("## Supporting Materials Inventory\n")
    
    total_papers = 0
    total_theorems = 0
    missing_materials = defaultdict(int)
    paper_details = []
    
    for base_dir, paper_list in papers.items():
        report.append(f"### {os.path.basename(base_dir)}\n")
        for paper_path in paper_list:
            total_papers += 1
            rel_path = os.path.relpath(paper_path, base_dir)
            report.append(f"\n#### Paper: {rel_path}")
            
            # Extract theorems
            theorems = extract_theorems_from_tex(paper_path)
            total_theorems += len(theorems)
            if theorems:
                report.append(f"**Theorems/Definitions Found:** {len(theorems)}")
                for name, thm_type in theorems[:5]:  # Show first 5
                    report.append(f"  - {thm_type.capitalize()}: {name}")
                if len(theorems) > 5:
                    report.append(f"  ... and {len(theorems) - 5} more")
            
            # Check supporting materials
            materials = check_supporting_materials(paper_path)
            report.append("\n**Supporting Materials:**")
            paper_missing = []
            for material_type, exists in materials.items():
                status = "✅" if exists else "❌"
                report.append(f"  - {status} {material_type}")
                if not exists:
                    missing_materials[material_type] += 1
                    paper_missing.append(material_type)
            
            paper_details.append({
                'path': rel_path,
                'theorems': len(theorems),
                'missing': paper_missing
            })
    
    # Summary
    report.append("\n## Summary Statistics\n")
    report.append(f"- **Total Papers Audited:** {total_papers}")
    report.append(f"- **Total Theorems/Definitions Found:** {total_theorems}")
    report.append("\n**Missing Materials Count:**")
    for material_type, count in sorted(missing_materials.items(), key=lambda x: -x[1]):
        percentage = (count / total_papers * 100) if total_papers > 0 else 0
        report.append(f"  - {material_type}: {count} papers ({percentage:.1f}%)")
    
    # Papers needing attention
    report.append("\n## Papers Requiring Attention\n")
    papers_needing_work = [p for p in paper_details if p['missing']]
    if papers_needing_work:
        for paper in sorted(papers_needing_work, key=lambda x: -len(x['missing'])):
            report.append(f"- **{paper['path']}**")
            report.append(f"  - Missing: {', '.join(paper['missing'])}")
            report.append(f"  - Theorems: {paper['theorems']}")
    else:
        report.append("✅ All papers have complete supporting materials!")
    
    return "\n".join(report)

def create_standard_structure(paper_path: str):
    """Create standard supporting materials directory structure."""
    paper_dir = os.path.dirname(paper_path)
    supporting_dir = os.path.join(paper_dir, "supporting_materials")
    
    for material_type in REQUIRED_DIRS.keys():
        material_path = os.path.join(supporting_dir, material_type)
        os.makedirs(material_path, exist_ok=True)
        
        # Create README in each directory
        readme_path = os.path.join(material_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, 'w') as f:
                f.write(f"# {material_type.replace('_', ' ').title()}\n\n")
                f.write(f"This directory contains {material_type} for the paper.\n")

def main():
    """Main audit function."""
    print("🔍 Scanning for papers...")
    papers = find_all_papers(BASE_DIRS)
    
    total = sum(len(p) for p in papers.values())
    print(f"📄 Found {total} papers across {len(BASE_DIRS)} directories")
    
    print("\n📊 Generating audit report...")
    report = generate_audit_report(papers)
    
    # Save report
    report_path = "/Users/coo-koba42/dev/PAPER_AUDIT_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"✅ Report saved to: {report_path}")
    
    # Create standard structures for papers missing materials
    print("\n📁 Creating standard directory structures...")
    for base_dir, paper_list in papers.items():
        for paper_path in paper_list:
            materials = check_supporting_materials(paper_path)
            if not all(materials.values()):
                create_standard_structure(paper_path)
                print(f"  Created structure for: {os.path.basename(paper_path)}")
    
    print("\n✅ Audit complete!")
    return report_path

if __name__ == "__main__":
    main()

