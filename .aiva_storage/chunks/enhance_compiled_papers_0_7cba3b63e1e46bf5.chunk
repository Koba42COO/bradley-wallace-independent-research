#!/usr/bin/env python3
"""
Enhance compiled papers with full LaTeX content extraction and better formatting.
"""

import os
import re
from pathlib import Path


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



def extract_full_latex_content(tex_path):
    """Extract and clean LaTeX content for reading."""
    try:
        with open(tex_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Remove document structure
        content = re.sub(r'\\documentclass.*?\n', '', content)
        content = re.sub(r'\\usepackage.*?\n', '', content)
        content = re.sub(r'\\begin\{document\}', '', content)
        content = re.sub(r'\\end\{document\}', '', content)
        content = re.sub(r'\\maketitle', '', content)
        content = re.sub(r'\\tableofcontents', '', content)
        
        # Convert sections to markdown
        content = re.sub(r'\\section\{([^}]+)\}', r'## \1', content)
        content = re.sub(r'\\subsection\{([^}]+)\}', r'### \1', content)
        content = re.sub(r'\\subsubsection\{([^}]+)\}', r'#### \1', content)
        
        # Convert equations
        content = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', r'$$\1$$', content, flags=re.DOTALL)
        content = re.sub(r'\$([^$]+)\$', r'$\1$', content)
        
        # Convert itemize/enumerate
        content = re.sub(r'\\begin\{itemize\}', '', content)
        content = re.sub(r'\\end\{itemize\}', '', content)
        content = re.sub(r'\\begin\{enumerate\}', '', content)
        content = re.sub(r'\\end\{enumerate\}', '', content)
        content = re.sub(r'\\item\s+', '- ', content)
        
        # Convert text formatting
        content = re.sub(r'\\textbf\{([^}]+)\}', r'**\1**', content)
        content = re.sub(r'\\textit\{([^}]+)\}', r'*\1*', content)
        content = re.sub(r'\\emph\{([^}]+)\}', r'*\1*', content)
        
        # Clean up LaTeX commands
        content = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', content)
        content = re.sub(r'\\[a-zA-Z]+', '', content)
        
        # Clean up extra whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    except Exception as e:
        return f"Error extracting content: {e}"

def enhance_compiled_paper(compiled_path, tex_path):
    """Enhance a compiled paper with full content."""
    paper_name = os.path.splitext(os.path.basename(tex_path))[0]
    
    # Read existing compiled paper
    with open(compiled_path, 'r', encoding='utf-8') as f:
        compiled_content = f.read()
    
    # Extract full LaTeX content
    full_content = extract_full_latex_content(tex_path)
    
    # Find where to insert full content (after abstract)
    if "## Paper Overview" in compiled_content:
        # Insert full content section before Paper Overview
        insert_pos = compiled_content.find("## Paper Overview")
        enhanced = compiled_content[:insert_pos]
        enhanced += "## Full Paper Content\n\n"
        enhanced += "<details>\n<summary>Click to expand full paper content</summary>\n\n"
        enhanced += full_content[:50000]  # Limit to 50k chars
        if len(full_content) > 50000:
            enhanced += "\n\n*[Content truncated - see source .tex file for complete paper]*\n"
        enhanced += "\n\n</details>\n\n"
        enhanced += "---\n\n"
        enhanced += compiled_content[insert_pos:]
    else:
        enhanced = compiled_content
    
    # Write enhanced version
    with open(compiled_path, 'w', encoding='utf-8') as f:
        f.write(enhanced)

def main():
    """Enhance all compiled papers."""
    compiled_dir = "/Users/coo-koba42/dev/bradley-wallace-independent-research/compiled_papers"
    papers = find_all_papers()
    
    print("🔧 Enhancing compiled papers with full content...")
    
    for paper_path in papers:
        paper_name = os.path.splitext(os.path.basename(paper_path))[0]
        compiled_path = os.path.join(compiled_dir, f"{paper_name}_COMPILED.md")
        
        if os.path.exists(compiled_path):
            enhance_compiled_paper(compiled_path, paper_path)
            print(f"✅ Enhanced: {paper_name}")
    
    print("\n✅ All papers enhanced!")

if __name__ == '__main__':
    from compile_papers_for_reading import find_all_papers
    main()

