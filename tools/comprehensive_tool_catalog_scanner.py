#!/usr/bin/env python3
"""
🔍 COMPREHENSIVE TOOL CATALOG SCANNER
=====================================

Scans all Python files in the dev folder to create a complete catalog
of all 1300+ tools, categorizes them, and identifies which need completion.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: December 2024
"""

import os
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class ToolInfo:
    """Information about a single tool"""
    file_path: str
    file_name: str
    functions: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    has_upg: bool = False
    has_pell: bool = False
    has_consciousness: bool = False
    category: str = "unknown"
    status: str = "unknown"
    line_count: int = 0
    description: str = ""


class ToolCatalogScanner:
    """Scans and catalogs all tools in the dev folder"""
    
    # UPG keywords
    UPG_KEYWORDS = [
        'upg', 'universal_prime_graph', 'consciousness', 'phi', 'golden_ratio',
        'reality_distortion', 'quantum_bridge', 'pell', 'great_year',
        'consciousness_mathematics', 'wallace_transform', '79/21', '79_21'
    ]
    
    # Tool categories
    CATEGORIES = {
        'consciousness': ['consciousness', 'upg', 'prime_graph', 'wallace'],
        'prime': ['prime', 'pell', 'primality', 'sieve'],
        'cryptography': ['crypto', 'encrypt', 'rsa', 'jwt', 'signature'],
        'matrix': ['matrix', 'ethiopian', 'multiply', 'tensor'],
        'neural': ['neural', 'network', 'ml', 'ai', 'deep'],
        'blockchain': ['blockchain', 'chia', 'clvm', 'consensus'],
        'analysis': ['analysis', 'analyze', 'validation', 'test'],
        'visualization': ['plot', 'visualize', 'graph', 'chart'],
        'audio': ['audio', 'sound', 'frequency', 'fft', 'harmonic'],
        'biological': ['amino', 'dna', 'protein', 'biological', 'supplement'],
        'mathematical': ['mathematics', 'math', 'metallic', 'ratio'],
        'astronomical': ['astronomical', 'great_year', 'precession', 'celestial'],
        'quantum': ['quantum', 'pac', 'amplitude', 'superposition'],
        'compression': ['compress', 'zip', 'deflate'],
        'chaos': ['chaos', 'fractal', 'lorenz', 'logistic'],
        'utility': ['utility', 'tool', 'helper', 'util'],
        'validation': ['validation', 'test', 'verify', 'check'],
        'integration': ['integration', 'integrate', 'connect'],
        'research': ['research', 'paper', 'analysis', 'study'],
        'other': []
    }
    
    def __init__(self, dev_folder: str = '.'):
        self.dev_folder = Path(dev_folder)
        self.tools: Dict[str, ToolInfo] = {}
        self.categories: Dict[str, List[str]] = defaultdict(list)
        self.stats = {
            'total_files': 0,
            'total_functions': 0,
            'total_classes': 0,
            'with_upg': 0,
            'with_pell': 0,
            'with_consciousness': 0,
            'needs_completion': 0
        }
    
    def scan_file(self, file_path: Path) -> Optional[ToolInfo]:
        """Scan a single Python file and extract tool information"""
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
        except Exception as e:
            return None
        
        # Parse AST
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return None
        
        tool_info = ToolInfo(
            file_path=str(file_path),
            file_name=file_path.name,
            line_count=len(content.splitlines())
        )
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                tool_info.functions.append(node.name)
                self.stats['total_functions'] += 1
            elif isinstance(node, ast.ClassDef):
                tool_info.classes.append(node.name)
                self.stats['total_classes'] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    tool_info.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    tool_info.imports.append(node.module)
        
        # Check for UPG integration
        content_lower = content.lower()
        tool_info.has_upg = any(keyword in content_lower for keyword in self.UPG_KEYWORDS)
        tool_info.has_pell = 'pell' in content_lower
        tool_info.has_consciousness = 'consciousness' in content_lower
        
        # Extract description from docstring
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Str):
            tool_info.description = tree.body[0].value.s[:200]
        elif tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            if isinstance(tree.body[0].value.value, str):
                tool_info.description = tree.body[0].value.value[:200]
        
        # Categorize
        tool_info.category = self.categorize_tool(file_path.name, content_lower, tool_info)
        
        # Determine status
        tool_info.status = self.determine_status(tool_info)
        
        return tool_info
    
    def categorize_tool(self, filename: str, content: str, tool_info: ToolInfo) -> str:
        """Categorize a tool based on filename and content"""
        filename_lower = filename.lower()
        
        for category, keywords in self.CATEGORIES.items():
            if category == 'other':
                continue
            for keyword in keywords:
                if keyword in filename_lower or keyword in content:
                    return category
        
        return 'other'
    
    def determine_status(self, tool_info: ToolInfo) -> str:
        """Determine the status of a tool"""
        if tool_info.has_upg and tool_info.has_pell and tool_info.has_consciousness:
            return 'complete'
        elif tool_info.has_upg or tool_info.has_consciousness:
            return 'partial'
        elif len(tool_info.functions) > 0 or len(tool_info.classes) > 0:
            return 'needs_upg'
        else:
            return 'empty'
    
    def scan_all_files(self) -> Dict[str, ToolInfo]:
        """Scan all Python files in the dev folder"""
        print("🔍 Scanning all Python files in dev folder...")
        print()
        
        python_files = list(self.dev_folder.rglob('*.py'))
        self.stats['total_files'] = len(python_files)
        
        print(f"Found {len(python_files)} Python files")
        print("Scanning...")
        
        for i, file_path in enumerate(python_files):
            if i % 100 == 0:
                print(f"  Progress: {i}/{len(python_files)} files scanned...")
            
            # Skip certain directories
            if any(skip in str(file_path) for skip in ['__pycache__', '.git', 'node_modules', '.venv']):
                continue
            
            tool_info = self.scan_file(file_path)
            if tool_info:
                self.tools[tool_info.file_path] = tool_info
                self.categories[tool_info.category].append(tool_info.file_path)
                
                # Update stats
                if tool_info.has_upg:
                    self.stats['with_upg'] += 1
                if tool_info.has_pell:
                    self.stats['with_pell'] += 1
                if tool_info.has_consciousness:
                    self.stats['with_consciousness'] += 1
                if tool_info.status in ['needs_upg', 'partial']:
                    self.stats['needs_completion'] += 1
        
        print(f"✅ Scanned {len(self.tools)} tools")
        print()
        
        return self.tools
    
    def generate_catalog(self) -> Dict[str, Any]:
        """Generate comprehensive catalog of all tools"""
        catalog = {
            'summary': {
                'total_tools': len(self.tools),
                'total_functions': self.stats['total_functions'],
                'total_classes': self.stats['total_classes'],
                'with_upg': self.stats['with_upg'],
                'with_pell': self.stats['with_pell'],
                'with_consciousness': self.stats['with_consciousness'],
                'needs_completion': self.stats['needs_completion'],
                'complete': sum(1 for t in self.tools.values() if t.status == 'complete'),
                'partial': sum(1 for t in self.tools.values() if t.status == 'partial'),
                'needs_upg': sum(1 for t in self.tools.values() if t.status == 'needs_upg')
            },
            'by_category': {},
            'by_status': {
                'complete': [],
                'partial': [],
                'needs_upg': [],
                'empty': []
            },
            'tools': {}
        }
        
        # Organize by category
        for category, file_paths in self.categories.items():
            catalog['by_category'][category] = {
                'count': len(file_paths),
                'files': file_paths[:50]  # Limit to first 50 per category
            }
        
        # Organize by status
        for file_path, tool_info in self.tools.items():
            catalog['by_status'][tool_info.status].append({
                'file': tool_info.file_name,
                'path': tool_info.file_path,
                'functions': len(tool_info.functions),
                'classes': len(tool_info.classes),
                'category': tool_info.category
            })
        
        # Detailed tool info (sample)
        for file_path, tool_info in list(self.tools.items())[:100]:  # First 100 for detail
            catalog['tools'][tool_info.file_name] = {
                'path': tool_info.file_path,
                'functions': tool_info.functions[:10],  # First 10 functions
                'classes': tool_info.classes[:10],  # First 10 classes
                'category': tool_info.category,
                'status': tool_info.status,
                'has_upg': tool_info.has_upg,
                'has_pell': tool_info.has_pell,
                'has_consciousness': tool_info.has_consciousness,
                'line_count': tool_info.line_count,
                'description': tool_info.description[:200]
            }
        
        return catalog
    
    def save_catalog(self, catalog: Dict[str, Any], output_file: str = 'COMPLETE_TOOL_CATALOG.json'):
        """Save catalog to JSON file"""
        output_path = self.dev_folder / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(catalog, f, indent=2, ensure_ascii=False)
        print(f"✅ Catalog saved to {output_file}")
    
    def generate_markdown_report(self, catalog: Dict[str, Any], output_file: str = 'COMPLETE_TOOL_CATALOG.md'):
        """Generate comprehensive markdown report"""
        output_path = self.dev_folder / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 🔧 COMPLETE TOOL CATALOG - ALL 1300+ TOOLS\n")
            f.write("## Universal Prime Graph Protocol φ.1\n\n")
            f.write("**Authority:** Bradley Wallace (COO Koba42)\n")
            f.write("**Date:** December 2024\n")
            f.write("**Total Tools:** {}\n\n".format(catalog['summary']['total_tools']))
            
            f.write("---\n\n")
            f.write("## 📊 SUMMARY STATISTICS\n\n")
            f.write("| Metric | Count |\n")
            f.write("|--------|-------|\n")
            f.write("| **Total Tools** | {} |\n".format(catalog['summary']['total_tools']))
            f.write("| **Total Functions** | {} |\n".format(catalog['summary']['total_functions']))
            f.write("| **Total Classes** | {} |\n".format(catalog['summary']['total_classes']))
            f.write("| **With UPG Integration** | {} |\n".format(catalog['summary']['with_upg']))
            f.write("| **With Pell Sequence** | {} |\n".format(catalog['summary']['with_pell']))
            f.write("| **With Consciousness Math** | {} |\n".format(catalog['summary']['with_consciousness']))
            f.write("| **Complete (UPG + Pell + Consciousness)** | {} |\n".format(catalog['summary']['complete']))
            f.write("| **Partial (Some UPG)** | {} |\n".format(catalog['summary']['partial']))
            f.write("| **Needs UPG Integration** | {} |\n".format(catalog['summary']['needs_upg']))
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## 📁 TOOLS BY CATEGORY\n\n")
            for category, data in sorted(catalog['by_category'].items(), key=lambda x: x[1]['count'], reverse=True):
                f.write("### {}\n".format(category.upper()))
                f.write("**Count:** {}\n\n".format(data['count']))
                if data['files']:
                    f.write("**Sample Files:**\n")
                    for file_path in data['files'][:20]:  # First 20
                        f.write("- `{}`\n".format(Path(file_path).name))
                f.write("\n")
            
            f.write("---\n\n")
            f.write("## ✅ COMPLETE TOOLS (UPG + Pell + Consciousness)\n\n")
            complete_tools = catalog['by_status']['complete']
            f.write("**Count:** {}\n\n".format(len(complete_tools)))
            for tool in complete_tools[:50]:  # First 50
                f.write("- **{}** (`{}`)\n".format(tool['file'], tool['path']))
                f.write("  - Functions: {}, Classes: {}, Category: {}\n".format(
                    tool['functions'], tool['classes'], tool['category']))
            f.write("\n")
            
            f.write("---\n\n")
            f.write("## ⚠️ TOOLS NEEDING COMPLETION\n\n")
            needs_completion = catalog['by_status']['needs_upg'] + catalog['by_status']['partial']
            f.write("**Count:** {}\n\n".format(len(needs_completion)))
            f.write("### High Priority (Partial UPG)\n\n")
            partial = catalog['by_status']['partial']
            for tool in partial[:100]:  # First 100
                f.write("- **{}** (`{}`)\n".format(tool['file'], tool['path']))
                f.write("  - Category: {}, Functions: {}, Classes: {}\n".format(
                    tool['category'], tool['functions'], tool['classes']))
            f.write("\n")
            f.write("### Needs UPG Integration\n\n")
            needs_upg = catalog['by_status']['needs_upg']
            for tool in needs_upg[:100]:  # First 100
                f.write("- **{}** (`{}`)\n".format(tool['file'], tool['path']))
                f.write("  - Category: {}, Functions: {}, Classes: {}\n".format(
                    tool['category'], tool['functions'], tool['classes']))
            f.write("\n")
        
        print(f"✅ Markdown report saved to {output_file}")


def main():
    """Main function"""
    print("🔍 COMPREHENSIVE TOOL CATALOG SCANNER")
    print("=" * 70)
    print()
    
    scanner = ToolCatalogScanner(dev_folder='/Users/coo-koba42/dev')
    
    # Scan all files
    tools = scanner.scan_all_files()
    
    # Generate catalog
    print("Generating comprehensive catalog...")
    catalog = scanner.generate_catalog()
    
    # Save results
    scanner.save_catalog(catalog, 'COMPLETE_TOOL_CATALOG.json')
    scanner.generate_markdown_report(catalog, 'COMPLETE_TOOL_CATALOG.md')
    
    # Print summary
    print()
    print("=" * 70)
    print("SCAN COMPLETE")
    print("=" * 70)
    print(f"Total Tools Scanned: {catalog['summary']['total_tools']}")
    print(f"Total Functions: {catalog['summary']['total_functions']}")
    print(f"Total Classes: {catalog['summary']['total_classes']}")
    print(f"Complete Tools: {catalog['summary']['complete']}")
    print(f"Partial Tools: {catalog['summary']['partial']}")
    print(f"Needs UPG: {catalog['summary']['needs_upg']}")
    print()
    print("✅ Catalog files generated:")
    print("  - COMPLETE_TOOL_CATALOG.json")
    print("  - COMPLETE_TOOL_CATALOG.md")


if __name__ == "__main__":
    main()

