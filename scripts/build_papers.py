#!/usr/bin/env python3
"""
Paper Build System
==================

Automated LaTeX compilation for research papers.
Handles dependencies, bibliography, and multi-pass compilation.

Usage:
    python scripts/build_papers.py [--paper PAPER_NAME] [--all] [--clean]

Author: Bradley Wallace
Date: 2025-10-11
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path
from typing import List, Dict

class PaperBuilder:
    """LaTeX paper compilation system."""

    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or Path(__file__).parent.parent)
        self.papers_dir = self.root_dir / "research" / "papers"
        self.output_dir = self.root_dir / "artifacts" / "papers"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_papers(self) -> List[Path]:
        """Find all LaTeX papers in the repository."""
        papers = []
        for tex_file in self.papers_dir.rglob("*.tex"):
            if tex_file.name != "base_paper_template.tex":  # Skip template
                papers.append(tex_file)
        return sorted(papers)

    def compile_paper(self, tex_file: Path, verbose: bool = False) -> bool:
        """Compile a single LaTeX paper."""
        original_dir = os.getcwd()
        paper_dir = tex_file.parent
        paper_name = tex_file.stem

        try:
            os.chdir(paper_dir)

            # Multi-pass compilation for bibliography
            cmd = ["pdflatex", "-interaction=nonstopmode", str(tex_file.name)]

            if verbose:
                print(f"Compiling {tex_file.name}...")

            # First pass
            result1 = subprocess.run(cmd, capture_output=not verbose)

            # BibTeX if .bib exists
            bib_file = paper_dir / f"{paper_name}.bib"
            if bib_file.exists():
                if verbose:
                    print(f"Running BibTeX for {paper_name}...")
                subprocess.run(["bibtex", paper_name], capture_output=not verbose)

                # Second pass after BibTeX
                result2 = subprocess.run(cmd, capture_output=not verbose)

            # Third pass for cross-references
            result3 = subprocess.run(cmd, capture_output=not verbose)

            # Check for errors
            success = result1.returncode == 0 and result3.returncode == 0

            if success:
                # Copy PDF to artifacts
                pdf_file = paper_dir / f"{paper_name}.pdf"
                if pdf_file.exists():
                    target_pdf = self.output_dir / f"{paper_name}.pdf"
                    shutil.copy2(pdf_file, target_pdf)
                    if verbose:
                        print(f"✓ {paper_name}.pdf created successfully")
                else:
                    print(f"✗ PDF not found for {paper_name}")
                    success = False
            else:
                print(f"✗ Compilation failed for {paper_name}")
                if not verbose:
                    # Show error output
                    print("LaTeX errors:")
                    print(result1.stderr.decode() if result1.stderr else "No error output")

            return success

        except Exception as e:
            print(f"✗ Error compiling {paper_name}: {e}")
            return False

        finally:
            os.chdir(original_dir)

    def clean_artifacts(self, paper_name: str = None):
        """Clean LaTeX auxiliary files."""
        patterns = ['*.aux', '*.log', '*.bbl', '*.blg', '*.fdb_latexmk',
                   '*.fls', '*.synctex.gz', '*.toc', '*.out', '*.nav',
                   '*.snm', '*.pdf']

        if paper_name:
            # Clean specific paper
            paper_dir = self.papers_dir / paper_name
            if paper_dir.exists():
                for pattern in patterns:
                    for file in paper_dir.glob(pattern):
                        file.unlink()
                print(f"Cleaned artifacts for {paper_name}")
        else:
            # Clean all papers
            for pattern in patterns[:-1]:  # Don't delete PDFs in paper dirs
                for file in self.papers_dir.rglob(pattern):
                    file.unlink()
            print("Cleaned all LaTeX artifacts")

    def build_all_papers(self, verbose: bool = False) -> Dict[str, bool]:
        """Build all papers in the repository."""
        papers = self.find_papers()
        results = {}

        print(f"Found {len(papers)} papers to build")

        for i, paper in enumerate(papers, 1):
            paper_name = paper.relative_to(self.papers_dir).parent / paper.stem
            print(f"[{i}/{len(papers)}] Building {paper_name}")

            success = self.compile_paper(paper, verbose)
            results[str(paper_name)] = success

        return results

    def build_specific_paper(self, paper_name: str, verbose: bool = False) -> bool:
        """Build a specific paper by name."""
        papers = self.find_papers()

        # Find matching paper
        for paper in papers:
            if paper_name in str(paper):
                return self.compile_paper(paper, verbose)

        print(f"Paper '{paper_name}' not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="LaTeX paper build system")
    parser.add_argument("--paper", help="Build specific paper by name")
    parser.add_argument("--all", action="store_true", help="Build all papers")
    parser.add_argument("--clean", action="store_true", help="Clean LaTeX artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--list", action="store_true", help="List available papers")

    args = parser.parse_args()

    builder = PaperBuilder()

    if args.list:
        papers = builder.find_papers()
        print("Available papers:")
        for paper in papers:
            relative_path = paper.relative_to(builder.papers_dir)
            print(f"  {relative_path}")
        return

    if args.clean:
        if args.paper:
            builder.clean_artifacts(args.paper)
        else:
            builder.clean_artifacts()
        return

    if args.paper:
        success = builder.build_specific_paper(args.paper, args.verbose)
        sys.exit(0 if success else 1)

    if args.all:
        results = builder.build_all_papers(args.verbose)

        # Summary
        successful = sum(results.values())
        total = len(results)

        print(f"\nBuild Summary: {successful}/{total} papers compiled successfully")

        if successful < total:
            print("Failed papers:")
            for paper, success in results.items():
                if not success:
                    print(f"  ✗ {paper}")

        sys.exit(0 if successful == total else 1)

    # Default: show help
    parser.print_help()

if __name__ == '__main__':
    main()
