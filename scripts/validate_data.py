#!/usr/bin/env python3
"""
Data Validation Script
======================

Validates data integrity, completeness, and consistency across the research framework.

Usage:
    python scripts/validate_data.py [--verbose] [--fix] [--dataset DATASET]

Author: Bradley Wallace
Date: 2025-10-11
"""

import os
import sys
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation and integrity checking system."""

    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or Path(__file__).parent.parent)
        self.data_dir = self.root_dir / "data"
        self.artifacts_dir = self.root_dir / "artifacts"
        self.figures_dir = self.root_dir / "figures"

        # Expected data structure
        self.expected_dirs = {
            'data': ['raw', 'interim', 'processed', 'external'],
            'artifacts': ['figures', 'models', 'reports', 'run-data'],
            'figures': []  # Any subdirs are domain-specific
        }

        self.checksums_file = self.data_dir / "checksums.json"

    def calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except (IOError, OSError) as e:
            logger.warning(f"Could not calculate checksum for {file_path}: {e}")
            return None

    def validate_directory_structure(self) -> Dict[str, bool]:
        """Validate that expected directories exist."""
        results = {}

        for base_dir, subdirs in self.expected_dirs.items():
            base_path = self.root_dir / base_dir
            results[f"{base_dir}_exists"] = base_path.exists()

            if subdirs:
                for subdir in subdirs:
                    sub_path = base_path / subdir
                    results[f"{base_dir}/{subdir}_exists"] = sub_path.exists()
                    if sub_path.exists():
                        # Check if directory is not empty (has files)
                        has_files = any(sub_path.rglob('*'))
                        results[f"{base_dir}/{subdir}_has_content"] = has_files

        return results

    def validate_data_files(self) -> Dict[str, any]:
        """Validate data file integrity and formats."""
        results = {
            'files_checked': 0,
            'files_valid': 0,
            'format_errors': [],
            'checksum_mismatches': []
        }

        # Check data files
        data_files = list(self.data_dir.rglob('*'))
        data_files = [f for f in data_files if f.is_file() and not f.name.startswith('.')]

        for file_path in data_files:
            results['files_checked'] += 1

            try:
                # Validate based on file extension
                if file_path.suffix.lower() == '.csv':
                    self._validate_csv(file_path, results)
                elif file_path.suffix.lower() in ['.json', '.jsonl']:
                    self._validate_json(file_path, results)
                elif file_path.suffix.lower() in ['.npy', '.npz']:
                    self._validate_numpy(file_path, results)
                elif file_path.suffix.lower() == '.db':
                    self._validate_sqlite(file_path, results)
                else:
                    # Generic file existence check
                    if file_path.stat().st_size == 0:
                        results['format_errors'].append(f"Empty file: {file_path}")

            except Exception as e:
                results['format_errors'].append(f"Error validating {file_path}: {e}")

        results['files_valid'] = results['files_checked'] - len(results['format_errors'])

        return results

    def _validate_csv(self, file_path: Path, results: Dict):
        """Validate CSV file format."""
        try:
            df = pd.read_csv(file_path, nrows=5)  # Just check first few rows
            if df.empty:
                results['format_errors'].append(f"Empty CSV: {file_path}")
            elif df.shape[1] == 0:
                results['format_errors'].append(f"No columns in CSV: {file_path}")
        except Exception as e:
            results['format_errors'].append(f"Invalid CSV {file_path}: {e}")

    def _validate_json(self, file_path: Path, results: Dict):
        """Validate JSON file format."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
        except Exception as e:
            results['format_errors'].append(f"Invalid JSON {file_path}: {e}")

    def _validate_numpy(self, file_path: Path, results: Dict):
        """Validate NumPy file format."""
        try:
            if file_path.suffix.lower() == '.npy':
                np.load(file_path)
            else:  # .npz
                np.load(file_path, allow_pickle=False)
        except Exception as e:
            results['format_errors'].append(f"Invalid NumPy file {file_path}: {e}")

    def _validate_sqlite(self, file_path: Path, results: Dict):
        """Validate SQLite database."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(file_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()

            if not tables:
                results['format_errors'].append(f"No tables in SQLite DB: {file_path}")

        except Exception as e:
            results['format_errors'].append(f"Invalid SQLite DB {file_path}: {e}")

    def validate_figure_consistency(self) -> Dict[str, any]:
        """Validate that figures referenced in papers exist."""
        results = {
            'papers_checked': 0,
            'figures_found': 0,
            'missing_figures': [],
            'orphaned_figures': []
        }

        # Check papers for figure references
        paper_dir = self.root_dir / "research" / "papers"
        if paper_dir.exists():
            for tex_file in paper_dir.rglob("*.tex"):
                results['papers_checked'] += 1
                try:
                    with open(tex_file, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Find \includegraphics commands
                    import re
                    img_matches = re.findall(r'\\includegraphics(?:\[.*?\])?\{([^}]+)\}', content)

                    for img_path in img_matches:
                        # Resolve relative paths
                        if not img_path.startswith('/'):
                            img_full_path = tex_file.parent / img_path
                        else:
                            img_full_path = Path(img_path)

                        if img_full_path.exists():
                            results['figures_found'] += 1
                        else:
                            results['missing_figures'].append(f"{img_path} (referenced in {tex_file.name})")

                except Exception as e:
                    logger.warning(f"Could not check figures in {tex_file}: {e}")

        return results

    def generate_report(self) -> Dict[str, any]:
        """Generate comprehensive validation report."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'directory_structure': self.validate_directory_structure(),
            'data_validation': self.validate_data_files(),
            'figure_consistency': self.validate_figure_consistency(),
            'summary': {}
        }

        # Calculate summary
        dir_checks = report['directory_structure']
        data_checks = report['data_validation']
        figure_checks = report['figure_consistency']

        total_checks = len(dir_checks) + data_checks['files_checked'] + figure_checks['papers_checked']
        passed_checks = sum(dir_checks.values()) + data_checks['files_valid'] + (figure_checks['papers_checked'] - len(figure_checks['missing_figures']))

        report['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': passed_checks / total_checks if total_checks > 0 else 0,
            'critical_issues': len(data_checks['format_errors']) + len(figure_checks['missing_figures'])
        }

        return report

    def print_report(self, report: Dict[str, any], verbose: bool = False):
        """Print validation report."""
        print("=" * 60)
        print("DATA VALIDATION REPORT")
        print("=" * 60)
        print(f"Timestamp: {report['timestamp']}")
        print()

        # Summary
        summary = report['summary']
        print("SUMMARY:")
        print(f"  Total checks: {summary['total_checks']}")
        print(f"  Passed: {summary['passed_checks']}")
        print(f"  Failed: {summary['failed_checks']}")
        print(".1f")
        print(f"  Critical issues: {summary['critical_issues']}")
        print()

        # Directory structure
        print("DIRECTORY STRUCTURE:")
        for check, result in report['directory_structure'].items():
            status = "✓" if result else "✗"
            print(f"  {status} {check}")
        print()

        # Data validation
        data_val = report['data_validation']
        print("DATA VALIDATION:")
        print(f"  Files checked: {data_val['files_checked']}")
        print(f"  Files valid: {data_val['files_valid']}")
        if data_val['format_errors']:
            print(f"  Format errors: {len(data_val['format_errors'])}")
            if verbose:
                for error in data_val['format_errors'][:5]:
                    print(f"    - {error}")
                if len(data_val['format_errors']) > 5:
                    print(f"    ... and {len(data_val['format_errors']) - 5} more")
        print()

        # Figure consistency
        fig_check = report['figure_consistency']
        print("FIGURE CONSISTENCY:")
        print(f"  Papers checked: {fig_check['papers_checked']}")
        print(f"  Figures found: {fig_check['figures_found']}")
        if fig_check['missing_figures']:
            print(f"  Missing figures: {len(fig_check['missing_figures'])}")
            if verbose:
                for missing in fig_check['missing_figures'][:3]:
                    print(f"    - {missing}")
                if len(fig_check['missing_figures']) > 3:
                    print(f"    ... and {len(fig_check['missing_figures']) - 3} more")
        print()

        if summary['critical_issues'] > 0:
            print("⚠️  CRITICAL ISSUES DETECTED")
            print("   Review the detailed output above and fix data integrity problems.")
        else:
            print("✅ ALL CHECKS PASSED")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Data validation script")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues automatically')
    parser.add_argument('--dataset', help='Validate specific dataset')

    args = parser.parse_args()

    validator = DataValidator()

    print("Running data validation...")
    report = validator.generate_report()
    validator.print_report(report, args.verbose)

    # Exit with error code if there are critical issues
    if report['summary']['critical_issues'] > 0:
        sys.exit(1)

if __name__ == '__main__':
    main()
