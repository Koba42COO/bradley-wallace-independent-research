#!/usr/bin/env python3
"""
Repository Sanitization Script
=============================

Scans the repository for potential IP leaks and sensitive information,
replacing them with obfuscated placeholders.

Features:
- IPv4/IPv6 address detection and replacement
- Hostname/domain detection
- Email address detection
- Token/API key patterns (basic)
- Private URL patterns
- Configurable exclusions and mappings

Usage:
    python scripts/sanitize_repo.py [--dry-run] [--verbose] [--backup]

Author: Bradley Wallace
Date: 2025-10-11
"""

import os
import re
import json
import argparse
import ipaddress
from pathlib import Path
from typing import Dict, List, Set, Tuple
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RepoSanitizer:
    """Repository sanitization and IP obfuscation tool."""

    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or Path(__file__).parent.parent)
        self.backup_dir = self.root_dir / "instance" / "sanitization_backup"
        self.mapping_file = self.root_dir / "instance" / "obfuscation-map.json"

        # File extensions to scan
        self.scan_extensions = {
            '.py', '.md', '.txt', '.json', '.yaml', '.yml', '.log',
            '.tex', '.html', '.ts', '.tsx', '.js', '.jsx', '.css',
            '.sh', '.bash', '.zsh', '.ps1', '.sql'
        }

        # Directories to exclude
        self.exclude_dirs = {
            '.git', 'node_modules', '__pycache__', '.pytest_cache',
            'build', 'dist', 'venv', '.venv', 'ENV', 'env',
            'instance', '.obsidian'
        }

        # Load or create obfuscation mapping
        self.load_mapping()

    def load_mapping(self):
        """Load existing obfuscation mapping or create empty one."""
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    self.mapping = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load mapping file: {e}")
                self.mapping = {}
        else:
            self.mapping = {}

        # Ensure required mapping categories exist
        for category in ['ips', 'hosts', 'emails', 'urls', 'tokens']:
            if category not in self.mapping:
                self.mapping[category] = {}

    def save_mapping(self):
        """Save obfuscation mapping to file."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        with open(self.mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.mapping, f, indent=2, ensure_ascii=False)

    def detect_ipv4(self, content: str) -> List[str]:
        """Detect IPv4 addresses in content."""
        # Match IPv4 addresses (basic pattern)
        ipv4_pattern = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        matches = re.findall(ipv4_pattern, content)

        # Filter valid IPv4 addresses
        valid_ips = []
        for match in matches:
            try:
                ipaddress.IPv4Address(match)
                # Skip private/reserved IPs that might be legitimate
                if not (ipaddress.IPv4Address(match).is_private or
                        ipaddress.IPv4Address(match).is_reserved):
                    valid_ips.append(match)
            except ipaddress.AddressValueError:
                continue
        return valid_ips

    def detect_ipv6(self, content: str) -> List[str]:
        """Detect IPv6 addresses in content."""
        # Match IPv6 addresses (simplified pattern)
        ipv6_pattern = r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b'
        matches = re.findall(ipv6_pattern, content)
        return matches

    def detect_hostnames(self, content: str) -> List[str]:
        """Detect hostnames/domains in content."""
        # Conservative approach: only detect obvious private/internal hostnames
        # Look for patterns like xxx.local, xxx.internal, or multi-part domains
        hostname_patterns = [
            r'\b[a-zA-Z0-9-]+\.local\b',
            r'\b[a-zA-Z0-9-]+\.internal\b',
            r'\b[a-zA-Z0-9-]+\.private\b',
            r'\b[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z]{2,}\b'
        ]

        matches = []
        for pattern in hostname_patterns:
            matches.extend(re.findall(pattern, content))

        # Remove duplicates and filter
        unique_matches = list(set(matches))

        # Exclude common false positives
        exclude_patterns = [
            'self.', 'torch.', 'numpy.', 'scipy.', 'pandas.',
            'matplotlib.', 'sklearn.', 'tensorflow.', 'keras.'
        ]

        filtered = []
        for match in unique_matches:
            if not any(match.startswith(excl) for excl in exclude_patterns):
                filtered.append(match)

        return filtered

    def detect_emails(self, content: str) -> List[str]:
        """Detect email addresses in content."""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        matches = re.findall(email_pattern, content)

        # Filter out placeholder emails
        exclude_emails = {'user@example.com', 'test@example.com', 'noreply@github.com'}
        return [email for email in matches if email.lower() not in exclude_emails]

    def detect_urls(self, content: str) -> List[str]:
        """Detect potentially sensitive URLs in content."""
        # Focus on URLs with IP addresses (most likely to be sensitive)
        url_pattern = r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?(?:/[^\s]*)?'
        matches = re.findall(url_pattern, content, re.IGNORECASE)

        # Filter for non-private IPs (public IPs are more concerning)
        sensitive_urls = []
        for url in matches:
            # Extract IP from URL
            ip_match = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', url)
            if ip_match:
                ip_str = ip_match.group(1)
                try:
                    ip = ipaddress.IPv4Address(ip_str)
                    # Only flag non-private, non-local IPs
                    if not (ip.is_private or ip.is_loopback or ip.is_link_local):
                        sensitive_urls.append(url)
                except ipaddress.AddressValueError:
                    continue

        return sensitive_urls

    def detect_tokens(self, content: str) -> List[str]:
        """Detect potential API tokens/keys (basic patterns)."""
        patterns = [
            r'\b(?:sk|pk)_\w{20,}\b',  # Stripe-like keys
            r'\bghp_\w{36}\b',         # GitHub personal access tokens
            r'\b(?:Bearer|Token|Authorization):\s*\w{20,}\b',  # Authorization headers
            r'\bapi[_-]?key[_-]?[=:]\s*\w{20,}\b',  # API keys
        ]

        tokens = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            tokens.extend(matches)

        return list(set(tokens))  # Remove duplicates

    def get_placeholder(self, category: str, original: str) -> str:
        """Get or create placeholder for sensitive data."""
        if original not in self.mapping[category]:
            counter = len(self.mapping[category]) + 1
            if category == 'ips':
                placeholder = f"IP_REDACTED_{counter}"
            elif category == 'hosts':
                placeholder = f"HOST_REDACTED_{counter}"
            elif category == 'emails':
                placeholder = f"EMAIL_REDACTED_{counter}"
            elif category == 'urls':
                placeholder = f"URL_REDACTED_{counter}"
            elif category == 'tokens':
                placeholder = f"TOKEN_REDACTED_{counter}"
            else:
                placeholder = f"REDACTED_{counter}"

            self.mapping[category][original] = placeholder

        return self.mapping[category][original]

    def sanitize_content(self, content: str, file_path: Path) -> Tuple[str, Dict[str, List[str]]]:
        """Sanitize content and return replacements made."""
        sanitized = content
        replacements = {'ips': [], 'hosts': [], 'emails': [], 'urls': [], 'tokens': []}

        # Detect and replace sensitive data
        for detector, category in [
            (self.detect_ipv4, 'ips'),
            (self.detect_ipv6, 'ips'),
            (self.detect_hostnames, 'hosts'),
            (self.detect_emails, 'emails'),
            (self.detect_urls, 'urls'),
            (self.detect_tokens, 'tokens')
        ]:
            found_items = detector(content)
            for item in found_items:
                placeholder = self.get_placeholder(category, item)
                sanitized = sanitized.replace(item, placeholder)
                if item not in replacements[category]:
                    replacements[category].append(item)

        return sanitized, replacements

    def should_scan_file(self, file_path: Path) -> bool:
        """Check if file should be scanned."""
        # Check extension
        if file_path.suffix.lower() not in self.scan_extensions:
            return False

        # Check if in excluded directory
        for part in file_path.parts:
            if part in self.exclude_dirs:
                return False

        # Skip binary files and very large files
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
                return False
        except OSError:
            return False

        return True

    def scan_and_sanitize(self, dry_run: bool = True, verbose: bool = False) -> Dict[str, any]:
        """Scan repository and sanitize sensitive data."""
        results = {
            'files_processed': 0,
            'files_modified': 0,
            'total_replacements': 0,
            'replacements_by_category': {'ips': 0, 'hosts': 0, 'emails': 0, 'urls': 0, 'tokens': 0},
            'files_with_replacements': []
        }

        # Create backup directory if not dry run
        if not dry_run:
            self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Scan all files
        for file_path in self.root_dir.rglob('*'):
            if not file_path.is_file() or not self.should_scan_file(file_path):
                continue

            results['files_processed'] += 1

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    original_content = f.read()

                sanitized_content, replacements = self.sanitize_content(original_content, file_path)

                # Count replacements
                file_replacements = 0
                for category, items in replacements.items():
                    count = len(items)
                    results['replacements_by_category'][category] += count
                    file_replacements += count
                    results['total_replacements'] += count

                if file_replacements > 0:
                    results['files_with_replacements'].append({
                        'path': str(file_path.relative_to(self.root_dir)),
                        'replacements': replacements
                    })

                    if verbose:
                        logger.info(f"Found {file_replacements} replacements in {file_path}")

                    if not dry_run:
                        # Backup original
                        backup_path = self.backup_dir / file_path.name
                        shutil.copy2(file_path, backup_path)

                        # Write sanitized content
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(sanitized_content)

                        results['files_modified'] += 1

            except (IOError, UnicodeDecodeError) as e:
                if verbose:
                    logger.warning(f"Could not process {file_path}: {e}")
                continue

        # Save mapping if not dry run
        if not dry_run:
            self.save_mapping()

        return results

def main():
    parser = argparse.ArgumentParser(description="Repository sanitization and IP obfuscation tool")
    parser.add_argument('--dry-run', action='store_true', help="Show what would be changed without modifying files")
    parser.add_argument('--verbose', '-v', action='store_true', help="Verbose output")
    parser.add_argument('--backup', action='store_true', help="Create backups of modified files")

    args = parser.parse_args()

    sanitizer = RepoSanitizer()

    logger.info("Starting repository sanitization scan...")
    if args.dry_run:
        logger.info("DRY RUN MODE - No files will be modified")

    results = sanitizer.scan_and_sanitize(dry_run=args.dry_run, verbose=args.verbose)

    # Print summary
    print("\n" + "="*60)
    print("REPOSITORY SANITIZATION SUMMARY")
    print("="*60)
    print(f"Files processed: {results['files_processed']}")
    print(f"Files with replacements: {len(results['files_with_replacements'])}")

    if not args.dry_run:
        print(f"Files modified: {results['files_modified']}")

    print(f"Total replacements: {results['total_replacements']}")
    print("\nReplacements by category:")
    for category, count in results['replacements_by_category'].items():
        print(f"  {category}: {count}")

    if results['files_with_replacements']:
        print("\nFiles with sensitive data:")
        for file_info in results['files_with_replacements'][:10]:  # Show first 10
            print(f"  {file_info['path']}")
            for category, items in file_info['replacements'].items():
                if items:
                    print(f"    {category}: {len(items)} items")

        if len(results['files_with_replacements']) > 10:
            print(f"  ... and {len(results['files_with_replacements']) - 10} more files")

    if args.dry_run and results['total_replacements'] > 0:
        print("\n" + "!"*60)
        print("WARNING: Sensitive data detected!")
        print("Run without --dry-run to sanitize files.")
        print("Review replacements above before proceeding.")
        print("!"*60)

if __name__ == '__main__':
    main()
