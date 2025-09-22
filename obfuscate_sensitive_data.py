#!/usr/bin/env python3
"""
SquashPlot Sensitive Data Obfuscation Script
============================================

This script obfuscates sensitive information in the codebase before
pushing to GitHub. It handles:
- IP addresses
- API keys
- Private keys
- Passwords
- Personal information
- Network configurations
"""

import os
import re
import hashlib
from pathlib import Path

def obfuscate_ip_addresses(content):
    """Obfuscate IP addresses in content"""
    # Replace real IP addresses with placeholder
    patterns = [
        (r'\b192\.168\.\d+\.\d+\b', '192.168.xxx.xxx'),
        (r'\b10\.\d+\.\d+\.\d+\b', '10.xxx.xxx.xxx'),
        (r'\b172\.\d+\.\d+\.\d+\b', '172.xxx.xxx.xxx'),
        (r'\b127\.0\.0\.1\b', '127.0.0.1'),  # Keep localhost as is
        (r'\blocalhost\b', 'localhost'),  # Keep localhost as is
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)

    return content

def obfuscate_api_keys(content):
    """Obfuscate API keys and tokens"""
    patterns = [
        (r'api_key\s*=\s*["\'][^"\']+["\']', 'api_key = "OBFUSCATED_API_KEY"'),
        (r'secret_key\s*=\s*["\'][^"\']+["\']', 'secret_key = "OBFUSCATED_SECRET_KEY"'),
        (r'token\s*=\s*["\'][^"\']+["\']', 'token = "OBFUSCATED_TOKEN"'),
        (r'password\s*=\s*["\'][^"\']+["\']', 'password = "OBFUSCATED_PASSWORD"'),
        (r'private_key\s*=\s*["\'][^"\']+["\']', 'private_key = "OBFUSCATED_PRIVATE_KEY"'),
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    return content

def obfuscate_personal_data(content):
    """Obfuscate personal information"""
    patterns = [
        (r'\b\d{3}-\d{2}-\d{4}\b', 'XXX-XX-XXXX'),  # SSN
        (r'\b\d{3}-\d{3}-\d{4}\b', 'XXX-XXX-XXXX'),  # Phone
        (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'user@domain.com'),  # Email
        (r'\b\d{4}\s+\w+\s+\w+\b', 'YYYY STREET NAME'),  # Address
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    return content

def obfuscate_wallet_data(content):
    """Obfuscate wallet and seed information"""
    patterns = [
        (r'seed\s*=\s*["\'][^"\']+["\']', 'seed = "OBFUSCATED_SEED"'),
        (r'wallet_key\s*=\s*["\'][^"\']+["\']', 'wallet_key = "OBFUSCATED_WALLET_KEY"'),
        (r'mnemonic\s*=\s*["\'][^"\']+["\']', 'mnemonic = "OBFUSCATED_MNEMONIC"'),
        (r'xch1[a-z0-9]+\b', 'xch1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'),  # Chia addresses
    ]

    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)

    return content

def obfuscate_file(filepath):
    """Obfuscate sensitive data in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        original_content = content

        # Apply all obfuscation functions
        content = obfuscate_ip_addresses(content)
        content = obfuscate_api_keys(content)
        content = obfuscate_personal_data(content)
        content = obfuscate_wallet_data(content)

        # Only write if content changed
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Obfuscated: {filepath}")
            return True
        else:
            return False

    except Exception as e:
        print(f"❌ Error obfuscating {filepath}: {e}")
        return False

def find_files_to_obfuscate():
    """Find files that might contain sensitive information"""
    sensitive_extensions = ['.py', '.js', '.json', '.txt', '.md', '.html', '.yaml', '.yml']

    files_to_check = []
    root_dir = Path(__file__).parent

    for ext in sensitive_extensions:
        files_to_check.extend(root_dir.rglob(f'*{ext}'))

    return files_to_check

def main():
    """Main obfuscation function"""
    print("🔒 SquashPlot Sensitive Data Obfuscation")
    print("=" * 50)

    # Find files to check
    files_to_check = find_files_to_obfuscate()
    print(f"📁 Found {len(files_to_check)} files to check")

    # Files to skip (already known to be safe or don't need obfuscation)
    skip_files = {
        '.gitignore',
        'LICENSE',
        'README.md',
        'obfuscate_sensitive_data.py',
        'requirements.txt',
        'setup.py',
        'test_squashplot.py'
    }

    obfuscated_count = 0
    checked_count = 0

    for filepath in files_to_check:
        filename = filepath.name

        # Skip certain files
        if filename in skip_files:
            continue

        checked_count += 1

        if obfuscate_file(filepath):
            obfuscated_count += 1

    print("\n📊 Obfuscation Complete:")
    print(f"   • Files checked: {checked_count}")
    print(f"   • Files obfuscated: {obfuscated_count}")
    print(f"   • Files unchanged: {checked_count - obfuscated_count}")

    if obfuscated_count > 0:
        print("\n⚠️  WARNING:")
        print("   Sensitive data has been obfuscated in the above files.")
        print("   Original values have been replaced with placeholder text.")
        print("   Make sure to restore real values in your local development environment.")
    else:
        print("\n✅ No sensitive data found!")
        print("   All files appear to be safe for public repository.")

if __name__ == "__main__":
    main()
