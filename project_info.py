#!/usr/bin/env python3
"""
SquashPlot Project Information
==============================

Displays information about all files in the SquashPlot project.
"""

import os
from pathlib import Path
from datetime import datetime

def get_file_info(filepath):
    """Get file information"""
    path = Path(filepath)
    if not path.exists():
        return None

    stat = path.stat()
    return {
        'name': path.name,
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
        'type': 'directory' if path.is_dir() else 'file'
    }

def main():
    """Display project information"""
    print("🌟 SquashPlot - Complete Replit Project")
    print("=" * 50)

    # Core files
    core_files = [
        "main.py",
        "squashplot.py",
        "whitelist_signup.py",
        "compression_validator.py",
        "squashplot_web_interface.html",
        "requirements.txt",
        ".replit",
        "replit.nix",
        "setup.py",
        "test_squashplot.py",
        "project_info.py"
    ]

    # Documentation files
    docs_files = [
        "README.md",
        "REPLIT_README.md",
        "SQUASHPLOT_TECHNICAL_WHITEPAPER.md",
        "twitter_bio.txt"
    ]

    print("\n📁 CORE FILES:")
    print("-" * 30)

    for file in core_files:
        info = get_file_info(file)
        if info:
            print("<25")
        else:
            print("<25")
    print("\n📚 DOCUMENTATION:")
    print("-" * 30)

    for file in docs_files:
        info = get_file_info(file)
        if info:
            print("<25")
        else:
            print("<25")
    print("\n🚀 QUICK START:")
    print("-" * 30)
    print("1. Run setup: python setup.py")
    print("2. Start web: python main.py --web")
    print("3. Open browser to access interface")
    print("4. Test with: python test_squashplot.py")

    print("\n🎯 FEATURES:")
    print("-" * 30)
    print("✅ Basic Version (FREE)")
    print("   • 42% compression ratio")
    print("   • 2x processing speed")
    print("   • Proven algorithms")
    print("   • Web & CLI interfaces")
    print()
    print("🚀 Pro Version (Whitelist)")
    print("   • Up to 70% compression")
    print("   • Up to 2x faster processing")
    print("   • Enhanced algorithms")
    print("   • Priority support")

    print("\n🛠️ TECHNOLOGIES:")
    print("-" * 30)
    print("• Python 3.8+")
    print("• Flask web framework")
    print("• NumPy for data processing")
    print("• Zlib/Bz2/Lzma compression")
    print("• HTML/CSS/JavaScript frontend")

    print("\n📦 DEPENDENCIES:")
    print("-" * 30)
    print("See requirements.txt for full list")
    print("• numpy - Data processing")
    print("• flask - Web framework")
    print("• requests - HTTP client")
    print("• psutil - System monitoring")

    print("\n🔧 DEVELOPMENT:")
    print("-" * 30)
    print("• pytest - Testing framework")
    print("• black - Code formatting")
    print("• flake8 - Linting")
    print("• mypy - Type checking")

    # Project statistics
    total_files = len(core_files) + len(docs_files)
    total_size = 0

    for files in [core_files, docs_files]:
        for file in files:
            info = get_file_info(file)
            if info and info['type'] == 'file':
                total_size += info['size']

    print("\n📊 PROJECT STATS:")
    print("-" * 30)
    print(f"• Total files: {total_files}")
    print(f"• Total size: {total_size:,} bytes")
    print(".1f")
    print("• Platform: Replit ready")
    print("• Python version: 3.8+ required")

    print("\n🎉 READY TO DEPLOY!")
    print("-" * 30)
    print("This SquashPlot project is fully configured for Replit.")
    print("Just fork this repository and start developing!")

if __name__ == "__main__":
    main()
