#!/usr/bin/env python3
"""
SquashPlot Setup Script for Replit
==================================

This script helps set up SquashPlot on Replit by:
- Installing dependencies
- Checking system requirements
- Running basic tests
- Setting up the environment
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main setup function"""
    print("🚀 SquashPlot Setup for Replit")
    print("=" * 40)

    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python Version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 8):
        print("❌ Python 3.8 or higher required")
        return False

    print("✅ Python version OK")

    # Check if we're on Replit
    on_replit = os.environ.get('REPL_ID') is not None
    if on_replit:
        print("✅ Running on Replit")
    else:
        print("ℹ️  Not running on Replit (that's OK)")

    # Install dependencies
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False

    # Check for required files
    print("\n📁 Checking files...")
    required_files = [
        "main.py",
        "squashplot.py",
        "squashplot_web_interface.html",
        "requirements.txt"
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False

    print("✅ All required files present")

    # Run basic tests
    print("\n🧪 Running basic tests...")

    try:
        # Test import
        from squashplot import SquashPlotCompressor
        compressor = SquashPlotCompressor(pro_enabled=False)
        print("✅ Basic compression test passed")

        # Test Pro compression
        compressor_pro = SquashPlotCompressor(pro_enabled=True)
        print("✅ Pro compression test passed")

    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

    # Check web interface
    web_file = Path("squashplot_web_interface.html")
    if web_file.exists():
        size = web_file.stat().st_size
        print(f"✅ Web interface ready ({size:,} bytes)")
    else:
        print("❌ Web interface not found")
        return False

    # Success message
    print("\n" + "=" * 40)
    print("🎉 SquashPlot Setup Complete!")
    print("=" * 40)
    print()
    print("🚀 To start SquashPlot:")
    print("   Web Interface: python main.py --web")
    print("   CLI: python main.py --cli")
    print("   Demo: python main.py --demo")
    print()
    print("🌐 Web interface will be available at:")
    print("   https://your-replit-name.replit.dev")
    print()
    print("📚 Documentation:")
    print("   REPLIT_README.md - Setup and usage guide")
    print("   README.md - General SquashPlot documentation")
    print()
    print("🧪 Test commands:")
    print("   python squashplot.py --benchmark")
    print("   python compression_validator.py --size 50")
    print()
    print("💡 Pro tip: Request whitelist access for enhanced features:")
    print("   python squashplot.py --whitelist-request your@email.com")

    return True

def cleanup():
    """Clean up any temporary files"""
    print("\n🧹 Cleaning up...")

    # Remove any __pycache__ directories
    for pycache in Path(".").rglob("__pycache__"):
        if pycache.is_dir():
            import shutil
            shutil.rmtree(pycache)
            print(f"   Removed {pycache}")

    print("✅ Cleanup complete")

if __name__ == "__main__":
    try:
        success = main()
        cleanup()

        if success:
            print("\n🎯 Ready to compress some Chia plots!")
            sys.exit(0)
        else:
            print("\n❌ Setup failed - please check the errors above")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        cleanup()
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        cleanup()
        sys.exit(1)
