#!/usr/bin/env python3
"""
SquashPlot Deployment Script
============================

Quick deployment options for SquashPlot package.
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Deployment menu"""
    print("🚀 SquashPlot Deployment Options")
    print("=" * 40)

    options = {
        "1": ("Replit (Recommended)", deploy_replit),
        "2": ("Local Development", deploy_local),
        "3": ("Docker", deploy_docker),
        "4": ("Test Everything", run_tests),
        "5": ("Package Info", show_info),
        "6": ("Exit", exit_app)
    }

    while True:
        print("\n📋 Deployment Options:")
        for key, (name, _) in options.items():
            print(f"   {key}. {name}")

        choice = input("\nChoose option (1-6): ").strip()

        if choice in options:
            func = options[choice][1]
            func()
        else:
            print("❌ Invalid option. Please choose 1-6.")

def deploy_replit():
    """Replit deployment instructions"""
    print("\n🌐 Replit Deployment")
    print("-" * 30)
    print("1. Go to https://replit.com")
    print("2. Click 'Create' -> 'Fork from GitHub'")
    print("3. Enter: your-github-username/squashplot")
    print("4. Or upload this entire folder to a new Replit")
    print("5. Run: python setup.py")
    print("6. Run: python main.py --web")
    print("7. Click 'Open in new tab'")
    print()
    print("✅ Your SquashPlot will be live at:")
    print("   https://your-replit-name.replit.dev")

def deploy_local():
    """Local deployment"""
    print("\n💻 Local Deployment")
    print("-" * 30)

    try:
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            print("❌ Python 3.8+ required")
            return

        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Install dependencies
        print("\n📦 Installing dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])

        # Run setup
        print("\n🚀 Running setup...")
        subprocess.check_call([sys.executable, "setup.py"])

        # Start web interface
        print("\n🌐 Starting web interface...")
        print("   Access at: http://localhost:8080")
        print("   Press Ctrl+C to stop")

        subprocess.run([
            sys.executable, "main.py", "--web", "--port", "8080"
        ])

    except KeyboardInterrupt:
        print("\n⚠️  Deployment stopped by user")
    except Exception as e:
        print(f"❌ Deployment failed: {e}")

def deploy_docker():
    """Docker deployment instructions"""
    print("\n🐳 Docker Deployment")
    print("-" * 30)
    print("Docker deployment coming soon!")
    print("For now, use Replit or local deployment.")

def run_tests():
    """Run all tests"""
    print("\n🧪 Running Tests")
    print("-" * 30)

    try:
        subprocess.check_call([sys.executable, "test_squashplot.py"])
        print("✅ All tests passed!")
    except subprocess.CalledProcessError:
        print("❌ Some tests failed")
    except FileNotFoundError:
        print("❌ Test file not found")

def show_info():
    """Show package information"""
    print("\n📊 Package Information")
    print("-" * 30)

    try:
        subprocess.check_call([sys.executable, "project_info.py"])
    except subprocess.CalledProcessError:
        print("❌ Could not load package info")
    except FileNotFoundError:
        print("❌ Project info file not found")

def exit_app():
    """Exit the application"""
    print("\n👋 Thanks for using SquashPlot!")
    sys.exit(0)

if __name__ == "__main__":
    main()
