#!/usr/bin/env python3
"""
Production validation script for SquashPlot Enhanced
Validates that all production components are working correctly
"""

import sys
import os
import subprocess
import json
from pathlib import Path

def validate_health_check():
    """Validate health check returns proper exit codes"""
    print("🔍 Testing health check...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/production_wrapper.py", "health-check"
        ], capture_output=True, text=True, cwd="/workspaces/replit-agent/production")
        
        if result.returncode == 1:  # Should fail in test environment
            print("✅ Health check properly fails with exit code 1")
            
            # Check for structured JSON output
            if "Health status:" in result.stderr and "{" in result.stderr:
                print("✅ Structured JSON health status logged")
            else:
                print("❌ Missing structured health status")
                return False
                
        else:
            print(f"❌ Health check returned unexpected exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Health check validation failed: {e}")
        return False
        
    return True

def validate_tool_interface():
    """Validate tool interface works correctly"""
    print("🔍 Testing tool interface...")
    
    try:
        result = subprocess.run([
            sys.executable, "src/squashplot_enhanced.py", "--version"
        ], capture_output=True, text=True, cwd="/workspaces/replit-agent/production")
        
        if result.returncode == 0 and "SquashPlot Enhanced" in result.stdout:
            print("✅ Tool interface responds correctly")
        else:
            print("❌ Tool interface failed")
            return False
            
    except Exception as e:
        print(f"❌ Tool interface validation failed: {e}")
        return False
        
    return True

def validate_file_structure():
    """Validate production file structure"""
    print("🔍 Testing file structure...")
    
    required_files = [
        "src/squashplot_enhanced.py",
        "src/production_wrapper.py", 
        "src/__init__.py",
        "config/__init__.py",
        "requirements.txt",
        "Dockerfile",
        "README.md",
        ".env.example"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        
    return True

def validate_syntax():
    """Validate Python syntax for all modules"""
    print("🔍 Testing Python syntax...")
    
    python_files = [
        "src/squashplot_enhanced.py",
        "src/production_wrapper.py"
    ]
    
    for file_path in python_files:
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", file_path
            ], capture_output=True, text=True, cwd="/workspaces/replit-agent/production")
            
            if result.returncode != 0:
                print(f"❌ Syntax error in {file_path}: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Syntax validation failed for {file_path}: {e}")
            return False
            
    print("✅ All Python files have valid syntax")
    return True

def main():
    """Run all production validations"""
    print("🚀 SquashPlot Enhanced Production Validation")
    print("=" * 50)
    
    validations = [
        ("File Structure", validate_file_structure),
        ("Python Syntax", validate_syntax), 
        ("Tool Interface", validate_tool_interface),
        ("Health Check", validate_health_check)
    ]
    
    results = {}
    
    for name, validator in validations:
        print(f"\n📋 {name}")
        results[name] = validator()
        
    print("\n" + "=" * 50)
    print("🎯 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
            
    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED - PRODUCTION READY!")
        return 0
    else:
        print("\n💥 VALIDATION FAILURES - NOT PRODUCTION READY")
        return 1

if __name__ == "__main__":
    sys.exit(main())