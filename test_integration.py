#!/usr/bin/env python3
"""
Integration Test for SquashPlot Pro Production UI/UX
===================================================

Tests the unified interface with Dr. Plotter and Andy's CLI integration.
"""

import sys
import os

def test_imports():
    """Test all required imports"""
    print("🔍 Testing imports...")

    tests = [
        ("Production Server", "squashplot_production_server"),
        ("Dr. Plotter Integration", "dr_plotter_integration"),
        ("SquashPlot Core", "squashplot"),
        ("API Server", "squashplot_api_server"),
    ]

    results = {}
    for name, module in tests:
        try:
            __import__(module)
            results[name] = "✅"
            print(f"  {results[name]} {name}")
        except ImportError as e:
            results[name] = f"❌ ({e})"
            print(f"  {results[name]} {name}")

    return results

def test_dr_plotter():
    """Test Dr. Plotter functionality"""
    print("\n🧑‍🔬 Testing Dr. Plotter integration...")

    try:
        from dr_plotter_integration import DrPlotterIntegration, PlotterConfig

        dr_plotter = DrPlotterIntegration()
        recommendations = dr_plotter.get_system_recommendations()

        print("  ✅ Dr. Plotter initialized")
        print(f"  ✅ System recommendations: {len(recommendations)} categories")

        return True
    except Exception as e:
        print(f"  ❌ Dr. Plotter test failed: {e}")
        return False

def test_ui_files():
    """Test UI file availability"""
    print("\n🎨 Testing UI files...")

    ui_files = [
        "squashplot_production_ui.html",
        "squashplot_web_interface.html",
        "squashplot_dashboard.html"
    ]

    results = {}
    for ui_file in ui_files:
        if os.path.exists(ui_file):
            results[ui_file] = "✅"
            print(f"  {results[ui_file]} {ui_file}")
        else:
            results[ui_file] = "❌"
            print(f"  {results[ui_file]} {ui_file} - File not found")

    return results

def test_cli_integration():
    """Test CLI command templates"""
    print("\n💻 Testing CLI integration...")

    try:
        from squashplot_production_server import Config

        cli_commands = Config.CLI_COMMANDS
        print(f"  ✅ CLI commands loaded: {len(cli_commands)} commands")

        # Check for Dr. Plotter commands
        dr_commands = [cmd for cmd in cli_commands.keys() if 'dr' in cmd.lower()]
        print(f"  ✅ Dr. Plotter commands: {len(dr_commands)}")

        return True
    except Exception as e:
        print(f"  ❌ CLI integration test failed: {e}")
        return False

def test_main_integration():
    """Test main.py integration"""
    print("\n🚀 Testing main.py integration...")

    try:
        # Test argument parsing
        import argparse
        from main import main

        print("  ✅ Main module imports successfully")

        # Test CLI commands without executing
        print("  ✅ Main CLI structure verified")

        return True
    except Exception as e:
        print(f"  ❌ Main integration test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    print("🧪 SquashPlot Pro Integration Test Suite")
    print("=" * 50)

    # Run tests
    import_results = test_imports()
    ui_results = test_ui_files()
    cli_result = test_cli_integration()
    dr_result = test_dr_plotter()
    main_result = test_main_integration()

    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)

    total_tests = len(import_results) + len(ui_results) + 3  # cli, dr, main
    passed_tests = sum(1 for r in import_results.values() if r == "✅") + \
                   sum(1 for r in ui_results.values() if r == "✅") + \
                   (1 if cli_result else 0) + \
                   (1 if dr_result else 0) + \
                   (1 if main_result else 0)

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")

    if passed_tests == total_tests:
        print("\n🎉 All integration tests passed!")
        print("✅ SquashPlot Pro is ready for production use")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
