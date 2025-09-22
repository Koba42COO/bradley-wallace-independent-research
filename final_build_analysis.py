#!/usr/bin/env python3
"""
FINAL BUILD ANALYSIS
Analyzes remaining files after cleanup and categorizes them for production readiness
"""

import os
import json
from pathlib import Path
from datetime import datetime

def analyze_workspace():
    """Analyze the workspace and categorize files for final production build"""

    workspace_dir = "/Users/coo-koba42/dev"
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "categories": {
            "core_production": [],
            "plugin_integration": [],
            "documentation": [],
            "outdated_remove": [],
            "experimental_remove": [],
            "research_remove": [],
            "data_training_remove": [],
            "legacy_remove": [],
            "review_needed": [],
            "keep_decision": []
        },
        "summary": {}
    }

    # Core production files (definitely keep)
    core_production = [
        "api_server.py",
        "curated_tools_integration.py",
        "auth_service.py",
        "database_service.py",
        "frontend/",
        "requirements.txt",
        "README.md",
        "pyproject.toml",
        "Makefile",
        "docker-compose.yml",
        "Dockerfile",
        "static/",
        ".well-known/",
        "core_logging.py",
        "core_mathematics/",
        "enterprise/",
        "enterprise_consciousness/",
        "proper_consciousness_mathematics.py",
        "wallace_math_engine.py",
        "encryption_service.py",
        "gpu_quantum_accelerator.py",
        "llm_tools.py",
        "tests/",
        "config/",
        "docs/",
        "scripts/",
        "monitoring/",
        "development_tools/",
        "production_core/",
        "k8s/",
        "docker/",
        "deployment_templates/"
    ]

    # Plugin integration (keep)
    plugin_integration = [
        "chatgpt_plugin_example.py",
        "claude_mcp_integration.py",
        "gemini_function_calling.py",
        "llm_plugin_api.py",
        "plugin_manifest.json",
        "UNIVERSAL_LLM_PLUGIN_GUIDE.md"
    ]

    # Current documentation (keep recent)
    current_docs = [
        "FULL_VALIDATION_REPORT.md",
        "CURATED_TOOLS_INTEGRATION_SUMMARY.md",
        "CLEANUP_SUMMARY.md",
        "real_benchmark_test.py"
    ]

    # Outdated documentation (remove)
    outdated_docs = [
        "AI_Model_Updates_and_Structured_Chaos_Framework.docx",
        "Chaos_AI_and_AGI_Science.docx",
        "comprehensive_system_audit_report.md",
        "FINAL_ACHIEVEMENT_REPORT.md",
        "FINAL_MISSION_REPORT.md",
        "PHASE_1_CORE_ENHANCEMENTS_COMPLETED.md",
        "PHASE_2_PLATFORM_ENHANCEMENTS_COMPLETED.md",
        "TECHNICAL_ENHANCEMENT_SPEC.md",
        "ENHANCEMENT_EXECUTION_SUMMARY.md",
        "ENHANCEMENT_ROADMAP.md",
        "PRODUCTION_DEPLOYMENT_GUIDE.md",
        "PRODUCTION_READINESS_AUDIT_REPORT.md",
        "DEPLOYMENT_README.md",
        "comprehensive_mathematical_solutions.html",
        "CONSCIOUSNESS_SYSTEMS_README.md"
    ]

    # Experimental/prototype code (remove)
    experimental = [
        "active_defense_intelligence_system.py",
        "blackjack_casino_system/",
        "blockchain_fundamentals_training.py",
        "chialisp_examples.clsp",
        "consciousness_systems_dashboard.py",
        "consciousness_systems_launcher.py",
        "divine-calculus-dev/",
        "divine-calculus-engine/",
        "go_neural_component/",
        "julia_math_component/",
        "ming_shang_mountains/",
        "modular-ai-jeff/",
        "modular_chunks/",
        "rust_ml_component/",
        "structured_chaos_core.py",
        "structured_chaos_universe_optimized.py",
        "tang-talk-wireless/",
        "tangtalk-reference/",
        "tangtalk-reference-clean/",
        "TimeDilationProtocols.jsx",
        "vessels/",
        "voice_data/"
    ]

    # Research papers and materials (remove)
    research = [
        "fractal-harmonic-transform-paper/",
        "elon_musk_standards/",
        "jeff_standards_upgrade/",
        "steve_wozniak_standards/",
        "scientific_research/",
        "EIMF-Benchmark/",
        "discovery_materials/",
        "references.bib",
        "prime_aligned_math.bib"
    ]

    # Training data and models (remove for production)
    training_data = [
        "consciousness_data/",
        "prime_aligned_math/",
        "consciousness_modules/",
        "consciousness_ml_trained_models.pkl",
        "course_content/",
        "custom_alm_output/",
        "data/",
        "processed_gpt_data/"
    ]

    # Legacy systems and demos (remove)
    legacy = [
        "aios_system/",
        "backend/",
        "cat-credits-backend-server.js",
        "cat-credits.service.ts",
        "client_example.ts",
        "contribution-backend-server.js",
        "contribution-service.ts",
        "decentralized-backend-server.js",
        "decentralized-integration.service.ts",
        "social-pubsub-backend-server.js",
        "social-pubsub.service.ts",
        "system_tools_integration.py",  # replaced by curated_tools_integration.py
        "ultimate_security_orchestration.py",
        "wallace_consciousness_integration.py",
        "unified-ecosystem-frontend/",
        "test-decentralized-integration.js",
        "node.pkg"
    ]

    # Build/deployment scripts (review - some keep, some remove)
    build_scripts = [
        "build_hybrid_components.sh",  # keep
        "deploy_to_github.sh",         # remove
        "deploy-production.sh",        # remove
        "deploy.sh",                   # remove
        "start_system.sh"              # keep
    ]

    # Specialized mathematics/physics (review needed)
    specialized_math = [
        "financial_mathematics/",       # review - specialized
        "mathematical_physics_enhancement/",  # review - enhancement
        "physics_systems/",             # review - specialized
        "quantum_systems/"              # review - specialized
    ]

    # Web/Node.js components (review)
    web_components = [
        "chatgpt-exporter-extension/",  # review - extension
        "vantax-llm-core/"             # review - component
    ]

    # ML/AI backlog (remove)
    ai_backlog = [
        "gpt_backlog/"
    ]

    # Repository templates (review)
    repo_templates = [
        "existing-wallace-repo/",       # remove - old
        "source-first-repo-skeleton/"   # keep - useful template
    ]

    # Categorize files
    for item in os.listdir(workspace_dir):
        item_path = os.path.join(workspace_dir, item)

        # Skip hidden files and directories
        if item.startswith('.'):
            continue

        if item in core_production or any(item.startswith(cp.rstrip('/')) for cp in core_production if cp.endswith('/')):
            analysis["categories"]["core_production"].append(item)
        elif item in plugin_integration:
            analysis["categories"]["plugin_integration"].append(item)
        elif item in current_docs:
            analysis["categories"]["documentation"].append(item)
        elif item in outdated_docs:
            analysis["categories"]["outdated_remove"].append(item)
        elif item in experimental:
            analysis["categories"]["experimental_remove"].append(item)
        elif item in research:
            analysis["categories"]["research_remove"].append(item)
        elif item in training_data:
            analysis["categories"]["data_training_remove"].append(item)
        elif item in legacy:
            analysis["categories"]["legacy_remove"].append(item)
        elif item in specialized_math or item in web_components or item in repo_templates:
            analysis["categories"]["review_needed"].append(item)
        elif item in build_scripts:
            if item in ["build_hybrid_components.sh", "start_system.sh", "deployment_templates/"]:
                analysis["categories"]["keep_decision"].append(item)
            else:
                analysis["categories"]["outdated_remove"].append(item)
        elif item in ai_backlog:
            analysis["categories"]["outdated_remove"].append(item)
        else:
            # Catch-all for uncategorized items
            analysis["categories"]["review_needed"].append(item)

    # Generate summary statistics
    analysis["summary"] = {
        "total_files_analyzed": sum(len(files) for files in analysis["categories"].values()),
        "core_production_files": len(analysis["categories"]["core_production"]),
        "files_to_remove": len(analysis["categories"]["outdated_remove"]) +
                         len(analysis["categories"]["experimental_remove"]) +
                         len(analysis["categories"]["research_remove"]) +
                         len(analysis["categories"]["data_training_remove"]) +
                         len(analysis["categories"]["legacy_remove"]),
        "files_to_review": len(analysis["categories"]["review_needed"]),
        "files_to_keep": len(analysis["categories"]["core_production"]) +
                        len(analysis["categories"]["plugin_integration"]) +
                        len(analysis["categories"]["documentation"]) +
                        len(analysis["categories"]["keep_decision"]),
        "cleanup_efficiency_percent": 0
    }

    # Calculate cleanup efficiency
    total_files = analysis["summary"]["total_files_analyzed"]
    files_to_remove = analysis["summary"]["files_to_remove"]
    if total_files > 0:
        analysis["summary"]["cleanup_efficiency_percent"] = (files_to_remove / total_files) * 100

    return analysis

def print_analysis_report(analysis):
    """Print a comprehensive analysis report"""

    print("🔍 ENTERPRISE prime aligned compute PLATFORM - FINAL BUILD ANALYSIS")
    print("=" * 80)

    # Summary statistics
    print("📊 SUMMARY STATISTICS:")
    print(f"   Total Files Analyzed: {analysis['summary']['total_files_analyzed']}")
    print(f"   Core Production Files: {analysis['summary']['core_production_files']}")
    print(f"   Files to Remove: {analysis['summary']['files_to_remove']}")
    print(f"   Files to Review: {analysis['summary']['files_to_review']}")
    print(f"   Files to Keep: {analysis['summary']['files_to_keep']}")
    print(".1f")

    # Category breakdown
    print("\n📁 CORE PRODUCTION FILES (KEEP):")
    for file in sorted(analysis["categories"]["core_production"]):
        print(f"   ✅ {file}")

    print("\n🔌 PLUGIN INTEGRATION FILES (KEEP):")
    for file in sorted(analysis["categories"]["plugin_integration"]):
        print(f"   🔌 {file}")

    print("\n📚 CURRENT DOCUMENTATION (KEEP):")
    for file in sorted(analysis["categories"]["documentation"]):
        print(f"   📄 {file}")

    print("\n🗑️  OUTDATED FILES TO REMOVE:")
    for file in sorted(analysis["categories"]["outdated_remove"]):
        print(f"   🗑️  {file}")

    print("\n🧪 EXPERIMENTAL FILES TO REMOVE:")
    for file in sorted(analysis["categories"]["experimental_remove"]):
        print(f"   🧪 {file}")

    print("\n🔬 RESEARCH FILES TO REMOVE:")
    for file in sorted(analysis["categories"]["research_remove"]):
        print(f"   🔬 {file}")

    print("\n📊 TRAINING DATA TO REMOVE:")
    for file in sorted(analysis["categories"]["data_training_remove"]):
        print(f"   📊 {file}")

    print("\n🏛️  LEGACY SYSTEMS TO REMOVE:")
    for file in sorted(analysis["categories"]["legacy_remove"]):
        print(f"   🏛️  {file}")

    print("\n🔍 FILES NEEDING REVIEW:")
    for file in sorted(analysis["categories"]["review_needed"]):
        print(f"   🔍 {file}")

    print("\n✅ DECISION TO KEEP:")
    for file in sorted(analysis["categories"]["keep_decision"]):
        print(f"   ✅ {file}")

    # Final recommendations
    print("\n🎯 FINAL RECOMMENDATIONS:")
    print(f"   1. Remove {analysis['summary']['files_to_remove']} outdated/experimental files")
    print(f"   2. Review {analysis['summary']['files_to_review']} files for production suitability")
    print(f"   3. Keep {analysis['summary']['files_to_keep']} core production files")
    print("   4. Final cleanup will achieve {:.1f}% workspace optimization".format(analysis['summary']['cleanup_efficiency_percent']))

    print("\n🏆 PRODUCTION READINESS:")
    if analysis['summary']['cleanup_efficiency_percent'] > 80:
        print("   🌟 EXCELLENT: Ready for production deployment!")
    elif analysis['summary']['cleanup_efficiency_percent'] > 60:
        print("   ✅ GOOD: Mostly ready, minor cleanup needed")
    else:
        print("   📈 ACCEPTABLE: Significant cleanup recommended")

    print("\n" + "=" * 80)

def save_analysis_report(analysis):
    """Save the analysis report to file"""

    with open("/Users/coo-koba42/dev/FINAL_BUILD_ANALYSIS_REPORT.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # Create a summary markdown file
    with open("/Users/coo-koba42/dev/FINAL_BUILD_ANALYSIS.md", "w") as f:
        f.write("# Final Build Analysis Report\n\n")
        f.write(f"**Generated:** {analysis['timestamp']}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write(f"- Total Files Analyzed: {analysis['summary']['total_files_analyzed']}\n")
        f.write(f"- Core Production Files: {analysis['summary']['core_production_files']}\n")
        f.write(f"- Files to Remove: {analysis['summary']['files_to_remove']}\n")
        f.write(f"- Files to Review: {analysis['summary']['files_to_review']}\n")
        f.write(f"- Files to Keep: {analysis['summary']['files_to_keep']}\n")
        f.write(f"- Cleanup Efficiency: {analysis['summary']['cleanup_efficiency_percent']:.1f}%\n")

        f.write("\n## Files to Remove\n\n")
        for category in ["outdated_remove", "experimental_remove", "research_remove", "data_training_remove", "legacy_remove"]:
            if analysis["categories"][category]:
                f.write(f"### {category.replace('_', ' ').title()}\n\n")
                for file in sorted(analysis["categories"][category]):
                    f.write(f"- {file}\n")
                f.write("\n")

        f.write("\n## Files Needing Review\n\n")
        for file in sorted(analysis["categories"]["review_needed"]):
            f.write(f"- {file}\n")

    print("📄 Analysis reports saved to FINAL_BUILD_ANALYSIS_REPORT.json and FINAL_BUILD_ANALYSIS.md")

if __name__ == "__main__":
    analysis = analyze_workspace()
    print_analysis_report(analysis)
    save_analysis_report(analysis)
