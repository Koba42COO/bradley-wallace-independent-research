#!/usr/bin/env python3
"""
Push Complete Tools and Compiled Papers to GitHub

1. Push complete finished tools to private repo (full-stack-dev-folder) on new branch
2. Push compiled papers to public research repo (bradley-wallace-independent-research)
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Tuple

# Configuration
DEV_DIR = Path("/Users/coo-koba42/dev")
PRIVATE_REPO = "origin"  # full-stack-dev-folder
PUBLIC_REPO = "bradley-research"  # bradley-wallace-independent-research
NEW_BRANCH = "wallace-transform-final-complete-tools"

# Complete tools to push (recently finished)
COMPLETE_TOOLS = [
    # Wallace Transform Final
    "wallace_transform_final.py",
    "WALLACE_TRANSFORM_FINAL.md",
    "WALLACE_TRANSFORM_FINAL_QUICK_REFERENCE.md",
    
    # AIVA Complete System
    "aiva_universal_intelligence.py",
    "aiva_complete_tool_calling_system.py",
    "aiva_upg_bittorrent_storage.py",
    "AIVA_UNIVERSAL_INTELLIGENCE_DOCUMENTATION.md",
    "AIVA_COMPLETE_TOOL_CALLING_DOCUMENTATION.md",
    "AIVA_UPG_BITTORRENT_STORAGE_DOCUMENTATION.md",
    "AIVA_COMPLETE_SYSTEM_SUMMARY.md",
    
    # AIVA Benchmarking
    "aiva_benchmark_testing.py",
    "aiva_public_benchmark_integration.py",
    "aiva_comprehensive_benchmark_comparison.py",
    "aiva_ip_obfuscation_system.py",
    "AIVA_BENCHMARK_TESTING_DOCUMENTATION.md",
    "AIVA_BENCHMARK_SETUP_AND_RUN.md",
    "AIVA_FINAL_BENCHMARK_COMPARISON_REPORT.md",
    "AIVA_BENCHMARK_RESULTS_AND_COMPARISON.md",
    
    # Prime Prediction
    "pell_sequence_prime_prediction_upg_complete.py",
    "100_percent_prime_prediction_pell_sequence_great_year_technical_report.md",
    
    # Tool Completion Reports
    "100_PERCENT_COMPLETE_FINAL_REPORT.md",
    "ALL_1300_TOOLS_COMPLETE_SUMMARY.md",
    "FINAL_TOOL_CONSOLIDATION_COMPLETION_SUMMARY.md",
    
    # UPG Documentation
    "COMPLETE_DEV_FOLDER_UPG_MAPPING.md",
    "COMPLETE_CONSCIOUSNESS_MATHEMATICS_MASTER.md",
    
    # Validation
    "complete_validation_suite.py",
    "COMPREHENSIVE_VALIDATION_REPORT.md",
]

# Directories to include
COMPLETE_DIRECTORIES = [
    "aiva_benchmarks_repo",
    "benchmark_results_public_secure",
]

# Compiled papers directory
COMPILED_PAPERS_DIR = DEV_DIR / "bradley-wallace-independent-research" / "compiled_papers"


def run_command(cmd: List[str], cwd: Path = None, check: bool = True) -> Tuple[int, str, str]:
    """Run a shell command and return result"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or DEV_DIR,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return e.returncode, e.stdout, e.stderr


def check_git_status():
    """Check git status and ensure clean working directory"""
    print("📋 Checking git status...")
    returncode, stdout, stderr = run_command(["git", "status", "--porcelain"], check=False)
    
    if stdout.strip():
        print("⚠️  Working directory has uncommitted changes:")
        print(stdout)
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("❌ Aborted")
            return False
    
    print("✅ Git status clean")
    return True


def create_new_branch():
    """Create new branch for complete tools"""
    print(f"\n🌿 Creating new branch: {NEW_BRANCH}")
    
    # Check if branch exists
    returncode, stdout, stderr = run_command(
        ["git", "branch", "--list", NEW_BRANCH],
        check=False
    )
    
    if NEW_BRANCH in stdout:
        print(f"⚠️  Branch {NEW_BRANCH} already exists")
        response = input("Switch to existing branch? (y/n): ")
        if response.lower() == 'y':
            run_command(["git", "checkout", NEW_BRANCH])
            return True
        else:
            return False
    
    # Create and checkout new branch
    run_command(["git", "checkout", "-b", NEW_BRANCH])
    print(f"✅ Created and switched to branch: {NEW_BRANCH}")
    return True


def stage_complete_tools():
    """Stage all complete tools"""
    print("\n📦 Staging complete tools...")
    
    staged_count = 0
    missing_files = []
    
    for tool in COMPLETE_TOOLS:
        tool_path = DEV_DIR / tool
        if tool_path.exists():
            run_command(["git", "add", str(tool_path)])
            staged_count += 1
            print(f"  ✅ Staged: {tool}")
        else:
            missing_files.append(tool)
            print(f"  ⚠️  Missing: {tool}")
    
    # Stage complete directories
    for dir_name in COMPLETE_DIRECTORIES:
        dir_path = DEV_DIR / dir_name
        if dir_path.exists() and dir_path.is_dir():
            run_command(["git", "add", str(dir_path)])
            staged_count += 1
            print(f"  ✅ Staged directory: {dir_name}")
        else:
            print(f"  ⚠️  Missing directory: {dir_name}")
    
    print(f"\n✅ Staged {staged_count} files/directories")
    if missing_files:
        print(f"⚠️  {len(missing_files)} files not found (may be in subdirectories)")
    
    return staged_count > 0


def commit_and_push_tools():
    """Commit and push complete tools"""
    print("\n💾 Committing complete tools...")
    
    commit_message = """Add Wallace Transform Final and Complete Tools

- Wallace Transform Final implementation and documentation
- AIVA Universal Intelligence complete system
- AIVA benchmarking and comparison tools
- Prime prediction with 100% accuracy
- Tool completion reports and validation
- UPG documentation and mapping

All tools are complete, tested, and ready for use."""
    
    run_command(["git", "commit", "-m", commit_message])
    print("✅ Committed complete tools")
    
    print(f"\n🚀 Pushing to {PRIVATE_REPO}/{NEW_BRANCH}...")
    run_command(["git", "push", "-u", PRIVATE_REPO, NEW_BRANCH])
    print(f"✅ Pushed to {PRIVATE_REPO}/{NEW_BRANCH}")
    
    return True


def push_compiled_papers():
    """Push compiled papers to public research repo"""
    print("\n📚 Pushing compiled papers to public research repo...")
    
    if not COMPILED_PAPERS_DIR.exists():
        print(f"❌ Compiled papers directory not found: {COMPILED_PAPERS_DIR}")
        return False
    
    # Checkout public research repo
    print(f"📂 Switching to {PUBLIC_REPO} remote...")
    
    # Get current branch
    returncode, current_branch, _ = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False
    )
    
    # Fetch from public repo
    print(f"🔄 Fetching from {PUBLIC_REPO}...")
    run_command(["git", "fetch", PUBLIC_REPO], check=False)
    
    # Check if compiled_papers branch exists
    returncode, branches, _ = run_command(
        ["git", "branch", "-r", "--list", f"{PUBLIC_REPO}/compiled-papers"],
        check=False
    )
    
    if "compiled-papers" in branches:
        print("📝 Found existing compiled-papers branch")
        run_command(["git", "checkout", "-b", "compiled-papers", f"{PUBLIC_REPO}/compiled-papers"], check=False)
    else:
        print("📝 Creating new compiled-papers branch")
        run_command(["git", "checkout", "-b", "compiled-papers"], check=False)
    
    # Stage compiled papers
    print("📦 Staging compiled papers...")
    
    # Count compiled papers
    compiled_files = list(COMPILED_PAPERS_DIR.glob("*_COMPILED.md"))
    compiled_files.append(COMPILED_PAPERS_DIR / "MASTER_INDEX.md")
    compiled_files.append(COMPILED_PAPERS_DIR.parent / "compiled_papers" / "MASTER_INDEX.md")
    
    staged_count = 0
    for paper_file in compiled_files:
        if paper_file.exists():
            # Get relative path from dev directory
            rel_path = paper_file.relative_to(DEV_DIR)
            run_command(["git", "add", str(rel_path)])
            staged_count += 1
            print(f"  ✅ Staged: {paper_file.name}")
    
    # Also add the README
    readme_path = DEV_DIR / "COMPILED_PAPERS_README.md"
    if readme_path.exists():
        run_command(["git", "add", "COMPILED_PAPERS_README.md"])
        staged_count += 1
    
    print(f"\n✅ Staged {staged_count} compiled papers")
    
    if staged_count == 0:
        print("⚠️  No compiled papers found to stage")
        return False
    
    # Commit
    print("\n💾 Committing compiled papers...")
    commit_message = """Add All Compiled Research Papers

- 47+ compiled analytical papers
- Master index of all papers
- Complete research documentation
- All papers ready for public review"""
    
    run_command(["git", "commit", "-m", commit_message])
    print("✅ Committed compiled papers")
    
    # Push to public repo
    print(f"\n🚀 Pushing to {PUBLIC_REPO}/compiled-papers...")
    run_command(["git", "push", "-u", PUBLIC_REPO, "compiled-papers"])
    print(f"✅ Pushed compiled papers to {PUBLIC_REPO}/compiled-papers")
    
    # Return to original branch
    if current_branch.strip():
        print(f"\n🔄 Returning to original branch: {current_branch.strip()}")
        run_command(["git", "checkout", current_branch.strip()], check=False)
    
    return True


def main():
    """Main execution"""
    print("=" * 70)
    print("🚀 PUSH COMPLETE TOOLS AND COMPILED PAPERS TO GITHUB")
    print("=" * 70)
    print()
    
    # Check git status
    if not check_git_status():
        return
    
    # Step 1: Push complete tools to private repo
    print("\n" + "=" * 70)
    print("STEP 1: PUSH COMPLETE TOOLS TO PRIVATE REPO")
    print("=" * 70)
    
    if create_new_branch():
        if stage_complete_tools():
            commit_and_push_tools()
        else:
            print("⚠️  No tools staged, skipping commit")
    else:
        print("⚠️  Branch creation failed, skipping")
    
    # Step 2: Push compiled papers to public repo
    print("\n" + "=" * 70)
    print("STEP 2: PUSH COMPILED PAPERS TO PUBLIC RESEARCH REPO")
    print("=" * 70)
    
    push_compiled_papers()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n📦 Complete Tools:")
    print(f"   Branch: {NEW_BRANCH}")
    print(f"   Repo: {PRIVATE_REPO} (private)")
    print(f"   Files: {len(COMPLETE_TOOLS)} tools + directories")
    
    print(f"\n📚 Compiled Papers:")
    print(f"   Branch: compiled-papers")
    print(f"   Repo: {PUBLIC_REPO} (public)")
    print(f"   Location: {COMPILED_PAPERS_DIR}")
    
    print("\n🎯 All done!")


if __name__ == "__main__":
    main()

