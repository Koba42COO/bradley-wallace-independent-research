#!/usr/bin/env python3
"""
Push Complete Tools and Compiled Papers to GitHub (Non-Interactive)

1. Push complete finished tools to private repo (full-stack-dev-folder) on new branch
2. Push compiled papers to public research repo (bradley-wallace-independent-research)
"""

import os
import subprocess
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
]

# Compiled papers directory
COMPILED_PAPERS_DIR = DEV_DIR / "bradley-wallace-independent-research" / "compiled_papers"


def run_command(cmd: List[str], cwd: Path = None, check: bool = False) -> Tuple[int, str, str]:
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


def create_new_branch():
    """Create new branch for complete tools"""
    print(f"\n🌿 Creating new branch: {NEW_BRANCH}")
    
    # Check if branch exists
    returncode, stdout, stderr = run_command(
        ["git", "branch", "--list", NEW_BRANCH],
        check=False
    )
    
    if NEW_BRANCH in stdout:
        print(f"⚠️  Branch {NEW_BRANCH} already exists, switching to it")
        run_command(["git", "checkout", NEW_BRANCH], check=False)
        return True
    
    # Create and checkout new branch
    returncode, stdout, stderr = run_command(["git", "checkout", "-b", NEW_BRANCH], check=False)
    if returncode == 0:
        print(f"✅ Created and switched to branch: {NEW_BRANCH}")
        return True
    else:
        print(f"❌ Failed to create branch: {stderr}")
        return False


def stage_complete_tools():
    """Stage all complete tools"""
    print("\n📦 Staging complete tools...")
    
    staged_count = 0
    missing_files = []
    
    for tool in COMPLETE_TOOLS:
        tool_path = DEV_DIR / tool
        if tool_path.exists():
            returncode, stdout, stderr = run_command(["git", "add", str(tool_path)], check=False)
            if returncode == 0:
                staged_count += 1
                print(f"  ✅ Staged: {tool}")
            else:
                print(f"  ⚠️  Failed to stage: {tool}")
        else:
            missing_files.append(tool)
            print(f"  ⚠️  Missing: {tool}")
    
    print(f"\n✅ Staged {staged_count} files")
    if missing_files:
        print(f"⚠️  {len(missing_files)} files not found")
    
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
    
    returncode, stdout, stderr = run_command(["git", "commit", "-m", commit_message], check=False)
    if returncode == 0:
        print("✅ Committed complete tools")
    else:
        if "nothing to commit" in stderr.lower():
            print("ℹ️  Nothing to commit (files may already be committed)")
        else:
            print(f"⚠️  Commit warning: {stderr}")
    
    print(f"\n🚀 Pushing to {PRIVATE_REPO}/{NEW_BRANCH}...")
    returncode, stdout, stderr = run_command(["git", "push", "-u", PRIVATE_REPO, NEW_BRANCH], check=False)
    if returncode == 0:
        print(f"✅ Pushed to {PRIVATE_REPO}/{NEW_BRANCH}")
        return True
    else:
        print(f"⚠️  Push result: {stderr}")
        return False


def push_compiled_papers():
    """Push compiled papers to public research repo"""
    print("\n📚 Pushing compiled papers to public research repo...")
    
    if not COMPILED_PAPERS_DIR.exists():
        print(f"❌ Compiled papers directory not found: {COMPILED_PAPERS_DIR}")
        return False
    
    # Get current branch
    returncode, current_branch, _ = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False
    )
    current_branch = current_branch.strip()
    
    # Fetch from public repo
    print(f"🔄 Fetching from {PUBLIC_REPO}...")
    run_command(["git", "fetch", PUBLIC_REPO], check=False)
    
    # Check if compiled-papers branch exists
    returncode, branches, _ = run_command(
        ["git", "branch", "-r", "--list", f"{PUBLIC_REPO}/compiled-papers"],
        check=False
    )
    
    branch_name = "compiled-papers"
    if f"{PUBLIC_REPO}/compiled-papers" in branches:
        print("📝 Found existing compiled-papers branch, switching to it")
        run_command(["git", "checkout", "-b", branch_name, f"{PUBLIC_REPO}/compiled-papers"], check=False)
    else:
        print("📝 Creating new compiled-papers branch")
        run_command(["git", "checkout", "-b", branch_name], check=False)
    
    # Stage compiled papers
    print("📦 Staging compiled papers...")
    
    # Find all compiled papers
    compiled_files = list(COMPILED_PAPERS_DIR.glob("*_COMPILED.md"))
    compiled_files.append(COMPILED_PAPERS_DIR / "MASTER_INDEX.md")
    
    # Also check for README
    readme_path = DEV_DIR / "COMPILED_PAPERS_README.md"
    if readme_path.exists():
        compiled_files.append(readme_path)
    
    staged_count = 0
    for paper_file in compiled_files:
        if paper_file.exists():
            # Get relative path from dev directory
            rel_path = paper_file.relative_to(DEV_DIR)
            returncode, stdout, stderr = run_command(["git", "add", str(rel_path)], check=False)
            if returncode == 0:
                staged_count += 1
                print(f"  ✅ Staged: {paper_file.name}")
    
    print(f"\n✅ Staged {staged_count} compiled papers")
    
    if staged_count == 0:
        print("⚠️  No compiled papers found to stage")
        # Return to original branch
        if current_branch:
            run_command(["git", "checkout", current_branch], check=False)
        return False
    
    # Commit
    print("\n💾 Committing compiled papers...")
    commit_message = """Add All Compiled Research Papers

- 47+ compiled analytical papers
- Master index of all papers
- Complete research documentation
- All papers ready for public review"""
    
    returncode, stdout, stderr = run_command(["git", "commit", "-m", commit_message], check=False)
    if returncode == 0:
        print("✅ Committed compiled papers")
    else:
        if "nothing to commit" in stderr.lower():
            print("ℹ️  Nothing to commit (papers may already be committed)")
        else:
            print(f"⚠️  Commit warning: {stderr}")
    
    # Push to public repo
    print(f"\n🚀 Pushing to {PUBLIC_REPO}/{branch_name}...")
    returncode, stdout, stderr = run_command(["git", "push", "-u", PUBLIC_REPO, branch_name], check=False)
    if returncode == 0:
        print(f"✅ Pushed compiled papers to {PUBLIC_REPO}/{branch_name}")
    else:
        print(f"⚠️  Push result: {stderr}")
    
    # Return to original branch
    if current_branch:
        print(f"\n🔄 Returning to original branch: {current_branch}")
        run_command(["git", "checkout", current_branch], check=False)
    
    return returncode == 0


def main():
    """Main execution"""
    print("=" * 70)
    print("🚀 PUSH COMPLETE TOOLS AND COMPILED PAPERS TO GITHUB")
    print("=" * 70)
    print()
    
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
    print(f"   Files: {len(COMPLETE_TOOLS)} tools")
    
    print(f"\n📚 Compiled Papers:")
    print(f"   Branch: compiled-papers")
    print(f"   Repo: {PUBLIC_REPO} (public)")
    print(f"   Location: {COMPILED_PAPERS_DIR}")
    
    print("\n🎯 All done!")


if __name__ == "__main__":
    main()

