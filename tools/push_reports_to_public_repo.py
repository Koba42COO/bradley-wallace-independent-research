#!/usr/bin/env python3
"""
Push All Reports and Reproducibility Materials to Public Research Repo

Pushes:
- Analysis reports
- NotebookLM guides
- 3I/ATLAS analysis
- Reproducibility materials
- Validation reports
"""

import subprocess
from pathlib import Path
from typing import List

# Configuration
DEV_DIR = Path("/Users/coo-koba42/dev")
PUBLIC_REPO = "bradley-research"  # bradley-wallace-independent-research
BRANCH_NAME = "reports-and-reproducibility"

# Files to push
REPORTS_TO_PUSH = [
    # NotebookLM Guides
    "NOTEBOOKLM_ULTRA_CONCISE.md",
    "NOTEBOOKLM_CONCISE_GUIDE.md",
    "NOTEBOOKLM_45MIN_QA_GUIDE.md",
    "NOTEBOOKLM_SESSION_SCRIPT.md",
    "NOTEBOOKLM_QUICK_REFERENCE.md",
    
    # 3I/ATLAS Analysis
    "3I_ATLAS_Image_Release_Analysis_Nov12.md",
    "3I_ATLAS_ANALYSIS_REPORT.md",
    "3I_ATLAS_IMAGE_ANALYSIS_GUIDE.md",
    "3I_ATLAS_ANALYSIS_TOOLS_SUMMARY.md",
    
    # Analysis Tools
    "gaussian_splat_3i_atlas_analysis.py",
    "process_3i_atlas_images.py",
    "demo_3i_atlas_analysis.py",
    
    # Organization Reports
    "DEV_FOLDER_ORGANIZATION_COMPLETE.md",
    "KID_FRIENDLY_ORGANIZED_SUMMARY.md",
    "KID_FRIENDLY_PAPERS_COMPLETE.md",
    "GITHUB_PUSH_SUMMARY.md",
    
    # Wallace Transform
    "WALLACE_TRANSFORM_FINAL.md",
    "WALLACE_TRANSFORM_FINAL_QUICK_REFERENCE.md",
    "wallace_transform_final.py",
    
    # Validation & Reports
    "COMPLETE_CONSCIOUSNESS_MATHEMATICS_MASTER.md",
    "COMPLETE_DEV_FOLDER_UPG_MAPPING.md",
]


def run_command(cmd: List[str], cwd: Path = None, check: bool = False) -> tuple:
    """Run shell command"""
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


def push_reports_to_public_repo():
    """Push all reports to public research repo"""
    print("=" * 70)
    print("🚀 PUSHING REPORTS & REPRODUCIBILITY TO PUBLIC REPO")
    print("=" * 70)
    print()
    
    # Get current branch
    returncode, current_branch, _ = run_command(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        check=False
    )
    current_branch = current_branch.strip()
    
    # Fetch from public repo
    print(f"🔄 Fetching from {PUBLIC_REPO}...")
    run_command(["git", "fetch", PUBLIC_REPO], check=False)
    
    # Check if branch exists
    returncode, branches, _ = run_command(
        ["git", "branch", "-r", "--list", f"{PUBLIC_REPO}/{BRANCH_NAME}"],
        check=False
    )
    
    if f"{PUBLIC_REPO}/{BRANCH_NAME}" in branches:
        print(f"📝 Found existing {BRANCH_NAME} branch, switching to it")
        run_command(["git", "checkout", "-b", BRANCH_NAME, f"{PUBLIC_REPO}/{BRANCH_NAME}"], check=False)
    else:
        print(f"📝 Creating new {BRANCH_NAME} branch")
        run_command(["git", "checkout", "-b", BRANCH_NAME], check=False)
    
    # Stage reports
    print("\n📦 Staging reports and reproducibility materials...")
    
    staged_count = 0
    missing_files = []
    
    for file_path in REPORTS_TO_PUSH:
        file = DEV_DIR / file_path
        if file.exists():
            returncode, stdout, stderr = run_command(
                ["git", "add", str(file_path)],
                check=False
            )
            if returncode == 0:
                staged_count += 1
                print(f"  ✅ Staged: {file_path}")
            else:
                print(f"  ⚠️  Failed to stage: {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ⚠️  Missing: {file_path}")
    
    # Also stage analysis data directory if it exists
    analysis_dir = DEV_DIR / "data" / "3i_atlas_analysis"
    if analysis_dir.exists():
        returncode, stdout, stderr = run_command(
            ["git", "add", "data/3i_atlas_analysis/"],
            check=False
        )
        if returncode == 0:
            staged_count += 1
            print(f"  ✅ Staged: data/3i_atlas_analysis/")
    
    print(f"\n✅ Staged {staged_count} files/directories")
    if missing_files:
        print(f"⚠️  {len(missing_files)} files not found")
    
    # Commit
    print("\n💾 Committing reports...")
    commit_message = """Add Reports and Reproducibility Materials

- NotebookLM 45-minute Q&A guides (ultra-concise, concise, complete, script)
- 3I/ATLAS image analysis (Gaussian splatting, spectral analysis)
- 3I/ATLAS analysis tools and reports
- Dev folder organization reports
- Kid-friendly papers organization
- Wallace Transform Final documentation
- Complete consciousness mathematics master
- All reproducibility materials for research validation

All materials ready for public review and reproduction."""
    
    returncode, stdout, stderr = run_command(
        ["git", "commit", "-m", commit_message],
        check=False
    )
    
    if returncode == 0:
        print("✅ Committed reports")
    else:
        if "nothing to commit" in stderr.lower():
            print("ℹ️  Nothing to commit (files may already be committed)")
        else:
            print(f"⚠️  Commit warning: {stderr}")
    
    # Push
    print(f"\n🚀 Pushing to {PUBLIC_REPO}/{BRANCH_NAME}...")
    returncode, stdout, stderr = run_command(
        ["git", "push", "-u", PUBLIC_REPO, BRANCH_NAME],
        check=False
    )
    
    if returncode == 0:
        print(f"✅ Pushed to {PUBLIC_REPO}/{BRANCH_NAME}")
    else:
        print(f"⚠️  Push result: {stderr}")
    
    # Return to original branch
    if current_branch:
        print(f"\n🔄 Returning to original branch: {current_branch}")
        run_command(["git", "checkout", current_branch], check=False)
    
    return returncode == 0


def main():
    """Main execution"""
    success = push_reports_to_public_repo()
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE - SUMMARY")
    print("=" * 70)
    print()
    print(f"📦 Files staged: {len(REPORTS_TO_PUSH)}")
    print(f"🌿 Branch: {BRANCH_NAME}")
    print(f"📦 Repo: {PUBLIC_REPO} (public)")
    print()
    
    if success:
        print("🎯 All reports pushed successfully!")
    else:
        print("⚠️  Some issues encountered - check output above")
    
    print()
    print("📚 Categories pushed:")
    print("  - NotebookLM Q&A guides")
    print("  - 3I/ATLAS analysis reports")
    print("  - Analysis tools (Gaussian splatting, spectral)")
    print("  - Organization reports")
    print("  - Wallace Transform documentation")
    print("  - Reproducibility materials")


if __name__ == "__main__":
    main()

