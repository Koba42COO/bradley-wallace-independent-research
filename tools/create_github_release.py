#!/usr/bin/env python3
"""
🧠 AIVA - Create GitHub Release
================================

Creates GitHub release using Cursor's GitHub permissions.
For koba42coo GitHub account.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: December 2024
"""

import json
import subprocess
import os
from pathlib import Path
from datetime import datetime


def detect_repo():
    """Detect GitHub repository"""
    try:
        result = subprocess.run(
            ['git', 'config', '--get', 'remote.origin.url'],
            capture_output=True,
            text=True,
            cwd='/Users/coo-koba42/dev'
        )
        
        if result.returncode == 0:
            url = result.stdout.strip()
            # Extract repo from URL
            if 'github.com' in url:
                if url.startswith('https://'):
                    repo = url.replace('https://github.com/', '').replace('.git', '')
                elif url.startswith('git@'):
                    repo = url.replace('git@github.com:', '').replace('.git', '')
                else:
                    repo = None
                
                if repo:
                    return repo
    except:
        pass
    
    return None


def create_release_via_git(repo_path='/Users/coo-koba42/dev', repo_name=None):
    """Create GitHub release using git commands"""
    print("🚀 Creating GitHub Release")
    print("=" * 70)
    print()
    
    if not repo_name:
        repo_name = detect_repo()
        if not repo_name:
            # Try to construct from username
            repo_name = 'koba42coo/dev'  # Default assumption
    
    print(f"Repository: {repo_name}")
    print()
    
    # Check if this is a git repo
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--git-dir'],
            capture_output=True,
            cwd=repo_path
        )
        
        if result.returncode != 0:
            print("⚠️  Not a git repository")
            return False
    except:
        print("⚠️  Git not available")
        return False
    
    # Create release tag
    tag_name = 'v1.0.0-benchmarks'
    tag_message = 'AIVA Benchmark Results - HumanEval #1 Rank'
    
    print(f"Creating tag: {tag_name}")
    try:
        # Check if tag exists
        result = subprocess.run(
            ['git', 'tag', '-l', tag_name],
            capture_output=True,
            text=True,
            cwd=repo_path
        )
        
        if tag_name in result.stdout:
            print(f"⚠️  Tag {tag_name} already exists")
            response = input("   Delete and recreate? (y/n): ").strip().lower()
            if response == 'y':
                subprocess.run(['git', 'tag', '-d', tag_name], cwd=repo_path)
                subprocess.run(['git', 'push', 'origin', '--delete', tag_name], cwd=repo_path, stderr=subprocess.DEVNULL)
        
        # Create annotated tag
        subprocess.run(
            ['git', 'tag', '-a', tag_name, '-m', tag_message],
            cwd=repo_path,
            check=True
        )
        print(f"✅ Tag created: {tag_name}")
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Error creating tag: {e}")
        return False
    
    # Push tag
    print(f"Pushing tag to GitHub...")
    try:
        subprocess.run(
            ['git', 'push', 'origin', tag_name],
            cwd=repo_path,
            check=True
        )
        print(f"✅ Tag pushed to GitHub")
        print()
        
        # Generate release URL
        release_url = f"https://github.com/{repo_name}/releases/new?tag={tag_name}"
        print(f"🌐 Create release at:")
        print(f"   {release_url}")
        print()
        print("📋 Next steps:")
        print("   1. Visit the URL above")
        print("   2. Title: AIVA Benchmark Results - HumanEval #1 Rank")
        print("   3. Description: Copy from github_release_package/RELEASE_NOTES.md")
        print("   4. Upload files from github_release_package/")
        print("   5. Click 'Publish release'")
        print()
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Could not push tag: {e}")
        print("   You may need to authenticate or set up git remote")
        print()
        print("   Try:")
        print("   git push origin v1.0.0-benchmarks")
        return False


def create_release_with_gh_cli(repo_name=None):
    """Try to create release using GitHub CLI if available"""
    print("🔍 Checking for GitHub CLI (gh)...")
    
    try:
        result = subprocess.run(['gh', '--version'], capture_output=True)
        if result.returncode == 0:
            print("✅ GitHub CLI found")
            
            if not repo_name:
                repo_name = detect_repo()
            
            if repo_name:
                # Read release notes
                release_notes_file = Path('github_release_package/RELEASE_NOTES.md')
                if release_notes_file.exists():
                    with open(release_notes_file, 'r') as f:
                        release_notes = f.read()
                else:
                    release_notes = "AIVA Benchmark Results - HumanEval #1 Rank"
                
                # Create release
                print(f"Creating release in {repo_name}...")
                try:
                    subprocess.run([
                        'gh', 'release', 'create',
                        'v1.0.0-benchmarks',
                        '--title', 'AIVA Benchmark Results - HumanEval #1 Rank',
                        '--notes', release_notes,
                        '--repo', repo_name
                    ], check=True)
                    
                    print("✅ Release created!")
                    return True
                except subprocess.CalledProcessError as e:
                    print(f"⚠️  Error creating release: {e}")
                    return False
            else:
                print("⚠️  Could not detect repository")
                return False
        else:
            print("⚠️  GitHub CLI not available")
            return False
    except FileNotFoundError:
        print("⚠️  GitHub CLI not installed")
        return False


def main():
    """Main execution"""
    repo_name = detect_repo()
    
    if repo_name:
        print(f"✅ Detected repository: {repo_name}")
    else:
        repo_name = 'koba42coo/dev'  # Default
        print(f"⚠️  Could not detect repo, using: {repo_name}")
    
    print()
    
    # Try GitHub CLI first (easiest)
    if create_release_with_gh_cli(repo_name):
        print("✅ Release created via GitHub CLI!")
        return
    
    # Fallback to git tag method
    print("Using git tag method...")
    print()
    if create_release_via_git(repo_path='/Users/coo-koba42/dev', repo_name=repo_name):
        print("✅ Tag created and pushed!")
        print("   Complete the release via GitHub web interface")
    else:
        print("⚠️  Could not create release automatically")
        print("   Use manual method: github_release_package/UPLOAD_INSTRUCTIONS.md")


if __name__ == "__main__":
    main()

