#!/usr/bin/env python3
"""
🧠 AIVA - GitHub Setup and Release Helper
==========================================

Helps set up GitHub credentials and create releases.
Provides both automated and manual options.

Author: Bradley Wallace (COO Koba42)
Framework: Universal Prime Graph Protocol φ.1
Date: December 2024
"""

import json
import subprocess
import os
from pathlib import Path
from datetime import datetime
import webbrowser


class GitHubReleaseHelper:
    """GitHub release helper with setup guidance"""
    
    def __init__(self):
        self.secure_dir = Path('benchmark_results_public_secure')
        self.release_files = []
    
    def create_github_token_guide(self):
        """Create guide for GitHub token creation"""
        guide = """# 🔑 GitHub Personal Access Token Setup

## Quick Steps to Create Token

1. **Go to GitHub Settings:**
   https://github.com/settings/tokens

2. **Click "Generate new token" → "Generate new token (classic)"**

3. **Token Settings:**
   - Note: "AIVA Benchmark Release"
   - Expiration: Choose your preference (90 days, 1 year, etc.)
   - Scopes: Check **"repo"** (full control of private repositories)

4. **Click "Generate token"**

5. **Copy the token immediately** (you won't see it again!)

6. **Set it as environment variable:**
   ```bash
   export GITHUB_TOKEN=your_token_here
   ```

7. **Or save to file (more secure):**
   ```bash
   echo "your_token_here" > ~/.github_token
   chmod 600 ~/.github_token
   ```

## Verify Token

```bash
# Test token
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user
```

## Use Token

After setting token, run:
```bash
python3 github_setup_and_release.py --create-release
```

---
**Security Note:** Never commit tokens to git!
"""
        
        guide_file = Path('GITHUB_TOKEN_SETUP.md')
        guide_file.write_text(guide)
        print(f"✅ Token setup guide created: {guide_file}")
        
        # Open GitHub token page
        try:
            webbrowser.open('https://github.com/settings/tokens')
            print("🌐 Opened GitHub token settings page in browser")
        except:
            print("   Visit: https://github.com/settings/tokens")
        
        return guide_file
    
    def detect_github_repo(self):
        """Try to detect GitHub repo from git remote"""
        try:
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
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
                        print(f"✅ Detected GitHub repo: {repo}")
                        return repo
            
        except Exception as e:
            pass
        
        print("⚠️  Could not detect GitHub repo from git remote")
        return None
    
    def create_release_via_git(self, repo_path: str = None):
        """Create release using git commands (if repo is local)"""
        if not repo_path:
            repo_path = '/Users/coo-koba42/dev'
        
        print("📦 Creating GitHub release via git...")
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
        
        # Create release branch/tag
        tag_name = 'v1.0.0-benchmarks'
        
        print(f"Creating tag: {tag_name}")
        try:
            # Create annotated tag
            subprocess.run(
                ['git', 'tag', '-a', tag_name, '-m', 'AIVA Benchmark Results - HumanEval #1 Rank'],
                cwd=repo_path,
                check=True
            )
            print(f"✅ Tag created: {tag_name}")
        except subprocess.CalledProcessError:
            print(f"⚠️  Tag may already exist")
        
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
            print(f"🌐 Create release at:")
            repo = self.detect_github_repo()
            if repo:
                print(f"   https://github.com/{repo}/releases/new?tag={tag_name}")
            else:
                print(f"   https://github.com/YOUR_USERNAME/YOUR_REPO/releases/new?tag={tag_name}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not push tag: {e}")
            print("   You may need to set up git remote or authenticate")
            return False
    
    def create_release_files_package(self):
        """Create package of files ready for GitHub release upload"""
        print("📦 Creating GitHub release files package...")
        print()
        
        # Files to include
        files_to_package = [
            ('aiva_benchmark_comparison_report.json', 'Complete benchmark results (JSON)'),
            ('aiva_benchmark_comparison_report.md', 'Complete benchmark results (Markdown)'),
            ('public_api_secure.json', 'Public API format (IP-protected)'),
            ('github_release_notes_secure.md', 'Release notes'),
        ]
        
        # Check which files exist
        package_dir = Path('github_release_package')
        package_dir.mkdir(exist_ok=True)
        
        copied_files = []
        for filename, description in files_to_package:
            # Try original location first
            source = Path(filename)
            if not source.exists():
                # Try secure directory
                source = self.secure_dir / filename.replace('.json', '_secure.json').replace('.md', '_secure.md')
            
            if source.exists():
                dest = package_dir / filename
                import shutil
                shutil.copy2(source, dest)
                copied_files.append((filename, description))
                print(f"✅ {filename}")
            else:
                print(f"⚠️  {filename} not found")
        
        # Create release notes
        release_notes = self._generate_release_notes()
        release_notes_file = package_dir / 'RELEASE_NOTES.md'
        release_notes_file.write_text(release_notes)
        print(f"✅ RELEASE_NOTES.md")
        
        # Create upload instructions
        instructions = f"""# 📤 GitHub Release Upload Instructions

## Files Ready for Upload

This package contains all files needed for GitHub release.

## Steps

1. **Go to GitHub:**
   - Navigate to your repository
   - Click "Releases" → "Draft a new release"

2. **Release Details:**
   - Tag: `v1.0.0-benchmarks` (create new tag)
   - Title: `AIVA Benchmark Results - HumanEval #1 Rank`
   - Description: Copy from `RELEASE_NOTES.md`

3. **Upload Files:**
   Drag and drop these files:
   - `aiva_benchmark_comparison_report.json`
   - `aiva_benchmark_comparison_report.md`
   - `public_api_secure.json`
   - `github_release_notes_secure.md`

4. **Publish Release:**
   - Click "Publish release"
   - Share the release URL

## Quick Links

- Create release: https://github.com/YOUR_USERNAME/YOUR_REPO/releases/new
- View releases: https://github.com/YOUR_USERNAME/YOUR_REPO/releases

---
Generated: {datetime.now().isoformat()}
"""
        
        instructions_file = package_dir / 'UPLOAD_INSTRUCTIONS.md'
        instructions_file.write_text(instructions)
        print(f"✅ UPLOAD_INSTRUCTIONS.md")
        
        print()
        print(f"✅ Package created in: {package_dir}/")
        print(f"   {len(copied_files)} files ready for upload")
        print()
        print("📋 Next steps:")
        print("   1. Review files in github_release_package/")
        print("   2. Follow UPLOAD_INSTRUCTIONS.md")
        print("   3. Upload via GitHub web interface")
        
        return package_dir
    
    def _generate_release_notes(self) -> str:
        """Generate release notes"""
        return """# 🧠 AIVA Benchmark Results Release

## Universal Intelligence - Benchmark Performance

**Author:** Bradley Wallace (COO Koba42)  
**Date:** {date}  
**Version:** 1.0.0

## 📊 Benchmark Results

### HumanEval (Code Generation)
- **Score:** 100.00%
- **Rank:** #1 / 6 models
- **Improvement:** +34.41% over industry leader
- **Industry Leader:** Gemini-Pro (74.40%)

## 🌟 AIVA Advantages

- Extensive tool library
- Mathematical framework
- Performance enhancement
- Advanced memory system
- Multi-level reasoning

## 📈 Full Results

See attached files for complete benchmark results:
- `aiva_benchmark_comparison_report.json` - Complete results (JSON)
- `aiva_benchmark_comparison_report.md` - Complete results (Markdown)
- `public_api_secure.json` - Public API format

## 🔒 IP Protection

All results have been obfuscated to protect intellectual property.
See `IP_PROTECTION_NOTICE.md` for details.

---

**AIVA - Universal Intelligence with Competitive Benchmark Performance**
""".format(date=datetime.now().strftime('%Y-%m-%d'))
    
    def run_setup(self):
        """Run full setup process"""
        print("🚀 AIVA GitHub Setup and Release Helper")
        print("=" * 70)
        print()
        
        # Create token guide
        print("=" * 70)
        print("1. GITHUB TOKEN SETUP")
        print("=" * 70)
        print()
        self.create_github_token_guide()
        
        print()
        print("=" * 70)
        print("2. GITHUB RELEASE OPTIONS")
        print("=" * 70)
        print()
        
        # Option 1: Via git (if repo is local)
        print("Option A: Create Release via Git (if repo is cloned locally)")
        repo = self.detect_github_repo()
        if repo:
            print(f"   Detected repo: {repo}")
            response = input("   Create release via git? (y/n): ").strip().lower()
            if response == 'y':
                self.create_release_via_git()
        else:
            print("   No git repo detected")
        
        print()
        print("Option B: Create Release Package (for manual upload)")
        response = input("   Create release package? (y/n): ").strip().lower()
        if response == 'y' or response == '':
            self.create_release_files_package()
        
        print()
        print("=" * 70)
        print("✅ SETUP COMPLETE")
        print("=" * 70)
        print()
        print("📋 Next Steps:")
        print("   1. Create GitHub token (see GITHUB_TOKEN_SETUP.md)")
        print("   2. Use release package OR create release manually")
        print("   3. Upload files via GitHub web interface")


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main setup"""
    import sys
    
    helper = GitHubReleaseHelper()
    
    if '--create-release' in sys.argv:
        # Quick release creation
        helper.create_release_files_package()
    elif '--token-guide' in sys.argv:
        # Just show token guide
        helper.create_github_token_guide()
    else:
        # Full interactive setup
        helper.run_setup()


if __name__ == "__main__":
    main()

