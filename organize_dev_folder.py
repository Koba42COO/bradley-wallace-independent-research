#!/usr/bin/env python3
"""
Organize Dev Folder - Comprehensive File Organization

Organizes the dev folder into logical categories:
- tools/ - All Python tools and scripts
- documentation/ - All markdown docs and reports
- research/ - Research papers and analysis
- benchmarks/ - Benchmark results and testing
- projects/ - Major project directories
- config/ - Configuration files
- data/ - Data files and outputs
- archives/ - Old/backup files
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import re

# Configuration
DEV_DIR = Path("/Users/coo-koba42/dev")

# Organization structure
ORGANIZATION = {
    "tools": {
        "name": "tools",
        "description": "Python tools and scripts",
        "patterns": [
            r".*\.py$",
            r".*\.sh$",
        ],
        "keywords": ["tool", "script", "system", "framework", "utility"],
        "exclude": ["test", "backup", "__pycache__", ".pyc"]
    },
    "documentation": {
        "name": "documentation",
        "description": "Documentation, reports, and guides",
        "patterns": [
            r".*\.md$",
            r".*\.txt$",
            r".*\.rst$",
        ],
        "keywords": ["doc", "readme", "guide", "report", "summary", "index"],
        "exclude": ["kid_friendly"]
    },
    "research": {
        "name": "research",
        "description": "Research papers and analysis",
        "patterns": [
            r".*\.tex$",
            r".*research.*",
            r".*analysis.*",
            r".*paper.*",
        ],
        "keywords": ["research", "paper", "analysis", "study", "thesis"],
        "exclude": []
    },
    "benchmarks": {
        "name": "benchmarks",
        "description": "Benchmark results and testing",
        "patterns": [
            r".*benchmark.*",
            r".*test.*",
            r".*validation.*",
        ],
        "keywords": ["benchmark", "test", "validation", "result"],
        "exclude": ["__pycache__"]
    },
    "data": {
        "name": "data",
        "description": "Data files, outputs, and results",
        "patterns": [
            r".*\.json$",
            r".*\.csv$",
            r".*\.png$",
            r".*\.jpg$",
            r".*\.wav$",
            r".*output.*",
            r".*result.*",
        ],
        "keywords": ["data", "output", "result", "json", "csv"],
        "exclude": ["__pycache__"]
    },
    "config": {
        "name": "config",
        "description": "Configuration files",
        "patterns": [
            r".*\.yaml$",
            r".*\.yml$",
            r".*\.toml$",
            r".*\.ini$",
            r".*\.cfg$",
            r".*\.env.*",
            r".*config.*",
        ],
        "keywords": ["config", "setting", "yaml", "toml", "ini"],
        "exclude": []
    },
    "projects": {
        "name": "projects",
        "description": "Major project directories",
        "patterns": [],
        "keywords": ["project", "repo", "module"],
        "exclude": [],
        "directories": [
            "bradley-wallace-independent-research",
            "consciousness_research_docs",
            "consciousness-mathematics",
            "fractal-harmonic-transform",
            "The-Wallace-Transformation-A-Complete-Unified-Framework-for-Consciousness-Mathematics-and-Reality",
            "Nonlinear-Approach-to-the-Riemann-Hypothesis",
            "Nonlinear-approach-to-The-Rheimann-Hypothesis-",
            "MicroManipulatorStepper",
            "clvm",
            "chia-blockchain",
            "gold_standard_benchmark",
            "aiva_benchmarks_repo",
            "benchmark_results_public",
            "benchmark_results_public_secure",
        ]
    }
}

# Files to keep in root
KEEP_IN_ROOT = [
    "README.md",
    "LICENSE",
    ".gitignore",
    ".gitattributes",
    ".cursorrules",
    ".cursorignore",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
]


def should_exclude(filepath: Path, exclude_list: List[str]) -> bool:
    """Check if file should be excluded"""
    filepath_str = str(filepath).lower()
    return any(exclude.lower() in filepath_str for exclude in exclude_list)


def matches_pattern(filepath: Path, patterns: List[str]) -> bool:
    """Check if file matches any pattern"""
    filename = filepath.name
    return any(re.match(pattern, filename, re.IGNORECASE) for pattern in patterns)


def matches_keywords(filepath: Path, keywords: List[str]) -> bool:
    """Check if file matches keywords"""
    filepath_str = str(filepath).lower()
    return any(keyword.lower() in filepath_str for keyword in keywords)


def categorize_file(filepath: Path) -> Tuple[str, bool]:
    """Categorize a file into organization structure"""
    # Skip if should be kept in root
    if filepath.name in KEEP_IN_ROOT:
        return "root", True
    
    # Skip directories
    if filepath.is_dir():
        return "skip", False
    
    # Skip hidden files and cache
    if filepath.name.startswith('.') or '__pycache__' in str(filepath):
        return "skip", False
    
    # Check each category
    for category, config in ORGANIZATION.items():
        # Check if in excluded directories
        if should_exclude(filepath, config.get("exclude", [])):
            continue
        
        # Check patterns
        if config["patterns"] and matches_pattern(filepath, config["patterns"]):
            return category, False
        
        # Check keywords
        if config["keywords"] and matches_keywords(filepath, config["keywords"]):
            return category, False
    
    # Default: keep in root or create "misc" category
    return "misc", False


def organize_files(dry_run: bool = True) -> Dict[str, List[Path]]:
    """Organize files into categories"""
    print("=" * 70)
    print("📁 ORGANIZING DEV FOLDER")
    print("=" * 70)
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be moved")
    else:
        print("🚀 LIVE MODE - Files will be moved")
    print()
    
    # Create category directories
    organized = {category: [] for category in ORGANIZATION.keys()}
    organized["misc"] = []
    organized["root"] = []
    organized["skip"] = []
    
    # Get all files in root
    root_files = [f for f in DEV_DIR.iterdir() if f.is_file()]
    
    print(f"📄 Found {len(root_files)} files in root directory")
    print()
    
    # Categorize files
    for filepath in root_files:
        category, keep = categorize_file(filepath)
        organized[category].append(filepath)
        
        if category != "skip":
            action = "KEEP" if keep else f"MOVE → {category}/"
            print(f"  {action}: {filepath.name}")
    
    print()
    
    # Create directories if not dry run
    if not dry_run:
        for category in ORGANIZATION.keys():
            category_dir = DEV_DIR / ORGANIZATION[category]["name"]
            category_dir.mkdir(exist_ok=True)
            print(f"📁 Created/verified: {category}/")
    
    print()
    return organized


def move_files(organized: Dict[str, List[Path]], dry_run: bool = True):
    """Move files to their categories"""
    print("=" * 70)
    print("📦 MOVING FILES")
    print("=" * 70)
    print()
    
    moved_count = 0
    error_count = 0
    
    for category, files in organized.items():
        if category in ["root", "skip"]:
            continue
        
        if not files:
            continue
        
        # Handle misc category
        if category == "misc":
            category_dir = DEV_DIR / "misc"
        else:
            category_dir = DEV_DIR / ORGANIZATION[category]["name"]
        
        print(f"\n📂 {category.upper()} ({len(files)} files)")
        
        for filepath in files:
            dest = category_dir / filepath.name
            
            # Handle duplicates
            if dest.exists() and not dry_run:
                # Add number suffix
                base = dest.stem
                ext = dest.suffix
                counter = 1
                while dest.exists():
                    dest = category_dir / f"{base}_{counter}{ext}"
                    counter += 1
            
            if dry_run:
                print(f"  🔍 Would move: {filepath.name} → {category}/{dest.name}")
            else:
                try:
                    shutil.move(str(filepath), str(dest))
                    moved_count += 1
                    print(f"  ✅ Moved: {filepath.name} → {category}/{dest.name}")
                except Exception as e:
                    error_count += 1
                    print(f"  ❌ Error moving {filepath.name}: {e}")
    
    print()
    return moved_count, error_count


def organize_projects():
    """Organize major project directories"""
    print("=" * 70)
    print("📁 ORGANIZING PROJECT DIRECTORIES")
    print("=" * 70)
    print()
    
    projects_dir = DEV_DIR / "projects"
    projects_dir.mkdir(exist_ok=True)
    
    project_dirs = ORGANIZATION["projects"]["directories"]
    moved_count = 0
    
    for project_name in project_dirs:
        project_path = DEV_DIR / project_name
        
        if project_path.exists() and project_path.is_dir():
            dest = projects_dir / project_name
            
            if dest.exists():
                print(f"  ⚠️  Already exists: projects/{project_name}")
            else:
                try:
                    shutil.move(str(project_path), str(dest))
                    moved_count += 1
                    print(f"  ✅ Moved: {project_name} → projects/")
                except Exception as e:
                    print(f"  ❌ Error moving {project_name}: {e}")
        else:
            print(f"  ⚠️  Not found: {project_name}")
    
    print()
    return moved_count


def create_organization_index(organized: Dict[str, List[Path]]):
    """Create index of organization"""
    print("=" * 70)
    print("📋 CREATING ORGANIZATION INDEX")
    print("=" * 70)
    print()
    
    index_path = DEV_DIR / "ORGANIZATION_INDEX.md"
    
    index_content = """# 📁 Dev Folder Organization Index

**Date Organized:** $(date)  
**Status:** Organized by category

---

## 📂 Directory Structure

"""
    
    for category, config in ORGANIZATION.items():
        files = organized.get(category, [])
        index_content += f"""
### {config['name'].upper()}/
**Description:** {config['description']}  
**Files:** {len(files)}

"""
        if files:
            for filepath in sorted(files)[:20]:  # Show first 20
                index_content += f"- {filepath.name}\n"
            if len(files) > 20:
                index_content += f"- ... and {len(files) - 20} more\n"
    
    # Misc files
    misc_files = organized.get("misc", [])
    if misc_files:
        index_content += f"""
### misc/
**Description:** Miscellaneous files  
**Files:** {len(misc_files)}

"""
        for filepath in sorted(misc_files)[:20]:
            index_content += f"- {filepath.name}\n"
        if len(misc_files) > 20:
            index_content += f"- ... and {len(misc_files) - 20} more\n"
    
    index_content += """
---

## 📊 Statistics

"""
    
    total_files = sum(len(files) for files in organized.values() if isinstance(files, list))
    
    for category, files in organized.items():
        if category not in ["skip"]:
            count = len(files)
            if count > 0:
                index_content += f"- **{category}:** {count} files\n"
    
    index_content += f"""
- **Total Files Organized:** {total_files}

---

## 🎯 How to Use

1. **Tools** - All Python scripts and utilities
2. **Documentation** - All markdown docs and reports
3. **Research** - Research papers and analysis
4. **Benchmarks** - Benchmark results and testing
5. **Data** - Data files, outputs, and results
6. **Config** - Configuration files
7. **Projects** - Major project directories
8. **Misc** - Files that don't fit other categories

---

*This organization makes it easier to find and manage files!*

"""
    
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"  ✅ Created: ORGANIZATION_INDEX.md")


def main():
    """Main execution"""
    import sys
    
    # Check for dry-run flag
    dry_run = "--dry-run" in sys.argv or "-d" in sys.argv
    
    # Step 1: Organize files
    organized = organize_files(dry_run=dry_run)
    
    # Step 2: Move files (if not dry run)
    if not dry_run:
        moved_count, error_count = move_files(organized, dry_run=False)
        
        # Step 3: Organize projects
        projects_moved = organize_projects()
        
        # Step 4: Create index
        create_organization_index(organized)
        
        # Summary
        print("=" * 70)
        print("✅ ORGANIZATION COMPLETE")
        print("=" * 70)
        print()
        print(f"📦 Files moved: {moved_count}")
        print(f"⚠️  Errors: {error_count}")
        print(f"📁 Projects moved: {projects_moved}")
        print()
        print("📋 See ORGANIZATION_INDEX.md for details")
    else:
        print("=" * 70)
        print("🔍 DRY RUN COMPLETE")
        print("=" * 70)
        print()
        print("Run without --dry-run to actually organize files")
        print()
        
        # Show summary
        for category, files in organized.items():
            if category != "skip" and files:
                print(f"{category}: {len(files)} files")


if __name__ == "__main__":
    main()

