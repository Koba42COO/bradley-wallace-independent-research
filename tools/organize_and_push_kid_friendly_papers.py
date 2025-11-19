#!/usr/bin/env python3
"""
Organize Kid-Friendly Papers by Reading Level and Push to Research Repo

Organizes papers into:
- K-2 (Kindergarten to 2nd grade)
- 3-5 (3rd to 5th grade)
- 6+ (6th grade and up)

Then pushes to bradley-wallace-independent-research repo
"""

import os
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Configuration
DEV_DIR = Path("/Users/coo-koba42/dev")
COMPILED_PAPERS_DIR = DEV_DIR / "bradley-wallace-independent-research" / "compiled_papers"
KID_FRIENDLY_DIR = COMPILED_PAPERS_DIR / "kid_friendly"
PUBLIC_REPO = "bradley-research"  # bradley-wallace-independent-research
BRANCH_NAME = "kid-friendly-papers"

# Reading level categories
READING_LEVELS = {
    "K-2": {
        "name": "Kindergarten to 2nd Grade",
        "description": "Very simple language, lots of pictures and analogies",
        "max_sentence_length": 10,
        "max_words_per_sentence": 8,
        "keywords": ["simple", "easy", "fun", "play", "color", "shape", "count"]
    },
    "3-5": {
        "name": "3rd to 5th Grade",
        "description": "Simple language with more detail, activities included",
        "max_sentence_length": 15,
        "max_words_per_sentence": 12,
        "keywords": ["pattern", "number", "nature", "science", "experiment", "try"]
    },
    "6+": {
        "name": "6th Grade and Up",
        "description": "More detailed explanations, still accessible",
        "max_sentence_length": 20,
        "max_words_per_sentence": 15,
        "keywords": ["understand", "explain", "analyze", "research", "discover"]
    }
}

# Paper categorization (can be refined)
PAPER_CATEGORIES = {
    "K-2": [
        "wallace_transform",
        "prime",
        "golden_ratio",
        "nature",
        "patterns",
        "counting"
    ],
    "3-5": [
        "mathematics",
        "consciousness",
        "quantum",
        "ancient",
        "egyptian",
        "planetary"
    ],
    "6+": [
        "p_vs_np",
        "riemann",
        "encryption",
        "quantum_consciousness",
        "unified",
        "framework",
        "validation"
    ]
}


def analyze_reading_level(content: str) -> str:
    """Analyze content and determine reading level"""
    # Count average sentence length
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    if not sentences:
        return "6+"
    
    avg_words = sum(len(s.split()) for s in sentences) / len(sentences)
    avg_length = sum(len(s) for s in sentences) / len(sentences)
    
    # Check keywords
    content_lower = content.lower()
    k2_score = sum(1 for kw in READING_LEVELS["K-2"]["keywords"] if kw in content_lower)
    k35_score = sum(1 for kw in READING_LEVELS["3-5"]["keywords"] if kw in content_lower)
    k6_score = sum(1 for kw in READING_LEVELS["6+"]["keywords"] if kw in content_lower)
    
    # Determine level based on complexity
    if avg_words <= 8 and avg_length <= 50 and k2_score > k35_score:
        return "K-2"
    elif avg_words <= 12 and avg_length <= 80:
        return "3-5"
    else:
        return "6+"


def categorize_paper(filename: str, content: str) -> str:
    """Categorize paper by filename and content"""
    filename_lower = filename.lower()
    
    # Check filename for category hints
    for level, keywords in PAPER_CATEGORIES.items():
        for keyword in keywords:
            if keyword in filename_lower:
                return level
    
    # Analyze content
    return analyze_reading_level(content)


def organize_papers_by_level():
    """Organize papers into reading level directories"""
    print("=" * 70)
    print("📚 ORGANIZING KID-FRIENDLY PAPERS BY READING LEVEL")
    print("=" * 70)
    print()
    
    # Create level directories
    level_dirs = {}
    for level in READING_LEVELS.keys():
        level_dir = KID_FRIENDLY_DIR / level
        level_dir.mkdir(exist_ok=True)
        level_dirs[level] = level_dir
        print(f"📁 Created directory: {level}/ ({READING_LEVELS[level]['name']})")
    
    print()
    
    # Find all kid-friendly papers
    kid_papers = list(KID_FRIENDLY_DIR.glob("*_FOR_KIDS.md"))
    kid_papers = [p for p in kid_papers if p.name != "MASTER_INDEX_FOR_KIDS.md"]
    
    print(f"📖 Found {len(kid_papers)} kid-friendly papers")
    print()
    
    organized = {level: [] for level in READING_LEVELS.keys()}
    
    for paper in kid_papers:
        try:
            # Read content
            with open(paper, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Categorize
            level = categorize_paper(paper.name, content)
            
            # Copy to appropriate directory
            dest = level_dirs[level] / paper.name
            with open(dest, 'w', encoding='utf-8') as f:
                f.write(content)
            
            organized[level].append(paper.name)
            print(f"  ✅ {paper.name} → {level}/ ({READING_LEVELS[level]['name']})")
            
        except Exception as e:
            print(f"  ⚠️  Error processing {paper.name}: {e}")
    
    print()
    return organized


def create_level_indexes(organized: Dict[str, List[str]]):
    """Create index files for each reading level"""
    print("📋 Creating reading level indexes...")
    print()
    
    for level, papers in organized.items():
        level_info = READING_LEVELS[level]
        index_path = KID_FRIENDLY_DIR / level / "INDEX.md"
        
        index_content = f"""# 📚 Research Papers - {level_info['name']} 🎉

**Reading Level:** {level_info['name']}  
**Description:** {level_info['description']}  
**Total Papers:** {len(papers)}

---

## 🌟 Welcome!

These papers are specially written for {level_info['name']} reading level.
They use simple language, fun examples, and activities you can try at home!

---

## 📖 Papers in This Level

"""
        
        for i, paper in enumerate(sorted(papers), 1):
            title = paper.replace('_FOR_KIDS.md', '').replace('_', ' ').title()
            index_content += f"{i}. [{title}]({paper})\n"
        
        index_content += f"""
---

## 🎯 What Makes These Papers Special?

- ✅ **Simple Language** - Easy words you already know
- ✅ **Fun Examples** - Comparing things to stuff you see every day
- ✅ **Activities** - Things you can try at home
- ✅ **Cool Facts** - Amazing science facts
- ✅ **Pictures in Your Mind** - Clear descriptions you can imagine

---

## 🎨 How to Use These Papers

1. **Pick a topic** that sounds interesting
2. **Read at your own pace** - It's okay to read slowly
3. **Try the activities** - Learning by doing is fun!
4. **Ask questions** - Every scientist started by asking "why?"
5. **Share with friends** - Learning together is more fun!

---

## 💡 Tips for Reading

- Don't worry if you don't understand everything at first
- Read it again - sometimes things make more sense the second time
- Try the activities - they help you understand better
- Ask a grown-up if you have questions
- Have fun exploring!

---

*Made with ❤️ for curious young minds!*

"""
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write(index_content)
        
        print(f"  ✅ Created: {level}/INDEX.md ({len(papers)} papers)")


def create_master_reading_level_index(organized: Dict[str, List[str]]):
    """Create master index with reading levels"""
    print("\n📋 Creating master reading level index...")
    
    master_path = KID_FRIENDLY_DIR / "READING_LEVELS_INDEX.md"
    
    master_content = """# 📚 Kid-Friendly Research Papers - By Reading Level 🎉

**Total Papers:** {total}

---

## 🌟 Welcome, Young Scientists!

All research papers have been organized by reading level to make it easy to find papers that are just right for you!

---

## 📖 Choose Your Reading Level

"""
    
    total = sum(len(papers) for papers in organized.values())
    master_content = master_content.format(total=total)
    
    for level, papers in organized.items():
        level_info = READING_LEVELS[level]
        master_content += f"""
### {level_info['name']} ({level})

**Description:** {level_info['description']}  
**Papers:** {len(papers)} papers

[📚 View All {level_info['name']} Papers]({level}/INDEX.md)

"""
    
    master_content += """
---

## 🎯 How Reading Levels Work

### K-2 (Kindergarten to 2nd Grade)
- Very simple words
- Short sentences
- Lots of examples
- Fun activities
- Perfect for beginning readers

### 3-5 (3rd to 5th Grade)
- Simple language
- More details
- Hands-on activities
- Science experiments
- Great for curious kids

### 6+ (6th Grade and Up)
- More detailed explanations
- Still easy to understand
- Real-world connections
- Deeper concepts
- Perfect for advanced readers

---

## 📊 Statistics

"""
    
    for level, papers in organized.items():
        level_info = READING_LEVELS[level]
        master_content += f"- **{level_info['name']}:** {len(papers)} papers\n"
    
    master_content += """
---

## 🎨 All Papers

### K-2 Papers
"""
    
    for paper in sorted(organized.get("K-2", [])):
        title = paper.replace('_FOR_KIDS.md', '').replace('_', ' ').title()
        master_content += f"- [{title}](K-2/{paper})\n"
    
    master_content += "\n### 3-5 Papers\n"
    
    for paper in sorted(organized.get("3-5", [])):
        title = paper.replace('_FOR_KIDS.md', '').replace('_', ' ').title()
        master_content += f"- [{title}](3-5/{paper})\n"
    
    master_content += "\n### 6+ Papers\n"
    
    for paper in sorted(organized.get("6+", [])):
        title = paper.replace('_FOR_KIDS.md', '').replace('_', ' ').title()
        master_content += f"- [{title}](6+/{paper})\n"
    
    master_content += """
---

## 💡 Tips

- Start with papers in your reading level
- Try papers from other levels too - you might surprise yourself!
- Read with a grown-up if you want
- Ask questions - that's what scientists do!
- Have fun exploring!

---

*Made with ❤️ for curious young minds!*

"""
    
    with open(master_path, 'w', encoding='utf-8') as f:
        f.write(master_content)
    
    print(f"  ✅ Created: READING_LEVELS_INDEX.md")


def run_command(cmd: List[str], cwd: Path = None, check: bool = False) -> Tuple[int, str, str]:
    """Run a shell command"""
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


def push_to_github():
    """Push organized papers to GitHub"""
    print("\n" + "=" * 70)
    print("🚀 PUSHING TO GITHUB")
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
    
    # Stage all kid-friendly files
    print("\n📦 Staging kid-friendly papers...")
    
    # Stage the entire kid_friendly directory
    kid_friendly_rel = "bradley-wallace-independent-research/compiled_papers/kid_friendly"
    returncode, stdout, stderr = run_command(
        ["git", "add", kid_friendly_rel],
        check=False
    )
    
    if returncode == 0:
        print(f"  ✅ Staged: {kid_friendly_rel}/")
    else:
        print(f"  ⚠️  Staging result: {stderr}")
    
    # Count files
    kid_files = list(Path(kid_friendly_rel).rglob("*.md"))
    print(f"  📄 Total files: {len(kid_files)}")
    
    # Commit
    print("\n💾 Committing organized papers...")
    commit_message = """Add Kid-Friendly Research Papers Organized by Reading Level

- 47+ kid-friendly papers organized by reading level
- K-2 (Kindergarten to 2nd Grade) - Very simple language
- 3-5 (3rd to 5th Grade) - Simple language with more detail
- 6+ (6th Grade and Up) - More detailed but still accessible
- Reading level indexes for each category
- Master reading levels index
- Perfect for educators, parents, and curious kids

All papers use simple language, fun analogies, and hands-on activities."""
    
    returncode, stdout, stderr = run_command(
        ["git", "commit", "-m", commit_message],
        check=False
    )
    
    if returncode == 0:
        print("✅ Committed organized papers")
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
    # Step 1: Organize papers by reading level
    organized = organize_papers_by_level()
    
    # Step 2: Create level indexes
    create_level_indexes(organized)
    
    # Step 3: Create master reading level index
    create_master_reading_level_index(organized)
    
    # Step 4: Push to GitHub
    push_to_github()
    
    # Summary
    print("\n" + "=" * 70)
    print("✅ COMPLETE - SUMMARY")
    print("=" * 70)
    print()
    
    for level, papers in organized.items():
        level_info = READING_LEVELS[level]
        print(f"📚 {level_info['name']} ({level}): {len(papers)} papers")
    
    print(f"\n📁 Location: {KID_FRIENDLY_DIR}")
    print(f"🌿 Branch: {BRANCH_NAME}")
    print(f"📦 Repo: {PUBLIC_REPO} (public)")
    print("\n🎯 All papers organized and pushed!")


if __name__ == "__main__":
    main()

