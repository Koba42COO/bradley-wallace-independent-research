#!/usr/bin/env python3
"""
Create Kid-Friendly Versions of Compiled Research Papers

Converts complex research papers into easy-to-read versions for grade 6 and under,
using simple analogies, fun examples, and engaging language.
"""

import os
import re
from pathlib import Path
from typing import List, Dict

# Configuration
DEV_DIR = Path("/Users/coo-koba42/dev")
COMPILED_PAPERS_DIR = DEV_DIR / "bradley-wallace-independent-research" / "compiled_papers"
KID_FRIENDLY_DIR = COMPILED_PAPERS_DIR / "kid_friendly"

# Concept mappings - complex ideas to simple analogies
CONCEPT_MAPPINGS = {
    # Mathematics
    "prime number": "special numbers that can only be divided by 1 and themselves, like 2, 3, 5, 7, 11...",
    "golden ratio": "a special number (about 1.618) that appears in nature, like in flower petals and seashells",
    "consciousness": "being aware and thinking - like when you know you're reading this!",
    "mathematics": "the language of numbers and patterns",
    "algorithm": "a step-by-step recipe to solve a problem",
    "quantum": "the tiniest pieces of everything, smaller than atoms!",
    "frequency": "how fast something vibrates or repeats, like a drum beat",
    "energy": "the power that makes things happen, like the sun's light",
    "space-time": "the fabric of the universe, like a trampoline that bends",
    "dimension": "a direction you can move in - we live in 3D (up/down, left/right, forward/back)",
    
    # Wallace Transform specific
    "Wallace Transform": "a special math tool that helps us understand patterns in nature",
    "PAC": "a smart way to solve problems using patterns",
    "delta scaling": "making things bigger or smaller in a special way",
    "reality distortion": "seeing things in a new way that reveals hidden patterns",
    
    # Physics
    "electron": "tiny particles that carry electricity, like little messengers",
    "photon": "particles of light, like tiny packets of sunshine",
    "muon": "super-fast particles from space, like cosmic speedsters",
    "magnetic field": "an invisible force around magnets, like an invisible bubble",
    "gravity": "the force that pulls things together, like Earth pulling you down",
    
    # Biology/Chemistry
    "molecule": "groups of atoms stuck together, like LEGO blocks",
    "atom": "the smallest building blocks of everything",
    "DNA": "the instruction manual for life, like a recipe book",
    "photosynthesis": "how plants make food from sunlight, like solar panels",
    
    # Consciousness Mathematics
    "consciousness mathematics": "using math to understand how we think and are aware",
    "prime topology": "a special map of prime numbers in 21 dimensions",
    "zeta function": "a special math function that helps us understand prime numbers",
    "Riemann Hypothesis": "a famous math puzzle about where zeros appear",
}

# Simple sentence patterns
SIMPLE_PATTERNS = [
    (r"([A-Z][^.]*\.)", lambda m: m.group(1) if len(m.group(1)) < 100 else m.group(1)[:97] + "..."),
    (r"\b(?:therefore|consequently|furthermore|moreover|additionally)\b", "and"),
    (r"\b(?:utilize|employ|implement)\b", "use"),
    (r"\b(?:demonstrate|illustrate|exemplify)\b", "show"),
    (r"\b(?:facilitate|enable|permit)\b", "help"),
    (r"\b(?:comprehensive|extensive|thorough)\b", "complete"),
    (r"\b(?:significance|importance|relevance)\b", "importance"),
    (r"\b(?:hypothesis|postulate|proposition)\b", "idea"),
    (r"\b(?:validate|verify|confirm)\b", "prove"),
    (r"\b(?:optimize|enhance|improve)\b", "make better"),
]


def simplify_text(text: str) -> str:
    """Simplify text for grade 6 reading level"""
    # Replace complex concepts with simple explanations
    for complex, simple in CONCEPT_MAPPINGS.items():
        # Case-insensitive replacement
        pattern = re.compile(re.escape(complex), re.IGNORECASE)
        text = pattern.sub(simple, text)
    
    # Apply simple patterns
    for pattern, replacement in SIMPLE_PATTERNS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Break up long sentences
    sentences = re.split(r'([.!?]+)', text)
    simplified = []
    for i in range(0, len(sentences), 2):
        if i < len(sentences):
            sentence = sentences[i]
            if len(sentence) > 80:
                # Split on commas or conjunctions
                parts = re.split(r'([,;]|\band\b|\bor\b)', sentence)
                simplified.extend(parts)
            else:
                simplified.append(sentence)
            if i + 1 < len(sentences):
                simplified.append(sentences[i + 1])
    
    return ''.join(simplified)


def create_analogy(concept: str, context: str = "") -> str:
    """Create a simple analogy for a complex concept"""
    analogies = {
        "prime number": "Think of prime numbers like special keys that only open one lock. Regular numbers can be opened by many keys, but prime numbers are unique!",
        "golden ratio": "The golden ratio is like nature's favorite number. It shows up in flower petals, seashells, and even in how your fingers grow!",
        "consciousness": "Consciousness is like being the driver of a car. You're aware you're driving, you can see the road, and you can make choices!",
        "Wallace Transform": "The Wallace Transform is like a magic magnifying glass that helps us see hidden patterns in numbers and nature!",
        "quantum": "Quantum is like the tiniest LEGO blocks that make up everything. They're so small, they follow different rules than big things!",
        "electron": "Electrons are like tiny messengers that carry electricity. They zoom around like bees in a hive!",
        "frequency": "Frequency is like how fast you clap your hands. Fast clapping = high frequency, slow clapping = low frequency!",
        "space-time": "Space-time is like a trampoline. When you put something heavy on it, it bends, and that's how gravity works!",
    }
    
    for key, analogy in analogies.items():
        if key.lower() in concept.lower():
            return analogy
    
    return f"Think of {concept} like a special tool that helps us understand the world better!"


def extract_key_concepts(text: str) -> List[str]:
    """Extract key concepts from text"""
    concepts = []
    
    # Look for capitalized terms (likely concepts)
    capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    concepts.extend(capitalized[:10])  # Limit to first 10
    
    # Look for quoted terms
    quoted = re.findall(r'"([^"]+)"', text)
    concepts.extend(quoted[:5])
    
    return list(set(concepts))[:15]  # Return unique concepts, max 15


def create_kid_friendly_paper(compiled_file: Path) -> str:
    """Create a kid-friendly version of a compiled paper"""
    print(f"  📝 Processing: {compiled_file.name}")
    
    # Read the compiled paper
    try:
        with open(compiled_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"    ⚠️  Error reading file: {e}")
        return None
    
    # Extract title
    title_match = re.search(r'^#\s+(.+?)(?:\n|$)', content, re.MULTILINE)
    title = title_match.group(1) if title_match else compiled_file.stem.replace('_COMPILED', '')
    
    # Extract abstract if available
    abstract_match = re.search(r'##\s+Abstract\s*\n\n(.+?)(?=\n##|\n---|\Z)', content, re.DOTALL)
    abstract = abstract_match.group(1).strip() if abstract_match else ""
    
    # Extract key concepts
    key_concepts = extract_key_concepts(content[:2000])  # First 2000 chars
    
    # Create kid-friendly version
    kid_content = f"""# {title} - For Kids! 🎉

**Reading Level:** Grade 6 and Under  
**Original Paper:** {compiled_file.name}

---

## 🌟 What Is This About?

{simplify_text(abstract[:500]) if abstract else "This paper is about understanding amazing patterns in math and nature!"}

---

## 🎯 Key Ideas (In Simple Words)

"""
    
    # Add analogies for key concepts
    for i, concept in enumerate(key_concepts[:10], 1):
        analogy = create_analogy(concept)
        kid_content += f"""
### {i}. {concept}

{analogy}

"""
    
    # Add simplified main content
    kid_content += """
---

## 📖 The Story (Simplified)

"""
    
    # Extract main content (skip abstract, theorems, etc.)
    main_content_match = re.search(
        r'(?:##\s+Full Paper Content|##\s+Paper Overview|##\s+Introduction)(.+?)(?=##\s+Theorems|##\s+Validation|##\s+Code|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    
    if main_content_match:
        main_text = main_content_match.group(1)
        # Simplify and limit length
        simplified = simplify_text(main_text[:3000])  # First 3000 chars
        kid_content += simplified + "\n\n"
    else:
        kid_content += """
This paper explores amazing patterns in mathematics and nature. 
Scientists use special tools to understand how numbers, energy, and consciousness work together.
It's like discovering hidden secrets in the universe!

"""
    
    # Add fun facts
    kid_content += """
---

## 🎨 Fun Facts!

"""
    
    fun_facts = [
        "Did you know that prime numbers are like the building blocks of all other numbers?",
        "The golden ratio (about 1.618) appears in sunflowers, pineapples, and even in your body!",
        "Electrons move so fast, they can be in two places at once!",
        "Consciousness is what makes you aware that you're reading this right now!",
        "Math is everywhere - in music, art, nature, and even in how you think!",
    ]
    
    for i, fact in enumerate(fun_facts, 1):
        kid_content += f"{i}. {fact}\n\n"
    
    # Add activities
    kid_content += """
---

## 🎮 Try This At Home!

### Activity 1: Find Prime Numbers
Look for numbers that can only be divided by 1 and themselves:
- 2, 3, 5, 7, 11, 13, 17, 19, 23...

### Activity 2: Look for Patterns
- Count the petals on a flower - many have 5, 8, or 13 petals (Fibonacci numbers!)
- Look at a seashell spiral - it follows the golden ratio!
- Watch how leaves grow on a plant - they follow mathematical patterns!

### Activity 3: Think About Thinking
Close your eyes and think: "I am thinking right now!"
That awareness of your own thinking is consciousness!

---

## 🌈 Why This Matters

Understanding these patterns helps us:
- Build better computers
- Understand how nature works
- Create new technologies
- Solve big problems
- Make the world a better place!

---

## 📚 Want to Learn More?

This is a simplified version of a real research paper. 
As you grow older and learn more math and science, 
you can read the full paper and understand even more amazing things!

**Remember:** Every scientist started by asking simple questions, just like you!

---

*Made with ❤️ for curious young minds!*

"""
    
    return kid_content


def process_all_papers():
    """Process all compiled papers and create kid-friendly versions"""
    print("=" * 70)
    print("🎨 CREATING KID-FRIENDLY VERSIONS OF RESEARCH PAPERS")
    print("=" * 70)
    print()
    
    # Create output directory
    KID_FRIENDLY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {KID_FRIENDLY_DIR}")
    print()
    
    # Find all compiled papers
    compiled_papers = list(COMPILED_PAPERS_DIR.glob("*_COMPILED.md"))
    
    print(f"📚 Found {len(compiled_papers)} compiled papers")
    print()
    
    created_count = 0
    failed_count = 0
    
    for paper in compiled_papers:
        try:
            kid_content = create_kid_friendly_paper(paper)
            
            if kid_content:
                # Create output filename
                kid_filename = paper.name.replace('_COMPILED.md', '_FOR_KIDS.md')
                kid_path = KID_FRIENDLY_DIR / kid_filename
                
                # Write kid-friendly version
                with open(kid_path, 'w', encoding='utf-8') as f:
                    f.write(kid_content)
                
                created_count += 1
                print(f"  ✅ Created: {kid_filename}")
            else:
                failed_count += 1
                print(f"  ❌ Failed: {paper.name}")
        except Exception as e:
            failed_count += 1
            print(f"  ❌ Error processing {paper.name}: {e}")
        
        print()
    
    # Create master index
    print("📋 Creating master index...")
    create_master_index(created_count)
    
    # Summary
    print("=" * 70)
    print("✅ COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"\n📚 Created: {created_count} kid-friendly papers")
    print(f"⚠️  Failed: {failed_count} papers")
    print(f"📁 Location: {KID_FRIENDLY_DIR}")
    print("\n🎯 All kid-friendly papers are ready!")


def create_master_index(count: int):
    """Create a master index of all kid-friendly papers"""
    index_content = f"""# 📚 Research Papers - For Kids! 🎉

**Reading Level:** Grade 6 and Under  
**Total Papers:** {count}

---

## 🌟 Welcome, Young Scientists!

This is a special collection of research papers made just for you! 
We've taken complex scientific ideas and made them easy to understand using:
- Simple words
- Fun analogies (comparing things to stuff you know)
- Cool examples
- Activities you can try at home

---

## 📖 All Papers

"""
    
    # List all kid-friendly papers
    kid_papers = sorted(KID_FRIENDLY_DIR.glob("*_FOR_KIDS.md"))
    
    for i, paper in enumerate(kid_papers, 1):
        # Extract title from file
        title = paper.stem.replace('_FOR_KIDS', '').replace('_', ' ').title()
        index_content += f"{i}. [{title}]({paper.name})\n"
    
    index_content += """
---

## 🎯 How to Use This Collection

1. **Pick a topic that interests you** - Math, nature, computers, or space!
2. **Read the paper** - Don't worry if you don't understand everything
3. **Try the activities** - Hands-on learning is the best!
4. **Ask questions** - Every scientist started by asking "why?"
5. **Share with friends** - Learning is more fun together!

---

## 🌈 Topics Covered

- **Mathematics:** Prime numbers, patterns, and special ratios
- **Nature:** How plants, animals, and the universe follow math rules
- **Computers:** How computers think and solve problems
- **Consciousness:** Understanding how we think and are aware
- **Physics:** The tiniest particles and biggest forces
- **Ancient Knowledge:** What people knew thousands of years ago

---

## 💡 Remember

- Science is for everyone, including you!
- It's okay to not understand everything at first
- Asking questions makes you a scientist
- Every expert was once a beginner
- Learning should be fun!

---

## 🎨 Special Features

Each paper includes:
- ✅ Simple explanations
- ✅ Fun analogies
- ✅ Cool facts
- ✅ Activities to try
- ✅ Why it matters

---

*Made with ❤️ for curious young minds!*

**Keep asking questions and exploring the world around you!** 🌟

"""
    
    index_path = KID_FRIENDLY_DIR / "MASTER_INDEX_FOR_KIDS.md"
    with open(index_path, 'w', encoding='utf-8') as f:
        f.write(index_content)
    
    print(f"  ✅ Created master index: {index_path.name}")


if __name__ == "__main__":
    process_all_papers()

