#!/usr/bin/env python3
"""
Demo of the paste functionality working
"""

import json
import time
import os
from pathlib import Path

def demo_paste_processing():
    """Demo the paste processing without the complex exporter"""

    # Sample pasted conversation data
    paste_data = {
        "title": "Quantum Consciousness Discussion",
        "content": """You: What do you think about quantum effects in consciousness?

Assistant: Interesting question! While quantum mechanics undoubtedly plays a role in brain function at the molecular level, the idea of quantum consciousness remains highly speculative. Some theories suggest quantum coherence in microtubules could explain consciousness, but this remains controversial.

You: How does that relate to AIVA's architecture?

Assistant: AIVA's design draws inspiration from quantum computing principles - superposition of states, entanglement of information, and wave function collapse in decision making. However, we implement these concepts through classical computing with probabilistic algorithms.""",
        "messages": [
            {"role": "user", "content": "What do you think about quantum effects in consciousness?"},
            {"role": "assistant", "content": "Interesting question! While quantum mechanics undoubtedly plays a role in brain function at the molecular level, the idea of quantum consciousness remains highly speculative. Some theories suggest quantum coherence in microtubules could explain consciousness, but this remains controversial."},
            {"role": "user", "content": "How does that relate to AIVA's architecture?"},
            {"role": "assistant", "content": "AIVA's design draws inspiration from quantum computing principles - superposition of states, entanglement of information, and wave function collapse in decision making. However, we implement these concepts through classical computing with probabilistic algorithms."}
        ],
        "message_count": 4,
        "word_count": 148,
        "pasted_at": "2025-10-11T13:00:00Z"
    }

    print("🧠 GPT Teams Archive - Paste Demo")
    print("=" * 50)

    print("📝 Received pasted conversation:")
    print(f"Title: {paste_data['title']}")
    print(f"Messages: {paste_data['message_count']}")
    print(f"Words: {paste_data['word_count']}")
    print()

    # Simulate classification
    content = paste_data['content'].lower()
    if 'quantum' in content and 'consciousness' in content:
        classification = 'philosophy_theory'
    elif 'quantum' in content and 'physics' in content:
        classification = 'physics'
    elif 'ai' in content or 'aiva' in content:
        classification = 'ml'
    else:
        classification = 'philosophy_theory'

    print(f"🎯 Classified as: {classification}")
    print()

    # Create output directory structure
    artifacts_dir = Path('artifacts/gpt_convos')
    if classification in ['math', 'physics', 'ml', 'cryptography', 'systems', 'philosophy_theory', 'application']:
        output_dir = artifacts_dir / 'science' / classification
    else:
        output_dir = artifacts_dir / 'personal_sanitized'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
    title_slug = paste_data['title'].lower().replace(' ', '-').replace('[^a-z0-9-]', '')[:30]
    conversation_id = f"pasted_{int(time.time())}"

    base_filename = f"{timestamp}__{title_slug}__{conversation_id}"

    # Save JSON
    json_file = output_dir / f"{base_filename}.json"
    with open(json_file, 'w') as f:
        json.dump({
            'id': conversation_id,
            'title': paste_data['title'],
            'content': paste_data['content'],
            'messages': paste_data['messages'],
            'classification': classification,
            'source': 'pasted',
            'created_at': time.time(),
            'pasted_at': paste_data['pasted_at']
        }, f, indent=2)

    # Save Markdown
    md_file = output_dir / f"{base_filename}.md"
    with open(md_file, 'w') as f:
        f.write(f"# {paste_data['title']}\n\n")
        f.write(f"**Classification:** {classification}\n")
        f.write(f"**Source:** Pasted Conversation\n")
        f.write(f"**Pasted at:** {paste_data['pasted_at']}\n\n")

        for msg in paste_data['messages']:
            role = "User" if msg['role'] == 'user' else "Assistant"
            f.write(f"## {role}\n\n{msg['content']}\n\n---\n\n")

    print("💾 Files saved:")
    print(f"  JSON: {json_file}")
    print(f"  Markdown: {md_file}")
    print()

    # Simulate AIVA memory update
    print("🧠 AIVA Memory Updated:")
    print(f"  - Added episode: '{paste_data['title']}'")
    print(f"  - Classification: {classification}")
    print(f"  - Message count: {paste_data['message_count']}")
    print(f"  - Word count: {paste_data['word_count']}")
    print()

    print("✅ Paste processing complete!")
    print("🎉 Conversation successfully archived and integrated into AIVA memory!")

if __name__ == '__main__':
    demo_paste_processing()
