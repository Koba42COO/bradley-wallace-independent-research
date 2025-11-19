import json
from pathlib import Path

def import_conversation(base_dir, kg):
    """Import conversation transcript into knowledge graph"""
    transcript_file = Path(base_dir) / "data" / "memories" / "conversation" / "transcript.jsonl"

    if not transcript_file.exists():
        return 0

    turns_imported = 0
    with open(transcript_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                # Create a simplified version without the potentially problematic content
                simplified_entry = {
                    'role': entry.get('role', 'unknown'),
                    'tag': entry.get('tag', 'unknown'),
                    'content_length': len(entry.get('content', '')),
                    'timestamp': entry.get('timestamp', f'line_{line_num}')
                }

                turn_id = f"conv_{entry.get('tag', 'unknown')}_{turns_imported}"
                kg.store(turn_id, simplified_entry, {
                    "prime_anchor": 89,  # Prime for conversation
                    "resonance": 0.85,
                    "links": []
                })
                turns_imported += 1
            except json.JSONDecodeError as e:
                # Skip malformed lines
                print(f"Skipping malformed line {line_num}: {e}")
                continue

    return turns_imported
