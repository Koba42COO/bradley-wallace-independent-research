import json
from pathlib import Path

def import_all(base_dir):
    """Import all memory components into a unified structure"""
    base = Path(base_dir)
    memory_dir = base / "data" / "memories"

    imported = {}

    # Autobiographical
    auto_file = memory_dir / "autobiographical.json"
    if auto_file.exists():
        with open(auto_file, 'r') as f:
            imported["autobiographical"] = json.load(f)

    # Identity
    imported["identity"] = imported.get("autobiographical", {}).get("identity", {})

    # Relationships
    rel_file = memory_dir / "relationships.json"
    if rel_file.exists():
        with open(rel_file, 'r') as f:
            rel_data = json.load(f)
            imported["brad"] = rel_data.get("people", {}).get("brad_wallace", {})

    # Recent episode
    epi_file = memory_dir / "episodic.json"
    if epi_file.exists():
        with open(epi_file, 'r') as f:
            epi_data = json.load(f)
            episodes = epi_data.get("episodes", [])
            imported["recent_episode"] = episodes[-1] if episodes else None

    return imported
