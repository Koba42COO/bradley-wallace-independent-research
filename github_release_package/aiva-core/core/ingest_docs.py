import json
from pathlib import Path

def ingest_docs(base_dir, kg):
    """Ingest documentation into knowledge graph"""
    docs_dir = Path(base_dir) / "data" / "docs"

    if not docs_dir.exists():
        return 0

    docs_imported = 0
    for doc_file in docs_dir.glob("*.json"):
        with open(doc_file, 'r') as f:
            doc_data = json.load(f)

        doc_id = f"doc_{doc_file.stem}"
        kg.store(doc_id, doc_data, {
            "prime_anchor": 139,  # Prime for documentation
            "resonance": 0.92,
            "links": []
        })
        docs_imported += 1

    return docs_imported
