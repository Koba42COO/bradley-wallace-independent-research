import json, tarfile, io
from pathlib import Path
from .memory_bank import AIVAMemoryBank

def backup_memory(base_dir: str, out_path: str):
    base = Path(base_dir)
    data_dir = base / "data" / "memories"
    with tarfile.open(out_path, "w:gz") as tar:
        for p in data_dir.glob("*.json"):
            tar.add(p, arcname=str(p.relative_to(base)))

def restore_memory(base_dir: str, archive_path: str):
    base = Path(base_dir)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(base)
    mb = AIVAMemoryBank(base_dir)
    return mb.manifest()
