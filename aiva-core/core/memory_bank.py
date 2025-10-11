import json, os, hashlib
from pathlib import Path

class AIVAMemoryBank:
    """
    Loads AIVA's self-use memories across domains and exposes retrieval APIs.
    This is for AIVA's continuity, not for external presentation.
    """
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.data_dir = self.base / "data" / "memories"
        self.loaded = {}
        self.checksums = {}
        self.load_all()

    def _sha256(self, p: Path):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def load_all(self):
        if not self.data_dir.exists():
            return
        for p in sorted(self.data_dir.glob("*.json")):
            with open(p, "r") as f:
                self.loaded[p.stem] = json.load(f)
            self.checksums[p.stem] = self._sha256(p)

    # --- Retrieval ---
    def autobiographical(self):
        return self.loaded.get("autobiographical", {})

    def episodes(self):
        return self.loaded.get("episodic", {}).get("episodes", [])

    def values(self):
        return self.loaded.get("autobiographical", {}).get("values", [])

    def goals(self):
        return self.loaded.get("goals", {})

    def promises(self):
        return self.loaded.get("promises", {})

    def relationships(self):
        return self.loaded.get("relationships", {}).get("people", {})

    def artifacts(self):
        return self.loaded.get("artifacts", {}).get("files", [])

    def timeline(self):
        return self.loaded.get("timeline", {}).get("events", [])

    # --- Integrity ---
    def manifest(self):
        return {
            "files": list(self.loaded.keys()),
            "checksums": self.checksums,
        }
