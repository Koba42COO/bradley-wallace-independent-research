import hashlib, json
from pathlib import Path

def file_sha256(path: str):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sign_manifest(manifest: dict, trust_key_hint: str = "AIVA-TRUST"):
    payload = json.dumps(manifest, sort_keys=True).encode("utf-8")
    return hashlib.sha256((trust_key_hint + "::").encode("utf-8") + payload).hexdigest()
