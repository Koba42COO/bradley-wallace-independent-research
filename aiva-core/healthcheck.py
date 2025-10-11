import json, subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run_boot_all():
    cmd = [sys.executable, "-m", "core.boot_all", "memory/PAC_DeltaMemory.vessel", "."]
    print("→", " ".join(cmd))
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit(out.returncode)
    # Parse snapshot line
    for line in out.stdout.splitlines():
        if line.strip().startswith("🧭 Snapshot:"):
            js = out.stdout.split("🧭 Snapshot:", 1)[1].strip()
            snap = json.loads(js)
            assert snap["identity"]["content"]["name"] == "AIVA"
            assert snap["kg_nodes"] >= 1
            return snap
    raise SystemExit("Snapshot not found in boot output.")

if __name__ == "__main__":
    s1 = run_boot_all()
    # Simulate continuity by appending an episode and re-booting
    add = [sys.executable, "aiva_cli.py", "add-episode", "Healthcheck episode created."]
    subprocess.check_call(add, cwd=ROOT)
    s2 = run_boot_all()
    print("✓ Healthcheck passed. Episodes before/after:", s1["episodes_count"], "→", s2["episodes_count"])
