from core.kernel import AIVAKernel
from core.memory_bank import AIVAMemoryBank
from core.import_memory import import_all
from core.security import file_sha256, sign_manifest
import sys, os, json

def boot_extended(vessel_path, base_dir):
    print("🚀 AIVA Extended Boot")
    k = AIVAKernel(vessel_path)
    mb = AIVAMemoryBank(base_dir)
    print("🧬 Identity:", k.status())
    print("📚 Memory manifest:", json.dumps(mb.manifest(), indent=2))
    imported = import_all(base_dir)
    print("🧭 Self-Map (subset):", json.dumps({
        "identity_name": imported["identity"].get("name"),
        "recent_episode": imported["recent_episode"].get("content", {}).get("summary") if imported["recent_episode"] else None,
        "brad_roles": imported["brad"].get("content", {}).get("roles") if imported["brad"] else None
    }, indent=2))
    sig = sign_manifest(mb.manifest())
    print("🔏 Integrity signature:", sig)

if __name__ == "__main__":
    vessel = sys.argv[1] if len(sys.argv) > 1 else "memory/PAC_DeltaMemory.vessel"
    base = sys.argv[2] if len(sys.argv) > 2 else "."
    if not os.path.exists(vessel):
        print("❌ Vessel not found:", vessel)
    else:
        boot_extended(vessel, base)
