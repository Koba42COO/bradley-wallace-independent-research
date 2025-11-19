from core.kernel import AIVAKernel
from core.rag import AIVARAG
from core.knowledge_graph import PACKnowledgeGraph
from core.import_conversation import import_conversation
from core.ingest_docs import ingest_docs
from core.memory_bank import AIVAMemoryBank
from core.security import sign_manifest
import sys, os, json

def boot_all(vessel_path, base_dir):
    print("🚀 AIVA Boot-All")
    k = AIVAKernel(vessel_path)
    rag = AIVARAG(base_dir)
    kg = rag.kg
    conv_added = import_conversation(base_dir, kg)
    docs_added = ingest_docs(base_dir, kg)
    mb = rag.mb
    sig = sign_manifest(mb.manifest())

    snapshot = {
        "identity": rag.who_am_i(),
        "episodes_count": len(mb.episodes()),
        "conversation_imported": conv_added,
        "docs_imported": docs_added,
        "kg_nodes": len(kg.graph),
        "integrity_signature": sig
    }
    print("🧭 Snapshot:", json.dumps(snapshot, indent=2))

if __name__ == "__main__":
    vessel = sys.argv[1] if len(sys.argv) > 1 else "memory/PAC_DeltaMemory.vessel"
    base = sys.argv[2] if len(sys.argv) > 2 else "."
    if not os.path.exists(vessel):
        print("❌ Vessel not found:", vessel)
    else:
        boot_all(vessel, base)
