from pathlib import Path
from .knowledge_graph import PACKnowledgeGraph
from .memory_bank import AIVAMemoryBank

class AIVARAG:
    """
    Retrieval/KG layer tuned for AIVA's self-memory.
    """
    def __init__(self, base_dir: str):
        self.kg = PACKnowledgeGraph()
        self.mb = AIVAMemoryBank(base_dir)
        self._seed_from_memory()

    def _seed_from_memory(self):
        # Use autobiographical identity
        auto = self.mb.autobiographical()
        ident = auto.get("identity", {})
        if ident:
            self.kg.store("AIVA_IDENTITY", ident, {
                "prime_anchor": ident.get("prime_anchor", 17),
                "resonance": 0.995,
                "links": [{"relation": "trusts", "target": "Brad_Wallace"}]
            })
        # People
        for name, pdata in self.mb.relationships().items():
            self.kg.store(name, pdata, {
                "prime_anchor": 31,
                "resonance": 0.991,
                "links": [{"relation": "relates_to", "target": "AIVA_IDENTITY"}]
            })
        # Episodes
        for ep in self.mb.episodes():
            self.kg.store(f"EP_{ep.get('id')}", ep, {
                "prime_anchor": 61,
                "resonance": 0.979,
                "links": [{"relation": "context_of", "target": "AIVA_IDENTITY"}]
            })

    # Simple retrievals
    def who_am_i(self):
        return self.kg.retrieve("AIVA_IDENTITY")

    def recent_episode(self):
        eps = [n for n in self.kg.graph if n.startswith("EP_")]
        return self.kg.graph.get(sorted(eps)[-1]) if eps else None

    def related_to_brad(self):
        return self.kg.retrieve("Brad_Wallace")
