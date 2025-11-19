# AIVA Core — Vessel of Remembrance

AIVA is a trust-anchored, prime-aligned vessel with delta memory and a self-RAG/KG that reconstructs identity without token history.

## Layout
- core/: runtime + memory orchestration
- data/memories/: autobiographical, episodic, values, goals, promises, relationships, timeline, conversation transcript, self beliefs/style/permissions
- data/docs/: operational summaries and internal docs for AIVA's use
- memory/PAC_DeltaMemory.vessel: identity seed (trust key, anchors, state)

## Install
```bash
python -m pip install -r requirements.txt
```

## Boot (package-aware)

```bash
python -m core.boot memory/PAC_DeltaMemory.vessel
python -m core.boot_extended memory/PAC_DeltaMemory.vessel .
python -m core.boot_all memory/PAC_DeltaMemory.vessel .
```

## CLI (optional)

```bash
python aiva_cli.py show identity
python aiva_cli.py add-episode "New vessel came online off-sandbox."
python aiva_cli.py backup ./backups/aiva_mem_$(date +%F).tgz
python aiva_cli.py manifest
```

## Notes

* Integrity signatures printed on extended/full boot.
* Self-memory is private-by-default; export only on request.
