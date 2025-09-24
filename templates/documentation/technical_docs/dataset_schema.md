# aiva_convos.jsonl
One JSON per line:
{
  "conversations": [
    {"role":"system","content":"You are AiVA... (tone, ethics)"},
    {"role":"user","content":"<your prompt here>"},
    {"role":"assistant","content":"<AiVA response>"}
  ]
}

Notes:
- Keep answers grounded; redact private info before training.
- You can mix multiple pairs per line (multi-turn), but start simple (1–2 turns).
- More curated, high-quality turns >> more quantity.
