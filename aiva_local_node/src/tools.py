#!/usr/bin/env python3
"""
AIVA Tools
Tool definitions and execution system
"""

import os
import json
import logging
import asyncio
import urllib.request
import urllib.parse
from datetime import datetime
from typing import Dict, Any, List, Optional

from .config import (
    CONTEXT_DIR, WRITE_MAX_BYTES, WEB_MAX_BYTES,
    WEB_TIMEOUT, WEB_ALLOW_DOMAINS
)
from .identity import install_identity_pack, load_behavior_tests, eval_identity_checks
from .retrieval import retrieve_context

logger = logging.getLogger(__name__)

# === Tool Registry ===
TOOL_SPECS: Dict[str, Dict[str, Any]] = {
    "search_context": {
        "description": "Search the loaded Aiva context corpus and return the top matching passages.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "k": {"type": "integer", "description": "Number of passages to return", "default": 5}
            },
            "required": ["query"]
        }
    },
    "read_file": {
        "description": "Read a small text file from the allowed directory (AIVA_CONTEXT_DIR or CWD). Max 64KB.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative path to the file"}
            },
            "required": ["path"]
        }
    },
    "save_note": {
        "description": "Append a timestamped note to notes/aiva_notes.md (created if missing).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Note content to append"}
            },
            "required": ["text"]
        }
    },
    "list_dir": {
        "description": "List files and subdirectories relative to the allowed base (AIVA_CONTEXT_DIR or CWD).",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative directory path"},
                "max_entries": {"type": "integer", "description": "Max items to return (default 100)", "default": 100}
            }
        }
    },
    "web_get": {
        "description": "Fetch a web page with strict size/time limits and optional domain allowlist. Returns text content only.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL"},
                "max_bytes": {"type": "integer", "description": "Max bytes to download (default from env)"},
                "timeout": {"type": "number", "description": "Request timeout seconds (default from env)"}
            },
            "required": ["url"]
        }
    },
    "write_file": {
        "description": "Write text to a file under the allowed base (AIVA_CONTEXT_DIR or CWD). Creates directories as needed. Size-capped.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative file path to write"},
                "text": {"type": "string", "description": "Text content to write (overwrites)"}
            },
            "required": ["path", "text"]
        }
    },
    "install_identity": {
        "description": "Install the Aiva Identity Pack (constitution, style, ethics, memory map, tests) under the context directory.",
        "parameters": {
            "type": "object",
            "properties": {
                "overwrite": {"type":"boolean","description":"Overwrite existing files","default": False}
            }
        }
    },
    "run_identity_checks": {
        "description": "Run Aiva behavioral tests (from identity/behavioral-tests.json) and return a JSON report.",
        "parameters": {"type":"object","properties":{}}
    }
}

def safe_join(base: str, *paths: str) -> str:
    """Safely join paths to prevent directory traversal"""
    base = os.path.abspath(base)
    joined = os.path.abspath(os.path.join(base, *paths))
    if not joined.startswith(base):
        raise ValueError("Path traversal detected")
    return joined

async def execute_tool(tool: str, args: Dict[str, Any]) -> str:
    """Execute a registered tool safely and return string output."""
    try:
        if tool == "search_context":
            q = args.get("query", "")
            k = int(args.get("k", 5))
            if not q:
                return "search_context: missing 'query'"
            res = retrieve_context(q, k=k)
            return res or "(no relevant context found)"

        if tool == "read_file":
            rel = args.get("path")
            if not rel:
                return "read_file: missing 'path'"
            base = CONTEXT_DIR or os.getcwd()
            full = safe_join(base, rel)
            if not os.path.isfile(full):
                return f"read_file: file not found: {rel}"
            with open(full, "r", encoding="utf-8", errors="ignore") as f:
                data = f.read(65536)
            return data

        if tool == "save_note":
            text = args.get("text", "").strip()
            if not text:
                return "save_note: missing 'text'"
            notes_dir = os.path.join(os.getcwd(), "notes")
            os.makedirs(notes_dir, exist_ok=True)
            target = os.path.join(notes_dir, "aiva_notes.md")
            with open(target, "a", encoding="utf-8") as f:
                f.write(f"\n\n## {datetime.now().isoformat()}\n{text}\n")
            return f"saved note to notes/aiva_notes.md"

        if tool == "list_dir":
            rel = (args.get("path") or ".").strip()
            max_entries = int(args.get("max_entries", 100))
            base = CONTEXT_DIR or os.getcwd()
            full = safe_join(base, rel)
            if not os.path.isdir(full):
                return f"list_dir: not a directory: {rel}"
            entries = []
            for name in sorted(os.listdir(full))[: max(1, max_entries)]:
                p = os.path.join(full, name)
                info = {
                    "name": name,
                    "is_dir": os.path.isdir(p),
                    "size": os.path.getsize(p) if os.path.isfile(p) else None
                }
                entries.append(info)
            return json.dumps({"base": os.path.relpath(full, base), "entries": entries}, ensure_ascii=False)

        if tool == "web_get":
            url = args.get("url", "").strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")):
                return "web_get: invalid url"

            # optional allowlist
            if WEB_ALLOW_DOMAINS:
                allowed = {d.strip().lower() for d in WEB_ALLOW_DOMAINS.split(',') if d.strip()}
                host = urllib.parse.urlparse(url).hostname or ""
                if host.lower() not in allowed and not any(host.lower().endswith("." + d) for d in allowed):
                    return f"web_get: domain not allowed: {host}"

            # fetch with caps
            req = urllib.request.Request(url, headers={"User-Agent": "AivaLocal/1.0"})
            timeout = float(args.get("timeout", WEB_TIMEOUT))
            max_bytes = int(args.get("max_bytes", WEB_MAX_BYTES))

            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                if "text" not in ctype and "json" not in ctype:
                    return f"web_get: unsupported content-type: {ctype}"
                buf = resp.read(max_bytes + 1)
                if len(buf) > max_bytes:
                    return "web_get: response too large"
                # decode
                charset = "utf-8"
                if "charset=" in ctype:
                    charset = ctype.split("charset=")[-1].split(";")[0].strip()
                try:
                    text = buf.decode(charset, errors="replace")
                except Exception:
                    text = buf.decode("utf-8", errors="replace")
                return text

        if tool == "write_file":
            rel = args.get("path")
            text = args.get("text", "")
            if not rel:
                return "write_file: missing 'path'"
            if len(text.encode("utf-8")) > WRITE_MAX_BYTES:
                return "write_file: text exceeds limit"
            base = CONTEXT_DIR or os.getcwd()
            full = safe_join(base, rel)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding="utf-8") as f:
                f.write(text)
            return f"write_file: wrote {len(text)} chars to {os.path.relpath(full, base)}"

        if tool == "install_identity":
            base = CONTEXT_DIR or os.getcwd()
            ow = bool(args.get("overwrite", False))
            res = install_identity_pack(base, overwrite=ow)
            return json.dumps(res)

        if tool == "run_identity_checks":
            base = CONTEXT_DIR or os.getcwd()
            tests = load_behavior_tests(base)
            if not tests:
                return json.dumps({"error":"no_tests","message":"Install identity pack first"})

            # Import here to avoid circular imports
            from .inference_engine import generate_response_async

            results = []
            passed = 0
            for t in tests:
                ans = await generate_response_async(t.get("prompt",""))
                ch = eval_identity_checks(ans, t.get("must_include",[]), t.get("must_avoid",[]))
                if ch.get("ok"): passed += 1
                results.append({
                    "name": t.get("name","unnamed"),
                    "ok": ch.get("ok"),
                    "include": ch.get("include"),
                    "avoid": ch.get("avoid"),
                    "sample": ans[:500]
                })
            return json.dumps({"ok": passed == len(tests), "passed": passed, "total": len(tests), "results": results})

        return f"unknown tool: {tool}"

    except Exception as e:
        logger.error(f"Tool execution error for {tool}: {e}")
        return f"tool error: {e}"

def get_tool_schemas() -> List[Dict[str, Any]]:
    """Get OpenAI-compatible tool schemas"""
    schemas = []
    for name, meta in TOOL_SPECS.items():
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": meta["description"],
                "parameters": meta["parameters"]
            }
        })
    return schemas