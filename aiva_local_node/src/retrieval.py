#!/usr/bin/env python3
"""
AIVA Context Retrieval
Context retrieval and embedding functions for memory and knowledge
"""

import os
import json
import logging
import math
from typing import List, Dict, Any, Optional

from .config import (
    CONTEXT_DIR, MAX_CONTEXT_CHARS, MAX_CONTEXT_PASSAGES,
    USE_EMBEDDINGS, EMB_MODEL_NAME
)

logger = logging.getLogger(__name__)

# Global variables for retrieval system
_emb_model = None  # loaded SentenceTransformer or None
emb_items: List[Dict[str, Any]] = []  # {"path": str, "text": str, "vec": List[float]}
corpus_docs: List[Dict[str, Any]] = []  # each: {"path": str, "text": str}

def tokenize(text: str) -> List[str]:
    """Simple tokenization for text processing"""
    return [t.lower() for t in json.dumps(text).split()]  # simple, dependency-free

def score_passage(query: str, passage: str) -> float:
    """Score passage relevance using Jaccard similarity"""
    q = set(tokenize(query))
    p = set(tokenize(passage))
    if not q or not p:
        return 0.0
    # Jaccard overlap as a crude similarity
    return len(q & p) / len(q | p)

def load_corpus(dir_path: str) -> List[Dict[str, Any]]:
    """Load text corpus from directory"""
    docs: List[Dict[str, Any]] = []
    if not os.path.isdir(dir_path):
        logger.warning(f"AIVA_CONTEXT_DIR set but not a directory: {dir_path}")
        return docs

    for root, _, files in os.walk(dir_path):
        for fn in files:
            if any(fn.lower().endswith(ext) for ext in [".md", ".mdx", ".txt", ".json"]):
                full = os.path.join(root, fn)
                try:
                    with open(full, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                        # split into rough paragraphs to keep retrieval snippets concise
                        for chunk in text.split("\n\n"):
                            chunk = chunk.strip()
                            if chunk:
                                docs.append({"path": full, "text": chunk})
                except Exception as e:
                    logger.warning(f"Failed to read context file {full}: {e}")

    return docs

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

def maybe_load_embedder():
    """Load embedding model if not already loaded"""
    global _emb_model
    if _emb_model is not None:
        return _emb_model
    try:
        from sentence_transformers import SentenceTransformer
        _emb_model = SentenceTransformer(EMB_MODEL_NAME)
        logger.info(f"Embeddings model loaded: {EMB_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Embeddings disabled (failed to load {EMB_MODEL_NAME}): {e}")
        _emb_model = None
    return _emb_model

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for list of texts"""
    model = maybe_load_embedder()
    if not model:
        return []
    try:
        vecs = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        # ensure list-of-floats
        return [list(map(float, v)) for v in vecs]
    except Exception as e:
        logger.warning(f"Embedding encode failed: {e}")
        return []

def retrieve_context(query: str, k: int = MAX_CONTEXT_PASSAGES, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Retrieve relevant context for a query

    Args:
        query: Search query
        k: Number of passages to retrieve
        max_chars: Maximum character limit

    Returns:
        Formatted context string
    """
    if not query:
        return ""

    # Embeddings-first if available
    if USE_EMBEDDINGS and emb_items:
        qv = embed_texts([query])
        if qv and qv[0]:
            scored = sorted(emb_items, key=lambda it: cosine_similarity(qv[0], it["vec"]), reverse=True)
            selected, total = [], 0
            for it in scored:
                t = it["text"]
                if total + len(t) + 2 > max_chars:
                    break
                selected.append(t)
                total += len(t) + 2
                if len(selected) >= k:
                    break
            return "\n\n".join(selected)

    # Fallback to Jaccard-overlap retrieval
    if not corpus_docs:
        return ""

    ranked = sorted(corpus_docs, key=lambda d: score_passage(query, d["text"]), reverse=True)
    selected, total = [], 0
    for d in ranked[: max(k, 1) * 4]:
        if total + len(d["text"]) + 2 > max_chars:
            break
        selected.append(d["text"])
        total += len(d["text"]) + 2
        if len(selected) >= k:
            break
    return "\n\n".join(selected)

def build_embedding_index():
    """Build or rebuild embeddings index from corpus_docs."""
    global emb_items
    if not USE_EMBEDDINGS:
        emb_items = []
        return
    if not corpus_docs:
        emb_items = []
        return
    texts = [d["text"] for d in corpus_docs]
    vecs = embed_texts(texts)
    if not vecs or len(vecs) != len(texts):
        emb_items = []
        return
    emb_items = []
    for d, v in zip(corpus_docs, vecs):
        emb_items.append({"path": d["path"], "text": d["text"], "vec": v})
    logger.info(f"Embeddings index built with {len(emb_items)} items")

def initialize_retrieval():
    """Initialize the retrieval system"""
    global corpus_docs
    if CONTEXT_DIR:
        try:
            corpus_docs = load_corpus(CONTEXT_DIR)
            logger.info(f"Loaded Aiva context corpus: {len(corpus_docs)} passages from {CONTEXT_DIR}")
            build_embedding_index()
        except Exception as e:
            logger.warning(f"Failed to load Aiva context corpus: {e}")

def get_retrieval_stats() -> Dict[str, Any]:
    """Get statistics about the retrieval system"""
    return {
        "context_loaded": bool(corpus_docs),
        "context_docs": len(corpus_docs),
        "embeddings_enabled": USE_EMBEDDINGS,
        "embeddings_items": len(emb_items),
        "emb_model": EMB_MODEL_NAME if USE_EMBEDDINGS else None
    }