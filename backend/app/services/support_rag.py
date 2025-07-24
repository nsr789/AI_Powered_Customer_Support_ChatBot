"""
Support-RAG helper.

Answers FAQ-style questions by retrieving the most relevant snippets from the
Chroma collection ``support_kb`` that is built by `support_loader`.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from app.core.vector_store import get_collection

__all__ = ["answer"]


def _vector_top_k(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """Vector-similarity retrieval via Chroma."""
    col = get_collection("support_kb")
    result = col.query(query_texts=[query], n_results=k)
    return list(zip(result["documents"][0], result["metadatas"][0]))


def _keyword_fallback(query: str) -> Tuple[str, dict] | None:
    """
    Simple keyword scan across **all** documents.

    Returns the first (text, metadata) tuple whose text contains ANY token
    from the query (case-insensitive).  Returns ``None`` if nothing matches.
    """
    col = get_collection("support_kb")
    docs = col.get(include=["documents", "metadatas"])
    tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))

    for text, meta in zip(docs["documents"], docs["metadatas"]):
        if any(tok in text.lower() for tok in tokens):
            return text, meta
    return None


def _retrieve_docs(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """
    Retrieve documents using vector search with a keyword-based safety net.
    """
    pairs = _vector_top_k(query, k=k)

    # If vector search failed to produce anything relevant, fall back.
    if not pairs or all(not p[0] for p in pairs):
        kb_hit = _keyword_fallback(query)
        pairs = [kb_hit] if kb_hit else []

    return pairs


def answer(query: str) -> Dict:
    """
    Build a response dict consumed by the LangGraph router.

    Keys:
    - ``tool``    : literal "support"
    - ``answer``  : best-matching document text (or apology)
    - ``sources`` : list[dict]   – metadata for each returned doc
    """
    pairs = _retrieve_docs(query, k=3)

    if not pairs:
        # Still nothing? fall back to generic apology but keep sources key.
        return {
            "tool": "support",
            "answer": (
                "I'm sorry, I couldn't find an article covering that topic. "
                "Please reach out to our human support team."
            ),
            "sources": [],
        }

    best_doc, _ = pairs[0]
    sources = [meta or {} for _, meta in pairs]

    return {
        "tool": "support",
        "answer": best_doc,
        "sources": sources,
    }
