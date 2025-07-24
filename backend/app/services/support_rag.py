"""
Light-weight Retrieval-Augmented QA over the support knowledge-base.

The router expects `support_answer(query: str) -> dict` that returns:
    {
        "answer":  str,
        "tool":    "support",
        "sources": List[dict]   # metadata of cited docs
    }
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.core.vector_store import get_collection

# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #


def _vector_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Semantic similarity search using Chroma’s native API."""
    col = get_collection("support_kb")
    res = col.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"],
    )
    # unwrap single-query response
    return [
        {"content": doc, "metadata": meta}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


def _keyword_search(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Fallback keyword scan across every stored doc/metadata."""
    col = get_collection("support_kb")
    tokens = [re.escape(tok) for tok in query.lower().split()]
    pattern = re.compile("|".join(tokens), re.I)

    res = col.get(include=["documents", "metadatas"])
    hits: List[Dict[str, Any]] = []
    for doc, meta in zip(res["documents"], res["metadatas"]):
        if pattern.search(doc.lower()) or pattern.search(str(meta).lower()):
            hits.append({"content": doc, "metadata": meta})
            if len(hits) >= k:
                break
    return hits


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Hybrid retrieval: vector first, keyword fallback."""
    docs = _vector_search(query, k)
    if not docs:
        docs = _keyword_search(query, k)
    return docs


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def support_answer(query: str) -> Dict[str, Any]:
    """Return answer + sources dict expected by the agent router."""
    docs = _retrieve(query)

    # If nothing found, craft graceful fallback
    if not docs:
        return {
            "answer": (
                "I'm sorry, I couldn't find an article covering that topic. "
                "Please reach out to our human support team."
            ),
            "tool": "support",
            "sources": [],
        }

    primary = docs[0]
    snippet = primary["content"].strip()
    title = primary["metadata"].get("title") or primary["metadata"].get("heading", "")

    # Compose concise answer
    answer_parts = []
    if title:
        answer_parts.append(title)
    answer_parts.append(snippet)
    answer = ": ".join(answer_parts)

    return {
        "answer": answer,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }
