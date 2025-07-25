"""
Simple RAG helper for support articles.

Hybrid retrieval:
    • deterministic pseudo-embeddings  (keeps project offline-friendly)
    • lexical keyword fallback         (guarantees obvious hits)

Public API
----------
support_answer(query)  -> dict        (preferred name)
answer(query)          -> dict        (back-compat alias)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from app.core.llm import EmbeddingModel
from app.core.vector_store import get_collection

# --------------------------------------------------------------------------- #
__all__ = ["support_answer", "answer"]  # <- both symbols are now public

_EMBEDDER = EmbeddingModel()
_COLLECTION_NAME = "support_kb"


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _fetch_all() -> Dict[str, Any]:
    col = get_collection(_COLLECTION_NAME)
    return col.get(include=["documents", "embeddings", "metadatas"])


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    store = _fetch_all()
    if not store["documents"]:
        return []

    q_emb = _EMBEDDER.embed(query)
    scored = [
        {
            "document": d,
            "metadata": m,
            "score": _euclidean(q_emb, e),
        }
        for d, e, m in zip(
            store["documents"], store["embeddings"], store["metadatas"]
        )
    ]
    scored.sort(key=lambda x: x["score"])
    candidates = scored[: k * 2]

    q_tokens = {tok.lower() for tok in query.split()}
    best: List[Dict[str, Any]] = [
        item for item in candidates if any(t in item["document"].lower() for t in q_tokens)
    ][:k]

    return best or scored[:k]


def support_answer(query: str) -> Dict[str, Any]:
    docs = _retrieve(query)
    if not docs:
        return {
            "answer": (
                "I'm sorry, I couldn't find an article covering that topic. "
                "Please reach out to our human support team."
            ),
            "tool": "support",
            "sources": [],
        }

    best_doc = docs[0]["document"].strip()
    first_para = best_doc.split("\n\n", 1)[0].replace("\n", " ").strip()

    return {
        "answer": first_para,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }


# --------------------------------------------------------------------------- #
# Back-compat export expected by earlier phases/tests
answer = support_answer
