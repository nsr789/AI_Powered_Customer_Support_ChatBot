"""
Simple RAG helper for support articles.

We keep a deterministic, offline-friendly embedder but add a lexical fallback
so that obviously-relevant docs (e.g. containing “return”) are always found.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from app.core.llm import EmbeddingModel
from app.core.vector_store import get_collection

# --------------------------------------------------------------------------- #
_EMBEDDER = EmbeddingModel()         # deterministic stub used in earlier phases
_COLLECTION_NAME = "support_kb"


def _euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _fetch_all() -> Dict[str, Any]:
    """Return *all* docs, embeddings & metadata from the collection."""
    col = get_collection(_COLLECTION_NAME)
    return col.get(include=["documents", "embeddings", "metadatas"])


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Vector-similarity search **with** lexical fallback.

    Returns a list of dicts with keys: document, metadata, score.
    """
    store = _fetch_all()

    if not store["documents"]:
        return []

    q_emb = _EMBEDDER.embed(query)

    # ---- vector similarity --------------------------------------------------
    scored = []
    for doc, emb, meta in zip(
        store["documents"], store["embeddings"], store["metadatas"]
    ):
        dist = _euclidean(q_emb, emb)
        scored.append({"document": doc, "metadata": meta, "score": dist})

    scored.sort(key=lambda d: d["score"])
    top_vec = scored[: k * 2]  # broaden a bit before lexical filter

    # ---- lexical filter fallback -------------------------------------------
    q_tokens = {tok.lower() for tok in query.split()}
    best = []
    for item in top_vec:
        doc_lc = item["document"].lower()
        if any(tok in doc_lc for tok in q_tokens):
            best.append(item)
        if len(best) >= k:
            break

    # If nothing matched lexically, just return the vector top-k
    return best or scored[:k]


def support_answer(query: str) -> Dict[str, Any]:
    """Return answer, tool tag and source list for the agent-router."""
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

    # Use first paragraph of the best match as the answer
    best_doc = docs[0]["document"].strip()
    first_para = best_doc.split("\n\n", 1)[0].replace("\n", " ").strip()

    return {
        "answer": first_para,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }
