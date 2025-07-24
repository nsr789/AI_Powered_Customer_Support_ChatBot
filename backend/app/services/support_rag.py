"""Support-KB retrieve-and-answer utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel  # FAKE/Open-AI wrapper used everywhere

# One global embedder instance (same object used by the ingester)
_embedder = EmbeddingModel()


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return *k* most relevant chunks from the support KB."""
    collection = get_collection("support_kb")

    # Embed the query **locally** to stay offline-friendly
    q_emb: List[float] = _embedder.embed([query])[0]  # type: ignore[index]

    try:
        res = collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    docs: List[Dict[str, Any]] = []
    for content, meta in zip(res.get("documents", [[]])[0],
                             res.get("metadatas", [[]])[0]):
        docs.append({"page_content": content, "metadata": meta or {}})
    return docs


def _select_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Pick the first chunk that contains any keyword from *query*."""
    tokens = set(re.findall(r"\w+", query.lower()))
    for d in docs:
        if any(tok in d["page_content"].lower() for tok in tokens):
            return d["page_content"]
    # fallback - should rarely occur if we have docs
    return docs[0]["page_content"] if docs else ""


def support_answer(query: str) -> Dict[str, Any]:
    """
    Answer a support question using the internal knowledge-base.

    Output format required by the agent router:
        {"answer": str, "tool": "support", "sources": [metadata, ...]}
    """
    docs = _retrieve(query)
    if docs:
        answer_text = _select_answer(query, docs)
    else:
        answer_text = (
            "I'm sorry, I couldn't find an article covering that topic. "
            "Please reach out to our human support team."
        )

    return {"answer": answer_text, "tool": "support",
            "sources": [d["metadata"] for d in docs]}


# --- Backwards-compat import used by agent_router.py -------------------------
answer = support_answer
__all__ = ["support_answer", "answer"]
