"""Support-KB retrieve-and-answer helpers."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Set

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel

_embedder = EmbeddingModel()  # deterministic stub

# --------------------------------------------------------------------------- #
#  tiny English stop-list – enough for our test corpus                        #
# --------------------------------------------------------------------------- #
_STOP: Set[str] = {
    "the", "is", "are", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "what", "why", "how",
}


def _keyword_tokens(text: str) -> Set[str]:
    """Return meaningful lowercase tokens from *text*."""
    tokens = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
    return tokens - _STOP


# --------------------------------------------------------------------------- #
#                               Retrieval                                     #
# --------------------------------------------------------------------------- #
def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return up to *k* docs relevant to *query* (vector → lexical fallback)."""
    col = get_collection("support_kb")
    docs: List[Dict[str, Any]] = []

    # 1) vector search (may be poor with stub embeddings)
    try:
        vec = _embedder.embed(query)
        res = col.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "metadatas"],
        )
        for c, m in zip(res.get("documents", [[]])[0],
                        res.get("metadatas", [[]])[0]):
            docs.append({"page_content": c, "metadata": m or {}})
    except Exception:
        pass

    if docs:  # good enough
        return docs[:k]

    # 2) lexical scoring ------------------------------------------------------
    keywords = _keyword_tokens(query)
    if not keywords:
        return []

    all_blob = col.get(include=["documents", "metadatas"])
    scored: list[tuple[int, Dict[str, Any]]] = []
    for content, meta in zip(all_blob.get("documents", []),
                             all_blob.get("metadatas", [])):
        text_low = content.lower()
        hit_cnt = sum(1 for kw in keywords if kw in text_low)
        if hit_cnt:
            scored.append((hit_cnt, {"page_content": content, "metadata": meta or {}}))

    # sort by hit-count ↓, preserve original order for ties
    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:k]]


def _pick_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Choose the best paragraph from *docs* for the user."""
    if not docs:
        return ""
    kw = _keyword_tokens(query)
    for d in docs:
        if any(w in d["page_content"].lower() for w in kw):
            return d["page_content"]
    return docs[0]["page_content"]


# --------------------------------------------------------------------------- #
def support_answer(query: str) -> Dict[str, Any]:
    """Return answer + sources dict required by agent-router."""
    docs = _retrieve(query)
    answer_txt = (
        _pick_answer(query, docs)
        if docs
        else "I'm sorry, I couldn't find an article covering that topic. "
             "Please reach out to our human support team."
    )
    return {
        "answer": answer_txt,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }


# backward-compatibility alias
answer = support_answer
__all__ = ["support_answer", "answer"]
