"""Support-KB retrieve-and-answer helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel

_embedder = EmbeddingModel()  # deterministic stub

# --------------------------------------------------------------------------- #
_STOP: Set[str] = {
    "the", "is", "are", "a", "an", "and", "or", "of",
    "to", "in", "on", "for", "with", "what", "why", "how",
}


def _keyword_tokens(text: str) -> Set[str]:
    toks = {t for t in re.findall(r"\w+", text.lower()) if len(t) > 2}
    return toks - _STOP


# --------------------------------------------------------------------------- #
def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return up to *k* docs relevant to *query*."""
    col = get_collection("support_kb")

    # 1) vector search --------------------------------------------------------
    try:
        vec = _embedder.embed(query)
        res = col.query(
            query_embeddings=[vec],
            n_results=k,
            include=["documents", "metadatas"],
        )
        docs = [
            {"page_content": c, "metadata": m or {}}
            for c, m in zip(res["documents"][0], res["metadatas"][0])
        ]
        if docs:
            return docs
    except Exception:
        pass  # fall back to lexical

    # 2) lexical fallback -----------------------------------------------------
    kw = _keyword_tokens(query)
    if not kw:
        return []

    blob = col.get(include=["documents", "metadatas"])
    scored: list[tuple[int, Dict[str, Any]]] = []
    for content, meta in zip(blob["documents"], blob["metadatas"]):
        haystack = f"{meta.get('title','')} {content}".lower()
        hits = sum(1 for w in kw if w in haystack)
        if hits:
            scored.append((hits, {"page_content": content, "metadata": meta or {}}))

    scored.sort(key=lambda t: -t[0])
    return [d for _, d in scored[:k]]


def _pick_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Return the single best paragraph to show the user."""
    if not docs:
        return ""
    kw = _keyword_tokens(query)
    for d in docs:
        full_text = f"{d['metadata'].get('title','')}\n\n{d['page_content']}"
        if any(w in full_text.lower() for w in kw):
            return full_text
    # fallback to first
    first = docs[0]
    return f"{first['metadata'].get('title','')}\n\n{first['page_content']}"


def support_answer(query: str) -> Dict[str, Any]:
    """Return answer + sources dict required by the agent-router."""
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


# compatibility export
answer = support_answer
__all__ = ["support_answer", "answer"]
