"""
Support-RAG helper.

Provides retrieval-augmented generation for customer-support questions against
our small Markdown knowledge‑base ingested into the `support_kb` Chroma
collection.

Public API: ``answer(query: str) -> dict`` used by the agent router.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from langchain_community.vectorstores import Chroma

from app.core.vector_store import get_collection

__all__ = ["answer"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vector_top_k(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """Retrieve *k* most similar chunks via the embedding index."""
    col = get_collection("support_kb")
    result = col.query(query_texts=[query], n_results=k)
    return list(zip(result["documents"][0], result["metadatas"][0]))


def _match_tokens(text: str, tokens: set[str]) -> bool:
    lowered = text.lower()
    return any(tok in lowered for tok in tokens)


def _keyword_fallback(query: str) -> Tuple[str, dict] | None:
    """Scan every stored chunk and metadata for keyword overlap.

    The fallback ensures *some* answer is produced even when the vector search
    fails (e.g. because the embedding model isn't great for small datasets).
    """
    col = get_collection("support_kb")
    store = col.get(include=["documents", "metadatas"])

    tokens = set(re.findall(r"[a-zA-Z]+", query.lower()))

    for text, meta in zip(store["documents"], store["metadatas"]):
        meta_blob = " ".join(str(v) for v in (meta or {}).values())
        if _match_tokens(text, tokens) or _match_tokens(meta_blob, tokens):
            return text, meta
    return None


def _retrieve_docs(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """Hybrid retrieval: vector first, then keyword fallback."""
    pairs = [p for p in _vector_top_k(query, k) if p[0]]
    if not pairs:
        kb_hit = _keyword_fallback(query)
        if kb_hit:
            pairs.append(kb_hit)
    return pairs

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _compose_answer(doc_text: str, meta: dict) -> str:
    title = (meta or {}).get("title") or (meta or {}).get("heading")
    if title and title.lower() not in doc_text.lower():
        return f"{title}\n\n{doc_text}"
    return doc_text


def answer(query: str) -> Dict[str, Any]:
    """Return a JSON‑like dict expected by the agent graph."""
    docs = _retrieve_docs(query)
    if not docs:
        return {
            "tool": "support",
            "answer": (
                "I'm sorry, I couldn't find an article covering that topic. "
                "Please reach out to our human support team."
            ),
            "sources": [],
        }

    best_text, best_meta = docs[0]
    return {
        "tool": "support",
        "answer": _compose_answer(best_text, best_meta),
        "sources": [m or {} for _, m in docs],
    }
