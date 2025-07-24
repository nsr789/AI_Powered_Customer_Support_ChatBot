"""Retrieve-and-answer helper for the support knowledge-base."""

from __future__ import annotations

from typing import Any, Dict, List
import re

from app.core.vector_store import get_collection


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k chunks from the support KB."""
    collection = get_collection("support_kb")
    try:
        res = collection.query(
            query_texts=[query],
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
    """
    Choose the most relevant doc chunk.

    Preference order:
    1. First chunk containing *any* keyword from the query.
    2. Fallback to docs[0] if no keyword hit.
    """
    tokens = set(re.findall(r"\w+", query.lower()))
    for d in docs:
        if any(tok in d["page_content"].lower() for tok in tokens):
            return d["page_content"]
    return docs[0]["page_content"] if docs else ""


def support_answer(query: str) -> Dict[str, Any]:
    """
    Generate an answer for *query* from the support KB.

    Returns the dict expected by the agent router:
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

    return {
        "answer": answer_text,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }


# Back-compat for earlier imports
answer = support_answer
__all__ = ["support_answer", "answer"]
