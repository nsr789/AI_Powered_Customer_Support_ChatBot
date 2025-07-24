"""Retrieve-and-generate answers from the support knowledge-base."""

from __future__ import annotations

from typing import Any, Dict, List

from app.core.vector_store import get_collection


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Return top-k document chunks relevant to *query*.

    We query the underlying Chroma collection directly; no LangChain wrapper
    is required (and the wrapper’s API changed in recent versions).
    """
    collection = get_collection("support_kb")  # chromadb.api.models.Collection

    try:
        result = collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"],
        )
    except Exception:
        return []

    docs: List[Dict[str, Any]] = []
    # `query()` returns lists-of-lists keyed by 'documents' / 'metadatas'
    for content, meta in zip(result.get("documents", [[]])[0],
                             result.get("metadatas", [[]])[0]):
        docs.append({"page_content": content, "metadata": meta or {}})

    return docs


def support_answer(query: str) -> Dict[str, Any]:
    """Return answer + source metadata dict consumed by the agent router."""
    docs = _retrieve(query)

    if docs:
        answer_text = docs[0]["page_content"]
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


# --------------------------------------------------------------------------- #
# Legacy alias expected by older imports / tests.
answer = support_answer
__all__ = ["support_answer", "answer"]
