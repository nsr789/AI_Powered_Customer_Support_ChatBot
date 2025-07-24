"""
Support-RAG helper.

Answers FAQ-style questions by retrieving the most relevant snippets from the
Chroma collection ``support_kb`` that is built by `support_loader`.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from app.core.vector_store import get_collection

__all__ = ["answer"]


def _top_k(query: str, k: int = 3) -> List[Tuple[str, dict]]:
    """
    Retrieve *k* (text, metadata) pairs from the ``support_kb`` collection.

    The plain Chroma client is used instead of LangChain to avoid constructor
    incompatibilities across versions.
    """
    collection = get_collection("support_kb")  # chromadb.Collection

    # Chroma's `.query()` returns dicts of lists; unpack the first (only) query.
    result = collection.query(query_texts=[query], n_results=k)
    docs = result["documents"][0]
    metas = result["metadatas"][0]

    return list(zip(docs, metas))


def answer(query: str) -> Dict:
    """
    Build a response dict consumed by the LangGraph router.

    Keys required by the router / tests:
    - ``tool``    : literal string "support"
    - ``answer``  : best matching document text
    - ``sources`` : list of metadata dicts (may be empty if nothing found)
    """
    pairs = _top_k(query, k=3)

    if not pairs:
        return {
            "tool": "support",
            "answer": (
                "I'm sorry, I couldn't find an article covering that topic. "
                "Please reach out to our human support team."
            ),
            "sources": [],
        }

    # Use the highest-ranked document as the direct answer
    best_doc, _ = pairs[0]
    sources = [meta or {} for _, meta in pairs]

    return {
        "tool": "support",
        "answer": best_doc,
        "sources": sources,
    }
