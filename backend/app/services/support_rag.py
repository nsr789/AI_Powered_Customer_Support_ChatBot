"""Support-KB retrieve-and-answer utilities."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel

_embedder = EmbeddingModel()          # same stub embedder used at ingest


# --------------------------------------------------------------------------- #
#                           Retrieval helpers                                 #
# --------------------------------------------------------------------------- #
def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Return up to *k* support-KB chunks relevant to *query*.

    Strategy:
    1. Vector query (may be poor with stub embeddings).
    2. If that fails to yield anything, fall back to simple lexical matching.
    """
    collection = get_collection("support_kb")
    docs: List[Dict[str, Any]] = []

    # ---- 1) try vector search ------------------------------------------------
    try:
        q_vec = _embedder.embed(query)          # Stub returns List[float]
        res = collection.query(
            query_embeddings=[q_vec],           # batch size = 1
            n_results=k,
            include=["documents", "metadatas"],
        )
        for content, meta in zip(res.get("documents", [[]])[0],
                                 res.get("metadatas", [[]])[0]):
            docs.append({"page_content": content, "metadata": meta or {}})
    except Exception:
        pass                                    # ignore connection / other errors

    # ---- 2) lexical fall-back if nothing useful -----------------------------
    if not docs:
        tokens = set(re.findall(r"\w+", query.lower()))
        blob = collection.get(include=["documents", "metadatas"])
        for content, meta in zip(blob.get("documents", []),
                                 blob.get("metadatas", [])):
            if any(tok in content.lower() for tok in tokens):
                docs.append({"page_content": content, "metadata": meta or {}})
                if len(docs) >= k:
                    break

    return docs


def _select_answer(query: str, docs: List[Dict[str, Any]]) -> str:
    """Pick the best paragraph to return as the answer."""
    if not docs:
        return ""
    tokens = set(re.findall(r"\w+", query.lower()))
    for d in docs:
        if any(tok in d["page_content"].lower() for tok in tokens):
            return d["page_content"]
    return docs[0]["page_content"]             # fallback – first doc


# --------------------------------------------------------------------------- #
#                        Public entry-point                                   #
# --------------------------------------------------------------------------- #
def support_answer(query: str) -> Dict[str, Any]:
    """Return answer + source metadata bundle for the agent-router."""
    docs = _retrieve(query)

    answer_text = (
        _select_answer(query, docs)
        if docs
        else "I'm sorry, I couldn't find an article covering that topic. "
             "Please reach out to our human support team."
    )

    return {
        "answer": answer_text,
        "tool": "support",
        "sources": [d["metadata"] for d in docs],
    }


# backwards-compat for earlier imports
answer = support_answer
__all__ = ["support_answer", "answer"]
