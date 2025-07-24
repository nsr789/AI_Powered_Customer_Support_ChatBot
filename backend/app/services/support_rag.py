"""Retrieve-and-generate answers from the support knowledge-base."""
from __future__ import annotations

from typing import Any, Dict, List

from langchain_community.vectorstores import Chroma
from app.core.vector_store import get_collection


def _retrieve(query: str, k: int = 3) -> List[Dict[str, Any]]:
    """Return top-k documents that match the query."""
    vectordb = get_collection("support_kb")          # chromadb Collection
    retriever = Chroma(collection=vectordb).as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)


def support_answer(query: str) -> Dict[str, Any]:
    """Return answer dict consumed by the agent router."""
    docs = _retrieve(query)

    if docs:
        # Use first document’s text as the answer.
        answer_text = docs[0].page_content
    else:
        # Graceful fallback when nothing matches.
        answer_text = (
            "I'm sorry, I couldn't find an article covering that topic. "
            "Please reach out to our human support team."
        )

    return {
        "answer": answer_text,
        "tool": "support",
        "sources": [d.metadata for d in docs],
    }


# --------------------------------------------------------------------------- #
# Back-compat export expected by older imports/tests
answer = support_answer  # noqa: E305
__all__ = ["support_answer", "answer"]
