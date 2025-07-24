"""
Light-weight RAG helper for the customer-support knowledge-base.
"""

from typing import Dict, List

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

from app.core.vector_store import get_collection

__all__ = ["answer"]


def _retrieve_docs(query: str, k: int = 3) -> List[Document]:
    """Return top-k relevant support articles as LangChain `Document`s."""
    vectordb = get_collection("support_kb")
    retriever = Chroma(collection=vectordb).as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)  # type: ignore[call-arg]


def answer(query: str) -> Dict:
    """
    Answer a support question via RAG.

    Returns
    -------
    dict
        {
            "tool":    "support",
            "answer":  <str>,
            "sources": [<metadata-dict>, ...]
        }
    """
    docs = _retrieve_docs(query)

    # Fallback message when nothing matches
    if not docs:
        return {
            "tool": "support",
            "answer": (
                "Sorry, I couldn't find an article that answers that. "
                "Please contact support for further assistance."
            ),
            "sources": [],
        }

    # Simple heuristic: first doc’s body becomes the answer.
    top_doc = docs[0]
    sources = [d.metadata or {} for d in docs]

    return {
        "tool": "support",
        "answer": top_doc.page_content,
        "sources": sources,
    }
