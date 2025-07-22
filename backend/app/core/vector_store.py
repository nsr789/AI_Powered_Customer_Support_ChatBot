"""
Centralised Chroma client / collection helpers.

Collection name: `products`
Vector dim depends on embedding provider (dynamic).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List

import chromadb
from chromadb.utils.embedding_functions import EmbeddingFunction

from app.config import settings
from app.core.embeddings import get_default_provider

# --------------------------------------------------------------------------- #
# Client setup (persistent dir under project root by default)
# --------------------------------------------------------------------------- #
_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent.parent / ".chroma"),
)

_client = chromadb.PersistentClient(path=_PERSIST_DIR)


# --------------------------------------------------------------------------- #
# Embedding function adapter (Chroma expects __call__)
# --------------------------------------------------------------------------- #
class _Adapter(EmbeddingFunction):
    def __init__(self):
        self._provider = get_default_provider()

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self._provider.embed(texts)


_emb_fn = _Adapter()

_collection = _client.get_or_create_collection("products", embedding_function=_emb_fn)


def get_collection():
    """Return the singleton products collection."""
    return _collection
