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
from chromadb.config import Settings
from chromadb.utils.embedding_functions import EmbeddingFunction

from app.core.embeddings import get_default_provider

# --------------------------------------------------------------------------- #
# Client setup (embedded persistent DB – no separate server required)
# --------------------------------------------------------------------------- #
_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent.parent / ".chroma"),
)

_TELEMETRY = os.getenv("ANONYMIZED_TELEMETRY", "false").lower() == "true"

_settings = Settings(
    anonymized_telemetry=_TELEMETRY,
    is_persistent=True,
    persist_directory=_PERSIST_DIR,
)

# Embedded client; runs in-process
_client = chromadb.PersistentClient(path=_PERSIST_DIR, settings=_settings)

# --------------------------------------------------------------------------- #
# Embedding function adapter
# --------------------------------------------------------------------------- #
class _Adapter(EmbeddingFunction):
    """Wrap our embedding provider so Chroma can call it."""

    def __init__(self):
        self._provider = get_default_provider()

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self._provider.embed(texts)


_emb_fn = _Adapter()
_collection = _client.get_or_create_collection(
    name="products",
    embedding_function=_emb_fn,
)


def get_collection():
    """Return the singleton products collection."""
    return _collection
