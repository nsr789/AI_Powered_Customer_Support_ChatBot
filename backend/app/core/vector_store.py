"""
Centralised Chroma client / collection helpers.
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
# Defensive patch: if an old stub of `posthog` is imported, make capture a no-op
# --------------------------------------------------------------------------- #
def _silence_broken_posthog() -> None:
    try:
        import posthog  # type: ignore
        if not hasattr(posthog.capture, "__code__") or posthog.capture.__code__.co_argcount == 1:
            posthog.capture = lambda *_, **__: None  # type: ignore[assignment]
    except ImportError:
        pass


_silence_broken_posthog()

# --------------------------------------------------------------------------- #
# Client setup (embedded persistent DB – no external server)
# --------------------------------------------------------------------------- #
_PERSIST_DIR = os.getenv(
    "CHROMA_PERSIST_DIR",
    str(Path(__file__).resolve().parent.parent.parent / ".chroma"),
)

_SETTINGS = Settings(
    anonymized_telemetry=False,
    is_persistent=True,
    persist_directory=_PERSIST_DIR,
)

_client = chromadb.PersistentClient(path=_PERSIST_DIR, settings=_SETTINGS)

# --------------------------------------------------------------------------- #
# Embedding adapter
# --------------------------------------------------------------------------- #
class _Adapter(EmbeddingFunction):
    def __init__(self) -> None:
        self._provider = get_default_provider()

    def __call__(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self._provider.embed(texts)


_collection = _client.get_or_create_collection(
    name="products",
    embedding_function=_Adapter(),
)


def get_collection():
    """Return the singleton products collection."""
    return _collection
