"""
Embedding provider abstraction.

* `OpenAIEmbeddingProvider` – production, hits OpenAI API (text-embedding-3-small)
* `FakeEmbeddingProvider` – deterministic embeddings for unit tests
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from app.config import settings

# --------------------------------------------------------------------------- #
# Abstract base
# --------------------------------------------------------------------------- #


class EmbeddingProvider(ABC):
    """Strategy interface for embedding providers."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]: ...


# --------------------------------------------------------------------------- #
# OpenAI implementation
# --------------------------------------------------------------------------- #
class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Calls OpenAI embeddings endpoint."""

    def __init__(self) -> None:
        # Lazy import to avoid adding heavy dep when only running unit tests
        import openai  # type: ignore

        self._client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
        self._model = "text-embedding-3-small"

    def embed(self, texts: List[str]) -> List[List[float]]:
        res = self._client.embeddings.create(model=self._model, input=texts)
        return [record.embedding for record in res.data]  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- #
# Fake implementation (deterministic – good for tests)
# --------------------------------------------------------------------------- #
class FakeEmbeddingProvider(EmbeddingProvider):
    """Returns a deterministic 8-dim vector derived from text hash."""

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()  # 16 bytes
            vec = [b / 255.0 for b in h[:8]]  # 8-dim vector values ∈ [0,1]
            out.append(vec)
        return out


# --------------------------------------------------------------------------- #
# Provider factory util
# --------------------------------------------------------------------------- #
def get_default_provider() -> EmbeddingProvider:
    """Return OpenAI provider if key present, else Fake (offline)."""
    if settings.OPENAI_API_KEY:
        return OpenAIEmbeddingProvider()
    return FakeEmbeddingProvider()
