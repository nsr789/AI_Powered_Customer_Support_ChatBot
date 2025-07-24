"""
LLM & embedding helpers shared across services.
"""
from __future__ import annotations
import os
import hashlib
import random
from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage
from langchain.embeddings import OpenAIEmbeddings


# ─────────────────── Chat-LLM interface ───────────────────────────────────────
class LLMInterface:
    def stream(self, messages: list[dict | BaseMessage]) -> str:
        raise NotImplementedError


class OpenAIProvider(LLMInterface):
    def __init__(self, model: str = "gpt-3.5-turbo-0125"):
        self._client = ChatOpenAI(model=model, temperature=0.2, streaming=True)

    def stream(self, messages):
        for chunk in self._client.stream(messages):
            yield chunk.content or ""


class FakeLLM(LLMInterface):
    """Minimal deterministic stub for offline / CI."""

    _TEMPLATES = {
        "search": "search",
        "recommend": "recommend",
        "support": "support",
    }

    def stream(self, messages):
        last = messages[-1]["content"].lower()
        if any(w in last for w in ("how", "return", "policy", "shipping", "order")):
            yield "support"
        elif any(w in last for w in ("recommend", "suggest")):
            yield "recommend"
        else:
            yield "search"


# ─────────────────── Embedding wrapper ────────────────────────────────────────
class EmbeddingModel:
    """
    Thin wrapper returning a 1536-dim vector like OpenAI.
    Fakes deterministic vectors when OpenAI key absent.
    """

    def __init__(self):
        self._use_openai = bool(os.getenv("OPENAI_API_KEY"))
        if self._use_openai:
            self._model = OpenAIEmbeddings(model="text-embedding-3-small")

    def embed(self, text: str) -> List[float]:
        if self._use_openai:
            return self._model.embed_query(text)
        # deterministic fake embedding (hash → float vector)
        h = hashlib.sha256(text.encode()).digest()
        random.seed(h)
        return [random.random() for _ in range(1536)]
