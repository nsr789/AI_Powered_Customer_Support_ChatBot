"""
Centralised Chroma helpers.

• Uses a remote HttpClient **only** when CHROMA_HOST is set.
• Falls back to a local in-process client so CI tests run without a server.
• Works with both Chroma 0.4 and 0.5 API shapes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.models import Collection


# ────────────────────── choose client implementation ─────────────────────────
def _make_client() -> "chromadb.api.client.ClientAPI":
    """
    Return a Chroma client instance.

    Priority:
      1. If CHROMA_HOST is defined → try remote HttpClient.
      2. Else use local PersistentClient / Client (version-dependent).
    """
    host = os.getenv("CHROMA_HOST")  # only use remote when explicitly set
    if host:
        from chromadb import Settings

        settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=int(os.getenv("CHROMA_PORT", 8000)),
            anonymized_telemetry=False,
        )
        try:
            return chromadb.HttpClient(settings=settings)  # type: ignore[attr-defined]
        except Exception:  # remote unavailable → fall back silently
            pass

    # Local fallback ──────────────────────────────────────────────────────────
    data_dir = Path(os.getenv("CHROMA_DATA", tempfile.gettempdir())) / "chromadb"
    if hasattr(chromadb, "PersistentClient"):
        return chromadb.PersistentClient(path=str(data_dir))  # ≥ 0.5
    return chromadb.Client(path=str(data_dir))  # 0.4.x


_client = _make_client()


# ─────────────────────── helper for API version drift ─────────────────────────
def _collection_names() -> List[str]:
    """Return existing collection names across Chroma versions."""
    if hasattr(_client, "get_collection_names"):  # 0.5 HTTP client
        return _client.get_collection_names()  # type: ignore[attr-defined]
    if hasattr(_client, "list_collections"):  # 0.4/0.5 local client
        return [c.name for c in _client.list_collections()]  # type: ignore[attr-defined]
    raise AttributeError("Unable to list Chroma collections with this client")


# ─────────────────────────── public convenience API ───────────────────────────
def get_collection(name: str = "products") -> Collection:
    """
    Lazily create (if missing) and return a Chroma collection.

    Typical names:
      • "products"   – product vectors
      • "support_kb" – customer-support knowledge-base
    """
    if name not in _collection_names():
        _client.create_collection(name)  # type: ignore[attr-defined]
    return _client.get_collection(name)  # type: ignore[attr-defined]
