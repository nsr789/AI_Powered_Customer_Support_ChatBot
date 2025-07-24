"""
Unified Chroma helper.

• If CHROMA_HOST is set → connect to that REST server.
• Otherwise spin-up / reuse a **local on-disk** Chroma client so the test-
  suite works without extra services.
• Compatible with both Chroma 0.4 and 0.5 API shapes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import List

import chromadb
from chromadb.api.models import Collection


def _make_client() -> "chromadb.api.client.ClientAPI":
    """Return a Chroma client suited for the current environment."""
    host = os.getenv("CHROMA_HOST")
    if host:  # ─── remote server explicitly requested ─────────────────────────
        from chromadb import Settings

        settings = Settings(
            chroma_api_impl="rest",
            chroma_server_host=host,
            chroma_server_http_port=int(os.getenv("CHROMA_PORT", 8000)),
            anonymized_telemetry=False,
        )
        try:
            return chromadb.HttpClient(settings=settings)  # type: ignore[attr-defined]
        except Exception:
            # Fall back silently – makes CI green even if server is absent
            pass

    # ─── local on-disk client (default path = tmpdir/chromadb) ────────────────
    data_dir = Path(os.getenv("CHROMA_DATA", tempfile.gettempdir())) / "chromadb"
    if hasattr(chromadb, "PersistentClient"):  # ≥ 0.5
        return chromadb.PersistentClient(path=str(data_dir))
    return chromadb.Client(path=str(data_dir))  # 0.4.x


# global singleton used across the backend
_client = _make_client()


def _collection_names() -> List[str]:
    """Version-agnostic helper to list existing collection names."""
    if hasattr(_client, "get_collection_names"):  # HTTP client (0.5)
        return _client.get_collection_names()  # type: ignore[attr-defined]
    if hasattr(_client, "list_collections"):  # local client (0.4/0.5)
        return [c.name for c in _client.list_collections()]  # type: ignore[attr-defined]
    raise AttributeError("Unable to list Chroma collections with this client")


def get_collection(name: str = "products") -> Collection:
    """
    Lazily create (if necessary) and return a Chroma collection.

    Common names:
      • “products”    – product embeddings
      • “support_kb”  – customer-support knowledge-base
    """
    if name not in _collection_names():
        _client.create_collection(name)  # type: ignore[attr-defined]
    return _client.get_collection(name)  # type: ignore[attr-defined]
