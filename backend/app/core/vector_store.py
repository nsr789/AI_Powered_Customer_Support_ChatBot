"""
Centralised Chroma client helpers.

Handles both HTTP-client (remote server) and in-process client variants.
Compatible with Chroma ≥ 0.4 – some method names differ.
"""
from __future__ import annotations
import os
import chromadb
from chromadb.api.models import Collection

_settings = chromadb.Settings(
    chroma_api_impl="rest",
    chroma_server_host=os.getenv("CHROMA_HOST", "localhost"),
    chroma_server_http_port=int(os.getenv("CHROMA_PORT", 8000)),
    anonymized_telemetry=False,
)
_client = chromadb.HttpClient(settings=_settings)


def _collection_names() -> list[str]:
    """
    Return list of existing collection names, handling API differences
    between chromadb client classes.
    """
    # Newer API (0.5.x http client)
    if hasattr(_client, "get_collection_names"):
        return _client.get_collection_names()
    # Fallback (in-process client 0.4 / 0.5)
    if hasattr(_client, "list_collections"):
        return [c.name for c in _client.list_collections()]
    raise AttributeError("Unable to list Chroma collections with this client.")


def get_collection(name: str = "products") -> Collection:
    """
    Lazily create (if missing) and return a Chroma collection.

    Names used:
      • "products"   – product vectors
      • "support_kb" – customer-support knowledge-base
    """
    if name not in _collection_names():
        _client.create_collection(name)
    return _client.get_collection(name)
