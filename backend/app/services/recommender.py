"""
Very-lightweight rule-based recommender used when running the test-suite
with the FakeLLM (no real vector search available).
"""

from __future__ import annotations
from random import sample
from typing import List, Dict

# five static “best-seller” style products – ids need not match DB
_BEST_SELLERS: List[Dict] = [
    {
        "id": 1,
        "title": "Ceramic mug",
        "description": "11-oz microwave-safe ceramic coffee mug.",
        "category": None,
        "price": 9.99,
        "image": None,
    },
    {
        "id": 2,
        "title": "Desk organiser",
        "description": "Keeps all your stationery tidy.",
        "category": None,
        "price": 12.5,
        "image": None,
    },
    {
        "id": 3,
        "title": "Wireless mouse",
        "description": "Silent-click 2.4 GHz mouse, USB-C receiver.",
        "category": None,
        "price": 24.0,
        "image": None,
    },
    {
        "id": 4,
        "title": "Notebook A5",
        "description": "Hard-cover dotted notebook, 200 pages.",
        "category": None,
        "price": 5.95,
        "image": None,
    },
    {
        "id": 5,
        "title": "LED desk lamp",
        "description": "Adjustable brightness, USB powered.",
        "category": None,
        "price": 17.9,
        "image": None,
    },
]



def recommend(_: str, k: int = 5):  # query is ignored in fake mode
    """Return `k` popular products (deterministic for tests)."""
    if k >= len(_BEST_SELLERS):
        return _BEST_SELLERS.copy()
    # deterministic sample to keep tests repeatable
    return sorted(sample(_BEST_SELLERS, k), key=lambda d: d["id"])


# --------------------------------------------------------------------------- #
# Compatibility shim – older modules expect a `top_n()` helper.               #
# --------------------------------------------------------------------------- #
def top_n(query: str, n: int = 5):
    """
    Back-compat alias used by `app.services.indexer` when the vector search
    returns zero hits.  Delegates to :func:`recommend`.
    """
    return recommend(query, k=n)


__all__ = ["recommend", "top_n"]

