"""
Indexing & retrieval utilities for Product documents.
"""
from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from app.core.vector_store import get_collection
from app.models.product import Product
from app.services.recommender import top_n  # fallback if no matches


# --------------------------------------------------------------------------- #
# Index builder
# --------------------------------------------------------------------------- #
def build_product_index(session: Session) -> int:
    """
    Upsert *all* products into Chroma collection.

    Returns number of items indexed.
    """
    products: List[Product] = session.query(Product).all()  # type: ignore[attr-defined]
    if not products:
        return 0

    col = get_collection()
    col.upsert(
        ids=[str(p.id) for p in products],
        documents=[p.description or p.title for p in products],
        metadatas=[{"title": p.title, "price": float(p.price)} for p in products],
    )
    return len(products)


# --------------------------------------------------------------------------- #
# Query helper
# --------------------------------------------------------------------------- #
def search_products(session: Session, query: str, k: int = 5) -> List[Product]:
    """
    Semantic search in vector DB; fallback to top-N if no results.
    """
    col = get_collection()
    res = col.query(query_texts=[query], n_results=k)
    ids = [int(i) for i in res["ids"][0]] if res["ids"] and res["ids"][0] else []

    if not ids:
        return top_n(session, limit=k)

    ordering = {pid: idx for idx, pid in enumerate(ids)}
    items = (
        session.query(Product)  # type: ignore[attr-defined]
        .filter(Product.id.in_(ids))  # type: ignore[attr-defined]
        .all()
    )
    # Preserve vector ranking order
    items.sort(key=lambda p: ordering[p.id])
    return items
