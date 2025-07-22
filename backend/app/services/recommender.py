"""
Tiny placeholder recommender.

Phase-1 simply returns the first N products, ordered by ID.
Later phases will replace this with hybrid vector / rule-based logic.
"""
from __future__ import annotations

from typing import List

from sqlalchemy.orm import Session

from app.models.product import Product


def top_n(session: Session, limit: int = 5) -> List[Product]:
    """Return first *limit* products."""
    return (
        session.query(Product)  # type: ignore[attr-defined]
        .order_by(Product.id.asc())  # type: ignore[has-type]
        .limit(limit)
        .all()
    )
