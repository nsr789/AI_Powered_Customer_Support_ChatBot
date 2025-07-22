"""
Utilities to pull products from FakeStore API and persist them.

Designed for both CLI (scripts) and programmatic use. Network I/O is isolated
so tests can monkey-patch `requests.get`.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List

import requests
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.product import Product

log = logging.getLogger(__name__)
_FAKESTORE_PRODUCTS_URL = f"{settings.FAKESTORE_API_URL}/products"


# ------------------------------------------------------------------#
# Public helpers
# ------------------------------------------------------------------#
def fetch_products() -> List[dict[str, Any]]:
    """Hit FakeStore API and return raw JSON list."""
    resp = requests.get(_FAKESTORE_PRODUCTS_URL, timeout=10)
    resp.raise_for_status()
    return resp.json()  # type: ignore[return-value]


def save_products(session: Session, items: Iterable[dict[str, Any]]) -> int:
    """
    Persist products; upsert on primary-key conflict.

    Returns number of items committed.
    """
    added = 0
    for item in items:
        product = Product(
            id=item["id"],
            title=item["title"],
            description=item.get("description"),
            category=item.get("category"),
            price=item.get("price", 0),
            image=item.get("image"),
        )
        session.merge(product)  # Upsert
        added += 1
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise
    return added


# ------------------------------------------------------------------#
# CLI hook (optional)
# ------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    from app.core.database import SessionLocal

    with SessionLocal() as db:
        count = save_products(db, fetch_products())
        print(f"Inserted/updated {count} products.")
