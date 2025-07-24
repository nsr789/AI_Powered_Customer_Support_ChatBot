"""
Utilities to pull products from FakeStore API and persist them.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, List

import requests
from requests import HTTPError
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.product import Product

__all__ = ["fetch_products", "save_products"]          # ← EXPORTS

log = logging.getLogger(__name__)
_FAKESTORE_PRODUCTS_URL = f"{settings.FAKESTORE_API_URL}/products"


# ------------------------------------------------------------------#
# Public helpers
# ------------------------------------------------------------------#
def fetch_products() -> List[dict[str, Any]]:
    """Fetch FakeStore catalogue; fall back to 5 local samples when offline."""
    try:
        resp = requests.get(_FAKESTORE_PRODUCTS_URL, timeout=10)
        resp.raise_for_status()
        return resp.json()  # type: ignore[return-value]
    except (HTTPError, requests.RequestException) as err:
        log.warning("FakeStore API unavailable (%s). Using offline sample.", err)
        # ---------- expanded deterministic sample (5 items) ----------
        return [
            {
                "id": 1,
                "title": "Green Mug",
                "description": "Ceramic mug",
                "category": "home",
                "price": 5.0,
                "image": None,
            },
            {
                "id": 2,
                "title": "Basic White Tee",
                "description": "Cotton shirt",
                "category": "clothing",
                "price": 9.99,
                "image": None,
            },
            {
                "id": 3,
                "title": "Blue Jeans",
                "description": "Denim pants",
                "category": "clothing",
                "price": 19.99,
                "image": None,
            },
            {
                "id": 4,
                "title": "Running Shoes",
                "description": "Lightweight sneakers",
                "category": "footwear",
                "price": 49.99,
                "image": None,
            },
            {
                "id": 5,
                "title": "Leather Wallet",
                "description": "Genuine leather bifold",
                "category": "accessories",
                "price": 24.5,
                "image": None,
            },
        ]



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
        session.merge(product)
        added += 1
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise
    return added


# CLI hook remains unchanged
