"""
Product ORM entity mapped to the `products` table.

Represents the catalogue items we ingest from FakeStore.
"""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy import Column, Integer, Numeric, String, Text

from app.core.database import Base


class Product(Base):  # type: ignore
    __tablename__ = "products"

    id: int | Column[int] = Column(Integer, primary_key=True, index=True)
    title: str | Column[str] = Column(String(255), nullable=False)
    description: str | Column[str | None] = Column(Text, nullable=True)
    category: str | Column[str | None] = Column(String(100), nullable=True)
    price: Decimal | Column[Decimal] = Column(Numeric(10, 2), nullable=False)
    image: str | Column[str | None] = Column(String(255), nullable=True)

    # ------------------------------------------------------------------#
    # Utility helpers
    # ------------------------------------------------------------------#
    def as_dict(self) -> dict:
        """Serialize to a basic dict (for JSON responses)."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "price": float(self.price),
            "image": self.image,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Product id={self.id} title={self.title!r}>"
