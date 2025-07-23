"""
Product ORM entity mapped to the `products` table.
"""
from __future__ import annotations

from decimal import Decimal

from sqlalchemy import Integer, Numeric, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class Product(Base):  # type: ignore[call-arg]
    __tablename__ = "products"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    image: Mapped[str | None] = mapped_column(String(255), nullable=True)

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
