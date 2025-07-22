"""
Schema smoke-test: ensure Product table is created correctly.
"""
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.models.product import Product  # noqa: F401  (ensure model imported)


def test_product_table_exists() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)

    insp = inspect(engine)
    assert "products" in insp.get_table_names()
