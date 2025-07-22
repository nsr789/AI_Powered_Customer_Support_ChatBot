"""
Vector store indexing & query unit-tests using FakeEmbeddingProvider (default
because OPENAI_API_KEY is absent in CI).
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.services.indexer import build_product_index, search_products
from app.models.product import Product


def _memory_session():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


def _seed(session):
    session.add_all(
        [
            Product(
                id=1,
                title="Red Shirt",
                description="Bright red cotton shirt",
                category="clothing",
                price=10.0,
            ),
            Product(
                id=2,
                title="Blue Jeans",
                description="Classic denim jeans",
                category="clothing",
                price=20.0,
            ),
        ]
    )
    session.commit()


def test_index_and_query():
    session = _memory_session()
    _seed(session)

    indexed = build_product_index(session)
    assert indexed == 2

    res = search_products(session, "bright cotton", k=1)
    assert res[0].title == "Red Shirt"
