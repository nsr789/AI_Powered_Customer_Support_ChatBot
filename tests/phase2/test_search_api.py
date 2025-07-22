"""
Flask endpoint test via test client.
"""
import json

import pytest
from flask import Flask

from app.main import create_app
from app.core.database import SessionLocal, Base
from app.models.product import Product
from app.services.indexer import build_product_index


@pytest.fixture()
def client():
    # Use application factory
    app = create_app()
    app.config.update(TESTING=True)
    with app.test_client() as client:
        yield client


def _seed_db():
    with SessionLocal() as db:
        Base.metadata.create_all(bind=db.get_bind())
        db.add(Product(id=1, title="Green Mug", description="Ceramic mug", price=5.0))
        db.commit()
        build_product_index(db)


def test_search_endpoint(client):
    _seed_db()
    resp = client.get("/api/search?q=ceramic")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data[0]["title"] == "Green Mug"
