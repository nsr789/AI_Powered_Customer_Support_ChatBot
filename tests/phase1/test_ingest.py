"""
Unit-test the FakeStore loader with mocked network.
"""
import json
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base
from app.services.data_loader import fetch_products, save_products

_SAMPLE_JSON = [
    {
        "id": 1,
        "title": "Test Tee",
        "description": "Soft cotton t-shirt",
        "category": "clothing",
        "price": 9.99,
        "image": "https://example.com/tee.png",
    },
    {
        "id": 2,
        "title": "Coffee Mug",
        "description": "Ceramic mug",
        "category": "home",
        "price": 4.5,
        "image": "https://example.com/mug.png",
    },
]


def _memory_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine)()


@patch("app.services.data_loader.requests.get")
def test_fetch_and_save(mock_get: Mock) -> None:
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = _SAMPLE_JSON

    session = _memory_session()
    # Fetch mocked data
    data = fetch_products()
    assert data == _SAMPLE_JSON

    # Persist
    inserted = save_products(session, data)
    assert inserted == len(_SAMPLE_JSON)

    # Verify DB content
    titles = [p.title for p in session.query(Base.classes.products)]  # type: ignore[attr-defined]
    assert titles == ["Test Tee", "Coffee Mug"]
