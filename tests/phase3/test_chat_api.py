import json

from app.main import create_app
from app.core.database import SessionLocal
from app.services.data_loader import save_products, fetch_products

def _seed():
    with SessionLocal() as db:
        save_products(db, fetch_products())

def test_chat_search():
    _seed()
    app = create_app()
    client = app.test_client()
    resp = client.post("/api/chat", json={"query": "shirt"})
    chunks = b"".join(resp.response).decode()
    assert "results" in chunks
