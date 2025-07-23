"""Flask blueprint exposing API routes (Phase-2)."""
from __future__ import annotations

from flask import Blueprint, jsonify, request
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.services.indexer import search_products

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    """Lightweight health endpoint for uptime checks."""
    return jsonify({"status": "ok"}), 200


@api_bp.route("/search", methods=["GET"])
def search() -> tuple[list[dict], int]:
    """Semantic product search."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify([]), 200

    db: Session = next(get_db())
    results = search_products(db, query)
    return jsonify([p.as_dict() for p in results]), 200
