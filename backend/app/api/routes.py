"""Flask blueprint exposing API routes (Phase-1: health-check only)."""
from __future__ import annotations

from flask import Blueprint, jsonify

api_bp = Blueprint("api", __name__)


@api_bp.route("/health", methods=["GET"])
def health() -> tuple[dict, int]:
    """Lightweight health endpoint for uptime checks."""
    return jsonify({"status": "ok"}), 200
