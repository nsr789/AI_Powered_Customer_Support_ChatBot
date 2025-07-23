"""
Application factory and entry-point.

Boots Flask, registers blueprints, creates DB tables.
"""
from __future__ import annotations

from flask import Flask, jsonify

from app.api.routes import api_bp
from app.core.database import init_db


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)

    # ---- Root route ---------------------------------------------------------
    @app.route("/", methods=["GET"])
    def root():
        """Simple landing route for '/'."""
        return jsonify(
            {
                "message": "AI-Support Platform backend is running.",
                "available_endpoints": [
                    "/api/health",
                    "/api/search?q=<query>",
                ],
            }
        )

    # ---- Blueprints ---------------------------------------------------------
    app.register_blueprint(api_bp, url_prefix="/api")

    # ---- DB initialisation --------------------------------------------------
    with app.app_context():
        init_db()

    return app


# Local dev entry-point -------------------------------------------------------
if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=8000, debug=True)
