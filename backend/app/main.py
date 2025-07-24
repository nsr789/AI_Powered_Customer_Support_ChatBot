"""
Application factory and entry-point.

Boots Flask, registers blueprints, creates DB tables.
"""
from flask import Flask, jsonify
from flask_cors import CORS           # ← NEW

from app.api.routes import api_bp
from app.api.chat_routes import chat_bp
from app.core.database import init_db


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app)                         # ← enables “*” CORS on all routes

    @app.route("/", methods=["GET"])
    def root():
        return jsonify(
            {
                "message": "AI-Support Platform backend is running.",
                "available_endpoints": ["/api/health", "/api/search?q=<query>", "/api/chat"],
            }
        )

    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(chat_bp, url_prefix="/api")

    with app.app_context():
        init_db()

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8000, debug=True)
