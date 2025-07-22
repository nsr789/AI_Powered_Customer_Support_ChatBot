"""
Application factory and entry-point.

In Phase-1 we simply boot the Flask app, register API blueprint,
and ensure database tables are present.
"""
from __future__ import annotations

from flask import Flask

from app.api.routes import api_bp
from app.core.database import init_db


def create_app() -> Flask:
    """Flask application factory."""
    app = Flask(__name__)
    app.register_blueprint(api_bp, url_prefix="/api")

    # Lazily create tables at start-up
    with app.app_context():
        init_db()

    return app


# ------------------------------------------------------------------#
# Local dev entry-point
# ------------------------------------------------------------------#
if __name__ == "__main__":
    # Only for debugging; Render/Gunicorn will invoke `create_app`
    application = create_app()
    application.run(host="0.0.0.0", port=8000, debug=True)
