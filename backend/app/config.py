"""
Runtime configuration module.

Reads environment variables (or sensible defaults) so settings are centralised.
"""
from __future__ import annotations

import os
from pathlib import Path


class Settings:
    """Minimal settings container (no external deps)."""

    # General --------------------------------------------------------
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # Database -------------------------------------------------------
    # Default to SQLite file in project root for easy local use.
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        f"sqlite:///{Path(__file__).resolve().parent.parent.parent / 'app.db'}",
    )

    # External APIs --------------------------------------------------
    FAKESTORE_API_URL: str = os.getenv("FAKESTORE_API_URL", "https://fakestoreapi.com")

    # Secrets --------------------------------------------------------
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")


settings = Settings()
