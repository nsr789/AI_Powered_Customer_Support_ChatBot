"""
Database helpers: SQLAlchemy engine, session factory, and init_db utility.
"""
from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from app.config import settings

# ------------------------------------------------------------------#
# Engine & session factory
# ------------------------------------------------------------------#
DATABASE_URL = settings.DATABASE_URL

_ENGINE_KWARGS = {}
if DATABASE_URL.startswith("sqlite"):
    _ENGINE_KWARGS["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, **_ENGINE_KWARGS)  # type: ignore
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base for ORM models
Base = declarative_base()


def init_db() -> None:
    """
    Create all tables (no-op if they exist).

    Called once at app start-up; imports model modules to make sure they are
    registered with the SQLAlchemy metadata before `create_all`.
    """
    from app.models import product  # noqa: F401  (register model)

    Base.metadata.create_all(bind=engine)


# Dependency helper for future routes/services
def get_db() -> Generator:
    """Fast dependency for obtaining and closing a session (used later)."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
