"""
Project-wide pytest configuration.

Ensures the `backend` directory is on the import path so that the
`app` package (defined in backend/app) can be imported from any test
without extra boilerplate.

This keeps Phase-0/1 tests unchanged and makes future tests portable
in CI, local virtual-envs, and IDE test runners.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Project root = two levels up from this file
ROOT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = ROOT_DIR / "backend"

# Prepend once; if the path is already present do nothing
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
