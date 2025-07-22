
"""Phase 0 structural smoke test: verifies baseline skeleton exists."""
import os
import pytest

REQUIRED_PATHS = [
    ".github/workflows/ci.yml",
    "backend/app/main.py",
    "backend/Dockerfile",
    "tests/phase0/test_structure.py",
    "docker-compose.yml",
    ".env.example",
]

@pytest.mark.parametrize("path", REQUIRED_PATHS)
def test_path_exists(path: str):
    assert os.path.exists(path), f"Missing required path: {path}"
