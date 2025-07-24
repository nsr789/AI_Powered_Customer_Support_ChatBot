import subprocess, sys, os, pathlib, json

def test_npm_lock_present():
    lock_file = pathlib.Path("frontend/package-lock.json")
    assert lock_file.exists(), "package-lock.json missing; run npm install"

    # quick sanity check: lockfile has react dep
    data = json.loads(lock_file.read_text())
    assert "react" in data["packages"][""]["dependencies"]
