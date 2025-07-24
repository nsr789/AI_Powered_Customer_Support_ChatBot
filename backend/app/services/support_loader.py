"""Load Markdown docs into the support-knowledge-base Chroma collection."""

from __future__ import annotations

import hashlib
import pathlib
import re
from typing import List

import frontmatter
from app.core.llm import EmbeddingModel
from app.core.vector_store import get_collection

_DOC_PATH = pathlib.Path(__file__).parents[2] / "data" / "support_kb"
_EMBED = EmbeddingModel()          # deterministic stub

# --------------------------------------------------------------------------- #
def _h1_title(markdown_body: str) -> str | None:
    """Return first H1 heading stripped of leading '# ' or None."""
    for line in markdown_body.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _files() -> List[pathlib.Path]:
    return sorted(f for f in _DOC_PATH.rglob("*.md") if f.is_file())


def main() -> None:
    """Idempotently (re)create the `support_kb` collection."""
    col = get_collection("support_kb")
    existing: set[str] = set(col.get(ids=[], include=["ids"])["ids"])

    texts: List[str] = []
    metas: List[dict] = []
    ids:   List[str] = []

    for f in _files():
        fm = frontmatter.loads(f.read_text(encoding="utf-8"))
        body: str = fm.content.strip()
        meta: dict = fm.metadata or {}

        # --- fill in missing title from first H1 ----------------------------
        if "title" not in meta:
            title = _h1_title(body)
            if title:
                meta["title"] = title

        doc_id = hashlib.sha256(f.as_posix().encode()).hexdigest()[:16]
        if doc_id in existing:
            continue  # already loaded

        texts.append(body)
        metas.append(meta)
        ids.append(doc_id)

    if texts:
        embeds = [_EMBED.embed(t) for t in texts]  # type: ignore[arg-type]
        col.add(ids=ids, embeddings=embeds, documents=texts, metadatas=metas)
    print("✅  Support knowledge-base ingested.")
