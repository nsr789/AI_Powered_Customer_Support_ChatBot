"""Load Markdown docs into the support knowledge-base Chroma collection."""
from __future__ import annotations

import hashlib
import pathlib
from typing import List, Dict

import frontmatter
from app.core.llm import EmbeddingModel
from app.core.vector_store import get_collection

# --------------------------------------------------------------------------- #
_KB_PATH = pathlib.Path(__file__).parents[2] / "data" / "support_kb"
_EMBEDDER = EmbeddingModel()                     # deterministic stub


def _first_h1(markdown: str) -> str | None:
    """Return the first '# Heading' line without the leading `# `."""
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def _markdown_files() -> List[pathlib.Path]:
    return sorted(_KB_PATH.rglob("*.md"))


def main() -> None:
    """Idempotently ingest support articles into the `support_kb` collection."""
    collection = get_collection("support_kb")

    # ---- FIX: fetch existing IDs without passing an empty list --------------
    existing_ids: set[str] = set(collection.get(include=["ids"])["ids"])

    new_docs: List[str] = []
    new_ids:  List[str] = []
    new_meta: List[Dict] = []

    for md_file in _markdown_files():
        content = md_file.read_text(encoding="utf-8")
        fm = frontmatter.loads(content)
        body = fm.content.strip()
        meta = fm.metadata or {}

        if "title" not in meta:
            if (title := _first_h1(body)):
                meta["title"] = title

        doc_id = hashlib.sha256(md_file.as_posix().encode()).hexdigest()[:16]
        if doc_id in existing_ids:
            continue  # already stored

        new_docs.append(body)
        new_ids.append(doc_id)
        new_meta.append(meta)

    if new_docs:  # only embed & add if there is something new
        embeddings = [_EMBEDDER.embed(t) for t in new_docs]  # type: ignore[arg-type]
        collection.add(
            ids=new_ids,
            documents=new_docs,
            metadatas=new_meta,
            embeddings=embeddings,
        )

    print("✅  Support knowledge-base ingested.")
