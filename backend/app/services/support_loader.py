"""
Ingest markdown or plaintext FAQ / policy docs into Chroma ("support_kb").
Run once during deployment:

    python -m app.services.support_loader
"""
from pathlib import Path
import frontmatter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel  # thin wrapper used elsewhere

_KB_DIR = Path(__file__).parents[2] / "docs" / "faq"


def _iter_docs():
    """Yield dicts with id, content, metadata for every *.md file."""
    for path in _KB_DIR.glob("*.md"):
        post = frontmatter.load(path)
        yield {
            "id": str(path),
            "content": post.content,
            "metadata": {"title": post.get("title", path.stem)},
        }


def main() -> None:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    collection = get_collection("support_kb")

    for doc in _iter_docs():
        for chunk in splitter.split_text(doc["content"]):
            collection.add(
                ids=[f"{doc['id']}#{hash(chunk)}"],
                documents=[chunk],
                metadatas=[doc["metadata"]],
                embeddings=[EmbeddingModel().embed(chunk)],
            )
    print("✅  Support knowledge-base ingested.")


if __name__ == "__main__":
    main()
