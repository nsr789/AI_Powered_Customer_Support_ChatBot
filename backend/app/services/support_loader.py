"""
Ingest markdown/txt FAQs into the vector store 'support_kb'.
Run from CLI:  python -m app.services.support_loader
"""
from pathlib import Path
import frontmatter
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.vector_store import get_collection
from app.core.llm import EmbeddingModel  # wrapper around OpenAIEmbedding / Fake

_KB_DIR = Path(__file__).parent.parent.parent / "docs" / "faq"


def _iter_docs():
    for p in _KB_DIR.glob("*.md"):
        post = frontmatter.load(p)
        yield {"id": str(p), "content": post.content, "metadata": {"title": post.get('title', p.stem)}}


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
    print("Support KB ingested.")


if __name__ == "__main__":
    main()
