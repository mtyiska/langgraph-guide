import os
import logging
from pathlib import Path
from typing import Optional
from config import KNOWLEDGE_BASE_DIR, EMBEDDING_MODEL, OLLAMA_BASE_URL, DATA_DIR

logger = logging.getLogger(__name__)


class VectorStoreWrapper:
    """ChromaDB + Ollama embeddings vector store."""

    def __init__(self, collection_name: str = "knowledge_base"):
        try:
            import chromadb
            from chromadb.config import Settings
            self._client = chromadb.PersistentClient(
                path=str(DATA_DIR / "chroma"),
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._available = True
        except ImportError:
            logger.warning("chromadb not installed — vector search disabled.")
            self._available = False

    def _embed(self, texts: list[str]) -> list[list[float]]:
        import requests
        results = []
        for text in texts:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=30,
            )
            results.append(resp.json()["embedding"])
        return results

    def index_directory(self, directory: str | Path = KNOWLEDGE_BASE_DIR):
        if not self._available:
            logger.warning("Vector store not available — skipping indexing.")
            return

        directory = Path(directory)
        files = [f for f in directory.rglob("*") if f.is_file() and f.suffix in {".md", ".txt"}]

        if not files:
            logger.info(f"No documents found in {directory}")
            return

        docs, metadatas, ids = [], [], []
        for f in files:
            try:
                text = f.read_text(encoding="utf-8")
                # Chunk at ~800 chars with 100-char overlap
                chunks = _chunk_text(text, chunk_size=800, overlap=100)
                for i, chunk in enumerate(chunks):
                    doc_id = f"{f.name}__chunk_{i}"
                    docs.append(chunk)
                    metadatas.append({"source": str(f), "filename": f.name, "chunk": i})
                    ids.append(doc_id)
            except Exception as e:
                logger.error(f"Failed to read {f}: {e}")

        if not docs:
            return

        embeddings = self._embed(docs)
        self._collection.upsert(
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info(f"Indexed {len(docs)} chunks from {len(files)} files.")

    def search(self, query: str, k: int = 5) -> list[tuple]:
        if not self._available:
            return []

        try:
            embedding = self._embed([query])[0]
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(k, self._collection.count() or 1),
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                score = 1 - dist
                output.append((_FakeDoc(doc, meta), score))
            return output
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

    def count(self) -> int:
        if not self._available:
            return 0
        return self._collection.count()


class _FakeDoc:
    """Minimal document object compatible with LangChain document interface."""
    def __init__(self, content: str, metadata: dict):
        self.page_content = content
        self.metadata     = metadata


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    chunks = []
    start  = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c for c in chunks if c.strip()]