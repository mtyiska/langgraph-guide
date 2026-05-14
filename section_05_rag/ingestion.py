import sys
sys.path.append("..")

import os
import hashlib
import chromadb
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from rich.progress import track
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")


def get_embedding_fn():
    return create_langchain_embedding(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    )


def get_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)


def get_collection(client=None, embedding_fn=None):
    if client is None:
        client = get_client()
    if embedding_fn is None:
        embedding_fn = get_embedding_fn()
    return client.get_or_create_collection(
        name="documents",
        embedding_function=embedding_fn
    )


def chunk_id(source: str, content: str) -> str:
    h = hashlib.md5(f"{source}:{content}".encode()).hexdigest()[:12]
    return f"{source}_{h}"


def ingest_documents(docs_dir: str = DOCS_DIR, verbose: bool = True) -> int:
    client = get_client()
    embedding_fn = get_embedding_fn()
    collection = get_collection(client, embedding_fn)

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )

    md_files = [f for f in os.listdir(docs_dir) if f.endswith(".md")]
    if not md_files:
        print(f"No markdown files found in {docs_dir}")
        return 0

    existing_ids = set(collection.get()["ids"])

    ids_to_add, docs_to_add, metas_to_add = [], [], []
    skipped = 0

    iterator = track(md_files, description="Processing documents...") if verbose else md_files

    for filename in iterator:
        filepath = os.path.join(docs_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        header_chunks = header_splitter.split_text(content)
        final_chunks = char_splitter.split_documents(header_chunks)

        for i, chunk in enumerate(final_chunks):
            cid = chunk_id(filename, chunk.page_content)

            if cid in existing_ids:
                skipped += 1
                continue

            header_path = " > ".join(filter(None, [
                chunk.metadata.get("h1"),
                chunk.metadata.get("h2"),
                chunk.metadata.get("h3"),
            ]))

            ids_to_add.append(cid)
            docs_to_add.append(chunk.page_content)
            metas_to_add.append({
                "source": filename,
                "chunk_index": i,
                "header_path": header_path or "root",
                "char_count": len(chunk.page_content),
            })

    if ids_to_add:
        collection.add(ids=ids_to_add, documents=docs_to_add, metadatas=metas_to_add)

    if verbose:
        print(f"\nIngestion complete:")
        print(f"  Added:   {len(ids_to_add)} chunks")
        print(f"  Skipped: {skipped} (already in collection)")
        print(f"  Total:   {collection.count()} chunks in collection")

    return len(ids_to_add)


def get_all_sources(collection) -> list[str]:
    results = collection.get(include=["metadatas"])
    sources = list({m["source"] for m in results["metadatas"]})
    return sorted(sources)


def get_source_chunk_counts(collection) -> dict[str, int]:
    results = collection.get(include=["metadatas"])
    counts = {}
    for m in results["metadatas"]:
        src = m["source"]
        counts[src] = counts.get(src, 0) + 1
    return counts


if __name__ == "__main__":
    ingest_documents()