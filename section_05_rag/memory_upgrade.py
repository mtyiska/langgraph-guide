import sys
sys.path.append("..")

import os
import json
import sqlite3
from datetime import datetime
from uuid import uuid4
import chromadb
from chromadb.utils.embedding_functions import create_langchain_embedding
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
MEMORY_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "section_04_memory", "memory.db")


def get_memory_collection():
    embedding_fn = create_langchain_embedding(
        OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name="memories",
        embedding_function=embedding_fn
    )


def save_memory_with_embedding(
    user_id: str,
    session_id: str,
    content: str,
    category: str,
    importance: float = 0.5
):
    memory_id = f"{user_id}_{uuid4().hex[:8]}"

    # Save to SQLite
    if os.path.exists(MEMORY_DB_PATH):
        conn = sqlite3.connect(MEMORY_DB_PATH)
        conn.execute(
            """INSERT INTO memories (user_id, session_id, content, category, importance, created_at, last_accessed)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (user_id, session_id, content, category, importance,
             datetime.utcnow().isoformat(), datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()

    # Save to ChromaDB for semantic retrieval
    try:
        collection = get_memory_collection()
        collection.add(
            ids=[memory_id],
            documents=[content],
            metadatas={"user_id": user_id, "category": category, "importance": importance}
        )
    except Exception as e:
        print(f"[memory_upgrade] ChromaDB save failed: {e}")


def retrieve_relevant_memories(
    user_id: str,
    context: str,
    n: int = 5
) -> list[str]:
    try:
        collection = get_memory_collection()
        results = collection.query(
            query_texts=[context],
            n_results=n,
            where={"user_id": user_id},
            include=["documents", "distances"]
        )
        if not results["documents"][0]:
            return []
        return [
            doc for doc, dist in zip(results["documents"][0], results["distances"][0])
            if dist < 1.2
        ]
    except Exception as e:
        print(f"[memory_upgrade] Retrieval failed: {e}")
        return []


if __name__ == "__main__":
    print("Testing memory upgrade...")
    save_memory_with_embedding(
        "test_user", "test_session",
        "User prefers concise bullet-point responses.",
        "preference", 0.9
    )
    save_memory_with_embedding(
        "test_user", "test_session",
        "User is working on Project Atlas data platform.",
        "fact", 0.8
    )
    save_memory_with_embedding(
        "test_user", "test_session",
        "User's name is Jordan.",
        "preference", 0.95
    )

    results = retrieve_relevant_memories(
        "test_user",
        "Tell me about the Atlas project and how I like to receive information"
    )
    print(f"\nRetrieved {len(results)} semantically relevant memories:")
    for r in results:
        print(f"  - {r}")