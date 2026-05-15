import sys
sys.path.append("..")

import os
import chromadb

from langchain_ollama import  ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from config import EMBEDDING_MODEL, PRIMARY_MODEL, OLLAMA_BASE_URL
from chromadb.api.types import EmbeddingFunction
import requests
DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")

class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        pass

    
    def __call__(self, input: list[str]) -> list[list[float]]:
        embeddings = []
        for text in input:
            resp = requests.post(
                f"{OLLAMA_BASE_URL}/api/embeddings",
                json={"model": EMBEDDING_MODEL, "prompt": text},
                timeout=30,
            )
            embeddings.append(resp.json()["embedding"])
        return embeddings

embedding_fn = OllamaEmbeddingFunction()




client = chromadb.EphemeralClient()
collection = client.get_or_create_collection(
    name="retrieval_quality",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}  # add this
)

def ingest_docs():
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200, separators=["\n\n", "\n", ". ", " "]
    )
    ids, documents, metadatas = [], [], []
    for filename in os.listdir(DOCS_DIR):
        if not filename.endswith(".md"):
            continue
        with open(os.path.join(DOCS_DIR, filename)) as f:
            content = f.read()
        header_chunks = header_splitter.split_text(content)
        final_chunks = char_splitter.split_documents(header_chunks)
        for i, chunk in enumerate(final_chunks):
            chunk_id = f"{filename}_{i}"
            ids.append(chunk_id)
            documents.append(chunk.page_content)
            metadatas.append({"source": filename, "chunk_index": i})
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(ids)} chunks from {DOCS_DIR}\n")

ingest_docs()

TEST_QUESTIONS = [
    {"query": "what is the total project budget?", "expected_source": None, "type": "factual"},
    {"query": "who is the project manager?", "expected_source": None, "type": "factual"},
    {"query": "when is the beta launch?", "expected_source": None, "type": "factual"},
    {"query": "what database was chosen and why?", "expected_source": None, "type": "factual"},
    {"query": "what is the deployment strategy?", "expected_source": None, "type": "factual"},
    {"query": "how much money is left and what percentage is that?", "expected_source": None, "type": "multi-hop"},
    {"query": "what framework is used for the API and what are the tradeoffs?", "expected_source": None, "type": "multi-hop"},
    {"query": "financial expenditure remaining reserves", "expected_source": None, "type": "vocab_mismatch"},
    {"query": "cost allocation workforce compensation", "expected_source": None, "type": "vocab_mismatch"},
    {"query": "infrastructure hosting deployment expenses", "expected_source": None, "type": "vocab_mismatch"},
]

llm = ChatOllama(model=PRIMARY_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

def naive_retrieve(query: str, n: int = 3) -> list[dict]:
    results = collection.query(
        query_texts=[query],
        n_results=n,
        include=["documents", "metadatas", "distances"]
    )
    return [
        {"doc": d, "source": m["source"], "distance": dist}
        for d, m, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )
    ]

def agentic_retrieve(query: str, max_hops: int = 3) -> tuple[list[dict], int]:
    all_results = []
    current_query = query
    hops = 0

    for hop in range(max_hops):
        hops += 1
        results = collection.query(
            query_texts=[current_query],
            n_results=3,
            include=["documents", "metadatas", "distances"]
        )
        hop_results = [
            {"doc": d, "source": m["source"], "distance": dist}
            for d, m, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            )
        ]
        all_results.extend(hop_results)

        top_distance = hop_results[0]["distance"] if hop_results else 999
        if top_distance < 0.5:
            break

        if hop < max_hops - 1:
            context = hop_results[0]["doc"][:300] if hop_results else ""
            refine_prompt = f"""
Original query: "{query}"
Top retrieved chunk: "{context}"
The retrieval quality was {'medium' if top_distance < 1.0 else 'low'} (distance={top_distance:.2f}).
Suggest a better search query using different terminology found in the retrieved text.
Return only the new query, nothing else.
"""
            response = llm.invoke([HumanMessage(content=refine_prompt)])
            current_query = response.content.strip().strip('"')

    seen = set()
    deduped = []
    for r in all_results:
        if r["doc"] not in seen:
            seen.add(r["doc"])
            deduped.append(r)

    return sorted(deduped, key=lambda x: x["distance"]), hops

print("=== Retrieval Quality Comparison ===\n")
naive_successes = 0
agentic_successes = 0

for case in TEST_QUESTIONS:
    query = case["query"]
    qtype = case["type"]

    naive_results = naive_retrieve(query)
    naive_top_dist = naive_results[0]["distance"] if naive_results else 999
    naive_success = naive_top_dist < 0.8

    agentic_results, hops_used = agentic_retrieve(query)
    agentic_top_dist = agentic_results[0]["distance"] if agentic_results else 999
    agentic_success = agentic_top_dist < 0.8

    if naive_success:
        naive_successes += 1
    if agentic_success:
        agentic_successes += 1

    naive_status = "✓" if naive_success else "✗"
    agentic_status = "✓" if agentic_success else "✗"

    print(f"[{qtype}] {query[:60]}")
    print(f"  Naive:    {naive_status} dist={naive_top_dist:.3f} | src={naive_results[0]['source'] if naive_results else 'none'}")
    print(f"  Agentic:  {agentic_status} dist={agentic_top_dist:.3f} | hops={hops_used} | src={agentic_results[0]['source'] if agentic_results else 'none'}")
    print()

total = len(TEST_QUESTIONS)
print(f"=== Summary ===")
print(f"Naive retrieval success rate:   {naive_successes}/{total} ({naive_successes/total:.0%})")
print(f"Agentic retrieval success rate: {agentic_successes}/{total} ({agentic_successes/total:.0%})")