import sys
sys.path.append("..")

import chromadb

from chromadb.api.types import EmbeddingFunction
import requests

class OllamaEmbeddingFunction(EmbeddingFunction):
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
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL



client = chromadb.EphemeralClient()

collection = client.get_or_create_collection(
    name="experiment_02",
    embedding_function=embedding_fn
)

chunks = [
    {"id": "c01", "text": "Project Orion has a total approved budget of $180,000 for Phase 1.", "source": "budget.md", "topic": "budget"},
    {"id": "c02", "text": "Q3 expenditure reached $42,000, representing 23% of the total budget.", "source": "budget.md", "topic": "budget"},
    {"id": "c03", "text": "The remaining budget available for Q4 is $138,000.", "source": "budget.md", "topic": "budget"},
    {"id": "c04", "text": "Sarah Chen is the Project Manager for Project Orion.", "source": "team.md", "topic": "team"},
    {"id": "c05", "text": "Dave Okafor is the Lead Engineer and owns the CI/CD pipeline.", "source": "team.md", "topic": "team"},
    {"id": "c06", "text": "Priya Nair is the Senior Designer. She delivered wireframes on March 29.", "source": "team.md", "topic": "team"},
    {"id": "c07", "text": "The internal demo for Project Orion is scheduled for May 3, 2024.", "source": "schedule.md", "topic": "schedule"},
    {"id": "c08", "text": "Weekly standups occur every Monday at 9am PST.", "source": "schedule.md", "topic": "schedule"},
    {"id": "c09", "text": "The project launch target is Q3 2024, specifically September 30.", "source": "schedule.md", "topic": "schedule"},
    {"id": "c10", "text": "The backend API is 40% complete with 8 of 20 endpoints implemented.", "source": "status.md", "topic": "status"},
    {"id": "c11", "text": "The frontend component library is 25% complete as of April 8.", "source": "status.md", "topic": "status"},
    {"id": "c12", "text": "Project status is YELLOW due to the PaymentCo API blocker.", "source": "status.md", "topic": "status"},
    {"id": "c13", "text": "PaymentCo API sandbox access was requested March 20 and is still pending.", "source": "status.md", "topic": "blocker"},
    {"id": "c14", "text": "Jordan Kim joins May 1 as Backend Engineer specialising in payment integrations.", "source": "team.md", "topic": "team"},
    {"id": "c15", "text": "Alex Torres joins May 1 as Frontend Engineer and React specialist.", "source": "team.md", "topic": "team"},
    {"id": "c16", "text": "Personnel costs total $121,500 for Q1 including engineering, design, QA, and PM.", "source": "budget.md", "topic": "budget"},
    {"id": "c17", "text": "Infrastructure costs including AWS, SaaS tools, and PaymentCo licensing total $17,000.", "source": "budget.md", "topic": "budget"},
    {"id": "c18", "text": "The tech stack is Python backend, React frontend, and PostgreSQL database.", "source": "technical.md", "topic": "technical"},
    {"id": "c19", "text": "The database schema has 14 tables and 3 junction tables.", "source": "technical.md", "topic": "technical"},
    {"id": "c20", "text": "CI/CD pipeline was completed April 3 — 12 days ahead of schedule.", "source": "status.md", "topic": "status"},
]

collection.add(
    ids=[c["id"] for c in chunks],
    documents=[c["text"] for c in chunks],
    metadatas=[{"source": c["source"], "topic": c["topic"]} for c in chunks]
)

print(f"Collection size: {collection.count()} documents\n")

queries = [
    "what is the project budget?",
    "who is on the team?",
    "when is the demo?",
    "what is blocking the project?",
    "how much has been spent?",
    "what is the tech stack?",
    "who handles payments?",
    "what is the project status?",
    "when do the new engineers start?",
    "how complete is the backend?",
]

print("=== 10 Query Results ===")
for query in queries:
    results = collection.query(
        query_texts=[query],
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )
    print(f"\nQuery: '{query}'")
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        relevance = "HIGH" if dist < 0.5 else "MED" if dist < 1.0 else "LOW"
        print(f"  [{relevance} {dist:.3f}] ({meta['source']}) {doc[:80]}...")

# n_results comparison
print("\n=== n_results Comparison (query: 'budget') ===")
for n in [3, 5, 10]:
    results = collection.query(query_texts=["budget"], n_results=n, include=["distances"])
    distances = results["distances"][0]
    print(f"  n={n}: distances range {min(distances):.3f} — {max(distances):.3f}")

# metadata filtering
print("\n=== Metadata Filter Test ===")
unfiltered = collection.query(
    query_texts=["project timeline"],
    n_results=3,
    include=["documents", "metadatas"]
)
filtered = collection.query(
    query_texts=["project timeline"],
    n_results=3,
    where={"source": "schedule.md"},
    include=["documents", "metadatas"]
)
print("Without filter:")
for doc, meta in zip(unfiltered["documents"][0], unfiltered["metadatas"][0]):
    print(f"  ({meta['source']}) {doc[:70]}...")
print("With source_filter='schedule.md':")
for doc, meta in zip(filtered["documents"][0], filtered["metadatas"][0]):
    print(f"  ({meta['source']}) {doc[:70]}...")