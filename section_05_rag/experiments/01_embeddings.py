import sys
sys.path.append("..")

import numpy as np
from langchain_ollama import OllamaEmbeddings
from config import EMBEDDING_MODEL, OLLAMA_BASE_URL

embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

test_sentences = [
    "cost reduction",
    "expense optimisation",
    "team meeting agenda",
    "Q3 budget summary",
    "third quarter financial summary",
    "employee onboarding process",
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox jumps over the lazy dog.",  # identical pair
    "project launch date confirmed for September",
    "PaymentCo API integration is blocked",
]

print("Generating embeddings...")
vectors = embeddings.embed_documents(test_sentences)

print(f"\nEmbedding dimensionality: {len(vectors[0])}")
print(f"Total sentences embedded: {len(vectors)}")


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


pairs = [
    (0, 1, "cost reduction vs expense optimisation — should be HIGH"),
    (0, 2, "cost reduction vs team meeting agenda — should be LOW"),
    (3, 4, "Q3 budget vs third quarter financial summary — should be HIGH"),
    (3, 5, "Q3 budget vs employee onboarding — should be LOW"),
    (6, 7, "identical sentences — should be ~1.0"),
    (8, 9, "project launch vs PaymentCo blocked — contextually related"),
]

print("\n=== Cosine Similarity Scores ===")
for i, j, label in pairs:
    score = cosine_similarity(vectors[i], vectors[j])
    print(f"  {score:.4f}  |  {label}")

# Adversarial pair — same keywords, different meaning
print("\n=== Adversarial Pair Test ===")
adversarial = [
    "The bank by the river flooded last night",
    "The bank approved my loan application today",
]
adv_vectors = embeddings.embed_documents(adversarial)
score = cosine_similarity(adv_vectors[0], adv_vectors[1])
print(f"  {score:.4f}  |  Same keyword 'bank', different meaning")
print(f"  {'High similarity — model struggles with polysemy' if score > 0.8 else 'Lower similarity — model handles context reasonably'}")

# embed_query vs embed_documents
print("\n=== embed_query vs embed_documents ===")
query_vec = embeddings.embed_query("what is the project budget?")
doc_vec = embeddings.embed_documents(["what is the project budget?"])[0]
score = cosine_similarity(query_vec, doc_vec)
print(f"  Similarity between embed_query and embed_documents on same text: {score:.4f}")
print(f"  {'Nearly identical — model does not differentiate' if score > 0.99 else 'Different — always use embed_query for queries'}")