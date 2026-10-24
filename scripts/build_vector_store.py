import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.loader import load_chunks
from app.ingestion.vector_builder import VectorStore, build_vector_store

# 1. Build
docs = load_chunks("data/processed/chunks.json")
count = build_vector_store(docs)
print(f"✅ FAISS index built — {count} vectors saved to vectorstore/\n")

# 2. Smoke-test a few questions
store = VectorStore()
questions = [
    "Which service handles transcoding of uploaded audio?",
    "How are subscription payments handled?",
    "Where are playback events stored for analytics?",
]
for q in questions:
    print(f"Q: {q}")
    for r in store.search(q, k=3):
        print(f"   {r['score']:.3f}  {r['chunk_id']:<16} {r['metadata']['subsection']}")
    print()
