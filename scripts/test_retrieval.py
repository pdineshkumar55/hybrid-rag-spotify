"""Try both retrievers side by side.

    python scripts/test_retrieval.py
    python scripts/test_retrieval.py "Which services use PostgreSQL tables?"
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.graph_retriever import graph_search
from app.retrieval.vector_retriever import vector_search

questions = sys.argv[1:] or [
    "What happens after an artist uploads a track?",
    "Which services and tables implement the Playback & Player features?",
    "How does the app handle subscription payments?",
]

for q in questions:
    print("=" * 90)
    print(f"Q: {q}\n")

    vec = vector_search(q, k=4)
    print("VECTOR (FAISS):")
    for r in vec:
        print(f"   {r['score']:.3f}  {r['chunk_id']:<16} {r['metadata']['subsection']}")

    graph = graph_search(q, chunk_ids=[r["chunk_id"] for r in vec])
    print("\nGRAPH seeds (entities found):")
    for e in graph["entities"][:8]:
        print(f"   {e['score']:.2f}  {e['label']:<12} {e['name']}")
    print(f"\nGRAPH facts (top 12 of {len(graph['facts'])}):")
    for f in graph["facts"][:12]:
        print(f"   {f['score']:.2f}  hop{f['hop']}  {f['text']}  [{f['evidence']}]")
    print()
