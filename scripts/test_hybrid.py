"""See exactly what context the LLM will receive.

    python scripts/test_hybrid.py
    python scripts/test_hybrid.py "Which services use PostgreSQL tables?"
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.retrieval.hybrid import hybrid_retrieve

question = " ".join(sys.argv[1:]) or "What happens after an artist uploads a track?"
result = hybrid_retrieve(question)

print(f"Q: {question}\n")
print("FUSED CHUNKS (RRF):")
for c in result["chunks"]:
    vs = f"{c['vector_score']:.3f}" if c["vector_score"] is not None else "  -  "
    print(f"   {c['fused_score']:.4f}  vec={vs}  graph={c['graph_votes']:<5}  "
          f"{'+'.join(c['sources']):<12} {c['chunk_id']:<16} {c['subsection']}")

print(f"\n{len(result['facts'])} graph facts, {len(result['entities'])} entities")
print("\n" + "=" * 90 + "\nCONTEXT SENT TO THE LLM\n" + "=" * 90)
print(result["context"])
