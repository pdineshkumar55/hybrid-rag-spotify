"""Step 6a — Vector retrieval (FAISS).

Thin wrapper so the rest of the app never touches FAISS directly.
The index is loaded once and reused (important for FastAPI performance).
"""
from functools import lru_cache

from app.ingestion.vector_builder import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()


def vector_search(question: str, k: int = 5, min_score: float = 0.25) -> list[dict]:
    """Top-k chunks for the question.

    Returns: [{chunk_id, score, page_content, metadata}], score = cosine similarity (0..1).
    Chunks below min_score are dropped so unrelated questions return nothing.
    """
    results = get_vector_store().search(question, k=k)
    return [r for r in results if r["score"] >= min_score]


def multi_vector_search(queries: list[str], k: int = 5, min_score: float = 0.25,
                        rrf_k: int = 10) -> list[dict]:
    """Search FAISS with several phrasings of the question and merge with RRF.

    A chunk found by several phrasings, or ranked high by one, comes first.
    'score' stays the best cosine similarity seen, for display/thresholds.
    """
    merged: dict[str, dict] = {}
    for q in dict.fromkeys(q.strip() for q in queries if q.strip()):
        for rank, r in enumerate(vector_search(q, k=k, min_score=min_score), start=1):
            m = merged.setdefault(r["chunk_id"], {**r, "rrf": 0.0, "matched_queries": []})
            m["rrf"] += 1.0 / (rrf_k + rank)
            m["score"] = max(m["score"], r["score"])
            m["matched_queries"].append(q)
    ranked = sorted(merged.values(), key=lambda r: -r["rrf"])
    return ranked[:k]
