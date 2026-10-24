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
