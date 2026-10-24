"""Step 4 — Embed the chunks and store them in a FAISS index.

Files written to FAISS_INDEX_DIR (default: vectorstore/):
  index.faiss    the vectors (cosine similarity via normalised inner product)
  docstore.json  chunk text + metadata, in the same order as the vectors
  manifest.json  embedding model, dimension, count (to catch mismatches)
"""
import json
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings

BATCH_SIZE = 64


def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    return OpenAIEmbeddings(model=s.openai_embed_model, api_key=s.openai_api_key)


def _to_matrix(vectors: list[list[float]]) -> np.ndarray:
    mat = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(mat)  # after this, inner product == cosine similarity
    return mat


def build_vector_store(docs: list[Document], embeddings=None, index_dir: str | None = None) -> int:
    """Embed every chunk and save a fresh FAISS index. Returns the vector count."""
    s = get_settings()
    embeddings = embeddings or get_embeddings()
    index_dir = Path(index_dir or s.faiss_index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    texts = [d.page_content for d in docs]
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        vectors.extend(embeddings.embed_documents(texts[i : i + BATCH_SIZE]))

    mat = _to_matrix(vectors)
    index = faiss.IndexFlatIP(mat.shape[1])  # exact search, fine for small corpora
    index.add(mat)

    faiss.write_index(index, str(index_dir / "index.faiss"))
    (index_dir / "docstore.json").write_text(
        json.dumps(
            [{"page_content": d.page_content, "metadata": d.metadata} for d in docs],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (index_dir / "manifest.json").write_text(
        json.dumps(
            {
                "embed_model": getattr(embeddings, "model", "unknown"),
                "dimension": int(mat.shape[1]),
                "count": int(index.ntotal),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return int(index.ntotal)


class VectorStore:
    """Loads the saved index and runs similarity search."""

    def __init__(self, embeddings=None, index_dir: str | None = None):
        s = get_settings()
        self.embeddings = embeddings or get_embeddings()
        index_dir = Path(index_dir or s.faiss_index_dir)

        if not (index_dir / "index.faiss").exists():
            raise FileNotFoundError(
                f"No FAISS index in '{index_dir}'. Run: python scripts/build_vector_store.py"
            )

        self.index = faiss.read_index(str(index_dir / "index.faiss"))
        self.docs = json.loads((index_dir / "docstore.json").read_text(encoding="utf-8"))
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))

        current_model = getattr(self.embeddings, "model", "unknown")
        if manifest["embed_model"] != current_model:
            raise ValueError(
                f"Index was built with '{manifest['embed_model']}' but the app is using "
                f"'{current_model}'. Rebuild the index."
            )

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Return the top-k chunks: [{chunk_id, score, page_content, metadata}, ...]."""
        q = _to_matrix([self.embeddings.embed_query(query)])
        scores, idxs = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc = self.docs[idx]
            results.append(
                {
                    "chunk_id": doc["metadata"]["chunk_id"],
                    "score": round(float(score), 4),  # cosine similarity, higher = better
                    "page_content": doc["page_content"],
                    "metadata": doc["metadata"],
                }
            )
        return results