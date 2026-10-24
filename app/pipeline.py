"""The full question -> answer pipeline (used by FastAPI in Step 9).

    retrieve (FAISS + Neo4j)  ->  answer (OpenAI)  ->  verify (OpenAI)  ->  final answer
"""
import time

from app.generation.answer import generate_answer
from app.generation.verify import verify_answer
from app.retrieval.hybrid import hybrid_retrieve


def _sources(retrieval: dict) -> list[dict]:
    """Citation label -> where it came from, for display in the UI."""
    sources = []
    for i, c in enumerate(retrieval["chunks"], start=1):
        sources.append({
            "label": f"C{i}", "type": "passage", "chunk_id": c["chunk_id"],
            "title": f"{c['section']} > {c['subsection']}", "page": c["page"],
            "text": c["text"], "retrieved_by": c["sources"],
        })
    for i, f in enumerate(retrieval["facts"], start=1):
        sources.append({
            "label": f"F{i}", "type": "graph_fact", "text": f["text"],
            "source": f["source"], "source_label": f["source_label"], "rel": f["rel"],
            "target": f["target"], "target_label": f["target_label"],
            "evidence": f["evidence"], "chunk_ids": f["chunk_ids"], "hop": f["hop"],
        })
    for i, e in enumerate([e for e in retrieval["entities"] if e.get("description")], start=1):
        sources.append({
            "label": f"E{i}", "type": "entity", "text": f"{e['name']} ({e['label']}): {e['description']}",
        })
    return sources


def ask(question: str, verify: bool = True) -> dict:
    timings = {}

    t = time.perf_counter()
    retrieval = hybrid_retrieve(question)
    timings["retrieval_s"] = round(time.perf_counter() - t, 2)

    t = time.perf_counter()
    draft = generate_answer(question, retrieval["context"])
    timings["answer_s"] = round(time.perf_counter() - t, 2)

    verification = None
    final = draft
    if verify:
        t = time.perf_counter()
        verification = verify_answer(question, retrieval["context"], draft)
        timings["verify_s"] = round(time.perf_counter() - t, 2)
        if verification["verdict"] != "supported" and verification.get("corrected_answer"):
            final = verification["corrected_answer"]

    return {
        "question": question,
        "answer": final,
        "draft_answer": draft,
        "was_corrected": final != draft,
        "verification": verification,
        "sources": _sources(retrieval),
        "retrieval": {
            "chunks": [{k: c[k] for k in ("chunk_id", "subsection", "fused_score",
                                           "vector_score", "graph_votes", "sources")}
                       for c in retrieval["chunks"]],
            "fact_count": len(retrieval["facts"]),
            "entity_count": len(retrieval["entities"]),
        },
        "timings": timings,
    }
