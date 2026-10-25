"""The full question -> answer pipeline (used by FastAPI in Step 9).

    retrieve (FAISS + Neo4j)  ->  answer (OpenAI)  ->  verify (OpenAI)  ->  final answer
"""
import time

from app.cache import TTLCache, normalize_question
from app.config import get_settings
from app.generation.answer import generate_answer
from app.generation.verify import can_skip_verification, check_citations, verify_answer
from app.retrieval.hybrid import hybrid_retrieve

ANSWER_CACHE = TTLCache(maxsize=256, ttl_s=get_settings().answer_cache_ttl_s or 1)


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
    s = get_settings()
    mode = s.verify_mode if verify else "never"
    cache_key = (normalize_question(question), mode)

    if s.answer_cache_ttl_s > 0:
        cached = ANSWER_CACHE.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True, "timings": {"total_s": 0.0, "cache_hit": True}}

    timings = {}
    t = time.perf_counter()
    retrieval = hybrid_retrieve(question)
    timings["retrieval_s"] = round(time.perf_counter() - t, 2)
    timings.update(retrieval.get("timings", {}))

    t = time.perf_counter()
    draft = generate_answer(question, retrieval["context"])
    timings["answer_s"] = round(time.perf_counter() - t, 2)

    verification = None
    final = draft
    if mode != "never":
        skip, reason = (False, "")
        if mode == "auto":
            skip, reason = can_skip_verification(draft, retrieval["context"], retrieval,
                                                 s.verify_skip_min_score)
        if skip:
            verification = {
                "verdict": "auto_accepted", "confidence": "high", "claims": [],
                "corrected_answer": "", "reason": reason,
                "citations": check_citations(draft, retrieval["context"]),
            }
        else:
            t = time.perf_counter()
            verification = verify_answer(question, retrieval["context"], draft)
            verification["reason"] = reason or "verify_mode=always"
            timings["verify_s"] = round(time.perf_counter() - t, 2)
            if verification["verdict"] != "supported" and verification.get("corrected_answer"):
                final = verification["corrected_answer"]

    timings["total_s"] = round(sum(v for k, v in timings.items()
                                   if k in ("retrieval_s", "answer_s", "verify_s")), 2)
    result = _build_result(question, final, draft, verification, retrieval, timings)
    if s.answer_cache_ttl_s > 0:
        ANSWER_CACHE.set(cache_key, result)
    return {**result, "cached": False}


def _build_result(question, final, draft, verification, retrieval, timings) -> dict:
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
            "expansion": retrieval.get("expansion"),
            "fact_count": len(retrieval["facts"]),
            "entity_count": len(retrieval["entities"]),
        },
        "timings": timings,
    }
