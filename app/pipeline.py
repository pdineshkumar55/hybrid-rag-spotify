"""The full question -> answer pipeline (used by FastAPI).

    retrieve (FAISS + Neo4j [+ Cypher])  ->  answer (OpenAI)  ->  verify (auto)  ->  final answer

ask()         returns the whole result at once.
ask_stream()  yields events so the UI can show the answer while it is being written:
              {"type": "retrieval", ...}  -> {"type": "token", "text": ...} x N  -> {"type": "final", "result": ...}
"""
import time

from app.cache import TTLCache, normalize_question
from app.config import get_settings
from app.generation.answer import clean_answer, generate_answer, stream_answer
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
    gq = retrieval.get("graph_query")
    if gq and gq.get("rows") is not None:
        sources.append({
            "label": "Q1", "type": "graph_query", "text": gq.get("explanation", ""),
            "cypher": gq["cypher"], "rows": gq["rows"], "error": gq.get("error"),
        })
    return sources


def _mode(verify: bool) -> str:
    return get_settings().verify_mode if verify else "never"


def _cached(question: str, mode: str):
    if get_settings().answer_cache_ttl_s <= 0:
        return None
    hit = ANSWER_CACHE.get((normalize_question(question), mode))
    if hit is None:
        return None
    return {**hit, "cached": True, "timings": {"total_s": 0.0, "cache_hit": True}}


def _retrieve(question: str, timings: dict) -> dict:
    t = time.perf_counter()
    retrieval = hybrid_retrieve(question)
    timings["retrieval_s"] = round(time.perf_counter() - t, 2)
    timings.update(retrieval.get("timings", {}))
    return retrieval


def _finish(question: str, draft: str, retrieval: dict, timings: dict, mode: str) -> dict:
    """Verification (auto/always/never), result assembly and caching."""
    s = get_settings()
    verification, final = None, draft
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
    result = {
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
            "graph_query": retrieval.get("graph_query"),
            "fact_count": len(retrieval["facts"]),
            "entity_count": len(retrieval["entities"]),
        },
        "timings": timings,
    }
    if s.answer_cache_ttl_s > 0:
        ANSWER_CACHE.set((normalize_question(question), mode), result)
    return {**result, "cached": False}


def ask(question: str, verify: bool = True) -> dict:
    mode = _mode(verify)
    cached = _cached(question, mode)
    if cached is not None:
        return cached

    timings: dict = {}
    retrieval = _retrieve(question, timings)
    t = time.perf_counter()
    draft = generate_answer(question, retrieval["context"])
    timings["answer_s"] = round(time.perf_counter() - t, 2)
    return _finish(question, draft, retrieval, timings, mode)


def ask_stream(question: str, verify: bool = True):
    """Generator of events for streaming clients (see module docstring)."""
    mode = _mode(verify)
    cached = _cached(question, mode)
    if cached is not None:
        yield {"type": "retrieval", "cached": True}
        yield {"type": "token", "text": cached["answer"]}
        yield {"type": "final", "result": cached}
        return

    timings: dict = {}
    retrieval = _retrieve(question, timings)
    yield {"type": "retrieval", "cached": False,
           "chunks": len(retrieval["chunks"]), "facts": len(retrieval["facts"]),
           "retrieval_s": timings["retrieval_s"]}

    t = time.perf_counter()
    first_token_at = None
    parts = []
    for piece in stream_answer(question, retrieval["context"]):
        if first_token_at is None:
            first_token_at = round(time.perf_counter() - t, 2)
        parts.append(piece)
        yield {"type": "token", "text": piece}
    timings["answer_s"] = round(time.perf_counter() - t, 2)
    timings["first_token_s"] = round(timings["retrieval_s"] + (first_token_at or 0), 2)

    draft = clean_answer("".join(parts))
    yield {"type": "final", "result": _finish(question, draft, retrieval, timings, mode)}
