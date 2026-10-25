"""Step 7 — Hybrid retrieval: combine FAISS and Neo4j into one context.

1. FAISS returns the top-k chunks for the question.
2. Neo4j returns entities + facts (seeded by the question AND by those chunks).
3. Graph -> text: every fact carries the chunk_ids it came from, so the graph
   can "vote" for chunks FAISS ranked low or missed.
4. Reciprocal Rank Fusion (RRF) merges the two chunk rankings without having
   to compare cosine scores with graph scores directly.
5. build_context() formats passages [C1..], graph facts [F1..] and entity
   summaries [E1..] as the prompt context for the LLM (Step 8).
"""
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.config import get_settings
from app.retrieval.graph_retriever import graph_search, keyword_seeds, link_by_fulltext
from app.retrieval.query_expansion import expand_query
from app.retrieval.vector_retriever import get_vector_store, multi_vector_search, vector_search

RRF_K = 10  # small corpus -> small k keeps the top ranks distinct

# Shared worker pool: OpenAI and Neo4j calls are network-bound, so threads overlap well
_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="retrieval")


@lru_cache
def _chunk_lookup() -> dict[str, dict]:
    """chunk_id -> {page_content, metadata} from the FAISS docstore."""
    return {d["metadata"]["chunk_id"]: d for d in get_vector_store().docs}


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank)


def _safe_expand(question: str) -> dict:
    try:
        return expand_query(question)
    except Exception as exc:  # noqa: BLE001  (never fail a question because of this step)
        print(f"[query_expansion] skipped: {exc}")
        return {"queries": [question], "keywords": []}


def hybrid_retrieve(question: str, k_vector: int = 5, max_chunks: int = 6,
                    max_facts: int = 25, max_entities: int = 6) -> dict:
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    use_expansion = get_settings().query_expansion

    # Phase A (in parallel): expansion LLM call | FAISS for the original question |
    #                        graph full-text lookup for the original question
    f_exp = _POOL.submit(_safe_expand, question) if use_expansion else None
    f_vec = _POOL.submit(vector_search, question, k_vector)
    f_seed = _POOL.submit(link_by_fulltext, question)

    vec = f_vec.result()
    seeds = dict(f_seed.result())
    expansion = {"queries": [question], "keywords": []}
    timings["phase_a_s"] = round(time.perf_counter() - t0, 2)

    # Phase B (in parallel, only if expansion produced something): FAISS for all phrasings
    # in ONE embedding call | graph full-text lookup for the expansion keywords
    best = vec[0]["score"] if vec else 0.0
    if f_exp is not None and best >= get_settings().expansion_skip_min_score:
        # Strong direct match: answer from it instead of waiting for the expansion LLM call.
        # (The call finishes in the background and is cached for repeat questions.)
        expansion = {"queries": [question], "keywords": [],
                     "skipped": f"strong direct match ({best:.2f})"}
        f_exp = None
    if f_exp is not None:
        expansion = f_exp.result()
        timings["expansion_s"] = round(time.perf_counter() - t0, 2)
        t1 = time.perf_counter()
        f_multi = (_POOL.submit(multi_vector_search, expansion["queries"], k_vector)
                   if len(expansion["queries"]) > 1 else None)
        f_kw = _POOL.submit(keyword_seeds, expansion["keywords"])
        if f_multi is not None:
            vec = f_multi.result()
        for uid, score in f_kw.result().items():
            seeds[uid] = max(seeds.get(uid, 0), score)
        timings["phase_b_s"] = round(time.perf_counter() - t1, 2)

    # Phase C: graph expansion, seeded by question + keywords + the FAISS chunks
    t2 = time.perf_counter()
    graph = graph_search(question, chunk_ids=[r["chunk_id"] for r in vec], max_facts=max_facts,
                         precomputed_seeds=seeds)
    facts, entities = graph["facts"], graph["entities"][:max_entities]
    timings["graph_s"] = round(time.perf_counter() - t2, 2)

    # 3. Graph votes for chunks: sum of the scores of facts that cite each chunk
    graph_votes: dict[str, float] = {}
    for f in facts:
        for cid in f["chunk_ids"]:
            graph_votes[cid] = graph_votes.get(cid, 0.0) + f["score"]

    vector_rank = {r["chunk_id"]: i for i, r in enumerate(vec, start=1)}
    vector_score = {r["chunk_id"]: r["score"] for r in vec}
    graph_rank = {cid: i for i, (cid, _) in
                  enumerate(sorted(graph_votes.items(), key=lambda x: -x[1]), start=1)}

    # 4. Reciprocal Rank Fusion
    lookup = _chunk_lookup()
    fused = []
    for cid in set(vector_rank) | set(graph_rank):
        if cid not in lookup:
            continue
        score = 0.0
        sources = []
        if cid in vector_rank:
            score += _rrf(vector_rank[cid])
            sources.append("vector")
        if cid in graph_rank:
            score += _rrf(graph_rank[cid])
            sources.append("graph")
        doc = lookup[cid]
        fused.append({
            "chunk_id": cid,
            "text": doc["page_content"],
            "section": doc["metadata"]["section"],
            "subsection": doc["metadata"]["subsection"],
            "page": doc["metadata"]["page"],
            "vector_score": vector_score.get(cid),
            "graph_votes": round(graph_votes.get(cid, 0.0), 3),
            "fused_score": round(score, 4),
            "sources": sources,
        })
    fused.sort(key=lambda c: -c["fused_score"])
    chunks = fused[:max_chunks]

    return {
        "question": question,
        "expansion": expansion,
        "timings": timings,
        "chunks": chunks,
        "facts": facts,
        "entities": entities,
        "context": build_context(chunks, facts, entities),
    }


def build_context(chunks: list[dict], facts: list[dict], entities: list[dict]) -> str:
    """Prompt context with citation labels the LLM must use."""
    cid_to_label = {c["chunk_id"]: f"C{i}" for i, c in enumerate(chunks, start=1)}
    lines = ["## Document passages"]
    for c in chunks:
        lines.append(
            f"[{cid_to_label[c['chunk_id']]}] ({c['chunk_id']} | {c['section']} > "
            f"{c['subsection']} | page {c['page']})\n{c['text']}\n"
        )

    lines.append("## Knowledge graph facts")
    for i, f in enumerate(facts, start=1):
        cites = [cid_to_label[c] for c in f["chunk_ids"] if c in cid_to_label]
        origin = f"from {', '.join(cites)}" if cites else "graph only"
        lines.append(f"[F{i}] {f['text']}  ({f['evidence']}; {origin})")

    described = [e for e in entities if e.get("description")]
    if described:
        lines.append("\n## Entity summaries")
        for i, e in enumerate(described, start=1):
            lines.append(f"[E{i}] {e['name']} ({e['label']}): {e['description']}")

    return "\n".join(lines)
