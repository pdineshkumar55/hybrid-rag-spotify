"""Step 6b — Graph retrieval (Neo4j).

1. Entity linking — find the graph entities the question is about, two ways:
   a) full-text search of entity names/aliases/descriptions with the question words
   b) entities MENTIONED by the chunks FAISS returned (the hybrid bridge)
2. Expansion — collect the relationships around those entities:
   hop 1: every relationship of a seed entity
   hop 2: along dependency links only (e.g. Table <- Service, Service -> Service)
3. Each fact is scored and returned as a readable sentence plus its source chunk_ids.
"""
import re

from app.neo4j_client import run_query

FULLTEXT_INDEX = "spotify_entity_text"

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from",
    "how", "i", "in", "is", "it", "of", "on", "or", "should", "that", "the", "this", "to",
    "what", "when", "where", "which", "who", "why", "will", "with", "would", "app", "used",
    "use", "uses", "tell", "me", "about", "list", "all", "explain", "show", "give",
    "happens", "happen", "after", "before", "get", "gets", "work", "works", "not", "my", "we",
    "our", "there", "any", "has", "have", "handle", "handles", "handled",
}

KEYWORD_SEED_WEIGHT = 0.6  # expansion keywords count less than the question itself
MAX_FACTS_PER_SEED = 8       # hop-1 facts kept per seed entity
MAX_FACTS_PER_HOP2_NODE = 4  # hop-2 facts kept per intermediate node

# Relationship types worth following on the 2nd hop
HOP2_TYPES = ["OWNS_TABLE", "USES", "INTEGRATES_WITH", "NOTIFIES", "FEEDS",
              "AGGREGATES", "LOGS_EVENTS_TO", "PUBLISHES", "CONSUMES", "CARRIES"]


# ---------------------------------------------------------------- linking
def build_fulltext_query(question: str) -> str:
    """Turn a question into a safe Lucene query, boosting name matches."""
    words = re.findall(r"[a-z0-9_]+", question.lower())
    terms = [w for w in words if w not in STOPWORDS and len(w) > 1]
    if not terms:
        return ""
    parts = []
    for t in dict.fromkeys(terms):  # de-duplicate, keep order
        parts.append(f"name:{t}^3 aliases_text:{t}^2 description:{t}")
    return " ".join(parts)


def link_by_fulltext(question: str, limit: int = 12, rel_cutoff: float = 0.35) -> dict[str, float]:
    """uid -> normalised score (0..1) for entities matching the question text."""
    q = build_fulltext_query(question)
    if not q:
        return {}
    rows = run_query(
        f"""
        CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $q) YIELD node, score
        WHERE node:Spotify AND NOT node:Chunk
        RETURN node.uid AS uid, score
        ORDER BY score DESC LIMIT $limit
        """,
        q=q, limit=limit,
    )
    if not rows:
        return {}
    top = rows[0]["score"]
    return {r["uid"]: round(r["score"] / top, 3) for r in rows if r["score"] / top >= rel_cutoff}


def link_by_chunks(chunk_ids: list[str], weight: float = 0.5, per_chunk: int = 6) -> dict[str, float]:
    """uid -> score for entities mentioned by the FAISS chunks (closed labels first)."""
    if not chunk_ids:
        return {}
    rows = run_query(
        """
        UNWIND range(0, size($chunk_ids) - 1) AS rank
        MATCH (c:Spotify:Chunk {chunk_id: $chunk_ids[rank]})-[:MENTIONS]->(e:Spotify)
        WITH rank, e,
             CASE WHEN e:Service OR e:FeatureArea OR e:Layer OR e:PageGroup THEN 0 ELSE 1 END AS prio
        ORDER BY rank, prio
        WITH rank, collect(e.uid)[..$per_chunk] AS uids
        UNWIND uids AS uid
        RETURN rank, uid
        """,
        chunk_ids=chunk_ids, per_chunk=per_chunk,
    )
    seeds: dict[str, float] = {}
    for r in rows:
        score = round(weight / (1 + r["rank"]), 3)  # 1st chunk 0.5, 2nd 0.25, ...
        seeds[r["uid"]] = max(seeds.get(r["uid"], 0), score)
    return seeds


# ---------------------------------------------------------------- expansion
def _fact_text(f: dict) -> str:
    return f"{f['source']} ({f['source_label']}) -[{f['rel']}]-> {f['target']} ({f['target_label']})"


def expand(seeds: dict[str, float], max_facts: int = 40) -> tuple[list[dict], list[dict]]:
    """Return (entities, facts) around the seed entities."""
    if not seeds:
        return [], []
    uids = list(seeds)

    entities = run_query(
        """
        MATCH (e:Spotify) WHERE e.uid IN $uids
        RETURN e.uid AS uid, e.name AS name,
               [l IN labels(e) WHERE l <> 'Spotify'][0] AS label, e.description AS description
        """,
        uids=uids,
    )
    for e in entities:
        e["score"] = seeds[e["uid"]]
    entities.sort(key=lambda e: -e["score"])

    hop1 = run_query(
        """
        MATCH (e:Spotify)-[r]-(n:Spotify)
        WHERE e.uid IN $uids AND type(r) <> 'MENTIONS' AND NOT n:Chunk
        WITH e, r, startNode(r) AS a, endNode(r) AS b, n
        RETURN e.uid AS seed, 1 AS hop, type(r) AS rel, r.evidence AS evidence,
               coalesce(r.chunk_ids, []) AS chunk_ids,
               a.name AS source, [l IN labels(a) WHERE l <> 'Spotify'][0] AS source_label,
               b.name AS target, [l IN labels(b) WHERE l <> 'Spotify'][0] AS target_label,
               n.uid AS neighbour, n:Service AS neighbour_is_service
        """,
        uids=uids,
    )

    # 2nd hop: from every node reached in hop 1, but only along "dependency"
    # relationships, e.g. Table <-OWNS_TABLE- Service, Service -NOTIFIES-> Service.
    hop2_from: dict[str, float] = {}
    for f in hop1:
        if f["neighbour"] not in seeds:
            hop2_from[f["neighbour"]] = max(hop2_from.get(f["neighbour"], 0), seeds[f["seed"]] * 0.5)

    hop2 = []
    if hop2_from:
        hop2 = run_query(
            """
            MATCH (s:Spotify)-[r]-(n:Spotify)
            WHERE s.uid IN $uids AND type(r) IN $types AND NOT n:Chunk
            WITH s, r, startNode(r) AS a, endNode(r) AS b
            RETURN s.uid AS seed, 2 AS hop, type(r) AS rel, r.evidence AS evidence,
                   coalesce(r.chunk_ids, []) AS chunk_ids,
                   a.name AS source, [l IN labels(a) WHERE l <> 'Spotify'][0] AS source_label,
                   b.name AS target, [l IN labels(b) WHERE l <> 'Spotify'][0] AS target_label
            """,
            uids=list(hop2_from), types=HOP2_TYPES,
        )

    # Score and de-duplicate
    weights = {**hop2_from, **seeds}
    best: dict[tuple, dict] = {}
    for f in hop1 + hop2:
        score = weights.get(f["seed"], 0)
        if f["evidence"] == "inferred":
            score *= 0.8  # stated facts rank above inferred ones
        key = (f["source"], f["rel"], f["target"])
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "text": _fact_text(f),
                "source": f["source"], "source_label": f["source_label"],
                "rel": f["rel"],
                "target": f["target"], "target_label": f["target_label"],
                "evidence": f["evidence"], "chunk_ids": f["chunk_ids"],
                "hop": f["hop"], "score": round(score, 3), "via": f["seed"],
            }

    # Keep the best facts, but cap how many come from any single node, so a
    # hub like "tracks" or "PostgreSQL" can't fill the whole context.
    facts, per_node = [], {}
    for f in sorted(best.values(), key=lambda f: (-f["score"], f["hop"], f["text"])):
        cap = MAX_FACTS_PER_SEED if f["hop"] == 1 else MAX_FACTS_PER_HOP2_NODE
        if per_node.get(f["via"], 0) >= cap:
            continue
        per_node[f["via"]] = per_node.get(f["via"], 0) + 1
        facts.append(f)
        if len(facts) >= max_facts:
            break
    return entities, facts


# ---------------------------------------------------------------- public API
def keyword_seeds(keywords: list[str]) -> dict[str, float]:
    """Full-text seeds for expansion keywords, down-weighted (see KEYWORD_SEED_WEIGHT)."""
    if not keywords:
        return {}
    return {uid: round(score * KEYWORD_SEED_WEIGHT, 3)
            for uid, score in link_by_fulltext(" ".join(keywords)).items()}


def graph_search(question: str, chunk_ids: list[str] | None = None, max_facts: int = 40,
                 keywords: list[str] | None = None,
                 precomputed_seeds: dict[str, float] | None = None) -> dict:
    """Graph context for a question.

    chunk_ids: FAISS chunk IDs for the same question (optional) — their entities
               become extra seeds, which links the two retrievers together.
    keywords:  extra search terms from query expansion (document vocabulary).
    """
    # The user's own words get full weight; expansion keywords are searched
    # separately at a lower weight, so they add seeds without pushing out the
    # entities the question actually names (e.g. the "devices" table).
    if precomputed_seeds is not None:   # hybrid.py already ran the full-text lookups in parallel
        seeds = dict(precomputed_seeds)
    else:
        seeds = link_by_fulltext(question)
        for uid, score in keyword_seeds(keywords or []).items():
            seeds[uid] = max(seeds.get(uid, 0), score)
    for uid, score in link_by_chunks(chunk_ids or []).items():
        seeds[uid] = max(seeds.get(uid, 0), score)
    entities, facts = expand(seeds, max_facts=max_facts)
    return {"entities": entities, "facts": facts}
