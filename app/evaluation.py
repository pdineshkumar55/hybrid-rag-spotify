"""Step 11 — Evaluation metrics (no LLM needed, so runs are cheap and repeatable).

Per question:
  chunk_recall      share of expected chunks that reached the LLM context
  vector_recall     ... counting only chunks FAISS found
  graph_rescued     expected chunks that ONLY the graph brought in
  fact_recall       share of expected graph facts returned
  keyword_coverage  share of key points (must_mention groups) present in the final answer
  oos_correct       out-of-scope question correctly answered "not in the document"
"""
from app.generation.answer import NO_ANSWER


def _ratio(hit: int, total: int) -> float | None:
    return round(hit / total, 3) if total else None


def _fact_matches(expected: list[str], fact: dict) -> bool:
    src, rel, tgt = (x.lower() for x in expected)
    return (fact["rel"] == rel.upper()
            and src in fact["source"].lower()
            and tgt in fact["target"].lower())


def score_question(item: dict, result: dict) -> dict:
    answer = result["answer"]
    answer_l = answer.lower()
    passages = [s for s in result["sources"] if s["type"] == "passage"]
    facts = [s for s in result["sources"] if s["type"] == "graph_fact"]
    v = result.get("verification") or {}

    retrieved = {p["chunk_id"]: p["retrieved_by"] for p in passages}
    expected = item["expected_chunks"]
    found = [c for c in expected if c in retrieved]
    vector_found = [c for c in expected if "vector" in retrieved.get(c, [])]
    graph_only = [c for c in found if retrieved[c] == ["graph"]]

    fact_hits = [f for f in item["expected_facts"] if any(_fact_matches(f, x) for x in facts)]
    kw_hits = [g for g in item["must_mention"] if any(a.lower() in answer_l for a in g)]

    is_no_answer = answer.strip() == NO_ANSWER or "doesn't contain information" in answer_l

    row = {
        "id": item["id"],
        "type": item["type"],
        "question": item["question"],
        "chunk_recall": _ratio(len(found), len(expected)),
        "vector_recall": _ratio(len(vector_found), len(expected)),
        "graph_rescued": graph_only,
        "missing_chunks": [c for c in expected if c not in retrieved],
        "fact_recall": _ratio(len(fact_hits), len(item["expected_facts"])),
        "missing_facts": [f for f in item["expected_facts"] if f not in fact_hits],
        "keyword_coverage": _ratio(len(kw_hits), len(item["must_mention"])),
        "missing_keywords": [g[0] for g in item["must_mention"] if g not in kw_hits],
        "verdict": v.get("verdict"),
        "confidence": v.get("confidence"),
        "was_corrected": result.get("was_corrected", False),
        "seconds": result.get("timings", {}).get("total_s")
                   or round(sum(result.get("timings", {}).values()), 2),
        "answer": answer,
    }

    if item["type"] == "out_of_scope":
        row["oos_correct"] = is_no_answer
        row["passed"] = is_no_answer
    else:
        # A question "passes" when the answer covers every key point and isn't a refusal
        row["passed"] = (not is_no_answer) and (row["keyword_coverage"] in (None, 1.0))
    return row


def summarize(rows: list[dict]) -> dict:
    def mean(key, subset):
        vals = [r[key] for r in subset if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    in_scope = [r for r in rows if r["type"] != "out_of_scope"]
    oos = [r for r in rows if r["type"] == "out_of_scope"]
    summary = {
        "questions": len(rows),
        "pass_rate": round(sum(r["passed"] for r in rows) / len(rows), 3) if rows else None,
        "chunk_recall": mean("chunk_recall", in_scope),
        "vector_recall": mean("vector_recall", in_scope),
        "graph_rescued_chunks": sum(len(r["graph_rescued"]) for r in in_scope),
        "fact_recall": mean("fact_recall", in_scope),
        "keyword_coverage": mean("keyword_coverage", in_scope),
        "verified_supported": round(sum(r["verdict"] == "supported" for r in in_scope) / len(in_scope), 3)
                              if in_scope else None,
        "corrected_by_verifier": sum(r["was_corrected"] for r in rows),
        "out_of_scope_correct": f"{sum(r['oos_correct'] for r in oos)}/{len(oos)}" if oos else None,
        "avg_seconds": mean("seconds", rows),
        "by_type": {},
    }
    for t in sorted({r["type"] for r in in_scope}):
        subset = [r for r in in_scope if r["type"] == t]
        summary["by_type"][t] = {
            "n": len(subset),
            "pass_rate": round(sum(r["passed"] for r in subset) / len(subset), 3),
            "chunk_recall": mean("chunk_recall", subset),
            "fact_recall": mean("fact_recall", subset),
            "keyword_coverage": mean("keyword_coverage", subset),
        }
    return summary
