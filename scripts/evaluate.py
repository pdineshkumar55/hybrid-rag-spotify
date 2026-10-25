"""Step 11 — Run the evaluation set and score the app.

    python scripts/evaluate.py                       # all questions, with verification
    python scripts/evaluate.py --no-verify           # faster/cheaper (1 LLM call per question)
    python scripts/evaluate.py --ids q02 q11 q12     # only some questions
    python scripts/evaluate.py --api http://localhost:8000   # test the running API instead

Each run is saved to data/eval/runs/<timestamp>.json and compared with the previous run,
so every change to retrieval or the graph shows up as a measurable difference.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.evaluation import score_question, summarize

QUESTIONS = Path("data/eval/questions.json")
RUNS_DIR = Path("data/eval/runs")
METRICS = ["pass_rate", "chunk_recall", "vector_recall", "graph_rescued_chunks", "fact_recall",
           "keyword_coverage", "verified_supported", "verify_skipped", "corrected_by_verifier",
           "avg_seconds"]


def ask_local(question: str, verify: bool) -> dict:
    from app.pipeline import ask
    t = time.perf_counter()
    r = ask(question, verify=verify)
    r["timings"] = {**r["timings"], "total_s": round(time.perf_counter() - t, 2)}
    return r


def ask_api(base: str, question: str, verify: bool) -> dict:
    import requests
    r = requests.post(f"{base}/query", json={"question": question, "verify": verify}, timeout=180)
    r.raise_for_status()
    return r.json()


def fmt(x) -> str:
    if x is None:
        return "  -  "
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ids", nargs="*", help="only run these question ids")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--api", help="call a running API instead of the local pipeline")
    p.add_argument("--label", default="", help="short note saved with the run, e.g. 'baseline'")
    args = p.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))["questions"]
    if args.ids:
        items = [q for q in items if q["id"] in args.ids]

    rows = []
    print(f"Running {len(items)} questions (verify={not args.no_verify})\n")
    print(f"{'id':<4} {'type':<12} {'pass':<5} {'chunks':>6} {'facts':>6} {'keys':>6} {'verdict':<20} {'sec':>5}")
    for item in items:
        try:
            result = (ask_api(args.api, item["question"], not args.no_verify) if args.api
                      else ask_local(item["question"], not args.no_verify))
            row = score_question(item, result)
        except Exception as exc:  # noqa: BLE001
            row = {"id": item["id"], "type": item["type"], "question": item["question"],
                   "passed": False, "error": f"{type(exc).__name__}: {exc}",
                   "chunk_recall": None, "vector_recall": None, "graph_rescued": [],
                   "fact_recall": None, "keyword_coverage": None, "verdict": "error",
                   "confidence": None, "was_corrected": False, "seconds": None,
                   "oos_correct": False, "answer": ""}
        rows.append(row)
        print(f"{row['id']:<4} {row['type']:<12} {'✅' if row['passed'] else '❌':<5} "
              f"{fmt(row['chunk_recall']):>6} {fmt(row['fact_recall']):>6} "
              f"{fmt(row['keyword_coverage']):>6} {str(row['verdict']):<20} {fmt(row['seconds']):>5}")

    summary = summarize(rows)

    # ---- failures, so you know what to fix ----
    failed = [r for r in rows if not r["passed"]]
    if failed:
        print("\nFAILURES")
        for r in failed:
            print(f"  {r['id']} {r['question']}")
            if r.get("error"):
                print(f"     error: {r['error']}")
                continue
            if r.get("missing_chunks"):
                print(f"     missing chunks:   {r['missing_chunks']}")
            if r.get("missing_facts"):
                print(f"     missing facts:    {r['missing_facts']}")
            if r.get("missing_keywords"):
                print(f"     missing keywords: {r['missing_keywords']}")
            print(f"     answer: {r['answer'][:160].replace(chr(10), ' ')}")

    # ---- compare with the previous run ----
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # only compare with runs made in the same mode (with/without verification)
    previous = [f for f in sorted(RUNS_DIR.glob("*.json"))
                if json.loads(f.read_text(encoding="utf-8")).get("verify") == (not args.no_verify)]
    prev_summary = json.loads(previous[-1].read_text(encoding="utf-8"))["summary"] if previous else None

    print("\nSUMMARY" + (f"  (vs previous run {previous[-1].stem})" if prev_summary else ""))
    for m in METRICS:
        now = summary.get(m)
        line = f"  {m:<22} {fmt(now):>6}"
        if prev_summary and isinstance(now, (int, float)) and isinstance(prev_summary.get(m), (int, float)):
            delta = now - prev_summary[m]
            if abs(delta) > 1e-9:
                line += f"   ({'+' if delta > 0 else ''}{delta:.2f})"
        print(line)
    print(f"  {'out_of_scope_correct':<22} {summary['out_of_scope_correct']:>6}")
    if summary.get("avg_stage_s"):
        print("  avg time per stage:   " + "  ".join(f"{k}={v}" for k, v in summary["avg_stage_s"].items()))
    for t, s in summary["by_type"].items():
        print(f"  [{t}] n={s['n']} pass={fmt(s['pass_rate'])} chunks={fmt(s['chunk_recall'])} "
              f"facts={fmt(s['fact_recall'])} keys={fmt(s['keyword_coverage'])}")

    # ---- save (full runs only, so comparisons stay like-for-like) ----
    if not args.ids:
        out = RUNS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
        out.write_text(json.dumps({"label": args.label, "verify": not args.no_verify,
                                   "summary": summary, "rows": rows}, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
