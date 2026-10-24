"""Ask a question end to end from the terminal.

    python scripts/ask.py "What happens after an artist uploads a track?"
    python scripts/ask.py --no-verify "Which services own the playlists table?"
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.generation.answer import extract_citations
from app.pipeline import ask

args = sys.argv[1:]
verify = "--no-verify" not in args
question = " ".join(a for a in args if a != "--no-verify") or \
    "What happens after an artist uploads a track?"

r = ask(question, verify=verify)

print(f"Q: {r['question']}\n")
print("ANSWER" + (" (corrected by verifier)" if r["was_corrected"] else "") + ":")
print(r["answer"])

v = r["verification"]
if v:
    print(f"\nVERIFICATION: {v['verdict']}  |  confidence: {v['confidence']}")
    if v["citations"]["invalid"]:
        print(f"   invalid citations: {v['citations']['invalid']}")
    for c in v["claims"]:
        mark = {"supported": "✓", "unsupported": "?", "contradicted": "✗"}[c["status"]]
        note = f"  — {c['note']}" if c["note"] else ""
        print(f"   {mark} {c['claim']}  {c['citations']}{note}")
    if r["was_corrected"]:
        print("\nORIGINAL DRAFT:\n" + r["draft_answer"])

used = set(extract_citations(r["answer"]))
print("\nSOURCES CITED:")
for s in r["sources"]:
    if s["label"] in used:
        where = s.get("title") or s["type"]
        print(f"   [{s['label']}] {where}: {s['text'][:110].replace(chr(10), ' ')}")

print(f"\nTimings: {r['timings']}")
