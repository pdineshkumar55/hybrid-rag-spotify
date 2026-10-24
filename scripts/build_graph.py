"""Build the knowledge graph.

    python scripts/build_graph.py           # LLM extraction + load into Neo4j
    python scripts/build_graph.py --reuse   # skip the LLM, reload the cached extraction
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.ingestion.graph_builder import graph_stats, load_graph
from app.ingestion.graph_extractor import extract_graph, load_extraction, save_extraction
from app.ingestion.loader import load_chunks
from app.neo4j_client import run_query

CACHE = "data/processed/graph_extraction.json"

docs = load_chunks("data/processed/chunks.json")

if "--reuse" in sys.argv:
    print(f"1/2 Loading cached extraction from {CACHE}")
    extraction = load_extraction(CACHE)
else:
    print(f"1/2 Extracting entities with the LLM ({len(docs)} chunks + 1 cross-link pass)")
    extraction = extract_graph(docs)
    save_extraction(extraction, CACHE)
print(f"    {len(extraction['entities'])} entities, {len(extraction['relations'])} relations\n")

print("2/2 Loading into Neo4j Aura")
load_graph(docs, extraction)

nodes, rels = graph_stats()
print("\n✅ Graph built\n\nNodes:")
for n in nodes:
    print(f"   {n['label']:<14} {n['count']}")
print("\nRelationships:")
for r in rels:
    print(f"   {r['type']:<16} {str(r['evidence'] or '-'):<9} {r['count']}")

# ---- Sanity checks: questions vector search alone answers badly ----
print("\nService → Service dependencies:")
for row in run_query("""
    MATCH (a:Spotify:Service)-[r:NOTIFIES|FEEDS|AGGREGATES|LOGS_EVENTS_TO]->(b:Spotify:Service)
    RETURN a.name AS src, type(r) AS rel, b.name AS tgt ORDER BY src LIMIT 10
"""):
    print(f"   {row['src']} -[{row['rel']}]-> {row['tgt']}")

print("\nFeature area → service → tables:")
for row in run_query("""
    MATCH (fa:Spotify:FeatureArea)-[:IMPLEMENTED_BY]->(s:Service)
    OPTIONAL MATCH (s)-[:OWNS_TABLE]->(t:Table)
    RETURN fa.name AS area, s.name AS service, collect(t.name)[..4] AS tables
    ORDER BY area LIMIT 8
"""):
    print(f"   {row['area']} → {row['service']} → {row['tables']}")