"""Step 5 — Write the extracted graph into Neo4j Aura.

Everything is created under the :Spotify label, so the reset and all queries
only touch this project's nodes — your other app's data in the same Aura
instance is left alone.
"""
from typing import get_args

from langchain_core.documents import Document

from app.ingestion.graph_schema import PROJECT_LABEL, NodeLabel, RelType
from app.neo4j_client import run_query

P = PROJECT_LABEL
ALLOWED_LABELS = set(get_args(NodeLabel))
ALLOWED_RELS = set(get_args(RelType))


def setup_schema() -> None:
    # One uniqueness constraint, scoped to this project's nodes only
    run_query(f"CREATE CONSTRAINT spotify_uid IF NOT EXISTS FOR (n:{P}) REQUIRE n.uid IS UNIQUE")
    # Full-text index used in Step 6 to find entities mentioned in a question.
    # The 'english' analyzer stems words, so "uploads" also matches "upload".
    run_query("DROP INDEX spotify_entity_text IF EXISTS")
    run_query(
        f"CREATE FULLTEXT INDEX spotify_entity_text "
        f"FOR (n:{P}) ON EACH [n.name, n.aliases_text, n.description] "
        "OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}"
    )


def reset_project_graph() -> int:
    rows = run_query(f"MATCH (n:{P}) WITH count(n) AS c RETURN c")
    run_query(f"MATCH (n:{P}) DETACH DELETE n")
    return rows[0]["c"]


def write_chunks(docs: list[Document]) -> None:
    rows = [
        {
            "uid": f"chunk:{d.metadata['chunk_id']}",
            "chunk_id": d.metadata["chunk_id"],
            "text": d.page_content,
            "section": d.metadata["section"],
            "subsection": d.metadata["subsection"],
            "page": d.metadata["page"],
        }
        for d in docs
    ]
    run_query(
        f"""
        UNWIND $rows AS row
        MERGE (c:{P} {{uid: row.uid}})
        SET c:Chunk, c.chunk_id = row.chunk_id, c.text = row.text,
            c.section = row.section, c.subsection = row.subsection, c.page = row.page
        """,
        rows=rows,
    )


def write_entities(entities: list[dict]) -> None:
    by_label: dict[str, list[dict]] = {}
    for e in entities:
        if e["label"] in ALLOWED_LABELS:
            by_label.setdefault(e["label"], []).append(
                {**e, "aliases_text": " | ".join(e["aliases"])}
            )
    for label, rows in by_label.items():
        # Labels can't be query parameters; they come from the whitelist above.
        run_query(
            f"""
            UNWIND $rows AS row
            MERGE (n:{P} {{uid: row.uid}})
            SET n:{label}, n.name = row.name, n.description = row.description,
                n.aliases = row.aliases, n.aliases_text = row.aliases_text
            """,
            rows=rows,
        )


def write_relations(relations: list[dict]) -> None:
    by_type: dict[str, list[dict]] = {}
    for r in relations:
        if r["type"] in ALLOWED_RELS:
            by_type.setdefault(r["type"], []).append(r)
    for rel_type, rows in by_type.items():
        run_query(
            f"""
            UNWIND $rows AS row
            MATCH (a:{P} {{uid: row.source}})
            MATCH (b:{P} {{uid: row.target}})
            MERGE (a)-[r:{rel_type}]->(b)
            SET r.evidence = row.evidence, r.chunk_ids = row.chunk_ids
            """,
            rows=rows,
        )


def write_mentions(mentions: dict[str, list[str]]) -> None:
    rows = [
        {"chunk_uid": f"chunk:{cid}", "entity_uid": uid}
        for cid, uids in mentions.items()
        for uid in uids
    ]
    run_query(
        f"""
        UNWIND $rows AS row
        MATCH (c:{P} {{uid: row.chunk_uid}})
        MATCH (e:{P} {{uid: row.entity_uid}})
        MERGE (c)-[:MENTIONS]->(e)
        """,
        rows=rows,
    )


def load_graph(docs: list[Document], extraction: dict, reset: bool = True) -> None:
    setup_schema()
    if reset:
        removed = reset_project_graph()
        print(f"  removed {removed} old :{P} nodes (other projects untouched)")
    write_chunks(docs)
    write_entities(extraction["entities"])
    write_relations(extraction["relations"])
    write_mentions(extraction["mentions"])


def graph_stats() -> tuple[list[dict], list[dict]]:
    nodes = run_query(
        f"""
        MATCH (n:{P})
        UNWIND [l IN labels(n) WHERE l <> '{P}'] AS label
        RETURN label, count(*) AS count ORDER BY count DESC
        """
    )
    rels = run_query(
        f"""
        MATCH (:{P})-[r]->(:{P})
        RETURN type(r) AS type, r.evidence AS evidence, count(*) AS count
        ORDER BY type, evidence
        """
    )
    return nodes, rels
