"""Improvement — text-to-Cypher for counting / ranking / "list all" questions.

Neighbourhood expansion (graph_retriever.py) answers "what is connected to X".
Questions like "How many services ...?", "Which service owns the most tables?"
need an aggregate query over the whole graph, so the LLM writes one.

Safety: the generated query is validated (read-only keywords only, must use the
:Spotify label, LIMIT enforced) and executed in a READ transaction with a timeout,
so it cannot change data or touch the other project in the same Aura instance.
"""
import re

from neo4j import READ_ACCESS, unit_of_work
from pydantic import BaseModel, Field

from app.config import get_settings
from app.generation.answer import get_chat_llm
from app.ingestion.graph_schema import REL_RULES
from app.neo4j_client import get_driver

MAX_ROWS = 50
QUERY_TIMEOUT_S = 5.0

# Questions that need a whole-graph query rather than a neighbourhood lookup
AGGREGATE_RE = re.compile(
    r"\b(how many|count|number of|most|least|fewest|more than|less than|fewer than|at least|"
    r"at most|top \d+|rank|ranking|largest|smallest|biggest|every|list all|all (the )?"
    r"(services|tables|pages|routes|features|technologies|events|layers))\b",
    re.IGNORECASE,
)

FORBIDDEN_RE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL|GRANT|DENY|REVOKE)\b",
    re.IGNORECASE,
)


def _schema_text() -> str:
    rels = "\n".join(f"  (:{s})-[:{r}]->(:{t})" for r, pairs in REL_RULES.items() for s, t in sorted(pairs))
    return f"""Node labels (EVERY node also has the label :Spotify — always include it):
  Layer, Technology, Service, Table, Event, Page, PageGroup, FeatureArea, Feature, Chunk
Node properties: name (string), description (string), aliases (list of strings)
Chunk properties: chunk_id, text, section, subsection, page
Relationships (all also have properties: evidence = 'explicit' | 'inferred'):
{rels}
  (:Chunk)-[:MENTIONS]->(any entity)"""


CYPHER_SYSTEM = """You write ONE read-only Cypher query for a Neo4j knowledge graph that describes
the architecture of a Spotify-like web app.

{schema}

Rules:
- Use only MATCH / OPTIONAL MATCH / WHERE / WITH / RETURN / ORDER BY / LIMIT / UNWIND / aggregations.
- Every node pattern must include the :Spotify label, e.g. (s:Spotify:Service).
- Match names loosely: WHERE toLower(n.name) CONTAINS 'postgres'.
- Return readable columns (names and counts), never whole nodes. LIMIT {max_rows} or less.
- If the question cannot be answered from this schema, set answerable = false.

Examples:
Q: How many microservices are there?
MATCH (s:Spotify:Service) RETURN count(s) AS services
Q: Which service owns the most tables?
MATCH (s:Spotify:Service)-[:OWNS_TABLE]->(t:Spotify:Table)
RETURN s.name AS service, count(t) AS tables ORDER BY tables DESC LIMIT 5
Q: Which services own tables stored in PostgreSQL?
MATCH (db:Spotify:Technology)-[:STORES]->(t:Spotify:Table)<-[:OWNS_TABLE]-(s:Spotify:Service)
WHERE toLower(db.name) CONTAINS 'postgres'
RETURN s.name AS service, collect(DISTINCT t.name) AS tables"""


class CypherPlan(BaseModel):
    answerable: bool
    cypher: str = Field(description="The Cypher query, or empty string if not answerable")
    explanation: str = Field(description="One sentence: what the query computes")


def needs_graph_query(question: str) -> bool:
    return bool(AGGREGATE_RE.search(question))


def validate_cypher(cypher: str) -> str:
    """Return a safe query or raise ValueError."""
    q = cypher.strip().rstrip(";").strip()
    if not q:
        raise ValueError("empty query")
    if ";" in q:
        raise ValueError("multiple statements are not allowed")
    if FORBIDDEN_RE.search(q):
        raise ValueError("only read-only queries are allowed")
    if ":Spotify" not in q:
        raise ValueError("every node pattern must use the :Spotify label")
    if not re.search(r"\bLIMIT\s+\d+\b", q, re.IGNORECASE):
        q += f"\nLIMIT {MAX_ROWS}"
    return q


def _run_read(cypher: str) -> list[dict]:
    """Execute inside a READ transaction (writes are rejected by Neo4j) with a server-side timeout."""
    @unit_of_work(timeout=QUERY_TIMEOUT_S)
    def work(tx):
        return [r.data() for r in tx.run(cypher)][:MAX_ROWS]

    s = get_settings()
    with get_driver().session(database=s.neo4j_database or None,
                              default_access_mode=READ_ACCESS) as session:
        return session.execute_read(work)


def text_to_cypher(question: str, max_attempts: int = 2) -> dict:
    """{'cypher', 'explanation', 'rows', 'error'} — rows is None when nothing usable came back."""
    llm = get_chat_llm().with_structured_output(CypherPlan)
    system = CYPHER_SYSTEM.format(schema=_schema_text(), max_rows=MAX_ROWS)
    messages = [("system", system), ("human", question)]
    last = {"cypher": "", "explanation": "", "rows": None, "error": None}

    for _ in range(max_attempts):
        plan: CypherPlan = llm.invoke(messages)
        if not plan.answerable or not plan.cypher.strip():
            return {**last, "error": "not answerable from the graph schema"}
        try:
            cypher = validate_cypher(plan.cypher)
            rows = _run_read(cypher)
            return {"cypher": cypher, "explanation": plan.explanation, "rows": rows, "error": None}
        except Exception as exc:  # noqa: BLE001 — feed the error back and let the LLM fix the query
            last = {"cypher": plan.cypher, "explanation": plan.explanation, "rows": None,
                    "error": f"{type(exc).__name__}: {exc}"}
            messages += [("ai", plan.cypher),
                         ("human", f"That query failed: {last['error']}. Write a corrected query.")]
    return last
