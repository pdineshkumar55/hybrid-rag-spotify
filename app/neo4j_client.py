"""Shared Neo4j driver (used by graph building in Step 5 and retrieval in Step 6)."""
from functools import lru_cache

from neo4j import GraphDatabase, NotificationMinimumSeverity

from app.config import get_settings


@lru_cache
def get_driver():
    s = get_settings()
    driver = GraphDatabase.driver(
        s.neo4j_uri,
        auth=(s.neo4j_username, s.neo4j_password),
        # hide harmless server warnings (e.g. "relationship type does not exist yet")
        notifications_min_severity=NotificationMinimumSeverity.OFF,
        # Fail fast instead of hanging, and never reuse connections that Aura has
        # silently closed (e.g. after the laptop slept or the API sat idle)
        connection_timeout=10,              # seconds to open a new connection
        connection_acquisition_timeout=15,  # seconds to wait for a pooled connection
        max_connection_lifetime=300,        # recycle connections every 5 minutes
        liveness_check_timeout=30,          # re-check connections idle for >30 s before use
        keep_alive=True,
    )
    driver.verify_connectivity()
    return driver


def run_query(query: str, **params) -> list[dict]:
    """Run a Cypher query and return the rows as plain dicts."""
    s = get_settings()
    records, _, _ = get_driver().execute_query(query, parameters_=params, database_=s.neo4j_database)
    return [r.data() for r in records]
