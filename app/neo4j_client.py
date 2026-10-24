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
    )
    driver.verify_connectivity()
    return driver


def run_query(query: str, **params) -> list[dict]:
    """Run a Cypher query and return the rows as plain dicts."""
    s = get_settings()
    records, _, _ = get_driver().execute_query(
        query, parameters_=params, database_=s.neo4j_database or None
    )
    return [r.data() for r in records]
