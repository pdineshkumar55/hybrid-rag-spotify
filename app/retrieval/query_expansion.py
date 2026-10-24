"""Improvement — query expansion (multi-query retrieval).

Users ask in their own words ("an artist uploads a track"); the document uses
its own vocabulary ("Content Ingestion & Creator Portal", "transcoding uploads",
"track_uploaded"). One small LLM call rewrites the question into a few search
queries plus keywords, so both FAISS and the graph full-text search can match.
"""
from functools import lru_cache

from pydantic import BaseModel, Field

from app.generation.answer import get_chat_llm

EXPAND_SYSTEM = """You help search a software architecture document for a Spotify-like web app.
The document covers: tech stack (frontend, backend, databases, DevOps), 17 microservices
(API Gateway/BFF, Auth & Identity, User Profile, Catalog, Search & Discovery, Library & Playlist,
Playback & Session, Streaming/Media Delivery, Recommendation, Social & Activity,
Subscription & Billing, Ads, Content Ingestion & Creator Portal, Media Processor,
Analytics & Events, Notification, Admin & Backoffice), their database tables, data stores
(PostgreSQL, Redis, Elasticsearch, ClickHouse, S3/CDN, Kafka message queue and events),
frontend routes, and a feature checklist.

Rewrite the user's question into:
- queries: 3 short search queries that use the document's likely wording
  (service names, technical terms). Keep the original meaning; don't answer it.
- keywords: up to 8 key terms or names likely to appear in the relevant sections.
If the question is unrelated to the document, return the question itself as the only
query and an empty keyword list."""


class QueryPlan(BaseModel):
    queries: list[str] = Field(description="3 alternative search queries")
    keywords: list[str] = Field(description="Up to 8 key terms or names")


@lru_cache(maxsize=256)
def expand_query(question: str) -> dict:
    plan: QueryPlan = get_chat_llm().with_structured_output(QueryPlan).invoke([
        ("system", EXPAND_SYSTEM),
        ("human", question),
    ])
    return {
        "queries": [question, *[q for q in plan.queries if q.strip()][:3]],
        "keywords": [k for k in plan.keywords if k.strip()][:8],
    }
