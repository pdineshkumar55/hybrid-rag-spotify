"""Step 5 — The knowledge graph schema (from Step 1) as code.

The LLM may only produce these labels and relationship types; anything else is
dropped during validation, which keeps the graph clean and makes Cypher
labels safe to insert into queries.
"""
import re
from typing import Literal

from pydantic import BaseModel, Field

PROJECT_LABEL = "Spotify"  # every node also gets this label (separates it from the TN data)

# "Closed" labels come from the document's own headings. They are seeded from
# chunk metadata before the LLM runs, and the LLM may not invent new ones.
CLOSED_LABELS = {"Layer", "Service", "PageGroup", "FeatureArea"}

NodeLabel = Literal[
    "Layer", "Technology", "Service", "Table", "Event",
    "Page", "PageGroup", "FeatureArea", "Feature",
]
RelType = Literal[
    "BELONGS_TO", "OWNS_TABLE", "USES", "INTEGRATES_WITH", "NOTIFIES", "FEEDS",
    "AGGREGATES", "LOGS_EVENTS_TO", "STORES", "PART_OF", "IN_GROUP",
    "IMPLEMENTED_BY", "SERVED_BY", "PUBLISHES", "CONSUMES", "CARRIES",
]

# Allowed (source label, target label) pairs for each relationship type
REL_RULES: dict[str, set[tuple[str, str]]] = {
    "BELONGS_TO":      {("Technology", "Layer")},
    "OWNS_TABLE":      {("Service", "Table")},
    "USES":            {("Service", "Technology"), ("Page", "Technology")},
    "INTEGRATES_WITH": {("Service", "Technology")},
    "NOTIFIES":        {("Service", "Service")},
    "FEEDS":           {("Service", "Service")},
    "AGGREGATES":      {("Service", "Service")},
    "LOGS_EVENTS_TO":  {("Service", "Service")},
    "STORES":          {("Technology", "Table"), ("Technology", "Event")},
    "PART_OF":         {("Feature", "FeatureArea")},
    "IN_GROUP":        {("Page", "PageGroup")},
    "IMPLEMENTED_BY":  {("FeatureArea", "Service"), ("Feature", "Service")},
    "SERVED_BY":       {("Page", "Service")},
    "PUBLISHES":       {("Service", "Event")},
    "CONSUMES":        {("Service", "Event")},
    "CARRIES":         {("Technology", "Event")},   # message queue -> event it transports
}


# ---------- Structured output models for the LLM ----------
class Entity(BaseModel):
    label: NodeLabel
    name: str = Field(description="Canonical name, e.g. 'PostgreSQL', 'Media Processor Service', '/home'")
    description: str = Field(description="One short sentence from the text; empty string if none")
    aliases: list[str] = Field(description="Other names used for the same thing; may be empty")


class Relation(BaseModel):
    source_label: NodeLabel
    source_name: str
    type: RelType
    target_label: NodeLabel
    target_name: str
    evidence: Literal["explicit", "inferred"]


class ChunkExtraction(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


class CrossLinks(BaseModel):
    relations: list[Relation]


# ---------- Entity resolution helpers ----------
def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def match_key(label: str, name: str) -> str:
    """Looser key used to spot duplicates, e.g.
    'Catalog Service (Music Metadata)' and 'Catalog' -> 'catalog'."""
    name = re.sub(r"\(.*?\)", "", name)  # drop parentheses
    if label == "Service":
        name = re.sub(r"\bservices?\b", "", name, flags=re.I)
    if label == "FeatureArea":
        name = re.sub(r"^[A-M]\.\s*", "", name)  # "B. Playback" -> "Playback"
    if label == "Page" and name.strip() == "/":
        return "root"                           # the landing page route
    return slug(name)


def make_uid(label: str, name: str) -> str:
    return f"{label.lower()}:{match_key(label, name)}"


def rules_text() -> str:
    """Allowed relationship patterns, written out for the LLM prompt."""
    lines = []
    for rel, pairs in REL_RULES.items():
        for src, tgt in sorted(pairs):
            lines.append(f"- ({src}) -[{rel}]-> ({tgt})")
    return "\n".join(lines)
