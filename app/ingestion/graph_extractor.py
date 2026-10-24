"""Step 5 — Use the LLM to extract entities and relationships.

Pass 0: seed the "closed" entities (17 Services, 13 Feature Areas, 4 Layers,
        4 Page Groups) straight from the chunk headings — no LLM needed.
Pass 1: one LLM call per chunk -> entities + relationships stated in that chunk.
Pass 2: structural edges from the headings (Technology BELONGS_TO Layer,
        Feature PART_OF FeatureArea, Page IN_GROUP PageGroup).
Pass 3: one "cross-link" LLM call for links that span sections
        (Feature Area -> Service, Page -> Service, Service -> Event).
Everything is validated against graph_schema; rejected items are kept in
the "dropped" list of the JSON cache so you can see what the LLM got wrong.
"""
import json
import re
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.ingestion.graph_schema import (
    CLOSED_LABELS, REL_RULES, ChunkExtraction, CrossLinks, make_uid, match_key,
    rules_text, slug,
)

EXTRACT_SYSTEM = """You extract a knowledge graph from a software architecture document
for a Spotify-like web app. Extract ONLY entities that literally appear in THIS chunk.

Node labels:
- Technology: frameworks, libraries, databases, tools, cloud products, payment gateways.
  If a line lists alternatives with "/" (e.g. "Kafka/NATS/RabbitMQ"), create one Technology each.
- Table: database table names exactly as written (snake_case), e.g. "playlist_tracks".
- Event: message-queue events (e.g. "track_uploaded") or tracked user-event types
  (e.g. "playback events", "skips"). Metrics such as "daily active users" are NOT events.
- Page: a frontend route; name = the path, e.g. "/playlist/:id".
- Feature: an individual checklist item (only in the Feature Checklist section).
- Service, FeatureArea, Layer, PageGroup: FIXED lists below. Never invent new ones;
  refer to them only by these exact names.
If something fits no label, skip it.

Allowed relationships (source label -> target label). Use no other combinations:
{rules}
evidence = "explicit" when the text states it, "inferred" when it is a clear implication.

The chunk header in [brackets] tells you the section and subsection.
Known names (reuse them exactly when you mean the same thing):
{known}"""

CROSSLINK_SYSTEM = """You link parts of a Spotify-like web app architecture.
Using ONLY the names listed below (copy them exactly), return relationships:
- FeatureArea IMPLEMENTED_BY Service (each feature area -> the 1-3 services that implement it)
- Page SERVED_BY Service (each page -> the main 1-2 services that provide its data)
- Service PUBLISHES Event and Service CONSUMES Event (message-queue events only)
Mark every relationship evidence = "inferred". Skip anything you are unsure about.

{catalog}"""

# Which closed label a chunk heading defines, per section number
HEADING_LABEL = {1: "Layer", 2: "Service", 4: "PageGroup", 5: "FeatureArea"}
# Structural edges derived from headings: section -> (member label, rel type)
STRUCTURAL = {1: ("Technology", "BELONGS_TO"), 4: ("Page", "IN_GROUP"), 5: ("Feature", "PART_OF")}
NOT_A_GROUP = {"Global UI components"}


def heading_entity(meta: dict) -> tuple[str, str, list[str]] | None:
    """(label, canonical name, aliases) for the closed entity a chunk heading defines."""
    label = HEADING_LABEL.get(meta["section_no"])
    sub = meta["subsection"]
    if not label or sub in NOT_A_GROUP:
        return None
    name = re.sub(r"^(\d+\)|[A-M]\.)\s*", "", sub).strip()  # drop "4) " / "B. "
    base = re.sub(r"\(.*?\)", "", name)
    aliases = [p.strip() for p in base.split("/") if p.strip() and p.strip() != name]
    aliases += re.findall(r"\((.*?)\)", name)  # e.g. "BFF", "Music Metadata"
    return label, name, aliases


def in_text(name: str, chunk_text: str) -> bool:
    """True if the entity name appears in the chunk (ignoring case and punctuation)."""
    key = slug(name)
    if not key:  # purely symbolic names such as the "/" route
        return name.strip() in chunk_text
    return key in slug(chunk_text)


class GraphAccumulator:
    """Collects entities across chunks, merges duplicates, records what it rejects."""

    def __init__(self):
        self.entities: dict[str, dict] = {}      # uid -> entity
        self.alias_index: dict[tuple, str] = {}  # (label, match_key) -> uid
        self.relations: dict[tuple, dict] = {}   # (src_uid, type, tgt_uid) -> relation
        self.mentions: dict[str, set] = {}       # chunk_id -> {uid}
        self.dropped: list[dict] = []

    # ---------- entities ----------
    def resolve(self, label: str, name: str, fuzzy: bool = True) -> str | None:
        uid = self.alias_index.get((label, match_key(label, name)))
        if uid or not fuzzy or label not in CLOSED_LABELS:
            return uid
        # Closed labels: allow a short form, e.g. "Recommendation Service" ->
        # "Recommendation & Personalization Service" (only if exactly one candidate fits)
        words = set(match_key(label, name).split("-")) - {"and", ""}
        if not words:
            return None
        hits = {
            e["uid"] for e in self.entities.values()
            if e["label"] == label
            and words <= set(match_key(label, e["name"]).split("-"))
        }
        return hits.pop() if len(hits) == 1 else None

    def resolve_many(self, label: str, name: str) -> list[str]:
        """Like resolve, but also splits lists such as "Catalog/Streaming services"."""
        uid = self.resolve(label, name)
        if uid:
            return [uid]
        if label in CLOSED_LABELS and "/" in name:
            parts = [self.resolve(label, p) for p in name.split("/")]
            if parts and all(parts):
                return list(dict.fromkeys(parts))
        return []

    def _drop(self, chunk_id, item, reason):
        self.dropped.append({"chunk_id": chunk_id, "item": item, "reason": reason})
        return None

    def add_entity(self, label, name, description="", aliases=(), chunk_id=None,
                   seed=False, chunk_text=None, allow_feature=True) -> str | None:
        name = (name or "").strip()
        item = f"{label}:{name}"
        if not match_key(label, name):
            return self._drop(chunk_id, item, "empty name")

        uid = self.resolve(label, name, fuzzy=not seed) or next(
            (self.resolve(label, a, fuzzy=not seed) for a in aliases
             if self.resolve(label, a, fuzzy=not seed)), None
        )
        if uid is None and label in CLOSED_LABELS and not seed:
            many = self.resolve_many(label, name)  # e.g. "Catalog/Streaming services"
            if many:
                if chunk_id:
                    self.mentions.setdefault(chunk_id, set()).update(many)
                return many[0]
            return self._drop(chunk_id, item, f"unknown {label} (closed list)")
        if uid is None:
            if label == "Feature" and not allow_feature:
                return self._drop(chunk_id, item, "Feature outside the checklist section")
            if chunk_text is not None and not in_text(name, chunk_text):
                return self._drop(chunk_id, item, "not found in chunk text")
            uid = make_uid(label, name)
            self.entities[uid] = {
                "uid": uid, "label": label, "name": name,
                "description": (description or "").strip(), "aliases": [],
            }

        ent = self.entities[uid]
        if description and len(description) > len(ent["description"]):
            ent["description"] = description.strip()
        for alt in [name, *aliases]:
            if alt and alt != ent["name"] and alt not in ent["aliases"]:
                ent["aliases"].append(alt)
            if match_key(label, alt):
                self.alias_index.setdefault((label, match_key(label, alt)), uid)
        if chunk_id:
            self.mentions.setdefault(chunk_id, set()).add(uid)
        return uid

    # ---------- relations ----------
    def link(self, src: str, rel_type: str, tgt: str, evidence: str, chunk_id=None) -> bool:
        if src == tgt:
            return False
        key = (src, rel_type, tgt)
        existing = self.relations.get(key)
        if existing is None:
            self.relations[key] = {
                "source": src, "type": rel_type, "target": tgt,
                "evidence": evidence, "chunk_ids": [chunk_id] if chunk_id else [],
            }
        else:
            if evidence == "explicit":
                existing["evidence"] = "explicit"  # explicit wins over inferred
            if chunk_id and chunk_id not in existing["chunk_ids"]:
                existing["chunk_ids"].append(chunk_id)
        return True

    def add_relation(self, rel, chunk_id=None, chunk_text=None, allow_feature=True,
                     create=True) -> bool:
        item = f"({rel.source_label}:{rel.source_name})-[{rel.type}]->({rel.target_label}:{rel.target_name})"
        allowed = REL_RULES.get(rel.type, set())
        s_label, s_name, t_label, t_name = rel.source_label, rel.source_name, rel.target_label, rel.target_name
        if (s_label, t_label) not in allowed:
            if (t_label, s_label) in allowed:  # LLM got the direction backwards -> swap
                s_label, s_name, t_label, t_name = t_label, t_name, s_label, s_name
            else:
                self._drop(chunk_id, item, "label pair not allowed for this relationship")
                return False

        def endpoints(label, name) -> list[str]:
            uids = self.resolve_many(label, name)
            if uids or not create:
                return uids
            uid = self.add_entity(label, name, chunk_id=chunk_id,
                                  chunk_text=chunk_text, allow_feature=allow_feature)
            return [uid] if uid else []

        srcs, tgts = endpoints(s_label, s_name), endpoints(t_label, t_name)
        if not srcs or not tgts:
            self._drop(chunk_id, item, "endpoint unknown or rejected")
            return False
        for src in srcs:
            for tgt in tgts:
                self.link(src, rel.type, tgt, rel.evidence, chunk_id)
                if chunk_id:  # the chunk mentions both ends
                    self.mentions.setdefault(chunk_id, set()).update({src, tgt})
        return True

    # ---------- helpers ----------
    def known_names(self) -> str:
        by_label: dict[str, list[str]] = {}
        for e in self.entities.values():
            by_label.setdefault(e["label"], []).append(e["name"])
        return "\n".join(f"- {k}: {', '.join(sorted(v))}" for k, v in sorted(by_label.items()))

    def to_json(self) -> dict:
        return {
            "entities": list(self.entities.values()),
            "relations": list(self.relations.values()),
            "mentions": {k: sorted(v) for k, v in self.mentions.items()},
            "dropped": self.dropped,
        }


def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model, api_key=s.openai_api_key, temperature=0)


def extract_graph(docs: list[Document], llm=None, verbose: bool = True) -> dict:
    llm = llm or get_llm()
    chunk_llm = llm.with_structured_output(ChunkExtraction)
    link_llm = llm.with_structured_output(CrossLinks)
    acc = GraphAccumulator()

    # ---- Pass 0: seed closed entities from headings ----
    heading_uid: dict[str, str] = {}  # chunk_id -> uid of its heading entity
    for d in docs:
        h = heading_entity(d.metadata)
        if h:
            label, name, aliases = h
            heading_uid[d.metadata["chunk_id"]] = acc.add_entity(
                label, name, aliases=aliases, chunk_id=d.metadata["chunk_id"], seed=True)
    if verbose:
        seeded = Counter(acc.entities[u]["label"] for u in set(heading_uid.values()))
        print(f"  seeded from headings: {dict(seeded)}")

    # ---- Pass 1: per chunk ----
    system_rules = rules_text()
    for d in docs:
        cid, meta = d.metadata["chunk_id"], d.metadata
        chunk_text = d.page_content
        is_checklist = meta["section_no"] == 5
        result: ChunkExtraction = chunk_llm.invoke([
            ("system", EXTRACT_SYSTEM.format(rules=system_rules, known=acc.known_names())),
            ("human", d.page_content),
        ])
        ent_kept = sum(
            acc.add_entity(e.label, e.name, e.description, e.aliases, chunk_id=cid,
                           chunk_text=chunk_text, allow_feature=is_checklist) is not None
            for e in result.entities
        )
        rel_kept = sum(
            acc.add_relation(r, chunk_id=cid, chunk_text=chunk_text, allow_feature=is_checklist)
            for r in result.relations
        )
        if verbose:
            print(f"  {cid:<16} entities {ent_kept:>3}/{len(result.entities):<3} "
                  f"relations {rel_kept:>3}/{len(result.relations)}")

    # ---- Pass 2: structural edges from headings ----
    added = 0
    for d in docs:
        cid, sec = d.metadata["chunk_id"], d.metadata["section_no"]
        if sec in STRUCTURAL and cid in heading_uid:
            member_label, rel_type = STRUCTURAL[sec]
            for uid in acc.mentions.get(cid, ()):
                if acc.entities[uid]["label"] == member_label:
                    added += acc.link(uid, rel_type, heading_uid[cid], "explicit", cid)
    if verbose:
        print(f"  structural edges   {added} (BELONGS_TO / IN_GROUP / PART_OF)")

    # ---- Pass 3: cross-section links ----
    catalog_lines = []
    for label in ("FeatureArea", "Page", "Service", "Event"):
        catalog_lines.append(f"{label}:")
        for e in acc.entities.values():
            if e["label"] == label:
                catalog_lines.append(f"  - {e['name']}: {e['description']}")
    links: CrossLinks = link_llm.invoke([
        ("system", CROSSLINK_SYSTEM.format(catalog="\n".join(catalog_lines))),
        ("human", "Return the relationships."),
    ])
    kept = 0
    for r in links.relations:
        r.evidence = "inferred"
        kept += acc.add_relation(r, create=False)  # must point at existing entities
    if verbose:
        print(f"  cross-links        {kept}/{len(links.relations)} relations kept")
        if acc.dropped:
            reasons = Counter(x["reason"] for x in acc.dropped)
            print(f"  dropped items      {dict(reasons)}")

    return acc.to_json()


def save_extraction(data: dict, path: str = "data/processed/graph_extraction.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_extraction(path: str = "data/processed/graph_extraction.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
