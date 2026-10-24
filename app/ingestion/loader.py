"""Step 3 — Load the PDF and split it into section-aware chunks.

Every chunk gets a stable chunk_id (e.g. "spotify-s02-04") that is shared by
FAISS (Step 4) and Neo4j (Step 5), so the two stores can be joined later.
"""
import json
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

PROJECT = "spotify"
MAX_CHARS = 1200  # chunks longer than this get split further

# Header patterns found in this document
SECTION_RE = re.compile(r"^(\d+)\.\s+(.+)$")          # "2. Microservices (High-Level)"
UNDERLINE_RE = re.compile(r"^[-=]{3,}$")              # "-----" / "====="
SERVICE_RE = re.compile(r"^(\d+)\)\s+(.+)$")          # "4) Catalog Service (Music Metadata)"
FEATURE_AREA_RE = re.compile(r"^([A-M])\.\s+(.+)$")   # "B. Playback & Player"
GROUP_RE = re.compile(r"^-?\s*([A-Z][^:]{1,60}):$")   # "Frontend:", "- Public:", "Redis:"
SUMMARY_RE = re.compile(r"^This document summarizes")


def read_pdf_lines(pdf_path: str) -> list[tuple[int, str]]:
    """Return (page_number, line) for every non-empty line."""
    reader = PdfReader(pdf_path)
    lines = []
    for page_no, page in enumerate(reader.pages, start=1):
        for raw in (page.extract_text() or "").splitlines():
            line = raw.strip()
            if line:
                lines.append((page_no, line))
    return lines


def parse_sections(lines: list[tuple[int, str]]) -> list[dict]:
    """Walk the lines and group them into (section, subsection) blocks."""
    blocks: list[dict] = []
    section_no, section_title = 0, "Document Title"
    current = None

    def start_block(sub_title: str, page: int):
        nonlocal current
        current = {
            "section_no": section_no,
            "section": section_title,
            "subsection": sub_title,
            "page": page,
            "lines": [],
        }
        blocks.append(current)

    for i, (page, line) in enumerate(lines):
        nxt = lines[i + 1][1] if i + 1 < len(lines) else ""

        if UNDERLINE_RE.match(line):
            continue

        # Top-level section: "N. Title" followed by an underline
        m = SECTION_RE.match(line)
        if m and UNDERLINE_RE.match(nxt):
            section_no, section_title = int(m.group(1)), m.group(2).strip()
            start_block("Overview", page)
            continue

        if SUMMARY_RE.match(line):
            section_no, section_title = 99, "Document Summary"
            start_block("Summary", page)
            current["lines"].append(line)
            continue

        # Subsection headers (which pattern applies depends on the section)
        sub = None
        if section_no == 2 and (m := SERVICE_RE.match(line)):
            sub = f"{m.group(1)}) {m.group(2).strip()}"
        elif section_no == 5 and (m := FEATURE_AREA_RE.match(line)):
            sub = f"{m.group(1)}. {m.group(2).strip()}"
        elif section_no in (1, 3, 4) and (m := GROUP_RE.match(line)):
            sub = m.group(1).strip()

        if sub:
            start_block(sub, page)
            continue

        if current is None:  # title line before section 1
            start_block("Title", page)
        current["lines"].append(line)

    # Drop empty blocks (e.g. "Key pages/routes:" with nothing under it)
    return [b for b in blocks if b["lines"]]


def build_chunks(pdf_path: str) -> list[Document]:
    blocks = parse_sections(read_pdf_lines(pdf_path))
    splitter = RecursiveCharacterTextSplitter(chunk_size=MAX_CHARS, chunk_overlap=100)

    docs: list[Document] = []
    counters: dict[int, int] = {}
    for b in blocks:
        if b["section_no"] == 0:  # the title alone isn't worth embedding
            continue
        body = "\n".join(b["lines"])
        # A context header makes each chunk self-describing for embeddings
        header = f"[{b['section_no']}. {b['section']} > {b['subsection']}]"
        pieces = splitter.split_text(body) if len(body) > MAX_CHARS else [body]

        for piece in pieces:
            counters[b["section_no"]] = counters.get(b["section_no"], 0) + 1
            chunk_id = f"{PROJECT}-s{b['section_no']:02d}-{counters[b['section_no']]:02d}"
            docs.append(
                Document(
                    page_content=f"{header}\n{piece}",
                    metadata={
                        "chunk_id": chunk_id,
                        "project": PROJECT,
                        "section_no": b["section_no"],
                        "section": b["section"],
                        "subsection": b["subsection"],
                        "page": b["page"],
                        "source": Path(pdf_path).name,
                    },
                )
            )
    return docs


def save_chunks(docs: list[Document], out_path: str = "data/processed/chunks.json") -> None:
    """Save chunks so Step 4 (FAISS) and Step 5 (Neo4j) use the exact same IDs."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    Path(out_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_chunks(path: str = "data/processed/chunks.json") -> list[Document]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data]
