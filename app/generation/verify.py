"""Step 8b — Verify the answer against the same context.

Two checks:
1. Rule-based: every citation label in the answer must exist in the context.
2. LLM judge: splits the answer into claims and checks each against the cited
   sources; returns a verdict, a confidence and, if needed, a corrected answer.
"""
from typing import Literal

from pydantic import BaseModel, Field

from app.generation.answer import CITATION_RE, NO_ANSWER, extract_citations, get_chat_llm

VERIFY_SYSTEM = """You are a strict fact-checker for a retrieval-augmented answer.
You get the CONTEXT (graph query results [Q#], passages [C#], graph facts [F#], entity summaries [E#]),
the QUESTION and the ANSWER.

1. Split the answer into its individual factual claims.
2. For each claim decide:
   - "supported": the context states it (an "inferred" graph fact counts only if
     the answer presents it as likely/suggested)
   - "unsupported": the context does not state it
   - "contradicted": the context says otherwise
3. verdict: "supported" if all claims are supported, "partially_supported" if some
   are, "not_supported" if none are.
4. confidence: "high", "medium" or "low" that the answer is correct and complete.
5. corrected_answer: if any claim is unsupported or contradicted, rewrite the answer
   using only supported content, keeping the [Q#]/[C#]/[F#]/[E#] citations.
   Otherwise return an empty string.
Judge only against the context, never against outside knowledge."""


class ClaimCheck(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "contradicted"]
    citations: list[str] = Field(description="Labels such as C1, F2 that support or contradict it")
    note: str = Field(description="Short reason; empty string if supported")


class Verification(BaseModel):
    verdict: Literal["supported", "partially_supported", "not_supported"]
    confidence: Literal["high", "medium", "low"]
    claims: list[ClaimCheck]
    corrected_answer: str


def check_citations(answer: str, context: str) -> dict:
    used = extract_citations(answer)
    available = set(extract_citations(context))
    return {
        "used": used,
        "invalid": [c for c in used if c not in available],
        "has_citations": bool(used),
    }


def can_skip_verification(answer: str, context: str, retrieval: dict, min_score: float) -> tuple[bool, str]:
    """Cheap checks (no LLM). All must hold to skip the verification call:
    1. every citation label exists in the context, and there is at least one
    2. every content line of the answer carries a citation
    3. no "inferred" graph fact is cited (those are the riskiest claims)
    4. retrieval was confident: a strong FAISS match, or a chunk found by BOTH retrievers
    """
    if answer.strip() == NO_ANSWER:
        return True, "no-answer reply"
    cites = check_citations(answer, context)
    if not cites["has_citations"] or cites["invalid"]:
        return False, "missing or invalid citations"

    for line in answer.splitlines():
        text = line.strip().strip("-*• ").strip()
        if len(text) < 25 or text.endswith(":"):   # headings / short connective lines
            continue
        if not CITATION_RE.search(text):
            return False, "a sentence has no citation"

    inferred = {f"F{i}" for i, f in enumerate(retrieval["facts"], start=1) if f["evidence"] == "inferred"}
    if inferred & set(cites["used"]):
        return False, "cites an inferred graph fact"

    if "Q1" in cites["used"]:
        return True, "answer is based on an exact graph query result"

    chunks = retrieval["chunks"]
    best = max((c["vector_score"] or 0) for c in chunks) if chunks else 0
    both = any(set(c["sources"]) == {"vector", "graph"} for c in chunks)
    if best < min_score and not both:
        return False, f"weak retrieval (best score {best:.2f})"
    return True, f"grounded (best score {best:.2f}{', vector+graph agree' if both else ''})"


def verify_answer(question: str, context: str, answer: str, llm=None) -> dict:
    citations = check_citations(answer, context)

    if answer.strip() == NO_ANSWER:
        return {
            "verdict": "no_answer", "confidence": "high", "claims": [],
            "corrected_answer": "", "citations": citations,
        }

    llm = llm or get_chat_llm()
    judge = llm.with_structured_output(Verification)
    result: Verification = judge.invoke([
        ("system", VERIFY_SYSTEM),
        ("human", f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER:\n{answer}"),
    ])
    out = result.model_dump()

    # Rule-based checks can only lower confidence, never raise it
    if citations["invalid"] or not citations["has_citations"]:
        out["confidence"] = "low"
    out["citations"] = citations
    return out
