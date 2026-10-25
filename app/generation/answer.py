"""Step 8a — Generate a cited answer from the hybrid context."""
import re
from functools import lru_cache

from langchain_openai import ChatOpenAI

from app.config import get_settings

NO_ANSWER = "The document doesn't contain information to answer this question."

ANSWER_SYSTEM = """You are an assistant that answers questions about a software
architecture document for a Spotify-like web app.

Rules:
- Use ONLY the context below: graph query results [Q#], document passages [C#],
  knowledge graph facts [F#] and entity summaries [E#]. Do not use outside knowledge.
- [Q#] results are exact counts/lists computed over the whole graph: use them for
  "how many", "which ... most", "list all" questions, and cite them.
- After every sentence or bullet, cite the labels that support it, e.g. [C1][F3].
- Prefer passages [C#] for descriptions and facts [F#] for relationships
  (which service owns which table, what notifies what, etc.).
- Facts marked "inferred" were derived by the system, not stated in the document;
  if you rely on one, say "likely" or "the design suggests".
- If NOTHING in the context answers the question, reply exactly:
  "{no_answer}"
  If the context answers it even partly, answer what you can and never add that sentence.
- When a passage lists specific items (tables, events, routes, technologies),
  include every item it lists rather than summarising them away.
- Be concise. Use short bullets for lists or flows."""


@lru_cache
def get_chat_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.openai_chat_model, api_key=s.openai_api_key, temperature=0)


def _messages(question: str, context: str) -> list:
    return [
        ("system", ANSWER_SYSTEM.format(no_answer=NO_ANSWER)),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ]


def clean_answer(text: str) -> str:
    """Safety net: drop the refusal sentence if the model appended it to a real answer."""
    text = text.strip()
    if NO_ANSWER in text and text != NO_ANSWER:
        text = text.replace(NO_ANSWER, "").strip()
    return text


def generate_answer(question: str, context: str, llm=None) -> str:
    if not context.strip():
        return NO_ANSWER
    llm = llm or get_chat_llm()
    return clean_answer(llm.invoke(_messages(question, context)).content)


def stream_answer(question: str, context: str, llm=None):
    """Yield the answer piece by piece as the model writes it."""
    if not context.strip():
        yield NO_ANSWER
        return
    llm = llm or get_chat_llm()
    for chunk in llm.stream(_messages(question, context)):
        if chunk.content:
            yield chunk.content


CITATION_RE = re.compile(r"\[([CFEQ]\d+)\]")


def extract_citations(text: str) -> list[str]:
    """Labels like C1, F3, E2 used in a text, in order of first use."""
    return list(dict.fromkeys(CITATION_RE.findall(text)))
