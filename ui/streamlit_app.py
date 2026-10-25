"""Step 10 — Streamlit UI for the hybrid RAG app.

Run (with the API already running on port 8000):
    streamlit run ui/streamlit_app.py

The UI only talks to FastAPI over HTTP; it never imports the app code, so the
two can be deployed separately.
"""
import os
import time

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
DEFAULT_API = os.getenv("API_BASE_URL", "http://localhost:8000")

SAMPLE_QUESTIONS = [
    "What happens after an artist uploads a track?",
    "Which services and tables implement the Playback & Player features?",
    "How does the app handle subscription payments?",
    "Which services use PostgreSQL tables?",
    "What is stored in ClickHouse?",
]

VERDICT_STYLE = {
    "supported": ("success", "✅ Verified: every claim is supported by the sources"),
    "partially_supported": ("warning", "⚠️ Partially supported: some claims were removed or are unsupported"),
    "not_supported": ("error", "❌ Not supported by the sources"),
    "no_answer": ("info", "ℹ️ The document doesn't cover this question"),
    "auto_accepted": ("success", "⚡ Grounded: passed the fast checks, so the verification call was skipped"),
}
SOURCE_BADGE = {"vector": "🔎 vector", "graph": "🕸️ graph"}

st.set_page_config(page_title="Hybrid RAG · Spotify architecture", page_icon="🎧", layout="wide")


# ---------------------------------------------------------------- API helpers
def api_get(path: str, timeout: int = 10):
    r = requests.get(f"{st.session_state.api_url}{path}", timeout=timeout)
    r.raise_for_status()
    return r.json()


def api_post(path: str, payload: dict, headers: dict | None = None, timeout: int = 120):
    r = requests.post(f"{st.session_state.api_url}{path}", json=payload,
                      headers=headers or {}, timeout=timeout)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail")
        except ValueError:
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def api_stream(path: str, payload: dict, timeout: int = 120):
    """Yield the NDJSON events of a streaming endpoint one by one."""
    import json
    with requests.post(f"{st.session_state.api_url}{path}", json=payload,
                       stream=True, timeout=timeout) as r:
        if r.status_code >= 400:
            raise RuntimeError(f"{r.status_code}: {r.text}")
        for line in r.iter_lines(decode_unicode=True):
            if line:
                event = json.loads(line)
                if event.get("type") == "error":
                    raise RuntimeError(event["detail"])
                yield event


@st.cache_data(ttl=15, show_spinner=False)
def get_health(api_url: str) -> dict:
    """Cached for 15 s so the sidebar doesn't call /health on every click."""
    r = requests.get(f"{api_url}/health", timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- state
st.session_state.setdefault("api_url", DEFAULT_API)
st.session_state.setdefault("history", [])  # [{"question", "result" | "error"}]
st.session_state.setdefault("pending", None)


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state.api_url = st.text_input("API URL", st.session_state.api_url)
    verify = st.toggle("Verify answers (2nd LLM call)", value=True)

    st.subheader("Backend status")
    try:
        h = get_health(st.session_state.api_url)
        faiss_ok, neo_ok = h["faiss"]["ok"], h["neo4j"]["ok"]
        c1, c2 = st.columns(2)
        c1.metric("FAISS vectors", h["faiss"].get("vectors", "—"))
        c2.metric("Graph nodes", h["neo4j"].get("spotify_nodes", "—"))
        if faiss_ok and neo_ok:
            st.success(f"Connected · {h['models']['chat']}")
        else:
            st.warning("Degraded: " + ", ".join(
                n for n, ok in (("FAISS", faiss_ok), ("Neo4j", neo_ok)) if not ok) + " not ready")
    except Exception as exc:  # noqa: BLE001
        st.error(f"API not reachable at {st.session_state.api_url}\n\n{exc}")

    st.subheader("Try a question")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, width="stretch"):
            st.session_state.pending = q

    if st.session_state.history and st.button("🗑️ Clear chat", width="stretch"):
        st.session_state.history = []
        st.rerun()

    with st.expander("🛠️ Rebuild data (admin)"):
        token = st.text_input("Admin token", type="password",
                              help="Only needed if ADMIN_TOKEN is set in .env")
        rv = st.checkbox("Rebuild FAISS index", value=True)
        rg = st.checkbox("Rebuild knowledge graph", value=True)
        reuse = st.checkbox("Reuse cached LLM extraction (no LLM cost)", value=True)
        if st.button("Start ingest"):
            try:
                api_post("/ingest", {"rebuild_vectors": rv, "rebuild_graph": rg,
                                     "reuse_extraction": reuse},
                         headers={"X-Admin-Token": token} if token else None)
                with st.spinner("Ingest running..."):
                    while (status := api_get("/ingest/status"))["state"] == "running":
                        time.sleep(2)
                if status["state"] == "done":
                    st.success(f"Done: {status['summary']}")
                else:
                    st.error(status["error"])
                st.code("\n".join(status["log"]) or "(no log)")
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))


# ---------------------------------------------------------------- rendering
def graph_dot(facts: list[dict], limit: int = 18) -> str:
    """Graphviz DOT for the top facts (Streamlit renders it in the browser)."""
    def q(s: str) -> str:
        return '"' + s.replace('"', "'") + '"'

    colors = {"Service": "#cfe8ff", "Table": "#e8f5e9", "Technology": "#fff3cd",
              "Feature": "#f3e5f5", "FeatureArea": "#ede7f6", "Event": "#ffe0e0",
              "Page": "#e0f7fa", "Layer": "#eeeeee", "PageGroup": "#eeeeee"}
    nodes, edges = {}, []
    for f in facts[:limit]:
        nodes[f["source"]] = f["source_label"]
        nodes[f["target"]] = f["target_label"]
        style = "dashed" if f["evidence"] == "inferred" else "solid"
        edges.append(f'{q(f["source"])} -> {q(f["target"])} [label={q(f["rel"])}, style={style}, fontsize=9]')
    lines = ["digraph G {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontsize=10];']
    lines += [f'{q(n)} [fillcolor="{colors.get(lbl, "#ffffff")}"];' for n, lbl in nodes.items()]
    lines += edges + ["}"]
    return "\n".join(lines)


def render_result(result: dict, show_answer: bool = True):
    v = result.get("verification")
    if show_answer:
        st.markdown(result["answer"])

    if v:
        kind, text = VERDICT_STYLE.get(v["verdict"], ("info", v["verdict"]))
        getattr(st, kind)(f"{text}  ·  confidence: **{v['confidence']}**")
        if v.get("reason"):
            st.caption(f"Verification: {v['reason']}")
    if result.get("cached"):
        st.caption("⚡ Served from the answer cache")
        if v["citations"]["invalid"]:
            st.caption(f"Invalid citations removed/flagged: {', '.join(v['citations']['invalid'])}")
    if result.get("was_corrected"):
        with st.expander("Original draft (before verification)"):
            st.markdown(result["draft_answer"])

    used = set(v["citations"]["used"]) if v else set()
    passages = [s for s in result["sources"] if s["type"] == "passage"]
    facts = [s for s in result["sources"] if s["type"] == "graph_fact"]

    gq = next((s for s in result["sources"] if s["type"] == "graph_query"), None)
    if gq:
        with st.expander("🧮 Graph query (text-to-Cypher)" + ("  **(cited)**" if "Q1" in used else ""),
                         expanded="Q1" in used):
            st.caption(gq["text"])
            st.code(gq["cypher"], language="cypher")
            if gq["rows"]:
                st.dataframe(pd.DataFrame(gq["rows"]), hide_index=True, width="stretch")
            else:
                st.caption("No rows returned.")

    tab_src, tab_graph, tab_check, tab_debug = st.tabs(
        [f"📄 Passages ({len(passages)})", f"🕸️ Graph facts ({len(facts)})",
         "✔️ Verification", "🔧 Retrieval"])

    with tab_src:
        for s in passages:
            badges = " · ".join(SOURCE_BADGE.get(x, x) for x in s["retrieved_by"])
            cited = " **(cited)**" if s["label"] in used else ""
            with st.expander(f"[{s['label']}] {s['title']} · p.{s['page']} · {badges}{cited}",
                             expanded=s["label"] in used):
                st.text(s["text"])
                st.caption(s["chunk_id"])

    with tab_graph:
        if facts:
            st.graphviz_chart(graph_dot(facts), width="stretch")
            st.caption("Solid = stated in the document · dashed = inferred by the system")
            st.dataframe(
                pd.DataFrame([{
                    "label": f["label"], "cited": f["label"] in used, "fact": f["text"],
                    "evidence": f["evidence"], "hop": f["hop"],
                    "chunks": ", ".join(f["chunk_ids"]),
                } for f in facts]),
                hide_index=True, width="stretch",
            )
        else:
            st.info("No graph facts for this question.")

    with tab_check:
        if not v or not v["claims"]:
            st.info("No claims to check." if v else "Verification was turned off.")
        else:
            icon = {"supported": "✅", "unsupported": "❓", "contradicted": "❌"}
            for c in v["claims"]:
                note = f" — _{c['note']}_" if c["note"] else ""
                cites = " ".join(f"`{x}`" for x in c["citations"])
                st.markdown(f"{icon[c['status']]} {c['claim']} {cites}{note}")

    with tab_debug:
        gq_dbg = result["retrieval"].get("graph_query")
        if gq_dbg and gq_dbg.get("error"):
            st.caption(f"Text-to-Cypher: {gq_dbg['error']}")
        exp = result["retrieval"].get("expansion")
        if exp and exp.get("skipped"):
            st.caption(f"Query expansion not needed: {exp['skipped']}")
        if exp and len(exp.get("queries", [])) > 1:
            st.markdown("**Query expansion**")
            st.caption("Searched as: " + " · ".join(f"“{q}”" for q in exp["queries"]))
            if exp.get("keywords"):
                st.caption("Keywords: " + ", ".join(exp["keywords"]))
        st.dataframe(pd.DataFrame(result["retrieval"]["chunks"]), hide_index=True,
                     width="stretch")
        st.caption(f"{result['retrieval']['fact_count']} facts · "
                   f"{result['retrieval']['entity_count']} entities · timings: {result['timings']}")


# ---------------------------------------------------------------- main page
st.title("🎧 Hybrid RAG · Spotify-like architecture")
st.caption("Answers combine **FAISS** vector search and a **Neo4j** knowledge graph, "
           "generated and fact-checked by OpenAI.")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if "error" in turn:
            st.error(turn["error"])
        else:
            render_result(turn["result"])

question = st.chat_input("Ask about the architecture...") or st.session_state.pending
st.session_state.pending = None

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        status = st.empty()
        answer_box = st.empty()
        try:
            status.caption("🔎 Searching FAISS + Neo4j...")
            text, result = "", None
            for event in api_stream("/query/stream", {"question": question, "verify": verify}):
                if event["type"] == "retrieval":
                    status.caption("✍️ Writing the answer..." if not event.get("cached")
                                   else "⚡ From the answer cache")
                elif event["type"] == "token":
                    text += event["text"]
                    answer_box.markdown(text + "▌")          # typing cursor while streaming
                elif event["type"] == "final":
                    result = event["result"]
            status.empty()
            if result is None:
                raise RuntimeError("The stream ended without a final answer")
            # The verifier may have corrected the streamed draft: always show the final text
            answer_box.markdown(result["answer"])
            st.session_state.history.append({"question": question, "result": result})
            render_result(result, show_answer=False)
        except Exception as exc:  # noqa: BLE001
            status.empty()
            msg = f"Request failed: {exc}"
            st.session_state.history.append({"question": question, "error": msg})
            st.error(msg)
