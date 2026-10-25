"""Step 9 — FastAPI backend.

Run:   uvicorn app.api.main:app --reload --port 8000
Docs:  http://localhost:8000/docs

Endpoints
  GET  /health          FAISS + Neo4j status
  POST /query           question -> verified, cited answer
  POST /query/stream    same, streamed as NDJSON events (answer tokens arrive as written)
  GET  /graph/stats     node / relationship counts for the :Spotify graph
  POST /ingest          rebuild FAISS and/or the graph (background job)
  GET  /ingest/status   progress of the last ingest job
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.ingestion.graph_builder import graph_stats
from app.ingestion.ingest import run_ingest
from app.neo4j_client import run_query
from app.pipeline import ask, ask_stream
from app.retrieval.vector_retriever import get_vector_store


# ---------------------------------------------------------------- startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up once so the first question isn't slow. If the index or graph
    # isn't built yet, the API still starts and /health shows what's missing.
    try:
        get_vector_store()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] FAISS not loaded: {exc}")
    # Warm up Neo4j in the background so a slow/paused Aura can't block startup
    def _warm_neo4j():
        try:
            run_query("RETURN 1 AS ok")
            print("[startup] Neo4j connected")
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] Neo4j not reachable: {exc}")
    threading.Thread(target=_warm_neo4j, daemon=True).start()
    yield


app = FastAPI(
    title="Hybrid RAG — Spotify architecture",
    description="FAISS vector search + Neo4j knowledge graph + OpenAI, with answer verification.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------- models
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["What happens after an artist uploads a track?"])
    verify: bool = Field(True, description="Run the verification LLM call")


class IngestRequest(BaseModel):
    rebuild_vectors: bool = True
    rebuild_graph: bool = True
    reuse_extraction: bool = Field(False, description="Reuse cached LLM extraction (no LLM cost)")


# ---------------------------------------------------------------- health
_HEALTH_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="health")

@app.get("/health")
def health():
    s = get_settings()
    status = {"status": "ok", "models": {"chat": s.openai_chat_model, "embed": s.openai_embed_model}}

    try:
        store = get_vector_store()
        status["faiss"] = {"ok": True, "vectors": int(store.index.ntotal)}
    except Exception as exc:  # noqa: BLE001
        status["faiss"] = {"ok": False, "error": str(exc)}

    # Run the Neo4j check with a hard time limit so /health never hangs
    future = _HEALTH_POOL.submit(run_query, "MATCH (n:Spotify) RETURN count(n) AS c")
    try:
        count = future.result(timeout=8)[0]["c"]
        status["neo4j"] = {"ok": count > 0, "spotify_nodes": count}
    except FutureTimeout:
        status["neo4j"] = {"ok": False, "error": "Neo4j did not answer within 8 s "
                                                 "(Aura paused, or network problem)"}
    except Exception as exc:  # noqa: BLE001
        status["neo4j"] = {"ok": False, "error": str(exc)}

    if not (status["faiss"]["ok"] and status["neo4j"]["ok"]):
        status["status"] = "degraded"
    return status


# ---------------------------------------------------------------- query
@app.post("/query")
def query(req: QueryRequest):
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so the
    # blocking OpenAI / Neo4j calls don't freeze the server.
    t = time.perf_counter()
    try:
        result = ask(req.question.strip(), verify=req.verify)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Index not built: {exc}") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc
    result["timings"]["total_s"] = round(time.perf_counter() - t, 2)
    return result


@app.post("/query/stream")
def query_stream(req: QueryRequest):
    """Newline-delimited JSON: {"type":"retrieval"} -> {"type":"token"}... -> {"type":"final"}.
    Errors arrive as {"type":"error","detail":...} because the HTTP status is already sent."""
    def events():
        try:
            for event in ask_stream(req.question.strip(), verify=req.verify):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"type": "error", "detail": f"{type(exc).__name__}: {exc}"}) + "\n"

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------------- graph stats
@app.get("/graph/stats")
def stats():
    nodes, rels = graph_stats()
    return {"nodes": nodes, "relationships": rels}


# ---------------------------------------------------------------- ingest
_ingest_lock = threading.Lock()
_ingest_state: dict = {"state": "idle", "log": [], "summary": None, "error": None}


def _run_ingest_job(req: IngestRequest):
    def log(msg: str):
        _ingest_state["log"].append(f"{datetime.now(timezone.utc):%H:%M:%S} {msg}")

    try:
        _ingest_state["summary"] = run_ingest(
            rebuild_vectors=req.rebuild_vectors,
            rebuild_graph=req.rebuild_graph,
            reuse_extraction=req.reuse_extraction,
            log=log,
        )
        _ingest_state["state"] = "done"
    except Exception as exc:  # noqa: BLE001
        _ingest_state["state"] = "failed"
        _ingest_state["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _ingest_lock.release()


@app.post("/ingest", status_code=202)
def ingest(req: IngestRequest, background: BackgroundTasks,
           x_admin_token: str | None = Header(default=None)):
    token = get_settings().admin_token
    if token and x_admin_token != token:
        raise HTTPException(status_code=401, detail="Missing or wrong X-Admin-Token header")
    if not _ingest_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="An ingest job is already running")

    _ingest_state.update(state="running", log=[], summary=None, error=None)
    background.add_task(_run_ingest_job, req)
    return {"message": "Ingest started", "status_url": "/ingest/status"}


@app.get("/ingest/status")
def ingest_status():
    return _ingest_state
