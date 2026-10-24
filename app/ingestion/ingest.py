"""Run the whole ingestion (Steps 3-5) from code — used by POST /ingest.

PDF -> chunks.json -> FAISS index -> LLM extraction -> Neo4j graph
"""
from app.config import get_settings
from app.ingestion.graph_builder import graph_stats, load_graph
from app.ingestion.graph_extractor import extract_graph, load_extraction, save_extraction
from app.ingestion.loader import build_chunks, load_chunks, save_chunks
from app.ingestion.vector_builder import build_vector_store

CHUNKS_PATH = "data/processed/chunks.json"
EXTRACTION_PATH = "data/processed/graph_extraction.json"


def clear_caches() -> None:
    """Make the running API pick up the rebuilt index/graph without a restart."""
    from app.retrieval import hybrid, vector_retriever
    vector_retriever.get_vector_store.cache_clear()
    hybrid._chunk_lookup.cache_clear()


def run_ingest(rebuild_vectors: bool = True, rebuild_graph: bool = True,
               reuse_extraction: bool = False, log=print) -> dict:
    s = get_settings()
    summary: dict = {}

    log(f"Chunking {s.pdf_path}")
    docs = build_chunks(s.pdf_path)
    save_chunks(docs, CHUNKS_PATH)
    summary["chunks"] = len(docs)

    if rebuild_vectors:
        log("Building FAISS index")
        summary["vectors"] = build_vector_store(docs)

    if rebuild_graph:
        if reuse_extraction:
            log("Loading cached graph extraction")
            extraction = load_extraction(EXTRACTION_PATH)
        else:
            log("Extracting entities with the LLM")
            extraction = extract_graph(load_chunks(CHUNKS_PATH), verbose=False)
            save_extraction(extraction, EXTRACTION_PATH)
        log("Loading graph into Neo4j")
        load_graph(docs, extraction)
        nodes, _ = graph_stats()
        summary["graph_nodes"] = {n["label"]: n["count"] for n in nodes}

    clear_caches()
    log("Ingest finished")
    return summary
