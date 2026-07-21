"""
FastAPI wrapper around the RAG query engine.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /query         — {"question": "..."} → {"answer": "...", "sources": [...]}
    GET  /health        — liveness probe
    GET  /docs          — auto-generated OpenAPI docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.indexer import load_index, query, query_with_sources

logger = logging.getLogger(__name__)


# ── Lifespan: warm up the index on startup ───────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading RAG index on startup...")
    try:
        load_index()
        logger.info("Index loaded.")
    except FileNotFoundError:
        logger.warning(
            "No persisted index found — run `python main.py --rebuild-index` first."
        )
    yield


app = FastAPI(
    title="ENISA RAG Agent API",
    version="0.2.0",
    description="Query ENISA cybersecurity publications via hybrid RAG.",
    lifespan=lifespan,
)


# ── Schemas ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    use_hybrid: bool = Field(True, description="Use BM25+vector hybrid retrieval")
    include_sources: bool = Field(True, description="Return source citations")


class SourceInfo(BaseModel):
    title: str
    source_url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []


class GraphStatsResponse(BaseModel):
    nodes_total: int
    relationships_total: int
    nodes_by_type: dict[str, int]
    relationships_by_type: dict[str, int]


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/query", response_model=QueryResponse)
async def handle_query(req: QueryRequest):
    try:
        if req.include_sources:
            result = query_with_sources(req.question)
            return QueryResponse(
                answer=result["answer"],
                sources=[SourceInfo(**s) for s in result["sources"]],
            )
        else:
            answer = query(req.question, use_hybrid=req.use_hybrid)
            return QueryResponse(answer=answer)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Index not ready: {e}. Run `python main.py --rebuild-index`.",
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/stats", response_model=GraphStatsResponse)
async def graph_stats():
    """Node/edge counts by type for the Neo4j knowledge graph (GraphRAG layer)."""
    try:
        from rag.graph_indexer import get_graph_stats

        return GraphStatsResponse(**get_graph_stats())
    except Exception as e:
        logger.exception("Graph stats failed")
        raise HTTPException(
            status_code=503,
            detail=(
                f"Graph unavailable: {e}. Ensure Neo4j is running "
                "(`docker compose up -d neo4j`) and built "
                "(`python main.py --rebuild-graph`)."
            ),
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
