"""
LlamaIndex vector RAG over scraped ENISA PDFs.

Features:
- Hybrid retrieval: BM25 + vector, fused via QueryFusionRetriever
- Cross-encoder reranking (SentenceTransformerRerank)
- Sidecar abstract ingestion as extra Documents
- Section-aware metadata on chunks
- Incremental re-indexing via refresh_ref_docs()

Uses a local Ollama server (embeddings + LLM). No API keys required.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from llama_index.core import (
    Document,
    QueryBundle,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers import QueryFusionRetriever, VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever

from config import (
    METADATA_FILE,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_REQUEST_TIMEOUT,
    OLLAMA_REQUEST_TIMEOUT,
    OUTPUT_DIR,
    RAG_STORE_DIR,
)

logger = logging.getLogger(__name__)

# ── Model config ─────────────────────────────────────────────────────────────
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_LLM_MODEL = "llama3.2"

# Bound the generation context. Without this, llama-index defaults Ollama to the
# model's full context (128K for llama3.2), which forces a ~17GB load that runs
# on CPU and makes every answer take minutes. The RAG prompt only stuffs
# RERANK_TOP_K chunks (~512 tokens each) + the question, so 8K is ample.
OLLAMA_LLM_CONTEXT_WINDOW = 8192

CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval knobs
VECTOR_TOP_K = 15          # candidates from vector retriever
BM25_TOP_K = 15            # candidates from BM25 retriever
FUSION_TOP_K = 25          # passed to reranker
RERANK_TOP_K = 6           # final chunks to LLM
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_index: VectorStoreIndex | None = None
_nodes: list | None = None  # kept for BM25 retriever rebuild

# Cross-encoder weights take ~12s to load and the model is stateless, so it is
# built once per process rather than per query. Corpus-independent, so nothing
# invalidates it. _bm25 IS corpus-dependent and is cleared whenever _nodes change.
_reranker: SentenceTransformerRerank | None = None
_bm25: BM25Retriever | None = None


# ── Settings ─────────────────────────────────────────────────────────────────

def _configure_settings() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=OLLAMA_EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={"timeout": OLLAMA_REQUEST_TIMEOUT},
    )
    Settings.llm = Ollama(
        model=OLLAMA_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_LLM_REQUEST_TIMEOUT,
        context_window=OLLAMA_LLM_CONTEXT_WINDOW,
        temperature=0.0,
    )


# ── Metadata helpers ─────────────────────────────────────────────────────────

def _flatten_metadata(meta: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, list):
            out[str(k)] = ", ".join(str(i) for i in v)
        elif isinstance(v, str):
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def _path_lookup_from_metadata(metadata_by_uid: dict[str, dict]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    root = OUTPUT_DIR.resolve()

    for uid, rec in metadata_by_uid.items():
        base = {**rec, "uid": uid}
        flat = _flatten_metadata(base)
        lp = rec.get("local_path")
        if not lp:
            continue
        p = Path(lp)
        paths = {
            lp, str(p), str(p.resolve()),
            str((root / p.name).resolve()) if not p.is_absolute() else str(p.resolve()),
        }
        try:
            paths.add(str(p.resolve().relative_to(root)))
        except ValueError:
            pass
        for key in paths:
            lookup[key] = flat
        lookup[p.name] = flat
    return lookup


def _file_metadata_fn(lookup: dict[str, dict[str, str]]) -> Callable[[str], dict]:
    def _fn(file_path_str: str) -> dict:
        p = Path(file_path_str)
        for c in [file_path_str, str(p), str(p.resolve()), p.name]:
            if c in lookup:
                return dict(lookup[c])
        try:
            rel = str(p.resolve().relative_to(OUTPUT_DIR.resolve()))
            if rel in lookup:
                return dict(lookup[rel])
        except ValueError:
            pass
        return {}
    return _fn


# ── Sidecar abstract loading ────────────────────────────────────────────────

def _load_sidecar_documents(metadata_by_uid: dict[str, dict]) -> list[Document]:
    """
    Load .meta.json sidecars as extra Documents so abstracts
    and topic tags are retrievable independently of PDF chunks.
    """
    docs: list[Document] = []
    for uid, rec in metadata_by_uid.items():
        lp = rec.get("local_path")
        if not lp:
            continue
        sidecar = Path(lp).with_suffix(".meta.json")
        if not sidecar.exists():
            continue
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            continue
        abstract = meta.get("abstract", "")
        topics = meta.get("topics", [])
        if not abstract and not topics:
            continue
        text_parts = []
        if abstract:
            text_parts.append(f"Abstract: {abstract}")
        if topics:
            text_parts.append(f"Topics: {', '.join(topics)}")
        doc = Document(
            text="\n".join(text_parts),
            metadata=_flatten_metadata({
                **rec,
                "uid": uid,
                "chunk_type": "sidecar_abstract",
            }),
        )
        docs.append(doc)
    return docs


# ── Build ────────────────────────────────────────────────────────────────────

def load_documents() -> list[Document]:
    """
    Load the corpus: PDFs from OUTPUT_DIR plus sidecar abstracts, with
    scraper metadata attached to every Document.

    Shared by every retrieval backend (local vector store, MongoDB Atlas,
    graph) so each one indexes the *same* corpus — a precondition for the
    backend comparison in the README being meaningful.
    """
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    metadata_by_uid: dict[str, dict] = json.loads(METADATA_FILE.read_text())
    lookup = _path_lookup_from_metadata(metadata_by_uid)

    # Load PDFs
    reader = SimpleDirectoryReader(
        input_dir=str(OUTPUT_DIR),
        required_exts=[".pdf"],
        recursive=False,
        # SimpleDirectoryReader's hidden-file check walks the *whole* path, so
        # it finds nothing when the checkout sits under a dotted parent
        # directory. required_exts already restricts this to PDFs.
        exclude_hidden=False,
        file_metadata=_file_metadata_fn(lookup),
    )
    documents: list[Document] = reader.load_data()

    # Load sidecar abstracts
    sidecar_docs = _load_sidecar_documents(metadata_by_uid)
    if sidecar_docs:
        logger.info("Loaded %d sidecar abstract documents", len(sidecar_docs))
        documents.extend(sidecar_docs)

    if not documents:
        raise FileNotFoundError(f"No PDFs found in {OUTPUT_DIR}")

    return documents


def build_index(*, incremental: bool = False) -> VectorStoreIndex:
    """
    Build (or incrementally update) the vector index from PDFs + sidecars.

    Args:
        incremental: if True and an index already exists, only add new docs
                     via refresh_ref_docs() instead of rebuilding from scratch.
    """
    global _index, _nodes, _bm25
    _bm25 = None  # corpus is about to change
    _configure_settings()

    documents = load_documents()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Incremental path
    if incremental and RAG_STORE_DIR.exists() and any(RAG_STORE_DIR.iterdir()):
        logger.info("Incremental update — refreshing existing index")
        storage_context = StorageContext.from_defaults(persist_dir=str(RAG_STORE_DIR))
        index = load_index_from_storage(storage_context)
        index.refresh_ref_docs(documents)
        index.storage_context.persist(persist_dir=str(RAG_STORE_DIR))
        _index = index
        _nodes = list(index.docstore.docs.values())
        return index

    # Full rebuild
    RAG_STORE_DIR.mkdir(parents=True, exist_ok=True)
    storage_context = StorageContext.from_defaults()

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(RAG_STORE_DIR))
    _index = index
    _nodes = list(index.docstore.docs.values())
    return index


def load_index() -> VectorStoreIndex:
    """Reload persisted index from disk."""
    global _index, _nodes, _bm25
    _bm25 = None  # corpus is about to change
    _configure_settings()
    if not RAG_STORE_DIR.exists() or not any(RAG_STORE_DIR.iterdir()):
        raise FileNotFoundError(
            f"No persisted index at {RAG_STORE_DIR}. Run build_index() first."
        )
    storage_context = StorageContext.from_defaults(persist_dir=str(RAG_STORE_DIR))
    _index = load_index_from_storage(storage_context)
    _nodes = list(_index.docstore.docs.values())
    return _index


# ── Query ────────────────────────────────────────────────────────────────────

def get_reranker() -> SentenceTransformerRerank:
    """The shared cross-encoder reranker, loaded on first use.

    Also used by the Atlas backend, so both score candidates with the same
    model instance.
    """
    global _reranker
    if _reranker is None:
        _reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_K)
    return _reranker


def _get_bm25(nodes) -> BM25Retriever:
    """BM25 over the current corpus, rebuilt only when the corpus changes."""
    global _bm25
    if _bm25 is None:
        _bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=BM25_TOP_K)
    return _bm25


def _build_hybrid_retriever(index: VectorStoreIndex):
    """
    Hybrid retrieval half of the pipeline: BM25 + Vector → QueryFusionRetriever,
    plus the reranker that trims the fused candidates.

    Split out of _build_hybrid_query_engine() so retrieval can run on its own,
    without the LLM — see retrieve_sources().

    Returns: (retriever, reranker)
    """
    nodes = _nodes or list(index.docstore.docs.values())

    vector_retriever = VectorIndexRetriever(index=index, similarity_top_k=VECTOR_TOP_K)
    bm25_retriever = _get_bm25(nodes)

    fusion_retriever = QueryFusionRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        similarity_top_k=FUSION_TOP_K,
        num_queries=1,  # no query generation, just fuse
        use_async=False,  # run retrievers synchronously — avoids nested-async
        # crashes when engine.query() runs inside FastAPI's request threadpool
    )

    return fusion_retriever, get_reranker()


def _build_hybrid_query_engine(index: VectorStoreIndex):
    """
    Hybrid retriever: BM25 + Vector → QueryFusionRetriever → Reranker → LLM.
    """
    fusion_retriever, reranker = _build_hybrid_retriever(index)
    return RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
    )


def query(question: str, *, use_hybrid: bool = True) -> str:
    """
    Run a retrieval-augmented query.

    Args:
        question: natural language query
        use_hybrid: if True (default), use BM25+vector fusion + reranker.
                    If False, fall back to plain vector top-k.
    """
    global _index
    if _index is None:
        load_index()
    assert _index is not None

    if use_hybrid:
        try:
            engine = _build_hybrid_query_engine(_index)
        except Exception as e:
            logger.warning("Hybrid retriever failed (%s), falling back to vector-only", e)
            engine = _index.as_query_engine(similarity_top_k=RERANK_TOP_K)
    else:
        engine = _index.as_query_engine(similarity_top_k=RERANK_TOP_K)

    response = engine.query(question)
    return (response.response or str(response)).strip()


def format_source_nodes(nodes) -> list[dict]:
    """
    Deduplicate scored nodes into citation records.

    Shared with the Atlas backend so both report citations identically — the
    eval harness compares source titles across backends.
    """
    sources: list[dict] = []
    seen = set()
    for node in nodes:
        meta = node.metadata or {}
        title = meta.get("title", "Unknown")
        url = meta.get("source_url", "")
        key = (title, url)
        if key in seen:
            continue
        seen.add(key)
        sources.append({
            "title": title,
            "source_url": url,
            "score": round(node.score or 0.0, 4),
        })
    return sources


def format_sources(response) -> list[dict]:
    """Citation records for a query response."""
    return format_source_nodes(response.source_nodes)


def retrieve_sources(question: str, *, use_hybrid: bool = True) -> dict:
    """
    Retrieval only — same retriever and reranker as query_with_sources(), no LLM.

    Source recall never reads the generated answer, so the eval harness can skip
    generation entirely. That takes a question from ~60s of local Ollama
    inference to well under a second, which is what makes an iterate-and-measure
    loop over retrieval changes practical.

    Ollama must still be running: the query is embedded before it is searched.

    Returns: {"sources": [{"title": ..., "source_url": ..., "score": ...}, ...]}
    """
    global _index
    if _index is None:
        load_index()
    assert _index is not None

    reranker = None
    if use_hybrid:
        try:
            retriever, reranker = _build_hybrid_retriever(_index)
        except Exception as e:
            logger.warning("Hybrid retriever failed (%s), falling back to vector-only", e)
            retriever = VectorIndexRetriever(index=_index, similarity_top_k=RERANK_TOP_K)
    else:
        retriever = VectorIndexRetriever(index=_index, similarity_top_k=RERANK_TOP_K)

    bundle = QueryBundle(question)
    nodes = retriever.retrieve(bundle)
    if reranker is not None:
        nodes = reranker.postprocess_nodes(nodes, query_bundle=bundle)

    return {"sources": format_source_nodes(nodes)}


def query_with_sources(question: str) -> dict:
    """
    Like query() but also returns source nodes for citation.

    Returns: {"answer": str, "sources": [{"title": ..., "source_url": ..., "score": ...}, ...]}
    """
    global _index
    if _index is None:
        load_index()
    assert _index is not None

    try:
        engine = _build_hybrid_query_engine(_index)
    except Exception:
        engine = _index.as_query_engine(similarity_top_k=RERANK_TOP_K)

    response = engine.query(question)
    return {
        "answer": (response.response or str(response)).strip(),
        "sources": format_sources(response),
    }
