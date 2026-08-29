"""
MongoDB Atlas Vector Search backend — a third retrieval backbone over the
*same* ENISA corpus as rag/indexer.py (local vector store) and
rag/graph_indexer.py (Neo4j graph).

The point of this module is comparison, not replacement. It deliberately reuses
the local backend's corpus loader, chunk size, embedding model, candidate count
and reranker so that exactly one variable changes between the two backends:

    local  — chunks in ./rag_store, BM25 + vector fused by LlamaIndex in Python
    atlas  — chunks in MongoDB Atlas, $vectorSearch + $search fused by Atlas
             via Reciprocal Rank Fusion, server-side

One honest caveat for any writeup: embeddings are still generated locally by
Ollama, so every Atlas query pays a network round trip the local store does not.
Treat retrieval quality as the primary metric and latency as directional.

Requires MONGODB_URI in .env. Without it this module raises and nothing else in
the pipeline is affected.
"""

from __future__ import annotations

import logging

import pymongo
from llama_index.core import (
    QueryBundle,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.vector_stores.mongodb import MongoDBAtlasVectorSearch

from config import (
    EMBED_DIMENSIONS,
    MONGODB_COLLECTION,
    MONGODB_DB_NAME,
    MONGODB_URI,
    MONGODB_VECTOR_INDEX,
)

# Shared with the local backend on purpose — see module docstring.
from rag.indexer import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    FUSION_TOP_K,
    _configure_settings,
    format_source_nodes,
    format_sources,
    get_reranker,
    load_documents,
)

logger = logging.getLogger(__name__)

# Atlas needs a full-text index (not just the vector index) before it can run
# the $search half of a hybrid query.
FULLTEXT_INDEX_NAME = "fulltext_index"

# Atlas builds search indexes asynchronously. Block this long before giving up
# and telling the user to check the Atlas UI.
INDEX_BUILD_TIMEOUT_S = 180.0

_index: VectorStoreIndex | None = None


# ── Connection ───────────────────────────────────────────────────────────────

def _require_uri() -> str:
    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is not set. Add your Atlas connection string to .env:\n"
            '  MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/"\n'
            "See the MongoDB Atlas section of the README."
        )
    return MONGODB_URI


def get_client() -> pymongo.MongoClient:
    """Atlas client. `appname` tags the connection in Atlas's metrics view."""
    return pymongo.MongoClient(_require_uri(), appname="enisa-rag-agent")


def get_vector_store(client: pymongo.MongoClient | None = None) -> MongoDBAtlasVectorSearch:
    return MongoDBAtlasVectorSearch(
        mongodb_client=client or get_client(),
        db_name=MONGODB_DB_NAME,
        collection_name=MONGODB_COLLECTION,
        vector_index_name=MONGODB_VECTOR_INDEX,
        fulltext_index_name=FULLTEXT_INDEX_NAME,
    )


def ping() -> dict:
    """Cheap connectivity + state check. Used by --mongo-status and /health."""
    client = get_client()
    client.admin.command("ping")
    coll = client[MONGODB_DB_NAME][MONGODB_COLLECTION]
    try:
        search_indexes = [i["name"] for i in coll.list_search_indexes()]
    except Exception as e:  # collection may not exist yet
        logger.debug("Could not list search indexes: %s", e)
        search_indexes = []
    return {
        "connected": True,
        "database": MONGODB_DB_NAME,
        "collection": MONGODB_COLLECTION,
        "chunks": coll.estimated_document_count(),
        "search_indexes": search_indexes,
    }


# ── Search indexes ───────────────────────────────────────────────────────────

def ensure_search_indexes(store: MongoDBAtlasVectorSearch) -> None:
    """
    Create the vector and full-text search indexes if absent.

    Atlas rejects vectors whose dimension differs from the index declaration, so
    switching embedding models means dropping the index and rebuilding. Index
    creation is a no-op when the index already exists, making this safe to call
    on every build.
    """
    existing = {i["name"] for i in store.collection.list_search_indexes()}

    if MONGODB_VECTOR_INDEX not in existing:
        logger.info("Creating Atlas vector search index %r (%d dims)",
                    MONGODB_VECTOR_INDEX, EMBED_DIMENSIONS)
        store.create_vector_search_index(
            dimensions=EMBED_DIMENSIONS,
            path="embedding",
            similarity="cosine",
            wait_until_complete=INDEX_BUILD_TIMEOUT_S,
        )
    else:
        logger.info("Vector search index %r already exists", MONGODB_VECTOR_INDEX)

    if FULLTEXT_INDEX_NAME not in existing:
        logger.info("Creating Atlas full-text search index %r", FULLTEXT_INDEX_NAME)
        store.create_fulltext_search_index(
            field="text",
            wait_until_complete=INDEX_BUILD_TIMEOUT_S,
        )
    else:
        logger.info("Full-text search index %r already exists", FULLTEXT_INDEX_NAME)


# ── Build ────────────────────────────────────────────────────────────────────

def build_index(*, reset: bool = True) -> VectorStoreIndex:
    """
    Embed the corpus locally and write the chunks to Atlas.

    Args:
        reset: drop existing chunks first (default). Set False to append.

    Chunks are inserted before the search indexes are created: Atlas cannot
    build a search index on a collection that does not exist yet, and on a
    rebuild the index survives the document drop anyway.
    """
    global _index
    _configure_settings()

    documents = load_documents()
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    client = get_client()
    store = get_vector_store(client)

    if reset:
        deleted = client[MONGODB_DB_NAME][MONGODB_COLLECTION].delete_many({}).deleted_count
        if deleted:
            logger.info("Cleared %d existing chunks from %s.%s",
                        deleted, MONGODB_DB_NAME, MONGODB_COLLECTION)

    storage_context = StorageContext.from_defaults(vector_store=store)
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        transformations=[splitter],
        show_progress=True,
    )

    ensure_search_indexes(store)

    _index = index
    return index


def load_index() -> VectorStoreIndex:
    """Attach to the chunks already stored in Atlas — no re-embedding."""
    global _index
    _configure_settings()

    client = get_client()
    coll = client[MONGODB_DB_NAME][MONGODB_COLLECTION]
    if coll.estimated_document_count() == 0:
        raise RuntimeError(
            f"No chunks in {MONGODB_DB_NAME}.{MONGODB_COLLECTION}. "
            "Run `python main.py --rebuild-mongo` first."
        )

    _index = VectorStoreIndex.from_vector_store(get_vector_store(client))
    return _index


# ── Query ────────────────────────────────────────────────────────────────────

def _build_retriever(index: VectorStoreIndex, *, hybrid: bool):
    """
    Atlas-side retrieval, without the reranker or the LLM.

    In hybrid mode Atlas runs $vectorSearch and $search in one aggregation
    pipeline and fuses them with Reciprocal Rank Fusion server-side. That is the
    interesting contrast with the local backend, which pulls two candidate sets
    into Python and fuses them there.

    Split out of _build_query_engine() so retrieval can run on its own — see
    retrieve_sources().
    """
    return VectorIndexRetriever(
        index=index,
        similarity_top_k=FUSION_TOP_K,
        vector_store_query_mode=(
            VectorStoreQueryMode.HYBRID if hybrid else VectorStoreQueryMode.DEFAULT
        ),
    )


def _build_query_engine(index: VectorStoreIndex, *, hybrid: bool):
    """
    Atlas retrieval → cross-encoder reranker → LLM.

    FUSION_TOP_K is imported from the local backend, and get_reranker() returns
    the same shared cross-encoder instance, so both pipelines hand the LLM the
    same number of chunks chosen by the same reranker.
    """
    return RetrieverQueryEngine.from_args(
        retriever=_build_retriever(index, hybrid=hybrid),
        node_postprocessors=[get_reranker()],
    )


def query(question: str, *, use_hybrid: bool = True) -> str:
    """Run a retrieval-augmented query against Atlas. Mirrors indexer.query()."""
    return query_with_sources(question, use_hybrid=use_hybrid)["answer"]


def retrieve_sources(question: str, *, use_hybrid: bool = True) -> dict:
    """
    Retrieval only — Atlas search + reranker, no LLM. Mirrors
    indexer.retrieve_sources() so the eval harness scores both the same way.

    Returns: {"sources": [{"title", "source_url", "score"}, ...]}
    """
    global _index
    if _index is None:
        load_index()
    assert _index is not None

    bundle = QueryBundle(question)
    retriever = _build_retriever(_index, hybrid=use_hybrid)
    try:
        nodes = retriever.retrieve(bundle)
    except Exception as e:
        if not use_hybrid:
            raise
        # Same fallback as query_with_sources(): hybrid needs the full-text
        # index, and a missing or still-building one fails the whole query.
        logger.warning("Atlas hybrid retrieval failed (%s) — retrying vector-only", e)
        retriever = _build_retriever(_index, hybrid=False)
        nodes = retriever.retrieve(bundle)

    nodes = get_reranker().postprocess_nodes(nodes, query_bundle=bundle)

    return {"sources": format_source_nodes(nodes)}


def query_with_sources(question: str, *, use_hybrid: bool = True) -> dict:
    """
    Like query() but also returns source nodes for citation.

    Returns: {"answer": str, "sources": [{"title", "source_url", "score"}, ...]}
    """
    global _index
    if _index is None:
        load_index()
    assert _index is not None

    try:
        engine = _build_query_engine(_index, hybrid=use_hybrid)
        response = engine.query(question)
    except Exception as e:
        if not use_hybrid:
            raise
        # Hybrid needs the full-text index; a missing or still-building one
        # fails the whole pipeline. Vector-only still answers.
        logger.warning("Atlas hybrid query failed (%s) — retrying vector-only", e)
        engine = _build_query_engine(_index, hybrid=False)
        response = engine.query(question)

    return {
        "answer": (response.response or str(response)).strip(),
        "sources": format_sources(response),
    }
