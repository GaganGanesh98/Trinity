"""
Knowledge-graph construction for the GraphRAG layer.

This sits ALONGSIDE the existing vector pipeline in rag/indexer.py — it does not
replace it. The same ENISA PDFs are run through a LlamaIndex PropertyGraphIndex
with a domain-tuned SchemaLLMPathExtractor, producing a STRUCTURED cybersecurity
knowledge graph persisted in Neo4j.

Schema (see docs/graph-schema.md):
    entities   ThreatActor, Technique, Vulnerability, Malware,
               Sector, Mitigation, Regulation
    relations  USES, TARGETS, MITIGATES, EXPLOITS, REFERENCES

Everything runs locally via Ollama (LLM + embeddings) and a local Neo4j
(docker-compose). No API keys, no paid services.

CLI:  python main.py --rebuild-graph [--graph-limit N]
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Literal, Sequence

from llama_index.core import Document, Settings, SimpleDirectoryReader
from llama_index.core.indices.property_graph import PropertyGraphIndex
from llama_index.core.graph_stores.types import (
    KG_NODES_KEY,
    KG_RELATIONS_KEY,
    EntityNode,
    Relation,
)
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.schema import BaseNode, TransformComponent
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from llama_index.llms.ollama import Ollama
from pydantic import BaseModel, Field

from config import (
    METADATA_FILE,
    NEO4J_DATABASE,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_REQUEST_TIMEOUT,
    OUTPUT_DIR,
)

# Reuse the vector pipeline's settings + document-loading helpers verbatim so the
# graph is built from exactly the same chunks/metadata as the vector store.
from rag.indexer import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    _configure_settings,
    _file_metadata_fn,
    _load_sidecar_documents,
    _path_lookup_from_metadata,
)

logger = logging.getLogger(__name__)

# Graph extraction uses a dedicated model, separate from the generation LLM
# (config.OLLAMA_LLM_MODEL / llama3.2). SchemaLLMPathExtractor(strict=True) needs
# reliable structured output to stay schema-conformant, which the base llama3.2
# cannot do (it returns empty). qwen2.5 handles structured output well; the 3b
# variant is chosen so it fits comfortably in 8GB RAM and runs at usable speed
# (7b thrashes on low-memory machines). Extraction is an offline, one-time job.
# Swap via the GRAPH_LLM_MODEL env var. Requires: ollama pull qwen2.5:3b
GRAPH_LLM_MODEL = os.getenv("GRAPH_LLM_MODEL", "qwen2.5:3b")

# The shared _configure_settings() builds its Ollama LLM without a context_window,
# so llama-index defaults to the model's full context (128K for llama3.2) — which
# pins the model in memory and runs generation on CPU, making every call hang for
# minutes. Graph extraction only sees a single ~512-token chunk plus the schema
# prompt, so a small bounded context keeps each call fast.
GRAPH_LLM_CONTEXT_WINDOW = 8192


def _configure_graph_settings() -> None:
    """Reuse the shared embeddings, but override the LLM with the extraction model."""
    _configure_settings()  # sets nomic-embed-text embeddings globally
    Settings.llm = Ollama(
        model=GRAPH_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        request_timeout=OLLAMA_LLM_REQUEST_TIMEOUT,
        context_window=GRAPH_LLM_CONTEXT_WINDOW,
        temperature=0.0,
    )


# ── Domain schema ─────────────────────────────────────────────────────────────
# Kept as plain data so it's easy to read, diff, and document.

ENTITIES: list[str] = [
    "ThreatActor",
    "Technique",
    "Vulnerability",
    "Malware",
    "Sector",
    "Mitigation",
    "Regulation",
]

RELATIONS: list[str] = [
    "USES",
    "TARGETS",
    "MITIGATES",
    "EXPLOITS",
    "REFERENCES",
]

# Allowed (subject, relation, object) triples. Only edges matching one of these
# shapes are kept, so the graph stays structured (not free-form). These shapes are
# also what make multi-hop questions answerable, e.g.:
#   Sector <-TARGETS- ThreatActor -USES-> Technique <-MITIGATES- Mitigation
VALIDATION_SCHEMA: list[tuple[str, str, str]] = [
    ("ThreatActor", "USES", "Malware"),
    ("ThreatActor", "USES", "Technique"),
    ("ThreatActor", "TARGETS", "Sector"),
    ("ThreatActor", "TARGETS", "Vulnerability"),
    ("Malware", "EXPLOITS", "Vulnerability"),
    ("Technique", "EXPLOITS", "Vulnerability"),
    ("Technique", "TARGETS", "Sector"),
    ("Mitigation", "MITIGATES", "Technique"),
    ("Mitigation", "MITIGATES", "Vulnerability"),
    ("Mitigation", "MITIGATES", "Malware"),
    ("Regulation", "REFERENCES", "Mitigation"),
    ("Regulation", "REFERENCES", "Sector"),
    ("Regulation", "REFERENCES", "Technique"),
]

# Literal enums used to constrain the LLM's structured output to the schema.
_EntityType = Literal[tuple(ENTITIES)]  # type: ignore[valid-type]
_RelationType = Literal[tuple(RELATIONS)]  # type: ignore[valid-type]

_ALLOWED_TRIPLES: set[tuple[str, str, str]] = set(VALIDATION_SCHEMA)

MAX_TRIPLETS_PER_CHUNK = 10


# ── Schema-constrained extractor ──────────────────────────────────────────────
# NOTE: we intentionally do NOT use LlamaIndex's SchemaLLMPathExtractor(strict=True).
# On 8GB-class hardware that path is infeasible: its complex union schema makes
# small local models (llama3.2, qwen2.5:3b) return *nothing*, and the one model
# that handles it (qwen2.5:7b) runs at ~2 tok/s here (minutes per chunk).
#
# Instead we constrain a *flat* structured schema whose entity/relation types are
# Literal enums. A 3B model reliably satisfies this via Ollama structured output
# (~18s/chunk), and we validate every triple against VALIDATION_SCHEMA in Python.
# Same outcome — a structured, schema-conformant graph — but it actually runs
# locally and fast. Everything else (PropertyGraphIndex, Neo4j) is stock LlamaIndex.


class _ExtractedRelation(BaseModel):
    """One schema-typed edge the LLM is asked to emit."""

    subject: str = Field(description="Name of the subject entity")
    subject_type: _EntityType = Field(description="Type of the subject entity")
    relation: _RelationType = Field(description="Relationship type")
    object: str = Field(description="Name of the object entity")
    object_type: _EntityType = Field(description="Type of the object entity")


class _GraphExtraction(BaseModel):
    relations: list[_ExtractedRelation] = Field(default_factory=list)


_ALLOWED_SHAPES = "\n".join(f"  ({s})-[{r}]->({o})" for s, r, o in VALIDATION_SCHEMA)

_EXTRACT_PROMPT = PromptTemplate(
    "You are extracting a cybersecurity knowledge graph from a document excerpt.\n"
    "Extract entities and the relationships between them.\n\n"
    "Allowed entity types: " + ", ".join(ENTITIES) + "\n"
    "Allowed relation types: " + ", ".join(RELATIONS) + "\n"
    "Only emit relationships matching one of these shapes "
    "(subject_type)-[relation]->(object_type):\n" + _ALLOWED_SHAPES + "\n\n"
    "Rules:\n"
    "- Use the exact entity/relation type names above.\n"
    "- Skip anything that does not fit an allowed shape.\n"
    "- Return an empty list if the text contains no relevant relationships.\n\n"
    "Document excerpt:\n{text}\n"
)


class SchemaEnumPathExtractor(TransformComponent):
    """LlamaIndex KG extractor that constrains output to the domain schema via
    enum-typed structured prediction, then validates against VALIDATION_SCHEMA.

    Drop-in for a PropertyGraphIndex ``kg_extractors`` entry. Runs synchronously
    (local Ollama serves one request at a time anyway; concurrency just causes
    memory thrash on low-RAM machines).
    """

    llm: LLM
    max_triplets_per_chunk: int = MAX_TRIPLETS_PER_CHUNK

    def __call__(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        total = len(nodes)
        for i, node in enumerate(nodes, 1):
            self._extract(node)
            if show_progress and (i % 5 == 0 or i == total):
                logger.info("  extracted %d/%d chunks", i, total)
        return list(nodes)

    async def acall(
        self, nodes: Sequence[BaseNode], show_progress: bool = False, **kwargs: Any
    ) -> list[BaseNode]:
        # Serialized on purpose — see class docstring.
        return self.__call__(nodes, show_progress=show_progress, **kwargs)

    def _extract(self, node: BaseNode) -> None:
        try:
            result: _GraphExtraction = self.llm.structured_predict(
                _GraphExtraction, _EXTRACT_PROMPT, text=node.get_content()
            )
        except Exception as e:  # one bad chunk shouldn't abort the whole build
            logger.warning("Extraction failed for a chunk: %s", e)
            return

        existing_nodes = node.metadata.pop(KG_NODES_KEY, [])
        existing_rels = node.metadata.pop(KG_RELATIONS_KEY, [])
        entities: dict[str, EntityNode] = {}
        relations: list[Relation] = []

        for rel in result.relations[: self.max_triplets_per_chunk]:
            triple = (rel.subject_type, rel.relation, rel.object_type)
            if triple not in _ALLOWED_TRIPLES:
                continue  # discard non-conformant edges
            src = EntityNode(
                name=rel.subject,
                label=rel.subject_type,
                properties={"triplet_source_id": node.node_id},
            )
            dst = EntityNode(
                name=rel.object,
                label=rel.object_type,
                properties={"triplet_source_id": node.node_id},
            )
            entities[src.id] = src
            entities[dst.id] = dst
            relations.append(
                Relation(
                    label=rel.relation,
                    source_id=src.id,
                    target_id=dst.id,
                    properties={"triplet_source_id": node.node_id},
                )
            )

        node.metadata[KG_NODES_KEY] = existing_nodes + list(entities.values())
        node.metadata[KG_RELATIONS_KEY] = existing_rels + relations


# ── Graph store connection ────────────────────────────────────────────────────

# Neo4j is optional: the graph layer only exists when the docker-compose service
# is running. The driver's defaults assume a transient outage and retry with
# exponential backoff for 30s, which turns "Docker isn't started" into a half
# minute of stalled requests and log noise. Fail fast instead — an absent
# optional service should report itself immediately.
NEO4J_CONNECT_TIMEOUT_S = 3.0
NEO4J_RETRY_TIMEOUT_S = 2.0


def _graph_store(*, fail_fast: bool = False) -> Neo4jPropertyGraphStore:
    """Connect to the local Neo4j (docker-compose). Credentials from config/.env.

    Args:
        fail_fast: use short timeouts, for read-only status checks where a slow
            failure is worse than no answer. Leave False for index builds, which
            are long-running and should tolerate a blip.
    """
    # neo4j_kwargs is **kwargs on the store and is forwarded to the driver, so
    # these go in at the top level rather than nested under a dict.
    driver_kwargs: dict = {}
    if fail_fast:
        driver_kwargs = {
            "connection_timeout": NEO4J_CONNECT_TIMEOUT_S,
            "max_transaction_retry_time": NEO4J_RETRY_TIMEOUT_S,
        }

    return Neo4jPropertyGraphStore(
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        url=NEO4J_URI,
        database=NEO4J_DATABASE,
        **driver_kwargs,
    )


# ── Document loading (mirrors rag/indexer.build_index) ────────────────────────

def _pdf_paths(limit: int | None = None) -> list[Path]:
    paths = sorted(OUTPUT_DIR.glob("*.pdf"))
    return paths[:limit] if limit else paths


def _load_documents(limit: int | None = None) -> list[Document]:
    """Load PDFs (+ sidecar abstracts) for the selected files, same as the vector side."""
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_FILE}")

    metadata_by_uid: dict[str, dict] = json.loads(METADATA_FILE.read_text())
    lookup = _path_lookup_from_metadata(metadata_by_uid)

    paths = _pdf_paths(limit)
    if not paths:
        raise FileNotFoundError(f"No PDFs found in {OUTPUT_DIR}")

    reader = SimpleDirectoryReader(
        input_files=[str(p) for p in paths],
        file_metadata=_file_metadata_fn(lookup),
    )
    documents: list[Document] = reader.load_data()

    # Include only sidecars belonging to the selected PDFs.
    selected_names = {p.name for p in paths}
    filtered_meta = {
        uid: rec
        for uid, rec in metadata_by_uid.items()
        if Path(rec.get("local_path", "")).name in selected_names
    }
    sidecar_docs = _load_sidecar_documents(filtered_meta)
    if sidecar_docs:
        logger.info("Loaded %d sidecar abstract documents", len(sidecar_docs))
        documents.extend(sidecar_docs)

    logger.info("Loaded %d documents from %d PDFs", len(documents), len(paths))
    return documents


# ── Build ─────────────────────────────────────────────────────────────────────

def build_graph(
    *,
    limit: int | None = None,
    reset: bool = True,
) -> PropertyGraphIndex:
    """
    Extract the knowledge graph from the ENISA PDFs into Neo4j.

    Args:
        limit:  only process the first N PDFs (fast iteration). None = all.
        reset:  wipe existing graph first, so a rebuild is idempotent
                (mirrors the full-rebuild semantics of --rebuild-index).

    NOTE: extraction runs the local LLM over every chunk (~18s/chunk on
    qwen2.5:3b), so a full 10-PDF build takes a while. Use --graph-limit to
    smoke-test on fewer PDFs.
    """
    _configure_graph_settings()
    documents = _load_documents(limit=limit)
    graph_store = _graph_store()

    if reset:
        logger.info("Resetting existing graph (DETACH DELETE all nodes)")
        graph_store.structured_query("MATCH (n) DETACH DELETE n")

    kg_extractor = SchemaEnumPathExtractor(
        llm=Settings.llm,
        max_triplets_per_chunk=MAX_TRIPLETS_PER_CHUNK,
    )

    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    logger.info("Building property graph over %d documents...", len(documents))
    index = PropertyGraphIndex.from_documents(
        documents,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        kg_extractors=[kg_extractor],
        property_graph_store=graph_store,
        embed_kg_nodes=True,
        transformations=[splitter],
        use_async=False,  # serialize extraction (see SchemaEnumPathExtractor)
        show_progress=True,
    )
    stats = get_graph_stats()
    logger.info(
        "Graph built: %d nodes, %d relationships",
        stats["nodes_total"],
        stats["relationships_total"],
    )
    return index


def load_graph_index() -> PropertyGraphIndex:
    """Reconnect to the existing Neo4j graph (used by the Phase 2 retrievers)."""
    _configure_graph_settings()
    graph_store = _graph_store()
    return PropertyGraphIndex.from_existing(
        property_graph_store=graph_store,
        llm=Settings.llm,
        embed_model=Settings.embed_model,
        embed_kg_nodes=True,
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_graph_stats() -> dict[str, Any]:
    """
    Node/edge counts by type — powers GET /graph/stats.

    LlamaIndex labels entity nodes with `__Entity__` plus their domain type; the
    internal `__*__` labels are filtered out so only domain types are reported.

    Read-only status check, so it fails fast rather than retrying: Neo4j being
    absent is the normal state when Docker is not running.
    """
    graph_store = _graph_store(fail_fast=True)

    nodes_by_type: dict[str, int] = {}
    for row in graph_store.structured_query(
        """
        MATCH (n:`__Entity__`)
        UNWIND labels(n) AS label
        WITH label WHERE NOT label STARTS WITH '__'
        RETURN label AS type, count(*) AS count
        ORDER BY count DESC
        """
    ):
        nodes_by_type[row["type"]] = row["count"]

    rels_by_type: dict[str, int] = {}
    for row in graph_store.structured_query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY count DESC
        """
    ):
        rels_by_type[row["type"]] = row["count"]

    totals = graph_store.structured_query(
        "MATCH (n) WITH count(n) AS nodes "
        "MATCH ()-[r]->() RETURN nodes, count(r) AS rels"
    )
    nodes_total = totals[0]["nodes"] if totals else 0
    rels_total = totals[0]["rels"] if totals else 0

    return {
        "nodes_total": nodes_total,
        "relationships_total": rels_total,
        "nodes_by_type": nodes_by_type,
        "relationships_by_type": rels_by_type,
    }
