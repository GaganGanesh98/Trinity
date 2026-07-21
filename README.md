# ENISA RAG Agent

Local-first retrieval-augmented QA over **ENISA** cybersecurity publications
(threat landscapes, NIS2 guidance, AI-security frameworks). Everything runs on your
machine via [Ollama](https://ollama.com) and a local Neo4j — **no API keys, no paid
services**.

The project has two retrieval backbones over the *same* corpus:

1. **Vector RAG** — `VectorStoreIndex` with hybrid **BM25 + vector** fusion and
   cross-encoder reranking (`rag/indexer.py`).
2. **GraphRAG** — a **Neo4j** knowledge graph of cybersecurity entities and
   relationships, extracted with a schema-constrained LLM (`rag/graph_indexer.py`).

A FastAPI service exposes both, and an eval harness scores retrieval quality.

## Requirements

- Python 3.11+ (`venv/` provided)
- [Ollama](https://ollama.com) running locally with:
  ```bash
  ollama pull llama3.2          # generation (answers)
  ollama pull nomic-embed-text  # embeddings (vector store + graph nodes)
  ollama pull qwen2.5:3b        # graph entity/relation extraction (GraphRAG only)
  ```
  Extraction uses a structured-output model (`qwen2.5:3b`) so the knowledge graph
  stays schema-conformant; base llama3.2 can't reliably do strict schema
  extraction. The 3b size fits 8GB RAM and runs at usable speed (7b thrashes on
  low-memory machines). Override with the `GRAPH_LLM_MODEL` env var.
- Docker (for Neo4j) — only needed for the GraphRAG layer.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # local defaults are fine

# Vector store (ships prebuilt in rag_store/, rebuild if needed):
python main.py --rebuild-index

# Serve the API:
python -m uvicorn api:app --reload
# POST /query  ·  GET /health  ·  GET /graph/stats  ·  GET /docs
```

Configuration (corpus source, Ollama, Neo4j) lives in `config.py`; secrets in `.env`.

---

## GraphRAG

The GraphRAG layer adds a **graph database** alongside the vector store so the system can
answer questions that require *following relationships*, not just matching text.

### 1. Schema

A domain-tuned cybersecurity schema keeps the graph **structured, not free-form**:

- **Entities:** `ThreatActor`, `Technique`, `Vulnerability`, `Malware`, `Sector`,
  `Mitigation`, `Regulation`
- **Relations:** `USES`, `TARGETS`, `MITIGATES`, `EXPLOITS`, `REFERENCES`

Extraction uses a custom LlamaIndex extractor (`SchemaEnumPathExtractor`) that constrains
the LLM to emit only these types via an enum-typed structured schema, then validates every
edge against a whitelist of `(subject, relation, object)` shapes — non-conformant edges are
dropped. (LlamaIndex's built-in `SchemaLLMPathExtractor(strict=True)` needs a 7B+ model to
work and is impractically slow on low-RAM machines; this flat-schema approach runs on a 3B
model in ~18s/chunk. See [`docs/graph-schema.md`](docs/graph-schema.md).)

### 2. Build the graph

```bash
# Start Neo4j (local, free — browse at http://localhost:7474):
docker compose up -d neo4j

# Extract entities + relationships from enisa_docs/ into Neo4j:
python main.py --rebuild-graph              # all PDFs
python main.py --rebuild-graph --graph-limit 2   # fast smoke test (first 2 PDFs)
```

`--rebuild-graph` mirrors `--rebuild-index`: it rebuilds the store from `enisa_docs/`
without re-scraping. Extraction runs the local LLM over every chunk, so a full build is
slow — use `--graph-limit` while iterating. The build is idempotent (it resets the graph
first).

### 3. Inspect it

```bash
curl -s localhost:8000/graph/stats | python -m json.tool
```

```jsonc
{
  "nodes_total": 128,
  "relationships_total": 214,
  "nodes_by_type": { "Technique": 41, "ThreatActor": 22, "Sector": 12, "...": 0 },
  "relationships_by_type": { "USES": 63, "TARGETS": 48, "MITIGATES": 31, "...": 0 }
}
```

Or explore visually in the Neo4j browser at <http://localhost:7474>
(user `neo4j`, password from `.env`).

### Retrieval modes & evaluation

> **Roadmap.** Phase 1 (this) ships the graph store, construction, and `/graph/stats`.
> Phase 2 adds graph + hybrid retrievers to `/query` (`mode = vector | graph | hybrid`).
> Phase 3 adds a three-mode eval table over `eval_questions.json` (including multi-hop
> questions), quantifying where graph/hybrid beats vector-only.
