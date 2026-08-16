# Trinity

**Three retrieval architectures over one corpus — hybrid BM25+vector, Neo4j
GraphRAG, and MongoDB Atlas Vector Search — scored by the same eval harness.**

Retrieval-augmented QA over **ENISA** cybersecurity publications (threat
landscapes, NIS2 guidance, AI-security frameworks). The local backends run
entirely on your machine via [Ollama](https://ollama.com) and a local Neo4j —
**no API keys, no paid services**; the Atlas backend is optional and off by
default.

The name is the point: three retrieval backbones over the *same* corpus, so the
question "does the choice of retrieval architecture actually matter?" can be
answered by measurement rather than assertion.

1. **Vector RAG** — `VectorStoreIndex` with hybrid **BM25 + vector** fusion and
   cross-encoder reranking (`rag/indexer.py`).
2. **GraphRAG** — a **Neo4j** knowledge graph of cybersecurity entities and
   relationships, extracted with a schema-constrained LLM (`rag/graph_indexer.py`).
3. **MongoDB Atlas Vector Search** — the same chunks in a managed cloud vector
   store, with `$vectorSearch` + `$search` fused **server-side** by Atlas
   (`rag/mongo_indexer.py`). Optional; off unless `MONGODB_URI` is set.

A FastAPI service exposes them, and an eval harness scores retrieval quality
per backend.

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

---

## MongoDB Atlas Vector Search

The only backend here that is **not** local-first. It exists to be *compared*
against the local vector store, not to replace it: same corpus, same chunk size,
same `nomic-embed-text` embeddings, same candidate count, same cross-encoder
reranker. Exactly one thing changes — **where the similarity search runs**.

| | local | atlas |
| --- | --- | --- |
| Chunk storage | `./rag_store` (on disk) | Atlas collection |
| Lexical search | BM25, in-process | Atlas `$search` |
| Vector search | in-process | Atlas `$vectorSearch` |
| Fusion | LlamaIndex, in Python | Reciprocal Rank Fusion, server-side |

**Caveat worth stating in any writeup:** embeddings are still generated locally
by Ollama, so every Atlas query pays a network round trip the local store does
not. Treat retrieval quality as the primary metric and latency as directional.

### 1. Set up the cluster

In the [Atlas UI](https://cloud.mongodb.com), on a free **M0** cluster:

1. **Database Access** → *Add New Database User* → password auth. Give it
   *Read and write to any database*.
2. **Network Access** → *Add IP Address* → *Add Current IP Address*.
   (Re-add it when your IP changes — a stalled connection is usually this.)
3. **Clusters** → *Connect* → *Drivers* → *Python*, and copy the connection
   string.

Put it in `.env` (never commit it — `.env` is gitignored):

```bash
MONGODB_URI="mongodb+srv://<user>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority"
```

Percent-encode any of `@ : / ? # [ ]` in the password, or the URI won't parse.

Verify before indexing anything:

```bash
python main.py --mongo-status
```

### 2. Build the index

```bash
python main.py --rebuild-mongo
```

This embeds `enisa_docs/` locally, writes the chunks to Atlas, then creates two
search indexes (`vector_index` for `$vectorSearch`, `fulltext_index` for
`$search`). Index creation is idempotent and waits for the build to finish.

The vector index declares **768 dimensions** (`nomic-embed-text`). Atlas rejects
vectors of a different size, so changing the embedding model means dropping the
index and rebuilding.

### 3. Compare the backends

```bash
python eval.py --backend local
python eval.py --backend atlas
python eval.py --backend all --out results.json   # both + markdown table
```

### Results

Both backends, `eval_questions.json` (5 questions, 7 expected answer strings):

| Backend | Source recall | Answer recall |
| --- | --- | --- |
| local | 100% (5/5) | 100% (7/7) |
| atlas | 100% (5/5) | 100% (7/7) |

**This is a ceiling effect, not a tie on merit.** Every question names its target
document almost verbatim ("...in the ENISA Threat Landscape 2024"), so any working
retriever finds it. The current eval set has **no power to discriminate** between
backends — a 100%/100% draw means the measurement failed, not that the backends
are equivalent.

Retrieval latency, measured **without** the LLM and interleaved between backends so
machine-level drift hits both equally:

| Backend | Warm retrieval |
| --- | --- |
| local | ~0.15s |
| atlas | ~0.26s |

Atlas costs ~0.11s more per query — about what a round trip to the Frankfurt
cluster should cost. That is ~0.2% of end-to-end time, which local Ollama
generation (50–90s per question) dominates entirely.

Note that `eval.py`'s own `avg_latency_s` is **end-to-end** and should not be read
as a backend comparison: it is mostly generation time, and because the backends run
sequentially, whichever runs second absorbs more accumulated machine pressure. In
the recorded run that inflated `atlas` to 137s against `local`'s 58s — an artifact
of ordering, not a property of either store. Compare the retrieval-only numbers
above instead.

### Making the eval discriminate

The next step is a harder question set, targeting the axes where the backends
should actually diverge:

- **Lexical-vs-semantic splits** — questions phrased in vocabulary absent from the
  source text, where BM25 should fail and dense retrieval should win.
- **Multi-hop questions** — "which sectors are targeted by actors using technique
  X" — where the Neo4j graph should beat both vector backends.
- **Unanswerable questions**, to measure false-positive retrieval.

### Retrieval modes & evaluation

> **Roadmap.** Phase 1 (this) ships the graph store, construction, and `/graph/stats`.
> Phase 2 adds graph + hybrid retrievers to `/query` (`mode = vector | graph | hybrid`).
> Phase 3 adds a three-mode eval table over `eval_questions.json` (including multi-hop
> questions), quantifying where graph/hybrid beats vector-only.
