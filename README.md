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

## Am I in scope? (`/scope`)

Scope determination is **rule-based with no LLM in the path**. It follows a small
number of published tests, so a deterministic implementation is reproducible,
instant, auditable, and cannot hallucinate. Every answer cites the article it
came from.

```bash
curl -X POST http://127.0.0.1:8000/scope -H 'Content-Type: application/json' \
  -d '{"sector":"health","employees":800,"annual_turnover_eur":90000000}'
```

The tests, in order: **sector** (Annex I or II, Art. 2(1)) → **size-cap
exemptions** (Art. 2(2) puts DNS/TLD, trust services, public e-comms, sole
national providers and central government in scope *regardless of size*) →
**size** (at least medium under Recommendation 2003/361/EC) → **classification**
(large + Annex I ⇒ essential, otherwise important).

That second test matters more than it sounds. A six-person DNS provider is an
**essential entity** despite being micro-sized — a size-only checker gets that
exactly backwards.

Out-of-scope answers still return caveats, because "no" is rarely the end of it:
Art. 2(2)(d)–(g) contain judgement-based tests no tool can decide for you,
Member States may extend scope in transposition, and entities in scope must
impose security requirements on their suppliers (Art. 21(2)(d)) — so the
obligations often arrive contractually anyway.

---

## NIS2 readiness assessment

Searching public regulation is not a product — a search engine already does it.
The useful question is the one no search engine can answer, because it needs a
public regulation *and* a private document read together:

> *Does **our** incident response policy meet the NIS2 reporting requirements?*

`POST /assess` takes a security policy, incident response plan or supplier
policy and returns a per-domain gap report.

```bash
curl -F "file=@data/samples/sample_security_policy.txt" \
     http://127.0.0.1:8000/assess | python -m json.tool
```

```jsonc
{
  "document_name": "sample_security_policy.txt",
  "coverage_pct": 42.3,
  "high_severity_gaps": 3,
  "findings": [
    {
      "checkpoint_id": "NIS2-04",
      "domain": "Business continuity and crisis management",
      "article": "Art. 21(2)(c)",
      "status": "PARTIAL",
      "severity": "MEDIUM",
      "rationale": "Backups are taken and retained, but the document does not state recovery objectives or evidence of restoration testing.",
      "excerpt": "Backups of all production servers are taken nightly and retained for 30 days.",
      "excerpt_verified": true
    }
  ]
}
```

Assess a single domain with `-F "checkpoints=NIS2-03"`. `GET /assess/checkpoints`
lists all thirteen with the evidence each expects.

### Linking a document instead of uploading

`POST /assess/url` takes a link. One HTTP fetch covers more sharing methods than
it first appears — a published policy page, a direct PDF, a Google Drive/Docs
"anyone with the link" share, a public Notion or Confluence page — all of which
are just URLs. Drive and Docs links are rewritten to their direct-download form,
and HTML is reduced to text before assessment.

```bash
curl -X POST http://127.0.0.1:8000/assess/url -H 'Content-Type: application/json' \
  -d '{"url":"https://example.com/security-policy.pdf"}'
```

Provider OAuth (private Drive/SharePoint files) would extend this to documents
that are not shareable by link, but that is a much larger build and unnecessary
for the common case.

**This endpoint is an SSRF surface** and is guarded accordingly: only `http(s)`,
every hostname resolved and rejected if it maps to loopback, private, link-local
or reserved space, cloud metadata endpoints refused by name, redirects followed
manually so each hop is re-validated, and the body size-capped while streaming.
Without those controls, a caller could use the server to reach internal hosts
they cannot see themselves.

### Browser UI

`GET /nis2` serves a single self-contained page with both tools — the scope
questionnaire and the policy assessment (upload or link). The sector list and
the Art. 2(2) criteria are fetched from `/scope/sectors` at load, so the form
cannot drift out of sync with the rules the backend actually applies.

### Benchmarking the assessment model

Finding quality is model-dependent, so it is measured the same way retrieval is,
against hand-labelled ground truth in `nis2/ground_truth.json`:

```bash
python -m nis2.benchmark --models llama3.2 qwen2.5:7b
```

Three things are scored separately, since a model can be right for the wrong
reason: **status accuracy**, **gap recall** (of genuinely deficient checkpoints,
how many escaped being marked ADDRESSED — a false clean bill of health is the
costly error), and **rationale recall** (did it name the specific omission, such
as the missing Art. 23 deadlines).

#### Results

| Model | Status accuracy | Gap recall | Rationale recall | Findings quoted | Time |
| --- | --- | --- | --- | --- | --- |
| llama3.2 | 69% | 100% | 67% | 62% | 103s |
| qwen2.5:7b | 23%* | 100% | 67% | 15% | 846s |

**Gap recall is 100% for both**, and it is the metric that matters most: neither
model ever handed out a false clean bill of health on a checkpoint that was
genuinely deficient. Errors run toward over-reporting gaps, which costs review
time rather than creating false assurance.

`llama3.2` is the default: more accurate *and* eight times faster. The bigger
model is not the better one here.

\* The qwen figure is confounded and should not be read as a judgement
comparison. It quoted verbatim on only 15% of findings — it paraphrases where
llama3.2 copies — so the quote-verification rule below fired on most of its
output. What is being measured there is quoting discipline, not reasoning. A
fair comparison needs fuzzy quote matching, which is not built yet.

#### What the benchmark changed

The first run scored 38%, and inspecting the misses showed the fault was in the
rule, not the model. Findings that claimed coverage without a quotable sentence
were demoted to `UNCLEAR`, but four of the five demotions were checkpoints the
document genuinely did not address. When a model cannot quote a single
supporting sentence, the likeliest explanation is that the control is absent —
not that it is worded ambiguously.

Retargeting the demotion to `NOT_ADDRESSED` lifted status accuracy from **38% to
69%** with gap recall unchanged at 100%. The cost is visible in the table: the
document *does* carry an approved, annually-reviewed security policy, and it is
now reported as a gap because llama3.2 failed to quote it. A false gap costs a
few minutes of review; a false pass is what makes a compliance tool dangerous.

### Where the checkpoints come from

The thirteen requirement domains are not invented here — they mirror the
structure ENISA's *Technical Implementation Guidance* (June 2025) uses to
decompose the risk-management measures of **NIS2 Article 21(2)**, a document
that is already in the corpus. Every finding can therefore point at a published
requirement rather than at a checklist we made up.

### Two guarantees that make it usable

**Nothing is stored.** The upload is chunked and embedded into an in-memory
index that is discarded when the request ends. It never enters the vector store,
and embeddings and generation both run on local Ollama — no third-party API sees
the document. For a company being asked to hand over its internal security
policy, that property matters more than any feature.

**Every quote is verified.** The model is asked for a verbatim excerpt, and the
excerpt is checked to be a real substring of the upload (whitespace-normalised,
since PDF extraction breaks lines mid-sentence). Anything invented is dropped.

Further, a finding claiming `ADDRESSED` or `PARTIAL` **without** a verifiable
quote is demoted to `UNCLEAR` and marked for manual review. This is not
hypothetical: on a policy containing no cryptography section at all, llama3.2
returned `PARTIAL — "covers key management (rotation and storage)"`. It could not
produce a quote, because there was nothing to quote. In compliance a fabricated
finding is worse than no tool, so unsupported claims are not presented as
evidence of coverage.

### Limits

- **Not legal advice, and not a compliance certificate.** It surfaces gaps for a
  human to review.
- **Model quality is the binding constraint.** llama3.2 (3B) reasons adequately
  about presence and absence but misses nuance — on the sample it flagged
  incident handling as `PARTIAL` without noting the missing Art. 23 24h/72h
  reporting deadlines, which is the most consequential gap in that document. A
  larger local model (`qwen2.5:7b`) improves this at the cost of speed and RAM.
- Roughly 13 seconds per checkpoint, so a full thirteen-domain run takes a few
  minutes on a laptop.

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
