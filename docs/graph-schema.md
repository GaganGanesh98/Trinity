# Knowledge-graph schema (GraphRAG layer)

The GraphRAG layer extracts a **structured** cybersecurity knowledge graph from the
same ENISA PDFs the vector store uses, and persists it in Neo4j. The graph stays typed
rather than free-form (see *Extraction* below).

Source of truth: [`rag/graph_indexer.py`](../rag/graph_indexer.py)
(`ENTITIES`, `RELATIONS`, `VALIDATION_SCHEMA`).

## Entities

| Type            | Meaning                                                        |
| --------------- | -------------------------------------------------------------- |
| `ThreatActor`   | Adversary group / actor (e.g. ransomware gang, APT).           |
| `Technique`     | Attack technique or TTP (e.g. phishing, DDoS, supply-chain).   |
| `Vulnerability` | Weakness or CVE-class flaw that can be exploited.              |
| `Malware`       | Malicious software / tooling.                                  |
| `Sector`        | Targeted sector or domain (e.g. finance, public admin, space). |
| `Mitigation`    | Control, safeguard, or good-practice measure.                  |
| `Regulation`    | Regulatory / framework reference (e.g. NIS2, ENISA framework). |

## Relations

| Relation    | Shape (subject → object)                                                       |
| ----------- | ------------------------------------------------------------------------------ |
| `USES`      | `ThreatActor → Malware`, `ThreatActor → Technique`                             |
| `TARGETS`   | `ThreatActor → Sector`, `ThreatActor → Vulnerability`, `Technique → Sector`   |
| `EXPLOITS`  | `Malware → Vulnerability`, `Technique → Vulnerability`                         |
| `MITIGATES` | `Mitigation → Technique`, `Mitigation → Vulnerability`, `Mitigation → Malware` |
| `REFERENCES`| `Regulation → Mitigation`, `Regulation → Sector`, `Regulation → Technique`     |

Only these `(subject, relation, object)` triples are allowed. The full list lives in
`VALIDATION_SCHEMA`.

## Extraction

Extraction is done by `SchemaEnumPathExtractor` (a small custom LlamaIndex
`TransformComponent` in `rag/graph_indexer.py`), wired into a stock `PropertyGraphIndex`.
For each chunk it asks the LLM for a **flat structured object** whose `subject_type`,
`relation`, and `object_type` fields are constrained to the schema enums, then discards any
edge not in `VALIDATION_SCHEMA`.

Why not LlamaIndex's built-in `SchemaLLMPathExtractor(strict=True)`? Its complex union
schema needs a 7B+ model to produce anything, and such a model runs at ~2 tok/s on an 8GB
machine (minutes per chunk). The flat enum schema here is simple enough that a 3B model
(`qwen2.5:3b`) satisfies it reliably in ~18s/chunk — same structured, schema-conformant
result, but it actually runs locally. The generation/answer model stays `llama3.2`.

## Why these shapes: multi-hop retrieval

The schema is designed so that questions requiring **following relationships** — not just
matching text — become answerable by graph traversal. This is where graph/hybrid retrieval
should beat vector-only (measured in Phase 3).

Example question:

> *Which mitigations address techniques used by threat actors targeting the finance sector?*

This is a 3-hop traversal:

```
(Sector {name:"finance"})
    <-[:TARGETS]-  (ThreatActor)
    -[:USES]->     (Technique)
    <-[:MITIGATES]-(Mitigation)
```

As Cypher against the Neo4j store:

```cypher
MATCH (s:Sector)<-[:TARGETS]-(a:ThreatActor)-[:USES]->(t:Technique)<-[:MITIGATES]-(m:Mitigation)
WHERE toLower(s.name) CONTAINS 'finance'
RETURN DISTINCT a.name AS actor, t.name AS technique, m.name AS mitigation
```

A pure vector search can only surface chunks that *mention* finance; it cannot compose the
actor → technique → mitigation chain unless a single passage happens to state all of it.
The graph makes that composition explicit.

## Where it lives

- **Graph store:** Neo4j Community (local, via `docker-compose.yml`). Browse at
  <http://localhost:7474>.
- **Node labels:** entity nodes carry `:__Entity__` plus their domain-type label
  (e.g. `:ThreatActor`); source chunks are `:Chunk`. `GET /graph/stats` reports counts by
  domain type.
