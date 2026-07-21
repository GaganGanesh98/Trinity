"""
ENISA RAG Agent — configuration.

SourceConfig makes the crawler pluggable: swap BASE_URL, KEYWORDS, SEED_URLS,
and TOPIC_URLS to point at any Drupal-style publication index.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env (if present) so secrets stay out of source. Safe no-op if absent.
load_dotenv()


# ── Paths ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("./enisa_docs")
RAG_STORE_DIR = Path("./rag_store")
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# ── HTTP ─────────────────────────────────────────────────────────────────────
HEADERS = {
    "User-Agent": "Mozilla/5.0 (research-bot; gagan@enisa-rag-agent)"
}
MAX_PAGES = 5
REQUEST_DELAY = 1.5


# ── Source configuration (pluggable per corpus) ──────────────────────────────
@dataclass
class SourceConfig:
    """Everything the crawler needs to target a single publication source."""

    name: str
    base_url: str
    reports_url: str

    # Keywords for relevance filtering on listing pages
    keywords: list[str] = field(default_factory=list)

    # Direct publication page URLs — fetched unconditionally (bypass keywords)
    seed_urls: list[str] = field(default_factory=list)

    # Topic-filtered listing views to crawl alongside the main index
    topic_urls: list[str] = field(default_factory=list)

    # PDF link pattern on publication pages (regex on href)
    pdf_href_pattern: str = r"/sites/default/files/.*\.pdf$"


ENISA_SOURCE = SourceConfig(
    name="enisa",
    base_url="https://www.enisa.europa.eu",
    reports_url="https://www.enisa.europa.eu/publications",
    keywords=[
        # Existing
        "threat landscape",
        "NIS2",
        "ransomware",
        "vulnerability",
        "cyber threat",
        "incident response",
        "supply chain",
        # AI additions
        "artificial intelligence",
        " ai ",
        "machine learning",
        "multilayer framework",
        "ai cybersecurity",
        "ai threat",
    ],
    seed_urls=[
        # Multilayer Framework for Good Cybersecurity Practices for AI (Jun 2023)
        "https://www.enisa.europa.eu/publications/multilayer-framework-for-good-cybersecurity-practices-for-ai",
        # AI Threat Landscape — "Artificial Intelligence Cybersecurity Challenges" (Dec 2020)
        "https://www.enisa.europa.eu/publications/artificial-intelligence-cybersecurity-challenges",
    ],
    topic_urls=[
        # "AI & Next Gen Technologies" topic filter
        "https://www.enisa.europa.eu/publications?f%5B0%5D=topics:530",
    ],
    pdf_href_pattern=r"/sites/default/files/.*\.pdf$",
)

# Active source — swap this to point the whole pipeline at a different corpus
ACTIVE_SOURCE: SourceConfig = ENISA_SOURCE

# Convenience aliases so existing imports keep working
BASE_URL = ACTIVE_SOURCE.base_url
REPORTS_URL = ACTIVE_SOURCE.reports_url
KEYWORDS = ACTIVE_SOURCE.keywords


# ── Local RAG (Ollama) — used by rag/indexer.py ─────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_REQUEST_TIMEOUT = 120.0
OLLAMA_LLM_REQUEST_TIMEOUT = 300.0


# ── Neo4j (graph store) — used by rag/graph_indexer.py ───────────────────────
# Secrets come from .env (see .env.example). Defaults target the local
# docker-compose Neo4j and must be overridden for any real deployment.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "enisa-graph-dev")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")