"""
Download helpers — PDF fetching, metadata persistence, sidecar extraction.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from config import HEADERS, METADATA_FILE, OUTPUT_DIR


# ── Metadata persistence ────────────────────────────────────────────────────

def load_metadata() -> dict:
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata(meta: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_FILE.write_text(json.dumps(meta, indent=2))


# ── Hashing / dedup ─────────────────────────────────────────────────────────

def url_hash(url: str) -> str:
    """Truncated SHA-256 — safe for dedup keys and audit logs."""
    return hashlib.sha256(url.encode()).hexdigest()[:12]


def already_downloaded(url: str, metadata: dict) -> bool:
    return url_hash(url) in metadata


# ── Download ─────────────────────────────────────────────────────────────────

def download_pdf(pdf_url: str, dest_path: Path) -> bool:
    r = requests.get(pdf_url, headers=HEADERS, stream=True, timeout=60)
    r.raise_for_status()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    return True


def safe_filename(title: str, uid: str) -> str:
    clean = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title)
    return f"{clean[:80]}_{uid}.pdf"


# ── Sidecar: save page abstract + topics alongside the PDF ───────────────────

def save_sidecar(dest_pdf: Path, page_meta: dict) -> Path | None:
    """Write a .meta.json next to the PDF with abstract, topics, etc."""
    if not page_meta:
        return None
    sidecar_path = dest_pdf.with_suffix(".meta.json")
    sidecar_path.write_text(json.dumps(page_meta, indent=2))
    return sidecar_path


# ── Record download ──────────────────────────────────────────────────────────

def record_download(
    metadata: dict,
    uid: str,
    title: str,
    source_url: str,
    pdf_url: str,
    local_path: Path,
    date: str,
    *,
    topics: list[str] | None = None,
    abstract: str = "",
    source_type: str = "crawl",
) -> None:
    metadata[uid] = {
        "title": title,
        "source_url": source_url,
        "pdf_url": pdf_url,
        "local_path": str(local_path),
        "date": date,
        "topics": topics or [],
        "abstract": abstract,
        "source_type": source_type,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }
    save_metadata(metadata)