"""
ENISA RAG Agent — scraper entrypoint.

Three fetch phases:
1. Seed URLs   — known publications fetched unconditionally (bypass keywords)
2. Topic URLs  — topic-filtered listing views (e.g. AI & Next Gen)
3. Main crawl  — paginated /publications with keyword filtering
"""

from __future__ import annotations

import argparse
import time

from config import ACTIVE_SOURCE, MAX_PAGES, OUTPUT_DIR, REQUEST_DELAY
from scraper.downloader import (
    already_downloaded,
    download_pdf,
    load_metadata,
    record_download,
    safe_filename,
    save_sidecar,
    url_hash,
)
from scraper.fetcher import fetch_publication_links, get_pdf_url_from_page
from scraper.relevance import is_relevant


# ── Phase 1: Seed URLs (unconditional) ───────────────────────────────────────

def _fetch_single(
    url: str,
    metadata: dict,
    *,
    dry_run: bool,
    delay: float,
    source_type: str,
) -> bool:
    """Fetch a single publication page → PDF. Returns True if downloaded."""
    if already_downloaded(url, metadata):
        print(f"  [dupe] {url}")
        return False

    try:
        pdf_url, page_meta = get_pdf_url_from_page(url)
    except Exception as e:
        print(f"  [fail] {url}: {e}")
        return False

    if not pdf_url:
        print(f"  [nopdf] {url}")
        return False

    title = page_meta.get("title") or url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ").title()
    uid = url_hash(url)
    dest = OUTPUT_DIR / safe_filename(title, uid)

    if dry_run:
        print(f"  [dry] {pdf_url} -> {dest.name}")
        return False

    try:
        download_pdf(pdf_url, dest)
        save_sidecar(dest, page_meta)
        record_download(
            metadata, uid, title, url, pdf_url, dest,
            page_meta.get("date", "unknown"),
            topics=page_meta.get("topics"),
            abstract=page_meta.get("abstract", ""),
            source_type=source_type,
        )
        print(f"  [ok] {dest.name}")
        time.sleep(delay)
        return True
    except Exception as e:
        print(f"  [download fail] {e}")
        return False


def fetch_seeds(metadata: dict, *, dry_run: bool, delay: float) -> int:
    """Phase 1: fetch SEED_URLS unconditionally."""
    seeds = ACTIVE_SOURCE.seed_urls
    if not seeds:
        return 0
    print(f"\n── Seeds ({len(seeds)} URLs) ──")
    return sum(
        _fetch_single(url, metadata, dry_run=dry_run, delay=delay, source_type="seed")
        for url in seeds
    )


# ── Phase 2: Topic URL listings ──────────────────────────────────────────────

def fetch_topic_listings(
    metadata: dict,
    *,
    dry_run: bool,
    delay: float,
    max_pages: int,
) -> int:
    """Phase 2: crawl topic-filtered listing views."""
    topic_urls = ACTIVE_SOURCE.topic_urls
    if not topic_urls:
        return 0
    count = 0
    for base_url in topic_urls:
        print(f"\n── Topic listing: {base_url} ──")
        for page_num in range(max_pages):
            sep = "&" if "?" in base_url else "?"
            page_url = f"{base_url}{sep}page={page_num}"
            try:
                pubs = fetch_publication_links(page_url)
            except Exception as e:
                print(f"  Listing fetch failed: {e}")
                break
            if not pubs:
                break
            for pub in pubs:
                fetched = _fetch_single(
                    pub["url"], metadata,
                    dry_run=dry_run, delay=delay, source_type="topic",
                )
                if fetched:
                    count += 1
    return count


# ── Phase 3: Main paginated crawl ────────────────────────────────────────────

def run_scraper(
    max_pages: int = MAX_PAGES,
    delay: float = REQUEST_DELAY,
    focus_keywords: list[str] | None = None,
    dry_run: bool = False,
) -> dict:
    metadata = load_metadata()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 — seeds
    fetch_seeds(metadata, dry_run=dry_run, delay=delay)

    # Phase 2 — topic listings
    fetch_topic_listings(metadata, dry_run=dry_run, delay=delay, max_pages=max_pages)

    # Phase 3 — main /publications crawl with keyword filter
    reports_url = ACTIVE_SOURCE.reports_url
    print(f"\n── Main crawl: {reports_url} (max {max_pages} pages) ──")

    for page_num in range(max_pages):
        page_url = f"{reports_url}?page={page_num}"
        print(f"\n[Page {page_num + 1}] {page_url}")

        try:
            pubs = fetch_publication_links(page_url)
        except Exception as e:
            print(f"  Failed to fetch listing: {e}")
            break

        if not pubs:
            print("  No publications found — end of listing or check CSS selectors.")
            break

        for pub in pubs:
            title = pub["title"]
            url = pub["url"]
            date = pub["date"]
            description = pub.get("description", "")

            if not is_relevant(title, description, focus_keywords=focus_keywords):
                print(f"  [skip] {title[:60]}")
                continue

            if already_downloaded(url, metadata):
                print(f"  [dupe] {title[:60]}")
                continue

            print(f"  [fetch] {title[:60]}")
            try:
                pdf_url, page_meta = get_pdf_url_from_page(url)
            except Exception as e:
                print(f"    Page fetch failed: {e}")
                continue

            if not pdf_url:
                print(f"    No PDF found on page.")
                continue

            uid = url_hash(url)
            dest = OUTPUT_DIR / safe_filename(title, uid)

            if dry_run:
                print(f"    [dry-run] would download: {pdf_url}")
                print(f"    [dry-run] -> {dest.name}")
            else:
                try:
                    download_pdf(pdf_url, dest)
                    save_sidecar(dest, page_meta)
                    record_download(
                        metadata, uid, title, url, pdf_url, dest, date,
                        topics=page_meta.get("topics"),
                        abstract=page_meta.get("abstract", ""),
                        source_type="crawl",
                    )
                    print(f"    -> {dest.name}")
                except Exception as e:
                    print(f"    Download failed: {e}")

            if not dry_run:
                time.sleep(delay)

    print(f"\nTotal in store: {len(metadata)} documents.")
    return metadata


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape ENISA publications into enisa_docs/ for RAG indexing."
    )
    parser.add_argument(
        "--keywords",
        default=None,
        metavar="LIST",
        help='Comma-separated focus themes (overrides config.KEYWORDS).',
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        metavar="N",
        help=f"Maximum listing pages to crawl (default: {MAX_PAGES}).",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Rebuild the LlamaIndex vector store from enisa_docs/.",
    )
    parser.add_argument(
        "--rebuild-graph",
        action="store_true",
        help="Build the Neo4j knowledge graph (GraphRAG layer) from enisa_docs/. "
        "Requires the Neo4j container (docker compose up -d neo4j).",
    )
    parser.add_argument(
        "--graph-limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N PDFs when building the graph (fast iteration).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded; skip actual downloads.",
    )
    parser.add_argument(
        "--seeds-only",
        action="store_true",
        help="Only fetch seed URLs, skip listing crawl.",
    )
    args = parser.parse_args()

    focus: list[str] | None = None
    if args.keywords is not None:
        focus = [k.strip() for k in args.keywords.split(",") if k.strip()]

    max_pages = args.max_pages if args.max_pages is not None else MAX_PAGES

    # A bare store rebuild (--rebuild-index / --rebuild-graph) skips the network
    # scrape, so you can re-index or rebuild the graph offline. Any scrape-shaped
    # flag re-enables scraping (scrape THEN rebuild), preserving prior behavior.
    store_rebuild = args.rebuild_index or args.rebuild_graph
    scrape_flags = (
        args.seeds_only
        or args.dry_run
        or args.keywords is not None
        or args.max_pages is not None
    )
    do_scrape = (not store_rebuild) or scrape_flags

    if do_scrape:
        if args.seeds_only:
            meta = load_metadata()
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fetch_seeds(meta, dry_run=args.dry_run, delay=REQUEST_DELAY)
            print(f"\nTotal in store: {len(meta)} documents.")
        else:
            run_scraper(
                max_pages=max_pages,
                focus_keywords=focus,
                dry_run=args.dry_run,
            )

    if args.rebuild_index:
        if args.dry_run:
            print("\nSkipping --rebuild-index (dry-run mode).")
        else:
            from rag.indexer import build_index

            print("\nRebuilding RAG (vector) index...")
            build_index()

    if args.rebuild_graph:
        if args.dry_run:
            print("\nSkipping --rebuild-graph (dry-run mode).")
        else:
            from rag.graph_indexer import build_graph

            print("\nBuilding knowledge graph (Neo4j)...")
            build_graph(limit=args.graph_limit)
