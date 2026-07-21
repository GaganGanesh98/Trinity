"""
Fetch publication listings and individual publication pages from ENISA.

Includes structured metadata extraction (abstract, topics) for sidecar storage.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from config import ACTIVE_SOURCE, HEADERS

BASE_URL = ACTIVE_SOURCE.base_url

# ENISA /publications — "All publications" Drupal view
PUBLICATION_CARD_SELECTOR = "div.view-publications-index div.publications-item"
TITLE_LINK_SELECTOR = ".publication-content h3 a"
DATE_SELECTOR = "p.metadata time"
DESC_SELECTOR = ".publication-content .field--name-field-summary, .publication-content p.summary"


def _publications_from_html(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    publications: list[dict] = []
    for item in soup.select(PUBLICATION_CARD_SELECTOR):
        a_tag = item.select_one(TITLE_LINK_SELECTOR)
        if not a_tag or not a_tag.get("href"):
            continue
        title = a_tag.get_text(strip=True)
        link = urljoin(BASE_URL, a_tag["href"].strip())
        time_el = item.select_one(DATE_SELECTOR)
        date = ""
        if time_el:
            date = (time_el.get("datetime") or time_el.get_text(strip=True) or "").strip()
        # Try to grab the description/summary from the listing card
        desc_el = item.select_one(DESC_SELECTOR)
        description = desc_el.get_text(strip=True) if desc_el else ""
        publications.append({
            "title": title,
            "url": link,
            "date": date or "unknown",
            "description": description,
        })
    return publications


def _pdf_url_from_html(html: str) -> str | None:
    """
    Extract the primary PDF link from a publication page.

    Preference order:
    1. href matching the source's pdf_href_pattern (e.g. /sites/default/files/...pdf)
    2. Any href ending in .pdf
    Avoids false positives from generic "download" links.
    """
    soup = BeautifulSoup(html, "lxml")
    pattern = re.compile(ACTIVE_SOURCE.pdf_href_pattern, re.IGNORECASE)

    # Pass 1: prefer the canonical ENISA file path
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("?", 1)[0]
        if pattern.search(href):
            return urljoin(BASE_URL, a["href"].strip())

    # Pass 2: any .pdf link
    for a in soup.find_all("a", href=True):
        href = a["href"].strip().split("?", 1)[0]
        if href.lower().endswith(".pdf"):
            return urljoin(BASE_URL, a["href"].strip())

    return None


def extract_page_metadata(html: str) -> dict:
    """
    Pull structured metadata from a publication landing page:
    - abstract (first substantial paragraph or meta description)
    - topics (from tag links / taxonomy terms)
    - h1 title
    """
    soup = BeautifulSoup(html, "lxml")
    meta: dict = {}

    # Title from <h1>
    h1 = soup.select_one("h1")
    if h1:
        meta["title"] = h1.get_text(strip=True)

    # Abstract: meta description, or first <div class="field--name-body"> paragraph
    desc_tag = soup.find("meta", attrs={"name": "description"})
    if desc_tag and desc_tag.get("content"):
        meta["abstract"] = desc_tag["content"].strip()
    else:
        body_field = soup.select_one(".field--name-body, .field--name-field-summary")
        if body_field:
            first_p = body_field.find("p")
            if first_p:
                meta["abstract"] = first_p.get_text(strip=True)

    # Topics: ENISA uses taxonomy term links
    topic_links = soup.select(
        ".field--name-field-topics a, "
        ".field--name-field-tags a, "
        'a[href*="/topics/"]'
    )
    topics = list({a.get_text(strip=True) for a in topic_links if a.get_text(strip=True)})
    if topics:
        meta["topics"] = sorted(topics)

    return meta


def _html_playwright(
    url: str,
    *,
    wait_selector: str | None = None,
) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise ImportError(
            "playwright is required for JS-rendered pages. "
            "Install with: pip install playwright && playwright install chromium"
        ) from e

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            context = browser.new_context(extra_http_headers=dict(HEADERS))
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, timeout=20_000)
                except Exception:
                    pass
            return page.content()
        finally:
            browser.close()


def fetch_publication_links(url: str) -> list[dict]:
    """Fetch listing page → list of {title, url, date, description}."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    pubs = _publications_from_html(r.text)
    if pubs:
        return pubs
    html = _html_playwright(url, wait_selector=PUBLICATION_CARD_SELECTOR)
    return _publications_from_html(html)


def get_pdf_url_from_page(page_url: str) -> tuple[str | None, dict]:
    """
    Fetch a publication page → (pdf_url, page_metadata).

    page_metadata contains abstract, topics, h1 title extracted from the HTML.
    """
    r = requests.get(page_url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    html = r.text
    pdf = _pdf_url_from_html(html)
    page_meta = extract_page_metadata(html)

    if not pdf:
        html = _html_playwright(page_url, wait_selector=None)
        pdf = _pdf_url_from_html(html)
        if not page_meta:
            page_meta = extract_page_metadata(html)

    return pdf, page_meta
