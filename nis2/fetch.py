"""
Fetch a policy document from a URL, so users can link a document instead of
uploading a file.

One HTTP fetch covers more sharing methods than it first appears: a published
policy page, a direct PDF link, a Google Drive "anyone with the link" share, a
public Notion or Confluence page, and an anonymous SharePoint/OneDrive link are
all just URLs. Provider-specific OAuth integrations would add private documents
on top, but they are a much larger build and are not needed for the common case.

**Security note.** An endpoint that fetches arbitrary user-supplied URLs from the
server is a server-side request forgery (SSRF) primitive: without controls, a
caller can reach loopback, RFC1918 hosts, and cloud instance-metadata endpoints
that the server can see and they cannot. Every hop is therefore resolved and
checked against private address space *before* connecting, redirects are
followed manually so each new location is re-checked, and the response is size-
capped while streaming.
"""

from __future__ import annotations

import ipaddress
import logging
import re
import socket
from urllib.parse import parse_qs, urlparse, urlunparse

import requests

logger = logging.getLogger(__name__)

ALLOWED_SCHEMES = {"http", "https"}
MAX_BYTES = 10 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT_S = 15

# Cloud instance-metadata addresses. Blocked by the private-range check already,
# but named explicitly because they are the highest-value SSRF target.
_METADATA_HOSTS = {"169.254.169.254", "metadata.google.internal", "100.100.100.200"}

USER_AGENT = "Mozilla/5.0 (compatible; trinity-nis2-assessor/1.0)"


class FetchError(ValueError):
    """Raised for any URL we decline to fetch, or any fetch that fails."""


def _is_public_ip(host: str) -> bool:
    """Resolve `host` and require every answer to be a public address.

    All resolved addresses must pass: a hostname with one public and one private
    A record would otherwise let an attacker win the race.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise FetchError(f"Could not resolve host {host!r}: {e}") from None

    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    return True


def _validate(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise FetchError(f"Only http and https URLs are supported (got {parsed.scheme!r}).")
    if not parsed.hostname:
        raise FetchError("URL has no host.")
    if parsed.hostname.lower() in _METADATA_HOSTS:
        raise FetchError("Refusing to fetch cloud metadata endpoints.")
    if not _is_public_ip(parsed.hostname):
        raise FetchError(
            f"Refusing to fetch {parsed.hostname!r}: it resolves to a private, loopback "
            "or link-local address."
        )
    return url


def normalise_share_url(url: str) -> str:
    """
    Rewrite common share links into direct-download form.

    Google Drive and Docs share URLs render a viewer page rather than serving the
    file, so fetching them verbatim yields HTML chrome instead of the document.
    """
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()

    if host in ("drive.google.com", "www.drive.google.com"):
        # /file/d/<id>/view  or  /open?id=<id>
        m = re.search(r"/file/d/([A-Za-z0-9_-]+)", parsed.path)
        file_id = m.group(1) if m else (parse_qs(parsed.query).get("id") or [None])[0]
        if file_id:
            return f"https://drive.google.com/uc?export=download&id={file_id}"

    if host == "docs.google.com":
        # Export a Google Doc as plain text rather than the editor page.
        m = re.search(r"/document/d/([A-Za-z0-9_-]+)", parsed.path)
        if m:
            return f"https://docs.google.com/document/d/{m.group(1)}/export?format=txt"

    if host in ("github.com", "www.github.com") and "/blob/" in parsed.path:
        return urlunparse(parsed._replace(
            netloc="raw.githubusercontent.com", path=parsed.path.replace("/blob/", "/", 1)
        ))

    return url


def _html_to_text(html: bytes) -> bytes:
    """Strip a policy page down to readable text."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    # Collapse the blank-line runs that stripping tags leaves behind.
    return re.sub(r"\n{3,}", "\n\n", text).strip().encode()


def fetch_document(url: str) -> tuple[bytes, str]:
    """
    Fetch `url` and return (data, filename) ready for the assessor.

    HTML is reduced to text and returned as .txt; PDFs are returned as-is.
    Raises FetchError for anything we decline or that fails.
    """
    url = _validate(normalise_share_url(url.strip()))

    session = requests.Session()
    current = url
    for _ in range(MAX_REDIRECTS + 1):
        resp = session.get(
            current,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT_S,
            stream=True,
            allow_redirects=False,  # re-validate each hop ourselves
        )
        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise FetchError("Redirect without a Location header.")
            current = _validate(requests.compat.urljoin(current, location))
            continue
        break
    else:
        raise FetchError(f"Too many redirects (>{MAX_REDIRECTS}).")

    if resp.status_code != 200:
        raise FetchError(f"Fetch failed with HTTP {resp.status_code}.")

    # Declared length is a hint only; the streaming loop below is the real cap.
    declared = resp.headers.get("Content-Length")
    if declared and int(declared) > MAX_BYTES:
        raise FetchError(f"Document exceeds {MAX_BYTES // (1024 * 1024)} MB.")

    chunks, total = [], 0
    for chunk in resp.iter_content(8192):
        total += len(chunk)
        if total > MAX_BYTES:
            resp.close()
            raise FetchError(f"Document exceeds {MAX_BYTES // (1024 * 1024)} MB.")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise FetchError("Fetched document is empty.")

    ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
    name = (urlparse(current).path.rsplit("/", 1)[-1] or "document").strip()

    if ctype == "application/pdf" or data[:5] == b"%PDF-":
        return data, name if name.lower().endswith(".pdf") else f"{name or 'document'}.pdf"
    if ctype in ("text/html", "application/xhtml+xml"):
        return _html_to_text(data), f"{name or 'document'}.txt"
    if ctype.startswith("text/") or ctype in ("application/json", ""):
        return data, name if name.lower().endswith((".txt", ".md")) else f"{name or 'document'}.txt"

    raise FetchError(
        f"Unsupported content type {ctype!r}. Link a PDF, a plain-text file, or an HTML page."
    )
