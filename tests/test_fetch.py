"""
URL fetching: SSRF guard and share-link normalisation.

POST /assess/url turns a user-supplied string into a server-side HTTP request.
Without the guard, a caller can use the server to reach hosts only the server can
see — internal services and cloud instance metadata. Every case below is an
attack that must be refused before any connection is attempted.

These tests make no outbound network requests: refusals happen during validation,
and the addresses used are literals or resolve locally.
"""

from __future__ import annotations

import pytest

from nis2.fetch import FetchError, normalise_share_url


# ── SSRF guard ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "url,why",
    [
        ("http://127.0.0.1/",                          "loopback literal"),
        ("http://127.0.0.1:8000/admin",                "loopback with port"),
        ("http://localhost/",                          "loopback by name"),
        ("http://[::1]/",                              "IPv6 loopback"),
        ("http://169.254.169.254/latest/meta-data/",   "AWS/GCP metadata"),
        ("http://[fd00::1]/",                          "IPv6 unique-local"),
        ("http://10.0.0.5/",                           "RFC1918 10/8"),
        ("http://192.168.1.1/",                        "RFC1918 192.168/16"),
        ("http://172.16.0.1/",                         "RFC1918 172.16/12"),
        ("http://0.0.0.0/",                            "unspecified"),
    ],
)
def test_refuses_private_and_loopback_targets(url, why):
    from nis2.fetch import fetch_document

    with pytest.raises(FetchError):
        fetch_document(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://example.com/policy.pdf",
        "gopher://example.com/",
        "data:text/plain;base64,aGk=",
    ],
)
def test_refuses_non_http_schemes(url):
    from nis2.fetch import fetch_document

    with pytest.raises(FetchError, match="http"):
        fetch_document(url)


def test_metadata_host_is_refused_by_name():
    from nis2.fetch import fetch_document

    with pytest.raises(FetchError, match="metadata"):
        fetch_document("http://metadata.google.internal/computeMetadata/v1/")


def test_refusal_happens_before_any_connection(monkeypatch):
    """The guard must reject during validation, not after opening a socket."""
    import requests

    from nis2.fetch import fetch_document

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a request was attempted against a blocked host")

    monkeypatch.setattr(requests.Session, "get", explode)
    with pytest.raises(FetchError):
        fetch_document("http://169.254.169.254/latest/meta-data/")


# ── Share-link normalisation ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "given,expected",
    [
        ("https://drive.google.com/file/d/1AbC_dEf-123/view?usp=sharing",
         "https://drive.google.com/uc?export=download&id=1AbC_dEf-123"),
        ("https://drive.google.com/open?id=1AbC_dEf-123",
         "https://drive.google.com/uc?export=download&id=1AbC_dEf-123"),
        ("https://docs.google.com/document/d/1XyZ789/edit",
         "https://docs.google.com/document/d/1XyZ789/export?format=txt"),
        ("https://github.com/org/repo/blob/main/SECURITY.md",
         "https://raw.githubusercontent.com/org/repo/main/SECURITY.md"),
    ],
)
def test_rewrites_viewer_links_to_direct_downloads(given, expected):
    """A share URL serves the viewer page, not the file — fetching it verbatim
    yields HTML chrome instead of the document."""
    assert normalise_share_url(given) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/security-policy.pdf",
        "https://example.com/policies/",
        "https://github.com/org/repo",           # not a /blob/ path
    ],
)
def test_leaves_ordinary_urls_alone(url):
    assert normalise_share_url(url) == url


def test_normalisation_does_not_bypass_the_guard(monkeypatch):
    """A URL shaped like a share link must still be validated after rewriting.

    Normalisation runs first, so a rewrite that produced a blocked host — or a
    blocked host that merely looks like a share link — must still be refused
    without a connection being attempted.
    """
    import requests

    from nis2.fetch import fetch_document

    def explode(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("a request was attempted against a blocked host")

    monkeypatch.setattr(requests.Session, "get", explode)
    with pytest.raises(FetchError):
        fetch_document("http://127.0.0.1/org/repo/blob/main/SECURITY.md")
