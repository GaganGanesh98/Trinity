"""
HTTP surface.

Covers the endpoints that need no model: scope determination, the catalogues the
UI builds itself from, and the rejection paths. Endpoints that invoke the LLM are
covered by nis2/benchmark.py instead — they take minutes, which does not belong
in a suite meant to run on every change.

The index load is stubbed so startup is instant; none of these routes need it.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    import api

    # Startup normally loads the vector index (~20s). No route under test needs
    # it, so stub it out rather than pay for it on every run.
    api.load_index = lambda: None
    with TestClient(api.app) as c:
        yield c


# ── Scope ────────────────────────────────────────────────────────────────────

def test_large_hospital_is_essential(client):
    r = client.post("/scope", json={
        "sector": "health", "employees": 800,
        "annual_turnover_eur": 90_000_000, "balance_sheet_eur": 70_000_000,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["classification"] == "ESSENTIAL"
    assert body["in_scope"] is True
    assert body["obligations"] and body["reasoning"]
    assert "not legal advice" in body["disclaimer"].lower()


def test_micro_dns_provider_is_essential(client):
    r = client.post("/scope", json={
        "sector": "digital infrastructure", "employees": 6,
        "flags": {"is_dns_or_tld_provider": True},
    })
    assert r.json()["classification"] == "ESSENTIAL"


def test_small_food_business_is_out_of_scope(client):
    r = client.post("/scope", json={
        "sector": "food", "employees": 12, "annual_turnover_eur": 1_500_000,
    })
    body = r.json()
    assert body["in_scope"] is False
    assert body["obligations"] == []
    assert body["caveats"], "an out-of-scope answer must still explain the exceptions"


def test_sector_is_required(client):
    assert client.post("/scope", json={"employees": 800}).status_code == 422


def test_negative_headcount_is_rejected(client):
    r = client.post("/scope", json={"sector": "health", "employees": -5})
    assert r.status_code == 422


# ── Catalogues the UI depends on ─────────────────────────────────────────────

def test_sectors_catalogue(client):
    body = client.get("/scope/sectors").json()
    assert set(body) == {"annex_I", "annex_II", "size_exempt_criteria"}
    assert body["annex_I"]["sectors"] and body["size_exempt_criteria"]


def test_checkpoints_catalogue(client):
    body = client.get("/assess/checkpoints").json()
    assert len(body) == 13, "thirteen Article 21(2) requirement domains"
    ids = [c["id"] for c in body]
    assert ids == sorted(ids), "stable, ordered ids"
    for c in body:
        assert c["article"].startswith("Art. 21(2)") or "Art. 23" in c["article"]
        assert c["evidence_expected"]


# ── Upload and URL rejection paths ───────────────────────────────────────────

def test_empty_upload_is_rejected(client):
    r = client.post("/assess", files={"file": ("empty.txt", b"", "text/plain")})
    assert r.status_code == 400


def test_unsupported_file_type_is_rejected(client):
    r = client.post("/assess", files={"file": ("policy.docx", b"PK\x03\x04data", "application/octet-stream")})
    assert r.status_code == 400


def test_oversized_upload_is_rejected(client):
    import api

    r = client.post("/assess", files={
        "file": ("big.txt", b"x" * (api.MAX_UPLOAD_BYTES + 1), "text/plain")
    })
    assert r.status_code == 413


def test_unknown_checkpoint_id_is_rejected(client):
    r = client.post(
        "/assess",
        files={"file": ("p.txt", b"A security policy exists and is reviewed.", "text/plain")},
        data={"checkpoints": "NIS2-99"},
    )
    assert r.status_code == 400


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/policy.pdf",
    "http://169.254.169.254/latest/meta-data/",
    "file:///etc/passwd",
])
def test_ssrf_targets_are_rejected_at_the_http_layer(client, url):
    """The guard must hold through the endpoint, not only in the fetch module."""
    r = client.post("/assess/url", json={"url": url})
    assert r.status_code == 400


# ── Pages and status ─────────────────────────────────────────────────────────

def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


@pytest.mark.parametrize("path", ["/", "/nis2", "/about", "/privacy", "/terms"])
def test_pages_render(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


@pytest.mark.parametrize("path,active", [
    ("/about", "About"), ("/privacy", "Privacy"), ("/terms", "Terms"),
])
def test_legal_pages_mark_their_own_tab(client, path, active):
    html = client.get(path).text
    assert f'href="{path}"   aria-current="page"' in html or \
           f'href="{path}" aria-current="page"' in html


def test_every_page_has_a_home_link(client):
    for path in ["/", "/nis2", "/about", "/privacy", "/terms"]:
        assert 'aria-label="Home"' in client.get(path).text


def test_openapi_endpoints_are_grouped_and_named(client):
    """Operation names come from summary=; without it FastAPI exposes the Python
    function name, which is an internal identifier and not a product name."""
    spec = client.get("/openapi.json").json()
    for path, ops in spec["paths"].items():
        for method, op in ops.items():
            assert op.get("tags"), f"{method.upper()} {path} is ungrouped"
            assert op.get("summary"), f"{method.upper()} {path} has no summary"
            assert not op["summary"].startswith("Handle "), \
                f"{method.upper()} {path} exposes a function name: {op['summary']!r}"
