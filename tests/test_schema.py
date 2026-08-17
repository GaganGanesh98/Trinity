"""
Assessment schema: quote verification and report aggregation.

verify_excerpt is the anti-hallucination guarantee. If it returns True for text
that is not in the document, the product's central claim — that every finding is
backed by the customer's own words — is false. These tests exist to make that
failure loud.
"""

from __future__ import annotations

import pytest

from nis2.schema import Finding, Report, Severity, Status, verify_excerpt

DOC = """ACME LOGISTICS GMBH — INFORMATION SECURITY POLICY

4. BACKUPS
Backups of all production servers are taken nightly and retained for 30 days.
Backups are stored on a separate NAS device in the same building.
"""


# ── verify_excerpt ───────────────────────────────────────────────────────────

def test_accepts_a_real_quote():
    assert verify_excerpt("Backups of all production servers are taken nightly", DOC)


def test_rejects_an_invented_quote():
    assert not verify_excerpt("We perform quarterly penetration tests of all systems", DOC)


def test_tolerates_pdf_line_breaks_in_the_quote():
    """PDF extraction breaks sentences mid-line; a correctly copied quote must
    still verify, or the rule would punish honest models."""
    assert verify_excerpt("Backups of all production servers\nare taken   nightly", DOC)


def test_is_case_insensitive():
    assert verify_excerpt("BACKUPS OF ALL PRODUCTION SERVERS ARE TAKEN NIGHTLY", DOC)


@pytest.mark.parametrize("excerpt", ["", "   ", "Backups", "the same"])
def test_rejects_empty_and_too_short(excerpt):
    """Short fragments match by coincidence and prove nothing."""
    assert not verify_excerpt(excerpt, DOC)


def test_rejects_a_quote_stitched_from_separate_places():
    """Words that all appear in the document, in an order it never uses."""
    assert not verify_excerpt(
        "Backups of all production servers are tested quarterly for restoration", DOC
    )


def test_min_chars_is_configurable():
    assert verify_excerpt("30 days", DOC, min_chars=5)
    assert not verify_excerpt("30 days", DOC, min_chars=25)


# ── Report aggregation ───────────────────────────────────────────────────────

def _finding(cid, status, severity=Severity.MEDIUM):
    return Finding(
        checkpoint_id=cid, domain=f"d-{cid}", article="Art. 21(2)(a)",
        obligation="o", status=status, severity=severity, rationale="r", excerpt="",
    )


def test_counts_and_coverage():
    r = Report(document_name="p.txt", findings=[
        _finding("A", Status.ADDRESSED, Severity.NONE),
        _finding("B", Status.ADDRESSED, Severity.NONE),
        _finding("C", Status.PARTIAL),
        _finding("D", Status.NOT_ADDRESSED, Severity.HIGH),
    ]).finalise()

    assert (r.total, r.addressed, r.partial, r.not_addressed) == (4, 2, 1, 1)
    assert r.high_severity_gaps == 1
    # PARTIAL counts half: better than silence, worse than compliance.
    assert r.coverage_pct == pytest.approx(62.5)


def test_worst_findings_sort_first():
    r = Report(document_name="p.txt", findings=[
        _finding("A", Status.ADDRESSED, Severity.NONE),
        _finding("B", Status.UNCLEAR),
        _finding("C", Status.NOT_ADDRESSED, Severity.HIGH),
        _finding("D", Status.PARTIAL),
    ]).finalise()
    assert [f.status for f in r.findings] == [
        Status.NOT_ADDRESSED, Status.PARTIAL, Status.UNCLEAR, Status.ADDRESSED
    ]


def test_severity_orders_within_a_status():
    r = Report(document_name="p.txt", findings=[
        _finding("A", Status.NOT_ADDRESSED, Severity.LOW),
        _finding("B", Status.NOT_ADDRESSED, Severity.HIGH),
        _finding("C", Status.NOT_ADDRESSED, Severity.MEDIUM),
    ]).finalise()
    assert [f.severity for f in r.findings] == [Severity.HIGH, Severity.MEDIUM, Severity.LOW]


def test_addressed_findings_are_not_counted_as_gaps():
    r = Report(document_name="p.txt", findings=[
        _finding("A", Status.ADDRESSED, Severity.HIGH),  # severity is stale, status wins
    ]).finalise()
    assert r.high_severity_gaps == 0


def test_empty_report_does_not_divide_by_zero():
    r = Report(document_name="p.txt", findings=[]).finalise()
    assert r.total == 0 and r.coverage_pct == 0.0


def test_full_coverage_is_100():
    r = Report(document_name="p.txt", findings=[
        _finding(c, Status.ADDRESSED, Severity.NONE) for c in "AB"
    ]).finalise()
    assert r.coverage_pct == 100.0
