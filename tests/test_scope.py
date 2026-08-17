"""
Scope determination.

These assertions encode the directive's tests, not the implementation's current
behaviour — if a refactor changes an answer here, the answer is wrong, not the
test. Each case names the article it comes from.
"""

from __future__ import annotations

import pytest

from nis2.scope import (
    Classification,
    ScopeInput,
    Size,
    classify_size,
    determine,
    sectors,
)


# ── Size classification (Recommendation 2003/361/EC, applied by Art. 2(1)) ────

@pytest.mark.parametrize(
    "employees,turnover,balance,expected",
    [
        (9, 1e6, 1e6, Size.MICRO),
        (10, 1e6, 1e6, Size.SMALL),      # headcount is the binding criterion
        (49, 9e6, 9e6, Size.SMALL),
        (50, 9e6, 9e6, Size.MEDIUM),     # boundary into medium
        (249, 49e6, 42e6, Size.MEDIUM),
        (250, 1e6, 1e6, Size.LARGE),     # headcount alone promotes to large
        (100, 80e6, 80e6, Size.LARGE),   # financials alone promote to large
    ],
)
def test_size_boundaries(employees, turnover, balance, expected):
    assert classify_size(employees, turnover, balance) is expected


def test_size_unknown_without_headcount():
    assert classify_size(None, 10e6, 10e6) is None


def test_missing_financials_do_not_promote():
    """Absent data must not silently push an entity into scope.

    The financial ceilings are alternatives to each other, so an unknown value
    is treated as within the ceiling rather than as exceeding it.
    """
    assert classify_size(30, None, None) is Size.SMALL


def test_one_financial_within_ceiling_is_enough():
    # Turnover over the small ceiling but balance sheet under it: still small.
    assert classify_size(30, 12e6, 8e6) is Size.SMALL


# ── Classification (Art. 3) ──────────────────────────────────────────────────

def test_large_annex_i_is_essential():
    r = determine(ScopeInput("health", "healthcare providers", 800, 90e6, 70e6))
    assert r.classification is Classification.ESSENTIAL
    assert r.in_scope is True
    assert r.annex == "I"
    assert r.obligations


def test_medium_annex_i_is_important():
    r = determine(ScopeInput("transport", "road", 120, 30e6, 20e6))
    assert r.classification is Classification.IMPORTANT
    assert r.in_scope is True


def test_large_annex_ii_is_important_not_essential():
    """Annex II entities are never essential on size alone (Art. 3(1) covers Annex I)."""
    r = determine(ScopeInput("manufacturing", "motor vehicles", 5000, 900e6, 800e6))
    assert r.classification is Classification.IMPORTANT
    assert r.annex == "II"


def test_small_entity_is_out_of_scope():
    r = determine(ScopeInput("food", None, 12, 1.5e6, 1e6))
    assert r.in_scope is False
    assert r.obligations == []


def test_unlisted_sector_is_out_of_scope_regardless_of_size():
    r = determine(ScopeInput("advertising", None, 5000, 900e6, 800e6))
    assert r.in_scope is False
    assert r.annex is None


# ── Art. 2(2): in scope regardless of size ───────────────────────────────────

def test_micro_dns_provider_is_essential():
    """The case a size-only checker gets backwards.

    Six people, tiny financials — and still an essential entity, because
    Art. 2(2)(c) disapplies the size threshold for DNS providers.
    """
    r = determine(
        ScopeInput("digital infrastructure", "DNS service providers", 6, 400e3, 200e3,
                   flags={"is_dns_or_tld_provider": True})
    )
    assert r.classification is Classification.ESSENTIAL
    assert r.in_scope is True
    assert r.size is Size.MICRO
    assert any("2(2)(c)" in step for step in r.reasoning)


@pytest.mark.parametrize(
    "flag",
    [
        "provides_public_electronic_communications",
        "is_trust_service_provider",
        "is_dns_or_tld_provider",
        "is_sole_provider_in_member_state",
        "is_central_public_administration",
    ],
)
def test_every_size_cap_flag_overrides_the_size_test(flag):
    r = determine(ScopeInput("digital infrastructure", None, 3, 100e3, 50e3, flags={flag: True}))
    assert r.in_scope is True


def test_size_cap_flag_does_not_rescue_an_unlisted_sector():
    """Sector is tested before the size-cap exemption; failing it ends the analysis."""
    r = determine(ScopeInput("advertising", None, 3, 100e3, 50e3,
                             flags={"is_trust_service_provider": True}))
    assert r.in_scope is False


def test_unknown_flags_are_ignored():
    r = determine(ScopeInput("food", None, 10, 1e6, 1e6, flags={"made_up_criterion": True}))
    assert r.in_scope is False


# ── Explanation quality ──────────────────────────────────────────────────────

def test_every_result_cites_articles_and_carries_a_disclaimer():
    for inp in [
        ScopeInput("health", None, 800, 90e6, 70e6),
        ScopeInput("food", None, 12, 1e6, 1e6),
        ScopeInput("advertising", None, 500, 90e6, 70e6),
    ]:
        r = determine(inp)
        assert r.reasoning, "a determination must explain itself"
        assert any("Art." in s for s in r.reasoning)
        assert "not legal advice" in r.disclaimer.lower()


def test_out_of_scope_results_still_warn_about_supplier_obligations():
    """'No' is rarely the end of it — Art. 21(2)(d) reaches suppliers contractually."""
    r = determine(ScopeInput("food", None, 12, 1e6, 1e6))
    assert any("supplier" in c.lower() for c in r.caveats)


def test_missing_headcount_asks_for_it_rather_than_guessing():
    r = determine(ScopeInput("health", None, None, None, None))
    assert r.in_scope is False
    assert any("size test" in s.lower() for s in r.reasoning)


def test_essential_and_important_differ_only_in_supervision():
    ess = determine(ScopeInput("health", None, 800, 90e6, 70e6)).obligations
    imp = determine(ScopeInput("transport", None, 120, 30e6, 20e6)).obligations
    joined_e, joined_i = " ".join(ess).lower(), " ".join(imp).lower()
    # Both carry Art. 21 and Art. 23 duties...
    for text in (joined_e, joined_i):
        assert "art. 21(2)" in text and "art. 23" in text
    # ...and differ on supervision and penalty ceiling.
    assert "ex ante" in joined_e and "ex post" in joined_i


# ── Sector catalogue ─────────────────────────────────────────────────────────

def test_sector_catalogue_shape():
    s = sectors()
    assert set(s) == {"annex_I", "annex_II", "size_exempt_criteria"}
    assert s["annex_I"]["sectors"] and s["annex_II"]["sectors"]


def test_every_catalogued_sector_resolves_to_its_annex():
    """The UI builds its dropdown from this catalogue, so every option must work."""
    s = sectors()
    for annex, key in [("I", "annex_I"), ("II", "annex_II")]:
        for name in s[key]["sectors"]:
            assert determine(ScopeInput(name, None, 500, 90e6, 70e6)).annex == annex


def test_every_advertised_criterion_is_honoured():
    """A criterion offered in the UI that the engine ignores would be a silent lie."""
    for flag in sectors()["size_exempt_criteria"]:
        r = determine(ScopeInput("energy", None, 2, 50e3, 50e3, flags={flag: True}))
        assert r.in_scope is True, f"{flag} advertised but not honoured"
