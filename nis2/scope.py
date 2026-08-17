"""
NIS2 scope determination — "am I even in scope?"

Deliberately **rule-based, with no LLM anywhere in the path**. Scope follows
from a small number of published tests in Directive (EU) 2022/2555, and a
deterministic implementation is reproducible, instant, auditable, and cannot
hallucinate. Every answer cites the article it came from, so a reader can check
the reasoning rather than trust it.

The tests, in the order they are applied:

1. **Sector** — Annex I (high criticality) or Annex II (other critical). Outside
   both, the directive does not apply (Art. 2(1)).
2. **Size-cap exemption** — Art. 2(2) puts certain entity types in scope
   *regardless of size* (DNS, TLD registries, trust services, public e-comms,
   sole national providers, central public administration).
3. **Size** — otherwise the entity must reach at least medium size under
   Recommendation 2003/361/EC (Art. 2(1)).
4. **Classification** — large entities in Annex I sectors are *essential*;
   everything else in scope is *important* (Art. 3(1), 3(2)).

Not legal advice. Member States may extend scope in national transposition, and
Art. 2(2)(d)–(g) contain judgement-based criteria this cannot decide for you —
both are surfaced as caveats rather than silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

DIRECTIVE = "Directive (EU) 2022/2555 (NIS2)"


class Classification(str, Enum):
    ESSENTIAL = "ESSENTIAL"
    IMPORTANT = "IMPORTANT"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    LIKELY_OUT_OF_SCOPE = "LIKELY_OUT_OF_SCOPE"  # out on these answers, but a judgement test may pull it in


class Size(str, Enum):
    MICRO = "MICRO"
    SMALL = "SMALL"
    MEDIUM = "MEDIUM"
    LARGE = "LARGE"


# ── Sectors ──────────────────────────────────────────────────────────────────
# Annex I: sectors of high criticality. Annex II: other critical sectors.
# The split matters twice: it decides essential vs important, and Annex II
# entities are never essential on size alone.

ANNEX_I: dict[str, tuple[str, ...]] = {
    "energy": ("electricity", "district heating and cooling", "oil", "gas", "hydrogen"),
    "transport": ("air", "rail", "water", "road"),
    "banking": ("credit institutions",),
    "financial market infrastructure": ("trading venues", "central counterparties"),
    "health": ("healthcare providers", "EU reference laboratories",
               "pharmaceutical manufacturing", "medical device manufacturing"),
    "drinking water": ("suppliers and distributors",),
    "waste water": ("collection, disposal or treatment",),
    "digital infrastructure": ("internet exchange points", "DNS service providers",
                              "TLD name registries", "cloud computing services",
                              "data centre services", "content delivery networks",
                              "trust service providers",
                              "public electronic communications networks",
                              "publicly available electronic communications services"),
    "ict service management": ("managed service providers", "managed security service providers"),
    "public administration": ("central government", "regional government"),
    "space": ("ground-based infrastructure operators",),
}

ANNEX_II: dict[str, tuple[str, ...]] = {
    "postal and courier": ("postal service providers", "courier service providers"),
    "waste management": ("waste management",),
    "chemicals": ("manufacture, production and distribution",),
    "food": ("production, processing and distribution",),
    "manufacturing": ("medical devices", "computer, electronic and optical products",
                      "electrical equipment", "machinery and equipment", "motor vehicles",
                      "other transport equipment"),
    "digital providers": ("online marketplaces", "online search engines",
                          "social networking platforms"),
    "research": ("research organisations",),
}

# Art. 2(2): in scope regardless of size. Keys are the flags a caller can set.
SIZE_EXEMPT_CRITERIA: dict[str, str] = {
    "provides_public_electronic_communications":
        "Provider of public electronic communications networks or publicly available "
        "electronic communications services — Art. 2(2)(a)",
    "is_trust_service_provider":
        "Trust service provider — Art. 2(2)(b)",
    "is_dns_or_tld_provider":
        "TLD name registry, DNS service provider or domain registration service — Art. 2(2)(c)",
    "is_sole_provider_in_member_state":
        "Sole provider in a Member State of a service essential for critical societal or "
        "economic activities — Art. 2(2)(d)",
    "disruption_impacts_public_safety":
        "Disruption could significantly impact public safety, security or health — Art. 2(2)(e)",
    "disruption_causes_systemic_risk":
        "Disruption could induce significant systemic risk, particularly cross-border — Art. 2(2)(f)",
    "is_central_public_administration":
        "Public administration entity of central government — Art. 2(2)(h)",
    "is_critical_under_cer":
        "Identified as a critical entity under Directive (EU) 2022/2557 (CER) — Art. 3(1)(f)",
}


@dataclass
class ScopeInput:
    sector: str
    subsector: str | None = None
    employees: int | None = None
    annual_turnover_eur: float | None = None
    balance_sheet_eur: float | None = None
    flags: dict[str, bool] = field(default_factory=dict)


@dataclass
class ScopeResult:
    classification: Classification
    in_scope: bool
    size: Size | None
    annex: str | None
    reasoning: list[str]
    caveats: list[str]
    obligations: list[str]
    disclaimer: str = (
        "Indicative determination based on the criteria in "
        f"{DIRECTIVE}. Not legal advice. Member States may extend scope in national "
        "transposition; confirm with your national competent authority."
    )


def classify_size(
    employees: int | None,
    turnover: float | None,
    balance_sheet: float | None,
) -> Size | None:
    """
    Size class per Recommendation 2003/361/EC, as NIS2 Art. 2(1) applies it.

    The staff headcount is the binding criterion; the financial ceilings are
    alternatives to each other, so an entity stays in a class if *either* its
    turnover or its balance sheet total is within the ceiling.
    """
    if employees is None:
        return None

    def within(t_cap: float, b_cap: float) -> bool:
        # Unknown financials are treated as within the ceiling: absent data
        # should not silently promote an entity into scope.
        t_ok = turnover is None or turnover <= t_cap
        b_ok = balance_sheet is None or balance_sheet <= b_cap
        return t_ok or b_ok

    if employees < 10 and within(2e6, 2e6):
        return Size.MICRO
    if employees < 50 and within(10e6, 10e6):
        return Size.SMALL
    if employees < 250 and within(50e6, 43e6):
        return Size.MEDIUM
    return Size.LARGE


def _find_annex(sector: str) -> str | None:
    s = sector.strip().lower()
    if s in ANNEX_I:
        return "I"
    if s in ANNEX_II:
        return "II"
    return None


def determine(inp: ScopeInput) -> ScopeResult:
    """Apply the four tests in order and explain each step."""
    reasoning: list[str] = []
    caveats: list[str] = []

    # 1 — sector
    annex = _find_annex(inp.sector)
    if annex is None:
        return ScopeResult(
            classification=Classification.LIKELY_OUT_OF_SCOPE,
            in_scope=False,
            size=classify_size(inp.employees, inp.annual_turnover_eur, inp.balance_sheet_eur),
            annex=None,
            reasoning=[
                f"Sector {inp.sector!r} does not match a listed Annex I or Annex II sector, "
                "so NIS2 does not apply on the basis of sector (Art. 2(1))."
            ],
            caveats=[
                "Sector names here follow the Annexes. If your activity is described "
                "differently but falls within a listed sector, re-check with that sector selected.",
                "You may still be in scope as a supplier: entities in scope must impose "
                "security requirements on their direct suppliers (Art. 21(2)(d)), which "
                "reaches organisations the directive does not bind directly.",
            ],
            obligations=[],
        )
    reasoning.append(
        f"Sector {inp.sector!r} is listed in Annex {annex} "
        f"({'sectors of high criticality' if annex == 'I' else 'other critical sectors'})."
    )

    # 2 — size-cap exemptions (Art. 2(2))
    triggered = [SIZE_EXEMPT_CRITERIA[k] for k, v in inp.flags.items()
                 if v and k in SIZE_EXEMPT_CRITERIA]
    size = classify_size(inp.employees, inp.annual_turnover_eur, inp.balance_sheet_eur)

    if triggered:
        reasoning.extend(f"In scope regardless of size: {t}" for t in triggered)
        # Art. 3(1) makes these entity types essential where they sit in Annex I.
        cls = Classification.ESSENTIAL if annex == "I" else Classification.IMPORTANT
        reasoning.append(
            f"Classified as {cls.value.lower()} (Art. 3(1))."
            if annex == "I" else
            f"Annex II entity, classified as {cls.value.lower()} (Art. 3(2))."
        )
        return ScopeResult(
            classification=cls, in_scope=True, size=size, annex=annex,
            reasoning=reasoning, caveats=caveats, obligations=_obligations(cls),
        )

    # 3 — size threshold
    if size is None:
        return ScopeResult(
            classification=Classification.LIKELY_OUT_OF_SCOPE, in_scope=False,
            size=None, annex=annex,
            reasoning=reasoning + ["Employee count not supplied, so the size test could not be applied."],
            caveats=["Provide headcount, and turnover or balance sheet total, for a determination."],
            obligations=[],
        )

    reasoning.append(
        f"Size class: {size.value.lower()} "
        f"(Recommendation 2003/361/EC, applied by Art. 2(1))."
    )

    if size in (Size.MICRO, Size.SMALL):
        return ScopeResult(
            classification=Classification.LIKELY_OUT_OF_SCOPE, in_scope=False,
            size=size, annex=annex,
            reasoning=reasoning + [
                "Below the medium-sized threshold, so not in scope on size (Art. 2(1))."
            ],
            caveats=[
                "Art. 2(2)(d)–(g) can still pull a small entity into scope — for example if "
                "you are the sole provider of an essential service in a Member State, or a "
                "disruption would create systemic risk. These are judgement calls this tool "
                "does not make for you.",
                "Member States may designate additional entities in national transposition.",
                "Customers in scope must impose security requirements on their suppliers "
                "(Art. 21(2)(d)), so NIS2 obligations often arrive contractually regardless.",
            ],
            obligations=[],
        )

    # 4 — classification
    if annex == "I" and size == Size.LARGE:
        cls = Classification.ESSENTIAL
        reasoning.append(
            "Large entity in an Annex I sector, therefore an essential entity (Art. 3(1)(a))."
        )
    else:
        cls = Classification.IMPORTANT
        reasoning.append(
            "Medium-sized entity, or an Annex II entity, therefore an important entity (Art. 3(2))."
        )

    caveats.append(
        "Essential and important entities carry the same Art. 21 risk-management and "
        "Art. 23 reporting duties; they differ in supervision — essential entities face "
        "ex ante supervision, important entities ex post (Art. 32, 33)."
    )
    return ScopeResult(
        classification=cls, in_scope=True, size=size, annex=annex,
        reasoning=reasoning, caveats=caveats, obligations=_obligations(cls),
    )


def _obligations(cls: Classification) -> list[str]:
    if cls not in (Classification.ESSENTIAL, Classification.IMPORTANT):
        return []
    common = [
        "Register with your national competent authority (Art. 3(4)).",
        "Implement the risk-management measures of Art. 21(2) — the thirteen domains "
        "covered by this project's /assess endpoint.",
        "Report significant incidents: early warning within 24 hours, incident "
        "notification within 72 hours, final report within one month (Art. 23).",
        "Management bodies must approve the measures and can be held liable for "
        "non-compliance; they must undergo training (Art. 20).",
        "Impose security requirements on direct suppliers and service providers (Art. 21(2)(d)).",
    ]
    if cls is Classification.ESSENTIAL:
        common.append(
            "Subject to ex ante supervision: audits, inspections and information requests "
            "without prior evidence of breach (Art. 32). Fines up to €10 000 000 or 2% of "
            "total worldwide annual turnover, whichever is higher (Art. 34(4))."
        )
    else:
        common.append(
            "Subject to ex post supervision — action follows evidence of non-compliance "
            "(Art. 33). Fines up to €7 000 000 or 1.4% of total worldwide annual turnover, "
            "whichever is higher (Art. 34(5))."
        )
    return common


def sectors() -> dict:
    """Sector/subsector lists for a UI to render."""
    return {
        "annex_I": {"description": "Sectors of high criticality", "sectors": ANNEX_I},
        "annex_II": {"description": "Other critical sectors", "sectors": ANNEX_II},
        "size_exempt_criteria": SIZE_EXEMPT_CRITERIA,
    }
