"""
Structured output for the NIS2 readiness assessment.

The excerpt on every finding is validated to be a verbatim substring of the
uploaded document. In a compliance context a fabricated quotation is worse than
no tool at all: the reader cannot tell an invented obligation from a real one,
so the assessment stops being checkable. Anything the model paraphrases or
invents is dropped rather than shown.
"""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field


class Status(str, Enum):
    ADDRESSED = "ADDRESSED"          # the document demonstrably covers it
    PARTIAL = "PARTIAL"              # touched on, but incomplete against the obligation
    NOT_ADDRESSED = "NOT_ADDRESSED"  # nothing found in the document
    UNCLEAR = "UNCLEAR"              # text exists but is too vague to judge


class Severity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


# Ordering for report presentation: worst first, so the reader sees the gaps
# that matter before the things already handled.
_STATUS_RANK = {
    Status.NOT_ADDRESSED: 0,
    Status.PARTIAL: 1,
    Status.UNCLEAR: 2,
    Status.ADDRESSED: 3,
}
_SEVERITY_RANK = {
    Severity.HIGH: 0,
    Severity.MEDIUM: 1,
    Severity.LOW: 2,
    Severity.NONE: 3,
}


class Assessment(BaseModel):
    """What the LLM is asked to return for a single checkpoint."""

    status: Status
    severity: Severity
    rationale: str = Field(description="Two sentences at most, citing what the document does or omits.")
    excerpt: str = Field(
        default="",
        description="Verbatim quote from the uploaded document, or empty if nothing relevant exists.",
    )


class Finding(Assessment):
    """An assessment bound to its checkpoint, ready to render."""

    checkpoint_id: str
    domain: str
    article: str
    obligation: str
    evidence_expected: list[str] = Field(default_factory=list)

    # False when the model returned an excerpt that is not actually in the
    # document. The excerpt is cleared, and the report says so rather than
    # quietly dropping it.
    excerpt_verified: bool = True

    @property
    def sort_key(self) -> tuple[int, int, str]:
        return (_STATUS_RANK[self.status], _SEVERITY_RANK[self.severity], self.checkpoint_id)


class Report(BaseModel):
    document_name: str
    findings: list[Finding]

    # Which model produced the judgements. Recorded because finding quality is
    # model-dependent, so a report is only interpretable alongside it.
    model: str = ""

    # Headline counts, so a caller does not have to recompute them.
    total: int = 0
    addressed: int = 0
    partial: int = 0
    not_addressed: int = 0
    unclear: int = 0
    high_severity_gaps: int = 0
    coverage_pct: float = 0.0

    def finalise(self) -> "Report":
        """Sort worst-first and compute the summary counters."""
        self.findings.sort(key=lambda f: f.sort_key)
        self.total = len(self.findings)
        self.addressed = sum(f.status == Status.ADDRESSED for f in self.findings)
        self.partial = sum(f.status == Status.PARTIAL for f in self.findings)
        self.not_addressed = sum(f.status == Status.NOT_ADDRESSED for f in self.findings)
        self.unclear = sum(f.status == Status.UNCLEAR for f in self.findings)
        self.high_severity_gaps = sum(
            f.severity == Severity.HIGH and f.status != Status.ADDRESSED
            for f in self.findings
        )
        # PARTIAL counts as half — a policy that mentions backups but never tests
        # restores is genuinely better than silence, and worse than compliance.
        if self.total:
            self.coverage_pct = round(
                100 * (self.addressed + 0.5 * self.partial) / self.total, 1
            )
        return self


def _normalise(text: str) -> str:
    """Collapse whitespace so PDF line breaks don't defeat the quote check."""
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_excerpt(excerpt: str, document_text: str, *, min_chars: int = 25) -> bool:
    """
    True when `excerpt` really occurs in `document_text`.

    Whitespace is normalised on both sides because PDF extraction inserts line
    breaks mid-sentence, which would otherwise fail an exact match on a quote
    the model copied correctly. Very short excerpts are rejected outright: a
    handful of characters will match by chance and prove nothing.
    """
    if not excerpt or len(excerpt.strip()) < min_chars:
        return False
    return _normalise(excerpt) in _normalise(document_text)
