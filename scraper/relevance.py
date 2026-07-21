"""
Keyword-based relevance filter for publication listings.

Checks title AND description/topics — not just title.
"""

from __future__ import annotations

from config import KEYWORDS


def is_relevant(
    title: str,
    description: str = "",
    topics: list[str] | None = None,
    *,
    focus_keywords: list[str] | None = None,
) -> bool:
    """
    Return True if any keyword appears (case-insensitive) in
    title, description, or topic tags.
    """
    keywords = KEYWORDS if focus_keywords is None else focus_keywords
    if not keywords:
        return False

    # Build a single searchable text blob
    parts = [title]
    if description:
        parts.append(description)
    if topics:
        parts.extend(topics)
    haystack = " ".join(parts).lower()

    return any(kw.lower().strip() in haystack for kw in keywords if kw.strip())
