"""Shared rating vocabulary and deterministic heuristic parsers.

The legacy five-tier scale (Buy, Overweight, Hold, Underweight, Sell) is still
accepted for compatibility. The v1 investment committee final decision uses:
Strong Buy, Buy, Watch, Hold, Reduce, Avoid.
- The signal processor (rating extracted for downstream consumers)
- The memory log (rating tag stored alongside each decision entry)

Centralising it here avoids drift between those call sites.
"""

from __future__ import annotations

import re
from typing import Tuple


# Canonical, ordered 5-tier scale (most bullish to most bearish).
RATINGS_5_TIER: Tuple[str, ...] = (
    "Buy", "Overweight", "Hold", "Underweight", "Sell",
)
RATINGS_COMMITTEE: Tuple[str, ...] = (
    "Strong Buy", "Buy", "Watch", "Hold", "Reduce", "Avoid",
)
RATINGS_ALL: Tuple[str, ...] = RATINGS_COMMITTEE + RATINGS_5_TIER

_RATING_LOOKUP = {r.lower(): r for r in RATINGS_ALL}

# Matches "Rating: X" / "rating - X" / "Rating: **Strong Buy**" — tolerates
# markdown bold wrappers and either a colon or hyphen separator.
_RATING_LABEL_RE = re.compile(
    r"rating.*?[:\-]\s*\**([A-Za-z ]+)",
    re.IGNORECASE,
)


def _canonical(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.strip("*:.,` ").lower().split())
    if normalized in _RATING_LOOKUP:
        return _RATING_LOOKUP[normalized]
    for rating in sorted(RATINGS_ALL, key=len, reverse=True):
        if re.search(rf"\b{re.escape(rating.lower())}\b", normalized):
            return rating
    return None


def parse_rating(text: str, default: str = "Hold") -> str:
    """Heuristically extract a rating from prose text.

    Two-pass strategy:
    1. Look for an explicit "Rating: X" label (tolerant of markdown bold).
    2. Fall back to the first known rating phrase found anywhere in the text.

    Returns a canonical rating string, or ``default`` if no rating appears.
    """
    for line in text.splitlines():
        m = _RATING_LABEL_RE.search(line)
        parsed = _canonical(m.group(1) if m else None)
        if parsed:
            return parsed

    lowered = text.lower()
    for rating in RATINGS_ALL:
        if re.search(rf"\b{re.escape(rating.lower())}\b", lowered):
            return rating

    return default


def parse_committee_rating(text: str, default: str = "Watch") -> str:
    """Extract the v1 six-tier final committee rating.

    Legacy PM outputs are mapped into the new scale so old reports remain
    parseable by memory log and backend callers.
    """
    parsed = parse_rating(text, default=default)
    legacy_map = {
        "Overweight": "Buy",
        "Underweight": "Reduce",
        "Sell": "Avoid",
    }
    return legacy_map.get(parsed, parsed)


def rating_to_trade_signal(rating: str) -> str:
    """Map committee ratings to the old Buy/Hold/Sell transaction signal."""
    if rating in {"Strong Buy", "Buy"}:
        return "Buy"
    if rating in {"Reduce", "Avoid"}:
        return "Sell"
    return "Hold"


def conditional_action_for_rating(rating: str) -> str:
    mapping = {
        "Strong Buy": "Immediate entry possible",
        "Buy": "Wait for pullback",
        "Watch": "Watch only",
        "Hold": "Hold existing position only",
        "Reduce": "Reduce exposure",
        "Avoid": "Avoid new entry",
    }
    return mapping.get(rating, "Watch only")
