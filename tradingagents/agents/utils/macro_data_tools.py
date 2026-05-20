import json
import os
from datetime import datetime
from typing import Annotated

from langchain_core.tools import tool


def _normalize_market(market: str) -> str:
    value = (market or "").strip().upper()
    if value in {"US", "USA", "UNITED STATES"}:
        return "US"
    if value in {"KR", "KOREA", "KOR", "SOUTH KOREA"}:
        return "KR"
    return "GLOBAL"


def _source_status(market: str) -> dict[str, str]:
    fred_key = bool(os.getenv("FRED_API_KEY"))
    ecos_key = bool(os.getenv("BOK_ECOS_API_KEY"))

    if market == "US":
        return {
            "fred": "unavailable_not_implemented" if fred_key else "unavailable_missing_key",
            "ecos": "not_applicable",
        }
    if market == "KR":
        return {
            "fred": "not_applicable",
            "ecos": "unavailable_not_implemented" if ecos_key else "unavailable_missing_key",
        }
    return {
        "fred": "unavailable_not_implemented" if fred_key else "unavailable_missing_key",
        "ecos": "unavailable_not_implemented" if ecos_key else "unavailable_missing_key",
    }


def _unavailable_sources(status: dict[str, str]) -> list[dict[str, str]]:
    required_env = {
        "fred": "FRED_API_KEY",
        "ecos": "BOK_ECOS_API_KEY",
    }
    source_name = {
        "fred": "FRED",
        "ecos": "BOK ECOS",
    }
    blocks = []
    for source, state in status.items():
        if state == "not_applicable":
            continue
        if state == "unavailable_missing_key":
            blocks.append(
                {
                    "available": False,
                    "source": source_name[source],
                    "reason": "missing_api_key",
                    "required_env": required_env[source],
                    "impact": "External macro indicators were not used.",
                    "quality": "low",
                }
            )
        elif state == "unavailable_not_implemented":
            blocks.append(
                {
                    "available": False,
                    "source": source_name[source],
                    "reason": "phase_1_not_implemented",
                    "required_env": required_env[source],
                    "impact": "External macro indicators will be wired in a later phase.",
                    "quality": "low",
                }
            )
    return blocks


def build_macro_unavailable_snapshot(
    market: str,
    curr_date: str,
    lookback_months: int = 18,
) -> dict:
    normalized_market = _normalize_market(market)
    status = _source_status(normalized_market)
    unavailable = _unavailable_sources(status)
    return {
        "available": False,
        "as_of_date": curr_date,
        "market": normalized_market,
        "lookback_months": lookback_months,
        "source_status": status,
        "series": [],
        "derived": {
            "yield_curve_10y2y": None,
            "rate_pressure": "unknown",
            "inflation_pressure": "unknown",
            "growth_risk": "unknown",
            "fx_pressure": "unknown",
            "volatility_regime": "unknown",
            "macro_regime": "unavailable",
        },
        "unavailable_sources": unavailable,
        "data_revision_risk": False,
        "quality": "low",
        "quality_note": (
            "Phase 1 only exposes the macro snapshot contract and no-key "
            "fallback. External FRED/ECOS fetching is implemented in Phase 2."
        ),
        "cache_status": "skipped_no_fetch",
        "cache_created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
    }


@tool
def get_macro_indicators(
    market: Annotated[str, "Market code: US, KR, or GLOBAL"],
    curr_date: Annotated[str, "Current analysis date in yyyy-mm-dd format"],
    lookback_months: Annotated[
        int,
        "Number of months to inspect for raw/derived macro indicators",
    ] = 18,
) -> str:
    """
    Retrieve a structured macro indicator snapshot.

    Phase 1 intentionally avoids external API calls. It returns a stable
    unavailable block when FRED_API_KEY or BOK_ECOS_API_KEY is missing, and a
    non-crashing not-implemented block when keys are present. Phase 2 replaces
    this skeleton with cached FRED/ECOS fetchers and fixture-tested parsers.
    """
    snapshot = build_macro_unavailable_snapshot(market, curr_date, lookback_months)
    return json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
