"""Rule-based scoring layer for the investment committee workflow.

The scores are intentionally deterministic and conservative. They are not a
price forecast; they give the LLM a grounded starting point to interpret,
challenge, and explain instead of inventing scores from prose alone.
"""

from __future__ import annotations

from typing import Any


POSITIVE_TERMS = (
    "bullish",
    "constructive",
    "strong",
    "growth",
    "positive",
    "uptrend",
    "breakout",
    "improving",
    "beat",
    "상승",
    "강세",
    "개선",
    "호조",
    "긍정",
)

NEGATIVE_TERMS = (
    "bearish",
    "weak",
    "risk",
    "decline",
    "negative",
    "downtrend",
    "breakdown",
    "deteriorating",
    "miss",
    "하락",
    "약세",
    "위험",
    "부진",
    "악화",
    "부정",
)

QUALITY_TERMS = (
    "margin",
    "cash flow",
    "balance sheet",
    "profit",
    "earnings",
    "revenue",
    "quality",
    "마진",
    "현금흐름",
    "재무",
    "이익",
    "실적",
    "매출",
)

VALUATION_RISK_TERMS = (
    "expensive",
    "overvalued",
    "multiple",
    "valuation risk",
    "premium",
    "고평가",
    "밸류에이션 부담",
    "멀티플",
)


def clamp_score(value: int | float) -> int:
    return max(0, min(100, int(round(value))))


def pressure_to_score(value: str | None) -> int:
    mapping = {
        "low": 25,
        "calm": 25,
        "supportive": 30,
        "neutral": 50,
        "unknown": 50,
        "moderate": 60,
        "watch": 60,
        "balanced-watch": 58,
        "elevated": 72,
        "high": 78,
        "risk-sensitive": 72,
        "unavailable": 50,
    }
    return mapping.get((value or "unknown").lower(), 50)


def text_sentiment_score(text: str) -> int:
    lower = (text or "").lower()
    positives = sum(lower.count(term.lower()) for term in POSITIVE_TERMS)
    negatives = sum(lower.count(term.lower()) for term in NEGATIVE_TERMS)
    return clamp_score(50 + 8 * positives - 8 * negatives)


def text_risk_score(text: str, risk_terms: tuple[str, ...] = NEGATIVE_TERMS) -> int:
    lower = (text or "").lower()
    hits = sum(lower.count(term.lower()) for term in risk_terms)
    positives = sum(lower.count(term.lower()) for term in POSITIVE_TERMS)
    return clamp_score(45 + 7 * hits - 3 * positives)


def text_quality_score(text: str) -> int:
    lower = (text or "").lower()
    quality_hits = sum(lower.count(term.lower()) for term in QUALITY_TERMS)
    risk_hits = sum(lower.count(term.lower()) for term in NEGATIVE_TERMS)
    return clamp_score(50 + 6 * quality_hits - 4 * risk_hits)


def direction_from_score(score: int, bullish_when_high: bool = True) -> str:
    if bullish_when_high:
        if score >= 60:
            return "bullish"
        if score <= 40:
            return "bearish"
        return "neutral"
    if score >= 65:
        return "bearish"
    if score <= 35:
        return "bullish"
    return "neutral"


def quality_from_report(text: str, fallback: str = "medium") -> str:
    lower = (text or "").lower()
    if not lower.strip():
        return "low"
    if "unavailable" in lower or "데이터 없음" in lower or "missing" in lower:
        return "low"
    if len(lower) < 240:
        return fallback
    return "high"


def _macro_scores(macro_snapshot: dict[str, Any]) -> dict[str, int]:
    derived = macro_snapshot.get("derived", {}) if isinstance(macro_snapshot, dict) else {}
    rate_pressure = pressure_to_score(derived.get("rate_pressure"))
    inflation_pressure = pressure_to_score(derived.get("inflation_pressure"))
    fx_pressure = pressure_to_score(derived.get("fx_pressure"))
    volatility = pressure_to_score(derived.get("volatility_regime"))
    growth_risk = pressure_to_score(derived.get("growth_risk"))
    macro_risk = (rate_pressure + inflation_pressure + fx_pressure + volatility + growth_risk) / 5
    return {
        "macro_score": clamp_score(100 - macro_risk),
        "rate_pressure_score": rate_pressure,
        "inflation_pressure_score": inflation_pressure,
        "fx_pressure_score": fx_pressure,
        "volatility_score": volatility,
    }


def build_agent_scores(state: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic scores and structured analyst summaries from state."""
    market_score = text_sentiment_score(state.get("market_report", ""))
    sentiment_score = text_sentiment_score(state.get("sentiment_report", ""))
    news_event_risk = text_risk_score(state.get("news_report", ""))
    valuation_risk = text_risk_score(state.get("fundamentals_report", ""), VALUATION_RISK_TERMS)
    fundamental_quality = text_quality_score(state.get("fundamentals_report", ""))
    earnings_momentum = text_sentiment_score(state.get("fundamentals_report", ""))
    macro_scores = _macro_scores(state.get("macro_snapshot", {}))

    scores = {
        **macro_scores,
        "market_trend_score": market_score,
        "market_breadth_score": 50,
        "valuation_risk_score": valuation_risk,
        "fundamental_quality_score": fundamental_quality,
        "earnings_momentum_score": earnings_momentum,
        "sentiment_score": sentiment_score,
        "news_event_risk_score": news_event_risk,
    }

    macro_quality = state.get("macro_data_quality", {}).get("quality", "low")
    summaries = [
        {
            "agent": "macro_analyst",
            "as_of_date": state.get("trade_date", ""),
            "direction": direction_from_score(scores["macro_score"]),
            "score": scores["macro_score"],
            "confidence": 70 if macro_quality in {"high", "medium"} else 35,
            "data_quality": macro_quality,
            "key_evidence": [state.get("macro_snapshot", {}).get("derived", {})],
            "risks": [],
            "invalidating_conditions": [],
        },
        {
            "agent": "market_analyst",
            "as_of_date": state.get("trade_date", ""),
            "direction": direction_from_score(market_score),
            "score": market_score,
            "confidence": 55,
            "data_quality": quality_from_report(state.get("market_report", "")),
            "key_evidence": [],
            "risks": [],
            "invalidating_conditions": [],
        },
        {
            "agent": "sentiment_analyst",
            "as_of_date": state.get("trade_date", ""),
            "direction": direction_from_score(sentiment_score),
            "score": sentiment_score,
            "confidence": 50,
            "data_quality": quality_from_report(state.get("sentiment_report", "")),
            "key_evidence": [],
            "risks": ["crowding risk if sentiment is one-sided"],
            "invalidating_conditions": [],
        },
        {
            "agent": "news_event_analyst",
            "as_of_date": state.get("trade_date", ""),
            "direction": direction_from_score(news_event_risk, bullish_when_high=False),
            "score": news_event_risk,
            "confidence": 50,
            "data_quality": quality_from_report(state.get("news_report", "")),
            "key_evidence": [],
            "risks": [],
            "invalidating_conditions": [],
        },
        {
            "agent": "fundamentals_analyst",
            "as_of_date": state.get("trade_date", ""),
            "direction": direction_from_score(fundamental_quality),
            "score": fundamental_quality,
            "confidence": 55,
            "data_quality": quality_from_report(state.get("fundamentals_report", "")),
            "key_evidence": [],
            "risks": ["valuation risk"] if valuation_risk >= 65 else [],
            "invalidating_conditions": [],
        },
    ]
    return {
        **scores,
        "analyst_summaries": summaries,
        "score_notes": [
            "Scores are deterministic rule-based inputs for LLM interpretation.",
            "Pressure/risk scores are worse when higher; trend/quality scores are better when higher.",
        ],
    }


def aggregate_data_quality(state: dict[str, Any], agent_scores: dict[str, Any]) -> str:
    qualities = [item.get("data_quality", "low") for item in agent_scores.get("analyst_summaries", [])]
    if not qualities:
        return "low"
    if qualities.count("high") >= 3 and "low" not in qualities:
        return "high"
    if qualities.count("low") >= 3:
        return "low"
    return "medium"
