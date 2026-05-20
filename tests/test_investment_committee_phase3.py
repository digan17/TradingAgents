import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents import create_macro_analyst
from tradingagents.agents.managers.portfolio_manager import create_portfolio_manager
from tradingagents.agents.schemas import (
    CommitteeRating,
    ConditionalAction,
    PortfolioDecision,
)
from tradingagents.agents.utils import macro_data_tools
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.agents.utils.scoring import build_agent_scores
from tradingagents.graph.propagation import Propagator


def _pm_state(ticker: str = "AAPL", macro_snapshot: dict | None = None) -> dict:
    state = Propagator().create_initial_state(ticker, "2026-05-20")
    state.update(
        {
            "market_report": "Price trend is constructive with improving momentum and positive breakout risk.",
            "macro_report": "Macro regime is risk-sensitive but data quality is medium.",
            "macro_snapshot": macro_snapshot or {},
            "macro_data_quality": {"quality": "medium"},
            "sentiment_report": "Retail sentiment is bullish but crowding risk is moderate.",
            "news_report": "Recent news is positive, but regulatory risk remains unresolved.",
            "fundamentals_report": "Revenue, earnings, margin and cash flow quality are strong; valuation risk is premium.",
            "investment_plan": "Preliminary view: Buy candidate, but wait for pullback.",
            "trader_investment_plan": (
                "**Action**: Buy\n\n"
                "**Reasoning**: Constructive setup.\n\n"
                "**Entry Price**: 190.0\n\n"
                "**Stop Loss**: 180.0\n\n"
                "FINAL TRANSACTION PROPOSAL: **BUY**"
            ),
            "risk_debate_state": {
                "history": "Risk chair flags valuation risk; opportunity chair highlights catalysts.",
                "aggressive_history": "Upside catalyst exists.",
                "conservative_history": "Valuation and macro risk remain.",
                "neutral_history": "Wait for confirmation.",
                "judge_decision": "",
                "latest_speaker": "",
                "current_aggressive_response": "",
                "current_conservative_response": "",
                "current_neutral_response": "",
                "count": 1,
            },
        }
    )
    return state


def _structured_pm(decision: PortfolioDecision, captured: dict | None = None):
    structured = MagicMock()
    structured.invoke.side_effect = lambda prompt: (
        captured.__setitem__("prompt", prompt) if captured is not None else None
    ) or decision
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


@pytest.mark.unit
def test_scoring_layer_builds_required_scores_and_analyst_summaries():
    scores = build_agent_scores(_pm_state())

    for key in [
        "macro_score",
        "rate_pressure_score",
        "inflation_pressure_score",
        "fx_pressure_score",
        "volatility_score",
        "market_trend_score",
        "market_breadth_score",
        "valuation_risk_score",
        "fundamental_quality_score",
        "earnings_momentum_score",
        "sentiment_score",
        "news_event_risk_score",
    ]:
        assert key in scores
        assert 0 <= scores[key] <= 100

    assert {item["agent"] for item in scores["analyst_summaries"]} >= {
        "macro_analyst",
        "market_analyst",
        "sentiment_analyst",
        "news_event_analyst",
        "fundamentals_analyst",
    }


@pytest.mark.unit
def test_portfolio_manager_outputs_korean_report_and_decision_log_payload():
    captured = {}
    decision = PortfolioDecision(
        rating=CommitteeRating.WATCH,
        conditional_action=ConditionalAction.WATCH_ONLY,
        confidence=61,
        data_quality="medium",
        executive_summary="조건 충족 전까지 관찰이 우선입니다.",
        investment_thesis="기술적 흐름은 개선되지만 매크로와 밸류에이션 부담이 남아 있습니다.",
        entry_zone="190 이하 눌림 확인",
        invalidation_conditions=["180 하회", "실적 추정치 하향"],
        risk_summary=["밸류에이션 부담", "매크로 금리 압력"],
        sizing="신규 진입 보류",
        time_horizon="1-3개월",
        review_trigger="지지선 재확인 또는 실적 이벤트",
    )
    node = create_portfolio_manager(_structured_pm(decision, captured))
    result = node(_pm_state())

    assert "투자위원회 종합 리포트" in result["final_trade_decision"]
    assert "투자 자문이나 매수/매도 권유가 아닙니다" in result["final_trade_decision"]
    assert "FINAL TRANSACTION PROPOSAL" not in result["final_trade_decision"]
    assert "Rule-based scores JSON" in captured["prompt"]
    assert result["final_rating"] == "Watch"
    assert result["conditional_action"] == "Watch only"
    assert result["decision_log_payload"]["final_rating"] == "Watch"
    assert result["decision_log_payload"]["conditional_action"] == "Watch only"
    assert result["decision_log_payload"]["holding_horizon"] == "1-3개월"
    assert result["decision_log_payload"]["evaluation_targets"]["return_1m"] is None


@pytest.mark.smoke
def test_aapl_smoke_without_macro_keys_does_not_crash(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)

    payload = json.loads(get_macro_indicators.func("US", "2026-05-20", 18))
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="매크로 데이터 없음")
    macro_node = create_macro_analyst(llm)
    result = macro_node(Propagator().create_initial_state("AAPL", "2026-05-20"))

    assert payload["available"] is False
    assert result["macro_snapshot"]["market"] == "US"
    assert result["macro_data_quality"]["quality"] == "low"


@pytest.mark.smoke
def test_kr_smoke_with_mock_ecos_response(monkeypatch, tmp_path):
    from tradingagents.dataflows.config import set_config

    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("BOK_ECOS_API_KEY", "fixture-key")

    def fake_get_json(url, params=None):
        if "/D/" in url:
            rows = [
                {"TIME": "20260420", "DATA_VALUE": "1350.0"},
                {"TIME": "20260520", "DATA_VALUE": "1360.0"},
            ]
        else:
            rows = [
                {"TIME": "202604", "DATA_VALUE": "100.0"},
                {"TIME": "202605", "DATA_VALUE": "101.0"},
            ]
        return {"StatisticSearch": {"row": rows}}

    monkeypatch.setattr(macro_data_tools, "_get_json", fake_get_json)
    payload = json.loads(get_macro_indicators.func("KR", "2026-05-20", 18))

    assert payload["available"] is True
    assert payload["market"] == "KR"
    assert payload["source_status"]["ecos"] == "available"
