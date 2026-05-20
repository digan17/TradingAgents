import json

import pytest

from tradingagents.agents import create_macro_analyst
from tradingagents.agents.utils.macro_data_tools import get_macro_indicators
from tradingagents.graph.propagation import Propagator


@pytest.mark.unit
def test_macro_tool_returns_us_missing_key_block(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)

    payload = json.loads(get_macro_indicators.func("US", "2026-05-20", 18))

    assert payload["available"] is False
    assert payload["market"] == "US"
    assert payload["source_status"]["fred"] == "unavailable_missing_key"
    assert payload["source_status"]["ecos"] == "not_applicable"
    assert payload["series"] == []
    assert payload["quality"] == "low"
    assert payload["unavailable_sources"][0]["required_env"] == "FRED_API_KEY"


@pytest.mark.unit
def test_macro_tool_returns_kr_missing_key_block(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)

    payload = json.loads(get_macro_indicators.func("KR", "2026-05-20", 18))

    assert payload["available"] is False
    assert payload["market"] == "KR"
    assert payload["source_status"]["fred"] == "not_applicable"
    assert payload["source_status"]["ecos"] == "unavailable_missing_key"
    assert payload["unavailable_sources"][0]["required_env"] == "BOK_ECOS_API_KEY"


@pytest.mark.unit
def test_initial_state_contains_phase1_macro_and_decision_fields():
    state = Propagator().create_initial_state("AAPL", "2026-05-20")

    assert state["macro_report"] == ""
    assert state["macro_snapshot"] == {}
    assert state["macro_data_quality"] == {}
    assert state["agent_scores"] == {}
    assert state["final_rating"] == ""
    assert state["conditional_action"] == ""
    assert state["invalidation_conditions"] == []
    assert state["decision_log_payload"] == {}


@pytest.mark.unit
def test_macro_analyst_factory_is_importable():
    assert callable(create_macro_analyst)
