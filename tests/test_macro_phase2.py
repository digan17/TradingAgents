import json
from datetime import date

import pytest
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils import macro_data_tools
from tradingagents.agents.utils.macro_data_tools import (
    _cache_file,
    _read_cache,
    _write_cache,
    get_macro_indicators,
)
from tradingagents.dataflows.config import set_config
from tradingagents.graph.conditional_logic import ConditionalLogic
from tradingagents.graph.setup import GraphSetup
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _fred_fixture(series_id: str) -> dict:
    latest = "4.30" if series_id == "DGS2" else "4.50"
    return {
        "observations": [
            {"date": "2025-11-20", "value": "3.90"},
            {"date": "2026-02-20", "value": "4.05"},
            {"date": "2026-04-20", "value": "4.30"},
            {"date": "2026-05-20", "value": latest},
        ]
    }


def _ecos_fixture(url: str) -> dict:
    if "/D/" in url:
        rows = [
            {"TIME": "20251120", "DATA_VALUE": "1300.0"},
            {"TIME": "20260220", "DATA_VALUE": "1320.0"},
            {"TIME": "20260420", "DATA_VALUE": "1340.0"},
            {"TIME": "20260520", "DATA_VALUE": "1350.0"},
        ]
    else:
        rows = [
            {"TIME": "202510", "DATA_VALUE": "100.0"},
            {"TIME": "202601", "DATA_VALUE": "101.0"},
            {"TIME": "202603", "DATA_VALUE": "102.0"},
            {"TIME": "202604", "DATA_VALUE": "103.0"},
        ]
    return {"StatisticSearch": {"row": rows}}


@pytest.mark.unit
def test_fred_fixture_computes_latest_changes_zscore_and_cache(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setenv("FRED_API_KEY", "fixture-key")
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)

    calls = []

    def fake_get_json(url, params=None):
        calls.append((url, params))
        return _fred_fixture(params["series_id"])

    monkeypatch.setattr(macro_data_tools, "_get_json", fake_get_json)

    payload = json.loads(get_macro_indicators.func("US", "2026-05-20", 18))
    dgs10 = next(item for item in payload["series"] if item["id"] == "DGS10")

    assert payload["available"] is True
    assert payload["source_status"]["fred"] == "available"
    assert payload["derived"]["yield_curve_10y2y"] == 0.2
    assert dgs10["latest_observation_date"] == "2026-05-20"
    assert dgs10["latest_value"] == 4.5
    assert dgs10["change_1m"] == 0.2
    assert dgs10["change_3m"] == 0.45
    assert dgs10["change_6m"] == 0.6
    assert dgs10["zscore_18m"] is not None
    assert dgs10["data_lag_days"] == 0
    assert dgs10["quality"] == "ok"
    assert dgs10["cache_status"] == "miss"
    assert calls

    calls.clear()
    cached_payload = json.loads(get_macro_indicators.func("US", "2026-05-20", 18))
    assert cached_payload["cache_status"] == "hit"
    assert calls == []


@pytest.mark.unit
def test_ecos_fixture_computes_latest_changes_zscore_and_lag(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setenv("BOK_ECOS_API_KEY", "fixture-key")
    monkeypatch.setattr(macro_data_tools, "_get_json", lambda url, params=None: _ecos_fixture(url))
    monkeypatch.setattr(macro_data_tools, "_today_utc", lambda: date(2026, 5, 21))

    payload = json.loads(get_macro_indicators.func("KR", "2026-05-20", 18))
    base_rate = next(item for item in payload["series"] if item["id"] == "722Y001:0101000")

    assert payload["available"] is True
    assert payload["source_status"]["ecos"] == "available"
    assert payload["data_revision_risk"] is True
    assert base_rate["latest_observation_date"] == "2026-04-30"
    assert base_rate["latest_value"] == 103.0
    assert base_rate["change_1m"] == 1.0
    assert base_rate["change_3m"] == 2.0
    assert base_rate["change_6m"] == 3.0
    assert base_rate["zscore_18m"] is not None
    assert base_rate["quality"] == "ok"


@pytest.mark.unit
def test_macro_cache_key_includes_required_dimensions(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    _write_cache(
        "FRED",
        "DGS10",
        "US",
        "2026-05-20",
        18,
        {"observations": []},
    )

    cache_path = _cache_file("FRED", "DGS10", "US", "2026-05-20", 18)
    cached = _read_cache("FRED", "DGS10", "US", "2026-05-20", 18)

    assert cache_path.exists()
    assert "fred__DGS10__US__2026-05-20__18" in cache_path.name
    assert cached["payload"] == {"observations": []}


@pytest.mark.unit
def test_bad_fred_response_degrades_without_crashing(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setenv("FRED_API_KEY", "fixture-key")
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)
    monkeypatch.setattr(
        macro_data_tools,
        "_get_json",
        lambda url, params=None: (_ for _ in ()).throw(RuntimeError("bad response")),
    )

    payload = json.loads(get_macro_indicators.func("US", "2026-05-20", 18))

    assert payload["available"] is False
    assert payload["quality"] == "low"
    assert payload["source_status"]["fred"] == "unavailable_api_error"
    assert payload["unavailable_sources"][0]["reason"] == "api_error"
    assert payload["errors"]


@pytest.mark.unit
def test_unsupported_market_falls_back_to_global(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.delenv("BOK_ECOS_API_KEY", raising=False)

    payload = json.loads(get_macro_indicators.func("JP", "2026-05-20", 18))

    assert payload["market"] == "GLOBAL"
    assert payload["source_status"]["fred"] == "unavailable_missing_key"
    assert payload["source_status"]["ecos"] == "not_applicable"


@pytest.mark.unit
def test_should_continue_macro_routes_tool_calls():
    logic = ConditionalLogic()

    with_tool = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "get_macro_indicators",
                "args": {"market": "US", "curr_date": "2026-05-20"},
                "id": "macro-1",
            }
        ],
    )
    without_tool = AIMessage(content="macro report")

    assert logic.should_continue_macro({"messages": [with_tool]}) == "tools_macro"
    assert logic.should_continue_macro({"messages": [without_tool]}) == "Msg Clear Macro"


@pytest.mark.unit
def test_graph_setup_accepts_macro_only_and_default_macro_flow():
    tool_nodes = {
        "market": ToolNode([]),
        "macro": ToolNode([get_macro_indicators]),
        "social": ToolNode([]),
        "news": ToolNode([]),
        "fundamentals": ToolNode([]),
    }
    setup = GraphSetup(object(), object(), tool_nodes, ConditionalLogic())

    macro_only = setup.setup_graph(["macro"]).compile()
    default_flow = setup.setup_graph().compile()

    assert macro_only is not None
    assert default_flow is not None


@pytest.mark.unit
def test_trading_graph_default_and_tool_nodes_include_macro():
    default_selected = TradingAgentsGraph.__init__.__defaults__[0]
    tool_nodes = TradingAgentsGraph._create_tool_nodes(object())

    assert default_selected == ["market", "macro", "social", "news", "fundamentals"]
    assert "macro" in tool_nodes
