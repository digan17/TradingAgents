import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
    get_macro_indicators,
)


def _classify_market(ticker: str) -> str:
    ticker_upper = (ticker or "").upper()
    if ticker_upper.endswith((".KS", ".KQ")):
        return "KR"
    if "." in ticker_upper:
        return "GLOBAL"
    return "US"


def _macro_quality(snapshot: dict) -> dict:
    return {
        "quality": snapshot.get("quality", "low"),
        "quality_note": snapshot.get("quality_note", ""),
        "source_status": snapshot.get("source_status", {}),
        "data_revision_risk": snapshot.get("data_revision_risk", False),
    }


def create_macro_analyst(llm):
    def macro_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        market = _classify_market(ticker)
        instrument_context = build_instrument_context(ticker)

        snapshot_text = get_macro_indicators.func(market, current_date, 18)
        try:
            snapshot = json.loads(snapshot_text)
        except json.JSONDecodeError:
            snapshot = {
                "available": False,
                "as_of_date": current_date,
                "market": market,
                "source_status": {},
                "series": [],
                "derived": {},
                "quality": "low",
                "quality_note": "Macro tool returned invalid JSON.",
            }

        system_message = (
            "You are the Macro Analyst in a risk-first investment committee. "
            "Use only the macro snapshot supplied below. Do not infer unavailable "
            "macro indicators, and state clearly when external macro data was not "
            "used. Focus on rates, inflation, FX, oil, growth, yield curve, "
            "volatility, liquidity proxy, macro regime, and valuation multiple risk.\n\n"
            f"Macro snapshot JSON:\n```json\n{json.dumps(snapshot, ensure_ascii=False, indent=2)}\n```\n\n"
            "Produce a concise Korean macro report with: 기준일, 데이터 사용 가능 여부, "
            "source status, macro regime, key risks, and invalidating conditions."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "{system_message}\nFor your reference, the current date is "
                    "{current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result = (prompt | llm).invoke(state["messages"])
        return {
            "messages": [result],
            "macro_report": result.content,
            "macro_snapshot": snapshot,
            "macro_data_quality": _macro_quality(snapshot),
        }

    return macro_analyst_node
