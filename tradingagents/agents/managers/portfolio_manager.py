"""Portfolio Manager: synthesises the risk-analyst debate into the final decision.

Uses LangChain's ``with_structured_output`` so the LLM produces a typed
``PortfolioDecision`` directly, in a single call.  The result is rendered
back to markdown for storage in ``final_trade_decision`` so memory log,
CLI display, and saved reports continue to consume the same shape they do
today.  When a provider does not expose structured output, the agent falls
back gracefully to free-text generation.
"""

from __future__ import annotations

import hashlib
import json
import logging

from tradingagents.agents.schemas import PortfolioDecision, render_pm_decision
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_language_instruction,
)
from tradingagents.agents.utils.rating import (
    conditional_action_for_rating,
    parse_committee_rating,
)
from tradingagents.agents.utils.scoring import aggregate_data_quality, build_agent_scores
from tradingagents.agents.utils.structured import (
    bind_structured,
)


logger = logging.getLogger(__name__)


def _classify_market(ticker: str) -> str:
    ticker_upper = (ticker or "").upper()
    if ticker_upper.endswith((".KS", ".KQ")):
        return "KR"
    if "." in ticker_upper:
        return "GLOBAL"
    return "US"


def _snapshot_id(snapshot: dict) -> str:
    if not snapshot:
        return ""
    raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _fallback_decision_payload(state: dict, final_trade_decision: str, agent_scores: dict) -> dict:
    rating = parse_committee_rating(final_trade_decision)
    return {
        "rating": rating,
        "conditional_action": conditional_action_for_rating(rating),
        "confidence": 35,
        "data_quality": aggregate_data_quality(state, agent_scores),
        "entry_zone": "N/A",
        "invalidation_conditions": [],
        "sizing": "No new position until a structured execution plan is available.",
        "holding_horizon": "N/A",
        "review_trigger": "Re-run analysis when new data or price action changes the thesis.",
    }


def build_decision_log_payload(
    state: dict,
    *,
    final_rating: str,
    conditional_action: str,
    confidence: int,
    data_quality: str,
    entry_zone: str,
    invalidation_conditions: list[str],
    sizing: str,
    holding_horizon: str,
    review_trigger: str,
    agent_scores: dict,
) -> dict:
    ticker = state.get("company_of_interest", "")
    return {
        "analysis_date": state.get("trade_date", ""),
        "ticker": ticker,
        "market": _classify_market(ticker),
        "price_at_decision": 0,
        "agent_scores": {
            key: agent_scores.get(key, 0)
            for key in [
                "macro_score",
                "market_trend_score",
                "valuation_risk_score",
                "sentiment_score",
                "news_event_risk_score",
            ]
        },
        "final_rating": final_rating,
        "conditional_action": conditional_action,
        "confidence": confidence,
        "data_quality": data_quality,
        "entry_zone": entry_zone,
        "invalidation_conditions": invalidation_conditions,
        "sizing": sizing,
        "holding_horizon": holding_horizon,
        "review_trigger": review_trigger,
        "macro_snapshot_id": _snapshot_id(state.get("macro_snapshot", {})),
        "report_id": f"{ticker}-{state.get('trade_date', '')}",
        "evaluation_targets": {
            "return_1w": None,
            "return_1m": None,
            "return_3m": None,
            "max_drawdown_after_decision": None,
            "max_upside_after_decision": None,
            "error_reason": None,
            "next_rule_update": None,
        },
    }


def create_portfolio_manager(llm):
    structured_llm = bind_structured(llm, PortfolioDecision, "Portfolio Manager")

    def portfolio_manager_node(state) -> dict:
        agent_scores = state.get("agent_scores") or build_agent_scores(state)
        data_quality = aggregate_data_quality(state, agent_scores)
        instrument_context = build_instrument_context(state["company_of_interest"])

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        research_plan = state["investment_plan"]
        trader_plan = state["trader_investment_plan"]

        past_context = state.get("past_context", "")
        lessons_line = (
            f"- Lessons from prior decisions and outcomes:\n{past_context}\n"
            if past_context
            else ""
        )

        prompt = f"""As the Portfolio Manager, synthesize the risk analysts' debate and deliver the final investment committee decision.

{instrument_context}

---

**Final Rating Scale** (use exactly one):
- **Strong Buy**: 강한 매수 후보
- **Buy**: 매수 후보
- **Watch**: 관심/관찰
- **Hold**: 보유
- **Reduce**: 비중 축소
- **Avoid**: 회피

**Conditional Action Scale** (use exactly one):
- **Immediate entry possible**: 즉시 진입 가능
- **Wait for pullback**: 눌림목 대기
- **Watch only**: 관찰만
- **Hold existing position only**: 기존 보유만
- **Reduce exposure**: 비중 축소
- **Avoid new entry**: 신규 진입 회피

The rule-based scoring layer below is the source of truth for scores. Interpret
and challenge these scores; do not invent new scores.

**Rule-based scores JSON:**
```json
{json.dumps(agent_scores, ensure_ascii=False, indent=2)}
```

Overall deterministic data quality estimate: **{data_quality}**

**Context:**
- Research Manager's investment plan: **{research_plan}**
- Trader's transaction proposal: **{trader_plan}**
{lessons_line}
**Risk Analysts Debate History:**
{history}

---

Be decisive and ground every conclusion in specific evidence from the analysts.
The user-facing output must be Korean and must include that this is not financial advice.{get_language_instruction()}"""

        decision_payload = None
        if structured_llm is not None:
            try:
                decision = structured_llm.invoke(prompt)
                final_trade_decision = render_pm_decision(decision)
                decision_payload = {
                    "rating": decision.rating.value,
                    "conditional_action": decision.conditional_action.value,
                    "confidence": decision.confidence,
                    "data_quality": decision.data_quality,
                    "entry_zone": decision.entry_zone or "N/A",
                    "invalidation_conditions": decision.invalidation_conditions,
                    "sizing": decision.sizing,
                    "holding_horizon": decision.time_horizon or "N/A",
                    "review_trigger": decision.review_trigger or "N/A",
                }
            except Exception as exc:
                logger.warning(
                    "Portfolio Manager: structured-output invocation failed (%s); retrying once as free text",
                    exc,
                )

        if decision_payload is None:
            response = llm.invoke(prompt)
            final_trade_decision = response.content
            decision_payload = _fallback_decision_payload(
                state, final_trade_decision, agent_scores
            )

        decision_log_payload = build_decision_log_payload(
            state,
            final_rating=decision_payload["rating"],
            conditional_action=decision_payload["conditional_action"],
            confidence=decision_payload["confidence"],
            data_quality=decision_payload["data_quality"],
            entry_zone=decision_payload["entry_zone"],
            invalidation_conditions=decision_payload["invalidation_conditions"],
            sizing=decision_payload["sizing"],
            holding_horizon=decision_payload["holding_horizon"],
            review_trigger=decision_payload["review_trigger"],
            agent_scores=agent_scores,
        )

        new_risk_debate_state = {
            "judge_decision": final_trade_decision,
            "history": risk_debate_state["history"],
            "aggressive_history": risk_debate_state["aggressive_history"],
            "conservative_history": risk_debate_state["conservative_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_aggressive_response": risk_debate_state["current_aggressive_response"],
            "current_conservative_response": risk_debate_state["current_conservative_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": final_trade_decision,
            "agent_scores": agent_scores,
            "final_rating": decision_payload["rating"],
            "conditional_action": decision_payload["conditional_action"],
            "invalidation_conditions": decision_payload["invalidation_conditions"],
            "decision_log_payload": decision_log_payload,
        }

    return portfolio_manager_node
