"""Pydantic schemas used by agents that produce structured output.

The framework's primary artifact is still prose: each agent's natural-language
reasoning is what users read in the saved markdown reports and what the
downstream agents read as context.  Structured output is layered onto the
three decision-making agents (Research Manager, Trader, Portfolio Manager)
so that:

- Their outputs follow consistent section headers across runs and providers
- Each provider's native structured-output mode is used (json_schema for
  OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic)
- Schema field descriptions become the model's output instructions, freeing
  the prompt body to focus on context and the rating-scale guidance
- A render helper turns the parsed Pydantic instance back into the same
  markdown shape the rest of the system already consumes, so display,
  memory log, and saved reports keep working unchanged
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared rating types
# ---------------------------------------------------------------------------


class PortfolioRating(str, Enum):
    """5-tier rating used by the Research Manager and Portfolio Manager."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


class CommitteeRating(str, Enum):
    """6-tier final investment committee rating."""

    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    WATCH = "Watch"
    HOLD = "Hold"
    REDUCE = "Reduce"
    AVOID = "Avoid"


class ConditionalAction(str, Enum):
    """Final conditional action for user-facing reports."""

    IMMEDIATE_ENTRY = "Immediate entry possible"
    WAIT_FOR_PULLBACK = "Wait for pullback"
    WATCH_ONLY = "Watch only"
    HOLD_EXISTING_ONLY = "Hold existing position only"
    REDUCE_EXPOSURE = "Reduce exposure"
    AVOID_NEW_ENTRY = "Avoid new entry"


class TraderAction(str, Enum):
    """3-tier transaction direction used by the Trader.

    The Trader's job is to translate the Research Manager's investment plan
    into a concrete transaction proposal: should the desk execute a Buy, a
    Sell, or sit on Hold this round.  Position sizing and the nuanced
    Overweight / Underweight calls happen later at the Portfolio Manager.
    """

    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"


# ---------------------------------------------------------------------------
# Research Manager
# ---------------------------------------------------------------------------


class ResearchPlan(BaseModel):
    """Structured investment plan produced by the Research Manager.

    Hand-off to the Trader: the recommendation pins the directional view,
    the rationale captures which side of the bull/bear debate carried the
    argument, and the strategic actions translate that into concrete
    instructions the trader can execute against.
    """

    recommendation: PortfolioRating = Field(
        description=(
            "The investment recommendation. Exactly one of Buy / Overweight / "
            "Hold / Underweight / Sell. Reserve Hold for situations where the "
            "evidence on both sides is genuinely balanced; otherwise commit to "
            "the side with the stronger arguments."
        ),
    )
    rationale: str = Field(
        description=(
            "Conversational summary of the key points from both sides of the "
            "debate, ending with which arguments led to the recommendation. "
            "Speak naturally, as if to a teammate."
        ),
    )
    strategic_actions: str = Field(
        description=(
            "Concrete steps for the trader to implement the recommendation, "
            "including position sizing guidance consistent with the rating."
        ),
    )


def render_research_plan(plan: ResearchPlan) -> str:
    """Render a ResearchPlan to markdown for storage and the trader's prompt context."""
    return "\n".join([
        f"**Recommendation**: {plan.recommendation.value}",
        "",
        f"**Rationale**: {plan.rationale}",
        "",
        f"**Strategic Actions**: {plan.strategic_actions}",
    ])


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TraderProposal(BaseModel):
    """Structured transaction proposal produced by the Trader.

    The trader reads the Research Manager's investment plan and the analyst
    reports, then turns them into a concrete transaction: what action to
    take, the reasoning that justifies it, and the practical levels for
    entry, stop-loss, and sizing.
    """

    action: TraderAction = Field(
        description="The transaction direction. Exactly one of Buy / Hold / Sell.",
    )
    reasoning: str = Field(
        description=(
            "The case for this action, anchored in the analysts' reports and "
            "the research plan. Two to four sentences."
        ),
    )
    entry_price: Optional[float] = Field(
        default=None,
        description="Optional entry price target in the instrument's quote currency.",
    )
    stop_loss: Optional[float] = Field(
        default=None,
        description="Optional stop-loss price in the instrument's quote currency.",
    )
    position_sizing: Optional[str] = Field(
        default=None,
        description="Optional sizing guidance, e.g. '5% of portfolio'.",
    )
    holding_horizon: Optional[str] = Field(
        default=None,
        description="Optional proposed holding horizon, e.g. '2-6 weeks'.",
    )
    review_trigger: Optional[str] = Field(
        default=None,
        description="Optional condition that should trigger a review.",
    )


def render_trader_proposal(proposal: TraderProposal) -> str:
    """Render a TraderProposal to markdown.

    The trailing ``FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL**`` line is
    preserved for backward compatibility with the analyst stop-signal text
    and any external code that greps for it.
    """
    parts = [
        f"**Action**: {proposal.action.value}",
        "",
        f"**Reasoning**: {proposal.reasoning}",
    ]
    if proposal.entry_price is not None:
        parts.extend(["", f"**Entry Price**: {proposal.entry_price}"])
    if proposal.stop_loss is not None:
        parts.extend(["", f"**Stop Loss**: {proposal.stop_loss}"])
    if proposal.position_sizing:
        parts.extend(["", f"**Position Sizing**: {proposal.position_sizing}"])
    if proposal.holding_horizon:
        parts.extend(["", f"**Holding Horizon**: {proposal.holding_horizon}"])
    if proposal.review_trigger:
        parts.extend(["", f"**Review Trigger**: {proposal.review_trigger}"])
    parts.extend([
        "",
        f"FINAL TRANSACTION PROPOSAL: **{proposal.action.value.upper()}**",
    ])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------


class PortfolioDecision(BaseModel):
    """Structured output produced by the Portfolio Manager.

    The model fills every field as part of its primary LLM call; no separate
    extraction pass is required. Field descriptions double as the model's
    output instructions, so the prompt body only needs to convey context and
    the rating-scale guidance.
    """

    rating: CommitteeRating = Field(
        description=(
            "The final rating. Exactly one of Strong Buy / Buy / Watch / Hold / "
            "Reduce / Avoid."
        ),
    )
    conditional_action: ConditionalAction = Field(
        description=(
            "The conditional action. Exactly one of Immediate entry possible / "
            "Wait for pullback / Watch only / Hold existing position only / "
            "Reduce exposure / Avoid new entry."
        ),
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Final confidence from 0 to 100.",
    )
    data_quality: str = Field(
        description="Overall data quality: high, medium, or low.",
    )
    executive_summary: str = Field(
        description=(
            "A concise action plan covering entry strategy, position sizing, "
            "key risk levels, and time horizon. Two to four sentences."
        ),
    )
    investment_thesis: str = Field(
        description=(
            "Detailed reasoning anchored in specific evidence from the analysts' "
            "debate. If prior lessons are referenced in the prompt context, "
            "incorporate them; otherwise rely solely on the current analysis."
        ),
    )
    price_target: Optional[float] = Field(
        default=None,
        description="Optional target price in the instrument's quote currency.",
    )
    entry_zone: Optional[str] = Field(
        default=None,
        description="Suggested entry zone or condition. Use 'N/A' if no entry is advised.",
    )
    invalidation_conditions: list[str] = Field(
        default_factory=list,
        description="Conditions that invalidate the investment thesis.",
    )
    risk_summary: list[str] = Field(
        default_factory=list,
        description="Key risk factors that drive the final decision.",
    )
    sizing: str = Field(
        default="No new position until conditions improve.",
        description="Position sizing suggestion.",
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Optional recommended holding period, e.g. '3-6 months'.",
    )
    review_trigger: Optional[str] = Field(
        default=None,
        description="Trigger for re-review after the decision.",
    )


RATING_KO = {
    CommitteeRating.STRONG_BUY: "강한 매수 후보",
    CommitteeRating.BUY: "매수 후보",
    CommitteeRating.WATCH: "관심/관찰",
    CommitteeRating.HOLD: "보유",
    CommitteeRating.REDUCE: "비중 축소",
    CommitteeRating.AVOID: "회피",
}

ACTION_KO = {
    ConditionalAction.IMMEDIATE_ENTRY: "즉시 진입 가능",
    ConditionalAction.WAIT_FOR_PULLBACK: "눌림목 대기",
    ConditionalAction.WATCH_ONLY: "관찰만",
    ConditionalAction.HOLD_EXISTING_ONLY: "기존 보유만",
    ConditionalAction.REDUCE_EXPOSURE: "비중 축소",
    ConditionalAction.AVOID_NEW_ENTRY: "신규 진입 회피",
}


def render_pm_decision(decision: PortfolioDecision) -> str:
    """Render a PortfolioDecision as a Korean investment committee report."""
    risks = decision.risk_summary or ["명시된 핵심 리스크가 제한적입니다."]
    invalidations = decision.invalidation_conditions or ["명시된 무효화 조건 없음"]
    parts = [
        "# 투자위원회 종합 리포트",
        "",
        "## 1. 요약 판단",
        f"- 최종 등급: **{RATING_KO[decision.rating]}** (`{decision.rating.value}`)",
        f"- 조건부 액션: **{ACTION_KO[decision.conditional_action]}** (`{decision.conditional_action.value}`)",
        f"- 확신도: **{decision.confidence}/100**",
        f"- 데이터 품질: **{decision.data_quality}**",
        "",
        "## 2. 핵심 판단 근거",
        decision.executive_summary,
        "",
        "## 3. 투자 thesis",
        decision.investment_thesis,
        "",
        "## 4. 리스크와 무효화 조건",
        *[f"- 리스크: {risk}" for risk in risks],
        *[f"- 무효화 조건: {condition}" for condition in invalidations],
        "",
        "## 5. 실행 계획",
        f"- 진입 구간: {decision.entry_zone or 'N/A'}",
        f"- 사이징: {decision.sizing}",
        f"- 보유 기간: {decision.time_horizon or 'N/A'}",
        f"- 재검토 조건: {decision.review_trigger or 'N/A'}",
    ]
    if decision.price_target is not None:
        parts.append(f"- 목표가: {decision.price_target}")
    parts.extend([
        "",
        "## 6. 최종 결론",
        decision.executive_summary,
        "",
        "본 리포트는 투자 판단을 돕기 위한 리서치 보조 자료이며, 투자 자문이나 매수/매도 권유가 아닙니다.",
        "",
        f"**Rating**: {decision.rating.value}",
        f"**Conditional Action**: {decision.conditional_action.value}",
        f"**Confidence**: {decision.confidence}",
        f"**Data Quality**: {decision.data_quality}",
    ])
    return "\n".join(parts)
