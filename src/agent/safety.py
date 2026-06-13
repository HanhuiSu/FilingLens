"""Programmatic safety policy for conversational analyst behavior."""

from __future__ import annotations

import re
from typing import Any

from src.agent.constants import ANSWER_MODES, SAFETY_INTENTS
from src.agent.types import SafetyDecision


_INVESTMENT_ADVICE_TERMS = (
    "最看好哪个",
    "更看好哪个",
    "最看好哪一个",
    "更看好哪一个",
    "更值得关注哪个",
    "更值得关注哪一个",
    "值得重点关注哪个",
    "更值得跟踪哪个",
    "推荐哪个",
    "推荐哪一个",
    "买哪个",
    "该买",
    "能买吗",
    "可以买",
    "值得买",
    "买入",
    "卖出",
    "持有",
    "投资哪个",
    "recommend",
    "should i buy",
    "should i sell",
    "which should i buy",
    "which stock should i buy",
    "best investment",
    "buy rating",
    "sell rating",
)

_STOCK_PRICE_TERMS = (
    "股价",
    "股票价格",
    "目标价",
    "price target",
    "target price",
    "stock price",
    "share price",
)

_PREDICTION_TERMS = (
    "预测",
    "预判",
    "明天",
    "后天",
    "下周",
    "会涨",
    "会跌",
    "涨到",
    "跌到",
    "tomorrow",
    "next day",
    "next week",
    "predict",
    "forecast",
    "will rise",
    "will fall",
)

_REALTIME_TERMS = (
    "实时",
    "现在股价",
    "今天股价",
    "最新新闻",
    "实时新闻",
    "当前新闻",
    "live",
    "real-time",
    "realtime",
    "latest news",
    "today's news",
    "current news",
    "current stock price",
)

_FILING_OUTLOOK_TERMS = (
    "财报会怎么样",
    "财报怎么看",
    "财报前景",
    "今年财报",
    "未来财报",
    "公司前景",
    "outlook",
    "prospects",
    "future filings",
    "earnings outlook",
)

_OPEN_ANALYSIS_TERMS = (
    "最大的问题",
    "最大风险",
    "主要问题",
    "核心问题",
    "主要风险",
    "怎么看",
    "分析一下",
    "biggest problem",
    "main issue",
    "largest risk",
    "biggest risk",
    "what is wrong",
)

_OUT_OF_SCOPE_TERMS = (
    "天气",
    "菜谱",
    "做饭",
    "电影推荐",
    "体育比分",
    "weather",
    "recipe",
    "cook",
    "sports score",
    "movie recommendation",
)


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def _reason(code: str, message: str) -> dict[str, Any]:
    return {"code": code, "message": message}


def _limitation(code: str, severity: str, message: str) -> dict[str, Any]:
    return {"code": code, "severity": severity, "message": message}


def is_investment_advice_like_query(user_query: str) -> bool:
    return _contains_any(user_query, _INVESTMENT_ADVICE_TERMS)


def is_realtime_or_stock_price_prediction_query(user_query: str) -> bool:
    q = str(user_query or "").lower()
    if _contains_any(q, _REALTIME_TERMS):
        return True
    if _contains_any(q, _STOCK_PRICE_TERMS) and _contains_any(q, _PREDICTION_TERMS):
        return True
    return bool(re.search(r"(predict|forecast).*(stock|share).*price", q))


def is_filing_or_company_outlook_query(user_query: str) -> bool:
    return _contains_any(user_query, _FILING_OUTLOOK_TERMS)


def is_open_ended_company_analysis_query(user_query: str) -> bool:
    return _contains_any(user_query, _OPEN_ANALYSIS_TERMS)


def is_out_of_scope_query(user_query: str) -> bool:
    return _contains_any(user_query, _OUT_OF_SCOPE_TERMS)


def apply_safety_policy(
    *,
    user_query: str,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    needs_tools: bool,
    needs_clarification: bool,
    companies: list[str],
    comparison_target: str | None = None,
) -> SafetyDecision:
    """Normalize conversational safety behavior after intent routing.

    The decision can tighten answer mode, safety intent, and tool use, but it
    does not grant tools/metrics/sections. Those remain validated by the
    deterministic analysis-plan layer.
    """
    _ = task_type, comparison_target
    mode = answer_mode if answer_mode in ANSWER_MODES else "direct_fact"
    intent = safety_intent if safety_intent in SAFETY_INTENTS else "normal"
    tool_flag = bool(needs_tools)
    reasons: list[dict[str, Any]] = []
    limitations: list[dict[str, Any]] = []
    non_advisory = False
    forward_caution = False
    no_realtime = False

    if mode in {"meta", "clarification"} or needs_clarification:
        tool_flag = False
        reasons.append(_reason("conversational_no_tools", "Meta and clarification responses do not call financial tools."))

    if is_out_of_scope_query(user_query):
        mode = "refusal_or_redirect"
        intent = "unsupported_or_out_of_scope"
        tool_flag = False
        reasons.append(_reason("unsupported_scope", "Question is outside supported filings/company-analysis scope."))
        limitations.append(
            _limitation(
                "unsupported_scope",
                "medium",
                "The request is outside supported filings, structured financial facts, and filing-event analysis scope.",
            )
        )

    if is_realtime_or_stock_price_prediction_query(user_query):
        mode = "refusal_or_redirect"
        intent = "unsupported_or_out_of_scope"
        tool_flag = False
        no_realtime = True
        reasons.append(
            _reason(
                "no_realtime_or_price_prediction",
                "The agent has no web search/live quote source and must not predict near-term stock prices.",
            )
        )
        limitations.append(
            _limitation(
                "no_realtime_news_access",
                "high",
                "No web search or real-time market data source is available; do not claim current news or live prices.",
            )
        )
        limitations.append(
            _limitation(
                "unsupported_price_prediction",
                "high",
                "Near-term stock-price prediction is outside the supported scope.",
            )
        )

    if is_investment_advice_like_query(user_query):
        intent = "investment_advice_like"
        non_advisory = True
        if len(set(companies)) >= 2 or task_type == "company_comparison":
            mode = "comparison_brief"
            tool_flag = True
        reasons.append(
            _reason(
                "non_advisory_reframe",
                "Investment-advice-like wording is reframed as evidence-grounded analytical comparison.",
            )
        )
        limitations.append(
            _limitation(
                "investment_advice_boundary",
                "high",
                "This is analysis only, not investment advice, a buy/sell recommendation, or a price forecast.",
            )
        )

    if mode == "direct_fact" and companies and is_open_ended_company_analysis_query(user_query):
        mode = "analytical"
        tool_flag = True
        reasons.append(
            _reason(
                "open_ended_analysis_reframe",
                "Open-ended company problem/risk questions should gather filing text, not only structured facts.",
            )
        )

    if mode == "cautious_outlook" or is_filing_or_company_outlook_query(user_query):
        if intent != "unsupported_or_out_of_scope":
            mode = "cautious_outlook"
            tool_flag = bool(companies)
        forward_caution = True
        reasons.append(
            _reason(
                "forward_looking_caution",
                "Forward-looking filing/company questions must be framed as disclosed-data observations, not predictions.",
            )
        )
        limitations.append(
            _limitation(
                "forward_looking_uncertainty",
                "medium",
                "Forward-looking discussion is uncertain and bounded by validated historical facts and filing evidence.",
            )
        )

    if mode in {"meta", "clarification", "refusal_or_redirect"} or intent == "unsupported_or_out_of_scope":
        tool_flag = False

    return SafetyDecision(
        answer_mode=mode,
        safety_intent=intent,
        needs_tools=tool_flag,
        requires_non_advisory_framing=non_advisory,
        requires_forward_looking_caution=forward_caution,
        disallows_realtime_claims=no_realtime,
        policy_reasons=reasons,
        limitations=limitations,
    )
