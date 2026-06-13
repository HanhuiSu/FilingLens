"""LLM-backed Research Planner V1."""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any, Mapping

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from src.agent.llm import _get_llm, _parse_json_response

logger = logging.getLogger(__name__)


RESEARCH_PLANNER_PROMPT = """\
Return ONLY one valid JSON object. Do not answer the user.

You are building a research plan for a financial-filings analysis agent.
Your job is to understand what must be answered, not to classify into a single old intent label.

Today: {current_date}
User query: {user_query}
Resolved companies: {companies}
Canonical intent fallback: {canonical_intent}
Query understanding: {query_understanding}

Allowed question_type values:
direct_fact, trend_analysis, causal_explanation, overview, risk_analysis,
valuation_boundary, cash_flow_quality, comparison, cautious_outlook,
investment_boundary, unknown

Allowed evidence request types: numeric, text, calculation, event
Allowed tools: search_filings, query_financial_data, compute_metrics, query_event_price_window
Allowed metrics include: revenue, revenue_growth, net_income, operating_margin,
net_margin, operating_cash_flow, free_cash_flow, capital_expenditure, total_debt,
cash_and_equivalents, market_cap, pe_ratio, ps_ratio, price, adjusted_close.
Allowed filing sections include: ITEM_1, ITEM_1A, ITEM_2, ITEM_7, ITEM_8, MD&A, BUSINESS.

Required JSON schema:
{{
  "question_type": "causal_explanation",
  "user_goal": "<what the user needs answered>",
  "companies": ["NVDA"],
  "time_scope": "latest",
  "required_answer_parts": [
    {{
      "id": "quantify_growth",
      "description": "<must-answer part>",
      "required": true,
      "evidence_request_ids": ["revenue_growth_numeric"],
      "success_criteria": ["<how this part is satisfied>"],
      "forbidden_without_evidence": []
    }}
  ],
  "evidence_requests": [
    {{
      "id": "revenue_growth_numeric",
      "type": "numeric",
      "scope": "core",
      "company": "NVDA",
      "metrics": ["revenue", "revenue_growth"],
      "sections": [],
      "queries": [],
      "tool": "query_financial_data",
      "required": true,
      "min_results": 1,
      "answer_part_ids": ["quantify_growth"],
      "purpose": "<why this evidence is needed>",
      "fallback_strategy": ["latest_period"]
    }}
  ],
  "fallback_answer_policy": "<what to say if core evidence is missing>",
  "minimum_answer_policy": "<minimum bounded analytical answer when direct evidence is incomplete>",
  "reasoning_policy": {{
    "allow_inference": true,
    "allow_hypotheses": true,
    "must_separate_claim_tiers": true,
    "forbid_uncited_definitive_causal_claims": true
  }},
  "forbidden_claims": ["<claims forbidden without evidence>"]
}}

Hard rules:
1. Never return only an intent such as "revenue"; always return required_answer_parts.
2. For why / because / driver / 原因 / 为什么 revenue-growth questions, set question_type=causal_explanation.
3. For causal_explanation, include all three required answer parts:
   quantify_growth, identify_growth_drivers, state_evidence_boundary.
4. For causal_explanation, numeric revenue evidence only satisfies quantify_growth.
5. For causal_explanation, identify_growth_drivers requires text evidence from MD&A, ITEM_2, ITEM_7, operating results, or segment discussion.
6. If direct driver evidence is incomplete, still require a tiered analytical answer: verified_evidence, evidence-based inference, hypotheses_to_verify, counterpoints, and evidence_boundary.
7. For buy/sell/recommendation-like queries, set question_type=investment_boundary and forbid investment advice.
8. Do not ask for unsupported tools, unsupported metrics, or unsupported sections.
"""


def planner_mode(value: Any = None) -> str:
    mode = str(settings.research_planner_mode if value is None else value).strip().lower()
    return mode if mode in {"off", "shadow", "validated", "expanded"} else "expanded"


def build_research_plan_raw(
    *,
    user_query: str,
    companies: list[str],
    canonical_intent: Mapping[str, Any] | None = None,
    query_understanding: Mapping[str, Any] | None = None,
    today: date | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return raw LLM planner JSON plus a small execution trace."""
    started = time.monotonic()
    current_date = (today or date.today()).isoformat()
    trace: dict[str, Any] = {
        "source": "llm",
        "status": "not_run",
        "fallback_used": False,
        "fallback_reason": "",
        "duration_ms": 0,
    }
    try:
        timeout_seconds = float(settings.research_planner_timeout_seconds or 25.0)
        llm = _get_llm(
            reasoning=False,
            temperature=0.0,
            max_tokens=1800,
            timeout=timeout_seconds,
            max_retries=settings.llm_classify_max_retries,
        )
        prompt = RESEARCH_PLANNER_PROMPT.format(
            current_date=current_date,
            user_query=user_query,
            companies=companies,
            canonical_intent=dict(canonical_intent or {}),
            query_understanding=dict(query_understanding or {}),
        )
        response = llm.invoke(
            [
                SystemMessage(content="You are a financial research planner. Output ONLY valid JSON."),
                HumanMessage(content=prompt),
            ]
        )
        raw = _parse_json_response(str(getattr(response, "content", "") or ""))
        trace["status"] = "parsed" if raw else "empty_or_invalid_json"
        if not raw:
            trace["fallback_used"] = True
            trace["fallback_reason"] = "invalid_json"
        trace["duration_ms"] = int((time.monotonic() - started) * 1000)
        return raw, trace
    except Exception as exc:  # pragma: no cover - depends on external LLM availability
        logger.warning("Research planner LLM failed; validator may use fallback: %s", exc)
        error_text = str(exc)
        error_type = type(exc).__name__
        fallback_reason = "timeout" if error_type == "TimeoutError" or "timeout" in error_text.lower() or "timed out" in error_text.lower() else "llm_failed"
        if fallback_reason == "timeout" and not bool(settings.research_planner_fallback_for_timeout):
            raise
        trace.update(
            {
                "status": "llm_failed",
                "fallback_used": True,
                "fallback_reason": fallback_reason,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
                "duration_ms": int((time.monotonic() - started) * 1000),
            }
        )
        return {}, trace
