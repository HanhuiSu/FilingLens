# ruff: noqa: F401,F403,F405
"""Tool execution orchestration for the financial-analysis agent."""

from __future__ import annotations

import logging
from typing import Any

from config import settings
from src.agent import requirement_executor
from src.agent.constants import *
from src.agent.evidence_planner import build_evidence_plan
from src.agent.evidence_sufficiency import (
    build_requirement_status_map,
    build_trace_summary,
    collection_result,
    evaluate_evidence_sufficiency,
    normalize_dimension_status_contract,
    summarize_evidence_requirements,
)
from src.agent.evidence import _collect_event_rows, _collect_financial_rows, _ordered_unique_tickers, _period_year, _rows_for
from src.agent.progress import append_progress_event
from src.agent.query_plan import (
    _build_event_query,
    _build_retrieval_policy,
    _default_period_query,
    _detect_event_intent,
    _has_explicit_year,
    _infer_period_type,
    _is_recency_query,
    _resolve_query_plan,
)
from src.agent.state import AgentState
from src.tools.compute_metrics import compute_metrics
from src.tools.query_event_price_window import query_event_price_window
from src.tools.query_financial_data import query_financial_data
from src.tools.adapters.search_filings_tool import SearchFilingsTool
from src.tools.protocol import ToolExecutionContext, execute_tool_with_timeout
from src.tools.search_filings import search_filings, search_filings_lexical_fallback

logger = logging.getLogger(__name__)


def _progress(
    state: AgentState,
    event: str,
    status: str,
    message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    trace_id = str(state.get("trace_id") or "")
    if not trace_id:
        return
    append_progress_event(
        trace_id,
        event,
        status,
        message,
        node="execute_tools",
        metadata=metadata or {},
    )


def _search_filings_lexical_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = search_filings_lexical_fallback(
        ticker=str(payload.get("ticker", "")),
        query=str(payload.get("query", "")),
        top_k=int(payload.get("top_k", settings.retrieval_top_k) or settings.retrieval_top_k),
        form_type=payload.get("form_type"),
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
        section_allowlist=payload.get("section_allowlist"),
        strict_sections=bool(payload.get("strict_sections", False)),
        retrieval_profile=payload.get("retrieval_profile"),
        target_periods=payload.get("target_periods"),
        max_per_filing=payload.get("max_per_filing"),
        max_per_section=payload.get("max_per_section"),
        return_diagnostics=False,
    )
    return result if isinstance(result, list) else []


def _search_filings_payload(
    payload: dict[str, Any],
    state: AgentState,
    *,
    ticker: str,
    requirement_id: str = "",
) -> tuple[list[dict[str, Any]], bool, str]:
    context = ToolExecutionContext(
        trace_id=str(state.get("trace_id") or ""),
        requirement_id=requirement_id or None,
        company=ticker or None,
    )
    result = execute_tool_with_timeout(SearchFilingsTool(search_filings), payload, context)
    if result.ok:
        docs = result.data if isinstance(result.data, list) else []
        return docs, False, ""
    message = str(result.error.message) if result.error else "search_filings failed"
    if result.error and str(result.error.code) == "timeout":
        logger.warning(
            "search_filings timed out for %s requirement=%s; using lexical fallback",
            ticker,
            requirement_id or "-",
        )
        return _search_filings_lexical_payload(payload), True, message
    raise RuntimeError(message)

def _plan_and_requirements(state: AgentState) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan = dict(state.get("evidence_plan", {}) or {})
    if not plan:
        plan = build_evidence_plan(state).model_dump(exclude_none=True)
    requirements = [r for r in plan.get("evidence_requirements", []) if isinstance(r, dict)]
    return plan, requirements

def _requirements_of_type(requirements: list[dict[str, Any]], requirement_type: str) -> list[dict[str, Any]]:
    return [r for r in requirements if str(r.get("requirement_type", "")) == requirement_type]

def _requirement_matches(req: dict[str, Any], ticker: str, metric: str = "") -> bool:
    company = str(req.get("company") or "").upper()
    if company and company != str(ticker).upper():
        return False
    metrics = {str(m) for m in req.get("metrics", []) if str(m).strip()}
    single_metric = str(req.get("metric") or "")
    if single_metric:
        metrics.add(single_metric)
    return not metric or not metrics or metric in metrics

def _first_requirement_id(
    requirements: list[dict[str, Any]],
    requirement_type: str,
    ticker: str,
    metric: str = "",
) -> str:
    for req in requirements:
        if str(req.get("requirement_type", "")) != requirement_type:
            continue
        if _requirement_matches(req, ticker, metric):
            return str(req.get("requirement_id", ""))
    return ""

def _tools_from_requirements(requirements: list[dict[str, Any]], state: AgentState) -> list[str]:
    tools: list[str] = []
    for req in requirements:
        req_type = str(req.get("requirement_type", ""))
        if req_type == "numeric":
            tools.append("query_financial_data")
        elif req_type == "calculation":
            tools.append("compute_metrics")
        elif req_type == "text":
            tools.append("search_filings")
        elif req_type == "event" and (req.get("required") or str(state.get("event_intent", "none")) == "required"):
            tools.append("query_event_price_window")
    return list(dict.fromkeys(t for t in tools if t in ALLOWED_ANALYSIS_TOOLS))

def _metrics_from_requirements(requirements: list[dict[str, Any]]) -> list[str]:
    metrics: list[str] = []
    for req in requirements:
        if str(req.get("requirement_type", "")) not in {"numeric", "calculation"}:
            continue
        for metric in list(req.get("metrics", []) or []) + [req.get("metric")]:
            metric_text = str(metric or "").strip()
            if metric_text in ALLOWED_ANALYSIS_METRICS and metric_text not in metrics:
                metrics.append(metric_text)
    return metrics

def _tag_financial_result(result: Any, ticker: str, requirements: list[dict[str, Any]]) -> Any:
    if not isinstance(result, dict):
        return result
    numeric_reqs = _requirements_of_type(requirements, "numeric")
    for row in list(result.get("financial_facts", []) or []):
        if not isinstance(row, dict):
            continue
        rid = _first_requirement_id(numeric_reqs, "numeric", str(row.get("ticker", ticker)), str(row.get("metric", "")))
        if rid:
            row["requirement_id"] = rid
    for row in list(result.get("price_data", []) or []):
        if not isinstance(row, dict):
            continue
        row_ticker = str(row.get("ticker", ticker))
        for metric in row:
            if metric in {"ticker", "date"}:
                continue
            rid = _first_requirement_id(numeric_reqs, "numeric", row_ticker, str(metric))
            if rid:
                row["requirement_id"] = rid
                break
    return result

def _tag_event_result(result: Any, ticker: str, requirements: list[dict[str, Any]]) -> Any:
    if not isinstance(result, dict):
        return result
    event_reqs = _requirements_of_type(requirements, "event")
    rid = _first_requirement_id(event_reqs, "event", ticker)
    if not rid:
        return result
    for event in list(result.get("events", []) or []):
        if isinstance(event, dict):
            event["requirement_id"] = rid
    result["requirement_id"] = rid
    return result

def _build_collection_results(
    evidence_plan: dict[str, Any],
    tool_results: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    retry_counts: dict[str, int],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    requirements = [r for r in evidence_plan.get("evidence_requirements", []) if isinstance(r, dict)]
    for req in requirements:
        rid = str(req.get("requirement_id", ""))
        req_type = str(req.get("requirement_type", ""))
        min_results = int(req.get("min_results", 1) or 1)
        items: list[dict[str, Any]] = []
        if req_type == "text":
            items = [d for d in docs if str(d.get("requirement_id", "")) == rid]
        elif req_type == "numeric":
            for tr in tool_results:
                if tr.get("tool") != "query_financial_data":
                    continue
                data = tr.get("data", {}) if isinstance(tr.get("data"), dict) else {}
                for row in list(data.get("financial_facts", []) or []) + list(data.get("price_data", []) or []):
                    if isinstance(row, dict) and str(row.get("requirement_id", "")) == rid:
                        items.append(row)
        elif req_type == "calculation":
            items = [tr for tr in tool_results if tr.get("tool") == "compute_metrics" and str(tr.get("requirement_id", "")) == rid]
        elif req_type == "event":
            for tr in tool_results:
                if tr.get("tool") != "query_event_price_window":
                    continue
                data = tr.get("data", {}) if isinstance(tr.get("data"), dict) else {}
                for event in data.get("events", []) or []:
                    if isinstance(event, dict) and str(event.get("requirement_id", "")) == rid:
                        items.append(event)
        if len(items) >= min_results:
            status = "satisfied"
            failure_reason = None
        elif items:
            status = "partial"
            failure_reason = "below_min_results"
        else:
            status = "missing"
            failure_reason = "no_matching_evidence"
        results.append(
            collection_result(
                requirement_id=rid,
                status=status,
                evidence_type=req_type,
                items=items,
                failure_reason=failure_reason,
                retry_count=retry_counts.get(rid, 0),
            )
        )
    return results

def _structured_rows_for_ticker(tool_results: list[dict[str, Any]], ticker: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tr in tool_results:
        if tr.get("tool") != "query_financial_data":
            continue
        if str(tr.get("ticker", "")).upper() != str(ticker).upper():
            continue
        data = tr.get("data", {})
        out.extend(list(data.get("financial_facts", [])))
    return out

def _fact_structured_sufficient(
    tool_results: list[dict[str, Any]],
    ticker: str,
    requested_metrics: list[str],
) -> bool:
    rows = _structured_rows_for_ticker(tool_results, ticker)
    if not rows:
        return False
    metric_set = {str(m) for m in requested_metrics if str(m).strip()}
    if not metric_set:
        return True
    return any(str(r.get("metric", "")) in metric_set for r in rows)

def _target_periods_for_ticker(
    tool_results: list[dict[str, Any]],
    ticker: str,
    resolved_period_context: dict[str, Any],
) -> list[str]:
    common = resolved_period_context.get("common_periods")
    if isinstance(common, list) and common:
        return [str(x) for x in common if str(x).strip()][:4]

    rows = _structured_rows_for_ticker(tool_results, ticker)
    periods = sorted(
        {str(r.get("period_end", "")).strip() for r in rows if str(r.get("period_end", "")).strip()},
        reverse=True,
    )
    return periods[:4]

def _merge_execution_tools(selected: list[str], analysis_plan: dict[str, Any], state: AgentState) -> list[str]:
    """Apply validated plan hints without letting the raw LLM plan control tools."""
    if state.get("needs_tools") is False:
        return []
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    task_type = str(state.get("task_type", "fact_qa"))
    safety_intent = str(state.get("safety_intent", "normal"))
    event_intent = str(state.get("event_intent", "none"))
    selected = [tool for tool in selected if tool in ALLOWED_ANALYSIS_TOOLS]
    validated = [tool for tool in analysis_plan.get("validated_tools", []) if tool in ALLOWED_ANALYSIS_TOOLS]

    if answer_mode == "direct_fact":
        return list(dict.fromkeys(selected))

    merged = list(dict.fromkeys(selected + validated))
    if answer_mode == "cautious_outlook":
        merged.extend(["query_financial_data", "compute_metrics", "search_filings"])
    if answer_mode == "analytical":
        merged.append("search_filings")
    if task_type == "company_comparison" and safety_intent == "investment_advice_like":
        merged.extend(["query_financial_data", "compute_metrics", "search_filings"])
    if event_intent == "required":
        merged.append("query_event_price_window")
    elif "query_event_price_window" in merged:
        merged = [tool for tool in merged if tool != "query_event_price_window"]
    return list(dict.fromkeys(tool for tool in merged if tool in ALLOWED_ANALYSIS_TOOLS))

def _apply_plan_guided_retrieval_policy(
    retrieval_policy: dict[str, Any],
    analysis_plan: dict[str, Any],
    state: AgentState,
    selected_tools: list[str],
) -> dict[str, Any]:
    policy = dict(retrieval_policy)
    if "search_filings" not in selected_tools:
        return policy
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    task_type = str(state.get("task_type", "fact_qa"))
    safety_intent = str(state.get("safety_intent", "normal"))
    sections = [
        section
        for section in analysis_plan.get("section_preferences", [])
        if isinstance(section, str) and section in KNOWN_SEC_SECTIONS
    ]
    if answer_mode in {"cautious_outlook", "analytical"} or (
        task_type == "company_comparison" and safety_intent == "investment_advice_like"
    ):
        for section in ANALYTICAL_SECTION_PREFERENCES:
            if section not in sections:
                sections.append(section)
    if sections:
        policy["section_allowlist"] = list(dict.fromkeys(list(policy.get("section_allowlist") or []) + sections))
        policy["strict_sections"] = False
        policy["text_top_k"] = max(int(policy.get("text_top_k", 0) or 0), 4 if task_type == "company_comparison" else 5)
        policy["max_per_section"] = max(int(policy.get("max_per_section", 1) or 1), 2)
    return policy

def _should_use_requirement_executor(state: AgentState) -> bool:
    """Route analytical/conversational paths through requirement execution."""
    if state.get("needs_tools") is False:
        return False
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    if answer_mode == "direct_fact" and _is_profit_decline_query(str(state.get("user_query") or "")):
        return True
    if answer_mode == "direct_fact":
        return False
    if answer_mode in {"analytical", "cautious_outlook", "comparison_brief"}:
        return True
    if str(state.get("safety_intent", "")) == "investment_advice_like":
        return True
    return str(state.get("task_type", "")) in {"trend_analysis", "company_comparison", "report_summary"}


def _is_profit_decline_query(user_query: str) -> bool:
    query = str(user_query or "").lower()
    asks_why = "为什么" in query or "why" in query
    has_decline = any(
        term in query
        for term in (
            "利润下降",
            "利润下滑",
            "净利润下降",
            "净利润下滑",
            "盈利下降",
            "盈利下滑",
            "profit decline",
            "profit declined",
            "earnings decline",
            "earnings declined",
        )
    )
    return asks_why and has_decline

def _sync_requirement_executor_tools() -> None:
    """Preserve existing tests that monkeypatch tools via this module."""
    requirement_executor.query_financial_data = query_financial_data
    requirement_executor.search_filings = search_filings
    requirement_executor.compute_metrics = compute_metrics
    requirement_executor.query_event_price_window = query_event_price_window

def _trace_evidence_fields(
    evidence_plan: dict[str, Any],
    collection_results: list[dict[str, Any]],
    sufficiency: dict[str, Any],
    retry_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = summarize_evidence_requirements(evidence_plan, collection_results, sufficiency)
    synthesis_mode = str(summary.get("synthesis_mode") or "")
    dimension_contract = normalize_dimension_status_contract(
        dict(summary.get("dimension_status_by_id", summary.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(summary.get("satisfied_dimensions", summary.get("covered_dimensions", [])) or []),
        partial_dimensions=list(summary.get("partial_dimensions", []) or []),
        missing_dimensions=list(summary.get("missing_dimensions", []) or []),
        dimension_coverage_rate=summary.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=summary.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(summary.get("framework_sufficiency_status", "") or ""),
    )
    return {
        "evidence_plan_summary": summary,
        "evidence_requirements": list(evidence_plan.get("evidence_requirements", []) or []),
        "collected_evidence_by_requirement": summary.get("collected_evidence_by_requirement", {}),
        "requirement_status_map": build_requirement_status_map(evidence_plan, collection_results, sufficiency),
        "dimension_status_by_id": dimension_contract["dimension_status_by_id"],
        "dimension_status_map": dimension_contract["dimension_status_map"],
        "satisfied_dimensions": dimension_contract["satisfied_dimensions"],
        "covered_dimensions": dimension_contract["covered_dimensions"],
        "partial_dimensions": dimension_contract["partial_dimensions"],
        "missing_dimensions": dimension_contract["missing_dimensions"],
        "dimension_coverage_rate": dimension_contract["dimension_coverage_rate"],
        "weighted_dimension_coverage_rate": dimension_contract["weighted_dimension_coverage_rate"],
        "framework_sufficiency_status": dimension_contract["framework_sufficiency_status"],
        "evidence_sufficiency_summary": summary,
        "answer_part_status_by_id": dict(summary.get("answer_part_status_by_id", {}) or {}),
        "evidence_gap_by_answer_part": dict(summary.get("evidence_gap_by_answer_part", {}) or {}),
        "missing_required_answer_parts": list(summary.get("missing_required_answer_parts", []) or []),
        "partial_required_answer_parts": list(summary.get("partial_required_answer_parts", []) or []),
        "trace_summary": build_trace_summary(
            evidence_plan,
            collection_results,
            sufficiency,
            synthesis_mode=synthesis_mode,
        ),
        "missing_requirements": list(summary.get("missing_requirements", []) or []),
        "requirement_limitations": list(summary.get("requirement_limitations", []) or []),
        "retry_history": list(retry_history or []),
        "degradation_reason": summary.get("degradation_reason"),
        "synthesis_mode": synthesis_mode,
    }

def execute_agent_tools(state: AgentState) -> dict[str, Any]:
    """Invoke the selected tools for each company, collect results."""
    plan = _resolve_query_plan(state)
    evidence_plan, evidence_requirements = _plan_and_requirements(state)
    period_query = dict(plan.get("period_query", state.get("period_query", _default_period_query())))
    resolved_period_context = dict(plan.get("resolved_period_context", state.get("resolved_period_context", {})))
    comparison_basis_label = str(plan.get("comparison_basis_label", state.get("comparison_basis_label", "")))
    if state.get("needs_tools") is False:
        retrieval_policy = dict(state.get("retrieval_policy", {}))
        sufficiency = evaluate_evidence_sufficiency(evidence_plan, []).model_dump(exclude_none=True)
        trace_fields = _trace_evidence_fields(evidence_plan, [], sufficiency, [])
        retrieval_debug = {
            "policy": retrieval_policy,
            "event_calls": [],
            "search_calls": [],
            "search_skipped": [{"reason": "needs_tools_false"}],
        }
        return {
            "tool_results": [],
            "retrieved_docs": [],
            "evidence_loop_count": MAX_EVIDENCE_LOOPS,
            "period_query": period_query,
            "resolved_period_context": resolved_period_context,
            "comparison_basis_label": comparison_basis_label,
            "retrieval_policy": retrieval_policy,
            "retrieval_debug": retrieval_debug,
            "event_intent": "none",
            "market_reaction_requested": False,
            "event_query": state.get("event_query", {}),
            "event_results": list(state.get("event_results", [])),
            "market_reaction_evidence": [],
            "market_reaction_limitations": list(state.get("market_reaction_limitations", [])),
            "selected_tools": [],
            "validated_tools": [],
            "evidence_plan": evidence_plan,
            "evidence_collection_results": [],
            "evidence_sufficiency": sufficiency,
            "evidence_retry_history": [],
            "requirement_limitations": list(sufficiency.get("requirement_limitations", []) or []),
            "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
            "why_tools_skipped": [{"reason": "needs_tools_false", "message": "tools_skipped_by_answer_mode"}],
            **trace_fields,
        }

    companies = state.get("companies", [])
    comparison_target = state.get("comparison_target")
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    selected = _merge_execution_tools(list(state.get("selected_tools", [])), analysis_plan, state)
    for tool in _tools_from_requirements(evidence_requirements, state):
        if tool not in selected:
            selected.append(tool)
    time_range = state.get("time_range")
    metrics = list(state.get("requested_metrics", []))
    plan_metrics = [
        metric
        for metric in analysis_plan.get("metric_requirements", [])
        if isinstance(metric, str) and metric in ALLOWED_ANALYSIS_METRICS
    ]
    if plan_metrics:
        metrics = list(dict.fromkeys(metrics + plan_metrics))
    requirement_metrics = _metrics_from_requirements(evidence_requirements)
    if requirement_metrics:
        metrics = list(dict.fromkeys(metrics + requirement_metrics))
    task_type = state.get("task_type", "fact_qa")
    user_query = state.get("user_query", "")
    event_intent = str(state.get("event_intent", _detect_event_intent(user_query, task_type=str(task_type)))).lower()
    if event_intent not in EVENT_INTENT_TYPES:
        event_intent = "none"
    market_reaction_requested = bool(state.get("market_reaction_requested", event_intent == "required"))
    if event_intent != "required":
        selected = [tool for tool in selected if tool != "query_event_price_window"]
    elif "query_event_price_window" not in selected:
        selected = list(selected) + ["query_event_price_window"]
    event_query = dict(state.get("event_query", {}))
    if not event_query:
        event_query = _build_event_query(
            user_query=user_query,
            task_type=str(task_type),
            period_query=period_query,
        )
    retrieval_policy = dict(state.get("retrieval_policy", {}))
    if not retrieval_policy:
        retrieval_policy = _build_retrieval_policy(state, selected_tools=selected)
    retrieval_policy = _apply_plan_guided_retrieval_policy(retrieval_policy, analysis_plan, state, selected)
    retrieval_debug: dict[str, Any] = {
        "policy": retrieval_policy,
        "validated_tools": list(analysis_plan.get("validated_tools", selected)),
        "selected_tools": selected,
        "event_calls": [],
        "search_calls": [],
        "requirement_calls": [],
        "search_skipped": [],
    }
    event_results: list[dict[str, Any]] = list(state.get("event_results", []))
    market_reaction_limitations: list[str] = list(state.get("market_reaction_limitations", []))
    requirement_retry_counts: dict[str, int] = {}

    tickers = list(companies)
    if comparison_target and comparison_target not in tickers:
        tickers.append(comparison_target)

    if not tickers:
        logger.warning("execute_tools: no tickers — skipping all tool calls")
        sufficiency = evaluate_evidence_sufficiency(evidence_plan, []).model_dump(exclude_none=True)
        trace_fields = _trace_evidence_fields(evidence_plan, [], sufficiency, [])
        return {
            "tool_results": [{"tool": "_none", "error": "No company/ticker identified in the query"}],
            "retrieved_docs": [],
            "evidence_loop_count": MAX_EVIDENCE_LOOPS,  # skip retry
            "period_query": period_query,
            "resolved_period_context": resolved_period_context,
            "comparison_basis_label": comparison_basis_label,
            "retrieval_policy": retrieval_policy,
            "retrieval_debug": retrieval_debug,
            "event_intent": event_intent,
            "market_reaction_requested": market_reaction_requested,
            "event_query": event_query,
            "event_results": event_results,
            "market_reaction_evidence": [],
            "market_reaction_limitations": ["no_ticker_identified_for_event_query"],
            "selected_tools": selected,
            "validated_tools": list(analysis_plan.get("validated_tools", selected)),
            "evidence_plan": evidence_plan,
            "evidence_collection_results": [],
            "evidence_sufficiency": sufficiency,
            "evidence_retry_history": [],
            "requirement_limitations": list(sufficiency.get("requirement_limitations", []) or []),
            "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
            "why_tools_skipped": [{"reason": "no_ticker_identified", "message": "tools_skipped_without_company"}],
            **trace_fields,
        }

    if resolved_period_context.get("needs_clarification"):
        sufficiency = evaluate_evidence_sufficiency(evidence_plan, []).model_dump(exclude_none=True)
        trace_fields = _trace_evidence_fields(evidence_plan, [], sufficiency, [])
        return {
            "tool_results": [
                {
                    "tool": "_clarification",
                    "error": "period_clarification_needed",
                    "reason": resolved_period_context.get("clarification_reason", ""),
                }
            ],
            "retrieved_docs": [],
            "evidence_loop_count": MAX_EVIDENCE_LOOPS,
            "period_query": period_query,
            "resolved_period_context": resolved_period_context,
            "comparison_basis_label": comparison_basis_label,
            "retrieval_policy": retrieval_policy,
            "retrieval_debug": retrieval_debug,
            "event_intent": event_intent,
            "market_reaction_requested": market_reaction_requested,
            "event_query": event_query,
            "event_results": event_results,
            "market_reaction_evidence": [],
            "market_reaction_limitations": market_reaction_limitations,
            "selected_tools": selected,
            "validated_tools": list(analysis_plan.get("validated_tools", selected)),
            "evidence_plan": evidence_plan,
            "evidence_collection_results": [],
            "evidence_sufficiency": sufficiency,
            "evidence_retry_history": [],
            "requirement_limitations": list(sufficiency.get("requirement_limitations", []) or []),
            "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
            "why_tools_skipped": [{"reason": "period_clarification_needed", "message": "tools_skipped_until_period_clarified"}],
            **trace_fields,
        }

    date_start = time_range.get("start") if time_range else None
    date_end = time_range.get("end") if time_range else None
    section_allowlist = retrieval_policy.get("section_allowlist")
    strict_sections = bool(retrieval_policy.get("strict_sections", False))

    if _should_use_requirement_executor(state):
        _sync_requirement_executor_tools()
        req_out = requirement_executor.execute_evidence_requirements(
            state,
            evidence_plan,
            {
                "period_query": period_query,
                "resolved_period_context": resolved_period_context,
                "comparison_basis_label": comparison_basis_label,
                "retrieval_policy": retrieval_policy,
                "event_query": event_query,
                "event_intent": event_intent,
                "market_reaction_requested": market_reaction_requested,
                "date_start": date_start,
                "date_end": date_end,
                "tickers": tickers,
                "task_type": task_type,
                "user_query": user_query,
            },
        )
        req_debug = dict(req_out.get("retrieval_debug", {}) or {})
        req_debug["policy"] = retrieval_policy
        req_debug.setdefault("selected_tools", list(req_out.get("selected_tools", [])))
        req_debug.setdefault("validated_tools", list(req_out.get("validated_tools", [])))
        req_collection = list(req_out.get("evidence_collection_results", []))
        req_sufficiency = dict(req_out.get("evidence_sufficiency", {}) or {})
        req_retry_history = list(req_out.get("retry_history", []))
        req_trace_fields = _trace_evidence_fields(evidence_plan, req_collection, req_sufficiency, req_retry_history)
        return {
            "tool_results": list(req_out.get("tool_results", [])),
            "retrieved_docs": list(req_out.get("retrieved_docs", [])),
            "evidence_loop_count": state.get("evidence_loop_count", 0) + 1,
            "period_query": period_query,
            "resolved_period_context": resolved_period_context,
            "comparison_basis_label": comparison_basis_label,
            "retrieval_policy": retrieval_policy,
            "retrieval_debug": req_debug,
            "event_intent": event_intent,
            "market_reaction_requested": market_reaction_requested,
            "event_query": event_query,
            "event_results": list(req_out.get("event_results", [])),
            "market_reaction_evidence": _collect_event_rows(list(req_out.get("tool_results", []))),
            "market_reaction_limitations": list(req_out.get("market_reaction_limitations", [])),
            "selected_tools": list(req_out.get("selected_tools", [])),
            "validated_tools": list(req_out.get("validated_tools", [])),
            "evidence_plan": evidence_plan,
            "evidence_collection_results": req_collection,
            "evidence_sufficiency": req_sufficiency,
            "evidence_retry_history": req_retry_history,
            "requirement_limitations": list(req_sufficiency.get("requirement_limitations", []) or []),
            "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
            "why_tools_skipped": list(req_out.get("why_tools_skipped", [])),
            **req_trace_fields,
        }

    all_tool_results: list[dict[str, Any]] = list(state.get("tool_results", []))
    all_docs: list[dict[str, Any]] = list(state.get("retrieved_docs", []))

    # Phase 0: event-window lookup first for market-reaction intent.
    if "query_event_price_window" in selected:
        _progress(
            state,
            "tool_started",
            "started",
            "正在读取事件窗口市场数据。",
            metadata={"tool": "query_event_price_window", "ticker_count": len(tickers)},
        )
        event_returned = 0
        for ticker in tickers:
            payload = {
                "ticker": ticker,
                "event_type": event_query.get("event_type", "any"),
                "fiscal_period": event_query.get("fiscal_period"),
                "event_date": event_query.get("event_date"),
                "latest_n": event_query.get("latest_n", 4),
                "window_days": event_query.get("window_days", [1, 5, 10]),
                "sort_by": event_query.get("sort_by", "event_date"),
                "sort_order": event_query.get("sort_order", "desc"),
            }
            try:
                result = query_event_price_window.invoke(payload)
                result = _tag_event_result(result, ticker, evidence_requirements)
                events = list(result.get("events", [])) if isinstance(result, dict) else []
                event_returned += len(events)
                event_results.append({"ticker": ticker, "data": result})
                all_tool_results.append(
                    {
                        "tool": "query_event_price_window",
                        "ticker": ticker,
                        "requirement_id": str(result.get("requirement_id", "")) if isinstance(result, dict) else "",
                        "count": len(events),
                        "data": result,
                    }
                )
                retrieval_debug["event_calls"].append(
                    {
                        "ticker": ticker,
                        "requested_windows": payload.get("window_days", []),
                        "latest_n": payload.get("latest_n"),
                        "returned": len(events),
                    }
                )
                if market_reaction_requested and not events:
                    market_reaction_limitations.append(f"no_event_window_data_for_{ticker}")
            except Exception as exc:
                logger.warning("query_event_price_window failed for %s: %s", ticker, exc)
                all_tool_results.append(
                    {
                        "tool": "query_event_price_window",
                        "ticker": ticker,
                        "error": str(exc),
                    }
                )
                retrieval_debug["event_calls"].append(
                    {
                        "ticker": ticker,
                        "requested_windows": payload.get("window_days", []),
                        "latest_n": payload.get("latest_n"),
                        "returned": 0,
                        "error": str(exc),
                    }
                )
                market_reaction_limitations.append(f"event_query_failed_for_{ticker}")
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成事件窗口市场数据读取，返回 {event_returned} 条证据。",
            metadata={"tool": "query_event_price_window", "status": "passed", "returned": event_returned},
        )

    # Phase 1: structured queries (for period alignment and metric context).
    if "query_financial_data" in selected:
        _progress(
            state,
            "tool_started",
            "started",
            "正在读取结构化财务数据。",
            metadata={"tool": "query_financial_data", "ticker_count": len(tickers), "metrics": metrics},
        )
    financial_returned = 0
    for ticker in tickers:
        if "query_financial_data" not in selected:
            continue
        query_metrics = metrics if metrics else ["revenue", "net_income"]
        period_type = str(period_query.get("period_type") or _infer_period_type(state) or "")
        if period_type not in PERIOD_TYPES:
            period_type = _infer_period_type(state) or "latest"
        year_basis = str(period_query.get("year_basis") or "fiscal")
        target_period_type = str(resolved_period_context.get("target_period_type") or "")
        requires_fiscal_alignment = (
            year_basis == "fiscal"
            and (
                (period_type == "quarterly" and (period_query.get("year") is not None or period_query.get("quarter") is not None))
                or (period_type == "annual" and period_query.get("year") is not None)
                or (
                    period_type in {"latest", "trailing"}
                    and target_period_type == "quarterly"
                    and (period_query.get("year") is not None or period_query.get("quarter") is not None)
                )
                or (
                    period_type in {"latest", "trailing"}
                    and target_period_type == "annual"
                    and period_query.get("year") is not None
                )
            )
        )
        try:
            result = query_financial_data.invoke({
                "ticker": ticker,
                "metrics": query_metrics,
                "period_type": period_type,
                "target_period_type": resolved_period_context.get("target_period_type"),
                "year": period_query.get("year"),
                "quarter": period_query.get("quarter"),
                "trailing_n": period_query.get("trailing_n"),
                "year_basis": period_query.get("year_basis"),
                "comparison_basis": period_query.get("comparison_basis"),
                "strict_period_match": bool(resolved_period_context.get("strict_period_match", True)),
                "date_start": date_start,
                "date_end": date_end,
                "limit": 20,
            })

            # Guardrail: for recency questions without explicit year,
            # if strict date filter yields no data, retry unfiltered.
            if (
                (date_start or date_end)
                and _is_recency_query(user_query)
                and not _has_explicit_year(user_query)
                and not result.get("financial_facts")
                and not result.get("price_data")
            ):
                relaxed = query_financial_data.invoke({
                    "ticker": ticker,
                    "metrics": query_metrics,
                    "period_type": period_type,
                    "target_period_type": resolved_period_context.get("target_period_type"),
                    "year": period_query.get("year"),
                    "quarter": period_query.get("quarter"),
                    "trailing_n": period_query.get("trailing_n"),
                    "year_basis": period_query.get("year_basis"),
                    "comparison_basis": period_query.get("comparison_basis"),
                    "strict_period_match": bool(resolved_period_context.get("strict_period_match", True)),
                    "date_start": None,
                    "date_end": None,
                    "limit": 20,
                })
                if relaxed.get("financial_facts") or relaxed.get("price_data"):
                    logger.info(
                        "query_financial_data retry succeeded without stale date filter for %s",
                        ticker,
                    )
                    result = relaxed
            result = _tag_financial_result(result, ticker, evidence_requirements)
            if isinstance(result, dict):
                financial_returned += len(result.get("financial_facts", []) or []) + len(result.get("price_data", []) or [])

            period_context = result.get("period_context", {}) if isinstance(result, dict) else {}
            if requires_fiscal_alignment and period_context.get("fiscal_year_end_month") is None:
                resolved_period_context["needs_clarification"] = True
                resolved_period_context["clarification_reason"] = f"fiscal_year_end_unknown_for_{ticker}"
                clarification_collection = _build_collection_results(
                    evidence_plan,
                    all_tool_results,
                    all_docs,
                    requirement_retry_counts,
                )
                clarification_sufficiency = evaluate_evidence_sufficiency(
                    evidence_plan,
                    clarification_collection,
                ).model_dump(exclude_none=True)
                clarification_trace_fields = _trace_evidence_fields(
                    evidence_plan,
                    clarification_collection,
                    clarification_sufficiency,
                    [],
                )
                return {
                    "tool_results": all_tool_results
                    + [
                        {
                            "tool": "_clarification",
                            "ticker": ticker,
                            "error": "period_clarification_needed",
                            "reason": f"fiscal_year_end_unknown_for_{ticker}",
                        }
                    ],
                    "retrieved_docs": all_docs,
                    "evidence_loop_count": MAX_EVIDENCE_LOOPS,
                    "period_query": period_query,
                    "resolved_period_context": resolved_period_context,
                    "comparison_basis_label": comparison_basis_label,
                    "retrieval_policy": retrieval_policy,
                    "retrieval_debug": retrieval_debug,
                    "event_intent": event_intent,
                    "market_reaction_requested": market_reaction_requested,
                    "event_query": event_query,
                    "event_results": event_results,
                    "market_reaction_evidence": [],
                    "market_reaction_limitations": market_reaction_limitations + [f"fiscal_alignment_unknown_for_{ticker}"],
                    "selected_tools": selected,
                    "validated_tools": list(analysis_plan.get("validated_tools", selected)),
                    "evidence_plan": evidence_plan,
                    "evidence_collection_results": clarification_collection,
                    "evidence_sufficiency": clarification_sufficiency,
                    "evidence_retry_history": [],
                    "requirement_limitations": list(clarification_sufficiency.get("requirement_limitations", []) or []),
                    "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
                    "why_tools_skipped": [
                        {"reason": "fiscal_alignment_unknown", "message": f"tools_stopped_for_{ticker}"}
                    ],
                    **clarification_trace_fields,
                }

            all_tool_results.append({
                "tool": "query_financial_data",
                "ticker": ticker,
                "data": result,
            })
        except Exception as exc:
            logger.warning("query_financial_data failed for %s: %s", ticker, exc)
            all_tool_results.append({
                "tool": "query_financial_data",
                "ticker": ticker,
                "error": str(exc),
            })
    if "query_financial_data" in selected:
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成结构化财务数据读取，返回 {financial_returned} 条证据。",
            metadata={"tool": "query_financial_data", "status": "passed", "returned": financial_returned},
        )

    # Strict same-period filter before compute/text retrieval.
    if (
        task_type == "company_comparison"
        and "query_financial_data" in selected
        and period_query.get("comparison_basis") == "same_period"
    ):
        year_basis = str(period_query.get("year_basis", "fiscal"))
        target_period_type = str(resolved_period_context.get("target_period_type", "quarterly"))
        results_by_ticker: dict[str, list[dict[str, Any]]] = {}
        for tr in all_tool_results:
            if tr.get("tool") != "query_financial_data" or "data" not in tr:
                continue
            ticker = str(tr.get("ticker", ""))
            facts = list(tr.get("data", {}).get("financial_facts", []))
            results_by_ticker[ticker] = facts

        def _period_token(row: dict[str, Any]) -> str:
            if target_period_type == "quarterly":
                return str(row.get("period_end", ""))
            if year_basis == "calendar":
                return str(row.get("calendar_year", ""))
            return str(row.get("fiscal_year", ""))

        if len(results_by_ticker) >= 2:
            common: set[str] | None = None
            for facts in results_by_ticker.values():
                tokens = {_period_token(r) for r in facts if _period_token(r)}
                common = tokens if common is None else (common & tokens)
            common = common or set()

            if not common:
                resolved_period_context["same_period_match"] = False
                resolved_period_context["common_periods"] = []
            else:
                ordered_common = sorted(common, reverse=True)
                mode = str(period_query.get("period_type", "latest"))
                if mode == "latest":
                    selected_common = ordered_common[:1]
                elif mode == "trailing":
                    n = int(period_query.get("trailing_n") or 4)
                    selected_common = ordered_common[: max(1, n)]
                else:
                    selected_common = ordered_common
                selected_set = set(selected_common)

                for tr in all_tool_results:
                    if tr.get("tool") != "query_financial_data" or "data" not in tr:
                        continue
                    facts = list(tr.get("data", {}).get("financial_facts", []))
                    tr["data"]["financial_facts"] = [r for r in facts if _period_token(r) in selected_set]
                resolved_period_context["same_period_match"] = True
                resolved_period_context["common_periods"] = selected_common

    # Auto-compute if compute_metrics selected and we have structured data.
    if "compute_metrics" in selected:
        before_compute = len(all_tool_results)
        _progress(
            state,
            "tool_started",
            "started",
            "正在计算派生财务指标。",
            metadata={"tool": "compute_metrics"},
        )
        _auto_compute(state, all_tool_results, task_type)
        computed_count = len([item for item in all_tool_results[before_compute:] if item.get("tool") == "compute_metrics"])
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成派生指标计算，返回 {computed_count} 条计算证据。",
            metadata={"tool": "compute_metrics", "status": "passed", "returned": computed_count},
        )

    # Phase 2: requirement-aware text retrieval, falling back to legacy task-aware retrieval.
    text_requirements = _requirements_of_type(evidence_requirements, "text")
    if "search_filings" in selected and int(retrieval_policy.get("text_top_k", 0)) > 0 and text_requirements:
        for req in text_requirements:
            req_id = str(req.get("requirement_id", ""))
            req_ticker = str(req.get("company") or "")
            req_tickers = [req_ticker] if req_ticker else tickers
            req_sections = [s for s in req.get("section_preferences", []) if isinstance(s, str) and s in KNOWN_SEC_SECTIONS]
            req_query = str(req.get("retrieval_query") or user_query)
            for ticker in req_tickers:
                per_ticker_top_k = max(
                    int(req.get("min_results", 1) or 1),
                    int(retrieval_policy.get("text_top_k", settings.retrieval_top_k)),
                )
                if task_type == "company_comparison":
                    per_ticker_top_k = min(
                        per_ticker_top_k,
                        int(retrieval_policy.get("comparison_text_cap_per_company", 2)),
                    )
                payload: dict[str, Any] = {
                    "ticker": ticker,
                    "query": req_query,
                    "top_k": max(1, per_ticker_top_k),
                    "date_start": date_start,
                    "date_end": date_end,
                    "retrieval_profile": retrieval_policy.get("retrieval_profile", "default"),
                    "max_per_filing": retrieval_policy.get("max_per_filing"),
                    "max_per_section": retrieval_policy.get("max_per_section"),
                }
                if req_sections:
                    payload["section_allowlist"] = req_sections
                    payload["strict_sections"] = False
                target_periods = _target_periods_for_ticker(
                    all_tool_results,
                    ticker=ticker,
                    resolved_period_context=resolved_period_context,
                )
                if target_periods:
                    payload["target_periods"] = target_periods
                try:
                    _progress(
                        state,
                        "tool_started",
                        "started",
                        f"正在检索 {ticker} 的 SEC filing 文本证据。",
                        metadata={"tool": "search_filings", "requirement_id": req_id, "company": ticker},
                    )
                    docs_list, fallback_after_timeout, timeout_message = _search_filings_payload(
                        payload,
                        state,
                        ticker=ticker,
                        requirement_id=req_id,
                    )
                    if not docs_list and req.get("fallback_strategy"):
                        fallback_payload = dict(payload)
                        if "relax_sections" in req.get("fallback_strategy", []):
                            fallback_payload.pop("section_allowlist", None)
                            fallback_payload["strict_sections"] = False
                        if "fallback_user_query" in req.get("fallback_strategy", []):
                            fallback_payload["query"] = user_query
                        docs_list, fallback_after_timeout, timeout_message = _search_filings_payload(
                            fallback_payload,
                            state,
                            ticker=ticker,
                            requirement_id=req_id,
                        )
                        requirement_retry_counts[req_id] = requirement_retry_counts.get(req_id, 0) + 1
                    for doc in docs_list:
                        if isinstance(doc, dict):
                            doc["requirement_id"] = req_id
                    all_docs.extend(docs_list)
                    all_tool_results.append(
                        {
                            "tool": "search_filings",
                            "ticker": ticker,
                            "requirement_id": req_id,
                            "count": len(docs_list),
                        }
                    )
                    call_debug = {
                        "requirement_id": req_id,
                        "ticker": ticker,
                        "top_k": payload["top_k"],
                        "returned": len(docs_list),
                        "profile": payload.get("retrieval_profile", "default"),
                        "target_periods": payload.get("target_periods", []),
                        "section_allowlist": payload.get("section_allowlist"),
                        "retry_count": requirement_retry_counts.get(req_id, 0),
                    }
                    if fallback_after_timeout:
                        call_debug["fallback_after_timeout"] = True
                        call_debug["timeout_error"] = timeout_message
                    retrieval_debug["search_calls"].append(call_debug)
                    retrieval_debug["requirement_calls"].append(call_debug)
                    _progress(
                        state,
                        "tool_finished",
                        "completed" if docs_list else "warning",
                        f"已完成 SEC filing 检索，返回 {len(docs_list)} 条候选证据。",
                        metadata={
                            "tool": "search_filings",
                            "requirement_id": req_id,
                            "company": ticker,
                            "status": "passed" if docs_list else "missing",
                            "returned": len(docs_list),
                            "fallback_after_timeout": fallback_after_timeout,
                        },
                    )
                except Exception as exc:
                    logger.warning("search_filings failed for %s requirement %s: %s", ticker, req_id, exc)
                    all_tool_results.append(
                        {
                            "tool": "search_filings",
                            "ticker": ticker,
                            "requirement_id": req_id,
                            "error": str(exc),
                        }
                    )
                    retrieval_debug["requirement_calls"].append(
                        {
                            "requirement_id": req_id,
                            "ticker": ticker,
                            "top_k": payload["top_k"],
                            "returned": 0,
                            "error": str(exc),
                            "profile": payload.get("retrieval_profile", "default"),
                        }
                    )
                    _progress(
                        state,
                        "tool_finished",
                        "failed",
                        "SEC filing 检索失败。",
                        metadata={"tool": "search_filings", "requirement_id": req_id, "company": ticker, "error": str(exc)[:300]},
                    )
    elif "search_filings" in selected and int(retrieval_policy.get("text_top_k", 0)) > 0:
        for ticker in tickers:
            if (
                retrieval_policy.get("skip_fact_text_when_structured_sufficient")
                and task_type == "fact_qa"
                and _fact_structured_sufficient(all_tool_results, ticker=ticker, requested_metrics=metrics)
            ):
                retrieval_debug["search_skipped"].append(
                    {
                        "ticker": ticker,
                        "reason": "structured_evidence_sufficient_for_fact_qa",
                    }
                )
                continue

            per_ticker_top_k = int(retrieval_policy.get("text_top_k", settings.retrieval_top_k))
            if task_type == "company_comparison":
                per_ticker_top_k = min(
                    per_ticker_top_k,
                    int(retrieval_policy.get("comparison_text_cap_per_company", 2)),
                )
            payload: dict[str, Any] = {
                "ticker": ticker,
                "query": state["user_query"],
                "top_k": max(1, per_ticker_top_k),
                "date_start": date_start,
                "date_end": date_end,
                "retrieval_profile": retrieval_policy.get("retrieval_profile", "default"),
                "max_per_filing": retrieval_policy.get("max_per_filing"),
                "max_per_section": retrieval_policy.get("max_per_section"),
            }
            if section_allowlist:
                payload["section_allowlist"] = section_allowlist
                payload["strict_sections"] = bool(strict_sections)
            target_periods = _target_periods_for_ticker(
                all_tool_results,
                ticker=ticker,
                resolved_period_context=resolved_period_context,
            )
            if target_periods:
                payload["target_periods"] = target_periods
            try:
                _progress(
                    state,
                    "tool_started",
                    "started",
                    f"正在检索 {ticker} 的 SEC filing 文本证据。",
                    metadata={"tool": "search_filings", "company": ticker},
                )
                docs, fallback_after_timeout, timeout_message = _search_filings_payload(
                    payload,
                    state,
                    ticker=ticker,
                )
                doc_count = len(docs)
                all_docs.extend(docs)
                all_tool_results.append(
                    {
                        "tool": "search_filings",
                        "ticker": ticker,
                        "count": doc_count,
                    }
                )
                retrieval_debug["search_calls"].append(
                    {
                        "ticker": ticker,
                        "top_k": payload["top_k"],
                        "returned": doc_count,
                        "profile": payload.get("retrieval_profile", "default"),
                        "target_periods": payload.get("target_periods", []),
                        "section_allowlist": payload.get("section_allowlist"),
                        "fallback_after_timeout": True if fallback_after_timeout else None,
                        "timeout_error": timeout_message if fallback_after_timeout else None,
                    }
                )
                _progress(
                    state,
                    "tool_finished",
                    "completed" if doc_count else "warning",
                    f"已完成 SEC filing 检索，返回 {doc_count} 条候选证据。",
                    metadata={
                        "tool": "search_filings",
                        "company": ticker,
                        "status": "passed" if doc_count else "missing",
                        "returned": doc_count,
                        "fallback_after_timeout": fallback_after_timeout,
                    },
                )
            except Exception as exc:
                logger.warning("search_filings failed for %s: %s", ticker, exc)
                all_tool_results.append(
                    {
                        "tool": "search_filings",
                        "ticker": ticker,
                        "error": str(exc),
                    }
                )
                retrieval_debug["search_calls"].append(
                    {
                        "ticker": ticker,
                        "top_k": payload["top_k"],
                        "returned": 0,
                        "error": str(exc),
                        "profile": payload.get("retrieval_profile", "default"),
                    }
                )
                _progress(
                    state,
                    "tool_finished",
                    "failed",
                    "SEC filing 检索失败。",
                    metadata={"tool": "search_filings", "company": ticker, "error": str(exc)[:300]},
                )
    elif "search_filings" in selected:
        retrieval_debug["search_skipped"].append(
            {
                "reason": "policy_text_top_k_zero",
            }
        )

    evidence_collection_results = _build_collection_results(
        evidence_plan,
        all_tool_results,
        all_docs,
        requirement_retry_counts,
    )
    evidence_sufficiency = evaluate_evidence_sufficiency(
        evidence_plan,
        evidence_collection_results,
    ).model_dump(exclude_none=True)
    retry_history = [
        item
        for item in retrieval_debug.get("requirement_retry_history", [])
        if isinstance(item, dict)
    ]
    trace_evidence_fields = _trace_evidence_fields(evidence_plan, evidence_collection_results, evidence_sufficiency, retry_history)

    return {
        "tool_results": all_tool_results,
        "retrieved_docs": all_docs,
        "evidence_loop_count": state.get("evidence_loop_count", 0) + 1,
        "period_query": period_query,
        "resolved_period_context": resolved_period_context,
        "comparison_basis_label": comparison_basis_label,
        "retrieval_policy": retrieval_policy,
        "retrieval_debug": retrieval_debug,
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction_requested,
        "event_query": event_query,
        "event_results": event_results,
        "market_reaction_evidence": _collect_event_rows(all_tool_results),
        "market_reaction_limitations": list(dict.fromkeys(market_reaction_limitations)),
        "selected_tools": selected,
        "validated_tools": list(analysis_plan.get("validated_tools", selected)),
        "evidence_plan": evidence_plan,
        "evidence_collection_results": evidence_collection_results,
        "evidence_sufficiency": evidence_sufficiency,
        "evidence_retry_history": retry_history,
        "requirement_limitations": list(evidence_sufficiency.get("requirement_limitations", []) or []),
        "rejected_requirements": list(evidence_plan.get("rejected_requirements", [])),
        "why_tools_skipped": list(retrieval_debug.get("search_skipped", [])),
        **trace_evidence_fields,
    }

def _auto_compute(
    state: AgentState,
    tool_results: list[dict[str, Any]],
    task_type: str,
) -> None:
    """If we have financial_facts from query_financial_data, run compute_metrics."""
    evidence_plan, evidence_requirements = _plan_and_requirements(state)
    _ = evidence_plan
    computation = "growth"
    if task_type == "trend_analysis":
        computation = "qoq"
    elif task_type == "company_comparison":
        computation = "growth"

    comparable_index: dict[tuple[str, str], set[str]] = {}
    if task_type == "company_comparison":
        all_rows = _collect_financial_rows(tool_results)
        tickers = _ordered_unique_tickers(state, all_rows)
        if len(tickers) >= 2:
            t1, t2 = tickers[0], tickers[1]
            metrics = {str(r.get("metric", "")) for r in all_rows if r.get("metric")}
            for metric_name in metrics:
                # Annual: comparable by same calendar year.
                annual_1 = _rows_for(all_rows, t1, metric_name, "annual")
                annual_2 = _rows_for(all_rows, t2, metric_name, "annual")
                years_1 = {str(_period_year(r.get("period_end"))) for r in annual_1 if _period_year(r.get("period_end"))}
                years_2 = {str(_period_year(r.get("period_end"))) for r in annual_2 if _period_year(r.get("period_end"))}
                common_years = years_1 & years_2
                if common_years:
                    comparable_index[(metric_name, "annual")] = common_years

                # Quarterly: comparable by same period_end.
                q_1 = _rows_for(all_rows, t1, metric_name, "quarterly")
                q_2 = _rows_for(all_rows, t2, metric_name, "quarterly")
                periods_1 = {str(r.get("period_end", "")) for r in q_1 if r.get("period_end")}
                periods_2 = {str(r.get("period_end", "")) for r in q_2 if r.get("period_end")}
                common_periods = periods_1 & periods_2
                if common_periods:
                    comparable_index[(metric_name, "quarterly")] = common_periods

    for tr in list(tool_results):
        if tr.get("tool") != "query_financial_data" or "data" not in tr:
            continue
        facts = tr["data"].get("financial_facts", [])
        if len(facts) < 2:
            continue

        # Group by metric
        by_metric: dict[str, list] = {}
        for f in facts:
            by_metric.setdefault(f["metric"], []).append(f)

        for metric_name, rows in by_metric.items():
            sorted_rows = sorted(rows, key=lambda r: r["period_end"])
            if task_type == "company_comparison":
                period_type = str(sorted_rows[0].get("period_type", "")) if sorted_rows else ""
                allowed_tokens = comparable_index.get((metric_name, period_type), set())
                if not allowed_tokens:
                    continue
                filtered_rows: list[dict[str, Any]] = []
                for r in sorted_rows:
                    if period_type == "annual":
                        token = str(_period_year(r.get("period_end")))
                    else:
                        token = str(r.get("period_end", ""))
                    if token in allowed_tokens:
                        filtered_rows.append(r)
                sorted_rows = filtered_rows

            data_points = [
                {"period": r["period_end"], "value": r["value"]}
                for r in sorted_rows
                if r["value"] is not None
            ]
            if len(data_points) < 2:
                continue
            try:
                result = compute_metrics.invoke({
                    "data": data_points,
                    "computation": computation,
                })
                requirement_id = _first_requirement_id(
                    evidence_requirements,
                    "calculation",
                    str(tr.get("ticker", "")),
                    str(metric_name),
                )
                tool_results.append({
                    "tool": "compute_metrics",
                    "ticker": tr.get("ticker", ""),
                    "metric": metric_name,
                    "computation": computation,
                    "requirement_id": requirement_id,
                    "data": result,
                })
            except Exception as exc:
                logger.warning("compute_metrics failed: %s", exc)


infer_period_type = _infer_period_type
TOOL_REGISTRY = {
    "search_filings": search_filings,
    "query_financial_data": query_financial_data,
    "query_event_price_window": query_event_price_window,
    "compute_metrics": compute_metrics,
}
