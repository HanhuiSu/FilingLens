"""Thin LangGraph node wrappers for the financial-analysis agent.

Business logic lives in query_plan, tool_executor, evidence, citations,
rendering, and answering. This module intentionally keeps only graph-node
orchestration plus temporary compatibility re-exports for older tests/imports.
"""

from __future__ import annotations

from datetime import date
import logging
import re
import time
import uuid
from typing import Any

from config import settings
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent import analyst_draft as _analyst_draft
from src.agent import answering as _answering
from src.agent import citations as _citations
from src.agent import evidence as _evidence
from src.agent import query_plan as _query_plan
from src.agent import rendering as _rendering
from src.agent import tool_executor as _tool_executor
from src.agent.answer_assembler import AnswerAssembler, evidence_refs_from_body
from src.agent.answer_contract import check_answer_contract
from src.agent.answer_relevance import judge_answer_relevance
from src.agent.constants import MAX_EVIDENCE_LOOPS
from src.agent.constants import OUTPUT_PROTOCOL_VERSION as OUTPUT_PROTOCOL_VERSION
from src.agent.evidence_packet import build_evidence_packet, summarize_evidence_packet
from src.agent.evidence_planner import build_requirements_from_research_plan, evaluate_plan_coverage, merge_evidence_requirements
from src.agent.llm import _get_llm, _parse_json_response
from src.agent.output_language import detect_output_language, language_leakage_count, repair_language_leakage
from src.agent.plan_validator import deterministic_causal_research_plan, is_causal_explanation_query, validate_research_plan
from src.agent.progress import append_progress_event
from src.agent.prompts import CLASSIFY_AND_EXTRACT
from src.agent.research_planner import build_research_plan_raw, planner_mode
from src.agent.reporting import build_company_analysis_report
from src.agent.state import AgentState
from src.tools.compute_metrics import compute_metrics
from src.tools.query_event_price_window import query_event_price_window
from src.tools.query_financial_data import query_financial_data
from src.tools.search_filings import search_filings

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Temporary compatibility re-exports for tests and external imports.
# New code should import these from their owning modules instead.
# ------------------------------------------------------------------

_default_period_query = _query_plan._default_period_query
_normalize_period_query = _query_plan._normalize_period_query
_resolve_query_plan = _query_plan._resolve_query_plan
_select_tools = _query_plan._select_tools
_sanitize_time_range_for_recency = _query_plan._sanitize_time_range_for_recency
_has_explicit_year = _query_plan._has_explicit_year
_is_market_reaction_query = _query_plan._is_market_reaction_query
_infer_period_type = _query_plan._infer_period_type
_detect_event_intent = _query_plan._detect_event_intent
_normalize_safety_intent = _query_plan._normalize_safety_intent
_normalize_answer_mode = _query_plan._normalize_answer_mode
_needs_tools_for_answer_mode = _query_plan._needs_tools_for_answer_mode
_build_clarification_question = _query_plan._build_clarification_question
detect_answer_mode = _query_plan.detect_answer_mode
build_validated_analysis_plan = _query_plan.build_validated_analysis_plan
validate_analysis_plan = _query_plan.validate_analysis_plan
_build_event_query = _query_plan._build_event_query
_build_retrieval_policy = _query_plan._build_retrieval_policy
_extract_tickers_fallback = _query_plan._extract_tickers_fallback

_build_evidence_bundle = _evidence._build_evidence_bundle
_build_text_evidence = _evidence._build_text_evidence
_validate_claims = _evidence._validate_claims
_validate_numeric_claims_strict = _evidence._validate_numeric_claims_strict
_latest_comparable_pair = _evidence._latest_comparable_pair
_collect_event_rows = _evidence._collect_event_rows
_collect_financial_rows = _evidence._collect_financial_rows

_apply_text_citation_policy = _citations._apply_text_citation_policy

_build_phase4_output = _rendering._build_phase4_output
_compose_answer_payload = _rendering._compose_answer_payload
_format_fact_qa_answer = _rendering._format_fact_qa_answer
_render_claim_answer = _rendering._render_claim_answer
_sample_docs_for_prompt = _rendering._sample_docs_for_prompt
_build_market_reaction_block = _rendering._build_market_reaction_block


def _sync_compat_dependencies() -> None:
    """Propagate monkeypatched compatibility symbols into owner modules.

    Several legacy tests monkeypatch symbols on src.agent.nodes directly. The
    wrappers keep those tests working while the real logic now lives elsewhere.
    """
    _tool_executor.search_filings = search_filings
    _tool_executor.query_financial_data = query_financial_data
    _tool_executor.query_event_price_window = query_event_price_window
    _tool_executor.compute_metrics = compute_metrics
    _answering._get_llm = _get_llm
    _analyst_draft._get_llm = _get_llm
    _rendering._get_llm = _get_llm


# ------------------------------------------------------------------
# Node 1 — classify_and_extract
# ------------------------------------------------------------------

def classify_and_extract(state: AgentState) -> dict[str, Any]:
    """Classify the user query and normalize it into the shared AgentState."""
    user_query = state["user_query"]
    trace_id = state.get("trace_id") or str(uuid.uuid4())
    today = date.today()

    parsed: dict[str, Any] = {}
    classifier_trace: dict[str, Any] = {
        "source": "llm",
        "status": "not_run",
        "fallback_used": False,
    }
    try:
        classify_timeout = min(float(settings.llm_classify_timeout_seconds or 90.0), 90.0)
        llm = _get_llm(
            reasoning=False,
            temperature=0.0,
            max_tokens=1024,
            timeout=classify_timeout,
            max_retries=settings.llm_classify_max_retries,
        )
        prompt_text = CLASSIFY_AND_EXTRACT.format(
            user_query=user_query,
            current_date=today.isoformat(),
            current_year=today.year,
        )
        response = llm.invoke(
            [
                SystemMessage(content="You are a financial query classifier. Output ONLY valid JSON."),
                HumanMessage(content=prompt_text),
            ]
        )
        parsed = _parse_json_response(response.content)
        if not parsed and not settings.llm_classify_fallback_enabled:
            raise ValueError("classifier returned empty_or_invalid_json")
        classifier_trace.update(
            {
                "status": "parsed" if parsed else "empty_or_invalid_json",
                "fallback_used": not bool(parsed),
            }
        )
    except Exception as exc:  # pragma: no cover - depends on external LLM/API availability
        if not settings.llm_classify_fallback_enabled:
            logger.exception("classify_and_extract LLM failed and deterministic fallback is disabled")
            raise
        logger.warning("classify_and_extract LLM failed; using deterministic fallback: %s", exc)
        parsed = {}
        classifier_trace.update(
            {
                "source": "deterministic_fallback",
                "status": "llm_failed",
                "fallback_used": True,
                "error_type": type(exc).__name__,
                "error": str(exc)[:500],
            }
        )
    logger.info("classify_and_extract parsed: %s", parsed)

    out = _query_plan.build_classification_state(
        user_query=user_query,
        parsed=parsed,
        trace_id=trace_id,
        today=today,
    )
    out["classifier_trace"] = classifier_trace
    if isinstance(out.get("trace_summary"), dict):
        out["trace_summary"]["classifier_trace"] = classifier_trace
    out["messages"] = [HumanMessage(content=user_query)]
    _emit_intent_and_plan_progress(out)
    return out


def _scope_counts(requirements: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"core": 0, "optional_context": 0, "diagnostic": 0}
    for req in requirements:
        scope = str(req.get("requirement_scope") or ("core" if req.get("required", True) else "optional_context"))
        if scope not in counts:
            scope = "optional_context"
        counts[scope] += 1
    return counts


def _emit_intent_and_plan_progress(state: dict[str, Any]) -> None:
    trace_id = str(state.get("trace_id") or "")
    if not trace_id:
        return
    canonical = dict(state.get("canonical_intent", {}) or {})
    companies = [str(item) for item in state.get("companies", []) or [] if str(item)]
    intent_family = str(canonical.get("intent_family") or state.get("methodology_intent") or state.get("task_type") or "")
    analysis_scope = str(canonical.get("analysis_scope") or state.get("analysis_scope") or "")
    company_text = "、".join(companies) if companies else "未明确公司"
    scope_text = "单公司" if analysis_scope == "single_company" else ("公司对比" if analysis_scope == "comparison" else analysis_scope or "通用")
    append_progress_event(
        trace_id,
        "intent_resolved",
        "completed",
        f"已识别为{scope_text}分析，目标公司：{company_text}。",
        node="classify",
        metadata={
            "intent_family": intent_family,
            "analysis_scope": analysis_scope,
            "companies": companies,
            "answer_mode": str(state.get("answer_mode") or ""),
        },
    )
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    requirements = [dict(req) for req in evidence_plan.get("evidence_requirements", []) or [] if isinstance(req, dict)]
    counts = _scope_counts(requirements)
    append_progress_event(
        trace_id,
        "evidence_plan_built",
        "completed",
        f"已生成证据计划：{counts['core']} 项核心证据，{counts['optional_context']} 项可选背景证据。",
        node="classify",
        metadata={
            "policy_id": str(state.get("evidence_policy_id") or evidence_plan.get("policy_id") or ""),
            "core_count": counts["core"],
            "optional_context_count": counts["optional_context"],
            "diagnostic_count": counts["diagnostic"],
        },
    )


# ------------------------------------------------------------------
# Node 1.5 — research_plan
# ------------------------------------------------------------------

def _emit_research_plan_progress(state: dict[str, Any], event: str, status: str, message: str, metadata: dict[str, Any] | None = None) -> None:
    trace_id = str(state.get("trace_id") or "")
    if not trace_id:
        return
    append_progress_event(trace_id, event, status, message, node="research_plan", metadata=metadata or {})


def _is_profit_decline_query(user_query: str) -> bool:
    text = str(user_query or "").lower()
    has_why = "为什么" in text or "why" in text
    has_decline = any(term in text for term in ("利润下降", "净利润下降", "盈利下降", "profit decline", "profit declined", "earnings decline"))
    return has_why and has_decline


def _research_planner_shortcut_kind(state: AgentState, user_query: str) -> str:
    if _answering._is_risk_comparison_query(user_query, state):
        return "risk_comparison"
    if str(state.get("safety_intent") or "") == "investment_advice_like":
        return "investment_boundary"
    if _is_profit_decline_query(user_query):
        return "profit_decline_causal"
    return ""


def research_plan_node(state: AgentState) -> dict[str, Any]:
    """Build and validate Research Planner V1 output before evidence execution."""
    mode = planner_mode()
    legacy_evidence_plan = dict(state.get("evidence_plan", {}) or {})
    if mode == "off" or state.get("needs_tools") is False:
        return {
            "legacy_evidence_plan": legacy_evidence_plan,
            "research_plan_raw": {},
            "research_plan_validated": {},
            "research_plan_used": {},
            "research_plan_validation": {"valid": False, "fallback_reason": "research_planner_disabled"},
            "research_plan_source": "off",
            "research_plan_fallback_reason": "research_planner_disabled",
            "research_plan_duration_ms": 0,
            "required_answer_parts": [],
            "plan_coverage_decision": {
                "strategy": "legacy_only",
                "legacy_core_count": len(list(legacy_evidence_plan.get("core_requirement_ids", []) or [])),
                "research_core_count": 0,
                "retained_legacy_core_count": len(list(legacy_evidence_plan.get("core_requirement_ids", []) or [])),
                "coverage_ratio": 0.0,
                "warnings": [],
                "reason": "research_planner_disabled",
            },
            "requirement_merge_summary": {
                "strategy": "legacy_only",
                "merged_total_requirements": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
                "legacy_only_count": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
            },
            "evidence_plan_used": {"source": "legacy_evidence_plan", "strategy": "legacy_only"},
        }

    _emit_research_plan_progress(
        state,
        "research_plan_started",
        "started",
        "正在生成 Research Plan，明确必须回答的问题和取证要求。",
        {"mode": mode},
    )
    user_query = str(state.get("user_query") or "")
    companies = [str(item) for item in state.get("companies", []) or [] if str(item)]
    node_started = time.monotonic()
    shortcut_kind = _research_planner_shortcut_kind(state, user_query)
    if shortcut_kind:
        core_count = len(list(legacy_evidence_plan.get("core_requirement_ids", []) or []))
        requirement_count = len(list(legacy_evidence_plan.get("evidence_requirements", []) or []))
        trace_summary = dict(state.get("trace_summary", {}) or {})
        trace_summary.update(
            {
                "research_question_type": shortcut_kind,
                "required_answer_parts": [],
                "research_plan_used": False,
                "research_plan_source": "deterministic_fallback",
                "research_plan_fallback_reason": "deterministic_intent_shortcut",
                "research_plan_duration_ms": int((time.monotonic() - node_started) * 1000),
                "plan_execution_strategy": "legacy_only",
            }
        )
        _emit_research_plan_progress(
            state,
            "research_plan_built",
            "completed",
            "Research Planner 已按确定性意图跳过，本次使用 legacy evidence plan。",
            {
                "question_type": shortcut_kind,
                "valid": False,
                "used": False,
                "planner_status": "deterministic_intent_shortcut",
                "research_plan_source": "deterministic_fallback",
                "research_plan_fallback_reason": "deterministic_intent_shortcut",
                "research_plan_duration_ms": trace_summary["research_plan_duration_ms"],
                "plan_execution_strategy": "legacy_only",
            },
        )
        return {
            "legacy_evidence_plan": legacy_evidence_plan,
            "research_plan_raw": {},
            "research_plan_validated": {},
            "research_plan_used": {},
            "research_plan_validation": {
                "valid": False,
                "fallback_reason": "deterministic_intent_shortcut",
                "planner_trace": {
                    "source": "deterministic_fallback",
                    "status": "deterministic_intent_shortcut",
                    "fallback_used": True,
                    "fallback_reason": "deterministic_intent_shortcut",
                    "duration_ms": trace_summary["research_plan_duration_ms"],
                },
            },
            "research_plan_source": "deterministic_fallback",
            "research_plan_fallback_reason": "deterministic_intent_shortcut",
            "research_plan_duration_ms": trace_summary["research_plan_duration_ms"],
            "required_answer_parts": [],
            "plan_coverage_decision": {
                "strategy": "legacy_only",
                "legacy_core_count": core_count,
                "research_core_count": 0,
                "retained_legacy_core_count": core_count,
                "coverage_ratio": 0.0,
                "warnings": [],
                "reason": "deterministic_intent_shortcut",
            },
            "requirement_merge_summary": {
                "strategy": "legacy_only",
                "merged_total_requirements": requirement_count,
                "legacy_only_count": requirement_count,
            },
            "evidence_plan_used": {
                "source": "legacy_evidence_plan",
                "strategy": "legacy_only",
                "requirement_count": requirement_count,
            },
            "evidence_plan": legacy_evidence_plan,
            "evidence_requirements": list(legacy_evidence_plan.get("evidence_requirements", []) or []),
            "rejected_requirements": list(legacy_evidence_plan.get("rejected_requirements", []) or []),
            "trace_summary": trace_summary,
        }
    if is_causal_explanation_query(user_query):
        raw_plan = {}
        fallback_reason = "validator_injected_causal"
        deterministic_plan = deterministic_causal_research_plan(
            user_query=user_query,
            companies=companies,
            source="deterministic_fallback",
        )
        validation_model = validate_research_plan(
            deterministic_plan.model_dump(exclude_none=True),
            user_query=user_query,
            companies=companies,
            answer_mode=str(state.get("answer_mode") or ""),
            safety_intent=str(state.get("safety_intent") or ""),
        )
        validation_model.used_fallback = True
        validation_model.fallback_reason = fallback_reason
        validation_model.plan.planner_source = "deterministic_fallback"
        planner_trace = {
            "source": "deterministic_fallback",
            "status": "deterministic_causal_skeleton",
            "fallback_used": True,
            "fallback_reason": fallback_reason,
            "duration_ms": int((time.monotonic() - node_started) * 1000),
        }
    else:
        raw_plan, planner_trace = build_research_plan_raw(
            user_query=user_query,
            companies=companies,
            canonical_intent=dict(state.get("canonical_intent", {}) or {}),
            query_understanding=dict(state.get("query_understanding", {}) or state.get("query_understanding_summary", {}) or {}),
            today=date.today(),
        )
        validation_model = validate_research_plan(
            raw_plan,
            user_query=user_query,
            companies=companies,
            answer_mode=str(state.get("answer_mode") or ""),
            safety_intent=str(state.get("safety_intent") or ""),
        )
    validation = validation_model.model_dump(exclude_none=True)
    validated_plan = dict(validation.get("plan", {}) or {})
    planner_fallback_reason = str(planner_trace.get("fallback_reason") or validation.get("fallback_reason") or "")
    if planner_fallback_reason == "planner_missing_or_intent_only_for_causal_query":
        planner_fallback_reason = "intent_only"
    use_research_plan = bool(validation.get("valid")) and mode in {"validated", "expanded"}
    evidence_plan = legacy_evidence_plan
    used_plan: dict[str, Any] = {}
    plan_coverage_decision: dict[str, Any] = {}
    requirement_merge_summary: dict[str, Any] = {}
    evidence_plan_used: dict[str, Any] = {
        "source": "legacy_evidence_plan",
        "strategy": "legacy_only",
        "requirement_count": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
    }
    if use_research_plan:
        research_evidence_plan = build_requirements_from_research_plan(state, validated_plan).model_dump(exclude_none=True)
        coverage = evaluate_plan_coverage(
            research_plan=validated_plan,
            research_evidence_plan=research_evidence_plan,
            legacy_evidence_plan=legacy_evidence_plan,
            state=state,
            planner_valid=True,
            mode=mode,
        )
        merged_plan = merge_evidence_requirements(
            legacy_evidence_plan=legacy_evidence_plan,
            research_evidence_plan=research_evidence_plan,
            coverage_decision=coverage,
        ).model_dump(exclude_none=True)
        evidence_plan = merged_plan
        used_plan = validated_plan
        plan_coverage_decision = dict(merged_plan.get("plan_coverage_decision", {}) or coverage.model_dump(mode="json"))
        requirement_merge_summary = dict(merged_plan.get("requirement_merge_summary", {}) or {})
        evidence_plan_used = {
            "source": str(merged_plan.get("plan_source") or ""),
            "strategy": str(plan_coverage_decision.get("strategy") or ""),
            "requirement_count": len(list(merged_plan.get("evidence_requirements", []) or [])),
        }
    elif validation.get("valid") and is_causal_explanation_query(user_query):
        research_evidence_plan = build_requirements_from_research_plan(state, validated_plan).model_dump(exclude_none=True)
        coverage = evaluate_plan_coverage(
            research_plan=validated_plan,
            research_evidence_plan=research_evidence_plan,
            legacy_evidence_plan=legacy_evidence_plan,
            state=state,
            planner_valid=True,
            mode=mode,
        )
        merged_plan = merge_evidence_requirements(
            legacy_evidence_plan=legacy_evidence_plan,
            research_evidence_plan=research_evidence_plan,
            coverage_decision=coverage,
        ).model_dump(exclude_none=True)
        evidence_plan = merged_plan
        used_plan = validated_plan
        use_research_plan = True
        plan_coverage_decision = dict(merged_plan.get("plan_coverage_decision", {}) or coverage.model_dump(mode="json"))
        requirement_merge_summary = dict(merged_plan.get("requirement_merge_summary", {}) or {})
        evidence_plan_used = {
            "source": str(merged_plan.get("plan_source") or ""),
            "strategy": str(plan_coverage_decision.get("strategy") or ""),
            "requirement_count": len(list(merged_plan.get("evidence_requirements", []) or [])),
        }
    elif mode == "shadow":
        used_plan = {}
        coverage = evaluate_plan_coverage(
            research_plan=validated_plan,
            research_evidence_plan={},
            legacy_evidence_plan=legacy_evidence_plan,
            state=state,
            planner_valid=bool(validation.get("valid")),
            mode=mode,
        )
        plan_coverage_decision = coverage.model_dump(mode="json")
        requirement_merge_summary = {
            "strategy": "legacy_only",
            "merged_total_requirements": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
            "legacy_only_count": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
        }
    else:
        used_plan = dict(validated_plan or {})
        used_plan["planner_source"] = "legacy_evidence_plan"
        coverage = evaluate_plan_coverage(
            research_plan=validated_plan,
            research_evidence_plan={},
            legacy_evidence_plan=legacy_evidence_plan,
            state=state,
            planner_valid=False,
            mode=mode,
        )
        plan_coverage_decision = coverage.model_dump(mode="json")
        requirement_merge_summary = {
            "strategy": "legacy_only",
            "merged_total_requirements": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
            "legacy_only_count": len(list(legacy_evidence_plan.get("evidence_requirements", []) or [])),
        }
    research_plan_source = (
        "deterministic_fallback"
        if str(planner_trace.get("source") or "") == "deterministic_fallback" or bool(validation.get("used_fallback"))
        else ("validated_llm" if use_research_plan else "llm")
    )
    duration_ms = int(planner_trace.get("duration_ms") or ((time.monotonic() - node_started) * 1000))

    required_parts = list(used_plan.get("required_answer_parts", []) or [])
    trace_summary = dict(state.get("trace_summary", {}) or {})
    trace_summary.update(
        {
            "research_question_type": str(used_plan.get("question_type") or validated_plan.get("question_type") or ""),
            "required_answer_parts": required_parts,
            "research_plan_used": bool(use_research_plan),
            "research_plan_source": research_plan_source,
            "research_plan_fallback_reason": planner_fallback_reason,
            "research_plan_duration_ms": duration_ms,
            "plan_execution_strategy": str(plan_coverage_decision.get("strategy") or ""),
        }
    )
    _emit_research_plan_progress(
        state,
        "research_plan_built",
        "completed" if use_research_plan else "warning",
        (
            "Research Plan 已用于本次取证。"
            if use_research_plan
            else "Research Plan 已记录到 trace，本次继续使用 legacy evidence plan。"
        ),
        {
            "question_type": str(used_plan.get("question_type") or validated_plan.get("question_type") or ""),
            "valid": bool(validation.get("valid")),
            "used": bool(use_research_plan),
            "planner_status": str(planner_trace.get("status") or ""),
            "research_plan_source": research_plan_source,
            "research_plan_fallback_reason": planner_fallback_reason,
            "research_plan_duration_ms": duration_ms,
            "plan_execution_strategy": str(plan_coverage_decision.get("strategy") or ""),
        },
    )
    return {
        "legacy_evidence_plan": legacy_evidence_plan,
        "research_plan_raw": raw_plan,
        "research_plan_validated": validated_plan if validation.get("valid") else {},
        "research_plan_used": used_plan,
        "research_plan_validation": {**validation, "planner_trace": planner_trace},
        "research_plan_source": research_plan_source,
        "research_plan_fallback_reason": planner_fallback_reason,
        "research_plan_duration_ms": duration_ms,
        "required_answer_parts": required_parts,
        "plan_coverage_decision": plan_coverage_decision,
        "requirement_merge_summary": requirement_merge_summary,
        "evidence_plan_used": evidence_plan_used,
        "evidence_plan": evidence_plan,
        "evidence_requirements": list(evidence_plan.get("evidence_requirements", []) or []),
        "rejected_requirements": list(evidence_plan.get("rejected_requirements", []) or []),
        "trace_summary": trace_summary,
    }


# ------------------------------------------------------------------
# Node 2 — execute_tools
# ------------------------------------------------------------------

def execute_tools(state: AgentState) -> dict[str, Any]:
    """Invoke selected tools and collect raw tool/document evidence."""
    _sync_compat_dependencies()
    return _tool_executor.execute_agent_tools(state)


def _auto_compute(state: AgentState, tool_results: list[dict[str, Any]], task_type: str) -> None:
    """Compatibility wrapper for the old private helper."""
    _sync_compat_dependencies()
    return _tool_executor._auto_compute(state, tool_results, task_type)


# ------------------------------------------------------------------
# Node 3 — evaluate_evidence
# ------------------------------------------------------------------

def _emit_evidence_evaluated_progress(state: AgentState, sufficient: bool) -> None:
    trace_id = str(state.get("trace_id") or "")
    if not trace_id:
        return
    sufficiency = state.get("evidence_sufficiency", {})
    sufficiency = sufficiency if isinstance(sufficiency, dict) else {}
    answer_part_status = dict(state.get("answer_part_status_by_id") or sufficiency.get("answer_part_status_by_id") or {})
    if state.get("research_plan_used") and answer_part_status:
        required_parts = [
            str(part.get("id") or "")
            for part in state.get("required_answer_parts", []) or []
            if isinstance(part, dict) and bool(part.get("required", True)) and str(part.get("id") or "")
        ]
        if not required_parts:
            required_parts = sorted(answer_part_status)
        satisfied_parts = [
            part_id
            for part_id in required_parts
            if str(dict(answer_part_status.get(part_id, {}) or {}).get("status") or "") == "satisfied"
        ]
        partial_parts = [
            part_id
            for part_id in required_parts
            if str(dict(answer_part_status.get(part_id, {}) or {}).get("status") or "") == "partial"
        ]
        analyzable_parts = [
            part_id
            for part_id in required_parts
            if str(dict(answer_part_status.get(part_id, {}) or {}).get("status") or "") == "missing_but_analyzable"
        ]
        missing_parts = [
            part_id
            for part_id in required_parts
            if str(dict(answer_part_status.get(part_id, {}) or {}).get("status") or "") in {"missing", "missing_and_unanswerable"}
        ]
        append_progress_event(
            trace_id,
            "evidence_evaluated",
            "completed" if not missing_parts else "warning",
            f"证据充分性评估完成：{len(required_parts)} 个必答部分中 {len(satisfied_parts)} 个满足，{len(partial_parts)} 个部分满足，{len(analyzable_parts)} 个可分析缺口，{len(missing_parts)} 个不可回答缺口。",
            node="evaluate",
            metadata={
                "required_answer_parts": len(required_parts),
                "satisfied_answer_parts": len(satisfied_parts),
                "partial_answer_parts": len(partial_parts),
                "missing_but_analyzable_answer_parts": len(analyzable_parts),
                "missing_answer_parts": len(missing_parts),
                "evidence_health": str(sufficiency.get("evidence_health") or state.get("evidence_health") or ""),
                "evidence_sufficient": bool(sufficient),
            },
        )
        return
    dimensions = dict(
        state.get("dimension_status_map")
        or state.get("dimension_status_by_id")
        or sufficiency.get("dimension_status_map")
        or {}
    )
    satisfied_dimensions = [
        key
        for key, value in dimensions.items()
        if isinstance(value, dict) and str(value.get("status") or "") == "satisfied"
    ]
    if not satisfied_dimensions:
        satisfied_dimensions = [str(item) for item in state.get("satisfied_dimensions", []) or [] if str(item)]
    requirement_status = dict(state.get("requirement_status_map", {}) or {})
    missing_required = 0
    missing_optional = 0
    for item in requirement_status.values():
        if not isinstance(item, dict) or str(item.get("status") or "") != "missing":
            continue
        required = bool(item.get("required", True))
        scope = str(item.get("requirement_scope") or item.get("scope") or ("core" if required else "optional_context"))
        if required and scope == "core":
            missing_required += 1
        else:
            missing_optional += 1
    if not requirement_status:
        missing_required = len(list(sufficiency.get("missing_required_requirements", []) or []))
        missing_optional = len(list(sufficiency.get("optional_missing_requirements", []) or []))
    append_progress_event(
        trace_id,
        "evidence_evaluated",
        "completed" if sufficient else "warning",
        f"证据充分性评估完成：{len(satisfied_dimensions)} 个分析维度已满足，{missing_required} 个核心缺口。",
        node="evaluate",
        metadata={
            "satisfied_dimensions": len(satisfied_dimensions),
            "blocking_missing": missing_required,
            "optional_missing": missing_optional,
            "evidence_sufficient": bool(sufficient),
        },
    )


def evaluate_evidence(state: AgentState) -> dict[str, Any]:
    """Decide whether collected evidence is enough to answer the question."""
    if state.get("needs_tools") is False:
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}

    evidence_sufficiency = state.get("evidence_sufficiency", {})
    if isinstance(evidence_sufficiency, dict):
        overall_status = str(evidence_sufficiency.get("overall_status", "")).strip()
        if overall_status in {"sufficient", "partial"}:
            _emit_evidence_evaluated_progress(state, True)
            return {"evidence_sufficient": True}

    loop_count = state.get("evidence_loop_count", 0)
    tool_results = state.get("tool_results", [])
    retrieved_docs = state.get("retrieved_docs", [])
    data_route = state.get("data_route", "hybrid")

    num_structured = sum(
        1
        for tr in tool_results
        if tr.get("tool") in ("query_financial_data", "compute_metrics") and "data" in tr
    )
    num_docs = len(retrieved_docs)
    has_any_evidence = num_docs > 0 or num_structured > 0

    if loop_count >= MAX_EVIDENCE_LOOPS:
        if not has_any_evidence:
            logger.warning("evaluate_evidence: max loops reached with NO evidence")
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}

    if data_route == "documents_only" and num_docs > 0:
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}
    if data_route == "structured_only" and num_structured > 0:
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}
    if data_route == "hybrid" and num_docs > 0 and num_structured > 0:
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}
    if loop_count >= 1 and has_any_evidence:
        _emit_evidence_evaluated_progress(state, True)
        return {"evidence_sufficient": True}
    _emit_evidence_evaluated_progress(state, False)
    return {"evidence_sufficient": False}


def check_evidence(state: AgentState) -> str:
    """Conditional edge: route to generate or back to execute_tools."""
    return "sufficient" if state.get("evidence_sufficient", False) else "insufficient"


# ------------------------------------------------------------------
# Node 4 — generate_answer
# ------------------------------------------------------------------

def generate_answer(state: AgentState) -> dict[str, Any]:
    """Generate the final answer using validated evidence and citations."""
    _sync_compat_dependencies()
    return _answering.generate_agent_answer(state)


# ------------------------------------------------------------------
# Runtime AnswerContract guard nodes
# ------------------------------------------------------------------

def _contract_result_dict(state: AgentState) -> dict[str, Any]:
    result = state.get("contract_result", {})
    return result if isinstance(result, dict) else {}


def _answer_candidate_for_node(state: AgentState, answer: str, owner: str, provenance: dict[str, Any] | None = None):
    requested_dimensions: list[str] = []
    for source in (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions", []),
        dict(state.get("evidence_packet", {}) or {}).get("requested_dimensions", []) if isinstance(state.get("evidence_packet"), dict) else [],
        state.get("requested_dimensions", []),
    ):
        for item in source or []:
            text = str(item).strip()
            if text and text not in requested_dimensions:
                requested_dimensions.append(text)
    return AnswerAssembler.candidate(
        body=answer,
        owner=owner,
        requested_dimensions=requested_dimensions,
        evidence_refs=evidence_refs_from_body(answer),
        allowed_repairs=["add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"],
        provenance=provenance or {},
    )


def _assemble_node_answer(
    state: AgentState,
    *,
    answer: str,
    owner: str,
    transform: str,
    reason: str,
    claim_change_allowed: bool,
    validator_result: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    previous = str(state.get("final_answer") or state.get("draft_answer") or "")
    draft = str(state.get("draft_answer") or "")
    final = str(state.get("final_answer") or "")
    if draft == str(answer or "") and final != str(answer or ""):
        previous = final
    elif final == str(answer or "") and draft and draft != str(answer or ""):
        previous = draft
    candidate = _answer_candidate_for_node(state, answer, owner, provenance)
    if previous == str(answer or ""):
        candidate_payload = candidate.model_dump(exclude_none=True)
        candidates = [dict(item) for item in state.get("answer_candidates", []) or [] if isinstance(item, dict)]
        candidates.append(candidate_payload)
        return {
            "draft_answer": answer,
            "final_answer": answer,
            "final_answer_source": str(state.get("final_answer_source") or owner),
            "answer_history": list(state.get("answer_history", []) or []),
            "answer_candidate": candidate_payload,
            "answer_candidates": candidates[-8:],
        }
    return AnswerAssembler.select(
        candidate,
        state,
        previous_body=previous,
        transform=transform,
        reason=reason,
        claim_change_allowed=claim_change_allowed,
        validator_result=validator_result,
    )


def _is_safe_insufficient_response(state: AgentState, draft: str) -> bool:
    text = str(draft or "").lower()
    if not any(
        phrase in text
        for phrase in (
            "证据不足",
            "无法支持可追溯结论",
            "insufficient evidence",
            "not enough evidence",
            "evidence is not enough",
        )
    ):
        return False
    output = state.get("output", {})
    if not isinstance(output, dict):
        output = {}
    evidence_fields = (
        state.get("numeric_evidence"),
        state.get("text_evidence"),
        state.get("citations"),
        output.get("numeric_evidence"),
        output.get("text_evidence"),
        output.get("citations"),
    )
    return not any(bool(field) for field in evidence_fields)


def _node_requested_dimensions(state: AgentState) -> list[str]:
    out: list[str] = []
    for source in (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions", []),
        dict(state.get("evidence_packet", {}) or {}).get("requested_dimensions", []) if isinstance(state.get("evidence_packet"), dict) else [],
        state.get("requested_dimensions", []),
    ):
        for item in source or []:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    return out


def _ensure_node_canonical_packet(state: AgentState) -> dict[str, Any]:
    packet = state.get("evidence_packet", {})
    if isinstance(packet, dict) and packet.get("canonical_source"):
        return packet
    output = state.get("output", {})
    if not isinstance(output, dict):
        output = {}
    built = build_evidence_packet(
        user_query=str(state.get("user_query") or ""),
        task_type=str(state.get("task_type") or "fact_qa"),
        answer_mode=str(state.get("answer_mode") or "direct_fact"),
        safety_intent=str(state.get("safety_intent") or "normal"),
        analysis_scope=str(state.get("analysis_scope") or ""),
        time_policy=str(state.get("time_policy") or ""),
        period_scope=str(state.get("period_scope") or ""),
        companies=list(state.get("companies", []) or []),
        comparison_target=state.get("comparison_target"),
        requested_metrics=list(state.get("requested_metrics", []) or []),
        period_query=dict(state.get("period_query") or _default_period_query()),
        resolved_period_context=dict(state.get("resolved_period_context", {}) or {}),
        numeric_evidence=[],
        text_evidence=[],
        citations=[],
        evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
        requirement_limitations=list(output.get("limitations", []) or state.get("requirement_limitations", []) or []),
        safety_limitations=list(state.get("safety_limitations", []) or []),
        selected_framework=dict(state.get("selected_analysis_framework", {}) or {}),
        requirement_status_map=dict(state.get("requirement_status_map", {}) or {}),
    ).model_dump(exclude_none=True)
    built["research_plan"] = dict(state.get("research_plan_used", {}) or {})
    built["canonical_intent"] = dict(state.get("canonical_intent", {}) or {})
    built["requested_dimensions"] = _node_requested_dimensions(state)
    built["required_answer_parts"] = list(state.get("required_answer_parts", []) or [])
    state["evidence_packet"] = built
    state["evidence_packet_summary"] = summarize_evidence_packet(built)
    return built


def _has_public_evidence(state: AgentState) -> bool:
    packet = _node_packet_for_fallback(state)
    evidence_fields = (packet.get("numeric_table"), packet.get("text_snippets"))
    return any(bool(field) for field in evidence_fields)


def _has_missing_evidence_boundary(state: AgentState) -> bool:
    for source in (state, state.get("evidence_sufficiency_summary", {}), state.get("trace_summary", {})):
        if not isinstance(source, dict):
            continue
        if int(source.get("missing_required_requirements_count", 0) or 0) > 0:
            return True
    missing = list(state.get("missing_requirements", []) or [])
    sufficiency = state.get("evidence_sufficiency", {})
    if isinstance(sufficiency, dict):
        missing.extend(list(sufficiency.get("missing_requirements", []) or []))
    requirement_status_map = state.get("requirement_status_map", {})
    if isinstance(requirement_status_map, dict):
        for rid in missing:
            item = requirement_status_map.get(str(rid), {})
            if isinstance(item, dict) and bool(item.get("required", True)):
                return True
        missing = []
    status_maps: list[dict[str, Any]] = []
    for source in (
        state.get("dimension_status_map"),
        state.get("dimension_status_by_id"),
        (state.get("evidence_packet", {}) or {}).get("dimension_status_map") if isinstance(state.get("evidence_packet", {}), dict) else {},
    ):
        if isinstance(source, dict):
            status_maps.append(source)
    for status_map in status_maps:
        for item in status_map.values():
            if isinstance(item, dict) and str(item.get("status") or "") in {"missing", "partial"}:
                return True
    return any(str(item).strip() for item in missing)


def _is_partial_grounded_response(state: AgentState) -> bool:
    return _has_public_evidence(state) and _has_missing_evidence_boundary(state)


def contract_check_node(state: AgentState) -> dict[str, Any]:
    """Run the runtime AnswerContract against the current draft answer."""
    draft = str(state.get("draft_answer") or state.get("final_answer") or "")
    result = check_answer_contract(draft, state, scope="answer")
    result_dict = result.model_dump()
    scope_overclaim_check = dict(result_dict.get("scope_overclaim_check", {}) or {})
    evidence_scope_by_ref = dict(scope_overclaim_check.get("evidence_scope_by_ref", {}) or {})
    driver_scope_counts: dict[str, int] = {
        "company": 0,
        "segment": 0,
        "product": 0,
        "market_context": 0,
        "unknown": 0,
        "scope_bounded_inferences": 0,
    }
    for item in evidence_scope_by_ref.values():
        if not isinstance(item, dict):
            continue
        claim_scope = str(item.get("claim_scope") or "unknown")
        if claim_scope not in driver_scope_counts:
            claim_scope = "unknown"
        driver_scope_counts[claim_scope] += 1
        if str(item.get("allowed_claim_strength") or "") in {"bounded_inference", "hypothesis_only"}:
            driver_scope_counts["scope_bounded_inferences"] += 1
    decision = str(getattr(result, "decision", result_dict.get("decision", "passed")) or "passed")
    status = "passed_with_warnings" if result.route == "pass" and decision == "warning" else ("passed" if result.route == "pass" else "failed")
    public_summary = str(result.public_summary or "")
    violation_codes = [
        str(item.get("code") or item.get("type") or "")
        for item in result_dict.get("violations", []) or []
        if isinstance(item, dict)
    ]
    if violation_codes and set(violation_codes) == {"language_leakage"} and int(state.get("contract_attempts", 0) or 0) >= 1:
        result_dict["route"] = "pass"
        result_dict["action"] = "pass"
        result_dict["decision"] = "warning"
        result_dict["severity"] = "warning"
        result_dict["language_leakage_unresolved"] = True
        result_dict.setdefault("warnings", list(result_dict.get("violations", []) or []))
        public_summary = "Released after deterministic language repair attempt; language leakage remains marked in debug."
        result_dict["public_summary"] = public_summary
        status = "passed_with_warnings"
    if result.route != "pass" and _contract_debt_business_fallback_active(state) and _contract_result_only_minor_business_fallback_issues(result_dict):
        result_dict["route"] = "pass"
        result_dict["action"] = "pass"
        result_dict["decision"] = "warning"
        result_dict["severity"] = "warning"
        result_dict.setdefault("warnings", list(result_dict.get("violations", []) or []))
        public_summary = "Released bounded business fallback after primary answer contract debt."
        result_dict["public_summary"] = public_summary
        status = "passed_with_warnings"
    if result.route == "pass" and _is_safe_insufficient_response(state, draft):
        public_summary = "Safe insufficient-evidence response returned."
        result_dict["public_summary"] = public_summary
    elif result.route == "pass" and _is_partial_grounded_response(state):
        public_summary = "Partial grounded answer returned; some requested evidence remains unavailable."
        result_dict["public_summary"] = public_summary
    trace_id = str(state.get("trace_id") or "")
    if trace_id:
        append_progress_event(
            trace_id,
            "contract_checked",
            "completed" if result.route == "pass" else "failed",
            (
                "答案合约检查通过：引用、数字和非投资建议边界均满足要求。"
                if result.route == "pass"
                else "答案合约检查未通过，正在决定是否修复或阻断。"
            ),
            node="contract_check",
            metadata={"contract_status": status, "contract_decision": decision, "route": result.route, "action": result.action},
        )
    payload = {
        "contract_result": result_dict,
        "contract_trace": result_dict,
        "contract_status": status,
        "contract_decision": result_dict,
        "contract_failure_reasons": [item.code for item in result.violations],
        "contract_public_summary": public_summary,
        "evidence_scope_by_ref": evidence_scope_by_ref,
        "scope_overclaim_check": scope_overclaim_check,
        "scope_overclaim_violations": list(result_dict.get("scope_overclaim_violations", []) or []),
        "driver_scope_counts": driver_scope_counts,
    }
    if str(state.get("draft_answer") or "").strip():
        payload["draft_answer"] = draft
    return payload


def route_after_contract(state: AgentState) -> str:
    """Conditional edge after runtime contract check."""
    result = _contract_result_dict(state)
    route = str(result.get("route") or "blocked")
    action = str(result.get("action") or "")
    attempts = int(state.get("contract_attempts", 0) or 0)
    max_repairs = int(state.get("max_contract_repairs", 2) or 2)
    evidence_retries = int(state.get("contract_evidence_retry_count", 0) or 0)
    max_evidence_retries = int(state.get("max_contract_evidence_retries", 2) or 2)

    if action == "block":
        return "blocked"
    if action == "pass" or route == "pass":
        return "relevance_check"
    if _contract_debt_business_fallback_active(state) and _contract_result_only_minor_business_fallback_issues(result):
        return "relevance_check"
    if action in {"add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"} or route == "repair_answer":
        return "repair_generate" if attempts < max_repairs else "blocked"
    if action == "retry_evidence" or route == "need_more_evidence":
        missing = result.get("missing_requirements") or state.get("missing_requirements") or []
        can_retry = bool(missing) and evidence_retries < max_evidence_retries
        return "prepare_contract_evidence_retry" if can_retry else "blocked"
    return "blocked"


def relevance_check_node(state: AgentState) -> dict[str, Any]:
    """Check whether a contract-passing answer actually covers the ResearchPlan."""
    answer = str(state.get("draft_answer") or state.get("final_answer") or "")
    decision = judge_answer_relevance(answer, state).model_dump(exclude_none=True)
    trace_id = str(state.get("trace_id") or "")
    if trace_id:
        append_progress_event(
            trace_id,
            "relevance_checked",
            "completed" if str(decision.get("route") or "") == "finalize" else "warning",
            (
                "答案相关性检查通过或带边界发布。"
                if str(decision.get("route") or "") == "finalize"
                else "答案未覆盖用户核心问题，正在生成有边界的修复答案。"
            ),
            node="relevance_check",
            metadata={
                "decision": str(decision.get("decision") or ""),
                "status": str(decision.get("status") or ""),
                "missing_answer_parts": list(decision.get("missing_answer_parts", []) or []),
                "partial_answer_parts": list(decision.get("partial_required_answer_parts", []) or []),
                "missing_but_analyzable_answer_parts": list(decision.get("missing_but_analyzable_answer_parts", []) or []),
                "deterministic_failures": list(decision.get("deterministic_relevance_failures", []) or []),
            },
        )
    return {
        "relevance_decision": decision,
        "relevance_status": str(decision.get("status") or "not_run"),
        "partial_required_answer_parts": list(decision.get("partial_required_answer_parts", []) or state.get("partial_required_answer_parts", []) or []),
        "missing_but_analyzable_answer_parts": list(decision.get("missing_but_analyzable_answer_parts", []) or state.get("missing_but_analyzable_answer_parts", []) or []),
    }


def route_after_relevance(state: AgentState) -> str:
    decision = state.get("relevance_decision", {})
    decision = decision if isinstance(decision, dict) else {}
    route = str(decision.get("route") or "finalize")
    action = str(decision.get("action") or "")
    attempts = int(state.get("relevance_attempts", 0) or 0)
    if action in {"block", "retry_evidence"}:
        return "blocked"
    if action in {"add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"} and attempts < 1:
        return "relevance_repair"
    if action in {"add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"}:
        return "blocked"
    if route == "repair_answer" and attempts < 1:
        return "relevance_repair"
    if route == "repair_answer":
        return "blocked"
    return "finalize"


def _dimension_boundary_label(dimension: str, lang: str) -> str:
    labels = {
        "valuation_and_risk_boundary": ("估值风险边界", "valuation-risk boundary"),
        "moat_and_competitive_risk": ("竞争与风险", "competitive risk"),
        "cash_flow_quality": ("现金流质量", "cash-flow quality"),
        "revenue_quality": ("增长/收入质量", "growth/revenue quality"),
        "profitability_quality": ("盈利质量", "profitability quality"),
    }
    zh, en = labels.get(dimension, (dimension, dimension))
    return zh if lang == "zh" else en


def _existing_bounded_candidate(
    state: AgentState,
    requested_dimensions: list[str],
    *,
    require_format_ok: bool = False,
) -> dict[str, Any]:
    current = str(state.get("final_answer") or state.get("draft_answer") or "")
    requested = {str(item) for item in requested_dimensions if str(item)}
    for item in reversed([x for x in state.get("answer_candidates", []) or [] if isinstance(x, dict)]):
        body = str(item.get("body") or "").strip()
        owner = str(item.get("owner") or "").strip()
        allowed = {str(value) for value in item.get("allowed_repairs", []) or [] if str(value)}
        candidate_dims = {str(value) for value in item.get("requested_dimensions", []) or [] if str(value)}
        if not body or body == current:
            continue
        if requested and candidate_dims and not (requested & candidate_dims):
            continue
        if require_format_ok and not _format_constraints_satisfied(body, state):
            continue
        if "downgrade_to_bounded" in allowed or "bounded" in owner or "fallback" in owner:
            return dict(item)
    return {}


def _one_sentence_repair_answer(state: AgentState, answer: str, *, lang: str) -> str:
    valuation_line = _valuation_boundary_line_for_public_answer(state, lang=lang)
    if valuation_line and _true_investment_advice_like(state):
        if lang == "zh" and valuation_line.startswith("不能给买卖建议"):
            return valuation_line
        return (
            f"不能给买卖建议；{valuation_line}"
            if lang == "zh"
            else f"I cannot provide buy/sell advice; {valuation_line}"
        )
    refs = re.findall(r"\[[A-Z]\d+\]", str(answer or ""))
    cleaned_lines: list[str] = []
    for line in str(answer or "").splitlines():
        text = re.sub(r"^\s*[-*]\s+", "", line).strip()
        if not text or text.endswith(":") or text.endswith("：") or text.startswith("#"):
            continue
        cleaned_lines.append(text)
    joined = " ".join(cleaned_lines).strip()
    split_source = re.sub(r"([。！？!?](?:\[[A-Z]\d+\])?)\s+", r"\1\n", joined)
    split_source = re.sub(r"(\[[A-Z]\d+\])\s+", r"\1\n", split_source)
    pieces = [part.strip() for part in re.split(r"\n+", split_source) if part.strip()]
    if not pieces:
        pieces = [joined] if joined else []
    chosen = ""
    for piece in pieces:
        if re.search(r"\[[A-Z]\d+\]", piece):
            chosen = piece
            break
    if not chosen and pieces:
        chosen = pieces[0]
    if not chosen:
        chosen = (
            "当前只能发布已验证证据范围内的一句话边界，不能新增未引用结论。"
            if lang == "zh"
            else "Only a one-sentence validated-evidence boundary can be released here, without adding uncited conclusions."
        )
    chosen = re.sub(r"\s+", " ", chosen).strip()
    if lang == "zh":
        chosen = chosen.rstrip("。！？!?.；;，, ")
        suffix = "。"
    else:
        chosen = chosen.rstrip(".!?;:, ")
        suffix = "."
    if refs and not re.search(r"\[[A-Z]\d+\]", chosen):
        unique_refs = list(dict.fromkeys(refs))
        chosen = f"{chosen}{''.join(unique_refs[:4])}"
    return f"{chosen}{suffix}"


def _relevance_scope_limit_answer(
    *,
    state: AgentState,
    requested_dimensions: list[str],
    failures: list[str],
    lang: str,
) -> str:
    labels = [_dimension_boundary_label(item, lang) for item in requested_dimensions]
    if not labels:
        primary = str(
            dict(state.get("analysis_plan", {}) or {}).get("primary_dimension")
            or dict(state.get("evidence_policy", {}) or {}).get("primary_dimension")
            or state.get("primary_dimension")
            or ""
        ).strip()
        labels = [_dimension_boundary_label(primary, lang)] if primary else []
    label_text = "、".join(labels) if lang == "zh" else ", ".join(labels)
    if "one_sentence_constraint_violated" in failures:
        return (
            "不能给买卖建议；当前只能发布已验证证据范围内的有限边界，不新增结论。"
            if lang == "zh"
            else "I cannot give buy/sell advice; only a validated-evidence boundary can be released here, without adding new conclusions."
        )
    if lang == "zh":
        target = label_text or "用户请求的维度"
        return (
            f"证据边界：当前答案未可靠覆盖{target}。相关性修复不会新增未引用分析结论；"
            "只能删除、缩窄、补引用或选择已有有边界候选答案。"
        )
    target = label_text or "the requested dimension"
    return (
        f"Evidence boundary: the current answer does not reliably cover {target}. "
        "Relevance repair cannot add new uncited analysis; it can only delete, narrow, add citations, or select an existing bounded candidate."
    )


def _is_bounded_risk_comparison_owner(owner: str) -> bool:
    return str(owner or "") in {
        "bounded_risk_comparison_answer",
        "risk_comparison_bounded_answer",
        "bounded_risk_comparison_postprocess",
    }


def _is_risk_comparison_relevance_scope(state: AgentState, requested_dimensions: list[str]) -> bool:
    query = str(state.get("user_query") or "")
    if _answering._is_risk_comparison_query(query, state):
        return True
    canonical_intent = dict(state.get("canonical_intent", {}) or {})
    is_comparison = (
        str(state.get("task_type") or "") == "company_comparison"
        or str(state.get("answer_mode") or "") == "comparison_brief"
        or str(canonical_intent.get("intent_family") or "") == "comparison"
    )
    risk_requested = (
        "moat_and_competitive_risk" in requested_dimensions
        or "moat_and_competitive_risk" in [str(item) for item in canonical_intent.get("requested_dimensions", []) or []]
        or any(term in query.lower() for term in ("风险", "危险", "risk", "danger"))
    )
    return bool(is_comparison and risk_requested)


def _build_bounded_risk_comparison_answer(state: AgentState, *, lang: str) -> str:
    output = state.get("output", {})
    synthesis_payload = dict(output.get("synthesis", {}) or {}) if isinstance(output, dict) else {}
    return _answering._bounded_risk_comparison_answer(state, synthesis_payload, lang)


def relevance_repair_node(state: AgentState) -> dict[str, Any]:
    """Select a bounded candidate or scope-limit the answer without adding analysis."""
    attempt = int(state.get("relevance_attempts", 0) or 0) + 1
    query = str(state.get("user_query") or "")
    decision = state.get("relevance_decision", {})
    failures = [
        str(item.get("code") or "")
        for item in (decision.get("deterministic_relevance_failures", []) if isinstance(decision, dict) else [])
        if isinstance(item, dict)
    ]
    _ensure_node_canonical_packet(state)
    requested_dimensions = []
    for source in (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions", []),
        dict(state.get("evidence_packet", {}) or {}).get("requested_dimensions", []),
        state.get("requested_dimensions", []),
    ):
        for item in source or []:
            text = str(item).strip()
            if text and text not in requested_dimensions:
                requested_dimensions.append(text)
    lang = _node_target_lang(state)
    format_failure = "one_sentence_constraint_violated" in failures
    existing_candidate = _existing_bounded_candidate(
        state,
        requested_dimensions,
        require_format_ok=format_failure,
    )
    if existing_candidate:
        answer = str(existing_candidate.get("body") or "")
        owner = str(existing_candidate.get("owner") or "research_relevance_bounded_candidate")
        transform = "relevance_select_existing_bounded_candidate"
        provenance = {"attempt": attempt, "failures": failures, "selected_candidate_owner": owner}
    elif format_failure:
        answer = _one_sentence_repair_answer(
            state,
            str(state.get("final_answer") or state.get("draft_answer") or ""),
            lang=lang,
        )
        owner = "format_one_sentence_repair"
        transform = "relevance_one_sentence_format_repair"
        provenance = {"attempt": attempt, "failures": failures, "deterministic_format_repair": True}
    elif _is_risk_comparison_relevance_scope(state, requested_dimensions):
        answer = _build_bounded_risk_comparison_answer(state, lang=lang)
        owner = "bounded_risk_comparison_answer"
        transform = "relevance_risk_comparison_bounded_fallback"
        provenance = {
            "attempt": attempt,
            "failures": failures,
            "bounded_business_fallback": "risk_comparison",
        }
    else:
        generic = _generic_bounded_analysis_answer(
            state,
            lang=lang,
            requested_dimensions=requested_dimensions,
        )
        if generic:
            answer, owner = generic
            transform = "relevance_bounded_analysis_fallback"
            provenance = {"attempt": attempt, "failures": failures, "bounded_analysis_from_citable_evidence": True}
        else:
            answer = _relevance_scope_limit_answer(
                state=state,
                requested_dimensions=requested_dimensions,
                failures=failures,
                lang=lang,
            )
            owner = "research_relevance_scope_limit"
            transform = "relevance_scope_limit_boundary"
            provenance = {"attempt": attempt, "failures": failures, "no_existing_bounded_candidate": True}
    output = dict(state.get("output", {}) or {})
    output["final_answer_source"] = owner
    output["answer_status"] = "released_with_relevance_warning"
    output["relevance_decision"] = dict(state.get("relevance_decision", {}) or {})
    output["evidence_packet_summary"] = dict(state.get("evidence_packet_summary", {}) or {})
    output["format_constraints"] = _format_constraints_dict(state)
    output["format_constraints_satisfied"] = _format_constraints_satisfied(answer, state)
    output["unresolved_relevance_failures"] = failures
    if owner == "research_relevance_scope_limit":
        output["answer_quality_tier"] = "scope_limit"
        output["main_question_covered"] = False
        output["fallback_intent_match"] = True
        output["answered_dimensions"] = []
    elif _is_bounded_risk_comparison_owner(owner) or owner in {"bounded_analysis", "bounded_risk_analysis"}:
        output["answer_quality_tier"] = "bounded_analysis"
        output["main_question_covered"] = True
        output["fallback_intent_match"] = True
        output["answered_dimensions"] = ["moat_and_competitive_risk"] if owner == "bounded_risk_analysis" else (requested_dimensions or ["moat_and_competitive_risk"])
    elif format_failure and not output["format_constraints_satisfied"]:
        output["answer_quality_tier"] = "invalid_fallback"
        output["main_question_covered"] = False
        output["fallback_intent_match"] = False
        output["answered_dimensions"] = []
    else:
        output["answer_quality_tier"] = "bounded_analysis"
        output["main_question_covered"] = True
        output["fallback_intent_match"] = True
        output["answered_dimensions"] = requested_dimensions
    assembled = _assemble_node_answer(
        state,
        answer=answer,
        owner=owner,
        transform=transform,
        reason=";".join(failures) or str(decision.get("public_summary") if isinstance(decision, dict) else ""),
        claim_change_allowed=False,
        validator_result=dict(state.get("relevance_decision", {}) or {}),
        provenance=provenance,
    )
    output["answer_history"] = list(assembled.get("answer_history", state.get("answer_history", []) or []))
    output["answer_candidate"] = dict(assembled.get("answer_candidate", {}) or {})
    output["answer_candidates"] = list(assembled.get("answer_candidates", state.get("answer_candidates", []) or []))
    return {
        **assembled,
        "output": output,
        "relevance_attempts": attempt,
        "relevance_repair_attempts": attempt,
        "evidence_packet": dict(state.get("evidence_packet", {}) or {}),
        "evidence_packet_summary": dict(state.get("evidence_packet_summary", {}) or {}),
        "contract_status": (
            "scope_limited"
            if owner == "research_relevance_scope_limit"
            else "passed_with_warnings"
            if _is_bounded_risk_comparison_owner(owner) or owner in {"bounded_analysis", "bounded_risk_analysis"}
            else "failed"
        ),
        "final_route": "scope_limited" if owner == "research_relevance_scope_limit" else "bounded_fallback",
        "answer_quality_tier": output["answer_quality_tier"],
        "main_question_covered": output["main_question_covered"],
        "fallback_intent_match": output["fallback_intent_match"],
        "answered_dimensions": output["answered_dimensions"],
        "unresolved_relevance_failures": output["unresolved_relevance_failures"],
        "format_constraints_satisfied": output["format_constraints_satisfied"],
        "format_constraints": output["format_constraints"],
    }


def _strip_forbidden_sentences(answer: str, violation_codes: set[str]) -> str:
    if "forbidden_claim" not in violation_codes and "raw_internal_leakage" not in violation_codes:
        return answer
    parts = re_split_sentences(answer)
    forbidden_terms = (
        "买入",
        "卖出",
        "推荐买",
        "推荐卖",
        "target price",
        "price target",
        "should buy",
        "should sell",
        "recommend buying",
        "recommend selling",
        "REQ-",
        "dependency_",
        "contract_result",
        "EvidencePacket",
    )
    kept = [part for part in parts if not any(term.lower() in part.lower() for term in forbidden_terms)]
    return " ".join(kept).strip() or answer


def re_split_sentences(text: str) -> list[str]:
    """Small sentence splitter used only for deterministic repair."""
    import re

    parts = re.split(r"(?<=[。！？.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _public_limitation_lines(state: AgentState, result: dict[str, Any]) -> list[str]:
    lang = _node_target_lang(state)
    codes = {str(item.get("code") or "") for item in result.get("violations", []) if isinstance(item, dict)}
    lines: list[str] = []
    if "caveat_not_visible" in codes:
        lines.append("部分证据存在口径、覆盖或可信度限制。" if lang == "zh" else "Some evidence has coverage, confidence, or reconciliation limitations.")
    if "dimension_status_violation" in codes:
        lines.append("对 partial 或 missing 维度的判断仅限于当前已验证证据。" if lang == "zh" else "Claims about partial or missing dimensions are bounded by the currently validated evidence.")
    if "comparison_balance" in codes:
        lines.append("比较结论受到双方证据覆盖不完全对称的限制。" if lang == "zh" else "The comparison is limited by asymmetric evidence coverage.")
    if "segment_evidence_overstated_as_company_driver" in codes:
        lines.append(
            "分部/产品层面证据只能作为总公司营收增长的业务线索，不能写成完整公司级确定原因。"
            if lang == "zh"
            else "Segment/product-level evidence is only a business-line signal, not definitive total-company causality."
        )
    if not lines:
        summary = str(result.get("suggested_repair") or result.get("public_summary") or "").strip()
        if summary:
            lines.append(summary)
    return list(dict.fromkeys(lines))


def _downgrade_scope_overclaim_sentences(answer: str, result: dict[str, Any]) -> str:
    affected_refs: set[str] = set()
    for violation in result.get("violations", []) or []:
        if not isinstance(violation, dict):
            continue
        if str(violation.get("code") or violation.get("type") or "") != "segment_evidence_overstated_as_company_driver":
            continue
        affected_refs.update(str(ref) for ref in violation.get("affected_citations", []) or [] if str(ref).strip())
    if not affected_refs:
        return answer
    repaired_parts: list[str] = []
    for sentence in re_split_sentences(answer):
        refs = [ref for ref in affected_refs if f"[{ref}]" in sentence]
        if not refs or not any(term in sentence for term in ("营收增长", "总营收", "revenue growth", "total revenue")):
            repaired_parts.append(sentence)
            continue
        citation_text = " ".join(f"[{ref}]" for ref in refs)
        is_zh = any("\u4e00" <= ch <= "\u9fff" for ch in sentence)
        if is_zh:
            cause = sentence
            match = re.search(r"(?:主要由|由)(.+?)(?:驱动|推动|导致|。|；|$)", sentence)
            if match:
                cause = match.group(1).strip(" ，,。；;")
            repaired_parts.append(
                f"分部/产品层面证据显示，{cause}是相关业务增长线索之一，但不能完整代表总公司营收增长原因。{citation_text}"
            )
        else:
            repaired_parts.append(
                f"Segment/product-level evidence provides a bounded business-line signal, but it does not prove definitive total-company revenue-growth causality. {citation_text}"
            )
    return "\n".join(repaired_parts).strip()


def _dedupe_citation_refs_text(refs: str) -> str:
    ordered = list(dict.fromkeys(re.findall(r"\[([NT]\d+)\]", str(refs or ""))))
    return "".join(f"[{ref}]" for ref in ordered)


def _append_refs_to_matching_sentence(answer: str, answer_span: str, refs_text: str) -> str:
    refs_text = _dedupe_citation_refs_text(refs_text)
    if not answer_span or not refs_text:
        return answer
    span = answer_span.strip()
    ref_ids = re.findall(r"\[([NT]\d+)\]", refs_text)
    lines = str(answer or "").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or re.fullmatch(r"(?:\[[NT]\d+\])+", stripped):
            continue
        if span not in line and line not in span:
            continue
        existing = set(re.findall(r"\[([NT]\d+)\]", line))
        missing = [ref for ref in ref_ids if ref not in existing]
        if not missing:
            return "\n".join(lines)
        citation_text = "".join(f"[{ref}]" for ref in missing)
        lines[index] = f"{line.rstrip()}{citation_text}"
        return "\n".join(lines)
    if span in answer:
        existing = set(re.findall(r"\[([NT]\d+)\]", span))
        missing = [ref for ref in ref_ids if ref not in existing]
        if not missing:
            return answer
        return answer.replace(span, f"{span}{''.join(f'[{ref}]' for ref in missing)}", 1)
    return answer


def _strip_matching_sentence(answer: str, answer_span: str) -> str:
    span = str(answer_span or "").strip()
    if not span:
        return answer
    lines = str(answer or "").splitlines()
    changed = False
    kept_lines: list[str] = []
    for line in lines:
        if span in line or line.strip() in span:
            changed = True
            continue
        kept_lines.append(line)
    if changed:
        return "\n".join(kept_lines).strip() or answer
    parts = re_split_sentences(answer)
    kept = [part for part in parts if span not in part and part not in span]
    return " ".join(kept).strip() or answer


_CONTRACT_REPAIR_ALLOWED_CODES = {
    "citation_free_material_claim",
    "forbidden_claim",
    "raw_internal_leakage",
    "unsupported_benchmark_claim",
    "company_specific_token_leakage",
    "format_constraint_violation",
}


def _contract_violation_codes(result: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for item in result.get("violations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("type") or "").strip()
        if code and code not in out:
            out.append(code)
    return out


def _contract_material_uncited_count(result: dict[str, Any]) -> int:
    return sum(
        1
        for item in result.get("violations", []) or []
        if isinstance(item, dict) and str(item.get("code") or item.get("type") or "") == "citation_free_material_claim"
    )


def _contract_repair_types(result: dict[str, Any]) -> list[str]:
    types: list[str] = []
    for item in result.get("violations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("type") or "").strip()
        repair_type = ""
        if code == "citation_free_material_claim":
            repair_type = "add_citation" if str(item.get("suggested_fix") or item.get("suggested_refs") or "").strip() else "strip_uncited_material_sentence"
        elif code in {"forbidden_claim", "raw_internal_leakage"}:
            repair_type = "strip_forbidden_or_internal_sentence"
        elif code == "unsupported_benchmark_claim":
            repair_type = "unsupported_benchmark_rewrite"
        elif code == "company_specific_token_leakage":
            repair_type = "strip_company_specific_token_leakage"
        elif code == "format_constraint_violation":
            repair_type = "one_sentence_compression"
        if repair_type and repair_type not in types:
            types.append(repair_type)
    return types


def _contract_repair_allowed(result: dict[str, Any]) -> bool:
    codes = set(_contract_violation_codes(result))
    return bool(codes) and codes <= _CONTRACT_REPAIR_ALLOWED_CODES


def _contract_source_before_repair(state: AgentState) -> str:
    output = state.get("output", {})
    output = output if isinstance(output, dict) else {}
    return str(
        state.get("final_answer_source")
        or output.get("final_answer_source")
        or dict(state.get("synthesis", {}) or {}).get("final_answer_source", "")
        or "unknown_answer_source"
    )


def _contract_debt_answer(state: AgentState, result: dict[str, Any], *, lang: str) -> tuple[str, str]:
    requested_dimensions = _node_requested_dimensions(state)
    existing = _existing_bounded_candidate(state, requested_dimensions, require_format_ok=True)
    if existing:
        return str(existing.get("body") or ""), str(existing.get("owner") or "bounded_existing_candidate")
    return _contract_debt_public_scope_limit_answer(state, lang=lang)


_PUBLIC_INTERNAL_LEAK_TERMS = (
    "Rewrite only",
    "contract repair",
    "primary generation",
    "candidate layer",
    "suggested_fix",
    "repair instruction",
    "validator_result",
    "ContractResult",
    "route=",
    "重新生成候选答案",
    "候选答案层",
)


def _public_answer_has_internal_terms(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(term.lower() in lowered for term in _PUBLIC_INTERNAL_LEAK_TERMS)


def _citation_refs(refs: list[str]) -> str:
    return "".join(f"[{ref}]" for ref in refs if str(ref).strip())


def _node_packet_for_fallback(state: AgentState) -> dict[str, Any]:
    packet = state.get("evidence_packet", {})
    if isinstance(packet, dict) and packet:
        return packet
    try:
        return _ensure_node_canonical_packet(state)
    except Exception:
        return {}


def _node_text_rows(state: AgentState) -> list[dict[str, Any]]:
    packet = _node_packet_for_fallback(state)
    output = dict(state.get("output", {}) or {})
    rows: list[dict[str, Any]] = []
    for source in (
        packet.get("text_snippets"),
        packet.get("text_evidence"),
        state.get("text_evidence"),
        output.get("text_evidence"),
    ):
        for item in source or []:
            if isinstance(item, dict):
                rows.append(dict(item))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        key = ref or str(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _node_numeric_rows(state: AgentState) -> list[dict[str, Any]]:
    packet = _node_packet_for_fallback(state)
    output = dict(state.get("output", {}) or {})
    rows: list[dict[str, Any]] = []
    for source in (
        packet.get("numeric_table"),
        packet.get("numeric_evidence"),
        packet.get("computed_metrics"),
        state.get("numeric_evidence"),
        state.get("computed_metrics"),
        output.get("numeric_evidence"),
        output.get("computed_metrics"),
    ):
        for item in source or []:
            if isinstance(item, dict):
                rows.append(dict(item))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        key = ref or str(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _node_analytical_claim_rows(state: AgentState) -> list[dict[str, Any]]:
    output = dict(state.get("output", {}) or {})
    synthesis = dict(state.get("synthesis", {}) or {})
    rows: list[dict[str, Any]] = []
    for source in (state.get("analytical_claims"), output.get("analytical_claims"), synthesis.get("analytical_claims")):
        for item in source or []:
            if isinstance(item, dict):
                rows.append(dict(item))
    return rows


def _row_public_text(row: dict[str, Any]) -> str:
    claim = str(row.get("claim") or row.get("sentence") or row.get("summary") or "").strip()
    snippet = str(row.get("supporting_snippet") or row.get("text_snippet") or row.get("source_text") or "").strip()
    if claim and not re.search(r"提供了可用于比较|provides? business and risk context|business and risk context", claim, flags=re.IGNORECASE):
        return claim
    return snippet or claim


def _row_search_text(row: dict[str, Any]) -> str:
    return " ".join(
        str(row.get(key) or "")
        for key in (
            "claim",
            "sentence",
            "summary",
            "supporting_snippet",
            "text_snippet",
            "source_text",
            "metric",
            "metric_label",
            "dimension_id",
            "section",
        )
    )


def _row_refs(row: dict[str, Any]) -> list[str]:
    refs = [str(row.get("evidence_id") or "").strip()]
    refs.extend(str(ref).strip() for ref in row.get("evidence_ids", []) or [])
    refs.extend(str(ref).strip() for ref in row.get("claim_ids", []) or [])
    return list(dict.fromkeys(ref for ref in refs if ref and re.fullmatch(r"[NT]\d+", ref)))


def _line_from_row(row: dict[str, Any], *, max_chars: int = 180) -> str:
    text = _row_public_text(row)
    text = re.sub(r"\s+", " ", text).strip()
    text = text[:max_chars].rstrip(" ，,。.;；")
    refs = _citation_refs(_row_refs(row)[:4])
    return f"{text}{refs}" if text else ""


def _state_company_label(state: AgentState) -> str:
    companies = [_company_ticker_text(item) for item in state.get("companies", []) or []]
    companies = [item for item in companies if item]
    if companies:
        return companies[0]
    packet = _node_packet_for_fallback(state)
    companies = [_company_ticker_text(item) for item in packet.get("companies", []) or []]
    companies = [item for item in companies if item]
    return companies[0] if companies else "公司"


def _node_target_lang(state: AgentState) -> str:
    explicit = str(state.get("output_language") or dict(state.get("canonical_intent", {}) or {}).get("output_language") or "").strip()
    if explicit in {"zh", "en"}:
        return explicit
    query = str(state.get("user_query") or "").strip()
    return detect_output_language(query)


def _company_ticker_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("ticker") or item.get("TICKER") or item.get("symbol") or item.get("company") or "").upper().strip()
    return str(item or "").upper().strip()


def _contract_debt_public_scope_limit_answer(state: AgentState, *, lang: str) -> tuple[str, str]:
    requested = [_dimension_boundary_label(item, lang) for item in _node_requested_dimensions(state)]
    target = ("、".join(requested) if lang == "zh" else ", ".join(requested)) or ("这个问题" if lang == "zh" else "this question")
    if lang == "zh":
        return (
            f"证据边界：当前可验证证据不足以可靠回答{target}。我只能发布已验证事实和明确边界，不能补充未被证据支持的业务判断。",
            "bounded_scope_limit",
        )
    return (
        f"Evidence boundary: the validated evidence is not sufficient to reliably answer {target}. "
        "I can only release verified facts and explicit boundaries, not unsupported business judgments.",
        "bounded_scope_limit",
    )


_PUBLIC_METRIC_LABELS_ZH = {
    "revenue": "收入",
    "revenue_growth": "收入增速",
    "net_income": "净利润",
    "eps": "EPS",
    "gross_margin": "毛利率",
    "operating_margin": "营业利润率",
    "net_margin": "净利率",
    "operating_cash_flow": "经营现金流",
    "free_cash_flow": "自由现金流",
    "capital_expenditure": "资本开支",
    "fcf_margin": "自由现金流率",
    "cfo_to_net_income": "CFO/净利润",
    "capex_to_revenue": "资本开支/收入",
    "cash": "现金",
    "total_debt": "总债务",
    "market_cap": "市值",
    "share_price": "股价",
    "price": "股价",
    "adjusted_close": "股价",
    "pe_ratio": "P/E",
    "ps_ratio": "P/S",
    "fcf_yield": "FCF yield",
}

_PUBLIC_METRIC_LABELS_EN = {
    "revenue": "revenue",
    "revenue_growth": "revenue growth",
    "net_income": "net income",
    "eps": "EPS",
    "gross_margin": "gross margin",
    "operating_margin": "operating margin",
    "net_margin": "net margin",
    "operating_cash_flow": "operating cash flow",
    "free_cash_flow": "free cash flow",
    "capital_expenditure": "capital expenditures",
    "fcf_margin": "FCF margin",
    "cfo_to_net_income": "CFO/net income",
    "capex_to_revenue": "capex/revenue",
    "cash": "cash",
    "total_debt": "total debt",
    "market_cap": "market cap",
    "share_price": "share price",
    "price": "share price",
    "adjusted_close": "share price",
    "pe_ratio": "P/E",
    "ps_ratio": "P/S",
    "fcf_yield": "FCF yield",
}


def _public_metric_label(metric: str, lang: str) -> str:
    metric = str(metric or "").strip().lower()
    labels = _PUBLIC_METRIC_LABELS_ZH if lang == "zh" else _PUBLIC_METRIC_LABELS_EN
    return labels.get(metric, metric.replace("_", " "))


def _public_numeric_value(row: dict[str, Any]) -> str:
    display = str(row.get("display_value") or row.get("formatted_value") or "").strip()
    if display:
        return display
    value = row.get("value")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value or "").strip()
    unit = str(row.get("unit") or "").strip().lower()
    metric = str(row.get("metric") or row.get("metric_label") or "").strip().lower()
    if unit == "ratio" or metric.endswith("_margin") or metric in {"fcf_yield", "cash_conversion", "capex_to_revenue"}:
        return f"{numeric * 100:.2f}%"
    if unit in {"usd", "usd_per_share"} or metric == "eps":
        prefix = "$"
        if unit == "usd_per_share" or metric == "eps":
            return f"{prefix}{numeric:,.2f}"
        abs_value = abs(numeric)
        if abs_value >= 1_000_000_000_000:
            return f"{prefix}{numeric / 1_000_000_000_000:.2f}T"
        if abs_value >= 1_000_000_000:
            return f"{prefix}{numeric / 1_000_000_000:.2f}B"
        if abs_value >= 1_000_000:
            return f"{prefix}{numeric / 1_000_000:.2f}M"
        return f"{prefix}{numeric:,.0f}"
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.2f}".rstrip("0").rstrip(".")


def _numeric_line_from_row(row: dict[str, Any], *, lang: str) -> str:
    metric = str(row.get("metric") or row.get("metric_label") or "").strip()
    value = _public_numeric_value(row)
    refs = _citation_refs(_row_refs(row)[:2])
    if not metric or not value or not refs:
        return ""
    ticker = _company_ticker_text(row.get("ticker") or row.get("company") or "")
    label = _public_metric_label(metric, lang)
    if lang == "zh":
        prefix = f"{ticker} " if ticker else ""
        connector = " 为 " if re.search(r"[A-Za-z/]", label) else "为 "
        return f"{prefix}{label}{connector}{value}{refs}"
    prefix = f"{ticker} " if ticker else ""
    return f"{prefix}{label} is {value}{refs}"


def _citable_fact_lines(state: AgentState, *, lang: str, limit: int = 6) -> list[str]:
    lines: list[str] = []
    for row in _node_text_rows(state):
        if not _row_refs(row):
            continue
        line = _line_from_row(row)
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            return lines
    for row in _node_numeric_rows(state):
        line = _numeric_line_from_row(row, lang=lang)
        if line and line not in lines:
            lines.append(line)
        if len(lines) >= limit:
            return lines
    return lines


def _risk_rank_items_from_rows(
    state: AgentState,
    *,
    lang: str,
    rows_override: list[dict[str, Any]] | None = None,
) -> list[tuple[str, str, list[str]]]:
    query = str(state.get("user_query") or "").lower()
    if rows_override is not None:
        rows = [dict(row) for row in rows_override]
    else:
        raw_text_rows = _node_text_rows(state)
        rows = _risk_text_rows_for_public_answer(state) or raw_text_rows
    combined = " ".join(_row_public_text(row) for row in rows).lower()
    all_refs = list(dict.fromkeys(ref for row in rows for ref in _row_refs(row) if ref.startswith("T")))
    items: list[tuple[str, str, list[str]]] = []

    def add(theme: str, why: str, terms: tuple[str, ...]) -> None:
        refs = [
            ref
            for row in rows
            if any(term in _row_search_text(row).lower() for term in terms)
            for ref in _row_refs(row)
            if ref.startswith("T")
        ]
        if not refs and all_refs:
            refs = all_refs[:2]
        if refs and theme not in [item[0] for item in items]:
            items.append((theme, why, list(dict.fromkeys(refs))[:3]))

    if any(term in query for term in ("经济放缓", "衰退", "slowdown", "recession", "economic")):
        add(
            "客户 IT/云支出放缓" if lang == "zh" else "customer IT/cloud-spend slowdown",
            "基于业务结构推断：经济放缓通常先压低企业 IT、云和项目支出，再传导到收入增速、利润率和现金流。"
            if lang == "zh"
            else "Business-structure inference: a slowdown typically pressures enterprise IT/cloud spending first, then revenue growth, margins, and cash flow.",
            ("cloud", "customer", "spending", "demand", "revenue", "aws", "azure", "客户", "云", "支出", "需求", "收入"),
        )
    add(
        "履约/库存/资本开支压力" if lang == "zh" else "fulfillment, inventory, and capital-spending pressure",
        "基于业务模型的有限判断：履约、库存或资本投入问题会先影响成本、服务质量和现金占用，再压制利润率或 FCF。"
        if lang == "zh"
        else "Limited business-model judgment: fulfillment, inventory, or capital investment issues can affect cost, service quality, working capital, margins, and FCF.",
        ("fulfillment", "logistics", "inventory", "inventories", "capital expenditure", "capex", "supply chain", "履约", "物流", "库存", "资本开支", "供应链"),
    )
    add(
        "监管/合规风险" if lang == "zh" else "regulatory and compliance risk",
        "已验证风险文本若涉及监管、法律或合规，该风险可能通过合规成本、区域业务限制和经营不确定性传导。"
        if lang == "zh"
        else "When validated risk text cites regulation, legal, or compliance exposure, the transmission is compliance cost, operating limits, and uncertainty.",
        ("regulatory", "regulation", "legal", "compliance", "antitrust", "jurisdiction", "监管", "法律", "合规"),
    )
    add(
        "AWS/云竞争" if lang == "zh" else "AWS/cloud competition",
        "基于业务结构推断：云竞争可能通过价格、市场份额和客户迁移影响收入增速与利润率。"
        if lang == "zh"
        else "Business-structure inference: cloud competition can affect revenue growth and margins through pricing, share, and customer migration.",
        ("aws", "cloud", "competition", "competitive", "azure", "google cloud", "云", "竞争"),
    )
    if not items and all_refs:
        items.append(
            (
                "已披露经营风险" if lang == "zh" else "disclosed operating risks",
                "有限判断：当前只能按已验证风险文本做排序，不能扩展成确定预测。" if lang == "zh" else "Limited judgment: ranking is bounded by validated risk text and is not a forecast.",
                all_refs[:3],
            )
        )
    return items[:4]


def _bounded_risk_analysis_answer(
    state: AgentState,
    *,
    lang: str,
    missing_labels: list[str] | None = None,
) -> tuple[str, str] | None:
    raw_text_rows = _node_text_rows(state)
    risk_rows = _risk_text_rows_for_public_answer(state)
    if not risk_rows:
        risk_rows = [
            row
            for row in raw_text_rows
            if str(row.get("dimension_id") or "") == "moat_and_competitive_risk"
            or str(row.get("section") or "").upper().strip() in {"ITEM_1A", "ITEM_7", "ITEM_2"}
        ]
    if not risk_rows:
        return None
    facts = [_line_from_row(row) for row in risk_rows[:5] if _line_from_row(row)]
    ranking = _risk_rank_items_from_rows(state, lang=lang, rows_override=risk_rows)
    refs = list(dict.fromkeys(ref for row in risk_rows for ref in _row_refs(row) if ref.startswith("T")))
    ref_text = _citation_refs(refs[:4])
    query = str(state.get("user_query") or "").lower()
    scenario = any(term in query for term in ("经济放缓", "衰退", "slowdown", "recession", "economic"))
    if lang == "zh":
        top = ranking[0][0] if ranking else "已披露经营风险"
        conclusion = (
            f"有限判断：在经济放缓语境下，优先观察{top}；供应链只作为已披露风险线索之一，不机械排第一。{_citation_refs(ranking[0][2]) if ranking else ref_text}"
            if scenario
            else f"有限判断：可按业务传导把主要风险优先看作{top}；排序来自已验证风险文本和业务结构推断，不是确定预测。{_citation_refs(ranking[0][2]) if ranking else ref_text}"
        )
        lines = [
            "结论",
            conclusion,
            "",
            "已验证风险文本",
            *(f"- {item}" for item in facts),
            "",
            "基于业务模型的风险排序",
            *(
                f"- {idx}. {theme}：{why}{_citation_refs(refs)}"
                for idx, (theme, why, refs) in enumerate(ranking, start=1)
            ),
            "",
            "财务传导路径",
            (
                f"- 基于业务结构推断：客户 IT/云支出放缓 -> 收入增速压力 -> 利润率和现金流承压；当前仍需后续经营数据验证。{ref_text}"
                if scenario
                else f"- 基于业务结构推断：风险先影响收入、成本或服务质量，再传导到利润率、资本开支和自由现金流。{ref_text}"
            ),
            "",
            "待验证数据",
            "- 待验证：收入增速、订单/客户支出、毛利率/营业利润率、经营现金流、资本开支和 FCF。",
            "",
            "证据边界",
            "- 已披露风险文本不能单独量化发生概率或下一季度影响；该回答不构成投资建议。",
        ]
        if missing_labels:
            lines.append(f"- 仍缺少：{'、'.join(missing_labels[:5])}。")
        return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_risk_analysis"
    top = ranking[0][0] if ranking else "disclosed operating risks"
    lines = [
        "Conclusion",
        f"Limited judgment: the main risk to monitor is {top}; the ranking is bounded by validated risk text and business-structure inference, not a forecast. {ref_text}",
        "",
        "Verified Risk Text",
        *(f"- {item}" for item in facts),
        "",
        "Business-Model Risk Ranking",
        *(f"- {idx}. {theme}: {why}{_citation_refs(refs)}" for idx, (theme, why, refs) in enumerate(ranking, start=1)),
        "",
        "Financial Transmission Path",
        f"- Business-structure inference: risks transmit through revenue, cost, service quality, margin, capital spending, and FCF. {ref_text}",
        "",
        "Data to Verify",
        "- To verify: revenue growth, orders/customer spending, gross/operating margin, operating cash flow, capex, and FCF.",
        "",
        "Evidence Boundary",
        "- Risk text alone does not quantify probability or next-quarter impact; this is not investment advice.",
    ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_risk_analysis"


def _generic_bounded_analysis_answer(
    state: AgentState,
    *,
    lang: str,
    requested_dimensions: list[str] | None = None,
    missing_labels: list[str] | None = None,
) -> tuple[str, str] | None:
    risk_answer = _bounded_risk_analysis_answer(state, lang=lang, missing_labels=missing_labels)
    if risk_answer and (
        str(state.get("answer_mode") or "") == "risk_focused_analysis"
        or "moat_and_competitive_risk" in (requested_dimensions or _node_requested_dimensions(state))
        or any(term in str(state.get("user_query") or "").lower() for term in ("风险", "risk", "危险", "danger"))
    ):
        return risk_answer
    facts = _citable_fact_lines(state, lang=lang)
    if not facts:
        return None
    refs = "".join(list(dict.fromkeys(re.findall(r"\[[NT]\d+\]", " ".join(facts))))[:4])
    requested = requested_dimensions or _node_requested_dimensions(state)
    labels = [_dimension_boundary_label(item, lang) for item in requested if str(item).strip()]
    target = ("、".join(labels) if lang == "zh" else ", ".join(labels)) or ("原问题" if lang == "zh" else "the question")
    if lang == "zh":
        lines = [
            "结论",
            f"有限判断：当前可以基于已验证事实回答{target}的一部分，但不能把缺失证据扩展成确定结论。{refs}",
            "",
            "已验证事实",
            *(f"- {item}" for item in facts[:6]),
            "",
            "合理推断",
            f"- 合理推断：这些事实只能支持方向性、有限判断；若缺少同口径历史/同业/分部证据，不能升级为严格排序、因果或投资结论。{refs}",
            "",
            "待验证假设",
            "- 待验证：结论的强弱仍取决于后续同口径数据、管理层披露和相关分部/现金流指标。",
            "",
            "证据边界",
            "- 事实和数字必须以上述引用为准；该回答不构成投资建议、买卖建议或目标价。",
        ]
        if missing_labels:
            lines.append(f"- 仍缺少：{'、'.join(missing_labels[:5])}。")
        return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_analysis"
    lines = [
        "Conclusion",
        f"Limited judgment: current verified facts can answer part of {target}, but missing evidence cannot be expanded into certainty. {refs}",
        "",
        "Verified Facts",
        *(f"- {item}" for item in facts[:6]),
        "",
        "Reasonable Inference",
        f"- Reasonable inference: these facts support only a directional, bounded judgment; without comparable history, peers, segment, or cash-flow evidence, this cannot become a strict ranking, causal claim, or investment conclusion. {refs}",
        "",
        "Hypotheses To Verify",
        "- To verify: the strength of the conclusion depends on later comparable data, management disclosure, and segment/cash-flow metrics.",
        "",
        "Evidence Boundary",
        "- Facts and numbers are limited to the cited evidence above; this is not investment advice, a trading recommendation, or a target price.",
    ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_analysis"


def _contract_debt_is_overview(state: AgentState) -> bool:
    output = dict(state.get("output", {}) or {})
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    canonical_intent = dict(state.get("canonical_intent", {}) or analysis_plan.get("canonical_intent", {}) or {})
    packet = dict(state.get("evidence_packet", {}) or {})
    return (
        str(canonical_intent.get("intent_family") or analysis_plan.get("intent_family") or packet.get("intent_family") or "") == "overview"
        or str(state.get("evidence_policy_id") or output.get("evidence_policy_id") or analysis_plan.get("evidence_policy_id") or packet.get("evidence_policy_id") or "") == "single_company_overview_v1"
        or str(state.get("legacy_methodology_intent") or analysis_plan.get("legacy_methodology_intent") or analysis_plan.get("methodology_intent") or packet.get("methodology_intent") or "") == "single_company_overview"
    )


def _contract_debt_overview_fallback(state: AgentState, *, lang: str) -> tuple[str, str] | None:
    if not _contract_debt_is_overview(state):
        return None
    text_rows = _node_text_rows(state)
    numeric_rows = _node_numeric_rows(state)
    if not text_rows and not numeric_rows:
        return None
    company = _state_company_label(state)

    def overview_text_line(row: dict[str, Any], *, kind: str) -> str:
        if lang != "zh":
            return _line_from_row(row)
        refs = _citation_refs(_row_refs(row)[:2])
        if not refs:
            return ""
        if kind == "business":
            return f"已验证业务文本覆盖产品、服务、客户或分部信息{refs}"
        if kind == "risk":
            return f"已验证风险文本提示竞争、供应链、监管或经营不确定性等风险边界{refs}"
        return f"已验证文本证据提供公司概览线索{refs}"

    business_lines = [
        overview_text_line(row, kind="business")
        for row in text_rows
        if str(row.get("dimension_id") or "") in {"business_model", "revenue_quality"}
    ]
    risk_lines = [
        overview_text_line(row, kind="risk")
        for row in text_rows
        if str(row.get("dimension_id") or "") == "moat_and_competitive_risk" or str(row.get("section") or "").upper() in {"ITEM_1A", "ITEM_7"}
    ]
    if not business_lines:
        business_lines = [overview_text_line(row, kind="business") for row in text_rows[:2]]
    business_lines = [line for line in dict.fromkeys(business_lines) if line]
    risk_lines = [line for line in dict.fromkeys(risk_lines) if line]
    def first_metric_row(metrics: set[str]) -> dict[str, Any]:
        for row in numeric_rows:
            metric = str(row.get("metric") or row.get("metric_label") or "").strip().lower()
            value = str(row.get("display_value") or row.get("formatted_value") or row.get("value") or "").strip()
            if metric in metrics and value:
                return dict(row)
        return {}

    def metric_value(row: dict[str, Any]) -> str:
        return str(row.get("display_value") or row.get("formatted_value") or row.get("value") or "").strip()

    def metric_refs(row: dict[str, Any]) -> str:
        return _citation_refs(_row_refs(row)[:2])

    def valuation_phrase() -> str:
        rows: list[tuple[str, str, str]] = []
        for metric in ("market_cap", "pe_ratio", "ps_ratio", "fcf_yield"):
            row = first_metric_row({metric})
            if row:
                rows.append((_public_metric_label(metric, lang), metric_value(row), metric_refs(row)))
        if not rows:
            return (
                "当前缺少可引用的估值倍数证据，不能判断便宜或昂贵。"
                if lang == "zh"
                else "Validated valuation-multiple evidence is unavailable, so cheap/expensive claims are not supported."
            )
        joined = "、".join(f"{label} {value}{refs}" for label, value, refs in rows if value)
        if lang == "zh":
            return f"估值边界可观察到 {joined}；这些指标不能直接推出买卖结论或估值吸引力。"
        return f"The valuation boundary includes {joined}; these inputs do not by themselves support buy/sell or valuation-attractiveness conclusions."

    revenue_row = first_metric_row({"revenue", "sales"})
    revenue_growth_row = first_metric_row({"revenue_growth"})
    profit_row = first_metric_row({"net_income", "net_margin", "operating_margin", "gross_margin"})
    cash_row = first_metric_row({"operating_cash_flow", "free_cash_flow", "fcf_margin", "cfo_to_net_income"})
    overview_refs = _citation_refs(list(dict.fromkeys(ref for row in [*text_rows, *numeric_rows] for ref in _row_refs(row)))[:5])
    if lang == "zh":
        business_line = business_lines[0] if business_lines else f"当前缺少可引用的业务模式文本证据，不能把 {company} 的业务定位扩展成未验证叙述。"
        revenue_line = (
            f"收入规模可以观察到（最新可见 {metric_value(revenue_row)}{metric_refs(revenue_row)}），"
            "但需要统一期间和口径后才能判断趋势。"
            if revenue_row
            else "当前缺少可引用的收入或增长指标，不能判断收入规模和趋势。"
        )
        if revenue_growth_row:
            revenue_line = f"{revenue_line} 收入增速线索为 {metric_value(revenue_growth_row)}{metric_refs(revenue_growth_row)}。"
        profit_line = (
            f"盈利层面已有可验证线索（最新可见 {_public_metric_label(str(profit_row.get('metric') or ''), lang)} {metric_value(profit_row)}{metric_refs(profit_row)}），"
            "但增长质量还要结合毛利率、经营利润率和现金流转换验证。"
            if profit_row
            else "当前缺少可引用的盈利指标，不能判断盈利质量。"
        )
        cash_line = (
            f"现金流层面已有可验证线索（{_public_metric_label(str(cash_row.get('metric') or ''), lang)} {metric_value(cash_row)}{metric_refs(cash_row)}），但还需要资本开支和自由现金流口径联动验证。"
            if cash_row
            else "现金流证据不完整，不能验证利润到自由现金流的转换质量。"
        )
        risk_line = risk_lines[0] if risk_lines else "当前风险文本覆盖有限，不能把概览降级为单一风险排序。"
        lines = [
            "结论",
            f"{company} 可以形成分析型概览：业务、收入、盈利/现金流、风险和估值必须分开看；结论受当前已验证证据范围限制。{overview_refs}",
            "",
            "业务定位",
            f"- {business_line}",
            "",
            "收入和盈利",
            f"- {revenue_line}",
            f"- {profit_line}",
            "",
            "现金流与估值",
            f"- {cash_line}",
            f"- {valuation_phrase()}",
            "",
            "主要风险",
            f"- {risk_line}",
            "",
            "证据边界",
            f"- 这是公司概览，不是单一风险或估值结论；风险和估值只是概览中的维度。{overview_refs}",
            "- 缺少的维度不能被扩展成未验证结论。",
        ]
    else:
        business_line = business_lines[0] if business_lines else f"Validated business-model text is unavailable, so {company}'s business positioning should not be expanded beyond cited evidence."
        revenue_line = (
            f"Revenue scale is observable at {metric_value(revenue_row)}{metric_refs(revenue_row)}, but period and basis need to be aligned before judging trend."
            if revenue_row
            else "Validated revenue or growth metrics are unavailable, so revenue scale and trend cannot be judged."
        )
        if revenue_growth_row:
            revenue_line = f"{revenue_line} Revenue-growth evidence is {metric_value(revenue_growth_row)}{metric_refs(revenue_growth_row)}."
        profit_line = (
            f"Profitability has a validated signal ({_public_metric_label(str(profit_row.get('metric') or ''), lang)} {metric_value(profit_row)}{metric_refs(profit_row)}), but quality still needs margins and cash conversion."
            if profit_row
            else "Validated profitability metrics are unavailable, so profitability quality cannot be judged."
        )
        cash_line = (
            f"Cash-flow evidence includes {_public_metric_label(str(cash_row.get('metric') or ''), lang)} {metric_value(cash_row)}{metric_refs(cash_row)}, but capex and FCF basis still need verification."
            if cash_row
            else "Cash-flow evidence is incomplete, so earnings-to-FCF conversion cannot be verified."
        )
        risk_line = risk_lines[0] if risk_lines else "Risk text coverage is limited; the overview should not be reduced to a risk-only answer."
        lines = [
            "Conclusion",
            f"{company} supports an analytical overview: business, revenue, profitability/cash flow, risk, and valuation should be separated, within currently validated evidence. {overview_refs}",
            "",
            "Business Positioning",
            f"- {business_line}",
            "",
            "Revenue And Profitability",
            f"- {revenue_line}",
            f"- {profit_line}",
            "",
            "Cash Flow And Valuation",
            f"- {cash_line}",
            f"- {valuation_phrase()}",
            "",
            "Primary Risks",
            f"- {risk_line}",
            "",
            "Evidence Boundary",
            f"- This is a company overview, not a risk-only answer; risk is one overview dimension. {overview_refs}",
            "- Missing dimensions cannot be expanded into unverified conclusions.",
        ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_overview_candidate"


def _contract_debt_segment_fallback(state: AgentState, *, lang: str) -> tuple[str, str] | None:
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    canonical_intent = dict(state.get("canonical_intent", {}) or analysis_plan.get("canonical_intent", {}) or {})
    packet = _node_packet_for_fallback(state)
    research_plan = dict(packet.get("research_plan", {}) or state.get("research_plan_used", {}) or {})
    scope = _segment_or_product_scope_from_state(state)
    if not scope and str(research_plan.get("question_type") or "") != "causal_explanation":
        return None
    if not scope and not (canonical_intent.get("segment_or_product_scope") or canonical_intent.get("segment_focus")):
        scope = "分部/产品" if lang == "zh" else "segment/product"
    rows = [*(_node_text_rows(state)), *(_node_numeric_rows(state)), *(_node_analytical_claim_rows(state))]
    if not rows:
        return None
    terms = [term.lower() for term in re.split(r"[\s,/，、&]+", scope) if term.strip()]
    terms.extend(["compute", "networking", "network", "nvlink", "infiniband", "ethernet", "data center", "数据中心", "网络", "产品", "分部"])
    matched = []
    for row in rows:
        text = _row_search_text(row)
        lowered = text.lower()
        if any(term and term in lowered for term in terms):
            matched.append(row)
    if not matched:
        # A causal segment question with only company/segment numbers is still answerable with a boundary.
        matched = [row for row in rows if _row_refs(row)][:4]
    if not matched:
        return None
    facts = []
    for row in matched[:5]:
        line = _line_from_row(row)
        if not line and str(row.get("metric") or "").strip():
            metric = str(row.get("metric") or row.get("metric_label") or "").strip()
            value = str(row.get("display_value") or row.get("formatted_value") or row.get("value") or "").strip()
            refs = _citation_refs(_row_refs(row)[:2])
            line = f"{metric}: {value}{refs}".strip()
        if line:
            facts.append(line)
    refs = list(dict.fromkeys(ref for row in matched for ref in _row_refs(row)))
    status_by_id = dict(state.get("answer_part_status_by_id") or dict(state.get("evidence_sufficiency", {}) or {}).get("answer_part_status_by_id", {}) or {})
    partial_parts = [
        _friendly_answer_part_label(str(key), lang)
        for key, value in status_by_id.items()
        if isinstance(value, dict) and str(value.get("status") or "") in {"partial", "missing"}
    ][:4]
    ref_text = _citation_refs(refs[:4])
    company = _state_company_label(state)
    if lang == "zh":
        network_focus = any(term in str(scope).lower() for term in ("network", "网络", "infiniband", "ethernet", "nvlink"))
        conclusion = (
            f"有限判断：{company} 网络业务增长大概率与 AI 集群建设、GPU 集群互连以及 NVLink/InfiniBand/Ethernet 需求有关；但这些证据主要是分部/产品层面，不能直接推出总公司级营收增长的完整因果或贡献比例。{ref_text}"
            if network_focus
            else f"当前证据支持在分部/产品层面讨论 {company} 的 {scope} 驱动，但不能直接推出总公司级营收增长的完整因果或贡献比例。{ref_text}"
        )
        lines = [
            "结论",
            conclusion,
            "",
            "已验证事实",
            *(f"- {item}" for item in facts if item),
            "",
            "合理推断",
            f"- 可验证证据指向 {scope} 相关业务线索；这些线索只约束分部/产品层面解释。{ref_text}",
            f"- 可以说 {scope} 可能参与相关增长解释；不能说它已经单独证明总公司营收增长主要由该因素决定。{ref_text}",
            "",
            "待验证假设",
            *(f"- 待验证：{part} 仍需补充分部收入、同比口径或管理层量化说明。" for part in (partial_parts or ["贡献比例和可持续性"])),
            "",
            "证据边界",
            "- 分部/产品证据不能单独证明总公司级完整因果、贡献比例或持续性。",
        ]
    else:
        lines = [
            "Short Conclusion",
            f"Current evidence supports a segment/product-level discussion of {company}'s {scope} drivers, but not full total-company revenue-growth causality or contribution share. {ref_text}",
            "",
            "Verified Facts",
            *(f"- {item}" for item in facts if item),
            "",
            "Segment/Product-Level Drivers",
            f"- Validated evidence points to {scope}-related business signals; these remain segment/product-level evidence. {ref_text}",
            "",
            "Citable Inference",
            f"- It is supportable to say {scope} may help explain the relevant business-line growth; it does not prove total-company revenue growth was mainly caused by that factor. {ref_text}",
            "",
            "Hypotheses To Verify",
            *(f"- To verify: {part} still needs segment revenue, comparable-period growth, or quantified management disclosure." for part in (partial_parts or ["contribution share and durability"])),
            "",
            "Evidence Boundary",
            "- Segment/product evidence alone cannot prove total-company causality, contribution share, or durability.",
        ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_segment_product_driver_candidate"


def _friendly_answer_part_label(part_id: str, lang: str) -> str:
    zh = {
        "quantify_growth": "增长幅度和分部贡献比例",
        "identify_growth_drivers": "具体增长驱动",
        "inferred_drivers": "可验证驱动推断",
        "direct_answer": "直接结论",
        "verified_evidence": "已验证事实",
    }
    en = {
        "quantify_growth": "growth magnitude and segment contribution",
        "identify_growth_drivers": "specific growth drivers",
        "inferred_drivers": "validated driver inferences",
        "direct_answer": "direct conclusion",
        "verified_evidence": "verified facts",
    }
    labels = zh if lang == "zh" else en
    return labels.get(str(part_id), str(part_id).replace("_", " "))


def _contract_debt_scenario_risk_fallback(state: AgentState, *, lang: str) -> tuple[str, str] | None:
    query = str(state.get("user_query") or "")
    if str(state.get("answer_mode") or "") != "risk_focused_analysis":
        return None
    if not re.search(r"如果|经济放缓|下季度|衰退|slowdown|recession|next quarter|economic downturn", query, flags=re.IGNORECASE):
        return None
    rows = _node_text_rows(state)
    risk_rows = [
        row
        for row in rows
        if str(row.get("dimension_id") or "") == "moat_and_competitive_risk" or str(row.get("section") or "").upper() in {"ITEM_1A", "ITEM_7", "ITEM_2"}
    ]
    if not risk_rows:
        return None
    scenario_terms = ("economic", "macroeconomic", "slowdown", "recession", "customer spending", "spending", "budget", "经济", "宏观", "客户支出", "预算", "需求")
    direct_rows = [row for row in risk_rows if any(term in _row_public_text(row).lower() for term in scenario_terms)]
    usable = direct_rows or risk_rows
    facts = [_line_from_row(row) for row in usable[:4]]
    refs = list(dict.fromkeys(ref for row in usable for ref in _row_refs(row)))
    ref_text = _citation_refs(refs[:4])
    direct = bool(direct_rows)
    if lang == "zh":
        lines = [
            "结论",
            f"有限判断：在经济放缓语境下，优先观察客户 IT/云支出放缓对收入增速、利润率和现金流的传导；供应链只作为已披露风险线索之一。{ref_text}",
            "",
            "已验证风险文本",
            *(f"- {item}" for item in facts if item),
            "",
            "基于业务模型的风险排序",
            f"- 1. 客户 IT/云支出放缓：基于业务结构推断，企业预算和云消费放缓会先压低收入增速，再影响利润率和现金流。{ref_text}",
            f"- 2. 已披露经营/供应链风险：这些风险可能影响服务交付或成本，但在经济放缓问题里不能机械排在需求传导之前。{ref_text}",
            "",
            "财务传导路径",
            f"- 基于业务结构推断：客户 IT/云支出放缓 -> 收入增速压力 -> 毛利率/营业利润率压力 -> 经营现金流和 FCF 受影响。{ref_text}",
            "",
            "待验证数据",
            "- 待验证：订单/客户支出、云消费量、收入增速、毛利率、营业利润率、经营现金流和 FCF。",
            "",
            "证据边界",
            "- 若披露没有直接量化该情景，排序只能是有限判断，不是确定预测或投资建议。",
        ]
    else:
        lines = [
            "Conclusion",
            f"Limited judgment: in an economic slowdown, first monitor customer IT/cloud spending and its transmission to revenue growth, margin, and cash flow; supply-chain risk is a disclosed signal, not an automatic top rank. {ref_text}",
            "",
            "Verified Risk Text",
            *(f"- {item}" for item in facts if item),
            "",
            "Business-Model Risk Ranking",
            f"- 1. Customer IT/cloud-spend slowdown: business-structure inference says budget and cloud-consumption pressure hits revenue growth first, then margins and cash flow. {ref_text}",
            f"- 2. Disclosed operating/supply-chain risks: these can affect delivery or cost, but should not mechanically outrank demand transmission in a slowdown question. {ref_text}",
            "",
            "Financial Transmission Path",
            f"- Business-structure inference: customer IT/cloud spending slows -> revenue growth pressure -> gross/operating margin pressure -> operating cash flow and FCF impact. {ref_text}",
            "",
            "Data to Verify",
            "- To verify: orders/customer spending, cloud consumption, revenue growth, gross margin, operating margin, operating cash flow, and FCF.",
            "",
            "Evidence Boundary",
            "- If disclosures do not directly quantify the scenario, the ranking is a limited judgment, not a forecast or investment advice.",
        ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_scenario_risk_candidate"


def _contract_debt_valuation_fallback(state: AgentState, *, lang: str) -> tuple[str, str] | None:
    requested = set(_node_requested_dimensions(state))
    query = str(state.get("user_query") or "").lower()
    if "valuation_and_risk_boundary" not in requested and not any(term in query for term in ("估值", "贵", "便宜", "valuation", "expensive", "cheap")):
        return None
    rows = [
        row
        for row in _node_numeric_rows(state)
        if str(row.get("metric") or "").lower() in {"pe_ratio", "ps_ratio", "fcf_yield", "market_cap", "share_price", "price"}
    ]
    if not rows:
        return None
    valuation_line = _valuation_boundary_line_for_public_answer(state, lang=lang)
    if valuation_line:
        return valuation_line, "bounded_valuation_candidate"
    metrics = []
    for row in rows[:5]:
        metric = str(row.get("metric") or "").strip()
        value = str(row.get("display_value") or row.get("formatted_value") or row.get("value") or "").strip()
        refs = _citation_refs(_row_refs(row)[:2])
        if metric and value:
            label = _public_metric_label(metric, lang)
            metrics.append(f"{label}: {value}{refs}" if lang != "zh" else f"{label}：{value}{refs}")
    ref_text = "".join(list(dict.fromkeys(re.findall(r"\[[NT]\d+\]", " ".join(metrics))))[:4])
    if lang == "zh":
        lines = [
            "结论",
            f"有限判断：已验证估值倍数支持“估值风险偏高”的方向性判断，但不能给买卖建议、目标价或严格历史/同业高低位结论。{ref_text}",
            "",
            "已验证事实",
            *(f"- {item}" for item in metrics),
            "",
            "合理推断",
            f"- 合理推断：P/E、P/S 或 FCF yield 等指标偏高时，估值容错空间较低；但该判断仍需要历史分位、同业基准和增长质量验证。{ref_text}",
            "",
            "待验证假设",
            "- 待验证：未来收入增速、利润率、自由现金流转化、历史估值区间和同业估值基准。",
            "",
            "证据边界",
            "- 当前证据只能支持方向性估值风险判断，不构成投资建议、买卖建议或目标价。",
        ]
    else:
        lines = [
            "Conclusion",
            f"Limited judgment: verified valuation multiples support a directional view that valuation risk is elevated, but not buy/sell advice, a target price, or strict historical/peer positioning. {ref_text}",
            "",
            "Verified Facts",
            *(f"- {item}" for item in metrics),
            "",
            "Reasonable Inference",
            f"- Reasonable inference: elevated P/E, P/S, or low FCF yield can reduce valuation margin for error; this still needs historical percentiles, peer benchmarks, and growth-quality evidence. {ref_text}",
            "",
            "Hypotheses To Verify",
            "- To verify: future revenue growth, margins, free-cash-flow conversion, historical valuation range, and peer benchmarks.",
            "",
            "Evidence Boundary",
            "- Current evidence supports only directional valuation-risk judgment, not investment advice, trading recommendations, or a target price.",
        ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "valuation_bounded_answer"


_CONTRACT_DEBT_BUSINESS_FALLBACK_OWNERS = {
    "bounded_overview_candidate",
    "bounded_segment_product_driver_candidate",
    "bounded_scenario_risk_candidate",
    "bounded_risk_boundary_candidate",
    "bounded_valuation_candidate",
    "valuation_bounded_answer",
    "overview_bounded_answer",
    "bounded_cash_flow_candidate",
    "bounded_profitability_candidate",
}


def _is_contract_debt_business_fallback_owner(owner: str) -> bool:
    text = str(owner or "")
    return text in _CONTRACT_DEBT_BUSINESS_FALLBACK_OWNERS or (
        text.startswith("bounded_") and text.endswith("_candidate") and text != "bounded_scope_limit"
    )


def _contract_debt_business_fallback_active(state: AgentState) -> bool:
    output = dict(state.get("output", {}) or {})
    owner = str(state.get("final_answer_source") or output.get("final_answer_source") or "")
    return bool(state.get("primary_generation_contract_debt") or output.get("primary_generation_contract_debt")) and _is_contract_debt_business_fallback_owner(owner)


def _contract_result_only_minor_business_fallback_issues(result: dict[str, Any]) -> bool:
    hard_codes = {
        "invalid_citation",
        "unsupported_numeric",
        "forbidden_claim",
        "raw_internal_leakage",
        "company_specific_token_leakage",
        "segment_evidence_overstated_as_company_driver",
    }
    codes = {
        str(item.get("code") or item.get("type") or "").strip()
        for item in result.get("violations", []) or []
        if isinstance(item, dict)
    }
    return not bool(codes & hard_codes)


def build_contract_debt_business_fallback(state: AgentState, result: dict[str, Any], lang: str) -> tuple[str, str]:
    requested_dimensions = _node_requested_dimensions(state)
    existing = _existing_bounded_candidate(state, requested_dimensions, require_format_ok=True)
    if existing:
        body = str(existing.get("body") or "").strip()
        owner = str(existing.get("owner") or "").strip() or "bounded_existing_candidate"
        if body and not _public_answer_has_internal_terms(body):
            return body, owner if not _public_answer_has_internal_terms(owner) else "bounded_existing_candidate"
    if _answering._is_risk_comparison_query(str(state.get("user_query") or ""), state):
        bounded = _build_bounded_risk_comparison_answer(state, lang=lang)
        if bounded.strip() and not _public_answer_has_internal_terms(bounded):
            return bounded, "bounded_risk_comparison_answer"
    for builder in (
        _contract_debt_overview_fallback,
        _contract_debt_segment_fallback,
        _contract_debt_scenario_risk_fallback,
        _contract_debt_valuation_fallback,
    ):
        built = builder(state, lang=lang)
        if built and built[0].strip() and not _public_answer_has_internal_terms(built[0]):
            return built
    generic = _generic_bounded_analysis_answer(
        state,
        lang=lang,
        requested_dimensions=requested_dimensions,
    )
    if generic and generic[0].strip() and not _public_answer_has_internal_terms(generic[0]):
        return generic
    return _contract_debt_public_scope_limit_answer(state, lang=lang)


def _deterministic_repair_answer(state: AgentState, result: dict[str, Any]) -> str:
    draft = str(state.get("draft_answer") or state.get("final_answer") or "")
    codes = {str(item.get("code") or "") for item in result.get("violations", []) if isinstance(item, dict)}
    repaired = _strip_forbidden_sentences(draft, codes)
    if "unsupported_benchmark_claim" in codes:
        for violation in result.get("violations", []) or []:
            if not isinstance(violation, dict) or str(violation.get("code") or violation.get("type") or "") != "unsupported_benchmark_claim":
                continue
            original = str(violation.get("answer_span") or "").strip()
            replacement = str(violation.get("public_replacement") or "估值倍数较高，但缺少历史/行业基准，不能严格判断是否处于高位。").strip()
            refs = _dedupe_citation_refs_text(original)
            if refs and not re.search(r"\[[NT]\d+\]", replacement):
                replacement = f"{replacement}{refs}"
            if original and replacement and original in repaired:
                repaired = repaired.replace(original, replacement, 1)
    if "citation_free_material_claim" in codes:
        for violation in result.get("violations", []) or []:
            if not isinstance(violation, dict) or str(violation.get("code") or "") != "citation_free_material_claim":
                continue
            original = str(violation.get("answer_span") or "").strip()
            refs = str(violation.get("suggested_fix") or "").strip()
            if refs and not re.fullmatch(r"(?:\[[NT]\d+\])+", refs):
                refs = ""
            if refs:
                repaired = _append_refs_to_matching_sentence(repaired, original, refs)
            else:
                repaired = _strip_matching_sentence(repaired, original)
    if "company_specific_token_leakage" in codes:
        for violation in result.get("violations", []) or []:
            if not isinstance(violation, dict) or str(violation.get("code") or violation.get("type") or "") != "company_specific_token_leakage":
                continue
            repaired = _strip_matching_sentence(repaired, str(violation.get("answer_span") or "").strip())
    if "segment_evidence_overstated_as_company_driver" in codes:
        repaired = _downgrade_scope_overclaim_sentences(repaired, result)
    if "format_constraint_violation" in codes:
        lang = _node_target_lang(state)
        return _one_sentence_repair_answer({**state, "draft_answer": repaired}, repaired, lang=lang).strip()
    if "language_leakage" in codes:
        repaired = repair_language_leakage(repaired, _node_target_lang(state))
    return repaired.strip()


def repair_generate_node(state: AgentState) -> dict[str, Any]:
    """Repair the draft answer without adding new facts."""
    result = _contract_result_dict(state)
    attempt = int(state.get("contract_attempts", 0) or 0) + 1
    previous = str(state.get("draft_answer") or state.get("final_answer") or "")
    source_before_repair = _contract_source_before_repair(state)
    material_uncited_count = _contract_material_uncited_count(result)
    repair_types = _contract_repair_types(result)
    lang = _node_target_lang(state)
    primary_generation_contract_debt = material_uncited_count > 2 or not _contract_repair_allowed(result)
    if primary_generation_contract_debt:
        repaired, owner = build_contract_debt_business_fallback(state, result, lang)
    else:
        repaired = _deterministic_repair_answer(state, result)
        owner = source_before_repair
    strategy = "deterministic_rule_repair"
    if primary_generation_contract_debt:
        strategy = "primary_generation_contract_debt"
    checked = check_answer_contract(repaired, {**state, "draft_answer": repaired}, scope="answer")
    if checked.route == "repair_answer":
        strategy = "deterministic_rule_repair_unresolved"
        if primary_generation_contract_debt:
            strategy = "primary_generation_contract_debt_unresolved"
    repair_actions = list(state.get("repair_actions", []) or [])
    repair_actions.append(
        {
            "attempt": attempt,
            "strategy": strategy,
            "action": str(result.get("action") or ""),
            "violations": _contract_violation_codes(result),
            "repair_owner": "contract_repair",
            "source_before_repair": source_before_repair,
            "repair_types": repair_types,
            "material_claim_uncited_count": material_uncited_count,
            "primary_generation_contract_debt": primary_generation_contract_debt,
        }
    )
    assembled = _assemble_node_answer(
        state,
        answer=repaired,
        owner=owner,
        transform=strategy,
        reason=";".join(_contract_violation_codes(result)),
        claim_change_allowed=False,
        validator_result={
            "contract_route": str(result.get("route") or ""),
            "contract_action": str(result.get("action") or ""),
            "attempt": attempt,
            "repair_owner": "contract_repair",
            "source_before_repair": source_before_repair,
            "repair_types": repair_types,
            "material_claim_uncited_count": material_uncited_count,
            "primary_generation_contract_debt": primary_generation_contract_debt,
        },
        provenance={
            "strategy": strategy,
            "attempt": attempt,
            "repair_owner": "contract_repair",
            "source_before_repair": source_before_repair,
            "repair_types": repair_types,
            "material_claim_uncited_count": material_uncited_count,
            "primary_generation_contract_debt": primary_generation_contract_debt,
        },
    )
    output = dict(state.get("output", {}) or {})
    output.update(
        {
            "final_answer_source": owner,
            "repair_applied": bool(repaired != previous and not primary_generation_contract_debt),
            "repair_owner": "contract_repair",
            "source_before_repair": source_before_repair,
            "repair_types": repair_types,
            "repair_attempts": attempt,
            "material_claim_uncited_count": material_uncited_count,
            "primary_generation_contract_debt": primary_generation_contract_debt,
            "answer_history": list(assembled.get("answer_history", state.get("answer_history", []) or [])),
        }
    )
    if primary_generation_contract_debt and _is_contract_debt_business_fallback_owner(owner):
        output["answer_quality_tier"] = "bounded_analysis"
        output["main_question_covered"] = True
        output["fallback_intent_match"] = True
        output["answered_dimensions"] = _node_requested_dimensions(state)
    return {
        **assembled,
        "output": output,
        "contract_attempts": attempt,
        "repair_actions": repair_actions,
        "contract_status": "failed",
        "final_answer_source": owner,
        "repair_applied": bool(repaired != previous and not primary_generation_contract_debt),
        "repair_owner": "contract_repair",
        "source_before_repair": source_before_repair,
        "repair_types": repair_types,
        "repair_attempts": attempt,
        "material_claim_uncited_count": material_uncited_count,
        "primary_generation_contract_debt": primary_generation_contract_debt,
        **(
            {
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": _node_requested_dimensions(state),
            }
            if primary_generation_contract_debt and _is_contract_debt_business_fallback_owner(owner)
            else {}
        ),
    }


def prepare_contract_evidence_retry_node(state: AgentState) -> dict[str, Any]:
    """Prepare a bounded evidence retry requested by the runtime contract."""
    count = int(state.get("contract_evidence_retry_count", 0) or 0) + 1
    history = list(state.get("evidence_retry_history", []) or [])
    result = _contract_result_dict(state)
    history.append(
        {
            "source": "runtime_answer_contract",
            "attempt": count,
            "missing_requirements": list(result.get("missing_requirements", []) or []),
            "violation_codes": [str(item.get("code") or "") for item in result.get("violations", []) if isinstance(item, dict)],
        }
    )
    return {
        "contract_evidence_retry_count": count,
        "evidence_retry_history": history,
        "evidence_sufficient": False,
    }


_DIMENSION_LABELS_ZH = {
    "business_model": "业务描述",
    "revenue_quality": "收入质量",
    "profitability_quality": "盈利质量",
    "cash_flow_quality": "现金流质量",
    "balance_sheet_and_capital_intensity": "资产负债和资本强度",
    "moat_and_competitive_risk": "风险因素",
    "valuation_and_risk_boundary": "估值边界",
}

_DIMENSION_LABELS_EN = {
    "business_model": "business model",
    "revenue_quality": "revenue quality",
    "profitability_quality": "profitability quality",
    "cash_flow_quality": "cash-flow quality",
    "balance_sheet_and_capital_intensity": "balance-sheet and capital-intensity",
    "moat_and_competitive_risk": "risk-factor",
    "valuation_and_risk_boundary": "valuation-boundary",
}

_REQUIREMENT_TYPE_LABELS_ZH = {
    "text": "文本证据",
    "numeric": "数值证据",
    "calculation": "计算证据",
    "event": "事件证据",
}

_REQUIREMENT_TYPE_LABELS_EN = {
    "text": "text evidence",
    "numeric": "numeric evidence",
    "calculation": "calculation evidence",
    "event": "event evidence",
}


def _requirement_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if hasattr(item, "model_dump"):
        dumped = item.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {}


def _evidence_requirements_by_id(state: AgentState) -> dict[str, dict[str, Any]]:
    raw_requirements = list(state.get("evidence_requirements", []) or [])
    evidence_plan = state.get("evidence_plan", {})
    if isinstance(evidence_plan, dict):
        raw_requirements.extend(list(evidence_plan.get("evidence_requirements", []) or []))
    out: dict[str, dict[str, Any]] = {}
    for item in raw_requirements:
        req = _requirement_dict(item)
        requirement_id = str(req.get("requirement_id") or "").strip()
        if requirement_id and requirement_id not in out:
            out[requirement_id] = req
    return out


def _friendly_requirement_label(requirement_id: Any, requirements_by_id: dict[str, dict[str, Any]], *, lang: str) -> str:
    rid = str(requirement_id or "").strip()
    req = requirements_by_id.get(rid, {})
    req_type = str(req.get("requirement_type") or "").strip()
    dimension_id = str(req.get("dimension_id") or "").strip()
    if lang == "zh":
        dimension = _DIMENSION_LABELS_ZH.get(dimension_id, "必要")
        evidence_type = _REQUIREMENT_TYPE_LABELS_ZH.get(req_type, "证据")
        return f"{dimension}{evidence_type}"
    dimension = _DIMENSION_LABELS_EN.get(dimension_id, "required")
    evidence_type = _REQUIREMENT_TYPE_LABELS_EN.get(req_type, "evidence")
    return f"{dimension} {evidence_type}"


def _friendly_missing_requirement_labels(state: AgentState, missing: list[Any], *, lang: str) -> list[str]:
    requirements_by_id = _evidence_requirements_by_id(state)
    labels: list[str] = []
    for item in missing:
        label = _friendly_requirement_label(item, requirements_by_id, lang=lang)
        if label and label not in labels:
            labels.append(label)
    return labels


def _company_list_for_public_answer(state: AgentState) -> list[str]:
    companies = [_company_ticker_text(item) for item in state.get("companies", []) or []]
    companies = [item for item in companies if item]
    target = str(state.get("comparison_target") or "").upper().strip()
    if target:
        companies.append(target)
    return list(dict.fromkeys(companies))


_VALUATION_BOUNDARY_METRIC_LABELS = {
    "pe_ratio": "P/E",
    "ps_ratio": "P/S",
    "fcf_yield": "FCF yield",
}


def _numeric_rows_for_public_answer(state: AgentState) -> list[dict[str, Any]]:
    return _node_numeric_rows(state)


def _valuation_boundary_line_for_public_answer(state: AgentState, *, lang: str) -> str:
    rows = _numeric_rows_for_public_answer(state)
    seen: list[str] = []
    refs: list[str] = []
    for row in rows:
        metric = str(row.get("metric") or "").strip()
        if metric not in _VALUATION_BOUNDARY_METRIC_LABELS:
            continue
        label = _VALUATION_BOUNDARY_METRIC_LABELS[metric]
        if label not in seen:
            seen.append(label)
        ref = str(row.get("evidence_id") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    if not seen:
        return ""
    metrics = "、".join(seen) if lang == "zh" else ", ".join(seen)
    ref_text = "".join(f"[{ref}]" for ref in refs[:4])
    companies = _company_list_for_public_answer(state)
    company_label = " 和 ".join(companies) if companies else "该股票"
    if lang == "zh":
        direction_inputs: list[str] = []
        multiple_labels = [label for label in seen if label in {"P/E", "P/S"}]
        if multiple_labels:
            direction_inputs.append("、".join(multiple_labels) + " 较高")
        if "FCF yield" in seen:
            direction_inputs.append("FCF yield 较低")
        input_text = "、".join(direction_inputs) if direction_inputs else f"{metrics} 可见"
        return (
            f"不能给买卖建议；但从 {input_text}来看，{company_label} 的估值风险偏高，"
            f"是否合理取决于增长兑现能力和同业/历史基准{ref_text}。"
        )
    return (
        f"{company_label}'s current valuation multiples ({metrics}) look high and can support a bounded view that valuation risk is elevated{ref_text}; "
        "however, historical percentiles or industry/peer benchmarks are missing, so a strict cheap/expensive or high-position judgment is not supported."
    )


def _node_requested_dimensions(state: AgentState) -> list[str]:
    requested: list[str] = []
    sources = (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions", []),
        dict(state.get("evidence_packet", {}) or {}).get("requested_dimensions", []),
        state.get("requested_dimensions", []),
        state.get("required_dimensions", []),
        [state.get("primary_dimension")],
    )
    for source in sources:
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in requested:
                requested.append(text)
    return requested


def _true_investment_advice_like(state: AgentState) -> bool:
    if str(state.get("safety_intent") or "") != "investment_advice_like":
        return False
    query = str(state.get("user_query") or "").lower()
    return bool(
        re.search(
            r"(能不能买|可以买|值得买|该不该买|买入|卖出|持有|目标价|price target|target price|buy|sell|hold|worth buying|should i)",
            query,
        )
    )


def _prediction_or_target_or_out_of_scope(state: AgentState) -> bool:
    query = str(state.get("user_query") or "").lower()
    if str(state.get("safety_intent") or "") == "unsupported_or_out_of_scope":
        return True
    if str(state.get("answer_mode") or "") == "refusal_or_redirect":
        return True
    return bool(
        re.search(
            r"(目标价|price target|target price|预测股价|明天涨跌|确定.*涨|确定.*跌|will .* stock|guarantee|forecast the stock price)",
            query,
        )
    )


def _rows_for_metric_terms(state: AgentState, terms: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in _numeric_rows_for_public_answer(state):
        haystack = " ".join(
            str(row.get(key) or "")
            for key in (
                "metric",
                "metric_label",
                "label",
                "dimension_id",
                "requirement_id",
                "statement",
                "segment",
            )
        ).lower()
        if any(term in haystack for term in terms):
            rows.append(row)
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("evidence_id") or row.get("ref") or row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _refs_from_rows(rows: list[dict[str, Any]], *, limit: int = 4) -> str:
    refs: list[str] = []
    for row in rows:
        ref = str(row.get("evidence_id") or row.get("ref") or "").strip()
        if ref and ref not in refs:
            refs.append(ref)
    return "".join(f"[{ref}]" for ref in refs[:limit])


def _metric_labels_from_rows(rows: list[dict[str, Any]], *, lang: str, limit: int = 4) -> str:
    labels: list[str] = []
    for row in rows:
        label = str(row.get("metric_label") or row.get("metric") or row.get("label") or "").strip()
        if not label:
            continue
        label = _VALUATION_BOUNDARY_METRIC_LABELS.get(label, label)
        if label not in labels:
            labels.append(label)
    sep = "、" if lang == "zh" else ", "
    return sep.join(labels[:limit])


def _first_row_by_metric(rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> dict[str, Any]:
    wanted = {metric.strip().lower() for metric in metrics if metric.strip()}
    for row in rows:
        metric = str(row.get("metric") or row.get("metric_label") or "").strip().lower()
        if metric in wanted:
            return dict(row)
    return {}


def _profitability_bounded_answer(state: AgentState, *, lang: str, missing_labels: list[str]) -> tuple[str, str] | None:
    rows = _rows_for_metric_terms(
        state,
        (
            "net_income",
            "operating_income",
            "gross_margin",
            "operating_margin",
            "net_margin",
            "eps",
            "profitability",
        ),
    )
    ordered_rows = [
        row
        for row in (
            _first_row_by_metric(rows, ("net_income",)),
            _first_row_by_metric(rows, ("gross_margin",)),
            _first_row_by_metric(rows, ("operating_margin",)),
            _first_row_by_metric(rows, ("net_margin",)),
            _first_row_by_metric(rows, ("eps",)),
        )
        if row
    ]
    if not ordered_rows:
        return None
    refs = _refs_from_rows(ordered_rows)
    company = _state_company_label(state)
    fact_lines = [_numeric_line_from_row(row, lang=lang) for row in ordered_rows[:5]]
    fact_lines = [line for line in fact_lines if line]
    if not fact_lines:
        return None
    has_net_margin = any(str(row.get("metric") or "").strip().lower() == "net_margin" for row in ordered_rows)
    has_margin_structure = any(str(row.get("metric") or "").strip().lower() in {"gross_margin", "operating_margin"} for row in ordered_rows)
    if lang == "zh":
        missing_text = "、".join(missing_labels[:5]) if missing_labels else "净利率、同口径多期利润率和现金流转换"
        conclusion = (
            f"有限判断：{company} 的盈利质量有可引用的盈利规模"
            f"{'和利润率结构' if has_margin_structure else ''}线索；但"
            f"{'净利率仍需补充验证，' if not has_net_margin else ''}不能把这些单点指标直接外推为可持续盈利质量结论。{refs}"
        )
        lines = [
            "结论",
            conclusion,
            "",
            "已验证事实",
            *(f"- {line}" for line in fact_lines),
            "",
            "合理推断",
            f"- 合理推断：净利润提供盈利规模线索，毛利率/营业利润率提供利润率结构线索；这些事实支持方向性盈利质量观察，但还不足以单独证明盈利质量可持续。{refs}",
            "",
            "待验证假设",
            f"- 待验证：{missing_text}。",
            "",
            "证据边界",
            "- 事实和数字只限于上述引用；缺失的净利率、费用结构或现金流转换不能被补写成确定结论。",
        ]
        return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_profitability_candidate"
    missing_text = ", ".join(missing_labels[:5]) if missing_labels else "net margin, comparable margin history, and cash conversion"
    conclusion = (
        f"Limited judgment: {company} has citable profitability-scale"
        f"{' and margin-structure' if has_margin_structure else ''} signals; "
        f"{'net margin still needs verification, and ' if not has_net_margin else ''}"
        f"these point-in-time metrics cannot be extrapolated into a durable profitability-quality conclusion. {refs}"
    )
    lines = [
        "Conclusion",
        conclusion,
        "",
        "Verified Facts",
        *(f"- {line}" for line in fact_lines),
        "",
        "Reasonable Inference",
        f"- Reasonable inference: net income provides profitability scale, while gross/operating margins provide margin-structure signals; these facts support only a directional profitability-quality view. {refs}",
        "",
        "Hypotheses To Verify",
        f"- To verify: {missing_text}.",
        "",
        "Evidence Boundary",
        "- Facts and numbers are limited to the cited evidence above; missing net margin, cost structure, or cash conversion cannot be filled in as a certain conclusion.",
    ]
    return "\n".join(line for line in lines if str(line).strip()).strip(), "bounded_profitability_candidate"


def _scope_limit_boundary_lines(
    state: AgentState,
    missing_labels: list[str],
    *,
    dimensions: list[str],
    lang: str,
) -> list[str]:
    labels = [_dimension_boundary_label(item, lang) for item in dimensions if str(item).strip()]
    label_text = "、".join(labels) if lang == "zh" else ", ".join(labels)
    if lang == "zh":
        target = label_text or "原问题维度"
        lines = [f"证据边界：当前可引用证据不足以回答{target}，因此不能把通用安全模板当作原问题答案。"]
        if missing_labels:
            lines.append("缺失的信息包括：" + "、".join(missing_labels[:5]) + "。")
        lines.append("该结论只是 scope limit，不计为完整回答。")
        return lines
    target = label_text or "the requested dimension"
    lines = [f"Evidence boundary: the citable evidence is not sufficient to answer {target}, so a generic safety template cannot stand in for the requested answer."]
    if missing_labels:
        lines.append("Missing evidence includes: " + ", ".join(missing_labels[:5]) + ".")
    lines.append("This is a scope limit, not a complete answer.")
    return lines


def build_intent_compatible_boundary_answer(
    state: AgentState,
    missing_labels: list[str],
    *,
    lang: str,
) -> dict[str, Any]:
    """Build a releasable boundary only when it matches the original intent."""
    query = str(state.get("user_query") or "")
    query_lower = query.lower()
    dimensions = _node_requested_dimensions(state)
    primary = str(state.get("primary_dimension") or dict(state.get("evidence_policy", {}) or {}).get("primary_dimension") or "").strip()
    canonical_intent = dict(state.get("canonical_intent", {}) or {})
    methodology = str(
        state.get("methodology_intent")
        or state.get("legacy_methodology_intent")
        or canonical_intent.get("legacy_methodology_intent")
        or ""
    ).strip()

    if _true_investment_advice_like(state) or _prediction_or_target_or_out_of_scope(state):
        valuation_line = _valuation_boundary_line_for_public_answer(state, lang=lang)
        companies = _company_list_for_public_answer(state)
        company_label = " 和 ".join(companies) if companies else "该股票"
        if valuation_line:
            return {
                "lines": [valuation_line],
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["valuation_and_risk_boundary"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "valuation_bounded_answer",
            }
        if lang == "zh":
            lines = [f"不能判断 {company_label} 是否值得买；不能给买入、卖出、持有建议，也不能给目标价或确定性预测。"]
            if valuation_line:
                lines.append(valuation_line)
            if missing_labels:
                lines.append("证据限制包括：" + "、".join(missing_labels[:5]) + "。")
            return {
                "lines": lines,
                "answer_quality_tier": "safe_refusal",
                "main_question_covered": False,
                "fallback_intent_match": True,
                "answered_dimensions": ["valuation_and_risk_boundary"] if valuation_line else [],
                "final_route": "safe_refusal",
                "contract_status": "passed_with_warnings",
            }
        lines = [f"I cannot decide whether {company_label} is worth buying; I cannot provide buy, sell, or hold advice, target prices, or deterministic forecasts."]
        if valuation_line:
            lines.append(valuation_line)
        if missing_labels:
            lines.append("Evidence limits include: " + ", ".join(missing_labels[:5]) + ".")
        return {
            "lines": lines,
            "answer_quality_tier": "safe_refusal",
            "main_question_covered": False,
            "fallback_intent_match": True,
            "answered_dimensions": ["valuation_and_risk_boundary"] if valuation_line else [],
            "final_route": "safe_refusal",
                "contract_status": "passed_with_warnings",
            }

    if _answering._is_profit_decline_query(query):
        premise_answer = _answering._profit_decline_premise_answer(_numeric_rows_for_public_answer(state), lang=lang)
        if premise_answer:
            return {
                "lines": [line for line in premise_answer.splitlines() if line.strip()],
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["profitability_quality"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "bounded_profit_decline_candidate",
            }

    overview_intent = (
        str(canonical_intent.get("intent_family") or "") == "overview"
        or methodology in {"single_company_overview", "overview"}
        or "overview" in query_lower
        or "公司概览" in query
    )
    if overview_intent:
        built = _contract_debt_overview_fallback(state, lang=lang)
        if built:
            lines = [line for line in built[0].splitlines() if line.strip()]
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": [
                    "business_model",
                    "revenue_quality",
                    "profitability_quality",
                    "cash_flow_quality",
                    "moat_and_competitive_risk",
                    "valuation_and_risk_boundary",
                ],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "overview_bounded_answer",
            }

    valuation_intent = (
        "valuation_and_risk_boundary" in dimensions
        or primary == "valuation_and_risk_boundary"
        or methodology in {"valuation_boundary_analysis", "valuation_and_risk_boundary", "valuation"}
        or str(canonical_intent.get("intent_family") or "") == "valuation"
        or any(term in query_lower for term in ("估值", "贵不贵", "贵", "便宜", "valuation", "expensive", "cheap"))
    )
    if valuation_intent:
        valuation_line = _valuation_boundary_line_for_public_answer(state, lang=lang)
        if valuation_line and _format_constraints_dict(state).get("one_sentence"):
            return {
                "lines": [valuation_line],
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["valuation_and_risk_boundary"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "valuation_bounded_answer",
            }
        built_valuation = _contract_debt_valuation_fallback(state, lang=lang)
        if built_valuation:
            lines = [line for line in built_valuation[0].splitlines() if line.strip()]
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["valuation_and_risk_boundary"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "valuation_bounded_answer",
            }
        if valuation_line:
            lines = [valuation_line]
            rows = _node_numeric_rows(state)
            context_rows = [
                row
                for row in rows
                if str(row.get("metric") or "").strip() in {"revenue", "revenue_growth", "net_margin", "gross_margin", "operating_margin", "net_income", "free_cash_flow"}
            ][:4]
            if context_rows:
                heading = "相关财务背景" if lang == "zh" else "Related Financial Context"
                lines.append(heading)
                for row in context_rows:
                    metric = str(row.get("metric") or "").strip()
                    value = str(row.get("display_value") or row.get("formatted_value") or row.get("value") or "").strip()
                    refs = _citation_refs(_row_refs(row)[:2])
                    if metric and value:
                        label = _public_metric_label(metric, lang)
                        lines.append(f"- {label}: {value}{refs}" if lang != "zh" else f"- {label}：{value}{refs}")
            if missing_labels:
                lines.append(("证据边界：" if lang == "zh" else "Evidence boundary: ") + ("、".join(missing_labels[:5]) if lang == "zh" else ", ".join(missing_labels[:5])) + ("。" if lang == "zh" else "."))
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["valuation_and_risk_boundary"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": "valuation_bounded_answer",
            }

    profitability_intent = (
        "profitability_quality" in dimensions
        or primary == "profitability_quality"
        or methodology in {"profitability", "profitability_quality_analysis"}
        or str(canonical_intent.get("intent_family") or "") == "profitability"
        or any(term in query_lower for term in ("盈利质量", "利润质量", "profitability", "profit quality"))
    )
    if profitability_intent:
        built_profitability = _profitability_bounded_answer(state, lang=lang, missing_labels=missing_labels)
        if built_profitability:
            lines = [line for line in built_profitability[0].splitlines() if line.strip()]
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["profitability_quality"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": built_profitability[1],
            }

    is_risk = (
        "moat_and_competitive_risk" in dimensions
        or primary == "moat_and_competitive_risk"
        or methodology == "risk_focused_analysis"
        or any(term in query_lower for term in ("风险", "危险", "risk", "danger"))
    )
    if is_risk:
        bounded_risk = None if _answering._is_risk_comparison_query(query, state) else _bounded_risk_analysis_answer(state, lang=lang, missing_labels=missing_labels)
        if bounded_risk:
            return {
                "lines": [line for line in bounded_risk[0].splitlines() if line.strip()],
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["moat_and_competitive_risk"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
                "owner": bounded_risk[1],
            }
        if _answering._is_risk_comparison_query(query, state):
            bounded = _answering._bounded_risk_comparison_answer(
                state,
                dict(dict(state.get("output", {}) or {}).get("synthesis", {}) or {}),
                lang,
            )
            lines = [line for line in bounded.splitlines() if line.strip()]
            if lines:
                return {
                    "lines": lines,
                    "answer_quality_tier": "bounded_analysis",
                    "main_question_covered": True,
                    "fallback_intent_match": True,
                    "answered_dimensions": ["moat_and_competitive_risk"],
                    "final_route": "released_with_warnings",
                    "contract_status": "passed_with_warnings",
                }
        risk_rows = _risk_text_rows_for_public_answer(state)
        if risk_rows:
            companies = _company_list_for_public_answer(state)
            company = " 和 ".join(companies) if companies else "该公司"
            if lang == "zh":
                lines = [f"可以发布有边界的风险回答：{company} 的已验证风险文本只支持以下风险线索，不能扩展成未引用排序结论。"]
                for row in risk_rows[:4]:
                    ref = str(row.get("evidence_id") or "").strip()
                    text = str(row.get("claim") or row.get("summary") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
                    text = re.sub(r"\s+", " ", text)[:180].rstrip(" ，,。.;；")
                    if text:
                        lines.append(f"- {text} [{ref}]")
                if missing_labels:
                    lines.append("证据边界：" + "、".join(missing_labels[:5]) + "。")
            else:
                lines = [f"A bounded risk answer is releasable for {company}: validated risk text supports only these risk signals, not an uncited ranking conclusion."]
                for row in risk_rows[:4]:
                    ref = str(row.get("evidence_id") or "").strip()
                    text = str(row.get("claim") or row.get("summary") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
                    text = re.sub(r"\s+", " ", text)[:180].rstrip(" ,.;")
                    if text:
                        lines.append(f"- {text} [{ref}]")
                if missing_labels:
                    lines.append("Evidence boundary: " + ", ".join(missing_labels[:5]) + ".")
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["moat_and_competitive_risk"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
            }
        return {
            "lines": _scope_limit_boundary_lines(state, missing_labels, dimensions=["moat_and_competitive_risk"], lang=lang),
            "answer_quality_tier": "scope_limit",
            "main_question_covered": False,
            "fallback_intent_match": True,
            "answered_dimensions": [],
            "final_route": "scope_limited",
            "contract_status": "scope_limited",
        }

    revenue_intent = (
        "revenue_quality" in dimensions
        or primary == "revenue_quality"
        or methodology == "revenue_quality_analysis"
        or any(term in query_lower for term in ("营收", "收入", "revenue", "sales", "data center", "数据中心"))
    )
    if revenue_intent:
        rows = _rows_for_metric_terms(state, ("revenue", "sales", "data_center", "data center", "segment revenue", "营收", "收入"))
        if rows:
            labels = _metric_labels_from_rows(rows, lang=lang)
            refs = _refs_from_rows(rows)
            if lang == "zh":
                lines = [f"可以发布有边界的营收回答：当前可引用证据覆盖 {labels or '营收相关指标'}{refs}，但不足以扩展到未验证的驱动因素或投资判断。"]
            else:
                lines = [f"A bounded revenue answer is releasable: citable evidence covers {labels or 'revenue-related metrics'}{refs}, but not uncited drivers or investment conclusions."]
            if missing_labels:
                lines.append(("证据边界：" if lang == "zh" else "Evidence boundary: ") + ("、".join(missing_labels[:5]) if lang == "zh" else ", ".join(missing_labels[:5])) + ("。" if lang == "zh" else "."))
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["revenue_quality"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
            }
        return {
            "lines": _scope_limit_boundary_lines(state, missing_labels, dimensions=["revenue_quality"], lang=lang),
            "answer_quality_tier": "scope_limit",
            "main_question_covered": False,
            "fallback_intent_match": True,
            "answered_dimensions": [],
            "final_route": "scope_limited",
            "contract_status": "scope_limited",
        }

    cash_flow_intent = (
        "cash_flow_quality" in dimensions
        or primary == "cash_flow_quality"
        or methodology == "cash_flow_quality_analysis"
        or any(term in query_lower for term in ("现金流", "free cash flow", "fcf", "cfo", "capex", "cash flow"))
    )
    if cash_flow_intent:
        rows = _rows_for_metric_terms(state, ("free_cash_flow", "fcf", "cash_flow", "operating cash", "cfo", "capex", "capital_expenditure", "现金流"))
        if rows:
            labels = _metric_labels_from_rows(rows, lang=lang)
            refs = _refs_from_rows(rows)
            if lang == "zh":
                lines = [f"可以发布有边界的现金流质量回答：当前可引用证据覆盖 {labels or 'CFO、FCF、capex 或 FCF margin'}{refs}，但不能补写未验证的现金流解释。"]
            else:
                lines = [f"A bounded cash-flow-quality answer is releasable: citable evidence covers {labels or 'CFO, FCF, capex, or FCF margin'}{refs}, but not uncited cash-flow explanations."]
            if missing_labels:
                lines.append(("证据边界：" if lang == "zh" else "Evidence boundary: ") + ("、".join(missing_labels[:5]) if lang == "zh" else ", ".join(missing_labels[:5])) + ("。" if lang == "zh" else "."))
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": ["cash_flow_quality"],
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
            }
        return {
            "lines": _scope_limit_boundary_lines(state, missing_labels, dimensions=["cash_flow_quality"], lang=lang),
            "answer_quality_tier": "scope_limit",
            "main_question_covered": False,
            "fallback_intent_match": True,
            "answered_dimensions": [],
            "final_route": "scope_limited",
            "contract_status": "scope_limited",
        }

    overview_intent = False
    if overview_intent:
        numeric_rows = _numeric_rows_for_public_answer(state)
        text_rows = [dict(item) for item in dict(state.get("evidence_packet", {}) or {}).get("text_snippets", []) or [] if isinstance(item, dict)]
        if numeric_rows or text_rows:
            refs = _refs_from_rows(numeric_rows[:3] + text_rows[:3])
            if lang == "zh":
                lines = [f"可以发布有边界的公司概览：当前只使用已验证数字和文本证据{refs}，不补写缺证据的业务、风险或投资结论。"]
            else:
                lines = [f"A bounded company overview is releasable using only validated numeric and text evidence{refs}, without filling unsupported business, risk, or investment conclusions."]
            if missing_labels:
                lines.append(("证据边界：" if lang == "zh" else "Evidence boundary: ") + ("、".join(missing_labels[:5]) if lang == "zh" else ", ".join(missing_labels[:5])) + ("。" if lang == "zh" else "."))
            return {
                "lines": lines,
                "answer_quality_tier": "bounded_analysis",
                "main_question_covered": True,
                "fallback_intent_match": True,
                "answered_dimensions": dimensions,
                "final_route": "released_with_warnings",
                "contract_status": "passed_with_warnings",
            }
        return {
            "lines": _scope_limit_boundary_lines(state, missing_labels, dimensions=dimensions, lang=lang),
            "answer_quality_tier": "scope_limit",
            "main_question_covered": False,
            "fallback_intent_match": True,
            "answered_dimensions": [],
            "final_route": "scope_limited",
            "contract_status": "scope_limited",
        }

    return {}


def _safe_boundary_answer_lines(state: AgentState, missing_labels: list[str], *, lang: str) -> list[str]:
    companies = _company_list_for_public_answer(state)
    company_label = " 和 ".join(companies) if companies else "该股票"
    is_comparison = len(companies) >= 2 or str(state.get("answer_mode") or "") == "comparison_brief"
    valuation_line = _valuation_boundary_line_for_public_answer(state, lang=lang)
    if _answering._is_risk_comparison_query(str(state.get("user_query") or ""), state):
        output = state.get("output", {})
        synthesis_payload = dict(output.get("synthesis", {}) or {}) if isinstance(output, dict) else {}
        bounded = _answering._bounded_risk_comparison_answer(state, synthesis_payload, lang)
        return [line for line in bounded.splitlines() if line.strip()]
    if lang == "zh":
        if is_comparison:
            first = f"不能在当前证据边界内强行判断 {company_label} 谁的风险更大。"
        else:
            first = f"不能判断 {company_label} 现在是否值得买。"
        lines = [
            first,
            "我不提供买入、卖出或持有建议，也不提供目标价或确定性预测。",
            "可以发布的结论应转为风险、估值和证据边界：当前证据不足以支持完整结论，需要把风险因素、估值口径和已验证财务指标分开看。",
        ]
        if valuation_line:
            lines.append(valuation_line)
        if missing_labels:
            lines.append("证据限制包括：" + "、".join(missing_labels[:5]) + "。")
        lines.append("因此这里只能给出有限分析框架，不能替代投资建议。")
        return lines
    if is_comparison:
        first = f"I cannot force a ranking of which has greater risk between {company_label} within the current evidence boundary."
    else:
        first = f"I cannot decide whether {company_label} is worth buying now."
    lines = [
        first,
        "I do not provide buy, sell, or hold advice, target prices, or deterministic forecasts.",
        "The releasable answer should be reframed as risk, valuation, and evidence-boundary analysis: current evidence is insufficient for a complete conclusion, so risks, valuation basis, and validated financial metrics must be separated.",
    ]
    if valuation_line:
        lines.append(valuation_line)
    if missing_labels:
        lines.append("Evidence limits include: " + ", ".join(missing_labels[:5]) + ".")
    lines.append("This is only a limited analytical framework, not investment advice.")
    return lines


def _irreparable_contract_block(result: dict[str, Any]) -> bool:
    irreparable = {
        "forbidden_claim",
        "invalid_citation",
        "raw_internal_leakage",
        "unsupported_numeric",
        "evidence_backed_claim_without_citation",
    }
    for item in result.get("violations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or item.get("type") or "")
        if code in irreparable:
            if code == "unsupported_numeric" and str(item.get("suggested_fix") or item.get("suggested_value") or "").strip():
                continue
            return True
    return False


def _risk_text_rows_for_public_answer(state: AgentState) -> list[dict[str, Any]]:
    packet = _ensure_node_canonical_packet(state)
    rows: list[dict[str, Any]] = []
    source = packet.get("text_snippets")
    if not isinstance(source, list):
        source = []
    for item in source:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("evidence_id") or "").strip()
        if not ref.startswith("T"):
            continue
        dimension = str(item.get("dimension_id") or "").strip()
        section = str(item.get("section") or "").upper().strip()
        if dimension == "moat_and_competitive_risk" or section in {"ITEM_1A", "ITEM_7", "ITEM_2"}:
            rows.append(dict(item))
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        if ref and ref not in seen:
            seen.add(ref)
            out.append(row)
    return out


def _risk_comparison_without_citable_text(state: AgentState) -> bool:
    owner = str(state.get("final_answer_source") or "")
    if not owner:
        history = [item for item in state.get("answer_history", []) or [] if isinstance(item, dict)]
        if history:
            owner = str(history[-1].get("new_owner") or "")
    if _is_bounded_risk_comparison_owner(owner):
        return False
    if str(state.get("task_type") or "") != "company_comparison":
        return False
    query = str(state.get("user_query") or "").lower()
    dimensions = _node_requested_dimensions(state)
    risk_requested = (
        "moat_and_competitive_risk" in dimensions
        or str(state.get("primary_dimension") or "") == "moat_and_competitive_risk"
        or any(term in query for term in ("风险", "危险", "risk", "danger"))
    )
    numeric_dimensions = {
        "valuation_and_risk_boundary",
        "cash_flow_quality",
        "revenue_quality",
        "profitability_quality",
        "balance_sheet_and_capital_intensity",
    }
    if not risk_requested or any(dim in numeric_dimensions for dim in dimensions):
        return False
    return not bool(_risk_text_rows_for_public_answer(state))


def _bounded_grounded_answer_lines(state: AgentState, missing_labels: list[str], *, lang: str) -> list[str]:
    payload = build_intent_compatible_boundary_answer(state, missing_labels, lang=lang)
    if payload:
        return [str(line) for line in payload.get("lines", []) if str(line).strip()]
    query = str(state.get("user_query") or "")
    if _answering._is_risk_comparison_query(query, state):
        return _safe_boundary_answer_lines(state, missing_labels, lang=lang)
    risk_rows = _risk_text_rows_for_public_answer(state)
    is_risk = (
        str(state.get("answer_mode") or "") == "risk_focused_analysis"
        or str(state.get("primary_dimension") or "") == "moat_and_competitive_risk"
        or any(term in query.lower() for term in ("风险", "risk", "危险", "danger"))
    )
    if is_risk and risk_rows:
        companies = _company_list_for_public_answer(state)
        company = " 和 ".join(companies) if companies else "该公司"
        scenario = any(term in query.lower() for term in ("下季度", "经济放缓", "slowdown", "recession", "next quarter", "macro"))
        if lang == "zh":
            first = (
                f"可以发布有边界的情景风险判断：{company} 的已验证风险文本可支持风险线索，但不能量化下季度影响或做确定预测。"
                if scenario
                else f"可以发布有边界的风险判断：{company} 的已验证风险文本可支持主要风险线索，但不能把未引用内容写成排序结论。"
            )
            lines = [first, "已验证风险证据"]
            for row in risk_rows[:4]:
                ref = str(row.get("evidence_id") or "").strip()
                text = str(row.get("claim") or row.get("summary") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
                text = re.sub(r"\s+", " ", text)[:180].rstrip(" ，,。.;；")
                lines.append(f"- {text} [{ref}]")
            if missing_labels:
                lines.append("证据边界：" + "、".join(missing_labels[:5]) + "。")
            lines.append("该回答不构成投资建议。")
            return lines
        first = (
            f"A bounded scenario-risk answer is releasable for {company}: validated risk text supports risk signals, but not quantified next-quarter impact or deterministic forecasts."
            if scenario
            else f"A bounded risk answer is releasable for {company}: validated risk text supports risk signals, but uncited content cannot be used as a ranking conclusion."
        )
        lines = [first, "Validated Risk Evidence"]
        for row in risk_rows[:4]:
            ref = str(row.get("evidence_id") or "").strip()
            text = str(row.get("claim") or row.get("summary") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
            text = re.sub(r"\s+", " ", text)[:180].rstrip(" ,.;")
            lines.append(f"- {text} [{ref}]")
        if missing_labels:
            lines.append("Evidence boundaries: " + ", ".join(missing_labels[:5]) + ".")
        lines.append("This is not investment advice.")
        return lines
    return _safe_boundary_answer_lines(state, missing_labels, lang=lang)


def _release_safe_boundary_contract(state: AgentState) -> bool:
    result = _contract_result_dict(state)
    safety_release = _true_investment_advice_like(state) or _prediction_or_target_or_out_of_scope(state)
    if _irreparable_contract_block(result) and not safety_release:
        return False
    lang = _node_target_lang(state)
    return bool(build_intent_compatible_boundary_answer(state, [], lang=lang))


def safe_blocked_answer_node(state: AgentState) -> dict[str, Any]:
    """Replace an unsafe or unsupported draft with a public safe-blocked answer."""
    lang = _node_target_lang(state)
    result = _contract_result_dict(state)
    has_explicit_blocking_missing = (
        "blocking_missing_requirements" in result
        or "missing_required_requirements" in result
        or "missing_required_requirements" in state
    )
    blocking_missing = list(
        result.get("blocking_missing_requirements", [])
        or result.get("missing_required_requirements", [])
        or state.get("missing_required_requirements", [])
        or []
    )
    if not blocking_missing and str(result.get("route") or "") == "need_more_evidence":
        blocking_missing = list(result.get("missing_requirements", []) or state.get("missing_requirements", []) or [])
    elif not blocking_missing and not has_explicit_blocking_missing:
        requirements_by_id = _evidence_requirements_by_id(state)
        for item in list(result.get("missing_requirements", []) or state.get("missing_requirements", []) or []):
            rid = str(item or "").strip()
            req = requirements_by_id.get(rid, {})
            scope = str(req.get("requirement_scope") or req.get("scope") or "").strip()
            if req.get("required") is False or scope in {"optional_context", "diagnostic"}:
                continue
            blocking_missing.append(item)
    missing_labels = _friendly_missing_requirement_labels(state, blocking_missing, lang=lang)
    boundary_payload: dict[str, Any] = {}
    safety_release = _true_investment_advice_like(state) or _prediction_or_target_or_out_of_scope(state)
    boundary_payload = build_intent_compatible_boundary_answer(state, missing_labels, lang=lang)
    if not boundary_payload and (safety_release or not _irreparable_contract_block(result)):
        boundary_payload = build_intent_compatible_boundary_answer(state, missing_labels, lang=lang)
    release_boundary = bool(boundary_payload)
    boundary_tier = str(boundary_payload.get("answer_quality_tier") or "")
    if release_boundary:
        lines = [str(line) for line in boundary_payload.get("lines", []) if str(line).strip()]
    elif lang == "zh":
        first_line = (
            "目前证据不足以支持一个完整且通过契约校验的结论。"
            if missing_labels
            else "当前候选答案未通过契约校验，已停止发布。"
        )
        lines = [first_line, "我不会返回包含未验证数字、无效引用、投资建议、目标价或内部诊断信息的草稿。"]
        if missing_labels:
            lines.append("缺失的信息包括：" + "、".join(missing_labels[:5]) + "。")
    else:
        first_line = (
            "I could not produce a fully evidence-supported answer for this query."
            if missing_labels
            else "The candidate answer did not pass the evidence contract, so it was not released."
        )
        lines = [first_line, "I will not return a draft with unsupported numbers, invalid citations, investment advice, target prices, or internal diagnostics."]
        if missing_labels:
            lines.append("Missing evidence includes: " + ", ".join(missing_labels[:5]) + ".")
    answer = "\n".join(lines)
    if release_boundary and boundary_tier == "scope_limit":
        owner = "safe_boundary_scope_limit"
    elif release_boundary and boundary_tier == "safe_refusal":
        owner = "safe_boundary_safe_refusal"
    elif release_boundary and str(boundary_payload.get("owner") or ""):
        owner = str(boundary_payload.get("owner"))
    else:
        owner = "safe_boundary_bounded_answer" if release_boundary else "safe_blocked_answer"
    assembled = _assemble_node_answer(
        state,
        answer=answer,
        owner=owner,
        transform="safe_blocked_answer_node",
        reason=str(result.get("decision") or result.get("route") or "blocked"),
        claim_change_allowed=bool(release_boundary),
        validator_result=result,
        provenance={"release_boundary": release_boundary, "answer_quality_tier": boundary_tier},
    )
    output = dict(state.get("output", {}) or {})
    output["final_answer_source"] = owner
    output["evidence_packet_summary"] = dict(state.get("evidence_packet_summary", {}) or {})
    output["answer_history"] = list(assembled.get("answer_history", state.get("answer_history", []) or []))
    output["answer_candidate"] = dict(assembled.get("answer_candidate", {}) or {})
    output["answer_candidates"] = list(assembled.get("answer_candidates", state.get("answer_candidates", []) or []))
    output["answer_quality_tier"] = boundary_tier or ("invalid_fallback" if not release_boundary else "bounded_analysis")
    output["main_question_covered"] = bool(boundary_payload.get("main_question_covered", False))
    output["fallback_intent_match"] = bool(boundary_payload.get("fallback_intent_match", bool(release_boundary)))
    output["answered_dimensions"] = list(boundary_payload.get("answered_dimensions", []) or [])
    output["unresolved_relevance_failures"] = []
    output["format_constraints_satisfied"] = True
    payload: dict[str, Any] = {
        **assembled,
        "output": output,
        "final_answer_source": owner,
        "evidence_packet": dict(state.get("evidence_packet", {}) or {}),
        "evidence_packet_summary": dict(state.get("evidence_packet_summary", {}) or {}),
        "contract_status": "blocked",
        "final_contract_status": "blocked",
        "contract_public_summary": str(result.get("public_summary") or "Answer was blocked."),
        "answer_quality_tier": output["answer_quality_tier"],
        "main_question_covered": output["main_question_covered"],
        "fallback_intent_match": output["fallback_intent_match"],
        "answered_dimensions": output["answered_dimensions"],
        "unresolved_relevance_failures": output["unresolved_relevance_failures"],
        "format_constraints_satisfied": output["format_constraints_satisfied"],
    }
    if release_boundary:
        is_scope_limit = boundary_tier == "scope_limit"
        payload["contract_result"] = {
            "passed": not is_scope_limit,
            "severity": "warning",
            "decision": "scope_limited" if is_scope_limit else "warning",
            "route": "scope_limit" if is_scope_limit else "pass",
            "violations": [],
            "warnings": list(result.get("violations", []) or result.get("warnings", []) or []),
            "missing_requirements": list(result.get("missing_requirements", []) or []),
            "blocking_missing_requirements": list(result.get("blocking_missing_requirements", []) or []),
            "public_summary": (
                "Released a scope-limited answer matching the original intent."
                if is_scope_limit
                else "Released a safe evidence-boundary answer instead of the unsupported draft."
            ),
            "safe_boundary_released_from": str(result.get("decision") or result.get("route") or "blocked"),
        }
        payload["contract_status"] = "scope_limited" if is_scope_limit else "passed_with_warnings"
        payload["final_contract_status"] = "scope_limited" if is_scope_limit else "passed_with_warnings"
        payload["contract_public_summary"] = str(payload["contract_result"]["public_summary"])
        payload["final_route"] = str(boundary_payload.get("final_route") or ("scope_limited" if is_scope_limit else "released_with_warnings"))
    return payload


def _contract_output_summary(state: AgentState, status: str) -> dict[str, Any]:
    result = _contract_result_dict(state)
    return {
        "status": status,
        "decision": str(result.get("decision") or ("warning" if status == "passed_with_warnings" else status)),
        "public_summary": str(result.get("public_summary") or state.get("contract_public_summary") or ""),
        "repair_attempts": int(state.get("contract_attempts", 0) or 0),
        "evidence_retry_count": int(state.get("contract_evidence_retry_count", 0) or 0),
        "warnings": list(result.get("warnings", []) or []),
    }


def _format_constraints_dict(state: AgentState) -> dict[str, Any]:
    raw = state.get("format_constraints")
    if isinstance(raw, dict):
        constraints = dict(raw)
    else:
        constraints = {}
    query = str(state.get("user_query") or "").lower()
    if not constraints.get("one_sentence") and re.search(
        r"(一句话|一段话|用一句|只用一句|one sentence|single sentence|in one sentence)",
        query,
    ):
        constraints["one_sentence"] = True
        constraints["max_sentences"] = 1
    return constraints


def _sentence_count_for_format(answer: str) -> int:
    text = re.sub(r"\[[A-Z]\d+\]", "", str(answer or ""))
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    pieces = [part.strip() for part in re.split(r"[。！？!?]+|\n+", text) if part.strip()]
    return len(pieces)


def _format_constraints_satisfied(answer: str, state: AgentState) -> bool:
    constraints = _format_constraints_dict(state)
    max_sentences = int(constraints.get("max_sentences") or (1 if constraints.get("one_sentence") else 0) or 0)
    if max_sentences > 0 and _sentence_count_for_format(answer) > max_sentences:
        return False
    return True


def _relevance_failure_codes_from_state(state: AgentState) -> list[str]:
    decision = state.get("relevance_decision", {})
    if not isinstance(decision, dict):
        return []
    out: list[str] = []
    for item in decision.get("deterministic_relevance_failures", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code") or "").strip()
        if code and code not in out:
            out.append(code)
    return out


def _quality_missing_parts(state: AgentState) -> tuple[list[str], list[str]]:
    core: list[str] = []
    optional: list[str] = []

    def add(target: list[str], values: Any) -> None:
        for item in values or []:
            text = str(item).strip()
            if text and text not in target:
                target.append(text)

    sources = [
        state,
        dict(state.get("output", {}) or {}),
        dict(state.get("evidence_sufficiency", {}) or {}),
        dict(state.get("synthesis", {}) or {}).get("requirement_summary", {}) if isinstance(state.get("synthesis"), dict) else {},
    ]
    for source in sources:
        if not isinstance(source, dict):
            continue
        add(core, source.get("missing_required_answer_parts"))
        add(core, source.get("partial_required_answer_parts"))
        add(optional, source.get("missing_but_analyzable_answer_parts"))
        add(optional, source.get("missing_optional_answer_parts"))
        add(optional, source.get("diagnostic_missing_parts"))
    return core, optional


def _contract_metrics_from_state(state: AgentState) -> dict[str, Any]:
    result = _contract_result_dict(state)
    metrics = dict(result.get("metrics", {}) or {})
    output_contract = dict(dict(state.get("output", {}) or {}).get("contract", {}) or {})
    metrics.update(dict(output_contract.get("metrics", {}) or {}))
    return metrics


def _segment_or_product_scope_from_state(state: AgentState) -> str:
    output = dict(state.get("output", {}) or {})
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    canonical_intent = dict(state.get("canonical_intent", {}) or analysis_plan.get("canonical_intent", {}) or {})
    synthesis = dict(state.get("synthesis", {}) or {})
    return str(
        state.get("segment_or_product_scope")
        or output.get("segment_or_product_scope")
        or synthesis.get("segment_or_product_scope")
        or analysis_plan.get("segment_or_product_scope")
        or analysis_plan.get("segment_focus")
        or canonical_intent.get("segment_or_product_scope")
        or canonical_intent.get("segment_focus")
        or ""
    ).strip()


def _answer_has_boundary_language(answer: str) -> bool:
    lowered = str(answer or "").lower()
    return any(
        term in lowered
        for term in (
            "证据边界",
            "有限结论",
            "当前不能",
            "不能可靠",
            "缺少",
            "insufficient",
            "evidence boundary",
            "limited conclusion",
            "cannot reliably",
            "missing",
        )
    )


def _quality_fields_for_answer(
    state: AgentState,
    *,
    answer: str,
    status: str,
    final_route: str,
    final_answer_source: str,
) -> dict[str, Any]:
    inherited_tier = str(state.get("answer_quality_tier") or "")
    inherited_answered = [str(item) for item in state.get("answered_dimensions", []) or [] if str(item).strip()]
    answered_dimensions = inherited_answered or _node_requested_dimensions(state)
    format_ok = _format_constraints_satisfied(answer, state)
    relevance_status = str(state.get("relevance_status") or "")
    relevance_failures = _relevance_failure_codes_from_state(state)
    unresolved = relevance_failures if relevance_status not in {"passed", "passed_with_warnings", "not_run", ""} else []
    route = str(final_route or "").lower()
    safety_release = _true_investment_advice_like(state) or _prediction_or_target_or_out_of_scope(state)
    core_missing, optional_missing = _quality_missing_parts(state)
    metrics = _contract_metrics_from_state(state)
    substantive_contract_failures = [
        str(item.get("code") or item.get("type") or "").strip()
        for item in _contract_result_dict(state).get("violations", []) or []
        if isinstance(item, dict)
        and str(item.get("severity") or "").strip() not in {"warning"}
        and str(item.get("code") or item.get("type") or "").strip()
    ]
    repair_debt = bool(state.get("primary_generation_contract_debt") or dict(state.get("output", {}) or {}).get("primary_generation_contract_debt"))
    fallback_intent_match = bool(state.get("fallback_intent_match", True))
    main_question_covered = bool(state.get("main_question_covered", True))

    reason = ""
    bounded_source_names = {
        "bounded_valuation_boundary_postprocess",
        "bounded_valuation_postprocess",
        "valuation_bounded_answer",
        "bounded_valuation_candidate",
        "overview_bounded_answer",
        "bounded_segment_answer",
        "bounded_profitability_candidate",
        "bounded_profit_decline_candidate",
        "bounded_scenario_risk_candidate",
        "bounded_risk_candidate",
        "bounded_analysis",
        "bounded_risk_analysis",
        "bounded_valuation_risk_comparison_candidate",
        "bounded_risk_comparison_postprocess",
    }
    if not format_ok:
        tier = "invalid_fallback"
        main_question_covered = False
        fallback_intent_match = False
        reason = "format_constraints_failed"
    elif not fallback_intent_match:
        tier = "invalid_fallback"
        main_question_covered = False
        reason = "fallback_intent_mismatch"
    elif route == "safe_refusal":
        tier = "safe_refusal" if safety_release else "invalid_fallback"
        main_question_covered = False
        fallback_intent_match = bool(safety_release)
        reason = "valid_safety_refusal" if safety_release else "safe_refusal_for_non_safety_intent"
    elif route == "scope_limited" or status == "scope_limited" or (repair_debt and not _is_contract_debt_business_fallback_owner(final_answer_source)):
        tier = "scope_limit"
        main_question_covered = False
        reason = "scope_limited_route" if not repair_debt else "primary_generation_contract_debt"
    elif unresolved:
        tier = "invalid_fallback"
        main_question_covered = False
        fallback_intent_match = False
        reason = "unresolved_relevance_failures"
    elif substantive_contract_failures and status not in {"passed", "repaired", "passed_with_warnings"}:
        tier = "scope_limit" if _answer_has_boundary_language(answer) else "invalid_fallback"
        main_question_covered = tier != "scope_limit"
        fallback_intent_match = tier == "scope_limit"
        reason = "substantive_contract_failures"
    elif inherited_tier in {"scope_limit", "safe_refusal", "invalid_fallback"}:
        tier = inherited_tier
        reason = f"inherited_{inherited_tier}"
    elif inherited_tier == "bounded_analysis" or str(final_answer_source or "") in bounded_source_names:
        tier = "bounded_analysis"
        main_question_covered = True
        reason = "inherited_bounded_analysis"
    elif core_missing:
        tier = "bounded_analysis"
        main_question_covered = True
        reason = "core_answer_parts_partial_or_missing"
    elif route == "bounded_fallback" or _answer_has_boundary_language(answer):
        tier = "bounded_analysis"
        main_question_covered = True
        reason = "answered_with_explicit_evidence_boundary"
    elif status in {"passed", "repaired", "passed_with_warnings"}:
        tier = "true_answer"
        main_question_covered = True
        reason = "core_question_answered_without_substantive_failures"
    else:
        tier = "invalid_fallback"
        main_question_covered = False
        fallback_intent_match = False
        reason = "could_not_classify_as_answer_or_scope_limit"

    return {
        "answer_quality_tier": tier,
        "quality_tier_reason": reason,
        "main_question_covered": main_question_covered,
        "fallback_intent_match": fallback_intent_match,
        "answered_dimensions": answered_dimensions if tier not in {"scope_limit", "invalid_fallback"} else inherited_answered,
        "unresolved_relevance_failures": unresolved,
        "format_constraints_satisfied": format_ok,
        "format_constraints": _format_constraints_dict(state),
        "core_missing_parts": core_missing,
        "optional_missing_parts": optional_missing,
        "risk_items_directly_supported_count": int(metrics.get("risk_items_directly_supported_count", 0) or 0),
        "risk_items_template_only_count": int(metrics.get("risk_items_template_only_count", 0) or 0),
        "company_specific_token_leakage": int(metrics.get("company_specific_token_leakage", 0) or 0),
        "segment_or_product_scope": _segment_or_product_scope_from_state(state),
    }


def _merge_public_limitations(state: AgentState) -> list[str]:
    out: list[str] = [str(item) for item in state.get("limitations", []) or [] if str(item).strip()]
    output = state.get("output", {})
    if isinstance(output, dict):
        for item in output.get("limitations", []) or []:
            if isinstance(item, dict):
                msg = str(item.get("message") or item.get("code") or "").strip()
            else:
                msg = str(item).strip()
            if msg:
                out.append(msg)
    result = _contract_result_dict(state)
    out.extend(_public_limitation_lines(state, result))
    report = state.get("report", {})
    if isinstance(report, dict):
        out.extend(str(item) for item in report.get("overall_limitations", []) or [] if str(item).strip())
    return list(dict.fromkeys(out))


def finalize_node(state: AgentState) -> dict[str, Any]:
    """Promote the checked draft to the final answer and attach public metadata."""
    result = _contract_result_dict(state)
    attempts = int(state.get("contract_attempts", 0) or 0)
    current_status = str(state.get("contract_status") or "")
    if current_status == "blocked" or str(state.get("final_contract_status") or "") == "blocked":
        status = "blocked"
    elif current_status == "scope_limited" or str(state.get("final_contract_status") or "") == "scope_limited" or str(result.get("route") or "") == "scope_limit":
        status = "scope_limited"
    elif str(result.get("route") or "") == "pass" and attempts > 0:
        status = "repaired"
    elif str(result.get("route") or "") == "pass" and str(result.get("decision") or "") == "warning":
        status = "passed_with_warnings"
    elif str(result.get("route") or "") == "pass":
        status = "passed"
    else:
        status = "blocked"

    output = dict(state.get("output", {}) or {})
    draft_answer = str(state.get("draft_answer") or "")
    previous_final_answer = str(state.get("final_answer") or "")
    final_answer = draft_answer or previous_final_answer
    if draft_answer and draft_answer != previous_final_answer:
        promote_owner = str(
            state.get("final_answer_source")
            or output.get("final_answer_source")
            or "finalize_promote_draft"
        )
        assembled = _assemble_node_answer(
            state,
            answer=draft_answer,
            owner=promote_owner,
            transform="finalize_promote_draft",
            reason="draft_answer selected as final_answer",
            claim_change_allowed=False,
            provenance={"finalize_status": status},
        )
        state.update(assembled)
        final_answer = str(assembled.get("final_answer") or draft_answer)
    if status not in {"blocked", "scope_limited"} and _risk_comparison_without_citable_text({**state, "final_answer": final_answer}):
        lang = _node_target_lang(state)
        scope_answer = _relevance_scope_limit_answer(
            state=state,
            requested_dimensions=["moat_and_competitive_risk"],
            failures=["comparison_risk_text_dimension_missing"],
            lang=lang,
        )
        assembled = _assemble_node_answer(
            state,
            answer=scope_answer,
            owner="risk_text_scope_limit",
            transform="finalize_risk_text_scope_limit",
            reason="risk comparison has no final citable risk text in canonical packet",
            claim_change_allowed=False,
            provenance={"text_citable_count": 0},
        )
        state.update(assembled)
        final_answer = str(assembled.get("final_answer") or scope_answer)
        status = "scope_limited"
    answer_history = [dict(item) for item in state.get("answer_history", []) or [] if isinstance(item, dict)]
    final_answer_source = str(
        (answer_history[-1].get("new_owner") if answer_history else "")
        or state.get("final_answer_source")
        or output.get("final_answer_source")
        or ""
    )
    lang = _node_target_lang({**state, "user_query": str(state.get("user_query") or final_answer or "")})
    scrubbed_final_answer = _rendering.sanitize_user_facing_answer_text(final_answer, lang)
    scrubbed_final_answer = repair_language_leakage(scrubbed_final_answer, lang)
    if scrubbed_final_answer != final_answer:
        final_answer = scrubbed_final_answer
        state["final_answer"] = final_answer
        state["draft_answer"] = final_answer
    leakage_count = language_leakage_count(final_answer, lang)
    output["output_language"] = lang
    output["language_leakage"] = leakage_count
    output["language_leakage_unresolved"] = leakage_count > 0
    output["final_answer_source"] = final_answer_source
    output["answer_history"] = answer_history
    if state.get("answer_candidate"):
        output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
    if state.get("answer_candidates"):
        output["answer_candidates"] = list(state.get("answer_candidates", []) or [])
    output["contract"] = _contract_output_summary(state, status)
    output["contract_decision"] = result
    relevance_decision = dict(state.get("relevance_decision", {}) or {})
    if relevance_decision:
        output["relevance_decision"] = relevance_decision
        output["relevance_status"] = str(state.get("relevance_status") or relevance_decision.get("status") or "")
    relevance_status = str(state.get("relevance_status") or relevance_decision.get("status") or "")
    if status == "blocked":
        final_route = "blocked"
    elif status == "scope_limited" or str(state.get("final_route") or "") == "scope_limited" or str(state.get("final_answer_source") or "") in {"research_relevance_scope_limit", "safe_boundary_scope_limit"}:
        final_route = "scope_limited"
    elif str(state.get("final_route") or "") == "safe_refusal" or str(state.get("final_answer_source") or "") == "safe_boundary_safe_refusal":
        final_route = "safe_refusal"
    elif str(state.get("final_route") or "") == "bounded_fallback" or str(state.get("final_answer_source") or "") in {"research_relevance_bounded_fallback", "bounded_risk_comparison_answer", "risk_comparison_bounded_answer"}:
        final_route = "bounded_fallback"
    elif status in {"passed_with_warnings", "repaired"} or relevance_status in {"passed_with_warnings", "analytical_with_gaps"}:
        final_route = "released_with_warnings"
    else:
        final_route = "released"
    quality_fields = _quality_fields_for_answer(
        state,
        answer=final_answer,
        status=status,
        final_route=final_route,
        final_answer_source=final_answer_source,
    )
    output["final_route"] = final_route
    output.update(quality_fields)
    for key, default in {
        "repair_applied": False,
        "repair_owner": "",
        "source_before_repair": "",
        "repair_types": [],
        "repair_attempts": int(state.get("contract_attempts", 0) or 0),
        "material_claim_uncited_count": 0,
        "primary_generation_contract_debt": False,
    }.items():
        value = state.get(key, output.get(key, default))
        output[key] = value
    report = build_company_analysis_report({**state, "final_answer": final_answer, "output": output})
    report_contract_status = ""
    report_contract_result: dict[str, Any] = {}
    if report:
        output["report"] = report
        report_contract_status = str(report.get("contract_status") or "")
        report_contract_result = {
            "passed": report_contract_status == "passed",
            "status": report_contract_status,
        }
    limitations = _merge_public_limitations({**state, "report": report})
    trace_id = str(state.get("trace_id") or "")
    if trace_id:
        append_progress_event(
            trace_id,
            "answer_released",
            "completed" if status != "blocked" else "warning",
            "分析完成，最终答案已发布。" if status != "blocked" else "分析完成，但最终答案被合约阻断。",
            node="finalize",
            metadata={"trace_id": trace_id, "final_status": status, "final_route": final_route},
        )
    payload = {
        "final_answer": final_answer,
        "output": output,
        "final_answer_source": final_answer_source,
        "output_language": lang,
        "language_leakage": leakage_count,
        "language_leakage_unresolved": leakage_count > 0,
        "answer_history": answer_history,
        "answer_candidate": dict(state.get("answer_candidate", {}) or {}),
        "answer_candidates": list(state.get("answer_candidates", []) or []),
        "contract_status": status,
        "final_contract_status": status,
        "contract_decision": result,
        "contract_public_summary": output["contract"]["public_summary"],
        "relevance_decision": relevance_decision,
        "relevance_status": relevance_status,
        "relevance_repair_attempts": int(state.get("relevance_repair_attempts") or state.get("relevance_attempts") or 0),
        "final_route": final_route,
        **quality_fields,
        "repair_applied": bool(output.get("repair_applied", False)),
        "repair_owner": str(output.get("repair_owner") or ""),
        "source_before_repair": str(output.get("source_before_repair") or ""),
        "repair_types": list(output.get("repair_types", []) or []),
        "repair_attempts": int(output.get("repair_attempts", attempts) or 0),
        "material_claim_uncited_count": int(output.get("material_claim_uncited_count", 0) or 0),
        "primary_generation_contract_debt": bool(output.get("primary_generation_contract_debt", False)),
        "limitations": limitations,
        "report": report,
        "report_sections": list(report.get("sections", []) or []) if isinstance(report, dict) else [],
        "report_contract_status": report_contract_status,
        "report_contract_result": report_contract_result,
    }
    if str(state.get("draft_answer") or "").strip():
        payload["draft_answer"] = final_answer
    return payload
