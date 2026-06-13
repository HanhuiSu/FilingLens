"""API route handlers — POST /chat, GET /health, GET /trace/{trace_id}."""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from config import settings
from src.agent.analysis_framework import summarize_selected_analysis_framework
from src.agent.evidence_sufficiency import (
    build_trace_summary,
    normalize_dimension_status_contract,
    summarize_evidence_requirements,
)
from src.agent.progress import (
    append_progress_event,
    read_trace_payload,
    trace_path_for_id,
    utc_now_iso,
    validate_trace_id,
    write_trace_payload,
)
from src.agent.red_flags import detect_red_flags, serialize_red_flags
from src.api.models import (
    ChatRequest,
    ChatResponse,
    Citation,
    ErrorResponse,
    HealthResponse,
    TraceUiResponse,
    TraceResponse,
)
from src.api.trace_view import build_trace_ui_model

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────────

def _save_trace(trace_id: str, state: dict[str, Any]) -> Path:
    """Persist the full agent state to data/traces/{trace_id}.json."""
    safe_trace_id = validate_trace_id(trace_id)
    existing = read_trace_payload(safe_trace_id)

    serialisable: dict[str, Any] = {}
    skip_keys = {"messages"}
    for k, v in state.items():
        if k in skip_keys:
            continue
        try:
            json.dumps(v, default=str)
            serialisable[k] = v
        except (TypeError, ValueError):
            serialisable[k] = str(v)

    if existing.get("progress_events") and not serialisable.get("progress_events"):
        serialisable["progress_events"] = existing.get("progress_events")
    if existing.get("run_started_at") and not serialisable.get("run_started_at"):
        serialisable["run_started_at"] = existing.get("run_started_at")
    serialisable["trace_id"] = safe_trace_id
    return write_trace_payload(safe_trace_id, serialisable)


def _build_citations(raw: list[dict[str, Any]]) -> list[Citation]:
    """Convert raw citation dicts from agent state into Citation models."""
    out: list[Citation] = []
    for c in raw:
        out.append(
            Citation(
                source=c.get("ticker", c.get("source", "")),
                filing_type=c.get("form_type", c.get("filing_type", "")),
                period=c.get("fiscal_period", c.get("period", "")),
                section=c.get("section", ""),
                part=c.get("part", ""),
                quality=c.get("quality", ""),
                text_snippet=c.get("text_snippet", "")[:200],
                source_kind=c.get("source_kind", "document"),
                metric=c.get("metric", ""),
                period_type=c.get("period_type", ""),
                period_end=c.get("period_end", ""),
                filing_date=c.get("filing_date", ""),
                source_provider=c.get("source_provider", ""),
                source_url=c.get("source_url", ""),
                source_filing_id=c.get("source_filing_id", ""),
                confidence=c.get("confidence", ""),
                extraction_method=c.get("extraction_method", ""),
                source_tag=c.get("source_tag", ""),
                reconciliation_warning=c.get("reconciliation_warning", ""),
                section_fallback=bool(c.get("section_fallback", False)),
            )
        )
    return out


def _merge_requirement_entries(
    stored: dict[str, Any] | None,
    computed: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for rid, item in dict(stored or {}).items():
        merged[str(rid)] = dict(item) if isinstance(item, dict) else item
    for rid, item in dict(computed or {}).items():
        key = str(rid)
        if isinstance(item, dict):
            base = dict(merged.get(key, {})) if isinstance(merged.get(key, {}), dict) else {}
            for item_key, item_value in dict(item).items():
                if item_value is None or item_value == "":
                    base.setdefault(item_key, item_value)
                    continue
                base[item_key] = item_value
            merged[key] = base
        else:
            merged[key] = item
    return merged


def _infer_final_answer_source(data: dict[str, Any], synthesis_mode: str) -> str:
    explicit = (
        str(data.get("final_answer_source", "") or "")
        or str((data.get("output", {}) or {}).get("final_answer_source", "") or "")
        or str((data.get("synthesis", {}) or {}).get("final_answer_source", "") or "")
    )
    if explicit:
        return explicit
    answer_mode = str(data.get("answer_mode", "") or "")
    safety_intent = str(data.get("safety_intent", "") or "")
    if answer_mode in {"meta", "clarification", "refusal_or_redirect"} or safety_intent == "unsupported_or_out_of_scope":
        return "unsupported_or_refusal"
    validation = dict(data.get("draft_validation", {}) or data.get("analyst_draft_validation", {}) or {})
    accepted_draft = dict(validation.get("accepted_draft", {}) or {})
    revision_attempts = list(data.get("draft_revision_attempts", []) or [])
    if accepted_draft and bool(validation.get("passed", False)):
        return "analyst_draft_revised" if revision_attempts else "analyst_draft_initial"
    if dict(data.get("comparison_judgment_frame", {}) or {}) and (
        answer_mode == "comparison_brief" or str(data.get("task_type", "")) == "company_comparison"
    ):
        return "comparison_decision_fallback"
    if synthesis_mode == "conversational_short_circuit":
        return "unsupported_or_refusal"
    return "deterministic_synthesis"


def _selected_framework_id(data: dict[str, Any], trace_summary: dict[str, Any]) -> str:
    selected = data.get("selected_analysis_framework", {})
    if isinstance(selected, dict) and selected:
        summary = summarize_selected_analysis_framework(selected)
        framework_id = str(summary.get("id", "") or "").strip()
        if framework_id:
            return framework_id
    return str(trace_summary.get("analysis_framework_id", "") or "").strip()


def _active_dimension_ids(
    data: dict[str, Any],
    trace_summary: dict[str, Any],
    dimension_status_map: dict[str, Any],
) -> list[str]:
    packet = data.get("evidence_packet", {})
    if isinstance(packet, dict):
        active = [str(item) for item in packet.get("active_dimensions", []) or [] if str(item)]
        if active:
            return list(dict.fromkeys(active))
    selected = data.get("selected_analysis_framework", {})
    if isinstance(selected, dict) and selected:
        summary = summarize_selected_analysis_framework(selected)
        active = [str(item) for item in summary.get("active_dimension_ids", []) or [] if str(item)]
        if active:
            return list(dict.fromkeys(active))
    active = [str(item) for item in trace_summary.get("active_analysis_dimensions", []) or [] if str(item)]
    if active:
        return list(dict.fromkeys(active))
    return list(dict.fromkeys(str(key) for key in dimension_status_map.keys() if str(key)))


def _flatten_dimension_claims(dimension_status_map: dict[str, Any], active_dimensions: list[str], key: str) -> list[str]:
    claims: list[str] = []
    dimensions = active_dimensions or [str(item) for item in dimension_status_map.keys() if str(item)]
    for dimension_id in dimensions:
        item = dimension_status_map.get(dimension_id, {})
        if isinstance(item, dict):
            claims.extend(str(claim) for claim in item.get(key, []) or [] if str(claim).strip())
    return list(dict.fromkeys(claims))


def _methodology_red_flags(data: dict[str, Any], dimension_status_map: dict[str, Any]) -> list[dict[str, Any]]:
    packet = data.get("evidence_packet", {})
    packet = packet if isinstance(packet, dict) else {}
    packet_flags = packet.get("red_flags", [])
    if isinstance(packet_flags, list) and packet_flags:
        return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in packet_flags]
    state_flags = data.get("red_flags", [])
    if isinstance(state_flags, list) and state_flags:
        return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in state_flags]
    if dimension_status_map:
        return serialize_red_flags(detect_red_flags(packet or data, dimension_status_map))
    output_flags = (data.get("output", {}) or {}).get("red_flags", []) if isinstance(data.get("output", {}), dict) else []
    if isinstance(output_flags, list):
        return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in output_flags]
    return []


def _methodology_missing_flags(data: dict[str, Any], red_flags: list[dict[str, Any]]) -> list[dict[str, Any]]:
    packet = data.get("evidence_packet", {})
    packet = packet if isinstance(packet, dict) else {}
    packet_flags = packet.get("missing_evidence_flags", [])
    if isinstance(packet_flags, list) and packet_flags:
        return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in packet_flags]
    state_flags = data.get("missing_evidence_flags", [])
    if isinstance(state_flags, list) and state_flags:
        return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in state_flags]
    return [dict(flag) for flag in red_flags if str(flag.get("category", "")) == "missing_evidence"]


def _requirements_from_plan(evidence_plan: Any) -> list[dict[str, Any]]:
    if not isinstance(evidence_plan, dict):
        return []
    return [dict(item) for item in evidence_plan.get("evidence_requirements", []) or [] if isinstance(item, dict)]


def _metric_list_from_requirement(req: dict[str, Any]) -> list[str]:
    metrics = [str(item).strip() for item in req.get("metrics", []) or [] if str(item).strip()]
    metric = str(req.get("metric") or "").strip()
    if metric:
        metrics.append(metric)
    return list(dict.fromkeys(metrics))


def _methodology_metric_gaps(
    evidence_plan: Any,
    requirement_status_map: dict[str, Any],
    evidence_packet: dict[str, Any],
    active_dimensions: list[str],
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    available: dict[str, list[str]] = {dimension_id: [] for dimension_id in active_dimensions}
    missing: dict[str, list[str]] = {dimension_id: [] for dimension_id in active_dimensions}
    for req in _requirements_from_plan(evidence_plan):
        dimension_id = str(req.get("dimension_id") or "").strip()
        if not dimension_id:
            continue
        req_type = str(req.get("requirement_type") or "").strip()
        if req_type not in {"numeric", "calculation"}:
            continue
        metrics = _metric_list_from_requirement(req)
        if not metrics:
            continue
        rid = str(req.get("requirement_id") or "").strip()
        status_entry = requirement_status_map.get(rid, {})
        status = str(status_entry.get("status") if isinstance(status_entry, dict) else status_entry or "missing")
        target = available if status == "satisfied" else missing
        for metric in metrics:
            if metric not in target.setdefault(dimension_id, []):
                target[dimension_id].append(metric)
    numeric_by_dimension = evidence_packet.get("numeric_evidence_by_dimension", {}) if isinstance(evidence_packet, dict) else {}
    if isinstance(numeric_by_dimension, dict):
        for dimension_id, rows in numeric_by_dimension.items():
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                metric = str(row.get("metric") or "").strip()
                if metric and metric not in available.setdefault(str(dimension_id), []):
                    available[str(dimension_id)].append(metric)
    return (
        {dim: values for dim, values in available.items() if values or dim in active_dimensions},
        {dim: values for dim, values in missing.items() if values or dim in active_dimensions},
    )


def _methodology_text_by_dimension(evidence_packet: dict[str, Any], active_dimensions: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {dimension_id: [] for dimension_id in active_dimensions}
    text_by_dimension = evidence_packet.get("text_evidence_by_dimension", {}) if isinstance(evidence_packet, dict) else {}
    if isinstance(text_by_dimension, dict):
        for dimension_id, rows in text_by_dimension.items():
            clean_rows: list[dict[str, Any]] = []
            for row in rows or []:
                if not isinstance(row, dict):
                    continue
                clean_rows.append(
                    {
                        "evidence_id": str(row.get("evidence_id") or ""),
                        "ticker": str(row.get("ticker") or row.get("company") or ""),
                        "section": str(row.get("section") or ""),
                        "claim": str(row.get("claim") or ""),
                        "citation_ref": str(row.get("citation_ref") or ""),
                    }
                )
            out[str(dimension_id)] = clean_rows
    if not any(out.values()):
        for row in evidence_packet.get("text_snippets", []) or []:
            if not isinstance(row, dict):
                continue
            dimension_id = str(row.get("dimension_id") or "").strip()
            if not dimension_id:
                continue
            out.setdefault(dimension_id, []).append(
                {
                    "evidence_id": str(row.get("evidence_id") or ""),
                    "ticker": str(row.get("ticker") or row.get("company") or ""),
                    "section": str(row.get("section") or ""),
                    "claim": str(row.get("claim") or ""),
                    "citation_ref": str(row.get("citation_ref") or ""),
                }
            )
    return out


# ── Endpoints ─────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        llm_provider=str(settings.llm_provider),
        llm_base_url=str(settings.llm_base_url),
        llm_reasoning_model=str(settings.llm_reasoning_model),
        analyst_draft_enabled=bool(settings.analyst_draft_enabled),
        analyst_draft_max_attempts=int(settings.analyst_draft_max_attempts or 0),
        analyst_draft_max_tokens=int(settings.analyst_draft_max_tokens or 0),
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={500: {"model": ErrorResponse}},
)
def chat(req: ChatRequest):
    """Run the LangGraph agent in a worker thread and return structured results."""
    from src.agent.graph import compile_agent

    t0 = time.time()
    trace_id = req.client_trace_id or str(uuid.uuid4())
    run_started_at = utc_now_iso()
    append_progress_event(
        trace_id,
        "run_started",
        "started",
        "已接收研究请求，正在初始化分析任务。",
        node="api_chat",
        metadata={"query": req.query},
        run_started_at=run_started_at,
    )
    try:
        agent = compile_agent()
        result = agent.invoke({"user_query": req.query, "trace_id": trace_id, "run_started_at": run_started_at})
    except Exception as exc:
        logger.exception("Agent invocation failed")
        append_progress_event(
            trace_id,
            "run_failed",
            "failed",
            "分析过程中出现错误，已记录失败节点和错误摘要。",
            node="api_chat",
            metadata={"error": str(exc)[:500]},
        )
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed = time.time() - t0
    result["trace_id"] = trace_id
    result.setdefault("run_started_at", run_started_at)
    logger.info("chat completed: trace=%s elapsed=%.1fs", trace_id, elapsed)

    _save_trace(trace_id, result)

    return ChatResponse(
        answer=result.get("final_answer", ""),
        citations=_build_citations(result.get("citations", [])),
        used_tools=result.get("selected_tools", []),
        task_type=result.get("task_type", ""),
        trace_id=trace_id,
        output=result.get("output", {}),
        contract_status=str(result.get("contract_status", result.get("final_contract_status", "not_checked"))),
        canonical_intent=dict(result.get("canonical_intent", {}) or {}),
        answer_status=str(dict(result.get("output", {}) or {}).get("answer_status", "") or ""),
        contract_decision=dict(result.get("contract_decision", result.get("contract_result", {})) or {}),
        warnings=list(dict(result.get("output", {}) or {}).get("warnings", []) or []),
        repair_attempts=int(result.get("contract_attempts", 0) or 0),
        limitations=[str(item) for item in result.get("limitations", []) or [] if str(item).strip()],
        final_answer_source=str(result.get("final_answer_source") or dict(result.get("output", {}) or {}).get("final_answer_source") or ""),
        answer_history=list(result.get("answer_history", []) or dict(result.get("output", {}) or {}).get("answer_history", []) or []),
        answer_quality_tier=str(result.get("answer_quality_tier") or dict(result.get("output", {}) or {}).get("answer_quality_tier") or ""),
        quality_tier_reason=str(result.get("quality_tier_reason") or dict(result.get("output", {}) or {}).get("quality_tier_reason") or ""),
        main_question_covered=bool(result.get("main_question_covered", dict(result.get("output", {}) or {}).get("main_question_covered", True))),
        fallback_intent_match=bool(result.get("fallback_intent_match", dict(result.get("output", {}) or {}).get("fallback_intent_match", True))),
        answered_dimensions=list(result.get("answered_dimensions", []) or dict(result.get("output", {}) or {}).get("answered_dimensions", []) or []),
        unresolved_relevance_failures=list(result.get("unresolved_relevance_failures", []) or dict(result.get("output", {}) or {}).get("unresolved_relevance_failures", []) or []),
        format_constraints_satisfied=bool(result.get("format_constraints_satisfied", dict(result.get("output", {}) or {}).get("format_constraints_satisfied", True))),
        repair_applied=bool(result.get("repair_applied", dict(result.get("output", {}) or {}).get("repair_applied", False))),
        repair_owner=str(result.get("repair_owner") or dict(result.get("output", {}) or {}).get("repair_owner") or ""),
        source_before_repair=str(result.get("source_before_repair") or dict(result.get("output", {}) or {}).get("source_before_repair") or ""),
        repair_types=list(result.get("repair_types", []) or dict(result.get("output", {}) or {}).get("repair_types", []) or []),
        material_claim_uncited_count=int(result.get("material_claim_uncited_count", dict(result.get("output", {}) or {}).get("material_claim_uncited_count", 0)) or 0),
        core_missing_parts=list(result.get("core_missing_parts", []) or dict(result.get("output", {}) or {}).get("core_missing_parts", []) or []),
        optional_missing_parts=list(result.get("optional_missing_parts", []) or dict(result.get("output", {}) or {}).get("optional_missing_parts", []) or []),
        risk_items_directly_supported_count=int(result.get("risk_items_directly_supported_count", dict(result.get("output", {}) or {}).get("risk_items_directly_supported_count", 0)) or 0),
        risk_items_template_only_count=int(result.get("risk_items_template_only_count", dict(result.get("output", {}) or {}).get("risk_items_template_only_count", 0)) or 0),
        company_specific_token_leakage=int(result.get("company_specific_token_leakage", dict(result.get("output", {}) or {}).get("company_specific_token_leakage", 0)) or 0),
        output_language=str(result.get("output_language") or dict(result.get("output", {}) or {}).get("output_language") or ""),
        language_leakage=int(result.get("language_leakage", dict(result.get("output", {}) or {}).get("language_leakage", 0)) or 0),
        language_leakage_unresolved=bool(result.get("language_leakage_unresolved", dict(result.get("output", {}) or {}).get("language_leakage_unresolved", False))),
        segment_or_product_scope=str(result.get("segment_or_product_scope") or dict(result.get("output", {}) or {}).get("segment_or_product_scope") or ""),
    )


@router.get(
    "/trace/{trace_id}/ui",
    response_model=TraceUiResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_trace_ui(trace_id: str):
    """Retrieve a sanitized trace view model for the browser audit console."""
    try:
        trace_path = trace_path_for_id(trace_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found") from None
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    data = json.loads(trace_path.read_text())
    data.setdefault("trace_id", trace_id)
    return TraceUiResponse(**build_trace_ui_model(data))


@router.get(
    "/trace/{trace_id}",
    response_model=TraceResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_trace(trace_id: str):
    """Retrieve the stored trace for a given request."""
    try:
        trace_path = trace_path_for_id(trace_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found") from None
    if not trace_path.exists():
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")

    data = json.loads(trace_path.read_text())
    evidence_plan = data.get("evidence_plan", {})
    evidence_collection_results = data.get("evidence_collection_results", [])
    evidence_sufficiency = data.get("evidence_sufficiency", {})
    requirement_summary = summarize_evidence_requirements(
        evidence_plan if isinstance(evidence_plan, dict) else {},
        evidence_collection_results if isinstance(evidence_collection_results, list) else [],
        evidence_sufficiency if isinstance(evidence_sufficiency, dict) else {},
    )
    synthesis_mode = (
        str(data.get("synthesis_mode", "") or "")
        or str((data.get("output", {}) or {}).get("synthesis_mode", "") or "")
        or str((data.get("synthesis", {}) or {}).get("synthesis_mode", "") or "")
    )
    final_answer_source = _infer_final_answer_source(data, synthesis_mode)
    evidence_plan_summary = {
        **dict(data.get("evidence_plan_summary", {}) or {}),
        **requirement_summary,
    }
    evidence_sufficiency_summary = {
        **dict(data.get("evidence_sufficiency_summary", {}) or {}),
        **requirement_summary,
    }
    requirement_status_map = _merge_requirement_entries(
        dict(data.get("requirement_status_map", {}) or {}),
        dict(requirement_summary.get("requirement_status_map", {}) or {}),
    )
    final_requirement_status_map = _merge_requirement_entries(
        dict(data.get("final_requirement_status_map", {}) or {}),
        requirement_status_map,
    )
    computed_trace_summary = build_trace_summary(
        evidence_plan if isinstance(evidence_plan, dict) else {},
        evidence_collection_results if isinstance(evidence_collection_results, list) else [],
        evidence_sufficiency if isinstance(evidence_sufficiency, dict) else {},
        synthesis_mode=synthesis_mode,
    )
    trace_summary = {
        **dict(data.get("trace_summary", {}) or {}),
        **computed_trace_summary,
    }
    if not str(trace_summary.get("final_synthesis_mode", "") or "").strip():
        trace_summary["final_synthesis_mode"] = synthesis_mode
    collection_results = data.get("collection_evidence_collection_results", [])
    collection_sufficiency = data.get("collection_evidence_sufficiency", {})
    collection_summary = summarize_evidence_requirements(
        evidence_plan if isinstance(evidence_plan, dict) else {},
        collection_results if isinstance(collection_results, list) else [],
        collection_sufficiency if isinstance(collection_sufficiency, dict) else {},
    )
    dimension_contract = normalize_dimension_status_contract(
        dict(
            requirement_summary.get("dimension_status_by_id")
            or data.get("dimension_status_by_id")
            or requirement_summary.get("dimension_status_map")
            or data.get("dimension_status_map", {})
            or {}
        ),
        satisfied_dimensions=list(
            requirement_summary.get("satisfied_dimensions")
            or data.get("satisfied_dimensions")
            or requirement_summary.get("covered_dimensions")
            or data.get("covered_dimensions", [])
            or []
        ),
        partial_dimensions=list(requirement_summary.get("partial_dimensions", data.get("partial_dimensions", [])) or []),
        missing_dimensions=list(requirement_summary.get("missing_dimensions", data.get("missing_dimensions", [])) or []),
        dimension_coverage_rate=requirement_summary.get("dimension_coverage_rate", data.get("dimension_coverage_rate")),
        weighted_dimension_coverage_rate=requirement_summary.get(
            "weighted_dimension_coverage_rate",
            data.get("weighted_dimension_coverage_rate"),
        ),
        framework_sufficiency_status=str(
            requirement_summary.get("framework_sufficiency_status", data.get("framework_sufficiency_status", "")) or ""
        ),
    )
    dimension_status_by_id = dict(dimension_contract["dimension_status_by_id"])
    dimension_status_map = dict(dimension_contract["dimension_status_map"])
    active_dimensions = _active_dimension_ids(data, trace_summary, dimension_status_map)
    red_flags = _methodology_red_flags(data, dimension_status_map)
    missing_evidence_flags = _methodology_missing_flags(data, red_flags)
    evidence_packet = data.get("evidence_packet", {})
    evidence_packet = evidence_packet if isinstance(evidence_packet, dict) else {}
    allowed_claims = list(evidence_packet.get("allowed_claims", []) or data.get("allowed_claims", []) or [])
    forbidden_claims = list(evidence_packet.get("forbidden_claims", []) or data.get("forbidden_claims", []) or [])
    if not allowed_claims:
        allowed_claims = _flatten_dimension_claims(dimension_status_map, active_dimensions, "allowed_claims")
    if not forbidden_claims:
        forbidden_claims = _flatten_dimension_claims(dimension_status_map, active_dimensions, "forbidden_claims")
    available_metrics_by_dimension, missing_metrics_by_dimension = _methodology_metric_gaps(
        evidence_plan,
        final_requirement_status_map,
        evidence_packet,
        active_dimensions,
    )
    text_evidence_by_dimension = _methodology_text_by_dimension(evidence_packet, active_dimensions)
    return TraceResponse(
        trace_id=data.get("trace_id", trace_id),
        user_query=data.get("user_query", ""),
        output_language=str(data.get("output_language") or dict(data.get("output", {}) or {}).get("output_language") or ""),
        task_type=data.get("task_type", ""),
        answer_mode=data.get("answer_mode", "direct_fact"),
        safety_intent=data.get("safety_intent", "normal"),
        query_understanding_summary=dict(
            data.get("query_understanding_summary")
            or dict(data.get("trace_summary", {}) or {}).get("query_understanding_summary", {})
            or {}
        ),
        methodology_intent=str(data.get("methodology_intent") or dict(data.get("analysis_plan", {}) or {}).get("methodology_intent", "") or ""),
        canonical_intent=dict(data.get("canonical_intent") or dict(data.get("analysis_plan", {}) or {}).get("canonical_intent", {}) or {}),
        intent_merge_decision=dict(data.get("intent_merge_decision") or dict(data.get("canonical_intent", {}) or {}).get("intent_merge_decision", {}) or {}),
        evidence_policy_id=str(data.get("evidence_policy_id") or dict(data.get("analysis_plan", {}) or {}).get("evidence_policy_id", "") or ""),
        evidence_policy=dict(data.get("evidence_policy") or dict(data.get("analysis_plan", {}) or {}).get("evidence_policy", {}) or {}),
        analysis_scope=str(data.get("analysis_scope") or dict(data.get("analysis_plan", {}) or {}).get("analysis_scope", "") or ""),
        primary_dimension=str(data.get("primary_dimension") or dict(data.get("analysis_plan", {}) or {}).get("primary_dimension", "") or ""),
        required_dimensions=[
            str(item)
            for item in (
                data.get("required_dimensions")
                or dict(data.get("analysis_plan", {}) or {}).get("required_dimensions", [])
                or []
            )
            if str(item).strip()
        ],
        optional_dimensions=[
            str(item)
            for item in (
                data.get("optional_dimensions")
                or dict(data.get("analysis_plan", {}) or {}).get("optional_dimensions", [])
                or []
            )
            if str(item).strip()
        ],
        supporting_context_dimensions=[
            str(item)
            for item in (
                data.get("supporting_context_dimensions")
                or trace_summary.get("supporting_context_dimensions", [])
                or []
            )
            if str(item).strip()
        ],
        time_policy=str(data.get("time_policy") or dict(data.get("analysis_plan", {}) or {}).get("time_policy", "") or ""),
        period_scope=str(data.get("period_scope") or dict(data.get("analysis_plan", {}) or {}).get("period_scope", "") or ""),
        needs_clarification=bool(data.get("needs_clarification", False)),
        clarification_question=data.get("clarification_question"),
        needs_tools=bool(data.get("needs_tools", True)),
        data_route=data.get("data_route", ""),
        analysis_plan_raw=data.get("analysis_plan_raw", {}),
        analysis_plan=data.get("analysis_plan", {}),
        selected_analysis_framework=data.get("selected_analysis_framework", {}),
        research_plan_raw=data.get("research_plan_raw", {}),
        research_plan_validated=data.get("research_plan_validated", {}),
        research_plan_used=data.get("research_plan_used", {}),
        research_plan_validation=data.get("research_plan_validation", {}),
        research_plan_source=str(data.get("research_plan_source", "")),
        research_plan_fallback_reason=str(data.get("research_plan_fallback_reason", "")),
        research_plan_duration_ms=int(data.get("research_plan_duration_ms", 0) or 0),
        required_answer_parts=data.get("required_answer_parts", []),
        legacy_evidence_plan=data.get("legacy_evidence_plan", {}),
        plan_coverage_decision=data.get("plan_coverage_decision", {}),
        requirement_merge_summary=data.get("requirement_merge_summary", {}),
        evidence_plan_used=data.get("evidence_plan_used", {}),
        rejected_plan_items=data.get("rejected_plan_items", []),
        validated_tools=data.get("validated_tools", []),
        safety_decision=data.get("safety_decision", {}),
        safety_policy_reasons=data.get("safety_policy_reasons", []),
        safety_limitations=data.get("safety_limitations", []),
        evidence_plan=evidence_plan,
        evidence_plan_summary=evidence_plan_summary,
        evidence_requirements=list((evidence_plan or {}).get("evidence_requirements", []) or []) if isinstance(evidence_plan, dict) else [],
        evidence_collection_results=evidence_collection_results,
        evidence_sufficiency=evidence_sufficiency,
        evidence_sufficiency_summary=evidence_sufficiency_summary,
        answer_part_status_by_id=dict(requirement_summary.get("answer_part_status_by_id", data.get("answer_part_status_by_id", {})) or {}),
        evidence_gap_by_answer_part=dict(requirement_summary.get("evidence_gap_by_answer_part", data.get("evidence_gap_by_answer_part", {})) or {}),
        missing_required_answer_parts=list(requirement_summary.get("missing_required_answer_parts", data.get("missing_required_answer_parts", [])) or []),
        partial_required_answer_parts=list(requirement_summary.get("partial_required_answer_parts", data.get("partial_required_answer_parts", [])) or []),
        missing_but_analyzable_answer_parts=list(requirement_summary.get("missing_but_analyzable_answer_parts", data.get("missing_but_analyzable_answer_parts", [])) or []),
        missing_and_unanswerable_answer_parts=list(requirement_summary.get("missing_and_unanswerable_answer_parts", data.get("missing_and_unanswerable_answer_parts", [])) or []),
        evidence_health=str(requirement_summary.get("evidence_health", data.get("evidence_health", "")) or ""),
        tool_error_context=list(requirement_summary.get("tool_error_context", data.get("tool_error_context", [])) or []),
        collection_evidence_collection_results=collection_results if isinstance(collection_results, list) else [],
        collection_evidence_sufficiency=collection_sufficiency if isinstance(collection_sufficiency, dict) else {},
        collection_evidence_sufficiency_summary={
            **dict(data.get("collection_evidence_sufficiency_summary", {}) or {}),
            **collection_summary,
        },
        evidence_retry_history=data.get("evidence_retry_history", []),
        retry_history=data.get("retry_history", data.get("evidence_retry_history", [])),
        requirement_limitations=list(requirement_summary.get("requirement_limitations", data.get("requirement_limitations", [])) or []),
        collected_evidence_by_requirement=_merge_requirement_entries(
            dict(data.get("collected_evidence_by_requirement", {}) or {}),
            dict(requirement_summary.get("collected_evidence_by_requirement", {}) or {}),
        ),
        requirement_status_map=requirement_status_map,
        selected_framework=_selected_framework_id(data, trace_summary),
        active_dimensions=active_dimensions,
        dimension_status_by_id=dimension_status_by_id,
        dimension_status_map=dimension_status_map,
        satisfied_dimensions=list(dimension_contract["satisfied_dimensions"]),
        covered_dimensions=list(dimension_contract["covered_dimensions"]),
        partial_dimensions=list(dimension_contract["partial_dimensions"]),
        missing_dimensions=list(dimension_contract["missing_dimensions"]),
        dimension_coverage_rate=dimension_contract["dimension_coverage_rate"],
        weighted_dimension_coverage_rate=dimension_contract["weighted_dimension_coverage_rate"],
        framework_sufficiency_status=str(dimension_contract["framework_sufficiency_status"] or ""),
        red_flags=red_flags,
        missing_evidence_flags=missing_evidence_flags,
        forbidden_claims=[str(item) for item in forbidden_claims if str(item).strip()],
        allowed_claims=[str(item) for item in allowed_claims if str(item).strip()],
        available_metrics_by_dimension=available_metrics_by_dimension,
        missing_metrics_by_dimension=missing_metrics_by_dimension,
        text_evidence_by_dimension=text_evidence_by_dimension,
        final_methodology_coverage_rate=requirement_summary.get(
            "dimension_coverage_rate",
            data.get("final_methodology_coverage_rate", data.get("dimension_coverage_rate")),
        ),
        final_requirement_status_map=final_requirement_status_map,
        collection_requirement_status_map=_merge_requirement_entries(
            dict(data.get("collection_requirement_status_map", {}) or {}),
            dict(collection_summary.get("requirement_status_map", {}) or {}),
        ),
        evidence_validation_records=list(data.get("evidence_validation_records", []) or []),
        trace_summary=trace_summary,
        collection_trace_summary={
            **dict(data.get("collection_trace_summary", {}) or {}),
            **build_trace_summary(
                evidence_plan if isinstance(evidence_plan, dict) else {},
                collection_results if isinstance(collection_results, list) else [],
                collection_sufficiency if isinstance(collection_sufficiency, dict) else {},
                synthesis_mode=str((data.get("collection_trace_summary", {}) or {}).get("final_synthesis_mode", "") or ""),
            ),
        },
        missing_requirements=list(requirement_summary.get("missing_requirements", []) or []),
        degradation_reason=requirement_summary.get("degradation_reason"),
        validated_requirement_ids=list(data.get("validated_requirement_ids", []) or []),
        validated_numeric_evidence_count=int(data.get("validated_numeric_evidence_count", len(data.get("numeric_evidence", []) or [])) or 0),
        validated_text_evidence_count=int(data.get("validated_text_evidence_count", len(data.get("text_evidence", []) or [])) or 0),
        raw_retrieval_hits_by_requirement=dict(data.get("raw_retrieval_hits_by_requirement", {}) or {}),
        text_requirement_diagnostics=dict(data.get("text_requirement_diagnostics", {}) or {}),
        rejected_requirements=data.get("rejected_requirements", []),
        evidence_packet=evidence_packet,
        evidence_packet_summary=data.get("evidence_packet_summary", {}),
        comparison_judgment_frame=data.get("comparison_judgment_frame", {}),
        analyst_draft=data.get("analyst_draft", {}),
        analyst_draft_validation=data.get("analyst_draft_validation", {}),
        draft_validation=data.get("draft_validation", data.get("analyst_draft_validation", {})),
        draft_attempts=data.get("draft_attempts", []),
        draft_revision_attempts=data.get(
            "draft_revision_attempts",
            [
                item
                for item in data.get("draft_attempts", []) or []
                if int(item.get("attempt_index", item.get("attempt_number", 0)) or 0) > 1
            ],
        ),
        draft_violations=data.get("draft_violations", []),
        draft_final_status=str(data.get("draft_final_status", "")),
        draft_status=str(data.get("draft_status", "")),
        final_answer_source=final_answer_source,
        answer_history=list(data.get("answer_history", []) or dict(data.get("output", {}) or {}).get("answer_history", []) or []),
        answer_candidate=dict(data.get("answer_candidate", {}) or dict(data.get("output", {}) or {}).get("answer_candidate", {}) or {}),
        answer_candidates=list(data.get("answer_candidates", []) or dict(data.get("output", {}) or {}).get("answer_candidates", []) or []),
        draft_release_decision=dict(data.get("draft_release_decision", {}) or {}),
        synthesis=data.get("synthesis", {}),
        synthesis_strategy=data.get("synthesis_strategy", ""),
        synthesis_mode=synthesis_mode,
        analytical_claims=list(data.get("analytical_claims", []) or dict(data.get("synthesis", {}) or {}).get("analytical_claims", []) or []),
        claim_tiers=dict(data.get("claim_tiers", {}) or dict(data.get("synthesis", {}) or {}).get("claim_tiers", {}) or {}),
        analytical_reasoning_status=str(data.get("analytical_reasoning_status", "") or dict(data.get("synthesis", {}) or {}).get("analytical_reasoning_status", "")),
        unsupported_synthesis_items=data.get("unsupported_synthesis_items", []),
        why_tools_skipped=data.get("why_tools_skipped", []),
        companies=data.get("companies", []),
        comparison_target=data.get("comparison_target"),
        time_range=data.get("time_range"),
        period_query=data.get("period_query", {}),
        resolved_period_context=data.get("resolved_period_context", {}),
        comparison_basis_label=data.get("comparison_basis_label", ""),
        requested_metrics=data.get("requested_metrics", []),
        selected_tools=data.get("selected_tools", []),
        retrieval_policy=data.get("retrieval_policy", {}),
        retrieval_debug=data.get("retrieval_debug", {}),
        event_intent=str(data.get("event_intent", "none")),
        market_reaction_requested=bool(data.get("market_reaction_requested", False)),
        event_query=data.get("event_query", {}),
        event_results=data.get("event_results", []),
        market_reaction_evidence=data.get("market_reaction_evidence", []),
        market_reaction_limitations=data.get("market_reaction_limitations", []),
        tool_results=data.get("tool_results", []),
        numeric_evidence=data.get("numeric_evidence", []),
        text_evidence=data.get("text_evidence", []),
        unsupported_claims=data.get("unsupported_claims", []),
        numeric_citations=data.get("numeric_citations", []),
        text_citations=data.get("text_citations", []),
        citations=data.get("citations", []),
        output=data.get("output", {}),
        structured_sources=data.get("structured_sources", []),
        document_citations=data.get("document_citations", []),
        contract_result=data.get("contract_result", {}),
        contract_decision=data.get("contract_decision", data.get("contract_result", {})),
        contract_status=str(data.get("contract_status", "not_checked")),
        contract_attempts=int(data.get("contract_attempts", 0) or 0),
        repair_actions=data.get("repair_actions", []),
        final_contract_status=str(data.get("final_contract_status", data.get("contract_status", "not_checked"))),
        contract_public_summary=str(data.get("contract_public_summary", "")),
        contract_evidence_retry_count=int(data.get("contract_evidence_retry_count", 0) or 0),
        relevance_decision=dict(data.get("relevance_decision", {}) or {}),
        relevance_status=str(data.get("relevance_status", "not_run") or "not_run"),
        relevance_repair_attempts=int(data.get("relevance_repair_attempts", data.get("relevance_attempts", 0)) or 0),
        final_route=str(data.get("final_route", "") or dict(data.get("output", {}) or {}).get("final_route", "")),
        answer_quality_tier=str(data.get("answer_quality_tier") or dict(data.get("output", {}) or {}).get("answer_quality_tier") or ""),
        quality_tier_reason=str(data.get("quality_tier_reason") or dict(data.get("output", {}) or {}).get("quality_tier_reason") or ""),
        main_question_covered=bool(data.get("main_question_covered", dict(data.get("output", {}) or {}).get("main_question_covered", True))),
        fallback_intent_match=bool(data.get("fallback_intent_match", dict(data.get("output", {}) or {}).get("fallback_intent_match", True))),
        answered_dimensions=list(data.get("answered_dimensions", []) or dict(data.get("output", {}) or {}).get("answered_dimensions", []) or []),
        unresolved_relevance_failures=list(data.get("unresolved_relevance_failures", []) or dict(data.get("output", {}) or {}).get("unresolved_relevance_failures", []) or []),
        format_constraints_satisfied=bool(data.get("format_constraints_satisfied", dict(data.get("output", {}) or {}).get("format_constraints_satisfied", True))),
        format_constraints=dict(data.get("format_constraints", {}) or dict(data.get("output", {}) or {}).get("format_constraints", {}) or {}),
        repair_applied=bool(data.get("repair_applied", dict(data.get("output", {}) or {}).get("repair_applied", False))),
        repair_owner=str(data.get("repair_owner") or dict(data.get("output", {}) or {}).get("repair_owner") or ""),
        source_before_repair=str(data.get("source_before_repair") or dict(data.get("output", {}) or {}).get("source_before_repair") or ""),
        repair_types=list(data.get("repair_types", []) or dict(data.get("output", {}) or {}).get("repair_types", []) or []),
        material_claim_uncited_count=int(data.get("material_claim_uncited_count", dict(data.get("output", {}) or {}).get("material_claim_uncited_count", 0)) or 0),
        core_missing_parts=list(data.get("core_missing_parts", []) or dict(data.get("output", {}) or {}).get("core_missing_parts", []) or []),
        optional_missing_parts=list(data.get("optional_missing_parts", []) or dict(data.get("output", {}) or {}).get("optional_missing_parts", []) or []),
        risk_items_directly_supported_count=int(data.get("risk_items_directly_supported_count", dict(data.get("output", {}) or {}).get("risk_items_directly_supported_count", 0)) or 0),
        risk_items_template_only_count=int(data.get("risk_items_template_only_count", dict(data.get("output", {}) or {}).get("risk_items_template_only_count", 0)) or 0),
        company_specific_token_leakage=int(data.get("company_specific_token_leakage", dict(data.get("output", {}) or {}).get("company_specific_token_leakage", 0)) or 0),
        language_leakage=int(data.get("language_leakage", dict(data.get("output", {}) or {}).get("language_leakage", 0)) or 0),
        language_leakage_unresolved=bool(data.get("language_leakage_unresolved", dict(data.get("output", {}) or {}).get("language_leakage_unresolved", False))),
        segment_or_product_scope=str(data.get("segment_or_product_scope") or dict(data.get("output", {}) or {}).get("segment_or_product_scope") or ""),
        report=data.get("report", {}),
        report_contract_result=data.get("report_contract_result", {}),
        report_contract_status=str(data.get("report_contract_status", "")),
        evidence_loop_count=data.get("evidence_loop_count", 0),
        final_answer=data.get("final_answer", ""),
    )
