"""Requirement-based tool execution for conversational analyst paths."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from config import settings
from src.agent.constants import ALLOWED_ANALYSIS_METRICS, KNOWN_SEC_SECTIONS, PERIOD_TYPES
from src.agent.evidence_planner import normalize_planner_metric
from src.agent.evidence_sufficiency import collection_result, evaluate_evidence_sufficiency
from src.agent.metric_availability import normalize_metric_name
from src.agent.progress import append_progress_event
from src.agent.state import AgentState
from src.agent.types import TextEvidenceQuality
from src.tools.compute_metrics import compute_metrics
from src.tools.query_event_price_window import query_event_price_window
from src.tools.query_financial_data import query_financial_data
from src.tools.protocol import ToolExecutionContext, ToolResult
from src.tools.registry import build_default_tool_registry
from src.tools.search_filings import search_filings, search_filings_lexical_fallback

logger = logging.getLogger(__name__)

NO_TOOL_MODES = {"meta", "clarification", "refusal_or_redirect"}
LOW_QUALITY = {"low", "mixed"}
_TEXT_SUPPORT_STOPWORDS = {
    "about",
    "and",
    "are",
    "business",
    "company",
    "discussion",
    "filing",
    "from",
    "include",
    "includes",
    "item",
    "management",
    "operating",
    "results",
    "risk",
    "risks",
    "section",
    "that",
    "the",
    "this",
    "with",
}
TEXT_QUALITY_SEMANTIC_TERMS = {
    "revenue",
    "sales",
    "segment",
    "segments",
    "aws",
    "north america",
    "international",
    "operating income",
    "cash flow",
    "management",
    "discussion",
    "uncertainty",
    "regulation",
    "constraints",
    "challenges",
    "headwinds",
    "competition",
    "competitive",
    "inventory",
    "fulfillment",
    "risk",
    "risks",
    "business",
    "service",
    "services",
    "product",
    "products",
    "customer",
    "customers",
    "marketplace",
    "advertising",
    "prime",
    "demand",
    "supply",
    "margin",
    "profit",
}
GENERIC_TEXT_SUMMARIES = (
    "provides business and risk context",
    "relevant to comparison",
    "contains information",
    "provides context",
)


def _progress(
    state: AgentState,
    event: str,
    status: str,
    message: str,
    *,
    metadata: Mapping[str, Any] | None = None,
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
        metadata=dict(metadata or {}),
    )
_GENERIC_INTENT_QUERIES = {
    "biggest_problem": "operating challenges demand weakness margin pressure competitive pressure",
    "business_pressure": "operating challenges demand weakness margin pressure competitive pressure",
    "major_risks": "risk factors competition regulation supply chain demand uncertainty",
    "management_concern": "management discussion operating challenges demand softness execution headwinds",
    "comparison_risk": "risk factors operating challenges competitive pressure",
    "comparison_risk_context": "competition risk factors business risks competitive pressure",
    "comparison_key_difference": "business model operating leverage margin profile demand drivers",
    "comparison_context": "business context competitive position operating leverage",
    "single_company_operating_context": "management discussion operating results revenue margin",
    "single_company_business_model": "business overview products services revenue sources customers markets segments net sales reportable segments",
    "single_company_risk_context": "risk factors competition demand supply chain regulation customer concentration",
    "single_company_competition_context": "competitive position market position products customers industry competition",
    "risk_focused_risk_factors": "risk factors competition demand supply chain margin customer regulation",
    "risk_focused_management_context": "management discussion operating results demand margin revenue risk challenges",
    "growth_driver_text": "management discussion operating results revenue increased growth driven demand segment revenue",
    "revenue_growth_numeric": "revenue growth operating results",
    "business_model": "business overview products services revenue sources segments net sales reportable segments",
    "moat_and_competitive_risk": "risk factors competition demand uncertainty",
}


def _tool_registry():
    return build_default_tool_registry(
        {
            "query_financial_data": query_financial_data,
            "search_filings": search_filings,
            "compute_metrics": compute_metrics,
            "query_event_price_window": query_event_price_window,
        }
    )


def _tool_input_summary(tool_name: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if tool_name == "query_financial_data":
        return {"ticker": payload.get("ticker"), "metrics": payload.get("metrics"), "period_type": payload.get("period_type")}
    if tool_name == "search_filings":
        return {
            "ticker": payload.get("ticker"),
            "query": payload.get("query"),
            "top_k": payload.get("top_k"),
            "section_allowlist": payload.get("section_allowlist"),
            "strict_sections": payload.get("strict_sections"),
        }
    if tool_name == "compute_metrics":
        return {
            "computation": payload.get("computation"),
            "input_count": len(list(payload.get("data", []) or [])),
            "denominator_count": len(list(payload.get("denominator_data", []) or [])),
        }
    if tool_name == "query_event_price_window":
        return {
            "ticker": payload.get("ticker"),
            "event_type": payload.get("event_type"),
            "latest_n": payload.get("latest_n"),
            "window_days": payload.get("window_days"),
        }
    return {}


def _run_protocol_tool(tool_name: str, payload: dict[str, Any], context: Mapping[str, Any], req: Mapping[str, Any]) -> ToolResult:
    registry = _tool_registry()
    tool = registry.get(tool_name)
    execution_context = ToolExecutionContext(
        trace_id=str(context.get("trace_id", "")),
        requirement_id=str(req.get("requirement_id", "")),
        company=str(req.get("company", "")),
        dimension=str(req.get("dimension_id") or req.get("dimension") or ""),
        metadata={"user_query": context.get("user_query", "")},
    )
    result = registry.execute(tool_name, payload, execution_context)
    trace_item = result.trace_summary(
        tool_version=tool.spec.version,
        input_summary=_tool_input_summary(tool_name, payload),
        requirement_id=str(req.get("requirement_id", "")),
    )
    calls = context.setdefault("tool_call_results", []) if isinstance(context, dict) else None
    if isinstance(calls, list):
        trace_item["tool_call_id"] = f"tc_{len(calls) + 1:03d}"
        calls.append(trace_item)
    return result


def _raise_tool_error(result: ToolResult) -> None:
    if result.ok:
        return
    message = result.error.message if result.error else "unknown tool error"
    raise RuntimeError(message)


def _search_fallback_reason(result: ToolResult) -> tuple[str, str, str] | None:
    if result.ok or result.error is None:
        return None
    code = str(result.error.code or "")
    message = str(result.error.message or "")
    lowered = message.lower()
    if code == "timeout":
        return "timeout", code, message
    resource_markers = (
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "resource exhausted",
        "gpu memory",
        "memory allocation",
        "alloc failed",
        "oom",
    )
    if code in {"execution_error", "resource_exhausted"} and any(marker in lowered for marker in resource_markers):
        return "resource_error", code, message
    return None


def _requirements(evidence_plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(r) for r in evidence_plan.get("evidence_requirements", []) or [] if isinstance(r, Mapping)]


def _requirements_of_type(requirements: list[dict[str, Any]], requirement_type: str) -> list[dict[str, Any]]:
    return [r for r in requirements if str(r.get("requirement_type", "")) == requirement_type]


def _tools_from_requirements(requirements: list[dict[str, Any]], state: Mapping[str, Any]) -> list[str]:
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
    return list(dict.fromkeys(tools))


def _empty_result(
    state: Mapping[str, Any],
    evidence_plan: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    sufficiency = evaluate_evidence_sufficiency(evidence_plan, []).model_dump(exclude_none=True)
    return {
        "tool_results": [],
        "retrieved_docs": [],
        "event_results": list(state.get("event_results", []) or []),
        "evidence_collection_results": [],
        "evidence_validation_records": [],
        "evidence_sufficiency": sufficiency,
        "requirement_calls": [],
        "tool_call_results": [],
        "retry_history": [],
        "requirement_limitations": list(sufficiency.get("requirement_limitations", []) or []),
        "selected_tools": [],
        "validated_tools": [],
        "market_reaction_limitations": list(state.get("market_reaction_limitations", []) or []),
        "why_tools_skipped": [{"reason": reason, "message": "requirement_executor_skipped"}],
        "retrieval_debug": {
            "requirement_calls": [],
            "requirement_retry_history": [],
            "search_calls": [],
            "event_calls": [],
            "search_skipped": [{"reason": reason}],
            "tool_call_results": [],
        },
    }


def _period_type(req: Mapping[str, Any], period_query: Mapping[str, Any]) -> str:
    raw = str(req.get("period_type") or period_query.get("period_type") or "latest")
    if raw == "ttm":
        return "trailing"
    if raw not in PERIOD_TYPES:
        return "latest"
    return raw


def _metrics(req: Mapping[str, Any]) -> list[str]:
    metrics: list[str] = []
    for raw in list(req.get("metrics", []) or []) + [req.get("metric")]:
        raw_metric = str(raw or "").strip()
        metric = "price" if raw_metric == "price" else (normalize_planner_metric(raw_metric) or raw_metric)
        if metric in ALLOWED_ANALYSIS_METRICS and metric not in metrics:
            metrics.append(metric)
    return metrics or ["revenue", "net_income"]


def _price_metric_alias(metric: str) -> str:
    if metric in {"price", "share_price"}:
        return "adjusted_close"
    return metric


def _normalize_numeric_item(row: Mapping[str, Any], req: Mapping[str, Any], company: str) -> dict[str, Any]:
    period_end = str(row.get("period_end") or row.get("date") or "")
    role = str(req.get("evidence_role") or "")
    rid = str(req.get("requirement_id", ""))
    item = dict(row)
    item["requirement_id"] = rid
    item["source_requirement_id"] = rid
    item["company"] = str(item.get("company") or item.get("ticker") or company).upper()
    item["ticker"] = str(item.get("ticker") or company).upper()
    item["metric"] = str(item.get("metric") or "")
    item["period"] = period_end
    item["period_end"] = period_end
    item["period_type"] = str(item.get("period_type") or req.get("period_type") or "")
    item["period_scope"] = item["period_type"] or "unknown"
    item["value"] = item.get("value")
    item["unit"] = str(item.get("unit") or "")
    if role:
        item["role"] = role
        item["evidence_role"] = role
    item["source_provider"] = str(item.get("source_provider") or "")
    item["source_url"] = str(item.get("source_url") or "")
    item["source_filing_id"] = str(item.get("source_filing_id") or "")
    item["confidence"] = str(item.get("confidence") or "")
    item["extraction_method"] = str(item.get("extraction_method") or "")
    item["source_tag"] = str(item.get("source_tag") or "")
    item["reconciliation_warning"] = str(item.get("reconciliation_warning") or "")
    if not item["source_provider"] and item.get("source_tool") == "query_financial_data":
        item["source_provider"] = "structured"
    return item


def _normalize_price_items(row: Mapping[str, Any], req: Mapping[str, Any], company: str) -> list[dict[str, Any]]:
    requested = {_price_metric_alias(metric) for metric in _metrics(req)}
    out: list[dict[str, Any]] = []
    row_date = str(row.get("date") or row.get("period_end") or "")
    for metric in ("open", "high", "low", "close", "adjusted_close", "volume"):
        if metric not in requested or row.get(metric) is None:
            continue
        item = dict(row)
        item["requirement_id"] = str(req.get("requirement_id", ""))
        item["source_requirement_id"] = str(req.get("requirement_id", ""))
        item["company"] = str(item.get("company") or item.get("ticker") or company).upper()
        item["ticker"] = str(item.get("ticker") or company).upper()
        item["metric"] = metric
        item["period"] = row_date
        item["period_end"] = row_date
        item["period_type"] = "daily"
        item["value"] = item.get(metric)
        item["unit"] = ""
        item["source_provider"] = str(item.get("source_provider") or "yfinance")
        item["source_url"] = str(item.get("source_url") or "")
        item["source_filing_id"] = str(item.get("source_filing_id") or "")
        item["confidence"] = str(item.get("confidence") or "medium")
        item["extraction_method"] = str(item.get("extraction_method") or "api_price_history")
        item["source_tag"] = str(item.get("source_tag") or "")
        item["reconciliation_warning"] = str(item.get("reconciliation_warning") or "")
        out.append(item)
    return out


def _numeric_payload(req: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    period_query = dict(context.get("period_query", {}) or {})
    resolved = dict(context.get("resolved_period_context", {}) or {})
    period_type = _period_type(req, period_query)
    return {
        "ticker": str(req.get("company") or "").upper(),
        "metrics": _metrics(req),
        "period_type": period_type,
        "target_period_type": resolved.get("target_period_type"),
        "year": period_query.get("year"),
        "quarter": period_query.get("quarter"),
        "trailing_n": period_query.get("trailing_n"),
        "year_basis": period_query.get("year_basis"),
        "comparison_basis": period_query.get("comparison_basis"),
        "strict_period_match": bool(resolved.get("strict_period_match", True)),
        "date_start": context.get("date_start"),
        "date_end": context.get("date_end"),
        "limit": max(20, int(req.get("min_results", 1) or 1)),
    }


def _execute_numeric_requirement(req: dict[str, Any], context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    company = str(req.get("company") or "").upper()
    payload = _numeric_payload(req, context)
    requested_metrics = {_price_metric_alias(metric) for metric in _metrics(req)}
    try:
        protocol_result = _run_protocol_tool("query_financial_data", payload, context, req)
        _raise_tool_error(protocol_result)
        result = protocol_result.data
        data = dict(result or {}) if isinstance(result, Mapping) else {}
        items: list[dict[str, Any]] = []
        facts = list(data.get("financial_facts", []) or [])
        prices = list(data.get("price_data", []) or [])
        raw_returned_count = len(facts) + len(prices)
        rejected_reason = ""
        rejected_count = 0
        for row in facts:
            if isinstance(row, Mapping):
                item = _normalize_numeric_item(row, req, company)
                normalized_metric = normalize_planner_metric(str(item.get("metric") or "")) or str(item.get("metric") or "")
                item["metric"] = normalized_metric
                if normalized_metric in requested_metrics:
                    items.append(item)
                else:
                    rejected_count += 1
        price_items: list[dict[str, Any]] = []
        for row in prices:
            if isinstance(row, Mapping):
                price_items.extend(_normalize_price_items(row, req, company))
        items.extend(price_items)
        if raw_returned_count and not items:
            rejected_reason = "metric_mapping_failed" if rejected_count else "evidence_filter_mismatch"
        data["financial_facts"] = [item for item in items if str(item.get("metric", "")) not in {"open", "high", "low", "close", "adjusted_close", "volume"}]
        data["price_data"] = price_items
        status = "satisfied" if len(items) >= int(req.get("min_results", 1) or 1) else ("partial" if items else "missing")
        failure_reason = None if status == "satisfied" else ("below_min_results" if items else (rejected_reason or "no_matching_evidence"))
        tool_result = {
            "tool": "query_financial_data",
            "ticker": company,
            "requirement_id": rid,
            "count": len(items),
            "tool_returned_count": raw_returned_count,
            "validated_evidence_count": len(items),
            "rejected_evidence_reason": rejected_reason,
            "data": data,
        }
        return tool_result, collection_result(
            requirement_id=rid,
            status=status,
            evidence_type="numeric",
            items=items,
            failure_reason=failure_reason,
            company=company,
            evidence_role=str(req.get("evidence_role") or ""),
            tool_returned_count=raw_returned_count,
            validated_evidence_count=len(items),
            rejected_evidence_reason=rejected_reason,
        )
    except Exception as exc:  # pragma: no cover - explicitly tested with fakes
        logger.warning("query_financial_data failed for requirement %s: %s", rid, exc)
        return (
            {"tool": "query_financial_data", "ticker": company, "requirement_id": rid, "error": str(exc)},
            collection_result(
                requirement_id=rid,
                status="missing",
                evidence_type="numeric",
                items=[],
                failure_reason=f"query_financial_data_error:{exc}",
                company=company,
                evidence_role=str(req.get("evidence_role") or ""),
            ),
        )


def _causal_revenue_role_ids(requirements: list[dict[str, Any]], role: str) -> set[str]:
    return {
        str(req.get("requirement_id") or "")
        for req in requirements
        if str(req.get("evidence_role") or "") == role and str(req.get("requirement_id") or "")
    }


def _apply_causal_revenue_quality(
    *,
    requirements: list[dict[str, Any]],
    collection_results: list[dict[str, Any]],
    tool_results: list[dict[str, Any]],
) -> None:
    """Invalidate same-period comparator revenue before growth is computed."""
    current_ids = _causal_revenue_role_ids(requirements, "current_revenue")
    comparator_ids = _causal_revenue_role_ids(requirements, "comparator_revenue")
    if not current_ids or not comparator_ids:
        return
    current_items = [
        item
        for result in collection_results
        if str(result.get("requirement_id") or "") in current_ids and str(result.get("status") or "") in {"satisfied", "partial"}
        for item in result.get("items", []) or []
        if isinstance(item, Mapping) and str(item.get("metric") or "") == "revenue" and _row_period(item)
    ]
    if not current_items:
        return
    current = sorted(current_items, key=lambda item: _row_period(item))[-1]

    def invalid_reason(item: Mapping[str, Any]) -> str:
        if _same_period(current, item):
            return "same_period_comparator"
        if not _comparable_period_scope(current, item):
            return "incomparable_period_scope"
        if _safe_float(item.get("value")) == 0:
            return "zero_comparator"
        return ""

    invalid_by_req: dict[str, str] = {}
    for result in collection_results:
        rid = str(result.get("requirement_id") or "")
        if rid not in comparator_ids:
            continue
        raw_items = [dict(item) for item in result.get("items", []) or [] if isinstance(item, Mapping)]
        valid_items = [item for item in raw_items if not invalid_reason(item)]
        if valid_items:
            for item in valid_items:
                item["quality_status"] = "valid"
                item["role"] = "comparator_revenue"
                item["evidence_role"] = "comparator_revenue"
            result["items"] = valid_items
            result["status"] = "satisfied" if len(valid_items) >= int(result.get("min_results", 1) or 1) else "partial"
            result.pop("failure_reason", None)
            result["quality_status"] = "valid"
            continue
        reason = next((invalid_reason(item) for item in raw_items if invalid_reason(item)), "dependency_numeric_requirement_missing")
        invalid_by_req[rid] = reason
        result["items"] = []
        result["status"] = "missing"
        result["failure_reason"] = reason
        result["quality_status"] = reason
        result["current_revenue"] = _dependency_record(current, role="current_revenue")
    if not invalid_by_req:
        return
    for tool_result in tool_results:
        if tool_result.get("tool") != "query_financial_data":
            continue
        data = tool_result.get("data")
        if not isinstance(data, dict):
            continue
        for row in data.get("financial_facts", []) or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("requirement_id") or tool_result.get("requirement_id") or "")
            reason = invalid_by_req.get(rid)
            if not reason:
                continue
            row["quality_status"] = reason
            row["validation_failure_reason"] = reason
            row["exclude_from_evidence_matrix"] = True


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 2}


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def _section_anchor_terms(section: str) -> set[str]:
    normalized = str(section or "").upper().strip()
    if normalized == "ITEM_1A":
        return {"risk", "factors", "competition", "regulation", "uncertainty"}
    if normalized == "ITEM_7":
        return {"management", "discussion", "operating", "results", "demand", "margin"}
    if normalized == "ITEM_1":
        return {"business", "overview", "segments", "competition", "sales", "products", "services", "customers"}
    if normalized == "BUSINESS":
        return {"business", "overview", "segments", "competition", "sales", "products", "services", "customers"}
    if normalized == "ITEM_2":
        return {"operating", "results", "quarter", "demand", "pressure"}
    if normalized in {"MD&A", "MDA"}:
        return {"management", "discussion", "operating", "results", "margin"}
    return set()


def validate_text_evidence_quality(evidence: Mapping[str, Any]) -> TextEvidenceQuality:
    snippet = re.sub(
        r"\s+",
        " ",
        str(
            evidence.get("supporting_snippet")
            or evidence.get("text_snippet")
            or evidence.get("snippet")
            or evidence.get("text")
            or ""
        ),
    ).strip()
    summary = re.sub(r"\s+", " ", str(evidence.get("claim") or evidence.get("summary") or evidence.get("theme_name") or "")).strip()
    lowered = snippet.lower()
    snippet_length = len(snippet)
    is_header_only = bool(re.fullmatch(r"(?i)(?:part\s+[ivx]+\.?|item\s+\d+[a-z]?\.?|item\s+\d+[a-z]?\s*[.-]?\s*)", snippet.strip()))
    semantic_count = 0
    for term in TEXT_QUALITY_SEMANTIC_TERMS:
        if term in lowered:
            semantic_count += 1
    specificity = min(1.0, (semantic_count / 6.0) + min(snippet_length, 240) / 480.0)
    reason = ""
    if not snippet:
        reason = "snippet_support_failed"
    elif is_header_only:
        reason = "low_information_text_evidence"
    elif snippet_length < 80 and semantic_count < 2:
        reason = "low_information_text_evidence"
    elif semantic_count == 0:
        reason = "low_information_text_evidence"
    elif summary and any(item in summary.lower() for item in GENERIC_TEXT_SUMMARIES):
        reason = "low_information_text_evidence"
    return TextEvidenceQuality(
        is_valid=not bool(reason),
        reason=reason,
        snippet_length=snippet_length,
        semantic_term_count=semantic_count,
        is_section_header_only=is_header_only,
        specificity_score=round(float(specificity), 4),
    )


def _fallback_supporting_terms(snippet: str, section: str, ticker: str) -> list[str]:
    tokens = [
        tok
        for tok in re.findall(r"[a-z0-9][a-z0-9&._-]*", str(snippet or "").lower())
        if len(tok) > 3 and tok not in _TEXT_SUPPORT_STOPWORDS
    ]
    ranked = _ordered_unique(tokens)
    for anchor in sorted(_section_anchor_terms(section)):
        if anchor not in ranked:
            ranked.append(anchor)
    ticker_token = str(ticker or "").lower().strip()
    if ticker_token and ticker_token not in ranked:
        ranked.insert(0, ticker_token)
    return ranked[:12]


def _doc_stat(docs: list[Mapping[str, Any]], key: str, fallback: int = 0) -> int:
    values = [int(doc.get(key, 0) or 0) for doc in docs if isinstance(doc, Mapping)]
    if any(value > 0 for value in values):
        return max(values)
    return int(fallback or 0)


def _generic_query(req: Mapping[str, Any]) -> str:
    intent = str(req.get("retrieval_intent", "")).strip()
    if intent in _GENERIC_INTENT_QUERIES:
        return _GENERIC_INTENT_QUERIES[intent]
    purpose = str(req.get("purpose") or "").lower()
    sections = {str(s).upper() for s in req.get("section_preferences", []) or []}
    if "risk" in purpose or "ITEM_1A" in sections:
        return "risk factors"
    if "management" in purpose or "md&a" in purpose or "ITEM_7" in sections:
        return "management discussion operating results"
    if "business" in purpose or "ITEM_1" in sections:
        return "business overview"
    return "risk factors management discussion business overview operating results"


def _text_payload(
    req: Mapping[str, Any],
    context: Mapping[str, Any],
    ticker: str,
    *,
    query: str,
    section_allowlist: list[str] | None,
    strict_sections: bool,
) -> dict[str, Any]:
    retrieval_policy = dict(context.get("retrieval_policy", {}) or {})
    profile = str(req.get("retrieval_profile") or retrieval_policy.get("retrieval_profile") or "default")
    top_k = max(
        int(req.get("min_results", 1) or 1),
        int(retrieval_policy.get("text_top_k", settings.retrieval_top_k) or settings.retrieval_top_k),
    )
    payload: dict[str, Any] = {
        "ticker": ticker,
        "query": query,
        "top_k": max(1, top_k),
        "date_start": context.get("date_start"),
        "date_end": context.get("date_end"),
        "retrieval_profile": profile,
        "max_per_filing": retrieval_policy.get("max_per_filing"),
        "max_per_section": retrieval_policy.get("max_per_section"),
        "strict_sections": bool(strict_sections),
    }
    if section_allowlist:
        payload["section_allowlist"] = list(section_allowlist)
    target_periods = list(context.get("target_periods", []) or [])
    if target_periods:
        payload["target_periods"] = target_periods
    return payload


def _search_filings_lexical_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]] | dict[str, Any]:
    return search_filings_lexical_fallback(
        ticker=str(payload.get("ticker", "")),
        query=str(payload.get("query", "")),
        top_k=int(payload.get("top_k", 1) or 1),
        form_type=payload.get("form_type"),
        date_start=payload.get("date_start"),
        date_end=payload.get("date_end"),
        section_allowlist=payload.get("section_allowlist"),
        strict_sections=bool(payload.get("strict_sections", False)),
        retrieval_profile=payload.get("retrieval_profile"),
        target_periods=payload.get("target_periods"),
        max_per_filing=payload.get("max_per_filing"),
        max_per_section=payload.get("max_per_section"),
        return_diagnostics=True,
    )


def _response_docs_and_diagnostics(raw_response: Any) -> tuple[list[Any], dict[str, Any]]:
    if isinstance(raw_response, Mapping):
        return list(raw_response.get("items", []) or []), dict(raw_response.get("diagnostics", {}) or {})
    return raw_response if isinstance(raw_response, list) else [], {}


def _should_try_lexical_first(req: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
    if str(req.get("requirement_type") or "") != "text":
        return False
    if str(payload.get("retrieval_profile") or "") != "risk_summary":
        return False
    if not bool(payload.get("strict_sections", False)):
        return False
    sections = {str(section).upper().strip() for section in payload.get("section_allowlist", []) or []}
    if "ITEM_1A" not in sections:
        return False
    intent = str(req.get("retrieval_intent") or "")
    risk_intents = {
        "single_company_risk_context",
        "single_company_competition_context",
        "risk_focused_risk_factors",
    }
    return intent in risk_intents


def _append_lexical_tool_trace(
    *,
    context: Mapping[str, Any],
    req: Mapping[str, Any],
    payload: Mapping[str, Any],
    raw_hit_count: int,
    returned_count: int,
) -> None:
    calls = context.setdefault("tool_call_results", []) if isinstance(context, dict) else None
    if not isinstance(calls, list):
        return
    input_summary = _tool_input_summary("search_filings", payload)
    input_summary["backend"] = "duckdb_lexical"
    input_summary["lexical_first"] = True
    calls.append(
        {
            "tool_call_id": f"tc_{len(calls) + 1:03d}",
            "tool_name": "search_filings",
            "tool_version": "duckdb_lexical",
            "requirement_id": str(req.get("requirement_id", "")),
            "input_summary": input_summary,
            "ok": True,
            "latency_ms": None,
            "raw_count": raw_hit_count,
            "returned_count": returned_count,
            "warnings": ["DuckDB lexical retrieval used before vector search."],
            "provenance": [],
            "error": None,
        }
    )


def _normalize_text_doc(doc: Mapping[str, Any], req: Mapping[str, Any], query: str, ticker: str) -> dict[str, Any]:
    item = dict(doc)
    text = str(item.get("supporting_snippet") or item.get("text_snippet") or item.get("text") or "")
    snippet = text[:500]
    supporting_terms = [str(x) for x in item.get("supporting_terms", []) or [] if str(x).strip()]
    if not supporting_terms:
        supporting_terms = sorted((_tokens(query) | {ticker.lower()}) & _tokens(snippet))[:12]
    if not supporting_terms:
        supporting_terms = _fallback_supporting_terms(snippet, str(item.get("section", "")), ticker)
    item["requirement_id"] = str(req.get("requirement_id", ""))
    item["dimension_id"] = str(req.get("dimension_id", ""))
    item["framework_id"] = str(req.get("framework_id", ""))
    item["retrieval_intent"] = str(req.get("retrieval_intent", ""))
    item["analysis_purpose"] = str(req.get("analysis_purpose", ""))
    item["ticker"] = str(item.get("ticker") or ticker).upper()
    item["snippet"] = snippet
    item["text_snippet"] = str(item.get("text_snippet") or snippet)
    item["supporting_snippet"] = str(item.get("supporting_snippet") or snippet)
    item["supporting_terms"] = supporting_terms
    item["score_breakdown"] = dict(item.get("score_breakdown", {}) or {})
    text_quality = validate_text_evidence_quality(item)
    item["text_quality"] = text_quality.model_dump(exclude_none=True)
    item["text_quality_reason"] = text_quality.reason
    item["specificity_score"] = text_quality.specificity_score
    return item


def _text_doc_rejection_reason(
    doc: Mapping[str, Any],
    req: Mapping[str, Any],
    query: str,
    ticker: str,
    *,
    section_allowlist: list[str] | None = None,
    strict_sections: bool = False,
) -> str | None:
    section = str(doc.get("section") or "").upper()
    quality = str(doc.get("quality") or "").lower()
    allow = {str(x or "").upper().strip() for x in section_allowlist or [] if str(x or "").strip()}
    if strict_sections and allow and section not in allow:
        return "section_filter_dropped"
    if section in {"MIXED", "<MIXED>"} or "mixed" in section.lower() or quality in LOW_QUALITY:
        return "quality_filter_dropped"
    item = _normalize_text_doc(doc, req, query, ticker)
    if not str(item.get("supporting_snippet") or "").strip():
        return "snippet_support_failed"
    if not item.get("supporting_terms"):
        return "snippet_support_failed"
    text_quality = validate_text_evidence_quality(item)
    if not text_quality.is_valid:
        return text_quality.reason or "low_information_text_evidence"
    return None


def _usable_text_doc(doc: Mapping[str, Any], req: Mapping[str, Any], query: str, ticker: str) -> dict[str, Any] | None:
    if _text_doc_rejection_reason(doc, req, query, ticker) is not None:
        return None
    return _normalize_text_doc(doc, req, query, ticker)


def _text_rejected_snippet(doc: Mapping[str, Any], reason: str, attempt: Mapping[str, Any]) -> dict[str, Any]:
    text = str(doc.get("supporting_snippet") or doc.get("text_snippet") or doc.get("text") or "")
    snippet = re.sub(r"\s+", " ", text).strip()[:360]
    return {
        "reason": reason,
        "attempt": str(attempt.get("strategy", "")),
        "query": str(attempt.get("query", "")),
        "ticker": str(doc.get("ticker", "")),
        "filing_id": str(doc.get("filing_id", "")),
        "form_type": str(doc.get("form_type", "")),
        "fiscal_period": str(doc.get("fiscal_period", "")),
        "section": str(doc.get("section", "")),
        "quality": str(doc.get("quality", "")),
        "chunk_order": int(doc.get("chunk_order", 0) or 0),
        "score": doc.get("score"),
        "final_score": doc.get("final_score", doc.get("score")),
        "text_snippet": snippet,
        "supporting_snippet": snippet,
        "supporting_terms": list(doc.get("supporting_terms", []) or []),
    }


def _bump_reason(counts: dict[str, int], reason: str | None) -> None:
    if not reason:
        return
    counts[reason] = counts.get(reason, 0) + 1


def _unique_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, tuple[str, ...] | None, bool]] = set()
    for attempt in attempts:
        allow = attempt.get("section_allowlist")
        key = (
            str(attempt.get("query", "")),
            tuple(str(x) for x in allow) if isinstance(allow, list) else None,
            bool(attempt.get("strict_sections", False)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(attempt)
    return out


def _text_drop_stage(
    *,
    raw_hit_count: int,
    section_filtered_hit_count: int,
    usable_hit_count: int,
    status: str,
    rejection_reasons: Mapping[str, Any] | None = None,
) -> str:
    if status == "satisfied":
        return "satisfied"
    if raw_hit_count <= 0:
        return "no_raw_hits"
    if section_filtered_hit_count <= 0:
        return "section_filter_dropped"
    if usable_hit_count <= 0:
        reasons = {str(k): int(v or 0) for k, v in dict(rejection_reasons or {}).items()}
        if reasons.get("quality_filter_dropped", 0) > 0:
            return "quality_filter_dropped"
        if reasons.get("low_information_text_evidence", 0) > 0:
            return "low_information_text_evidence"
        if reasons.get("snippet_support_failed", 0) > 0:
            return "snippet_support_failed"
        if reasons.get("section_filter_dropped", 0) > 0:
            return "section_filter_dropped"
        return "snippet_support_failed"
    return "final_bundle_dropped"


def _primary_text_rejection_reason(rejection_reasons: Mapping[str, Any] | None) -> str:
    reasons = {str(k): int(v or 0) for k, v in dict(rejection_reasons or {}).items()}
    for reason in (
        "low_information_text_evidence",
        "quality_filter_dropped",
        "snippet_support_failed",
        "section_filter_dropped",
    ):
        if reasons.get(reason, 0) > 0:
            return reason
    return ""


def _execute_text_requirement(req: dict[str, Any], context: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rid = str(req.get("requirement_id", ""))
    ticker = str(req.get("company") or "").upper()
    if not ticker:
        ticker = str(next(iter(context.get("tickers", []) or []), "")).upper()
    primary_sections = [
        s
        for s in (req.get("primary_sections") or req.get("section_preferences") or [])
        if isinstance(s, str) and s in KNOWN_SEC_SECTIONS
    ]
    fallback_sections = [
        s
        for s in req.get("fallback_sections", []) or []
        if isinstance(s, str) and s in KNOWN_SEC_SECTIONS and s not in primary_sections
    ]
    relaxed_sections = _ordered_unique(primary_sections + fallback_sections)
    query = str(req.get("retrieval_query") or context.get("user_query") or "")
    broadened_queries = [
        str(x).strip()
        for x in req.get("broadened_queries", []) or []
        if str(x).strip()
    ]
    min_results = int(req.get("min_results", 1) or 1)
    broadened_query = broadened_queries[0] if broadened_queries else query
    relaxed_query = broadened_queries[1] if len(broadened_queries) > 1 else broadened_query
    attempts = _unique_attempts(
        [
            {
                "strategy": "strict_intent_query",
                "query": query,
                "section_allowlist": primary_sections,
                "strict_sections": True,
            },
            {
                "strategy": "strict_broadened_query",
                "query": broadened_query,
                "section_allowlist": primary_sections,
                "strict_sections": True,
            },
            {
                "strategy": "relaxed_sections_intent_query",
                "query": relaxed_query,
                "section_allowlist": relaxed_sections or None,
                "strict_sections": False,
            },
            {
                "strategy": "generic_query",
                "query": _generic_query(req),
                "section_allowlist": None,
                "strict_sections": False,
            },
        ]
    )
    usable: list[dict[str, Any]] = []
    retry_history: list[dict[str, Any]] = []
    all_docs: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    failure_reason = "no_matching_evidence"
    total_raw_hit_count = 0
    total_section_filtered_hit_count = 0
    snippet_support_passed_count = 0
    top_raw_snippets: list[dict[str, Any]] = []
    top_rejected_snippets: list[dict[str, Any]] = []
    rejection_reasons: dict[str, int] = {}

    for attempt_no, attempt in enumerate(attempts):
        stop_after_attempt = False
        payload = _text_payload(
            req,
            context,
            ticker,
            query=str(attempt["query"]),
            section_allowlist=attempt["section_allowlist"],
            strict_sections=bool(attempt["strict_sections"]),
        )
        payload["return_diagnostics"] = True
        try:
            fallback_after_timeout = False
            fallback_after_error = False
            fallback_error_code = ""
            fallback_error_message = ""
            lexical_first = False
            timeout_message = ""
            raw_response: Any | None = None
            if _should_try_lexical_first(req, payload):
                try:
                    lexical_response = _search_filings_lexical_payload(payload)
                    lexical_docs, _lexical_diag = _response_docs_and_diagnostics(lexical_response)
                    lexical_normalized = [
                        item
                        for doc in lexical_docs
                        if isinstance(doc, Mapping)
                        for item in [_usable_text_doc(doc, req, str(attempt["query"]), ticker)]
                        if item is not None
                    ]
                    unique_lexical_normalized: list[dict[str, Any]] = []
                    for item in lexical_normalized:
                        if item not in unique_lexical_normalized:
                            unique_lexical_normalized.append(item)
                    if len(unique_lexical_normalized) >= min_results:
                        raw_response = lexical_response
                        lexical_first = True
                except Exception as exc:
                    logger.debug("DuckDB lexical-first filing retrieval failed for %s: %s", rid, exc)

            if raw_response is None:
                protocol_result = _run_protocol_tool("search_filings", payload, context, req)
                fallback_reason = _search_fallback_reason(protocol_result)
                if fallback_reason:
                    fallback_kind, fallback_error_code, fallback_error_message = fallback_reason
                    fallback_after_timeout = fallback_kind == "timeout"
                    fallback_after_error = fallback_kind != "timeout"
                    timeout_message = fallback_error_message if fallback_after_timeout else ""
                    stop_after_attempt = True
                    raw_response = _search_filings_lexical_payload(payload)
                else:
                    _raise_tool_error(protocol_result)
                    raw_response = protocol_result.data
            docs, search_diag = _response_docs_and_diagnostics(raw_response)
            if fallback_after_timeout:
                search_diag["fallback_after_timeout"] = True
                search_diag["timeout_error"] = timeout_message
            if fallback_after_error:
                search_diag["fallback_after_error"] = True
                search_diag["fallback_error_code"] = fallback_error_code
                search_diag["fallback_error"] = fallback_error_message
            if lexical_first:
                search_diag["lexical_first"] = True
            raw_hit_count = int(search_diag.get("raw_hit_count", 0) or _doc_stat(docs, "retrieval_raw_hit_count", len(docs)))
            section_filtered_hit_count = int(search_diag.get("section_filtered_hit_count", 0) or _doc_stat(docs, "section_filtered_hit_count", len(docs)))
            raw_candidates = [
                item
                for item in list(search_diag.get("raw_candidates", []) or [])
                if isinstance(item, Mapping)
            ]
            if len(top_raw_snippets) < 10:
                top_raw_snippets.extend([dict(item) for item in raw_candidates[: 10 - len(top_raw_snippets)]])
            candidate_docs = raw_candidates or [doc for doc in docs if isinstance(doc, Mapping)]
            for candidate in candidate_docs:
                reason = _text_doc_rejection_reason(
                    candidate,
                    req,
                    str(attempt["query"]),
                    ticker,
                    section_allowlist=attempt["section_allowlist"],
                    strict_sections=bool(attempt["strict_sections"]),
                )
                _bump_reason(rejection_reasons, reason)
                if reason and len(top_rejected_snippets) < 10:
                    top_rejected_snippets.append(_text_rejected_snippet(candidate, reason, attempt))
            normalized = [
                item
                for doc in docs
                if isinstance(doc, Mapping)
                for item in [_usable_text_doc(doc, req, str(attempt["query"]), ticker)]
                if item is not None
            ]
            for item in normalized:
                if item not in usable:
                    usable.append(item)
            all_docs.extend(normalized)
            total_raw_hit_count += raw_hit_count
            total_section_filtered_hit_count += section_filtered_hit_count
            snippet_support_passed_count += len(normalized)
            failure_reason = "below_min_results" if usable else "no_matching_evidence"
            if fallback_after_timeout and len(usable) < min_results:
                failure_reason = (
                    "search_filings_timeout_fallback_no_usable_evidence"
                    if raw_hit_count
                    else "search_filings_timeout_fallback_empty"
                )
            if fallback_after_error and len(usable) < min_results:
                failure_reason = (
                    "search_filings_resource_fallback_no_usable_evidence"
                    if raw_hit_count
                    else "search_filings_resource_fallback_empty"
                )
            call = {
                "requirement_id": rid,
                "ticker": ticker,
                "attempt_no": attempt_no,
                "strategy": attempt["strategy"],
                "query": payload["query"],
                "section_allowlist": payload.get("section_allowlist"),
                "strict_sections": payload.get("strict_sections"),
                "top_k": payload["top_k"],
                "returned": len(docs),
                "raw_hit_count": raw_hit_count,
                "section_filtered_hit_count": section_filtered_hit_count,
                "snippet_support_passed_count": len(normalized),
                "usable_hit_count": len(normalized),
                "usable_count": len(normalized),
                "rejection_reasons": dict(rejection_reasons),
                "failure_reason": None if len(usable) >= min_results else failure_reason,
            }
            if fallback_after_timeout:
                call["fallback_after_timeout"] = True
                call["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
                call["timeout_error"] = timeout_message
            if fallback_after_error:
                call["fallback_after_error"] = True
                call["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
                call["fallback_error_code"] = fallback_error_code
                call["fallback_error"] = fallback_error_message
            if lexical_first:
                call["lexical_first"] = True
                call["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
            retry_history.append(call)
            calls.append(call)
            tool_item = {"tool": "search_filings", "ticker": ticker, "requirement_id": rid, "count": len(normalized), "attempt": attempt["strategy"]}
            if fallback_after_timeout:
                tool_item["fallback_after_timeout"] = True
                tool_item["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
                if timeout_message:
                    tool_item["timeout_error"] = timeout_message
                if len(usable) < min_results:
                    tool_item["failure_reason"] = failure_reason
            if fallback_after_error:
                tool_item["fallback_after_error"] = True
                tool_item["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
                if fallback_error_code:
                    tool_item["fallback_error_code"] = fallback_error_code
                if fallback_error_message:
                    tool_item["fallback_error"] = fallback_error_message
                if len(usable) < min_results:
                    tool_item["failure_reason"] = failure_reason
            if lexical_first:
                tool_item["lexical_first"] = True
                tool_item["backend"] = str(search_diag.get("backend") or "duckdb_lexical")
                _append_lexical_tool_trace(
                    context=context,
                    req=req,
                    payload=payload,
                    raw_hit_count=raw_hit_count,
                    returned_count=len(normalized),
                )
            tool_results.append(tool_item)
        except Exception as exc:  # pragma: no cover - explicitly tested with fakes
            failure_reason = f"search_filings_error:{exc}"
            call = {
                "requirement_id": rid,
                "ticker": ticker,
                "attempt_no": attempt_no,
                "strategy": attempt["strategy"],
                "query": payload["query"],
                "section_allowlist": payload.get("section_allowlist"),
                "strict_sections": payload.get("strict_sections"),
                "top_k": payload["top_k"],
                "returned": 0,
                "raw_hit_count": 0,
                "section_filtered_hit_count": 0,
                "snippet_support_passed_count": 0,
                "usable_hit_count": 0,
                "usable_count": 0,
                "failure_reason": failure_reason,
                "error": str(exc),
            }
            retry_history.append(call)
            calls.append(call)
            tool_results.append({"tool": "search_filings", "ticker": ticker, "requirement_id": rid, "error": str(exc), "attempt": attempt["strategy"]})
        if len(usable) >= min_results:
            break
        if stop_after_attempt:
            break

    retry_count = max(len(retry_history) - 1, 0)
    if len(usable) >= min_results:
        status = "satisfied"
        failure = None
    elif usable:
        status = "partial"
        failure = "below_min_results"
    else:
        status = "missing"
        failure = failure_reason
        primary_rejection_reason = _primary_text_rejection_reason(rejection_reasons)
        if primary_rejection_reason and failure in {"", "no_matching_evidence"}:
            failure = primary_rejection_reason
    result = collection_result(
        requirement_id=rid,
        status=status,
        evidence_type="text",
        items=usable[: max(min_results, len(usable))],
        failure_reason=failure,
        retry_count=retry_count,
        raw_hit_count=total_raw_hit_count,
        section_filtered_hit_count=total_section_filtered_hit_count,
        usable_hit_count=len(usable),
        snippet_support_passed_count=snippet_support_passed_count,
        text_claim_validated_count=0,
        company=ticker,
        retrieval_query=query,
        section_preferences=primary_sections,
        fallback_queries=broadened_queries,
        fallback_sections=fallback_sections,
        top_raw_snippets=top_raw_snippets[:10],
        top_rejected_snippets=top_rejected_snippets[:10],
        rejection_reasons=dict(rejection_reasons),
        drop_stage=_text_drop_stage(
            raw_hit_count=total_raw_hit_count,
            section_filtered_hit_count=total_section_filtered_hit_count,
            usable_hit_count=len(usable),
            status=status,
            rejection_reasons=rejection_reasons,
        ),
    )
    return all_docs, result, calls, retry_history, tool_results


def _calculation_result_value(metric: str, row: Mapping[str, Any]) -> float | None:
    for key in ("value", "margin", "ratio", "difference", "multiple"):
        value = _safe_float(row.get(key))
        if value is not None:
            return value
    if metric == "fcf_yield":
        value = _safe_float(row.get("ratio"))
        return value
    return None


def _calculation_item_rows(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    metric = str(item.get("metric") or "")
    rows: list[dict[str, Any]] = []
    for row in (dict(item.get("data", {}) or {}).get("results", []) or []):
        if not isinstance(row, Mapping):
            continue
        value = _calculation_result_value(metric, row)
        if value is None:
            continue
        period = str(row.get("period") or item.get("period") or item.get("period_end") or "")
        rows.append(
            {
                **dict(row),
                "requirement_id": str(item.get("requirement_id") or ""),
                "company": str(item.get("company") or item.get("ticker") or "").upper(),
                "ticker": str(item.get("ticker") or item.get("company") or "").upper(),
                "metric": metric,
                "period": period,
                "period_end": period,
                "period_type": str(row.get("period_type") or item.get("period_type") or ""),
                "value": value,
                "unit": str(
                    item.get("unit")
                    or (
                        "ratio"
                        if metric.endswith("_ratio")
                        or metric.endswith("_margin")
                        or metric in {"pe_ratio", "ps_ratio", "fcf_yield", "segment_profit_contribution"}
                        else ""
                    )
                ),
                "source_provider": str(item.get("source_provider") or "computed"),
                "confidence": str(item.get("confidence") or "high"),
                "extraction_method": str(item.get("extraction_method") or "programmatic_calculation"),
                "source_tag": str(item.get("source_tag") or row.get("source_tag") or ""),
            }
        )
    return rows


def _numeric_items_by_company(collection_results: list[dict[str, Any]], company: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for result in collection_results:
        if result.get("evidence_type") not in {"numeric", "calculation"} or result.get("status") != "satisfied":
            continue
        for item in result.get("items", []) or []:
            if str(item.get("company") or item.get("ticker") or "").upper() == company:
                if result.get("evidence_type") == "calculation":
                    items.extend(_calculation_item_rows(item))
                else:
                    items.append(item)
    return items


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _calculation_metrics(req: Mapping[str, Any]) -> set[str]:
    metrics: set[str] = set()
    for raw in list(req.get("metrics", []) or []) + [req.get("metric")]:
        metric = str(raw or "").strip()
        if metric:
            metrics.add(metric)
    return metrics


def _period_key(item: Mapping[str, Any]) -> tuple[str, str]:
    return (str(item.get("period_end") or item.get("period") or ""), str(item.get("period_type") or ""))


def _aligned_net_margin_points(numeric_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_metric: dict[str, dict[tuple[str, str], dict[str, Any]]] = {"revenue": {}, "net_income": {}}
    for item in numeric_items:
        metric = str(item.get("metric") or "")
        if metric not in by_metric:
            continue
        key = _period_key(item)
        if not key[0]:
            continue
        value = _safe_float(item.get("value"))
        if value is None:
            continue
        by_metric[metric][key] = item

    points: list[dict[str, Any]] = []
    for key in sorted(set(by_metric["revenue"]) & set(by_metric["net_income"])):
        revenue = _safe_float(by_metric["revenue"][key].get("value"))
        net_income = _safe_float(by_metric["net_income"][key].get("value"))
        if revenue is None or net_income is None or revenue == 0:
            continue
        points.append(
            {
                "period": key[0],
                "period_type": key[1],
                "numerator": net_income,
                "denominator": revenue,
                "numerator_requirement_id": str(by_metric["net_income"][key].get("requirement_id", "")),
                "denominator_requirement_id": str(by_metric["revenue"][key].get("requirement_id", "")),
            }
        )
    return points


def _aligned_ratio_points(
    numeric_items: list[dict[str, Any]],
    numerator_metric: str,
    denominator_metric: str,
) -> list[dict[str, Any]]:
    by_metric: dict[str, dict[tuple[str, str], dict[str, Any]]] = {numerator_metric: {}, denominator_metric: {}}
    for item in numeric_items:
        metric = str(item.get("metric") or "")
        if metric not in by_metric:
            continue
        key = _period_key(item)
        if not key[0] or _safe_float(item.get("value")) is None:
            continue
        by_metric[metric][key] = item
    points: list[dict[str, Any]] = []
    for key in sorted(set(by_metric[numerator_metric]) & set(by_metric[denominator_metric])):
        numerator = _safe_float(by_metric[numerator_metric][key].get("value"))
        denominator = _safe_float(by_metric[denominator_metric][key].get("value"))
        if numerator is None or denominator is None or denominator == 0:
            continue
        points.append(
            {
                "period": key[0],
                "period_type": key[1],
                "numerator": numerator,
                "denominator": denominator,
                "numerator_requirement_id": str(by_metric[numerator_metric][key].get("requirement_id", "")),
                "denominator_requirement_id": str(by_metric[denominator_metric][key].get("requirement_id", "")),
            }
        )
    return points


def _aligned_difference_points(
    numeric_items: list[dict[str, Any]],
    left_metric: str,
    right_metric: str,
) -> list[dict[str, Any]]:
    by_metric: dict[str, dict[tuple[str, str], dict[str, Any]]] = {left_metric: {}, right_metric: {}}
    for item in numeric_items:
        metric = str(item.get("metric") or "")
        if metric not in by_metric:
            continue
        key = _period_key(item)
        if not key[0] or _safe_float(item.get("value")) is None:
            continue
        by_metric[metric][key] = item
    points: list[dict[str, Any]] = []
    for key in sorted(set(by_metric[left_metric]) & set(by_metric[right_metric])):
        left = _safe_float(by_metric[left_metric][key].get("value"))
        right = _safe_float(by_metric[right_metric][key].get("value"))
        if left is None or right is None:
            continue
        if right_metric == "capital_expenditure":
            right = abs(right)
        points.append(
            {
                "period": key[0],
                "period_type": key[1],
                "value": left - right,
                "left_requirement_id": str(by_metric[left_metric][key].get("requirement_id", "")),
                "right_requirement_id": str(by_metric[right_metric][key].get("requirement_id", "")),
            }
        )
    return points


def _is_comparison_margin_requirement(req: Mapping[str, Any], task_type: str) -> bool:
    metrics = _calculation_metrics(req)
    rid = str(req.get("requirement_id") or "").upper()
    if "net_margin" in metrics:
        return True
    if task_type != "company_comparison":
        return False
    return bool(metrics & {"operating_margin", "gross_margin"}) or "MARGIN" in rid


def _execute_net_margin_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    aligned = _aligned_net_margin_points(numeric_items)
    if not aligned:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    data = [{"period": item["period"], "value": item["numerator"]} for item in aligned]
    denominator_data = [{"period": item["period"], "value": item["denominator"]} for item in aligned]
    payload = {"data": data, "denominator_data": denominator_data, "computation": "margin"}
    try:
        protocol_result = _run_protocol_tool("compute_metrics", payload, context, req)
        _raise_tool_error(protocol_result)
        result = protocol_result.data
    except Exception as exc:  # pragma: no cover - explicitly tested with fakes
        logger.warning("compute_metrics failed for requirement %s: %s", rid, exc)
        return {"tool": "compute_metrics", "ticker": company, "requirement_id": rid, "error": str(exc)}, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason=f"compute_metrics_error:{exc}",
        )

    enriched = dict(result or {}) if isinstance(result, Mapping) else {}
    results: list[dict[str, Any]] = []
    for i, row in enumerate(enriched.get("results", []) or []):
        item = dict(row) if isinstance(row, Mapping) else {}
        aligned_item = aligned[i] if i < len(aligned) else {}
        item.setdefault("period", str(aligned_item.get("period", "")))
        item["period_type"] = str(aligned_item.get("period_type", ""))
        item["numerator_metric"] = "net_income"
        item["denominator_metric"] = "revenue"
        item["source_tag"] = "net_income_over_revenue"
        item["numerator_requirement_id"] = str(aligned_item.get("numerator_requirement_id", ""))
        item["denominator_requirement_id"] = str(aligned_item.get("denominator_requirement_id", ""))
        results.append(item)
    enriched["results"] = results
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": "net_margin",
        "computation": "margin",
        "source_tag": "net_income_over_revenue",
        "data": enriched,
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _execute_ratio_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
    *,
    metric: str,
    numerator_metric: str,
    denominator_metric: str,
    source_tag: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    aligned = _aligned_ratio_points(numeric_items, numerator_metric, denominator_metric)
    if not aligned:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    results: list[dict[str, Any]] = []
    for item in aligned:
        ratio = item["numerator"] / item["denominator"]
        results.append(
            {
                "period": item["period"],
                "period_type": item["period_type"],
                "margin": round(ratio, 6),
                "margin_pct": f"{ratio * 100:.2f}%",
                "source_tag": source_tag,
                "numerator_metric": numerator_metric,
                "denominator_metric": denominator_metric,
                "numerator_requirement_id": item["numerator_requirement_id"],
                "denominator_requirement_id": item["denominator_requirement_id"],
            }
        )
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": metric,
        "computation": "margin",
        "source_tag": source_tag,
        "data": {"computation": "margin", "input_count": len(aligned), "results": results},
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _execute_net_debt_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    aligned = _aligned_difference_points(numeric_items, "total_debt", "cash_and_equivalents")
    if not aligned:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    results = [
        {
            "period": item["period"],
            "period_type": item["period_type"],
            "value": item["value"],
            "source_tag": "total_debt_minus_cash",
            "left_metric": "total_debt",
            "right_metric": "cash_and_equivalents",
            "left_requirement_id": item["left_requirement_id"],
            "right_requirement_id": item["right_requirement_id"],
        }
        for item in aligned
    ]
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": "net_debt",
        "computation": "difference",
        "source_tag": "total_debt_minus_cash",
        "data": {"computation": "difference", "input_count": len(results), "results": results},
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _execute_difference_metric_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
    *,
    metric: str,
    left_metric: str,
    right_metric: str,
    source_tag: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    aligned = _aligned_difference_points(numeric_items, left_metric, right_metric)
    if not aligned:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    results = [
        {
            "period": item["period"],
            "period_type": item["period_type"],
            "value": item["value"],
            "difference": item["value"],
            "source_tag": source_tag,
            "left_metric": left_metric,
            "right_metric": right_metric,
            "left_requirement_id": item["left_requirement_id"],
            "right_requirement_id": item["right_requirement_id"],
        }
        for item in aligned
    ]
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": metric,
        "computation": "difference",
        "source_tag": source_tag,
        "data": {"computation": "difference", "input_count": len(results), "results": results},
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _latest_metric_item(numeric_items: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    canonical_metric = normalize_metric_name(metric)
    rows = [
        row
        for row in numeric_items
        if normalize_metric_name(str(row.get("metric") or "")) == canonical_metric and _safe_float(row.get("value")) is not None
    ]
    if not rows:
        return None
    return sorted(rows, key=lambda row: str(row.get("period_end") or row.get("period") or ""), reverse=True)[0]


def _dependency_record(row: Mapping[str, Any] | None, *, role: str) -> dict[str, Any]:
    item = dict(row or {})
    return {
        "role": role,
        "metric": str(item.get("metric") or ""),
        "requirement_id": str(item.get("requirement_id") or ""),
        "period_end": str(item.get("period_end") or item.get("period") or ""),
        "period_type": str(item.get("period_type") or ""),
        "value": item.get("value"),
        "unit": str(item.get("unit") or ""),
        "source_provider": str(item.get("source_provider") or ""),
        "confidence": str(item.get("confidence") or ""),
        "reconciliation_warning": str(item.get("reconciliation_warning") or ""),
    }


def _dependency_confidence(rows: list[Mapping[str, Any] | None]) -> str:
    confidences = {str(dict(row or {}).get("confidence") or "").lower() for row in rows}
    providers = {str(dict(row or {}).get("source_provider") or "").lower() for row in rows}
    if "low" in confidences:
        return "low"
    if "medium" in confidences or "yfinance" in providers:
        return "medium"
    return "high"


def _dependency_warning(rows: list[Mapping[str, Any] | None]) -> str:
    warnings = [
        str(dict(row or {}).get("reconciliation_warning") or "").strip()
        for row in rows
        if str(dict(row or {}).get("reconciliation_warning") or "").strip()
    ]
    return ";".join(dict.fromkeys(warnings))


def _row_period(row: Mapping[str, Any] | None) -> str:
    return str(dict(row or {}).get("period_end") or dict(row or {}).get("period") or "")


def _row_period_scope(row: Mapping[str, Any] | None) -> str:
    return str(dict(row or {}).get("period_scope") or dict(row or {}).get("period_type") or "unknown")


def _revenue_growth_invalid_result(rid: str, reason: str, *, current: Mapping[str, Any] | None = None, comparator: Mapping[str, Any] | None = None) -> tuple[None, dict[str, Any]]:
    details = {
        "quality_status": reason,
        "current_revenue": _dependency_record(current, role="current_revenue") if current else {},
        "comparator_revenue": _dependency_record(comparator, role="comparator_revenue") if comparator else {},
    }
    return None, collection_result(
        requirement_id=rid,
        status="missing",
        evidence_type="calculation",
        items=[],
        failure_reason=reason,
        **details,
    )


def _same_period(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> bool:
    return bool(_row_period(a) and _row_period(a) == _row_period(b))


def _comparable_period_scope(current: Mapping[str, Any], comparator: Mapping[str, Any]) -> bool:
    current_scope = _row_period_scope(current)
    comparator_scope = _row_period_scope(comparator)
    if not current_scope or current_scope == "unknown" or not comparator_scope or comparator_scope == "unknown":
        return True
    aliases = {
        "fy": "annual",
        "year": "annual",
        "yearly": "annual",
        "quarter": "quarterly",
        "q": "quarterly",
        "ttm": "trailing",
        "trailing_twelve_months": "trailing",
    }
    return aliases.get(current_scope, current_scope) == aliases.get(comparator_scope, comparator_scope)


def _select_revenue_growth_pair(rows: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    if len(rows) < 2:
        return None, None, "dependency_numeric_requirement_missing"
    sorted_rows = sorted(rows, key=lambda item: str(item.get("period_end") or item.get("period") or ""))
    current_candidates = [row for row in sorted_rows if str(row.get("evidence_role") or row.get("role") or "") == "current_revenue"]
    current = (current_candidates[-1] if current_candidates else sorted_rows[-1]) if sorted_rows else None
    if not current:
        return None, None, "dependency_numeric_requirement_missing"
    current_period = _row_period(current)
    comparator_role_rows = [row for row in sorted_rows if str(row.get("evidence_role") or row.get("role") or "") == "comparator_revenue"]
    comparator_candidates = [
        row
        for row in (comparator_role_rows or sorted_rows)
        if _row_period(row) and _row_period(row) != current_period
    ]
    earlier = [row for row in comparator_candidates if _row_period(row) < current_period]
    comparator = (earlier[-1] if earlier else (comparator_candidates[-1] if comparator_candidates else None))
    if comparator is None:
        return current, (comparator_role_rows[-1] if comparator_role_rows else None), "same_period_comparator"
    comparator_value = _safe_float(comparator.get("value"))
    if comparator_value is None:
        return current, comparator, "dependency_numeric_requirement_missing"
    if comparator_value == 0:
        return current, comparator, "zero_comparator"
    if _same_period(current, comparator):
        return current, comparator, "same_period_comparator"
    if not _comparable_period_scope(current, comparator):
        return current, comparator, "incomparable_period_scope"
    return current, comparator, ""


def _execute_market_cap_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    price = _latest_metric_item(numeric_items, "price")
    shares = _latest_metric_item(numeric_items, "shares_outstanding")
    price_value = _safe_float((price or {}).get("value"))
    shares_value = _safe_float((shares or {}).get("value"))
    if price_value is None or shares_value is None:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    market_cap = price_value * shares_value
    dependencies = [
        _dependency_record(price, role="share_price"),
        _dependency_record(shares, role="shares_outstanding"),
    ]
    confidence = _dependency_confidence([price, shares])
    warning = _dependency_warning([price, shares])
    result = {
        "period": str((price or {}).get("period_end") or (price or {}).get("period") or ""),
        "period_type": "latest_price",
        "value": market_cap,
        "source_tag": "latest_price_times_shares_outstanding",
        "left_metric": str((price or {}).get("metric") or "adjusted_close"),
        "right_metric": "shares_outstanding",
        "left_requirement_id": str((price or {}).get("requirement_id") or ""),
        "right_requirement_id": str((shares or {}).get("requirement_id") or ""),
        "period_basis": "latest price x latest shares outstanding",
        "share_price": price_value,
        "price_date": str((price or {}).get("period_end") or (price or {}).get("period") or ""),
        "shares_outstanding": shares_value,
        "shares_period": str((shares or {}).get("period_end") or (shares or {}).get("period") or ""),
        "dependencies": dependencies,
        "source_provider": "computed",
        "confidence": confidence,
        "reconciliation_warning": warning,
    }
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": "market_cap",
        "unit": "USD",
        "computation": "valuation_multiple",
        "source_tag": "latest_price_times_shares_outstanding",
        "source_provider": "computed",
        "confidence": confidence,
        "reconciliation_warning": warning,
        "dependencies": dependencies,
        "data": {"computation": "valuation_multiple", "input_count": 2, "results": [result]},
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _execute_valuation_ratio_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
    *,
    metric: str,
    denominator_metric: str,
    source_tag: str,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    market_cap = _latest_metric_item(numeric_items, "market_cap")
    denominator = _latest_metric_item(numeric_items, denominator_metric)
    market_cap_value = _safe_float((market_cap or {}).get("value"))
    denominator_value = _safe_float((denominator or {}).get("value"))
    if market_cap_value in (None, 0) or denominator_value is None or (metric != "fcf_yield" and denominator_value == 0):
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    if metric == "fcf_yield":
        value = denominator_value / market_cap_value
        result_key = "ratio"
    else:
        value = market_cap_value / denominator_value
        result_key = "multiple"
    market_cap_dependencies = list((market_cap or {}).get("dependencies", []) or [])
    dependencies = [
        *[dict(item) for item in market_cap_dependencies if isinstance(item, Mapping)],
        _dependency_record(denominator, role=denominator_metric),
    ]
    confidence = _dependency_confidence([market_cap, denominator])
    warning = _dependency_warning([market_cap, denominator])
    statement_period = str((denominator or {}).get("period_end") or (denominator or {}).get("period") or "")
    result = {
        "period": str((market_cap or {}).get("period_end") or (market_cap or {}).get("period") or ""),
        "period_type": str((denominator or {}).get("period_type") or "latest"),
        result_key: value,
        "ratio": value,
        "value": value,
        "ratio_pct": f"{value * 100:.2f}%" if metric == "fcf_yield" else "",
        "multiple_label": f"{value:.2f}x" if metric != "fcf_yield" else "",
        "source_tag": source_tag,
        "numerator_metric": "free_cash_flow" if metric == "fcf_yield" else "market_cap",
        "denominator_metric": "market_cap" if metric == "fcf_yield" else denominator_metric,
        "numerator_requirement_id": str((denominator if metric == "fcf_yield" else market_cap or {}).get("requirement_id") or ""),
        "denominator_requirement_id": str((market_cap if metric == "fcf_yield" else denominator or {}).get("requirement_id") or ""),
        "period_basis": str((denominator or {}).get("period_type") or "latest"),
        "share_price": (market_cap or {}).get("share_price"),
        "price_date": (market_cap or {}).get("price_date"),
        "shares_outstanding": (market_cap or {}).get("shares_outstanding"),
        "shares_period": (market_cap or {}).get("shares_period"),
        "market_cap": market_cap_value,
        "market_cap_period": str((market_cap or {}).get("period_end") or (market_cap or {}).get("period") or ""),
        "statement_period": statement_period,
        f"{denominator_metric}_period": statement_period,
        "dependencies": dependencies,
        "source_provider": "computed",
        "confidence": confidence,
        "reconciliation_warning": warning,
    }
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": metric,
        "unit": "ratio",
        "computation": "valuation_multiple",
        "source_tag": source_tag,
        "source_provider": "computed",
        "confidence": confidence,
        "reconciliation_warning": warning,
        "dependencies": dependencies,
        "data": {"computation": "valuation_multiple", "input_count": 2, "results": [result]},
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
    )


def _execute_revenue_growth_requirement(
    req: dict[str, Any],
    numeric_items: list[dict[str, Any]],
    company: str,
    context: Mapping[str, Any],
    *,
    dependency_failure_reason: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    if dependency_failure_reason:
        return _revenue_growth_invalid_result(rid, dependency_failure_reason)
    rows = [
        row
        for row in sorted(numeric_items, key=lambda item: str(item.get("period_end") or item.get("period") or ""))
        if str(row.get("metric") or "") == "revenue" and _safe_float(row.get("value")) is not None
    ]
    current, comparator, invalid_reason = _select_revenue_growth_pair(rows)
    if invalid_reason:
        return _revenue_growth_invalid_result(rid, invalid_reason, current=current, comparator=comparator)
    if not current or not comparator:
        return _revenue_growth_invalid_result(rid, "dependency_numeric_requirement_missing", current=current, comparator=comparator)
    points = [
        {"period": str(comparator.get("period_end") or comparator.get("period") or ""), "value": comparator.get("value")},
        {"period": str(current.get("period_end") or current.get("period") or ""), "value": current.get("value")},
    ]
    payload = {"data": points, "computation": "growth"}
    try:
        protocol_result = _run_protocol_tool("compute_metrics", payload, context, req)
        _raise_tool_error(protocol_result)
        result = protocol_result.data
    except Exception as exc:  # pragma: no cover - explicitly tested with fakes
        logger.warning("compute_metrics failed for requirement %s: %s", rid, exc)
        return {"tool": "compute_metrics", "ticker": company, "requirement_id": rid, "error": str(exc)}, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason=f"compute_metrics_error:{exc}",
        )
    enriched = dict(result or {}) if isinstance(result, Mapping) else {}
    dependencies = [
        _dependency_record(current, role="current_revenue"),
        _dependency_record(comparator, role="comparator_revenue"),
    ]
    for row in enriched.get("results", []) or []:
        if isinstance(row, dict):
            if row.get("error"):
                continue
            row["source_tag"] = "revenue_period_growth"
            row["period_type"] = _row_period_scope(current)
            row["compare_period"] = _row_period(comparator)
            row["compare_value"] = comparator.get("value")
            row["current_requirement_id"] = str(current.get("requirement_id") or "")
            row["comparator_requirement_id"] = str(comparator.get("requirement_id") or "")
            row["dependencies"] = dependencies
    valid_results = [row for row in enriched.get("results", []) or [] if isinstance(row, Mapping) and row.get("growth") is not None]
    if not valid_results:
        return _revenue_growth_invalid_result(rid, "invalid_growth_dependencies", current=current, comparator=comparator)
    enriched["results"] = valid_results
    output_item = {
        "requirement_id": rid,
        "company": company,
        "ticker": company,
        "metric": "revenue_growth",
        "unit": "ratio",
        "computation": "growth",
        "source_tag": "revenue_period_growth",
        "source_provider": "computed",
        "confidence": _dependency_confidence([current, comparator]),
        "reconciliation_warning": _dependency_warning([current, comparator]),
        "dependencies": dependencies,
        "data": enriched,
    }
    return {"tool": "compute_metrics", **output_item}, collection_result(
        requirement_id=rid,
        status="satisfied",
        evidence_type="calculation",
        items=[output_item],
        quality_status="valid",
    )


def _execute_calculation_requirement(
    req: dict[str, Any],
    collection_results: list[dict[str, Any]],
    task_type: str,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    if "valuation_evidence_missing" in {str(item) for item in req.get("fallback_strategy", []) or []}:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="valuation_evidence_missing",
        )
    company = str(req.get("company") or "").upper()
    numeric_items = _numeric_items_by_company(collection_results, company)
    metrics = _calculation_metrics(req)
    if _is_comparison_margin_requirement(req, task_type):
        return _execute_net_margin_requirement(req, numeric_items, company, context)
    if "free_cash_flow" in metrics:
        return _execute_difference_metric_requirement(
            req,
            numeric_items,
            company,
            metric="free_cash_flow",
            left_metric="operating_cash_flow",
            right_metric="capital_expenditure",
            source_tag="operating_cash_flow_minus_abs_capex",
        )
    if metrics & {"cfo_to_net_income", "cash_conversion"}:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="cash_conversion" if "cash_conversion" in metrics else "cfo_to_net_income",
            numerator_metric="operating_cash_flow",
            denominator_metric="net_income",
            source_tag="operating_cash_flow_over_net_income",
        )
    if "fcf_margin" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="fcf_margin",
            numerator_metric="free_cash_flow",
            denominator_metric="revenue",
            source_tag="free_cash_flow_over_revenue",
        )
    if "segment_profit_contribution" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="segment_profit_contribution",
            numerator_metric="aws_operating_income",
            denominator_metric="consolidated_operating_income",
            source_tag="aws_operating_income_over_consolidated_operating_income",
        )
    if "net_debt" in metrics:
        return _execute_net_debt_requirement(req, numeric_items, company)
    if "debt_to_equity" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="debt_to_equity",
            numerator_metric="total_debt",
            denominator_metric="shareholders_equity",
            source_tag="total_debt_over_shareholders_equity",
        )
    if "capex_to_revenue" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="capex_to_revenue",
            numerator_metric="capital_expenditure",
            denominator_metric="revenue",
            source_tag="capital_expenditure_over_revenue",
        )
    if "receivables_to_revenue" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="receivables_to_revenue",
            numerator_metric="receivables",
            denominator_metric="revenue",
            source_tag="receivables_over_revenue",
        )
    if "inventory_to_revenue" in metrics:
        return _execute_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="inventory_to_revenue",
            numerator_metric="inventory",
            denominator_metric="revenue",
            source_tag="inventory_over_revenue",
        )
    if "market_cap" in metrics:
        return _execute_market_cap_requirement(req, numeric_items, company)
    if "pe_ratio" in metrics:
        return _execute_valuation_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="pe_ratio",
            denominator_metric="net_income",
            source_tag="market_cap_over_net_income",
        )
    if "ps_ratio" in metrics:
        return _execute_valuation_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="ps_ratio",
            denominator_metric="revenue",
            source_tag="market_cap_over_revenue",
        )
    if "fcf_yield" in metrics:
        return _execute_valuation_ratio_requirement(
            req,
            numeric_items,
            company,
            metric="fcf_yield",
            denominator_metric="free_cash_flow",
            source_tag="free_cash_flow_over_market_cap",
        )
    if "revenue_growth" in metrics:
        dependency_failure_reason = next(
            (
                str(result.get("failure_reason") or "")
                for result in collection_results
                if str(result.get("company") or "").upper() == company
                and str(result.get("evidence_role") or "") == "comparator_revenue"
                and str(result.get("failure_reason") or "") in {"same_period_comparator", "zero_comparator", "incomparable_period_scope"}
            ),
            "",
        )
        return _execute_revenue_growth_requirement(
            req,
            numeric_items,
            company,
            context,
            dependency_failure_reason=dependency_failure_reason,
        )
    if len(numeric_items) < 2:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    computation = "qoq" if task_type == "trend_analysis" else "growth"
    outputs: list[dict[str, Any]] = []
    tool_result: dict[str, Any] | None = None
    by_metric: dict[str, list[dict[str, Any]]] = {}
    for item in numeric_items:
        metric = str(item.get("metric") or "")
        if metric:
            by_metric.setdefault(metric, []).append(item)
    for metric, rows in by_metric.items():
        points = [
            {"period": str(row.get("period_end") or row.get("period") or ""), "value": row.get("value")}
            for row in sorted(rows, key=lambda r: str(r.get("period_end") or r.get("period") or ""))
            if row.get("value") is not None
        ]
        if len(points) < 2:
            continue
        payload = {"data": points, "computation": computation}
        try:
            protocol_result = _run_protocol_tool("compute_metrics", payload, context, req)
            _raise_tool_error(protocol_result)
            result = protocol_result.data
            item = {"requirement_id": rid, "company": company, "ticker": company, "metric": metric, "computation": computation, "data": result}
            outputs.append(item)
            tool_result = {"tool": "compute_metrics", **item}
        except Exception as exc:  # pragma: no cover - explicitly tested with fakes
            logger.warning("compute_metrics failed for requirement %s: %s", rid, exc)
            return {"tool": "compute_metrics", "ticker": company, "requirement_id": rid, "error": str(exc)}, collection_result(
                requirement_id=rid,
                status="missing",
                evidence_type="calculation",
                items=[],
                failure_reason=f"compute_metrics_error:{exc}",
            )
    if not outputs:
        return None, collection_result(
            requirement_id=rid,
            status="missing",
            evidence_type="calculation",
            items=[],
            failure_reason="dependency_numeric_requirement_missing",
        )
    return tool_result, collection_result(requirement_id=rid, status="satisfied", evidence_type="calculation", items=outputs)


def _execute_event_requirement(req: dict[str, Any], context: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rid = str(req.get("requirement_id", ""))
    company = str(req.get("company") or "").upper()
    event_query = dict(context.get("event_query", {}) or {})
    payload = {
        "ticker": company,
        "event_type": event_query.get("event_type", "any"),
        "fiscal_period": event_query.get("fiscal_period"),
        "event_date": event_query.get("event_date"),
        "latest_n": event_query.get("latest_n", 4),
        "window_days": event_query.get("window_days", [1, 5, 10]),
        "sort_by": event_query.get("sort_by", "event_date"),
        "sort_order": event_query.get("sort_order", "desc"),
    }
    try:
        protocol_result = _run_protocol_tool("query_event_price_window", payload, context, req)
        _raise_tool_error(protocol_result)
        result = protocol_result.data
        data = dict(result or {}) if isinstance(result, Mapping) else {}
        events: list[dict[str, Any]] = []
        for event in data.get("events", []) or []:
            if isinstance(event, Mapping):
                item = dict(event)
                item["requirement_id"] = rid
                events.append(item)
        data["events"] = events
        data["requirement_id"] = rid
        status = "satisfied" if len(events) >= int(req.get("min_results", 1) or 1) else ("partial" if events else "missing")
        failure = None if status == "satisfied" else ("below_min_results" if events else "no_matching_evidence")
        tool_result = {"tool": "query_event_price_window", "ticker": company, "requirement_id": rid, "count": len(events), "data": data}
        return tool_result, collection_result(requirement_id=rid, status=status, evidence_type="event", items=events, failure_reason=failure), {"ticker": company, "data": data}
    except Exception as exc:  # pragma: no cover - explicitly tested with fakes
        logger.warning("query_event_price_window failed for requirement %s: %s", rid, exc)
        return (
            {"tool": "query_event_price_window", "ticker": company, "requirement_id": rid, "error": str(exc)},
            collection_result(requirement_id=rid, status="missing", evidence_type="event", items=[], failure_reason=f"query_event_price_window_error:{exc}"),
            {"ticker": company, "data": {"events": [], "requirement_id": rid, "error": str(exc)}},
        )


def _evidence_validation_records(collection_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    tool_by_type = {
        "numeric": "query_financial_data",
        "calculation": "compute_metrics",
        "text": "search_filings",
        "event": "query_event_price_window",
    }
    for result in collection_results:
        evidence_type = str(result.get("evidence_type") or "")
        returned = int(
            result.get("tool_returned_count")
            or result.get("raw_hit_count")
            or result.get("section_filtered_hit_count")
            or len(result.get("items", []) or [])
            or 0
        )
        validated = int(result.get("validated_evidence_count") or result.get("usable_hit_count") or len(result.get("items", []) or []) or 0)
        rejected = str(result.get("rejected_evidence_reason") or result.get("failure_reason") or "")
        if not rejected and returned > validated:
            rejected = "evidence_filter_mismatch"
        records.append(
            {
                "requirement_id": str(result.get("requirement_id") or ""),
                "evidence_type": evidence_type,
                "tool": tool_by_type.get(evidence_type, ""),
                "tool_returned_count": returned,
                "validated_evidence_count": validated,
                "rejected_evidence_reason": rejected,
                "status": str(result.get("status") or ""),
            }
        )
    return records


def execute_evidence_requirements(
    state: AgentState,
    evidence_plan: Mapping[str, Any],
    execution_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute tools requirement-by-requirement for analytical answer modes."""
    context = dict(execution_context or {})
    context.setdefault("tool_call_results", [])
    requirements = _requirements(evidence_plan)
    answer_mode = str(state.get("answer_mode", evidence_plan.get("answer_mode", "direct_fact")))
    if state.get("needs_tools") is False:
        return _empty_result(state, evidence_plan, "needs_tools_false")
    if answer_mode in NO_TOOL_MODES:
        return _empty_result(state, evidence_plan, f"answer_mode_{answer_mode}")
    if not requirements:
        return _empty_result(state, evidence_plan, "no_evidence_requirements")

    selected_tools = _tools_from_requirements(requirements, state)
    tool_results: list[dict[str, Any]] = []
    retrieved_docs: list[dict[str, Any]] = []
    event_results: list[dict[str, Any]] = list(state.get("event_results", []) or [])
    collection_results: list[dict[str, Any]] = []
    requirement_calls: list[dict[str, Any]] = []
    retry_history: list[dict[str, Any]] = []
    event_calls: list[dict[str, Any]] = []

    numeric_requirements = _requirements_of_type(requirements, "numeric")
    if numeric_requirements:
        _progress(
            state,
            "tool_started",
            "started",
            "正在读取结构化财务数据。",
            metadata={"tool": "query_financial_data", "requirement_count": len(numeric_requirements)},
        )
    numeric_returned = 0
    for req in _requirements_of_type(requirements, "numeric"):
        tool_result, result = _execute_numeric_requirement(req, context)
        tool_results.append(tool_result)
        collection_results.append(result)
        requirement_calls.append(
            {
                "requirement_id": req.get("requirement_id", ""),
                "tool": "query_financial_data",
                "ticker": req.get("company", ""),
                "returned": result.get("tool_returned_count", len(result.get("items", []) or [])),
                "validated_evidence_count": result.get("validated_evidence_count", len(result.get("items", []) or [])),
                "rejected_evidence_reason": result.get("rejected_evidence_reason", ""),
                "status": result.get("status", ""),
                "failure_reason": result.get("failure_reason"),
            }
        )
    _apply_causal_revenue_quality(
        requirements=requirements,
        collection_results=collection_results,
        tool_results=tool_results,
    )
    numeric_returned = sum(len(result.get("items", []) or []) for result in collection_results if result.get("evidence_type") == "numeric")
    for call in requirement_calls:
        rid = str(call.get("requirement_id") or "")
        result = next((item for item in collection_results if str(item.get("requirement_id") or "") == rid), None)
        if result:
            call["returned"] = len(result.get("items", []) or [])
            if result.get("tool_returned_count") is not None:
                call["returned"] = result.get("tool_returned_count")
            call["validated_evidence_count"] = result.get("validated_evidence_count", len(result.get("items", []) or []))
            call["rejected_evidence_reason"] = result.get("rejected_evidence_reason", "")
            call["status"] = result.get("status", "")
            call["failure_reason"] = result.get("failure_reason")
    if numeric_requirements:
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成结构化财务数据读取，返回 {numeric_returned} 条证据。",
            metadata={"tool": "query_financial_data", "status": "passed", "returned": numeric_returned},
        )

    calculation_requirements = _requirements_of_type(requirements, "calculation")
    if calculation_requirements:
        _progress(
            state,
            "tool_started",
            "started",
            "正在计算派生财务指标。",
            metadata={"tool": "compute_metrics", "requirement_count": len(calculation_requirements)},
        )
    calculation_returned = 0
    for req in calculation_requirements:
        tool_result, result = _execute_calculation_requirement(
            req,
            collection_results,
            str(evidence_plan.get("task_type", state.get("task_type", ""))),
            context,
        )
        if tool_result:
            tool_results.append(tool_result)
        collection_results.append(result)
        calculation_returned += len(result.get("items", []) or [])
        requirement_calls.append(
            {
                "requirement_id": req.get("requirement_id", ""),
                "tool": "compute_metrics",
                "ticker": req.get("company", ""),
                "returned": len(result.get("items", []) or []),
                "status": result.get("status", ""),
                "failure_reason": result.get("failure_reason"),
            }
        )
    if calculation_requirements:
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成派生指标计算，返回 {calculation_returned} 条计算证据。",
            metadata={"tool": "compute_metrics", "status": "passed", "returned": calculation_returned},
        )

    for req in _requirements_of_type(requirements, "text"):
        req_id = str(req.get("requirement_id") or "")
        company = str(req.get("company") or "")
        dimension = str(req.get("dimension_id") or "")
        _progress(
            state,
            "tool_started",
            "started",
            f"正在检索 {company or '目标公司'} 的 SEC filing 文本证据。",
            metadata={"tool": "search_filings", "requirement_id": req_id, "company": company, "dimension": dimension},
        )
        docs, result, calls, retries, text_tool_results = _execute_text_requirement(req, context)
        retrieved_docs.extend(docs)
        tool_results.extend(text_tool_results)
        collection_results.append(result)
        requirement_calls.extend(calls)
        retry_history.extend(retries)
        _progress(
            state,
            "tool_finished",
            "completed" if str(result.get("status") or "") != "missing" else "warning",
            f"已完成 SEC filing 检索，返回 {len(docs)} 条候选证据。",
            metadata={
                "tool": "search_filings",
                "requirement_id": req_id,
                "company": company,
                "status": str(result.get("status") or ""),
                "returned": len(docs),
            },
        )

    event_requirements = _requirements_of_type(requirements, "event")
    if event_requirements:
        _progress(
            state,
            "tool_started",
            "started",
            "正在读取事件窗口市场数据。",
            metadata={"tool": "query_event_price_window", "requirement_count": len(event_requirements)},
        )
    event_returned = 0
    for req in event_requirements:
        tool_result, result, event_result = _execute_event_requirement(req, context)
        tool_results.append(tool_result)
        collection_results.append(result)
        event_results.append(event_result)
        event_returned += len(result.get("items", []) or [])
        event_calls.append(
            {
                "requirement_id": req.get("requirement_id", ""),
                "ticker": req.get("company", ""),
                "returned": len(result.get("items", []) or []),
                "status": result.get("status", ""),
                "failure_reason": result.get("failure_reason"),
            }
        )
        requirement_calls.append({"tool": "query_event_price_window", **event_calls[-1]})
    if event_requirements:
        _progress(
            state,
            "tool_finished",
            "completed",
            f"已完成事件窗口市场数据读取，返回 {event_returned} 条证据。",
            metadata={"tool": "query_event_price_window", "status": "passed", "returned": event_returned},
        )

    sufficiency = evaluate_evidence_sufficiency(evidence_plan, collection_results).model_dump(exclude_none=True)
    limitations = list(sufficiency.get("requirement_limitations", []) or [])
    retrieval_debug = {
        "policy": dict(context.get("retrieval_policy", {}) or {}),
        "selected_tools": selected_tools,
        "validated_tools": selected_tools,
        "event_calls": event_calls,
        "search_calls": [c for c in requirement_calls if c.get("strategy")],
        "requirement_calls": requirement_calls,
        "requirement_retry_history": retry_history,
        "search_skipped": [],
        "tool_call_results": list(context.get("tool_call_results", []) or []),
    }
    evidence_validation_records = _evidence_validation_records(collection_results)
    return {
        "tool_results": tool_results,
        "tool_call_results": list(context.get("tool_call_results", []) or []),
        "retrieved_docs": retrieved_docs,
        "event_results": event_results,
        "evidence_collection_results": collection_results,
        "evidence_validation_records": evidence_validation_records,
        "evidence_sufficiency": sufficiency,
        "requirement_calls": requirement_calls,
        "retry_history": retry_history,
        "requirement_limitations": limitations,
        "selected_tools": selected_tools,
        "validated_tools": selected_tools,
        "market_reaction_limitations": list(state.get("market_reaction_limitations", []) or []),
        "why_tools_skipped": [],
        "retrieval_debug": retrieval_debug,
    }
