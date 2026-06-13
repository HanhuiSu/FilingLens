"""Sanitized trace view model for the browser audit console."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.driver_evidence import apply_scope_aware_summary


_CITATION_RE = re.compile(r"\[([NTCE]\d+)\]")
_STATUS_ORDER = {"missing": 0, "missing_and_unanswerable": 0, "missing_but_analyzable": 1, "partial": 2, "failed": 3, "blocked": 4, "satisfied": 5, "passed": 6}
_TEXT_LIMIT = 500
_SNIPPET_LIMIT = 240


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any, *, limit: int = _TEXT_LIMIT) -> str:
    raw = str(value or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: limit - 1].rstrip() + "..."


def _public_dict(item: Mapping[str, Any], keys: list[str], *, text_limit: int = _TEXT_LIMIT) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        value = item.get(key)
        if isinstance(value, str):
            out[key] = _text(value, limit=text_limit)
        elif isinstance(value, (int, float, bool)) or value is None:
            out[key] = value
        elif isinstance(value, list):
            out[key] = [_text(v, limit=120) if not isinstance(v, (dict, list)) else v for v in value[:20]]
        elif isinstance(value, Mapping):
            out[key] = {
                str(k): _text(v, limit=120) if not isinstance(v, (dict, list)) else v
                for k, v in list(value.items())[:20]
            }
    return out


def _requirement_status_map(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    status_map: dict[str, dict[str, Any]] = {}
    for key in ("final_requirement_status_map", "requirement_status_map"):
        for rid, item in _as_dict(trace.get(key)).items():
            if isinstance(item, Mapping):
                status_map[str(rid)] = dict(item)
    for item in _as_list(trace.get("evidence_collection_results")):
        if not isinstance(item, Mapping):
            continue
        rid = str(item.get("requirement_id") or "")
        if not rid:
            continue
        update = {
            "status": item.get("status", ""),
            "failure_reason": item.get("failure_reason"),
            "collected_count": len(_as_list(item.get("items"))),
        }
        for key in (
            "raw_hit_count",
            "usable_hit_count",
            "evidence_role",
            "quality_status",
            "tool_returned_count",
            "validated_evidence_count",
            "rejected_evidence_reason",
        ):
            if item.get(key) is not None:
                update[key] = item.get(key)
        status_map.setdefault(rid, {}).update(update)
    return status_map


def _tool_for_requirement(req: Mapping[str, Any]) -> str:
    req_type = str(req.get("requirement_type") or req.get("evidence_type") or "")
    if req_type == "numeric":
        return "query_financial_data"
    if req_type == "calculation":
        return "compute_metrics"
    if req_type == "text":
        return "search_filings"
    if req_type == "event":
        return "query_event_price_window"
    return str(req.get("tool") or req.get("tool_name") or "")


def _build_evidence_plan(trace: Mapping[str, Any]) -> dict[str, Any]:
    plan = _as_dict(trace.get("evidence_plan"))
    requirements = _as_list(trace.get("evidence_requirements")) or _as_list(plan.get("evidence_requirements"))
    status_map = _requirement_status_map(trace)
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(requirements):
        if not isinstance(raw, Mapping):
            continue
        rid = str(raw.get("requirement_id") or raw.get("id") or f"REQ-{idx + 1}")
        status = _as_dict(status_map.get(rid))
        required = bool(raw.get("required", True))
        scope = _text(raw.get("requirement_scope") or ("core" if required else "optional_context"), limit=80)
        raw_status = _text(status.get("status") or raw.get("status") or "planned", limit=80)
        display_status = "optional_missing" if raw_status == "missing" and not required else raw_status
        row = {
            "requirement_id": rid,
            "dimension": _text(raw.get("dimension_id") or raw.get("dimension") or raw.get("analysis_dimension"), limit=120),
            "evidence_type": _text(raw.get("requirement_type") or raw.get("evidence_type"), limit=80),
            "evidence_role": _text(raw.get("evidence_role") or status.get("evidence_role"), limit=120),
            "answer_part_ids": _as_list(raw.get("answer_part_ids") or status.get("answer_part_ids")),
            "merged_from": _as_list(raw.get("merged_from")),
            "company": _text(raw.get("company") or raw.get("ticker"), limit=40),
            "period": _text(raw.get("period") or raw.get("period_type") or raw.get("fiscal_period"), limit=80),
            "tool": _tool_for_requirement(raw),
            "status": display_status,
            "status_label": "optional missing" if display_status == "optional_missing" else display_status,
            "raw_status": raw_status,
            "missing_reason": _text(status.get("failure_reason") or raw.get("missing_reason"), limit=160),
            "returned": status.get("collected_count"),
            "tool_returned_count": status.get("tool_returned_count"),
            "validated_evidence_count": status.get("validated_evidence_count"),
            "rejected_evidence_reason": _text(status.get("rejected_evidence_reason"), limit=160),
            "collected_count": status.get("collected_count"),
            "raw_hit_count": status.get("raw_hit_count"),
            "usable_hit_count": status.get("usable_hit_count"),
            "quality_status": _text(status.get("quality_status") or raw.get("quality_status"), limit=120),
            "required": required,
            "scope": scope,
            "blocking": scope == "core" and required and raw_status in {"missing", "partial", "rejected"},
        }
        rows.append(row)
    missing = [r for r in rows if r["raw_status"] == "missing"]
    missing_required = [r for r in missing if r["blocking"]]
    missing_optional = [r for r in missing if not r["required"]]
    partial = [r for r in rows if r["raw_status"] == "partial" and r["blocking"]]
    total_partial = [r for r in rows if r["raw_status"] == "partial"]
    scope_counts = {
        "core": len([r for r in rows if r["scope"] == "core"]),
        "optional_context": len([r for r in rows if r["scope"] == "optional_context"]),
        "diagnostic": len([r for r in rows if r["scope"] == "diagnostic"]),
    }
    requirements_by_scope = {
        scope: [r for r in rows if r["scope"] == scope]
        for scope in ("core", "optional_context", "diagnostic")
    }
    return {
        "requirements": rows,
        "requirements_by_scope": requirements_by_scope,
        "summary": {
            "requirement_count": len(rows),
            "missing_count": len(missing_required),
            "total_missing_count": len(missing),
            "missing_required_count": len(missing_required),
            "missing_optional_count": len(missing_optional),
            "partial_count": len(partial),
            "total_partial_count": len(total_partial),
            "satisfied_count": len([r for r in rows if r["status"] == "satisfied"]),
            "scope_counts": scope_counts,
            "core_count": scope_counts["core"],
            "optional_context_count": scope_counts["optional_context"],
            "diagnostic_count": scope_counts["diagnostic"],
        },
    }


def _build_research_plan(trace: Mapping[str, Any]) -> dict[str, Any]:
    raw = _safe_nested_dict(trace.get("research_plan_raw"), limit=800)
    validated = _safe_nested_dict(trace.get("research_plan_validated"), limit=800)
    used = _safe_nested_dict(trace.get("research_plan_used"), limit=800)
    validation = _safe_nested_dict(trace.get("research_plan_validation"), limit=800)
    gaps = _safe_nested_dict(trace.get("evidence_gap_by_answer_part"), limit=500)
    legacy = _as_dict(trace.get("legacy_evidence_plan"))
    requirements = _as_list(legacy.get("evidence_requirements"))
    return {
        "raw": raw,
        "validated": validated,
        "used": used,
        "validation": validation,
        "source": _text(trace.get("research_plan_source") or validation.get("research_plan_source") or validation.get("planner_trace", {}).get("source"), limit=120),
        "fallback_reason": _text(trace.get("research_plan_fallback_reason") or validation.get("fallback_reason"), limit=200),
        "duration_ms": int(trace.get("research_plan_duration_ms") or _as_dict(validation.get("planner_trace")).get("duration_ms") or 0),
        "required_answer_parts": _as_list(trace.get("required_answer_parts")) or _as_list(used.get("required_answer_parts")),
        "evidence_gap_by_answer_part": gaps,
        "answer_part_status_by_id": _safe_nested_dict(trace.get("answer_part_status_by_id"), limit=500),
        "missing_required_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_required_answer_parts"))],
        "partial_required_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("partial_required_answer_parts"))],
        "missing_but_analyzable_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_but_analyzable_answer_parts"))],
        "missing_and_unanswerable_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_and_unanswerable_answer_parts"))],
        "relevance_decision": _safe_nested_dict(trace.get("relevance_decision"), limit=500),
        "relevance_status": _text(trace.get("relevance_status"), limit=80),
        "legacy_evidence_plan": {
            "requirement_count": len(requirements),
            "evidence_policy_id": _text(legacy.get("evidence_policy_id") or _as_dict(legacy.get("evidence_policy")).get("policy_id"), limit=120),
            "expected_synthesis_style": _text(legacy.get("expected_synthesis_style"), limit=120),
            "requirements": [
                _public_dict(
                    item,
                    ["requirement_id", "requirement_type", "evidence_role", "answer_part_ids", "company", "metric", "metrics", "requirement_scope", "purpose"],
                    text_limit=200,
                )
                for item in requirements[:80]
                if isinstance(item, Mapping)
            ],
        },
        "summary": {
            "question_type": _text(used.get("question_type") or validated.get("question_type"), limit=120),
            "user_goal": _text(used.get("user_goal") or validated.get("user_goal"), limit=240),
            "used": bool(used),
            "valid": bool(validation.get("valid")),
            "source": _text(trace.get("research_plan_source") or validation.get("planner_trace", {}).get("source"), limit=120),
            "fallback_reason": _text(trace.get("research_plan_fallback_reason") or validation.get("fallback_reason"), limit=200),
            "duration_ms": int(trace.get("research_plan_duration_ms") or _as_dict(validation.get("planner_trace")).get("duration_ms") or 0),
            "required_answer_part_count": len(_as_list(trace.get("required_answer_parts")) or _as_list(used.get("required_answer_parts"))),
        },
    }


def _build_analytical_reasoning(trace: Mapping[str, Any]) -> dict[str, Any]:
    synthesis = _as_dict(trace.get("synthesis"))
    claims = _as_list(trace.get("analytical_claims")) or _as_list(synthesis.get("analytical_claims"))
    claim_tiers = _as_dict(trace.get("claim_tiers")) or _as_dict(synthesis.get("claim_tiers"))
    if not claim_tiers:
        def tier_value(item: Mapping[str, Any]) -> str:
            raw = item.get("tier")
            return str(getattr(raw, "value", raw) or "")
        claim_tiers = {
            "evidence_backed": len([c for c in claims if isinstance(c, Mapping) and tier_value(c) == "evidence_backed"]),
            "evidence_inferred": len([c for c in claims if isinstance(c, Mapping) and tier_value(c) == "evidence_inferred"]),
            "hypothesis_to_verify": len([c for c in claims if isinstance(c, Mapping) and tier_value(c) == "hypothesis_to_verify"]),
        }
    driver_scope_counts = _as_dict(trace.get("driver_scope_counts"))
    if not driver_scope_counts:
        evidence_scope = _build_evidence_scope(trace)["evidence_scope_by_ref"]
        driver_scope_counts = {
            "company": len([item for item in evidence_scope.values() if item.get("claim_scope") == "company"]),
            "segment": len([item for item in evidence_scope.values() if item.get("claim_scope") == "segment"]),
            "product": len([item for item in evidence_scope.values() if item.get("claim_scope") == "product"]),
            "market_context": len([item for item in evidence_scope.values() if item.get("claim_scope") == "market_context"]),
            "unknown": len([item for item in evidence_scope.values() if item.get("claim_scope") == "unknown"]),
            "scope_bounded_inferences": len(
                [
                    item
                    for item in evidence_scope.values()
                    if item.get("allowed_claim_strength") in {"bounded_inference", "hypothesis_only"}
                ]
            ),
        }
    return {
        "analytical_claims": [
            _public_dict(
                item,
                ["id", "text", "tier", "citation_refs", "supporting_claim_ids", "confidence", "caveat"],
                text_limit=500,
            )
            for item in claims[:80]
            if isinstance(item, Mapping)
        ],
        "claim_tiers": claim_tiers,
        "analytical_reasoning_status": _text(trace.get("analytical_reasoning_status") or synthesis.get("analytical_reasoning_status"), limit=80),
        "evidence_health": _text(trace.get("evidence_health") or synthesis.get("evidence_health"), limit=80),
        "driver_scope_counts": driver_scope_counts,
        "tool_error_context": [
            _public_dict(item, ["requirement_id", "status", "evidence_type", "kind", "failure_reason"], text_limit=240)
            for item in (_as_list(trace.get("tool_error_context")) or _as_list(synthesis.get("tool_error_context")))[:80]
            if isinstance(item, Mapping)
        ],
        "relevance_status": _text(trace.get("relevance_status"), limit=80),
    }


def _build_evidence_scope(trace: Mapping[str, Any]) -> dict[str, Any]:
    packet = _as_dict(trace.get("evidence_packet"))
    contract = _as_dict(trace.get("contract_result")) or _as_dict(trace.get("contract_decision"))
    scope_by_ref = _as_dict(trace.get("evidence_scope_by_ref")) or _as_dict(contract.get("scope_overclaim_check")).get("evidence_scope_by_ref")
    scope_by_ref = _as_dict(scope_by_ref)
    if not scope_by_ref:
        rows = _as_list(packet.get("text_snippets")) or _as_list(trace.get("text_evidence")) or _as_list(_as_dict(trace.get("output")).get("text_evidence"))
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            row = apply_scope_aware_summary(row)
            ref = str(row.get("citation_ref") or row.get("evidence_id") or "").strip()
            if not ref:
                continue
            claim_scope = str(row.get("claim_scope") or "").strip()
            driver_level = str(row.get("driver_level") or "").strip()
            if not claim_scope:
                claim_scope = {
                    "company_level_driver": "company",
                    "segment_level_driver": "segment",
                    "product_level_driver": "product",
                    "market_context": "market_context",
                    "risk_context": "market_context",
                }.get(driver_level, "unknown")
            scope_by_ref[ref] = {
                "evidence_id": ref,
                "driver_level": driver_level or "unknown",
                "driver_levels": list(row.get("driver_levels", []) or []),
                "claim_scope": claim_scope,
                "allowed_claim_strength": str(row.get("allowed_claim_strength") or ("definitive" if claim_scope == "company" else "bounded_inference")),
                "scope_reason": _text(row.get("scope_reason"), limit=220),
                "summary_scope_warning": _text(row.get("summary_scope_warning"), limit=120),
                "evidence_summary_scope_overclaim": bool(row.get("evidence_summary_scope_overclaim", False)),
            }
    return {
        "evidence_scope_by_ref": scope_by_ref,
        "rows": list(scope_by_ref.values()),
        "summary": {
            "company_level_driver_claims": len([item for item in scope_by_ref.values() if item.get("claim_scope") == "company"]),
            "segment_level_driver_claims": len([item for item in scope_by_ref.values() if item.get("claim_scope") == "segment"]),
            "product_level_driver_claims": len([item for item in scope_by_ref.values() if item.get("claim_scope") == "product"]),
            "scope_bounded_inferences": len([item for item in scope_by_ref.values() if item.get("allowed_claim_strength") in {"bounded_inference", "hypothesis_only"}]),
        },
    }


def _build_evidence_validation(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = _as_list(trace.get("evidence_validation_records"))
    if not records:
        for result in _as_list(trace.get("evidence_collection_results")):
            if not isinstance(result, Mapping):
                continue
            evidence_type = str(result.get("evidence_type") or "")
            records.append(
                {
                    "requirement_id": str(result.get("requirement_id") or ""),
                    "evidence_type": evidence_type,
                    "tool": _tool_for_requirement({"requirement_type": evidence_type}),
                    "tool_returned_count": int(
                        result.get("tool_returned_count")
                        or result.get("raw_hit_count")
                        or result.get("section_filtered_hit_count")
                        or len(_as_list(result.get("items")))
                        or 0
                    ),
                    "validated_evidence_count": int(result.get("validated_evidence_count") or result.get("usable_hit_count") or len(_as_list(result.get("items"))) or 0),
                    "rejected_evidence_reason": str(result.get("rejected_evidence_reason") or result.get("failure_reason") or ""),
                    "status": str(result.get("status") or ""),
                }
            )
    return [
        _public_dict(
            item,
            [
                "requirement_id",
                "evidence_type",
                "tool",
                "tool_returned_count",
                "validated_evidence_count",
                "rejected_evidence_reason",
                "status",
            ],
            text_limit=240,
        )
        for item in records[:200]
        if isinstance(item, Mapping)
    ]


def _evidence_rows(trace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packet = _as_dict(trace.get("evidence_packet"))
    output = _as_dict(trace.get("output"))
    numeric = (
        _as_list(packet.get("numeric_table"))
        or _as_list(trace.get("numeric_evidence"))
        or _as_list(output.get("numeric_evidence"))
    )
    text_rows = (
        _as_list(packet.get("text_snippets"))
        or _as_list(trace.get("text_evidence"))
        or _as_list(output.get("text_evidence"))
    )
    return [dict(r) for r in numeric if isinstance(r, Mapping)], [apply_scope_aware_summary(r) for r in text_rows if isinstance(r, Mapping)]


def _row_requirement_ids(row: Mapping[str, Any]) -> list[str]:
    req_ids = [
        str(req_id).strip()
        for req_id in _as_list(row.get("requirement_ids"))
        if str(req_id).strip()
    ]
    for key in ("source_requirement_id", "requirement_id"):
        rid = str(row.get(key) or "").strip()
        if rid and rid not in req_ids:
            req_ids.append(rid)
    return req_ids


def _build_evidence_packet(trace: Mapping[str, Any]) -> dict[str, Any]:
    numeric, text_rows = _evidence_rows(trace)
    status_map = _requirement_status_map(trace)
    numeric_view: list[dict[str, Any]] = []
    computed_view: list[dict[str, Any]] = []
    event_view: list[dict[str, Any]] = []
    for row in numeric[:200]:
        req_ids = _row_requirement_ids(row)
        source_requirement_id = str(row.get("source_requirement_id") or (req_ids[0] if req_ids else "")).strip()
        req_status = _as_dict(status_map.get(source_requirement_id))
        view = _public_dict(
            row,
            [
                "evidence_id",
                "requirement_id",
                "source_requirement_id",
                "requirement_ids",
                "ticker",
                "metric",
                "role",
                "evidence_role",
                "quality_status",
                "value",
                "unit",
                "period",
                "period_type",
                "period_scope",
                "period_end",
                "source_provider",
                "confidence",
                "source_url",
                "source_filing_id",
                "formula",
                "input_evidence_ids",
                "reconciliation_warning",
            ],
        )
        if source_requirement_id and not view.get("source_requirement_id"):
            view["source_requirement_id"] = source_requirement_id
        if req_ids and not view.get("requirement_ids"):
            view["requirement_ids"] = req_ids
        role = str(row.get("role") or row.get("evidence_role") or req_status.get("evidence_role") or "").strip()
        if role:
            view["role"] = view.get("role") or role
            view["evidence_role"] = view.get("evidence_role") or role
        quality_status = str(
            row.get("quality_status")
            or req_status.get("quality_status")
            or ("valid" if str(req_status.get("status") or "") == "satisfied" else req_status.get("failure_reason") or "")
            or ""
        ).strip()
        if quality_status:
            view["quality_status"] = view.get("quality_status") or quality_status
        provider = str(row.get("source_provider") or "")
        source_tool = str(row.get("source_tool") or "")
        if provider == "computed" or row.get("formula") or row.get("input_evidence_ids"):
            computed_view.append(view)
        elif source_tool == "query_event_price_window" or provider == "event_price_window":
            event_view.append(view)
        else:
            numeric_view.append(view)
    text_view = [
        _public_dict(
            row,
            [
                "evidence_id",
                "requirement_id",
                "ticker",
                "filing_id",
                "form_type",
                "fiscal_period",
                "section",
                "dimension_id",
                "citation_ref",
                "claim",
                "original_claim",
                "theme_name",
                "driver_level",
                "driver_levels",
                "claim_scope",
                "allowed_claim_strength",
                "scope_reason",
                "evidence_summary_scope_overclaim",
                "summary_scope_warning",
                "text_snippet",
                "supporting_snippet",
                "score",
                "quality",
                "retrieval_backend",
                "backend",
            ],
            text_limit=_SNIPPET_LIMIT,
        )
        for row in text_rows[:200]
    ]
    limitations = []
    for source in (
        _as_list(trace.get("requirement_limitations")),
        _as_list(_as_dict(trace.get("evidence_packet")).get("limitations")),
        _as_list(_as_dict(trace.get("output")).get("limitations")),
    ):
        for item in source:
            if isinstance(item, Mapping):
                message = item.get("message") or item.get("text") or item.get("code")
            else:
                message = item
            if message:
                limitations.append(_text(message, limit=200))
    return {
        "numeric_evidence": numeric_view,
        "text_evidence": text_view,
        "computed_metrics": computed_view,
        "event_evidence": event_view,
        "limitations": list(dict.fromkeys(limitations)),
        "summary": {
            "numeric_count": len(numeric_view),
            "text_count": len(text_view),
            "computed_count": len(computed_view),
            "event_count": len(event_view),
            "limitations_count": len(limitations),
        },
    }


def _build_dimensions(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    packet = _as_dict(trace.get("evidence_packet"))
    status_map = (
        _as_dict(trace.get("dimension_status_by_id"))
        or _as_dict(trace.get("dimension_status_map"))
        or _as_dict(packet.get("dimension_status_map"))
    )
    active = [str(x) for x in _as_list(trace.get("active_dimensions")) or _as_list(packet.get("active_dimensions"))]
    numeric, text_rows = _evidence_rows(trace)
    rows: list[dict[str, Any]] = []
    for dimension_id, raw in status_map.items():
        item = _as_dict(raw)
        support_ids = [str(x) for x in _as_list(item.get("supporting_evidence_ids")) if str(x)]
        evidence_count = len(support_ids)
        if not evidence_count:
            evidence_count = len(
                [
                    row
                    for row in numeric + text_rows
                    if str(row.get("dimension_id") or row.get("analysis_dimension") or "") == str(dimension_id)
                ]
            )
        rows.append(
            {
                "dimension_id": str(dimension_id),
                "status": _text(item.get("status") or "unknown", limit=40),
                "evidence_count": evidence_count,
                "supporting_evidence_ids": support_ids,
                "required_missing": [_text(x, limit=120) for x in _as_list(item.get("required_missing"))],
                "enhanced_missing": [_text(x, limit=120) for x in _as_list(item.get("enhanced_missing"))],
                "limitations": [_text(x, limit=200) for x in _as_list(item.get("limitations"))],
                "active": str(dimension_id) in active,
            }
        )
    for dimension_id in active:
        if not any(row["dimension_id"] == dimension_id for row in rows):
            rows.append(
                {
                    "dimension_id": dimension_id,
                    "status": "unknown",
                    "evidence_count": 0,
                    "supporting_evidence_ids": [],
                    "required_missing": [],
                    "enhanced_missing": [],
                    "limitations": [],
                    "active": True,
                }
            )
    return sorted(rows, key=lambda r: (_STATUS_ORDER.get(str(r.get("status")), 9), str(r.get("dimension_id"))))


def _evidence_map(trace: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    numeric, text_rows = _evidence_rows(trace)
    out: dict[str, dict[str, Any]] = {}
    for row in numeric:
        eid = str(row.get("evidence_id") or "")
        if eid:
            out[eid] = {"evidence_type": "numeric", **_public_dict(row, ["ticker", "metric", "period", "period_end", "source_provider"])}
    for row in text_rows:
        eid = str(row.get("evidence_id") or row.get("citation_ref") or "")
        if eid:
            out[eid] = {
                "evidence_type": "text",
                **_public_dict(
                    row,
                    [
                        "ticker",
                        "section",
                        "filing_id",
                        "fiscal_period",
                        "driver_level",
                        "driver_levels",
                        "claim_scope",
                        "allowed_claim_strength",
                        "scope_reason",
                    ],
                ),
            }
    return out


def _extract_citation_ids(trace: Mapping[str, Any]) -> list[str]:
    report = _as_dict(trace.get("report")) or _as_dict(_as_dict(trace.get("output")).get("report"))
    text_sources = [
        str(trace.get("final_answer") or ""),
        str(_as_dict(trace.get("output")).get("summary") or ""),
        str(report.get("markdown") or ""),
    ]
    ids: list[str] = []
    for source in text_sources:
        for match in _CITATION_RE.findall(source):
            if match not in ids:
                ids.append(match)
    return ids


def _build_citations(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    evidence = _evidence_map(trace)
    referenced = _extract_citation_ids(trace)
    rows: list[dict[str, Any]] = []
    for cid in referenced:
        source = evidence.get(cid, {})
        rows.append(
            {
                "citation_id": cid,
                "evidence_id": cid,
                "evidence_type": source.get("evidence_type", "unknown"),
                "company": source.get("ticker", ""),
                "metric": source.get("metric", ""),
                "section": source.get("section", ""),
                "period": source.get("period") or source.get("period_end") or source.get("fiscal_period", ""),
                "source_title": source.get("source_provider") or source.get("filing_id") or "",
                "driver_level": source.get("driver_level", ""),
                "driver_levels": source.get("driver_levels", []),
                "claim_scope": source.get("claim_scope", ""),
                "allowed_claim_strength": source.get("allowed_claim_strength", ""),
                "scope_reason": source.get("scope_reason", ""),
                "snippet": "",
                "used_in_answer": True,
                "valid": cid in evidence,
            }
        )
    for eid, source in evidence.items():
        if eid in referenced:
            continue
        rows.append(
            {
                "citation_id": eid,
                "evidence_id": eid,
                "evidence_type": source.get("evidence_type", "unknown"),
                "company": source.get("ticker", ""),
                "metric": source.get("metric", ""),
                "section": source.get("section", ""),
                "period": source.get("period") or source.get("period_end") or source.get("fiscal_period", ""),
                "source_title": source.get("source_provider") or source.get("filing_id") or "",
                "driver_level": source.get("driver_level", ""),
                "driver_levels": source.get("driver_levels", []),
                "claim_scope": source.get("claim_scope", ""),
                "allowed_claim_strength": source.get("allowed_claim_strength", ""),
                "scope_reason": source.get("scope_reason", ""),
                "snippet": "",
                "used_in_answer": False,
                "valid": True,
            }
        )
    return rows


def _contract_status(trace: Mapping[str, Any]) -> str:
    return str(trace.get("final_contract_status") or trace.get("contract_status") or "not_run")


def _node_status(trace: Mapping[str, Any], node_id: str) -> str:
    contract_status = _contract_status(trace)
    if node_id == "classify":
        return "passed" if trace.get("task_type") or trace.get("query_understanding_summary") else "idle"
    if node_id == "research_plan":
        validation = _as_dict(trace.get("research_plan_validation"))
        if validation.get("valid"):
            return "passed"
        if trace.get("research_plan_raw") or validation:
            return "warning"
        return "idle"
    if node_id == "execute_tools":
        if trace.get("needs_tools") is False:
            return "skipped"
        if _as_list(trace.get("tool_results")) or _as_list(trace.get("evidence_collection_results")):
            return "passed"
        return "idle"
    if node_id == "evaluate":
        return "passed" if trace.get("evidence_sufficiency") or trace.get("dimension_status_by_id") else "idle"
    if node_id == "generate":
        return "passed" if trace.get("final_answer") else "idle"
    if node_id == "contract_check":
        if contract_status in {"passed", "passed_with_warnings", "repaired"}:
            return contract_status
        if contract_status == "blocked":
            return "blocked"
        if contract_status in {"failed", "hard_fail"}:
            return "failed"
        return "idle"
    if node_id == "relevance_check":
        status = str(trace.get("relevance_status") or "")
        if status in {"passed", "passed_with_warnings", "released_with_warnings"}:
            return "passed" if status == "passed" else "warning"
        if status == "failed":
            return "failed"
        return "idle"
    if node_id == "relevance_repair":
        return "passed" if int(trace.get("relevance_repair_attempts") or trace.get("relevance_attempts") or 0) > 0 else "skipped"
    if node_id == "repair_generate":
        return "passed" if _as_list(trace.get("repair_actions")) else "skipped"
    if node_id == "prepare_contract_evidence_retry":
        return "retried" if int(trace.get("contract_evidence_retry_count") or 0) > 0 else "skipped"
    if node_id == "safe_blocked_answer":
        return "blocked" if contract_status == "blocked" else "skipped"
    if node_id == "finalize":
        if contract_status == "blocked":
            return "blocked"
        return "passed" if trace.get("final_answer") else "idle"
    return "idle"


def _build_workflow(trace: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = {
        "classify": "Classify",
        "research_plan": "Research Plan",
        "execute_tools": "Execute Tools",
        "evaluate": "Evaluate Evidence",
        "generate": "Generate Draft",
        "contract_check": "Contract Check",
        "relevance_check": "Relevance Check",
        "relevance_repair": "Relevance Repair",
        "repair_generate": "Repair",
        "prepare_contract_evidence_retry": "Evidence Retry",
        "safe_blocked_answer": "Safe Blocked Answer",
        "finalize": "Finalize",
    }
    nodes = [
        {
            "id": node_id,
            "label": label,
            "status": _node_status(trace, node_id),
            "retry_count": int(trace.get("contract_attempts") or 0) if node_id == "repair_generate" else 0,
            "summary": "",
        }
        for node_id, label in labels.items()
    ]
    repaired = bool(_as_list(trace.get("repair_actions")))
    relevance_repaired = int(trace.get("relevance_repair_attempts") or trace.get("relevance_attempts") or 0) > 0
    evidence_retry = int(trace.get("contract_evidence_retry_count") or 0) > 0
    blocked = _contract_status(trace) == "blocked"
    edges = [
        {"id": "classify-research_plan", "source": "classify", "target": "research_plan", "taken": True},
        {"id": "research_plan-execute_tools", "source": "research_plan", "target": "execute_tools", "taken": True},
        {"id": "execute_tools-evaluate", "source": "execute_tools", "target": "evaluate", "taken": True},
        {"id": "evaluate-generate", "source": "evaluate", "target": "generate", "taken": True},
        {"id": "generate-contract_check", "source": "generate", "target": "contract_check", "taken": True},
        {"id": "contract_check-relevance_check", "source": "contract_check", "target": "relevance_check", "label": "pass", "taken": not blocked},
        {"id": "relevance_check-finalize", "source": "relevance_check", "target": "finalize", "label": "pass", "taken": not blocked and not relevance_repaired},
        {
            "id": "relevance_check-relevance_repair",
            "source": "relevance_check",
            "target": "relevance_repair",
            "label": "repair_answer",
            "taken": relevance_repaired,
        },
        {
            "id": "relevance_repair-contract_check",
            "source": "relevance_repair",
            "target": "contract_check",
            "taken": relevance_repaired,
        },
        {
            "id": "contract_check-repair_generate",
            "source": "contract_check",
            "target": "repair_generate",
            "label": "repair_answer",
            "taken": repaired,
        },
        {
            "id": "repair_generate-contract_check",
            "source": "repair_generate",
            "target": "contract_check",
            "taken": repaired,
        },
        {
            "id": "contract_check-prepare_contract_evidence_retry",
            "source": "contract_check",
            "target": "prepare_contract_evidence_retry",
            "label": "need_more_evidence",
            "taken": evidence_retry,
        },
        {
            "id": "prepare_contract_evidence_retry-execute_tools",
            "source": "prepare_contract_evidence_retry",
            "target": "execute_tools",
            "taken": evidence_retry,
        },
        {
            "id": "contract_check-safe_blocked_answer",
            "source": "contract_check",
            "target": "safe_blocked_answer",
            "label": "blocked",
            "taken": blocked,
        },
        {
            "id": "safe_blocked_answer-finalize",
            "source": "safe_blocked_answer",
            "target": "finalize",
            "taken": blocked,
        },
    ]
    return nodes, edges


def _fallback_success_by_requirement(trace: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    out: dict[str, Mapping[str, Any]] = {}
    for raw in _as_list(_as_dict(trace.get("retrieval_debug")).get("requirement_calls")):
        if not isinstance(raw, Mapping):
            continue
        rid = str(raw.get("requirement_id") or "").strip()
        backend = str(raw.get("backend") or "").strip()
        usable_count = int(raw.get("usable_hit_count") or raw.get("usable_count") or raw.get("returned") or 0)
        if (
            rid
            and (bool(raw.get("fallback_after_timeout")) or bool(raw.get("fallback_after_error")))
            and backend == "duckdb_lexical"
            and usable_count > 0
            and not raw.get("failure_reason")
            and not raw.get("error")
        ):
            out[rid] = raw
    return out


def _is_fallback_tool_failure(raw: Mapping[str, Any]) -> bool:
    if bool(raw.get("ok", True)):
        return False
    error = raw.get("error")
    if isinstance(error, Mapping):
        return str(error.get("code") or "") in {"timeout", "execution_error", "resource_exhausted"}
    return "timed out" in str(error or "").lower()


def _fallback_warning(raw: Mapping[str, Any]) -> str:
    if bool(raw.get("fallback_after_timeout")):
        return "Vector search timed out; DuckDB lexical fallback succeeded."
    return "Vector search hit a resource error; DuckDB lexical fallback succeeded."


def _build_tool_calls(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    calls = _as_list(trace.get("tool_call_results"))
    if not calls:
        calls = _as_list(_as_dict(trace.get("retrieval_debug")).get("tool_call_results"))
    fallback_success = _fallback_success_by_requirement(trace)
    out: list[dict[str, Any]] = []
    for idx, raw in enumerate(calls):
        if not isinstance(raw, Mapping):
            continue
        item = _public_dict(
            raw,
            [
                "tool_call_id",
                "requirement_id",
                "tool_name",
                "tool_version",
                "input_summary",
                "ok",
                "latency_ms",
                "raw_count",
                "returned_count",
                "warnings",
                "provenance",
                "error",
            ],
        )
        rid = str(item.get("requirement_id") or "")
        fallback = fallback_success.get(rid)
        if fallback and _is_fallback_tool_failure(raw):
            warnings = list(item.get("warnings") or [])
            warnings.append(_fallback_warning(fallback))
            input_summary = dict(item.get("input_summary") or {})
            input_summary["backend"] = "duckdb_lexical"
            if bool(fallback.get("fallback_after_timeout")):
                input_summary["fallback_after_timeout"] = True
            if bool(fallback.get("fallback_after_error")):
                input_summary["fallback_after_error"] = True
                input_summary["fallback_error_code"] = fallback.get("fallback_error_code")
            input_summary["strategy"] = fallback.get("strategy")
            item.update(
                {
                    "ok": True,
                    "raw_count": fallback.get("raw_hit_count"),
                    "returned_count": fallback.get("usable_hit_count") or fallback.get("returned"),
                    "warnings": list(dict.fromkeys(warnings)),
                    "error": None,
                    "input_summary": input_summary,
                    "fallback_after_timeout": True if bool(fallback.get("fallback_after_timeout")) else False,
                    "fallback_after_error": True if bool(fallback.get("fallback_after_error")) else False,
                    "fallback_error_code": fallback.get("fallback_error_code"),
                    "backend": "duckdb_lexical",
                }
            )
        out.append(item)
    if out:
        return out
    for idx, raw in enumerate(_as_list(_as_dict(trace.get("retrieval_debug")).get("requirement_calls"))):
        if not isinstance(raw, Mapping):
            continue
        fallback_ok = (
            (bool(raw.get("fallback_after_timeout")) or bool(raw.get("fallback_after_error")))
            and str(raw.get("backend") or "") == "duckdb_lexical"
            and int(raw.get("usable_hit_count") or raw.get("usable_count") or raw.get("returned") or 0) > 0
            and not raw.get("failure_reason")
            and not raw.get("error")
        )
        out.append(
            {
                "tool_call_id": f"legacy_{idx + 1}",
                "requirement_id": _text(raw.get("requirement_id"), limit=120),
                "tool_name": _text(raw.get("tool") or raw.get("tool_name") or "search_filings", limit=80),
                "tool_version": "",
                "input_summary": {
                    "ticker": raw.get("ticker"),
                    "strategy": raw.get("strategy"),
                    "backend": raw.get("backend") if fallback_ok else None,
                    "fallback_after_timeout": True if fallback_ok and bool(raw.get("fallback_after_timeout")) else None,
                    "fallback_after_error": True if fallback_ok and bool(raw.get("fallback_after_error")) else None,
                    "fallback_error_code": raw.get("fallback_error_code") if fallback_ok else None,
                },
                "ok": fallback_ok or not bool(raw.get("error")),
                "latency_ms": None,
                "raw_count": raw.get("raw_hit_count"),
                "returned_count": raw.get("usable_hit_count") if fallback_ok else raw.get("returned"),
                "warnings": [_fallback_warning(raw)] if fallback_ok else [],
                "provenance": [],
                "error": None if fallback_ok else _text(raw.get("error") or raw.get("failure_reason"), limit=200),
                "fallback_after_timeout": True if fallback_ok and bool(raw.get("fallback_after_timeout")) else False,
                "fallback_after_error": True if fallback_ok and bool(raw.get("fallback_after_error")) else False,
                "fallback_error_code": raw.get("fallback_error_code") if fallback_ok else "",
                "backend": "duckdb_lexical" if fallback_ok else "",
            }
        )
    return out


def _build_contract(trace: Mapping[str, Any]) -> dict[str, Any]:
    result = _as_dict(trace.get("contract_result"))
    violations = _as_list(result.get("violations"))
    codes = []
    for item in violations:
        if isinstance(item, Mapping) and item.get("code"):
            codes.append(str(item.get("code")))
    warning_codes = []
    for item in _as_list(result.get("warnings")):
        if isinstance(item, Mapping) and item.get("code"):
            warning_codes.append(str(item.get("code")))
    return {
        "status": _contract_status(trace),
        "decision": _text(result.get("decision"), limit=80),
        "public_summary": _text(
            trace.get("contract_public_summary")
            or result.get("public_summary")
            or "Answer grounding checks are available in the trace.",
            limit=240,
        ),
        "severity": _text(result.get("severity"), limit=80),
        "route": _text(result.get("route"), limit=80),
        "violation_codes": list(dict.fromkeys(codes)),
        "warning_codes": list(dict.fromkeys(warning_codes)),
        "scope_overclaim_check": _safe_nested_dict(
            trace.get("scope_overclaim_check") or result.get("scope_overclaim_check"),
            limit=500,
        ),
        "scope_overclaim_violations": [
            _public_dict(item, ["type", "code", "message", "affected_citations", "answer_span", "citation_scopes"], text_limit=260)
            for item in (_as_list(trace.get("scope_overclaim_violations")) or _as_list(result.get("scope_overclaim_violations")))
            if isinstance(item, Mapping)
        ],
        "warnings": [
            _public_dict(item, ["code", "message", "severity", "dimension_id", "caveat_type"], text_limit=180)
            for item in _as_list(result.get("warnings"))
            if isinstance(item, Mapping)
        ],
        "repair_attempts": int(trace.get("contract_attempts") or 0),
        "repair_actions": [
            _public_dict(item, ["attempt", "violations", "action", "route"], text_limit=160)
            for item in _as_list(trace.get("repair_actions"))
            if isinstance(item, Mapping)
        ],
        "evidence_retry_count": int(trace.get("contract_evidence_retry_count") or 0),
        "report_contract_status": _text(trace.get("report_contract_status"), limit=80),
    }


def _safe_nested_dict(value: Any, *, limit: int = 400) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key, raw in value.items():
        if isinstance(raw, str):
            out[str(key)] = _text(raw, limit=limit)
        elif isinstance(raw, (int, float, bool)) or raw is None:
            out[str(key)] = raw
        elif isinstance(raw, list):
            out[str(key)] = [
                _text(item, limit=limit) if not isinstance(item, Mapping) else _safe_nested_dict(item, limit=limit)
                for item in raw[:20]
            ]
        elif isinstance(raw, Mapping):
            out[str(key)] = _safe_nested_dict(raw, limit=limit)
    return out


def _build_semantic_parser(trace: Mapping[str, Any]) -> dict[str, Any]:
    raw = _as_dict(trace.get("semantic_parser"))
    proposal = _as_dict(raw.get("proposal")) or _as_dict(trace.get("semantic_proposal"))
    return {
        "mode": _text(trace.get("semantic_parser_mode") or raw.get("mode"), limit=40),
        "ok": bool(raw.get("ok", False)),
        "source": _text(raw.get("source"), limit=80),
        "error": _text(raw.get("error"), limit=200),
        "diagnostics": _safe_nested_dict(raw.get("diagnostics")),
        "warnings": [
            _public_dict(item, ["field", "value", "reason", "detail"], text_limit=240)
            for item in _as_list(raw.get("warnings"))
            if isinstance(item, Mapping)
        ],
        "proposal": _public_dict(
            proposal,
            [
                "company_mentions",
                "analysis_scope",
                "methodology_intent",
                "requested_dimensions",
                "requested_metrics",
                "user_expectation",
                "safety_intent",
                "time_scope",
                "confidence",
                "ambiguity",
                "needs_clarification",
                "reasons",
            ],
            text_limit=240,
        ),
        "disagreement": _safe_nested_dict(raw.get("disagreement")),
    }


def _build_report(trace: Mapping[str, Any]) -> dict[str, Any] | None:
    report = _as_dict(trace.get("report")) or _as_dict(_as_dict(trace.get("output")).get("report"))
    if not report:
        return None
    sections = []
    for section in _as_list(report.get("sections")):
        if not isinstance(section, Mapping):
            continue
        sections.append(
            _public_dict(
                section,
                [
                    "section_id",
                    "title",
                    "section_status",
                    "citations",
                    "limitations",
                    "confidence",
                    "key_evidence_ids",
                    "contract_status",
                    "markdown",
                ],
                text_limit=1200,
            )
        )
    return {
        "title": _text(report.get("title"), limit=160),
        "company": _text(report.get("company"), limit=120),
        "ticker": _text(report.get("ticker"), limit=40),
        "period": _text(report.get("period"), limit=120),
        "report_type": _text(report.get("report_type"), limit=80),
        "sections": sections,
        "overall_limitations": [_text(x, limit=200) for x in _as_list(report.get("overall_limitations"))],
        "citations": [_text(x, limit=40) for x in _as_list(report.get("citations"))],
        "contract_status": _text(report.get("contract_status"), limit=80),
        "markdown": _text(report.get("markdown"), limit=20000),
        "generated_at": _text(report.get("generated_at"), limit=80),
    }


def _build_progress_events(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for item in _as_list(trace.get("progress_events")):
        if not isinstance(item, Mapping):
            continue
        event = _public_dict(
            item,
            ["event", "status", "message", "node", "timestamp", "elapsed_ms", "metadata"],
            text_limit=500,
        )
        if event.get("event") and event.get("status") and event.get("message") and event.get("timestamp"):
            metadata = _as_dict(item.get("metadata"))
            if metadata:
                event["metadata"] = {
                    str(key): (_text(value, limit=120) if isinstance(value, str) else value)
                    for key, value in list(metadata.items())[:20]
                    if isinstance(value, (str, int, float, bool)) or value is None
                }
            events.append(event)
    return events


def build_trace_ui_model(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable, public-safe trace model for frontend rendering."""
    nodes, edges = _build_workflow(trace)
    tool_calls = _build_tool_calls(trace)
    timeline = [
        {"event": "node", "node_id": node["id"], "status": node["status"], "summary": node.get("summary", "")}
        for node in nodes
        if node["status"] not in {"idle", "skipped"}
    ]
    timeline.extend(
        {
            "event": "tool_call",
            "node_id": "execute_tools",
            "tool_name": call.get("tool_name", ""),
            "ok": call.get("ok", False),
            "returned_count": call.get("returned_count"),
        }
        for call in tool_calls
    )
    query = _text(trace.get("user_query") or trace.get("query"), limit=500)
    companies = [{"ticker": _text(item, limit=40)} for item in _as_list(trace.get("companies"))]
    return {
        "trace_id": _text(trace.get("trace_id"), limit=120),
        "query": query,
        "user_query": query,
        "final_answer": _text(trace.get("final_answer") or trace.get("answer"), limit=30000),
        "local_trace_path": f"data/traces/{_text(trace.get('trace_id'), limit=120)}.json" if trace.get("trace_id") else "",
        "companies": companies,
        "task_type": _text(trace.get("task_type"), limit=80),
        "answer_mode": _text(trace.get("answer_mode"), limit=80),
        "canonical_intent": _safe_nested_dict(trace.get("canonical_intent")),
        "output_language": _text(trace.get("output_language") or _as_dict(trace.get("output")).get("output_language") or _as_dict(trace.get("canonical_intent")).get("output_language"), limit=20),
        "intent_merge_decision": _safe_nested_dict(trace.get("intent_merge_decision")),
        "evidence_policy_id": _text(trace.get("evidence_policy_id"), limit=120),
        "evidence_policy": _safe_nested_dict(trace.get("evidence_policy")),
        "research_plan_raw": _safe_nested_dict(trace.get("research_plan_raw"), limit=800),
        "research_plan_validated": _safe_nested_dict(trace.get("research_plan_validated"), limit=800),
        "research_plan_used": _safe_nested_dict(trace.get("research_plan_used"), limit=800),
        "research_plan_validation": _safe_nested_dict(trace.get("research_plan_validation"), limit=800),
        "required_answer_parts": _as_list(trace.get("required_answer_parts")),
        "legacy_evidence_plan": _safe_nested_dict(trace.get("legacy_evidence_plan"), limit=500),
        "plan_coverage_decision": _safe_nested_dict(trace.get("plan_coverage_decision"), limit=500),
        "requirement_merge_summary": _safe_nested_dict(trace.get("requirement_merge_summary"), limit=500),
        "evidence_plan_used": _safe_nested_dict(trace.get("evidence_plan_used"), limit=300),
        "semantic_parser_mode": _text(trace.get("semantic_parser_mode"), limit=40),
        "semantic_parser": _build_semantic_parser(trace),
        "semantic_proposal": _safe_nested_dict(trace.get("semantic_proposal")),
        "rule_methodology_intent": _text(trace.get("rule_methodology_intent"), limit=80),
        "proposed_methodology_intent": _text(trace.get("proposed_methodology_intent"), limit=80),
        "proposal_validation_warnings": [
            _public_dict(item, ["field", "value", "reason", "detail"], text_limit=240)
            for item in _as_list(trace.get("proposal_validation_warnings"))
            if isinstance(item, Mapping)
        ],
        "intent_conflict": bool(trace.get("intent_conflict", False)),
        "started_at": trace.get("started_at"),
        "finished_at": trace.get("finished_at"),
        "contract_status": _contract_status(trace),
        "contract_decision": _safe_nested_dict(trace.get("contract_decision") or trace.get("contract_result")),
        "relevance_decision": _safe_nested_dict(trace.get("relevance_decision"), limit=500),
        "relevance_status": _text(trace.get("relevance_status"), limit=80),
        "final_answer_source": _text(
            trace.get("final_answer_source")
            or _as_dict(trace.get("output")).get("final_answer_source")
            or _as_dict(trace.get("synthesis")).get("final_answer_source"),
            limit=120,
        ),
        "answer_history": _as_list(trace.get("answer_history")) or _as_list(_as_dict(trace.get("output")).get("answer_history")),
        "research_plan_source": _text(trace.get("research_plan_source"), limit=120),
        "research_plan_fallback_reason": _text(trace.get("research_plan_fallback_reason"), limit=200),
        "research_plan_duration_ms": int(trace.get("research_plan_duration_ms") or 0),
        "analytical_claims": _as_list(trace.get("analytical_claims")) or _as_list(_as_dict(trace.get("synthesis")).get("analytical_claims")),
        "claim_tiers": _as_dict(trace.get("claim_tiers")) or _as_dict(_as_dict(trace.get("synthesis")).get("claim_tiers")),
        "analytical_reasoning_status": _text(trace.get("analytical_reasoning_status") or _as_dict(trace.get("synthesis")).get("analytical_reasoning_status"), limit=80),
        "evidence_health": _text(trace.get("evidence_health") or _as_dict(trace.get("synthesis")).get("evidence_health"), limit=80),
        "tool_error_context": _as_list(trace.get("tool_error_context")) or _as_list(_as_dict(trace.get("synthesis")).get("tool_error_context")),
        "evidence_validation_records": _build_evidence_validation(trace),
        "relevance_repair_attempts": int(trace.get("relevance_repair_attempts") or trace.get("relevance_attempts") or 0),
        "final_route": _text(trace.get("final_route") or _as_dict(trace.get("output")).get("final_route"), limit=80),
        "answer_quality_tier": _text(trace.get("answer_quality_tier") or _as_dict(trace.get("output")).get("answer_quality_tier"), limit=80),
        "quality_tier_reason": _text(trace.get("quality_tier_reason") or _as_dict(trace.get("output")).get("quality_tier_reason"), limit=200),
        "main_question_covered": bool(trace.get("main_question_covered", _as_dict(trace.get("output")).get("main_question_covered", True))),
        "fallback_intent_match": bool(trace.get("fallback_intent_match", _as_dict(trace.get("output")).get("fallback_intent_match", True))),
        "answered_dimensions": _as_list(trace.get("answered_dimensions")) or _as_list(_as_dict(trace.get("output")).get("answered_dimensions")),
        "unresolved_relevance_failures": _as_list(trace.get("unresolved_relevance_failures")) or _as_list(_as_dict(trace.get("output")).get("unresolved_relevance_failures")),
        "format_constraints_satisfied": bool(trace.get("format_constraints_satisfied", _as_dict(trace.get("output")).get("format_constraints_satisfied", True))),
        "format_constraints": _as_dict(trace.get("format_constraints")) or _as_dict(_as_dict(trace.get("output")).get("format_constraints")),
        "repair_applied": bool(trace.get("repair_applied", _as_dict(trace.get("output")).get("repair_applied", False))),
        "repair_owner": _text(trace.get("repair_owner") or _as_dict(trace.get("output")).get("repair_owner"), limit=80),
        "source_before_repair": _text(trace.get("source_before_repair") or _as_dict(trace.get("output")).get("source_before_repair"), limit=120),
        "repair_types": _as_list(trace.get("repair_types")) or _as_list(_as_dict(trace.get("output")).get("repair_types")),
        "material_claim_uncited_count": int(trace.get("material_claim_uncited_count", _as_dict(trace.get("output")).get("material_claim_uncited_count", 0)) or 0),
        "core_missing_parts": _as_list(trace.get("core_missing_parts")) or _as_list(_as_dict(trace.get("output")).get("core_missing_parts")),
        "optional_missing_parts": _as_list(trace.get("optional_missing_parts")) or _as_list(_as_dict(trace.get("output")).get("optional_missing_parts")),
        "risk_items_directly_supported_count": int(trace.get("risk_items_directly_supported_count", _as_dict(trace.get("output")).get("risk_items_directly_supported_count", 0)) or 0),
        "risk_items_template_only_count": int(trace.get("risk_items_template_only_count", _as_dict(trace.get("output")).get("risk_items_template_only_count", 0)) or 0),
        "company_specific_token_leakage": int(trace.get("company_specific_token_leakage", _as_dict(trace.get("output")).get("company_specific_token_leakage", 0)) or 0),
        "language_leakage": int(trace.get("language_leakage", _as_dict(trace.get("output")).get("language_leakage", 0)) or 0),
        "language_leakage_unresolved": bool(trace.get("language_leakage_unresolved", _as_dict(trace.get("output")).get("language_leakage_unresolved", False))),
        "segment_or_product_scope": _text(trace.get("segment_or_product_scope") or _as_dict(trace.get("output")).get("segment_or_product_scope"), limit=120),
        "draft_release_decision": _safe_nested_dict(trace.get("draft_release_decision")),
        "repair_attempts": int(trace.get("contract_attempts") or 0),
        "evidence_retry_count": int(trace.get("contract_evidence_retry_count") or 0),
        "nodes": nodes,
        "edges": edges,
        "timeline": timeline,
        "research_plan": _build_research_plan(trace),
        "analytical_reasoning": _build_analytical_reasoning(trace),
        "evidence_scope": _build_evidence_scope(trace),
        "evidence_scope_by_ref": _build_evidence_scope(trace)["evidence_scope_by_ref"],
        "scope_overclaim_check": _safe_nested_dict(
            trace.get("scope_overclaim_check") or _as_dict(trace.get("contract_result")).get("scope_overclaim_check"),
            limit=500,
        ),
        "scope_overclaim_violations": _as_list(trace.get("scope_overclaim_violations")) or _as_list(_as_dict(trace.get("contract_result")).get("scope_overclaim_violations")),
        "driver_scope_counts": _as_dict(trace.get("driver_scope_counts")),
        "evidence_plan": _build_evidence_plan(trace),
        "answer_part_status_by_id": _safe_nested_dict(trace.get("answer_part_status_by_id"), limit=500),
        "evidence_gap_by_answer_part": _safe_nested_dict(trace.get("evidence_gap_by_answer_part"), limit=500),
        "missing_required_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_required_answer_parts"))],
        "partial_required_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("partial_required_answer_parts"))],
        "missing_but_analyzable_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_but_analyzable_answer_parts"))],
        "missing_and_unanswerable_answer_parts": [_text(item, limit=120) for item in _as_list(trace.get("missing_and_unanswerable_answer_parts"))],
        "evidence_packet": _build_evidence_packet(trace),
        "dimensions": _build_dimensions(trace),
        "citations": _build_citations(trace),
        "contract": _build_contract(trace),
        "report": _build_report(trace),
        "tool_calls": tool_calls,
        "progress_events": _build_progress_events(trace),
    }
