"""Evidence requirement collection results and sufficiency decisions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

from src.agent.analysis_framework import get_fundamental_quality_analysis
from src.agent.metric_availability import (
    DIMENSION_CORE_METRICS,
    DIMENSION_ENHANCED_METRICS,
    normalize_metric_name,
)
from src.agent.types import EvidenceCollectionResult, EvidenceSufficiencyResult
from src.agent.driver_evidence import annotate_driver_evidence, classify_driver_levels

COLLECTION_STATUSES = {"satisfied", "partial", "missing", "rejected"}


@dataclass(frozen=True)
class DimensionSufficiency:
    dimension_id: str
    status: Literal["satisfied", "partial", "missing"]
    satisfied_requirements: list[str]
    missing_requirements: list[str]
    required_available: list[str]
    required_missing: list[str]
    enhanced_available: list[str]
    enhanced_missing: list[str]
    supporting_evidence_ids: list[str]
    allowed_claims: list[str]
    forbidden_claims: list[str]
    limitation: str | None
    limitations: list[str]


def collection_result(
    *,
    requirement_id: str,
    status: str,
    evidence_type: str,
    items: list[dict[str, Any]] | None = None,
    failure_reason: str | None = None,
    retry_count: int = 0,
    **extra: Any,
) -> dict[str, Any]:
    """Build a normalized EvidenceCollectionResult dict."""
    if status not in COLLECTION_STATUSES:
        status = "missing"
    return EvidenceCollectionResult(
        requirement_id=requirement_id,
        status=status,
        evidence_type=evidence_type,
        items=list(items or []),
        failure_reason=failure_reason,
        retry_count=max(int(retry_count or 0), 0),
        **dict(extra or {}),
    ).model_dump(exclude_none=True)


def _requirements(evidence_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [r for r in evidence_plan.get("evidence_requirements", []) or [] if isinstance(r, Mapping)]


def _results_by_requirement(results: list[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    out: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        rid = str(result.get("requirement_id", "")).strip()
        if rid:
            out.setdefault(rid, []).append(result)
    return out


def _result_status(req_results: list[Mapping[str, Any]]) -> str:
    statuses = {str(r.get("status", "")) for r in req_results}
    if "satisfied" in statuses:
        return "satisfied"
    if "partial" in statuses:
        return "partial"
    if "rejected" in statuses:
        return "rejected"
    return "missing"


def _requirement_failure_reason(req_results: list[Mapping[str, Any]]) -> str | None:
    for result in req_results:
        if result.get("failure_reason"):
            return str(result.get("failure_reason"))
        if result.get("rejected_evidence_reason"):
            return str(result.get("rejected_evidence_reason"))
    return None


def _requested_metric_set(requirement: Mapping[str, Any]) -> set[str]:
    metrics: set[str] = set()
    for raw in list(requirement.get("metrics", []) or []) + [requirement.get("metric")]:
        metric = normalize_metric_name(str(raw or ""))
        if metric:
            metrics.add(metric)
    return metrics


def _raw_metric_set(items: list[Mapping[str, Any]]) -> set[str]:
    return {normalize_metric_name(str(item.get("metric") or "")) for item in items if str(item.get("metric") or "").strip()}


def _valuation_quality_filter_reason(requirement: Mapping[str, Any], reason: str) -> str:
    if reason != "quality_filter_rejected":
        return reason
    rid = str(requirement.get("requirement_id") or "").upper().replace("-", "_")
    raw_metric = str(requirement.get("metric") or "").strip()
    raw_metrics = {str(item or "").strip() for item in requirement.get("metrics", []) or []}
    metric = normalize_metric_name(str(requirement.get("metric") or ""))
    metrics = {normalize_metric_name(str(item or "")) for item in requirement.get("metrics", []) or []}
    if "VALUATION_PRICE" in rid or raw_metric in {"price", "adjusted_close"} or metric in {"price", "adjusted_close"} or {"price", "adjusted_close"} & (metrics | raw_metrics):
        return "valuation_price_quality_filter_rejected"
    if "VALUATION_SHARES_OUTSTANDING" in rid or raw_metric == "shares_outstanding" or metric == "shares_outstanding" or "shares_outstanding" in (metrics | raw_metrics):
        return "valuation_shares_quality_filter_rejected"
    if "VALUATION_MARKET_CAP" in rid or raw_metric == "market_cap" or metric == "market_cap" or "market_cap" in (metrics | raw_metrics):
        return "valuation_market_cap_dependency_rejected"
    return reason


def _specific_numeric_validation_failure(
    requirement: Mapping[str, Any],
    raw_items: list[Mapping[str, Any]],
    raw_failure_reason: str | None,
) -> str:
    raw_failure = str(raw_failure_reason or "").strip()
    if raw_failure in {"period_mismatch", "metric_alias_mismatch", "quality_filter_rejected", "missing_source_requirement_id"}:
        return _valuation_quality_filter_reason(requirement, raw_failure)
    if "period" in raw_failure:
        return "period_mismatch"
    if raw_failure in {"metric_mapping_failed", "unsupported_planner_metric"}:
        return "metric_alias_mismatch"
    if raw_failure in {"evidence_filter_mismatch", "numeric_validation_failed"}:
        return _valuation_quality_filter_reason(requirement, "quality_filter_rejected")
    if raw_items:
        rid = str(requirement.get("requirement_id") or "").strip()
        if rid and not any(str(item.get("source_requirement_id") or item.get("requirement_id") or "").strip() == rid for item in raw_items):
            return "missing_source_requirement_id"
        requested = _requested_metric_set(requirement)
        returned = _raw_metric_set(raw_items)
        if requested and returned and requested.isdisjoint(returned):
            return "metric_alias_mismatch"
    return _valuation_quality_filter_reason(requirement, "quality_filter_rejected")


def _requirement_retry_count(req_results: list[Mapping[str, Any]]) -> int:
    retry_count = 0
    for result in req_results:
        retry_count = max(retry_count, int(result.get("retry_count", 0) or 0))
    return retry_count


def _result_stat(req_results: list[Mapping[str, Any]], key: str) -> int:
    value = 0
    for result in req_results:
        value = max(value, int(result.get(key, 0) or 0))
    return value


def _drop_stage(
    *,
    raw_hit_count: int,
    section_filtered_hit_count: int,
    usable_hit_count: int,
    snippet_support_passed_count: int,
    validated_text_claim_count: int,
    text_citation_kept_count: int,
    final_validated_text_count: int,
    failure_reason: str | None,
) -> str:
    if final_validated_text_count > 0:
        return "satisfied"
    if raw_hit_count <= 0:
        return "no_raw_hits"
    if section_filtered_hit_count <= 0:
        return "section_filter_dropped"
    if usable_hit_count <= 0:
        return "quality_filter_dropped"
    if snippet_support_passed_count <= 0:
        return "snippet_support_failed"
    if validated_text_claim_count <= 0:
        return "claim_validation_failed"
    if text_citation_kept_count <= 0:
        if failure_reason:
            return "citation_policy_dropped"
        return "citation_policy_dropped"
    return "final_bundle_dropped"


def _required_ids(requirements: list[Mapping[str, Any]]) -> list[str]:
    return [
        str(r.get("requirement_id", "")).strip()
        for r in requirements
        if bool(r.get("required", True)) and str(r.get("requirement_scope") or "core") == "core"
    ]


def _ids_of_type(requirements: list[Mapping[str, Any]], req_type: str, *, required_only: bool = False) -> list[str]:
    out: list[str] = []
    for req in requirements:
        if str(req.get("requirement_type", "")) != req_type:
            continue
        if required_only and (
            not bool(req.get("required", True))
            or str(req.get("requirement_scope") or "core") != "core"
        ):
            continue
        rid = str(req.get("requirement_id", "")).strip()
        if rid:
            out.append(rid)
    return out


def _normalize_rejected_requirements(rejected: list[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, item in enumerate(rejected):
        if not isinstance(item, Mapping):
            continue
        rid = str(item.get("requirement_id", "")).strip() or f"REJECTED_{idx + 1}"
        normalized.append({"requirement_id": rid, **dict(item)})
    return normalized


def _rejected_requirement_ids(rejected_requirements: list[Mapping[str, Any]]) -> set[str]:
    return {
        str(item.get("requirement_id", "")).strip()
        for item in rejected_requirements
        if str(item.get("requirement_id", "")).strip()
    }


def _preview_item(item: Mapping[str, Any]) -> dict[str, Any]:
    keep = (
        "requirement_id",
        "source_requirement_id",
        "ticker",
        "company",
        "metric",
        "role",
        "evidence_role",
        "period",
        "period_scope",
        "period_type",
        "period_end",
        "value",
        "unit",
        "quality_status",
        "validation_failure_reason",
        "compare_period",
        "compare_value",
        "current_requirement_id",
        "comparator_requirement_id",
        "dependencies",
        "source_provider",
        "filing_id",
        "section",
        "snippet",
        "text_snippet",
        "supporting_snippet",
        "supporting_terms",
        "claim",
        "driver_level",
        "driver_levels",
        "claim_scope",
        "allowed_claim_strength",
        "scope_reason",
        "event_date",
        "event_type",
        "fiscal_period",
    )
    out = {key: item.get(key) for key in keep if key in item}
    for text_key in ("snippet", "supporting_snippet"):
        if text_key in out:
            out[text_key] = str(out[text_key] or "")[:240]
    if any(key in item for key in ("snippet", "supporting_snippet", "text_snippet", "claim")):
        annotated = annotate_driver_evidence(item)
        out.setdefault("driver_level", annotated.get("driver_level"))
        out.setdefault("driver_levels", annotated.get("driver_levels", []))
        out.setdefault("claim_scope", annotated.get("claim_scope"))
        out.setdefault("allowed_claim_strength", annotated.get("allowed_claim_strength"))
        out.setdefault("scope_reason", annotated.get("scope_reason"))
    return out


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 1.0
    return round(numerator / denominator, 6)


def _required_answer_parts(evidence_plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    parts = evidence_plan.get("required_answer_parts")
    if not isinstance(parts, list):
        parts = dict(evidence_plan.get("sufficiency_criteria", {}) or {}).get("required_answer_parts", [])
    return [part for part in parts or [] if isinstance(part, Mapping)]


def _linked_requirement_ids_for_part(
    part: Mapping[str, Any],
    requirements: list[Mapping[str, Any]],
) -> list[str]:
    part_id = str(part.get("id") or "").strip()
    request_ids = {str(item).strip() for item in part.get("evidence_request_ids", []) or [] if str(item).strip()}
    linked: list[str] = []
    for req in requirements:
        rid = str(req.get("requirement_id") or "").strip()
        if not rid:
            continue
        answer_part_ids = {str(item).strip() for item in req.get("answer_part_ids", []) or [] if str(item).strip()}
        evidence_request_id = str(req.get("evidence_request_id") or "").strip()
        if part_id and part_id in answer_part_ids:
            linked.append(rid)
        elif evidence_request_id and evidence_request_id in request_ids:
            linked.append(rid)
    return sorted(dict.fromkeys(linked))


def _requirement_role(req: Mapping[str, Any], status: Mapping[str, Any] | None = None) -> str:
    status = status or {}
    role = str(req.get("evidence_role") or status.get("evidence_role") or "").strip()
    if role:
        return role
    nested = req.get("requirement")
    if isinstance(nested, Mapping):
        return str(nested.get("evidence_role") or "").strip()
    return ""


def _requirement_metrics(req: Mapping[str, Any]) -> set[str]:
    metrics = {str(metric).strip() for metric in req.get("metrics", []) or [] if str(metric).strip()}
    metric = str(req.get("metric") or "").strip()
    if metric:
        metrics.add(metric)
    return metrics


def _items_for_requirement(status_map: Mapping[str, Mapping[str, Any]], rid: str) -> list[dict[str, Any]]:
    item = status_map.get(rid, {})
    previews = item.get("items_preview", []) if isinstance(item, Mapping) else []
    return [dict(row) for row in previews if isinstance(row, Mapping)]


def _distinct_revenue_periods(status_map: Mapping[str, Mapping[str, Any]], rids: list[str]) -> set[str]:
    periods: set[str] = set()
    for rid in rids:
        for item in _items_for_requirement(status_map, rid):
            metric = str(item.get("metric") or "").strip()
            if metric and metric != "revenue":
                continue
            period = str(item.get("period_end") or item.get("period") or "").strip()
            if period:
                periods.add(period)
    return periods


def _text_driver_levels(status_map: Mapping[str, Mapping[str, Any]], rids: list[str]) -> set[str]:
    levels: set[str] = set()
    scope_to_level = {
        "company": "company_level_driver",
        "segment": "segment_level_driver",
        "product": "product_level_driver",
        "market_context": "market_context",
        "unknown": "unknown",
    }
    for rid in rids:
        for item in _items_for_requirement(status_map, rid):
            claim_scope = str(item.get("claim_scope") or "").strip()
            if claim_scope:
                levels.add(scope_to_level.get(claim_scope, "unknown"))
            elif item.get("driver_level"):
                levels.add(str(item.get("driver_level")))
            elif isinstance(item.get("driver_levels"), list):
                levels.update(str(level) for level in item.get("driver_levels", []) if str(level).strip())
            else:
                text = " ".join(
                    str(item.get(key) or "")
                    for key in ("claim", "supporting_snippet", "snippet", "text_snippet", "section")
                    if str(item.get(key) or "").strip()
                )
                levels.update(classify_driver_levels(text))
    return levels


def _causal_quantify_growth_status(
    *,
    linked_ids: list[str],
    linked_reqs: Mapping[str, Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    satisfied_ids: list[str],
    partial_ids: list[str],
) -> tuple[str, str, dict[str, Any]]:
    def role_ids(role: str, *, statuses: set[str] | None = None) -> list[str]:
        out: list[str] = []
        for rid in linked_ids:
            req = linked_reqs.get(rid, {})
            status = str(status_map.get(rid, {}).get("status") or "missing")
            if statuses and status not in statuses:
                continue
            if _requirement_role(req, status_map.get(rid, {})) == role:
                out.append(rid)
        return out

    current_ids = role_ids("current_revenue", statuses={"satisfied"})
    comparator_ids = role_ids("comparator_revenue", statuses={"satisfied"})
    calc_ids = role_ids("revenue_growth_calculation", statuses={"satisfied"})
    growth_text_ids = role_ids("revenue_growth_text", statuses={"satisfied"})
    all_current_role_ids = role_ids("current_revenue")
    all_comparator_role_ids = role_ids("comparator_revenue")
    all_calc_role_ids = role_ids("revenue_growth_calculation")
    all_growth_text_role_ids = role_ids("revenue_growth_text")
    all_revenue_numeric_ids = [
        rid
        for rid in linked_ids
        if str(linked_reqs.get(rid, {}).get("requirement_type") or "") == "numeric"
        and "revenue" in _requirement_metrics(linked_reqs.get(rid, {}))
        and str(status_map.get(rid, {}).get("status") or "") == "satisfied"
    ]
    revenue_growth_metric_ids = [
        rid
        for rid in linked_ids
        if "revenue_growth" in _requirement_metrics(linked_reqs.get(rid, {}))
        and str(status_map.get(rid, {}).get("status") or "") == "satisfied"
    ]
    periods = _distinct_revenue_periods(status_map, all_revenue_numeric_ids)
    current_ok = bool(current_ids) or bool(all_revenue_numeric_ids)
    comparator_ok = bool(comparator_ids and (not periods or len(periods) >= 2)) or len(periods) >= 2
    calc_ok = bool(calc_ids or revenue_growth_metric_ids)
    growth_text_levels = _text_driver_levels(status_map, growth_text_ids)
    total_growth_text_ok = bool(growth_text_ids and "company_level_driver" in growth_text_levels)
    growth_text_segment_only = bool(growth_text_ids and growth_text_levels and "company_level_driver" not in growth_text_levels)

    def role_quality(rids: list[str], *, default_role: str) -> dict[str, Any]:
        status = "missing"
        reason = "missing"
        if any(str(status_map.get(rid, {}).get("status") or "") == "satisfied" for rid in rids):
            status = "satisfied"
            reason = "valid"
        elif any(str(status_map.get(rid, {}).get("status") or "") == "partial" for rid in rids):
            status = "partial"
            reason = "partial"
        elif rids:
            status = "missing"
            reason = next(
                (
                    str(status_map.get(rid, {}).get("quality_status") or status_map.get(rid, {}).get("failure_reason") or "").strip()
                    for rid in rids
                    if str(status_map.get(rid, {}).get("quality_status") or status_map.get(rid, {}).get("failure_reason") or "").strip()
                ),
                "missing",
            )
        return {"role": default_role, "status": status, "quality_status": reason, "requirement_ids": rids}

    current_quality = role_quality(all_current_role_ids, default_role="current_revenue")
    comparator_quality = role_quality(all_comparator_role_ids, default_role="comparator_revenue")
    calc_quality = role_quality(all_calc_role_ids, default_role="revenue_growth_calculation")
    growth_text_quality = role_quality(all_growth_text_role_ids, default_role="revenue_growth_text")
    if growth_text_segment_only:
        growth_text_quality["quality_status"] = "segment_only"
    details = {
        "current_revenue_requirement_ids": current_ids,
        "comparator_revenue_requirement_ids": comparator_ids,
        "revenue_growth_calculation_requirement_ids": calc_ids,
        "revenue_growth_text_requirement_ids": growth_text_ids,
        "distinct_revenue_periods": sorted(periods),
        "growth_text_driver_levels": sorted(growth_text_levels),
        "current_revenue": current_quality,
        "comparator_revenue": comparator_quality,
        "revenue_growth_calculation": calc_quality,
        "revenue_growth_text": growth_text_quality,
    }
    if (current_ok and comparator_ok and calc_ok) or total_growth_text_ok:
        return "satisfied", "", details
    if current_ok and str(comparator_quality.get("quality_status") or "") == "same_period_comparator":
        return "partial", "growth_calc_invalid_same_period", details
    if current_ok and str(comparator_quality.get("quality_status") or "") == "incomparable_period_scope":
        return "partial", "growth_calc_invalid_incomparable_period", details
    if current_ok and str(calc_quality.get("quality_status") or "") in {"same_period_comparator", "invalid_growth_dependencies", "zero_comparator", "incomparable_period_scope"}:
        return "partial", f"growth_calc_invalid_{calc_quality['quality_status']}", details
    if current_ok and growth_text_segment_only:
        return "partial", "growth_calc_unavailable_or_segment_only", details
    if current_ok or comparator_ids or calc_ids or growth_text_ids or partial_ids or satisfied_ids:
        return "partial", "growth_calc_unavailable_or_segment_only", details
    return "missing", "revenue_growth_evidence_missing", details


def _causal_driver_status(
    *,
    linked_ids: list[str],
    linked_reqs: Mapping[str, Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    partial_ids: list[str],
) -> tuple[str, str, dict[str, Any]]:
    text_ids = [
        rid
        for rid in linked_ids
        if str(linked_reqs.get(rid, {}).get("requirement_type") or "") == "text"
        and str(status_map.get(rid, {}).get("status") or "") == "satisfied"
    ]
    levels = _text_driver_levels(status_map, text_ids)
    details = {"driver_text_requirement_ids": text_ids, "driver_levels": sorted(levels)}
    if "company_level_driver" in levels:
        return "satisfied", "", details
    if levels & {"segment_level_driver", "product_level_driver", "market_context"}:
        return "partial", "only_segment_or_product_driver_evidence", details
    if partial_ids:
        return "partial", "driver_text_evidence_partial", details
    any_signal = any(str(item.get("status") or "") in {"satisfied", "partial"} for item in status_map.values())
    any_degraded_request = any(
        str(item.get("status") or "") in {"missing", "rejected"}
        and str(item.get("failure_reason") or "").strip()
        for item in status_map.values()
    )
    if any_signal or any_degraded_request:
        return "missing_but_analyzable", "driver_text_evidence_missing_but_analyzable", details
    return "missing_and_unanswerable", "driver_text_evidence_missing", details


def _answer_part_status_value(status_map: Mapping[str, Mapping[str, Any]], part_id: str) -> str:
    item = status_map.get(part_id, {}) if isinstance(status_map, Mapping) else {}
    return str(dict(item or {}).get("status") or "")


def build_answer_part_status(
    evidence_plan: Mapping[str, Any],
    status_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize evidence gaps by ResearchPlan required answer part."""
    parts = _required_answer_parts(evidence_plan)
    requirements = _requirements(evidence_plan)
    question_type = str(dict(evidence_plan.get("research_plan", {}) or {}).get("question_type") or "")
    status_by_id: dict[str, dict[str, Any]] = {}
    gap_by_part: dict[str, dict[str, Any]] = {}
    missing_required_parts: list[str] = []
    partial_required_parts: list[str] = []
    missing_but_analyzable_parts: list[str] = []
    missing_and_unanswerable_parts: list[str] = []

    for part in parts:
        part_id = str(part.get("id") or "").strip()
        if not part_id:
            continue
        required = bool(part.get("required", True))
        linked_ids = _linked_requirement_ids_for_part(part, requirements)
        linked_statuses = {rid: str(status_map.get(rid, {}).get("status") or "missing") for rid in linked_ids}
        linked_reqs = {rid: next((req for req in requirements if str(req.get("requirement_id") or "") == rid), {}) for rid in linked_ids}
        satisfied_ids = [rid for rid, status in linked_statuses.items() if status == "satisfied"]
        partial_ids = [rid for rid, status in linked_statuses.items() if status == "partial"]
        missing_ids = [rid for rid, status in linked_statuses.items() if status in {"missing", "rejected"}]

        reason = ""
        details: dict[str, Any] = {}
        if part_id == "state_evidence_boundary":
            part_status = "satisfied"
        elif part_id == "evidence_boundary":
            part_status = "satisfied"
        elif question_type == "causal_explanation" and part_id == "direct_answer":
            part_status = "satisfied"
        elif question_type == "causal_explanation" and part_id == "verified_evidence":
            part_status = "satisfied" if satisfied_ids else ("partial" if partial_ids else "missing_but_analyzable")
        elif question_type == "causal_explanation" and part_id == "quantify_growth":
            part_status, reason, details = _causal_quantify_growth_status(
                linked_ids=linked_ids,
                linked_reqs=linked_reqs,
                status_map=status_map,
                satisfied_ids=satisfied_ids,
                partial_ids=partial_ids,
            )
        elif question_type == "causal_explanation" and part_id == "identify_growth_drivers":
            part_status, reason, details = _causal_driver_status(
                linked_ids=linked_ids,
                linked_reqs=linked_reqs,
                status_map=status_map,
                partial_ids=partial_ids,
            )
        elif question_type == "causal_explanation" and part_id == "inferred_drivers":
            driver_status = _answer_part_status_value(status_by_id, "identify_growth_drivers")
            if driver_status in {"satisfied", "partial", "missing_but_analyzable"}:
                part_status = driver_status
                reason = "inference_requires_boundary" if driver_status != "satisfied" else ""
            else:
                part_status = "missing_but_analyzable"
                reason = "inference_framework_available_without_direct_driver_text"
        elif question_type == "causal_explanation" and part_id in {"hypotheses_to_verify", "counterpoints"}:
            part_status = "satisfied"
        elif linked_ids:
            part_status = "satisfied" if satisfied_ids else ("partial" if partial_ids else "missing")
        else:
            part_status = "satisfied"

        if required and part_status in {"missing", "missing_and_unanswerable"}:
            missing_required_parts.append(part_id)
            if part_status == "missing_and_unanswerable":
                missing_and_unanswerable_parts.append(part_id)
        elif required and part_status == "partial":
            partial_required_parts.append(part_id)
        elif required and part_status == "missing_but_analyzable":
            missing_but_analyzable_parts.append(part_id)
        gap = {
            "part_id": part_id,
            "description": str(part.get("description") or ""),
            "required": required,
            "status": part_status,
            "linked_requirement_ids": linked_ids,
            "satisfied_requirement_ids": satisfied_ids,
            "partial_requirement_ids": partial_ids,
            "missing_requirement_ids": missing_ids,
            "reason": reason,
            **details,
        }
        if reason:
            pass
        elif question_type == "causal_explanation" and part_id == "identify_growth_drivers" and part_status != "satisfied":
            gap["reason"] = "driver_text_evidence_missing_but_analyzable" if part_status == "missing_but_analyzable" else "driver_text_evidence_missing"
        elif part_status != "satisfied":
            gap["reason"] = "linked_evidence_not_satisfied"
        status_by_id[part_id] = gap
        gap_by_part[part_id] = gap

    return {
        "answer_part_status_by_id": status_by_id,
        "evidence_gap_by_answer_part": gap_by_part,
        "missing_required_answer_parts": missing_required_parts,
        "partial_required_answer_parts": partial_required_parts,
        "missing_but_analyzable_answer_parts": missing_but_analyzable_parts,
        "missing_and_unanswerable_answer_parts": missing_and_unanswerable_parts,
        "answer_parts_fully_satisfied": not missing_required_parts and not partial_required_parts and not missing_but_analyzable_parts,
        "answer_parts_clean_pass": not missing_required_parts and not partial_required_parts and not missing_but_analyzable_parts,
    }


def _tool_error_context(collection_results: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    context: list[dict[str, Any]] = []
    for result in collection_results:
        status = str(result.get("status") or "")
        reason = str(result.get("failure_reason") or "").strip()
        if status not in {"missing", "partial", "rejected"} and not reason:
            continue
        lowered = reason.lower()
        if any(term in lowered for term in ("timeout", "timed out", "oom", "out of memory", "exception", "error")):
            kind = "tool_execution_error"
        elif status == "partial":
            kind = "retrieval_degraded"
        else:
            kind = "no_matching_evidence"
        context.append(
            {
                "requirement_id": str(result.get("requirement_id") or ""),
                "status": status,
                "evidence_type": str(result.get("evidence_type") or ""),
                "kind": kind,
                "failure_reason": reason,
            }
        )
    return context


def _evidence_health(
    *,
    overall_status: str,
    answer_part_summary: Mapping[str, Any],
    tool_context: list[dict[str, Any]],
) -> str:
    if tool_context and any(str(item.get("kind") or "") in {"tool_execution_error", "retrieval_degraded"} for item in tool_context):
        return "degraded"
    if overall_status in {"sufficient", "focused_sufficient"} and bool(answer_part_summary.get("answer_parts_clean_pass", True)):
        return "complete"
    if answer_part_summary.get("missing_but_analyzable_answer_parts"):
        return "degraded"
    if overall_status == "partial" or answer_part_summary.get("partial_required_answer_parts"):
        return "partial"
    if overall_status == "insufficient":
        return "failed"
    return "partial"


def _clean_dimension_ids(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return sorted({str(item).strip() for item in values if str(item).strip()})


def _dimension_ids_with_status(dimension_status_by_id: Mapping[str, Mapping[str, Any]], status: str) -> list[str]:
    return sorted(
        str(dimension_id)
        for dimension_id, item in dimension_status_by_id.items()
        if str(item.get("status", "")) == status
    )


def normalize_dimension_status_contract(
    dimension_status_by_id: Mapping[str, Any] | None,
    *,
    satisfied_dimensions: list[str] | None = None,
    partial_dimensions: list[str] | None = None,
    missing_dimensions: list[str] | None = None,
    dimension_coverage_rate: float | None = None,
    weighted_dimension_coverage_rate: float | None = None,
    framework_sufficiency_status: str | None = None,
    missing_required_requirements_count: int | None = None,
    missing_optional_requirements_count: int | None = None,
    missing_enhanced_requirements_count: int | None = None,
) -> dict[str, Any]:
    """Return the canonical DimensionStatus contract plus legacy aliases.

    `dimension_status_by_id` is canonical. `dimension_status_map` and
    `covered_dimensions` are compatibility aliases retained for existing traces
    and UI/eval consumers.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for raw_dimension_id, raw_item in dict(dimension_status_by_id or {}).items():
        dimension_id = str(raw_dimension_id).strip()
        if not dimension_id:
            continue
        if isinstance(raw_item, Mapping):
            by_id[dimension_id] = dict(raw_item)
        else:
            by_id[dimension_id] = {"status": str(raw_item or "unknown")}

    derived_satisfied = _dimension_ids_with_status(by_id, "satisfied")
    derived_partial = _dimension_ids_with_status(by_id, "partial")
    derived_missing = _dimension_ids_with_status(by_id, "missing")
    if by_id:
        satisfied = derived_satisfied
        partial = derived_partial
        missing = derived_missing
    else:
        satisfied = _clean_dimension_ids(satisfied_dimensions) if satisfied_dimensions is not None else []
        partial = _clean_dimension_ids(partial_dimensions) if partial_dimensions is not None else []
        missing = _clean_dimension_ids(missing_dimensions) if missing_dimensions is not None else []
    total = len(by_id)
    if dimension_coverage_rate is None:
        dimension_coverage_rate = _safe_rate(len(satisfied), total)
    if weighted_dimension_coverage_rate is None:
        weighted_dimension_coverage_rate = _safe_rate((len(satisfied) * 2) + len(partial), total * 2)
    if framework_sufficiency_status is None:
        if total <= 0 or len(satisfied) == total:
            framework_sufficiency_status = "sufficient"
        elif satisfied or partial:
            framework_sufficiency_status = "partial"
        else:
            framework_sufficiency_status = "insufficient"

    payload: dict[str, Any] = {
        "dimension_status_by_id": by_id,
        "dimension_status_map": by_id,
        "satisfied_dimensions": satisfied,
        "covered_dimensions": list(satisfied),
        "partial_dimensions": partial,
        "missing_dimensions": missing,
        "dimension_coverage_rate": dimension_coverage_rate,
        "weighted_dimension_coverage_rate": weighted_dimension_coverage_rate,
        "framework_sufficiency_status": str(framework_sufficiency_status or ""),
    }
    if missing_required_requirements_count is not None:
        payload["missing_required_requirements_count"] = int(missing_required_requirements_count or 0)
    if missing_optional_requirements_count is not None:
        payload["missing_optional_requirements_count"] = int(missing_optional_requirements_count or 0)
    if missing_enhanced_requirements_count is not None:
        payload["missing_enhanced_requirements_count"] = int(missing_enhanced_requirements_count or 0)
    return payload


def _companies_for(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    req_type: str,
    *,
    required_only: bool = False,
    satisfied_only: bool = False,
) -> set[str]:
    companies: set[str] = set()
    for req in requirements:
        if str(req.get("requirement_type", "")) != req_type:
            continue
        if required_only and (
            not bool(req.get("required", True))
            or str(req.get("requirement_scope") or "core") != "core"
        ):
            continue
        rid = str(req.get("requirement_id", "")).strip()
        if satisfied_only:
            status = str(status_map.get(rid, {}).get("status", ""))
            if status != "satisfied":
                continue
        company = str(req.get("company") or "").upper().strip()
        if company:
            companies.add(company)
    return companies


def _company_evidence_balance(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> float | None:
    required_numeric_companies = _companies_for(requirements, status_map, "numeric", required_only=True)
    required_text_companies = _companies_for(requirements, status_map, "text", required_only=True)
    if not required_numeric_companies and not required_text_companies:
        return None
    numeric_ok = _companies_for(
        requirements,
        status_map,
        "numeric",
        required_only=True,
        satisfied_only=True,
    )
    text_ok = _companies_for(
        requirements,
        status_map,
        "text",
        required_only=True,
        satisfied_only=True,
    )
    score = 0.0
    parts = 0
    if required_numeric_companies:
        parts += 1
        score += 1.0 if required_numeric_companies.issubset(numeric_ok) else 0.0
    if required_text_companies:
        parts += 1
        score += 1.0 if required_text_companies.issubset(text_ok) else 0.0
    return round(score / max(parts, 1), 6)


def _dimension_metric_satisfied(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    *,
    dimension_id: str,
    metric: str,
    requirement_type: str | None = None,
) -> bool:
    for req in requirements:
        if str(req.get("dimension_id") or "") != dimension_id:
            continue
        if requirement_type and str(req.get("requirement_type") or "") != requirement_type:
            continue
        metrics = {normalize_metric_name(str(item)) for item in req.get("metrics", []) or [] if str(item)}
        if str(req.get("metric") or ""):
            metrics.add(normalize_metric_name(str(req.get("metric"))))
        if normalize_metric_name(metric) not in metrics:
            continue
        rid = str(req.get("requirement_id", "")).strip()
        if str(status_map.get(rid, {}).get("status", "")) == "satisfied":
            return True
    return False


def _dimension_metric_has_signal(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    *,
    dimension_id: str,
    metric: str,
    requirement_type: str | None = None,
) -> bool:
    for req in requirements:
        if str(req.get("dimension_id") or "") != dimension_id:
            continue
        if requirement_type and str(req.get("requirement_type") or "") != requirement_type:
            continue
        metrics = {normalize_metric_name(str(item)) for item in req.get("metrics", []) or [] if str(item)}
        if str(req.get("metric") or ""):
            metrics.add(normalize_metric_name(str(req.get("metric"))))
        if normalize_metric_name(metric) not in metrics:
            continue
        rid = str(req.get("requirement_id", "")).strip()
        if str(status_map.get(rid, {}).get("status", "")) in {"satisfied", "partial"}:
            return True
    return False


def _dimension_any_text_signal(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    *,
    dimension_id: str,
) -> bool:
    for req in requirements:
        if str(req.get("dimension_id") or "") != dimension_id:
            continue
        if str(req.get("requirement_type") or "") != "text":
            continue
        rid = str(req.get("requirement_id") or "").strip()
        if str(status_map.get(rid, {}).get("status", "")) in {"satisfied", "partial"}:
            return True
    return False


def _metric_rule_dimension_status(
    *,
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
    dimension_id: str,
    fallback_status: str,
) -> str:
    def satisfied(metric: str, req_type: str | None = None) -> bool:
        return _dimension_metric_satisfied(
            requirements,
            status_map,
            dimension_id=dimension_id,
            metric=metric,
            requirement_type=req_type,
        )

    def signal(metric: str, req_type: str | None = None) -> bool:
        return _dimension_metric_has_signal(
            requirements,
            status_map,
            dimension_id=dimension_id,
            metric=metric,
            requirement_type=req_type,
        )

    if dimension_id == "revenue_quality":
        if satisfied("revenue_growth"):
            return "satisfied"
        if satisfied("revenue", "numeric"):
            return "partial"
        return "partial" if signal("revenue", "numeric") else "missing"
    if dimension_id == "profitability_quality":
        has_income = satisfied("net_income", "numeric")
        has_margin = any(satisfied(metric) for metric in ("net_margin", "gross_margin", "operating_margin"))
        has_signal = has_income or any(signal(metric) for metric in ("net_margin", "gross_margin", "operating_margin"))
        if has_income and has_margin:
            return "satisfied"
        if has_signal:
            return "partial"
        return "missing"
    if dimension_id == "cash_flow_quality":
        has_ocf = satisfied("operating_cash_flow")
        has_fcf = satisfied("free_cash_flow")
        has_conversion = any(satisfied(metric) for metric in ("cash_conversion", "cfo_to_net_income"))
        if has_ocf and has_fcf and has_conversion:
            return "satisfied"
        if has_ocf or has_fcf:
            return "partial"
        if any(signal(metric) for metric in ("operating_cash_flow", "free_cash_flow")):
            return "partial"
        return "missing"
    if dimension_id == "balance_sheet_and_capital_intensity":
        has_cash = satisfied("cash_and_equivalents", "numeric")
        has_debt = satisfied("total_debt", "numeric")
        has_assets = satisfied("total_assets", "numeric")
        has_liabilities = satisfied("total_liabilities", "numeric")
        if has_cash and has_debt and has_assets and has_liabilities:
            return "satisfied"
        if any(
            (has_cash, has_debt, has_assets, has_liabilities, satisfied("shareholders_equity", "numeric"))
        ):
            return "partial"
        if any(signal(metric) for metric in ("inventory", "receivables", "capital_expenditure", "capex_to_revenue")):
            return "partial"
        return "missing"
    if dimension_id == "moat_and_competitive_risk":
        return "satisfied" if _dimension_any_text_signal(requirements, status_map, dimension_id=dimension_id) else "missing"
    if dimension_id == "business_model":
        return "satisfied" if _dimension_any_text_signal(requirements, status_map, dimension_id=dimension_id) else "missing"
    if dimension_id == "valuation_and_risk_boundary":
        has_price = satisfied("price", "numeric") or satisfied("adjusted_close", "numeric")
        has_market_cap = satisfied("market_cap")
        has_pe = satisfied("pe_ratio")
        has_ps = satisfied("ps_ratio")
        has_fcf_yield = satisfied("fcf_yield")
        has_multiple = has_pe or has_ps or has_fcf_yield
        if has_market_cap and has_pe and has_ps:
            return "satisfied"
        if has_price or has_market_cap or has_multiple:
            return "partial"
        return "missing"
    return fallback_status


def _requirement_metric_names(req: Mapping[str, Any]) -> set[str]:
    metrics = {
        normalize_metric_name(str(item))
        for item in req.get("metrics", []) or []
        if str(item).strip()
    }
    if str(req.get("metric") or "").strip():
        metrics.add(normalize_metric_name(str(req.get("metric"))))
    return {metric for metric in metrics if metric}


def _dimension_metric_availability(
    *,
    dimension_id: str,
    reqs: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str], list[str], list[str], set[str]]:
    available: set[str] = set()
    required_planned: set[str] = set()
    optional_planned: set[str] = set()
    for req in reqs:
        req_metrics = _requirement_metric_names(req)
        if bool(req.get("required", True)):
            required_planned |= req_metrics
        else:
            optional_planned |= req_metrics
        rid = str(req.get("requirement_id") or "").strip()
        if str(status_map.get(rid, {}).get("status") or "") in {"satisfied", "partial"}:
            available |= req_metrics
    core_metric_set = set(DIMENSION_CORE_METRICS.get(dimension_id, ()))
    enhanced_metric_set = set(DIMENSION_ENHANCED_METRICS.get(dimension_id, ()))
    core = core_metric_set & required_planned
    enhanced = (enhanced_metric_set | (core_metric_set & optional_planned)) & (required_planned | optional_planned)
    enhanced -= core
    return (
        sorted(core & available),
        sorted(core - available),
        sorted(enhanced & available),
        sorted(enhanced - available),
        available,
    )


def _single_company_core_numeric_ready(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> bool:
    return _dimension_metric_satisfied(
        requirements,
        status_map,
        dimension_id="revenue_quality",
        metric="revenue",
        requirement_type="numeric",
    ) and _dimension_metric_satisfied(
        requirements,
        status_map,
        dimension_id="profitability_quality",
        metric="net_income",
        requirement_type="numeric",
    )


def _risk_focused_text_ready(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> bool:
    required_dimensions = {"moat_and_competitive_risk"}
    satisfied_dimensions: set[str] = set()
    for req in requirements:
        if str(req.get("requirement_type") or "") != "text":
            continue
        dimension_id = str(req.get("dimension_id") or "").strip()
        if dimension_id not in required_dimensions:
            continue
        rid = str(req.get("requirement_id") or "").strip()
        if str(status_map.get(rid, {}).get("status", "")) == "satisfied":
            satisfied_dimensions.add(dimension_id)
    return required_dimensions.issubset(satisfied_dimensions)


def _single_company_core_text_ready(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> bool:
    core_dimensions = {"business_model", "moat_and_competitive_risk"}
    for req in requirements:
        if str(req.get("requirement_type") or "") != "text":
            continue
        if str(req.get("dimension_id") or "") not in core_dimensions:
            continue
        rid = str(req.get("requirement_id") or "").strip()
        if str(status_map.get(rid, {}).get("status", "")) in {"satisfied", "partial"}:
            return True
    return False


def _explicit_required_dimensions(evidence_plan: Mapping[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            str(item).strip()
            for item in list(evidence_plan.get("required_dimensions", []) or [])
            if str(item).strip()
        )
    )


def _dimension_status_lookup(dimension_summary: Mapping[str, Any]) -> dict[str, str]:
    raw = dict(dimension_summary.get("dimension_status_by_id", dimension_summary.get("dimension_status_map", {})) or {})
    return {
        str(dimension_id): str(item.get("status", "missing"))
        for dimension_id, item in raw.items()
        if isinstance(item, Mapping)
    }


def _dimension_definitions() -> dict[str, dict[str, Any]]:
    return {dimension.id: dimension.__dict__ for dimension in get_fundamental_quality_analysis()}


def _label_metric(metric: str) -> str:
    labels = {
        "cash": "现金",
        "total_debt": "债务",
        "capital_expenditure": "资本开支",
        "total_assets": "总资产",
        "total_liabilities": "总负债",
        "shareholders_equity": "股东权益",
        "revenue_growth": "收入增长",
        "gross_profit": "毛利润",
        "operating_income": "营业利润",
        "gross_margin": "毛利率",
        "operating_margin": "营业利润率",
        "share_price": "价格",
        "market_cap": "市值",
        "pe_ratio": "P/E",
        "ps_ratio": "P/S",
        "fcf_yield": "FCF yield",
    }
    return labels.get(metric, metric)


def _dimension_limitation(
    dimension_id: str,
    status: str,
    *,
    required_available: list[str] | None = None,
    required_missing: list[str] | None = None,
    enhanced_missing: list[str] | None = None,
) -> str | None:
    required_available = list(required_available or [])
    required_missing = list(required_missing or [])
    enhanced_missing = list(enhanced_missing or [])
    if status == "satisfied":
        if not enhanced_missing:
            return None
        missing_text = "、".join(_label_metric(metric) for metric in enhanced_missing)
        if dimension_id == "balance_sheet_and_capital_intensity":
            return f"核心资产负债证据可用，但缺少{missing_text}，因此资本强度细分判断保留限制。"
        if dimension_id == "cash_flow_quality":
            return f"核心现金流证据可用，但缺少{missing_text}，因此现金流转化细分判断保留限制。"
        if dimension_id == "valuation_and_risk_boundary":
            return f"核心估值证据可用，但缺少{missing_text}，因此估值边界判断保留限制。"
        if dimension_id == "revenue_quality":
            return f"核心收入证据可用，但缺少{missing_text}，因此收入结构或趋势判断保留限制。"
        if dimension_id == "profitability_quality":
            return f"核心盈利证据可用，但缺少{missing_text}，因此盈利结构判断保留限制。"
        return f"核心证据可用，但缺少{missing_text}等增强指标，因此相关细分判断保留限制。"
    if dimension_id == "cash_flow_quality":
        if required_available:
            return "已有部分现金流证据，但缺少现金流转化率或自由现金流等增强指标，因此只能做有限现金流质量判断。"
        return "当前缺少经营现金流/自由现金流证据，不能判断利润现金含量。"
    if dimension_id == "balance_sheet_and_capital_intensity":
        if required_available:
            missing_text = "、".join(_label_metric(metric) for metric in (enhanced_missing or required_missing))
            if missing_text:
                return f"已有现金、债务和资本开支证据，但缺少{missing_text}，因此只能做有限资产负债判断。"
            return "已有部分资产负债和资本投入证据，因此只能做有限资产负债判断。"
        return "当前缺少现金、债务或资本开支证据，不能判断抗风险能力和资本投入强度。"
    if dimension_id == "valuation_and_risk_boundary":
        if required_available:
            return "已有部分估值证据，但缺少完整估值倍数，因此只能做估值边界观察，不能判断买卖或短期价格。"
        return "当前缺少估值证据，不能判断价格是否便宜或昂贵，也不能形成买卖建议。"
    if dimension_id == "revenue_quality" and status == "partial" and not required_missing:
        return "核心收入证据可用，但缺少更多收入结构/质量指标，因此只做有限收入质量判断。"
    if dimension_id == "moat_and_competitive_risk":
        return "当前缺少可验证风险文本证据，不能做具体竞争风险判断。"
    if dimension_id == "profitability_quality" and status == "partial":
        return "盈利能力只能基于已验证的净利润、收入或净利率证据做有限判断。"
    return "该分析维度证据不完整，只能做有限判断。"


def build_dimension_sufficiency(
    evidence_plan: Mapping[str, Any],
    status_map: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    requirements = _requirements(evidence_plan)
    by_dimension: dict[str, list[Mapping[str, Any]]] = {}
    all_by_dimension: dict[str, list[Mapping[str, Any]]] = {}
    dimension_names: dict[str, str] = {}
    for req in requirements:
        dimension_id = str(req.get("dimension_id") or "").strip()
        if not dimension_id:
            continue
        all_by_dimension.setdefault(dimension_id, []).append(req)
        dimension_names.setdefault(dimension_id, str(req.get("dimension_name") or dimension_id))
        if not bool(req.get("required", True)) or str(req.get("requirement_scope") or "core") != "core":
            continue
        by_dimension.setdefault(dimension_id, []).append(req)
    definitions = _dimension_definitions()
    dimension_status_map: dict[str, dict[str, Any]] = {}
    use_metric_rules = str(evidence_plan.get("analysis_scope") or "") == "single_company"
    for dimension_id, reqs in by_dimension.items():
        satisfied_ids: list[str] = []
        missing_ids: list[str] = []
        partial_signal = False
        for req in reqs:
            rid = str(req.get("requirement_id", "")).strip()
            status = str(status_map.get(rid, {}).get("status", "missing"))
            if status == "satisfied":
                satisfied_ids.append(rid)
            else:
                missing_ids.append(rid)
                if status == "partial":
                    partial_signal = True
        if satisfied_ids and not missing_ids:
            dimension_status = "satisfied"
        elif satisfied_ids or partial_signal:
            dimension_status = "partial"
        else:
            dimension_status = "missing"
        if use_metric_rules:
            dimension_status = _metric_rule_dimension_status(
                requirements=requirements,
                status_map=status_map,
                dimension_id=dimension_id,
                fallback_status=dimension_status,
            )
        required_available, required_missing, enhanced_available, enhanced_missing, metric_available = _dimension_metric_availability(
            dimension_id=dimension_id,
            reqs=all_by_dimension.get(dimension_id, reqs),
            status_map=status_map,
        )
        if use_metric_rules and dimension_id in DIMENSION_CORE_METRICS:
            if required_missing and required_available:
                dimension_status = "partial"
            elif required_missing and not required_available:
                dimension_status = "missing"
            elif partial_signal:
                dimension_status = "partial"
            elif required_available and not required_missing:
                dimension_status = "satisfied"
            elif metric_available and dimension_status == "missing":
                dimension_status = "partial"
        definition = definitions.get(dimension_id, {})
        allowed_claims = list(definition.get("allowed_claims", []) or [])
        forbidden_claims = list(definition.get("forbidden_claims", []) or [])
        if dimension_id == "cash_flow_quality" and dimension_status == "missing":
            forbidden_claims = list(dict.fromkeys([*forbidden_claims, "cash flow is strong", "cash flow is weak"]))
        if dimension_id == "valuation_and_risk_boundary" and dimension_status == "missing":
            forbidden_claims = list(dict.fromkeys([*forbidden_claims, "cheap", "expensive", "buy", "sell", "recommend"]))
        if dimension_id == "moat_and_competitive_risk" and dimension_status == "missing":
            forbidden_claims = list(dict.fromkeys([*forbidden_claims, "specific risk judgment"]))
        if dimension_id == "profitability_quality" and dimension_status == "partial":
            allowed_claims = ["based on net margin / net income evidence", *allowed_claims]
        limitation = _dimension_limitation(
            dimension_id,
            dimension_status,
            required_available=required_available,
            required_missing=required_missing,
            enhanced_missing=enhanced_missing,
        )
        if dimension_status == "partial" and not limitation and not missing_ids:
            limitation = "该分析维度核心证据可用，但增强指标不完整，因此只能做有限判断。"
        dimension_status_map[dimension_id] = asdict(
            DimensionSufficiency(
                dimension_id=dimension_id,
                status=dimension_status,  # type: ignore[arg-type]
                satisfied_requirements=sorted(satisfied_ids),
                missing_requirements=sorted(missing_ids),
                required_available=required_available or sorted(satisfied_ids),
                required_missing=required_missing,
                enhanced_available=enhanced_available,
                enhanced_missing=enhanced_missing,
                supporting_evidence_ids=sorted(satisfied_ids),
                allowed_claims=allowed_claims,
                forbidden_claims=forbidden_claims,
                limitation=limitation,
                limitations=(
                    [limitation]
                    if limitation
                    else []
                ),
            )
        )
        dimension_status_map[dimension_id]["dimension_name"] = dimension_names.get(dimension_id, dimension_id)

    for dimension_id, reqs in all_by_dimension.items():
        if dimension_id in dimension_status_map:
            continue
        satisfied_ids: list[str] = []
        partial_ids: list[str] = []
        for req in reqs:
            rid = str(req.get("requirement_id", "")).strip()
            status = str(status_map.get(rid, {}).get("status", "missing"))
            if status == "satisfied":
                satisfied_ids.append(rid)
            elif status == "partial":
                partial_ids.append(rid)
        if not satisfied_ids and not partial_ids:
            continue
        required_available, required_missing, enhanced_available, enhanced_missing, _metric_available = _dimension_metric_availability(
            dimension_id=dimension_id,
            reqs=reqs,
            status_map=status_map,
        )
        definition = definitions.get(dimension_id, {})
        optional_status = "satisfied" if satisfied_ids else "partial"
        limitation = _dimension_limitation(
            dimension_id,
            optional_status,
            required_available=required_available,
            required_missing=required_missing,
            enhanced_missing=enhanced_missing,
        )
        dimension_status_map[dimension_id] = asdict(
            DimensionSufficiency(
                dimension_id=dimension_id,
                status=optional_status,  # type: ignore[arg-type]
                satisfied_requirements=sorted(satisfied_ids),
                missing_requirements=[],
                required_available=required_available or sorted(satisfied_ids),
                required_missing=required_missing,
                enhanced_available=enhanced_available,
                enhanced_missing=enhanced_missing,
                supporting_evidence_ids=sorted([*satisfied_ids, *partial_ids]),
                allowed_claims=list(definition.get("allowed_claims", []) or []),
                forbidden_claims=list(definition.get("forbidden_claims", []) or []),
                limitation=limitation,
                limitations=([limitation] if limitation else []),
            )
        )
        dimension_status_map[dimension_id]["dimension_name"] = dimension_names.get(dimension_id, dimension_id)

    total = len(dimension_status_map)
    covered = sorted(
        dimension_id
        for dimension_id, item in dimension_status_map.items()
        if str(item.get("status", "")) == "satisfied"
    )
    missing = sorted(
        dimension_id
        for dimension_id, item in dimension_status_map.items()
        if str(item.get("status", "")) == "missing"
    )
    partial = [
        dimension_id
        for dimension_id, item in dimension_status_map.items()
        if str(item.get("status", "")) == "partial"
    ]
    if total <= 0:
        framework_status = "sufficient"
    elif len(covered) == total:
        framework_status = "sufficient"
    elif covered or partial:
        framework_status = "partial"
    else:
        framework_status = "insufficient"
    weighted_coverage = _safe_rate((len(covered) * 2) + len(partial), total * 2)
    return normalize_dimension_status_contract(
        dimension_status_map,
        satisfied_dimensions=covered,
        partial_dimensions=sorted(partial),
        missing_dimensions=missing,
        dimension_coverage_rate=_safe_rate(len(covered), total),
        weighted_dimension_coverage_rate=weighted_coverage,
        framework_sufficiency_status=framework_status,
    )


def _has_company_evidence_imbalance(
    requirements: list[Mapping[str, Any]],
    status_map: Mapping[str, Mapping[str, Any]],
) -> bool:
    for req_type in ("numeric", "text"):
        required_companies = _companies_for(requirements, status_map, req_type, required_only=True)
        if len(required_companies) < 2:
            continue
        satisfied_companies = _companies_for(
            requirements,
            status_map,
            req_type,
            required_only=True,
            satisfied_only=True,
        )
        if satisfied_companies and not required_companies.issubset(satisfied_companies):
            return True
    return False


def _validated_items_by_requirement(items: list[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        req_ids = [
            str(rid).strip()
            for rid in item.get("requirement_ids", []) or []
            if str(rid).strip()
        ]
        rid = str(item.get("requirement_id", "")).strip()
        if rid and rid not in req_ids:
            req_ids.append(rid)
        for req_id in req_ids:
            out.setdefault(req_id, []).append(dict(item))
    return out


def _validated_failure_reason(
    *,
    requirement: Mapping[str, Any],
    validated_count: int,
    raw_count: int,
    raw_items: list[Mapping[str, Any]] | None = None,
    raw_failure_reason: str | None,
    validation_failure_reason: str | None,
) -> str | None:
    min_results = max(int(requirement.get("min_results", 1) or 1), 1)
    if validated_count >= min_results:
        return None
    if validated_count > 0:
        return "below_min_results"

    req_type = str(requirement.get("requirement_type", ""))
    if validation_failure_reason:
        return validation_failure_reason
    if req_type == "calculation" and raw_failure_reason == "dependency_numeric_requirement_missing":
        return raw_failure_reason
    if req_type == "text" and raw_count > 0:
        return "no_validated_text_evidence"
    if req_type in {"numeric", "calculation", "event"} and raw_count > 0:
        return _specific_numeric_validation_failure(requirement, list(raw_items or []), raw_failure_reason)
    return raw_failure_reason or "no_matching_evidence"


def build_validated_collection_results(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
    *,
    validated_numeric_evidence: list[Mapping[str, Any]] | None = None,
    validated_text_evidence: list[Mapping[str, Any]] | None = None,
    validation_failure_reasons: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Project final validated evidence bundles back onto requirement results."""
    requirements = _requirements(evidence_plan)
    raw_by_req = _results_by_requirement(collection_results)
    numeric_by_req = _validated_items_by_requirement(list(validated_numeric_evidence or []))
    text_by_req = _validated_items_by_requirement(list(validated_text_evidence or []))
    failure_overrides = {str(k): str(v) for k, v in dict(validation_failure_reasons or {}).items() if str(k).strip() and str(v).strip()}

    final_results: list[dict[str, Any]] = []
    for req in requirements:
        rid = str(req.get("requirement_id", "")).strip()
        if not rid:
            continue
        req_type = str(req.get("requirement_type", ""))
        if req_type == "text":
            items = list(text_by_req.get(rid, []))
            evidence_type = "text"
        else:
            items = list(numeric_by_req.get(rid, []))
            evidence_type = req_type or "numeric"
        min_results = max(int(req.get("min_results", 1) or 1), 1)
        if len(items) >= min_results:
            status = "satisfied"
        elif items:
            status = "partial"
        else:
            status = "missing"
        raw_results = raw_by_req.get(rid, [])
        raw_items = [
            item
            for result in raw_results
            for item in result.get("items", []) or []
            if isinstance(item, Mapping)
        ]
        raw_hit_count = _result_stat(raw_results, "raw_hit_count")
        section_filtered_hit_count = _result_stat(raw_results, "section_filtered_hit_count")
        usable_hit_count = _result_stat(raw_results, "usable_hit_count")
        snippet_support_passed_count = _result_stat(raw_results, "snippet_support_passed_count")
        if usable_hit_count > 0:
            raw_hit_count = max(raw_hit_count, usable_hit_count)
            section_filtered_hit_count = max(section_filtered_hit_count, usable_hit_count)
            snippet_support_passed_count = max(snippet_support_passed_count, usable_hit_count)
        failure_reason = _validated_failure_reason(
            requirement=req,
            validated_count=len(items),
            raw_count=len(raw_items),
            raw_items=raw_items,
            raw_failure_reason=_requirement_failure_reason(raw_results),
            validation_failure_reason=failure_overrides.get(rid),
        )
        raw_detail = dict(raw_results[0]) if raw_results else {}
        raw_drop_stage = str(raw_detail.get("drop_stage", "") or "")
        computed_drop_stage = _drop_stage(
            raw_hit_count=raw_hit_count,
            section_filtered_hit_count=section_filtered_hit_count,
            usable_hit_count=usable_hit_count,
            snippet_support_passed_count=snippet_support_passed_count,
            validated_text_claim_count=0,
            text_citation_kept_count=0,
            final_validated_text_count=len(items) if req_type == "text" else 0,
            failure_reason=failure_reason,
        ) if req_type == "text" else None
        if (
            req_type == "text"
            and len(items) <= 0
            and usable_hit_count <= 0
            and raw_drop_stage in {"no_raw_hits", "section_filter_dropped", "quality_filter_dropped", "snippet_support_failed"}
        ):
            computed_drop_stage = raw_drop_stage
        final_results.append(
            collection_result(
                requirement_id=rid,
                status=status,
                evidence_type=evidence_type,
                items=items,
                failure_reason=failure_reason,
                retry_count=_requirement_retry_count(raw_results),
                framework_id=str(req.get("framework_id") or ""),
                dimension_id=str(req.get("dimension_id") or ""),
                dimension_name=str(req.get("dimension_name") or ""),
                analysis_purpose=str(req.get("analysis_purpose") or ""),
                raw_hit_count=raw_hit_count,
                section_filtered_hit_count=section_filtered_hit_count,
                usable_hit_count=usable_hit_count,
                snippet_support_passed_count=snippet_support_passed_count,
                text_claim_validated_count=0,
                final_validated_text_count=len(items) if req_type == "text" else 0,
                company=str(raw_detail.get("company") or req.get("company") or ""),
                retrieval_query=str(raw_detail.get("retrieval_query") or req.get("retrieval_query") or ""),
                section_preferences=list(raw_detail.get("section_preferences") or req.get("section_preferences") or []),
                fallback_queries=list(raw_detail.get("fallback_queries") or req.get("broadened_queries") or []),
                fallback_sections=list(raw_detail.get("fallback_sections") or req.get("fallback_sections") or []),
                top_raw_snippets=list(raw_detail.get("top_raw_snippets", []) or []),
                top_rejected_snippets=list(raw_detail.get("top_rejected_snippets", []) or []),
                rejection_reasons=dict(raw_detail.get("rejection_reasons", {}) or {}),
                drop_stage=computed_drop_stage,
            )
        )
    return final_results


def build_requirement_status_map(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
    sufficiency: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build a canonical requirement status map from the sufficiency truth source."""
    requirements = _requirements(evidence_plan)
    rejected_requirements = _normalize_rejected_requirements(
        list(evidence_plan.get("rejected_requirements", []) or (sufficiency or {}).get("rejected_requirements", []) or [])
    )
    rejected_ids = _rejected_requirement_ids(rejected_requirements)
    by_req = _results_by_requirement(collection_results)
    suff = dict(sufficiency or {})
    satisfied = {str(x).strip() for x in suff.get("satisfied_requirements", []) or [] if str(x).strip()}
    partial = {str(x).strip() for x in suff.get("partial_requirements", []) or [] if str(x).strip()}
    missing = {str(x).strip() for x in suff.get("missing_requirements", []) or [] if str(x).strip()}

    status_map: dict[str, dict[str, Any]] = {}
    for req in requirements:
        rid = str(req.get("requirement_id", "")).strip()
        if not rid:
            continue
        status = "missing"
        if rid in satisfied:
            status = "satisfied"
        elif rid in partial:
            status = "partial"
        elif rid in missing:
            status = "missing"
        elif rid in rejected_ids:
            status = "rejected"
        req_results = by_req.get(rid, [])
        failure_reason = _requirement_failure_reason(req_results)
        if not failure_reason:
            if status == "missing":
                failure_reason = "no_matching_evidence"
            elif status == "partial":
                failure_reason = "below_min_results"
            elif status == "rejected":
                failure_reason = "requirement_rejected"
        items: list[Mapping[str, Any]] = []
        for result in req_results:
            items.extend([x for x in result.get("items", []) or [] if isinstance(x, Mapping)])
        status_map[rid] = {
            "requirement_id": rid,
            "requirement_type": str(req.get("requirement_type", "")),
            "company": str(req.get("company") or ""),
            "purpose": str(req.get("purpose") or ""),
            "metric": str(req.get("metric") or ""),
            "metrics": list(req.get("metrics", []) or []),
            "framework_id": str(req.get("framework_id") or ""),
            "dimension_id": str(req.get("dimension_id") or ""),
            "dimension_name": str(req.get("dimension_name") or ""),
            "analysis_purpose": str(req.get("analysis_purpose") or ""),
            "required": bool(req.get("required", True)),
            "requirement_scope": str(req.get("requirement_scope") or ("core" if bool(req.get("required", True)) else "optional_context")),
            "status": status,
            "failure_reason": failure_reason,
            "quality_status": next(
                (str(result.get("quality_status")) for result in req_results if str(result.get("quality_status", "")).strip()),
                "valid" if status == "satisfied" else (failure_reason or "missing"),
            ),
            "retry_count": _requirement_retry_count(req_results),
            "item_count": len(items),
            "raw_hit_count": _result_stat(req_results, "raw_hit_count"),
            "section_filtered_hit_count": _result_stat(req_results, "section_filtered_hit_count"),
            "usable_hit_count": _result_stat(req_results, "usable_hit_count"),
            "snippet_support_passed_count": _result_stat(req_results, "snippet_support_passed_count"),
            "validated_text_claim_count": _result_stat(req_results, "validated_text_claim_count"),
            "text_claim_validated_count": _result_stat(req_results, "text_claim_validated_count")
            or _result_stat(req_results, "validated_text_claim_count"),
            "text_citation_kept_count": _result_stat(req_results, "text_citation_kept_count"),
            "final_validated_text_count": _result_stat(req_results, "final_validated_text_count"),
            "drop_stage": next(
                (str(result.get("drop_stage")) for result in req_results if str(result.get("drop_stage", "")).strip()),
                None,
            ),
            "retrieval_query": next(
                (str(result.get("retrieval_query")) for result in req_results if str(result.get("retrieval_query", "")).strip()),
                str(req.get("retrieval_query") or ""),
            ),
            "section_preferences": next(
                (list(result.get("section_preferences") or []) for result in req_results if result.get("section_preferences")),
                list(req.get("section_preferences") or []),
            ),
            "fallback_queries": next(
                (list(result.get("fallback_queries") or []) for result in req_results if result.get("fallback_queries")),
                list(req.get("broadened_queries") or []),
            ),
            "fallback_sections": next(
                (list(result.get("fallback_sections") or []) for result in req_results if result.get("fallback_sections")),
                list(req.get("fallback_sections") or []),
            ),
            "top_raw_snippets": next(
                (list(result.get("top_raw_snippets") or []) for result in req_results if result.get("top_raw_snippets")),
                [],
            ),
            "top_rejected_snippets": next(
                (list(result.get("top_rejected_snippets") or []) for result in req_results if result.get("top_rejected_snippets")),
                [],
            ),
            "rejection_reasons": next(
                (dict(result.get("rejection_reasons") or {}) for result in req_results if result.get("rejection_reasons")),
                {},
            ),
            "answer_part_ids": list(req.get("answer_part_ids", []) or []),
            "evidence_request_id": str(req.get("evidence_request_id") or ""),
            "evidence_role": str(req.get("evidence_role") or ""),
            "alternative_group": str(req.get("alternative_group") or ""),
            "items_preview": [_preview_item(item) for item in items[:3]],
            "requirement": dict(req),
        }

    for rejected in rejected_requirements:
        rid = str(rejected.get("requirement_id", "")).strip()
        if not rid or rid in status_map:
            continue
        status_map[rid] = {
            "requirement_id": rid,
            "requirement_type": str(rejected.get("requirement_type", "") or rejected.get("type", "")),
            "company": str(rejected.get("company") or ""),
            "purpose": str(rejected.get("purpose") or ""),
            "framework_id": str(rejected.get("framework_id") or ""),
            "dimension_id": str(rejected.get("dimension_id") or ""),
            "dimension_name": str(rejected.get("dimension_name") or ""),
            "analysis_purpose": str(rejected.get("analysis_purpose") or ""),
            "required": bool(rejected.get("required", True)),
            "requirement_scope": str(rejected.get("requirement_scope") or ("core" if bool(rejected.get("required", True)) else "optional_context")),
            "status": "rejected",
            "failure_reason": str(rejected.get("reason") or "requirement_rejected"),
            "retry_count": 0,
            "item_count": 0,
            "raw_hit_count": 0,
            "section_filtered_hit_count": 0,
            "usable_hit_count": 0,
            "snippet_support_passed_count": 0,
            "validated_text_claim_count": 0,
            "text_claim_validated_count": 0,
            "text_citation_kept_count": 0,
            "final_validated_text_count": 0,
            "drop_stage": "final_bundle_dropped",
            "retrieval_query": str(rejected.get("retrieval_query") or ""),
            "section_preferences": list(rejected.get("section_preferences") or []),
            "fallback_queries": list(rejected.get("broadened_queries") or []),
            "fallback_sections": list(rejected.get("fallback_sections") or []),
            "top_raw_snippets": [],
            "top_rejected_snippets": [],
            "rejection_reasons": {},
            "items_preview": [],
            "requirement": dict(rejected),
        }

    return status_map


def finalize_evidence_accounting(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
    *,
    validated_numeric_evidence: list[Mapping[str, Any]] | None = None,
    validated_text_evidence: list[Mapping[str, Any]] | None = None,
    validation_failure_reasons: Mapping[str, str] | None = None,
    synthesis_mode: str = "",
) -> dict[str, Any]:
    """Build the final canonical requirement ledger from validated evidence only."""
    final_collection_results = build_validated_collection_results(
        evidence_plan,
        collection_results,
        validated_numeric_evidence=validated_numeric_evidence,
        validated_text_evidence=validated_text_evidence,
        validation_failure_reasons=validation_failure_reasons,
    )
    final_sufficiency = evaluate_evidence_sufficiency(
        evidence_plan,
        final_collection_results,
    ).model_dump(exclude_none=True)
    final_summary = summarize_evidence_requirements(
        evidence_plan,
        final_collection_results,
        final_sufficiency,
    )
    dimension_contract = normalize_dimension_status_contract(
        dict(final_summary.get("dimension_status_by_id", final_summary.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(final_summary.get("satisfied_dimensions", final_summary.get("covered_dimensions", [])) or []),
        partial_dimensions=list(final_summary.get("partial_dimensions", []) or []),
        missing_dimensions=list(final_summary.get("missing_dimensions", []) or []),
        dimension_coverage_rate=final_summary.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=final_summary.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(final_summary.get("framework_sufficiency_status", "") or ""),
        missing_required_requirements_count=int(final_summary.get("missing_required_requirements_count", 0) or 0),
        missing_optional_requirements_count=int(final_summary.get("missing_optional_requirements_count", 0) or 0),
        missing_enhanced_requirements_count=int(final_summary.get("missing_enhanced_requirements_count", 0) or 0),
    )
    return {
        "evidence_collection_results": final_collection_results,
        "evidence_sufficiency": final_sufficiency,
        "evidence_sufficiency_summary": final_summary,
        "requirement_status_map": final_summary.get("requirement_status_map", {}),
        "dimension_status_by_id": dimension_contract["dimension_status_by_id"],
        "dimension_status_map": dimension_contract["dimension_status_map"],
        "satisfied_dimensions": dimension_contract["satisfied_dimensions"],
        "covered_dimensions": dimension_contract["covered_dimensions"],
        "partial_dimensions": dimension_contract["partial_dimensions"],
        "missing_dimensions": dimension_contract["missing_dimensions"],
        "dimension_coverage_rate": dimension_contract["dimension_coverage_rate"],
        "weighted_dimension_coverage_rate": dimension_contract["weighted_dimension_coverage_rate"],
        "framework_sufficiency_status": dimension_contract["framework_sufficiency_status"],
        "requirement_limitations": list(final_summary.get("requirement_limitations", []) or []),
        "missing_requirements": list(final_summary.get("missing_requirements", []) or []),
        "missing_required_requirements": list(final_summary.get("missing_required_requirements", []) or []),
        "missing_optional_requirements": list(final_summary.get("missing_optional_requirements", []) or []),
        "missing_enhanced_requirements": list(final_summary.get("missing_enhanced_requirements", []) or []),
        "missing_required_requirements_count": int(final_summary.get("missing_required_requirements_count", 0) or 0),
        "missing_optional_requirements_count": int(final_summary.get("missing_optional_requirements_count", 0) or 0),
        "missing_enhanced_requirements_count": int(final_summary.get("missing_enhanced_requirements_count", 0) or 0),
        "answer_part_status_by_id": dict(final_summary.get("answer_part_status_by_id", {}) or {}),
        "evidence_gap_by_answer_part": dict(final_summary.get("evidence_gap_by_answer_part", {}) or {}),
        "missing_required_answer_parts": list(final_summary.get("missing_required_answer_parts", []) or []),
        "partial_required_answer_parts": list(final_summary.get("partial_required_answer_parts", []) or []),
        "missing_but_analyzable_answer_parts": list(final_summary.get("missing_but_analyzable_answer_parts", []) or []),
        "missing_and_unanswerable_answer_parts": list(final_summary.get("missing_and_unanswerable_answer_parts", []) or []),
        "evidence_health": str(final_summary.get("evidence_health") or "complete"),
        "tool_error_context": list(final_summary.get("tool_error_context", []) or []),
        "degradation_reason": final_summary.get("degradation_reason"),
        "trace_summary": build_trace_summary(
            evidence_plan,
            final_collection_results,
            final_sufficiency,
            synthesis_mode=synthesis_mode,
        ),
        "validated_requirement_ids": sorted(
            {
                str(req_id).strip()
                for item in list(validated_numeric_evidence or []) + list(validated_text_evidence or [])
                for req_id in list(item.get("requirement_ids", []) or []) + [item.get("requirement_id", "")]
                if str(req_id).strip()
            }
        ),
        "validated_numeric_evidence_count": len(list(validated_numeric_evidence or [])),
        "validated_text_evidence_count": len(list(validated_text_evidence or [])),
    }


def _degradation_limitation(
    degradation_reason: str,
    *,
    company_evidence_balance: float | None,
) -> dict[str, Any] | None:
    messages = {
        "required_evidence_missing": ("high", "Required evidence is missing, so the answer cannot be fully grounded."),
        "rejected_required_requirement": ("high", "A required evidence requirement was rejected during validation."),
        "comparison_numeric_evidence_missing": ("high", "Comparable numeric evidence is insufficient for the requested comparison."),
        "numeric_only_comparison": (
            "medium",
            "Required filing text evidence is incomplete, so only a limited judgment based on structured financial evidence is allowed.",
        ),
        "text_evidence_missing": (
            "medium",
            "Required filing text evidence is missing, so open-ended analysis cannot be grounded.",
        ),
        "text_evidence_partial": (
            "medium",
            "Only partial validated filing text evidence is available, so only a limited text-grounded analysis is allowed.",
        ),
        "limited_outlook": (
            "medium",
            "Required filing text evidence is missing, so only a cautious numeric-only outlook is allowed.",
        ),
        "numeric_trend_evidence_missing": ("high", "Required numeric trend evidence is missing."),
    }
    if degradation_reason == "numeric_only_comparison" and company_evidence_balance is not None and company_evidence_balance < 1.0:
        code = "numeric_only_comparison"
    else:
        code = degradation_reason
    if code not in messages:
        return None
    severity, message = messages[code]
    return {"code": code, "severity": severity, "message": message}


def _build_requirement_limitations(
    *,
    evidence_plan: Mapping[str, Any],
    status_map: Mapping[str, Mapping[str, Any]],
    overall_status: str,
    degradation_reason: str | None,
    company_evidence_balance: float | None,
) -> list[dict[str, Any]]:
    limitations: list[dict[str, Any]] = []
    if overall_status not in {"sufficient", "focused_sufficient"}:
        for item in status_map.values():
            status = str(item.get("status", ""))
            if status not in {"missing", "partial", "rejected"}:
                continue
            reason = str(item.get("failure_reason") or "evidence_requirement_not_satisfied")
            limitations.append(
                {
                    "code": f"requirement_{status}",
                    "severity": "medium" if status == "partial" else "high",
                    "message": f"Evidence requirement {item.get('requirement_id')} was {status}: {reason}.",
                    "requirement_id": item.get("requirement_id", ""),
                    "status": status,
                    "failure_reason": reason,
                }
            )

    if degradation_reason:
        degradation_item = _degradation_limitation(
            degradation_reason,
            company_evidence_balance=company_evidence_balance,
        )
        if degradation_item is not None:
            limitations.append(degradation_item)

    task_type = str(evidence_plan.get("task_type", ""))
    answer_mode = str(evidence_plan.get("answer_mode", ""))
    if (
        (task_type == "company_comparison" or answer_mode == "comparison_brief")
        and _has_company_evidence_imbalance(_requirements(evidence_plan), status_map)
    ):
        limitations.append(
            {
                "code": "imbalanced_company_evidence",
                "severity": "medium",
                "message": "Evidence coverage is imbalanced across compared companies.",
            }
        )

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in limitations:
        key = (str(item.get("code", "")), str(item.get("requirement_id", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def summarize_evidence_requirements(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
    sufficiency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize requirement planning and canonical sufficiency state."""
    requirements = _requirements(evidence_plan)
    status_map = build_requirement_status_map(evidence_plan, collection_results, sufficiency)
    required = [
        req
        for req in requirements
        if bool(req.get("required", True)) and str(req.get("requirement_scope") or "core") == "core"
    ]
    required_numeric = [req for req in required if str(req.get("requirement_type", "")) == "numeric"]
    required_text = [req for req in required if str(req.get("requirement_type", "")) == "text"]
    rejected = [
        item
        for item in status_map.values()
        if str(item.get("status", "")) == "rejected"
    ]
    satisfied = sorted([rid for rid, item in status_map.items() if str(item.get("status", "")) == "satisfied"])
    partial = sorted([rid for rid, item in status_map.items() if str(item.get("status", "")) == "partial"])
    missing = sorted([rid for rid, item in status_map.items() if str(item.get("status", "")) == "missing"])
    missing_required = sorted(
        [
            rid
            for rid, item in status_map.items()
            if str(item.get("status", "")) == "missing"
            and bool(item.get("required", True))
            and str(item.get("requirement_scope") or "core") == "core"
        ]
    )
    missing_optional = sorted(
        [
            rid
            for rid, item in status_map.items()
            if str(item.get("status", "")) == "missing"
            and (
                not bool(item.get("required", True))
                or str(item.get("requirement_scope") or "core") != "core"
            )
        ]
    )
    suff = dict(sufficiency or {})
    required_numeric_satisfied = sum(
        1
        for req in required_numeric
        if str(status_map.get(str(req.get("requirement_id", "")).strip(), {}).get("status", "")) == "satisfied"
    )
    required_text_satisfied = sum(
        1
        for req in required_text
        if str(status_map.get(str(req.get("requirement_id", "")).strip(), {}).get("status", "")) == "satisfied"
    )
    company_evidence_balance = suff.get("company_evidence_balance")
    requirement_limitations = list(suff.get("requirement_limitations", []) or [])
    if not requirement_limitations:
        requirement_limitations = _build_requirement_limitations(
            evidence_plan=evidence_plan,
            status_map=status_map,
            overall_status=str(suff.get("overall_status", "")),
            degradation_reason=suff.get("degradation_reason"),
            company_evidence_balance=company_evidence_balance,
        )
    dimension_summary = build_dimension_sufficiency(evidence_plan, status_map)
    missing_enhanced_metrics = sorted(
        {
            str(metric)
            for item in dict(dimension_summary.get("dimension_status_by_id", dimension_summary.get("dimension_status_map", {})) or {}).values()
            if isinstance(item, Mapping)
            for metric in list(item.get("enhanced_missing", []) or [])
            if str(metric).strip()
        }
    )
    dimension_summary = normalize_dimension_status_contract(
        dict(dimension_summary.get("dimension_status_by_id", dimension_summary.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(dimension_summary.get("satisfied_dimensions", dimension_summary.get("covered_dimensions", [])) or []),
        partial_dimensions=list(dimension_summary.get("partial_dimensions", []) or []),
        missing_dimensions=list(dimension_summary.get("missing_dimensions", []) or []),
        dimension_coverage_rate=dimension_summary.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=dimension_summary.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(dimension_summary.get("framework_sufficiency_status", "") or ""),
        missing_required_requirements_count=len(missing_required),
        missing_optional_requirements_count=len(missing_optional),
        missing_enhanced_requirements_count=len(missing_enhanced_metrics),
    )
    answer_part_summary = build_answer_part_status(evidence_plan, status_map)
    tool_context = list(suff.get("tool_error_context", []) or []) or _tool_error_context(collection_results)
    evidence_health = str(suff.get("evidence_health") or _evidence_health(
        overall_status=str(suff.get("overall_status", "")),
        answer_part_summary=answer_part_summary,
        tool_context=tool_context,
    ))

    return {
        "requirement_count": len(status_map),
        "required_count": len(required),
        "satisfied_count": len(satisfied),
        "partial_count": len(partial),
        "missing_count": len(missing),
        "missing_required_requirements_count": len(missing_required),
        "missing_optional_requirements_count": len(missing_optional),
        "missing_enhanced_requirements_count": len(missing_enhanced_metrics),
        "rejected_count": len(rejected),
        "required_numeric_count": len(required_numeric),
        "required_text_count": len(required_text),
        "satisfied_required_numeric_count": required_numeric_satisfied,
        "satisfied_required_text_count": required_text_satisfied,
        "required_numeric_satisfied_rate": suff.get(
            "required_numeric_satisfied_rate",
            _safe_rate(required_numeric_satisfied, len(required_numeric)),
        ),
        "required_text_satisfied_rate": suff.get(
            "required_text_satisfied_rate",
            _safe_rate(required_text_satisfied, len(required_text)),
        ),
        "company_evidence_balance": company_evidence_balance,
        "satisfied_requirements": satisfied,
        "partial_requirements": partial,
        "missing_requirements": missing,
        "missing_required_requirements": missing_required,
        "missing_optional_requirements": missing_optional,
        "missing_enhanced_requirements": missing_enhanced_metrics,
        "rejected_requirements": [item.get("requirement") or item for item in rejected],
        "overall_status": str(suff.get("overall_status", "")),
        "degradation_reason": suff.get("degradation_reason"),
        "can_synthesize": bool(suff.get("can_synthesize", False)),
        "evidence_health": evidence_health,
        "tool_error_context": tool_context,
        "requirement_limitations": requirement_limitations,
        "collected_evidence_by_requirement": status_map,
        "requirement_status_map": status_map,
        **answer_part_summary,
        **dimension_summary,
    }


def build_trace_summary(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
    sufficiency: Mapping[str, Any] | None = None,
    *,
    synthesis_mode: str = "",
) -> dict[str, Any]:
    """Return a reviewer-friendly canonical trace summary."""
    summary = summarize_evidence_requirements(evidence_plan, collection_results, sufficiency)
    dimension_contract = normalize_dimension_status_contract(
        dict(summary.get("dimension_status_by_id", summary.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(summary.get("satisfied_dimensions", summary.get("covered_dimensions", [])) or []),
        partial_dimensions=list(summary.get("partial_dimensions", []) or []),
        missing_dimensions=list(summary.get("missing_dimensions", []) or []),
        dimension_coverage_rate=summary.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=summary.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(summary.get("framework_sufficiency_status", "") or ""),
        missing_required_requirements_count=int(summary.get("missing_required_requirements_count", 0) or 0),
        missing_optional_requirements_count=int(summary.get("missing_optional_requirements_count", 0) or 0),
        missing_enhanced_requirements_count=int(summary.get("missing_enhanced_requirements_count", 0) or 0),
    )
    return {
        "sufficiency_status": str(summary.get("overall_status", "")),
        "missing_requirements_count": dimension_contract["missing_required_requirements_count"],
        "total_missing_requirements_count": int(summary.get("missing_count", 0) or 0),
        "missing_required_requirements_count": dimension_contract["missing_required_requirements_count"],
        "missing_optional_requirements_count": dimension_contract["missing_optional_requirements_count"],
        "missing_enhanced_requirements_count": dimension_contract["missing_enhanced_requirements_count"],
        "limitations_count": len(list(summary.get("requirement_limitations", []) or [])),
        "required_numeric_satisfied_rate": summary.get("required_numeric_satisfied_rate"),
        "required_text_satisfied_rate": summary.get("required_text_satisfied_rate"),
        "company_evidence_balance": summary.get("company_evidence_balance"),
        "degradation_reason": summary.get("degradation_reason"),
        "dimension_coverage_rate": dimension_contract["dimension_coverage_rate"],
        "weighted_dimension_coverage_rate": dimension_contract["weighted_dimension_coverage_rate"],
        "framework_sufficiency_status": dimension_contract["framework_sufficiency_status"],
        "dimension_status_by_id": dimension_contract["dimension_status_by_id"],
        "dimension_status_map": dimension_contract["dimension_status_map"],
        "satisfied_dimensions": dimension_contract["satisfied_dimensions"],
        "covered_dimensions": dimension_contract["covered_dimensions"],
        "partial_dimensions": dimension_contract["partial_dimensions"],
        "missing_dimensions": dimension_contract["missing_dimensions"],
        "final_synthesis_mode": synthesis_mode,
        "answer_part_status_by_id": dict(summary.get("answer_part_status_by_id", {}) or {}),
        "evidence_gap_by_answer_part": dict(summary.get("evidence_gap_by_answer_part", {}) or {}),
        "missing_required_answer_parts": list(summary.get("missing_required_answer_parts", []) or []),
        "partial_required_answer_parts": list(summary.get("partial_required_answer_parts", []) or []),
        "missing_but_analyzable_answer_parts": list(summary.get("missing_but_analyzable_answer_parts", []) or []),
        "missing_and_unanswerable_answer_parts": list(summary.get("missing_and_unanswerable_answer_parts", []) or []),
        "answer_parts_fully_satisfied": bool(summary.get("answer_parts_fully_satisfied", True)),
        "answer_parts_clean_pass": bool(summary.get("answer_parts_clean_pass", True)),
        "evidence_health": str(summary.get("evidence_health") or "complete"),
        "tool_error_context": list(summary.get("tool_error_context", []) or []),
    }


def evaluate_evidence_sufficiency(
    evidence_plan: Mapping[str, Any],
    collection_results: list[Mapping[str, Any]],
) -> EvidenceSufficiencyResult:
    """Decide whether collected evidence satisfies a validated EvidencePlan."""
    requirements = _requirements(evidence_plan)
    rejected = _normalize_rejected_requirements(list(evidence_plan.get("rejected_requirements", []) or []))
    by_req = _results_by_requirement(collection_results)

    if not requirements and not rejected:
        return EvidenceSufficiencyResult(
            overall_status="sufficient",
            can_synthesize=False,
            requirement_limitations=[],
            required_numeric_satisfied_rate=1.0,
            required_text_satisfied_rate=1.0,
        )

    satisfied: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    for req in requirements:
        rid = str(req.get("requirement_id", "")).strip()
        if not rid:
            continue
        status = _result_status(by_req.get(rid, []))
        if status == "satisfied":
            satisfied.append(rid)
        elif status == "partial":
            partial.append(rid)
        else:
            missing.append(rid)

    rejected_ids = _rejected_requirement_ids(rejected)
    for rid in list(missing):
        if rid in rejected_ids:
            missing.remove(rid)

    required_ids = _required_ids(requirements)
    required_missing = [
        rid
        for rid in required_ids
        if rid not in satisfied and rid not in partial and rid not in rejected_ids
    ]
    required_partial = [rid for rid in required_ids if rid in partial]
    required_rejected = [
        item
        for item in rejected
        if bool(item.get("required", True)) and str(item.get("requirement_scope") or "core") == "core"
    ]
    required_satisfied = all(
        rid in satisfied for rid in required_ids if rid not in rejected_ids
    ) and not required_missing and not required_partial and not required_rejected
    any_required_signal = any(
        rid in satisfied or rid in partial for rid in required_ids if rid not in rejected_ids
    )

    task_type = str(evidence_plan.get("task_type", ""))
    answer_mode = str(evidence_plan.get("answer_mode", ""))
    safety_intent = str(evidence_plan.get("safety_intent", ""))
    analysis_scope = str(evidence_plan.get("analysis_scope", ""))
    methodology_intent = str(evidence_plan.get("methodology_intent", ""))

    overall = "sufficient" if required_satisfied else ("partial" if any_required_signal else "insufficient")
    reason: str | None = None if required_satisfied else "required_evidence_missing"
    can_synthesize = overall in {"sufficient", "partial"} and bool(satisfied)

    status_map = build_requirement_status_map(
        evidence_plan,
        collection_results,
        {
            "satisfied_requirements": satisfied,
            "partial_requirements": partial,
            "missing_requirements": missing,
            "rejected_requirements": rejected,
        },
    )

    required_numeric_ids = _ids_of_type(requirements, "numeric", required_only=True)
    required_text_ids = _ids_of_type(requirements, "text", required_only=True)
    any_numeric_satisfied = any(
        str(status_map.get(rid, {}).get("status", "")) == "satisfied"
        for rid in _ids_of_type(requirements, "numeric", required_only=False)
    )
    all_required_text_satisfied = all(
        str(status_map.get(rid, {}).get("status", "")) == "satisfied" for rid in required_text_ids
    )
    dimension_summary = build_dimension_sufficiency(evidence_plan, status_map)
    dimension_statuses = _dimension_status_lookup(dimension_summary)

    is_comparison = task_type == "company_comparison" or answer_mode == "comparison_brief"
    if is_comparison:
        required_numeric_companies = _companies_for(
            requirements,
            status_map,
            "numeric",
            required_only=True,
        )
        satisfied_numeric_companies = _companies_for(
            requirements,
            status_map,
            "numeric",
            required_only=True,
            satisfied_only=True,
        )
        if len(required_numeric_companies) >= 2 and not required_numeric_companies.issubset(satisfied_numeric_companies):
            overall = "partial" if satisfied_numeric_companies else "insufficient"
            reason = "comparison_numeric_evidence_missing"
            can_synthesize = bool(satisfied_numeric_companies)
        elif answer_mode == "comparison_brief" and safety_intent == "investment_advice_like":
            required_text_companies = _companies_for(
                requirements,
                status_map,
                "text",
                required_only=True,
            )
            satisfied_text_companies = _companies_for(
                requirements,
                status_map,
                "text",
                required_only=True,
                satisfied_only=True,
            )
            if required_text_companies and not required_text_companies.issubset(satisfied_text_companies):
                overall = "partial"
                reason = "numeric_only_comparison"
                can_synthesize = bool(satisfied_numeric_companies)

    if answer_mode == "cautious_outlook":
        if not any_numeric_satisfied:
            overall = "insufficient"
            reason = "numeric_trend_evidence_missing"
            can_synthesize = False
        elif not all_required_text_satisfied:
            overall = "partial"
            reason = "limited_outlook"
            can_synthesize = True

    is_risk_focused = answer_mode == "risk_focused_analysis" and analysis_scope == "single_company"
    if is_risk_focused:
        if _risk_focused_text_ready(requirements, status_map):
            overall = "focused_sufficient"
            reason = None
            can_synthesize = True
        else:
            any_required_text_signal = any(
                str(status_map.get(rid, {}).get("status", "")) in {"satisfied", "partial"}
                for rid in required_text_ids
            )
            overall = "partial" if any_required_text_signal else "insufficient"
            reason = "risk_text_evidence_partial" if any_required_text_signal else "risk_text_evidence_missing"
            can_synthesize = bool(any_required_text_signal)

    is_single_company_methodology = analysis_scope == "single_company" and not is_risk_focused
    if is_single_company_methodology:
        core_numeric_ready = _single_company_core_numeric_ready(requirements, status_map)
        core_text_ready = _single_company_core_text_ready(requirements, status_map)
        explicit_required_dimensions = _explicit_required_dimensions(evidence_plan)
        dimension_specific_intents = {
            "revenue_quality_analysis",
            "profitability_quality_analysis",
            "cash_flow_quality_analysis",
            "balance_sheet_analysis",
            "valuation_boundary_analysis",
        }
        if explicit_required_dimensions:
            required_dimension_statuses = [
                dimension_statuses.get(dimension_id, "missing")
                for dimension_id in explicit_required_dimensions
            ]
            if required_satisfied and all(status == "satisfied" for status in required_dimension_statuses):
                overall = "sufficient"
                reason = None
                can_synthesize = True
            elif any(status in {"satisfied", "partial"} for status in required_dimension_statuses) or any_required_signal:
                overall = "partial"
                reason = (
                    "dimension_evidence_partial"
                    if all(status != "missing" for status in required_dimension_statuses)
                    else "dimension_evidence_missing"
                )
                can_synthesize = bool(satisfied or partial)
            else:
                overall = "insufficient"
                reason = "dimension_evidence_missing"
                can_synthesize = False
        elif methodology_intent == "valuation_boundary_analysis":
            overall = "partial"
            reason = "valuation_evidence_missing"
            can_synthesize = True
        elif methodology_intent in dimension_specific_intents:
            if any_required_signal:
                overall = "partial" if not required_satisfied else "sufficient"
                reason = None if required_satisfied else "dimension_evidence_partial"
                can_synthesize = True
            else:
                overall = "insufficient"
                reason = "dimension_evidence_missing"
                can_synthesize = False
        elif core_numeric_ready:
            if not required_satisfied:
                overall = "partial"
                reason = "single_company_methodology_partial" if core_text_ready else "single_company_text_evidence_missing"
            can_synthesize = True
            if any(
                str(status_map.get(rid, {}).get("failure_reason", "")) == "valuation_evidence_missing"
                for rid in required_missing
            ) or any(
                str(req.get("requirement_id") or "") in required_missing
                and (
                    str(req.get("dimension_id") or "") == "valuation_and_risk_boundary"
                    or "valuation_evidence_missing" in {str(item) for item in req.get("fallback_strategy", []) or []}
                )
                for req in requirements
            ):
                reason = "valuation_evidence_missing"
        else:
            overall = "insufficient"
            reason = "core_numeric_evidence_missing"
            can_synthesize = False

    if not is_single_company_methodology and (answer_mode == "analytical" or task_type == "report_summary"):
        any_required_text_signal = any(
            str(status_map.get(rid, {}).get("status", "")) in {"satisfied", "partial"}
            for rid in required_text_ids
        )
        if required_text_ids and not all_required_text_satisfied:
            if any_required_text_signal:
                overall = "partial"
                can_synthesize = True
                reason = "text_evidence_partial"
            else:
                overall = "insufficient"
                can_synthesize = False
                reason = "text_evidence_missing"

    if required_rejected and not required_satisfied:
        reason = "rejected_required_requirement"

    required_numeric_satisfied_rate = _safe_rate(
        sum(1 for rid in required_numeric_ids if str(status_map.get(rid, {}).get("status", "")) == "satisfied"),
        len(required_numeric_ids),
    )
    required_text_satisfied_rate = _safe_rate(
        sum(1 for rid in required_text_ids if str(status_map.get(rid, {}).get("status", "")) == "satisfied"),
        len(required_text_ids),
    )
    company_evidence_balance = _company_evidence_balance(requirements, status_map)
    answer_part_summary = build_answer_part_status(evidence_plan, status_map)
    partial_answer_parts = list(answer_part_summary.get("partial_required_answer_parts", []) or [])
    missing_answer_parts = list(answer_part_summary.get("missing_required_answer_parts", []) or [])
    missing_but_analyzable_parts = list(answer_part_summary.get("missing_but_analyzable_answer_parts", []) or [])
    missing_unanswerable_parts = list(answer_part_summary.get("missing_and_unanswerable_answer_parts", []) or [])
    if missing_but_analyzable_parts and overall == "insufficient":
        overall = "partial"
        reason = "answer_part_missing_but_analyzable"
        can_synthesize = True
    elif (missing_answer_parts or partial_answer_parts or missing_but_analyzable_parts) and overall == "sufficient":
        overall = "partial"
        reason = (
            "answer_part_evidence_missing"
            if missing_answer_parts or missing_unanswerable_parts
            else ("answer_part_missing_but_analyzable" if missing_but_analyzable_parts else "answer_part_evidence_partial")
        )
        can_synthesize = bool(satisfied or partial or missing_but_analyzable_parts)
    limitations = _build_requirement_limitations(
        evidence_plan=evidence_plan,
        status_map=status_map,
        overall_status=overall,
        degradation_reason=reason,
        company_evidence_balance=company_evidence_balance,
    )
    missing_required = sorted(
        rid
        for rid, item in status_map.items()
        if str(item.get("status") or "") == "missing"
        and bool(item.get("required", True))
        and str(item.get("requirement_scope") or "core") == "core"
    )
    missing_optional = sorted(
        rid
        for rid, item in status_map.items()
        if str(item.get("status") or "") == "missing"
        and (
            not bool(item.get("required", True))
            or str(item.get("requirement_scope") or "core") != "core"
        )
    )
    dimension_contract = normalize_dimension_status_contract(
        dict(dimension_summary.get("dimension_status_by_id", dimension_summary.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(dimension_summary.get("satisfied_dimensions", dimension_summary.get("covered_dimensions", [])) or []),
        partial_dimensions=list(dimension_summary.get("partial_dimensions", []) or []),
        missing_dimensions=list(dimension_summary.get("missing_dimensions", []) or []),
        dimension_coverage_rate=dimension_summary.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=dimension_summary.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(dimension_summary.get("framework_sufficiency_status", "") or ""),
    )
    tool_context = _tool_error_context(collection_results)
    health = _evidence_health(
        overall_status=overall,
        answer_part_summary=answer_part_summary,
        tool_context=tool_context,
    )

    return EvidenceSufficiencyResult(
        overall_status=overall,
        satisfied_requirements=sorted(satisfied),
        partial_requirements=sorted(partial),
        missing_requirements=sorted(missing),
        rejected_requirements=rejected,
        degradation_reason=reason,
        can_synthesize=can_synthesize,
        requirement_limitations=limitations,
        required_numeric_satisfied_rate=required_numeric_satisfied_rate,
        required_text_satisfied_rate=required_text_satisfied_rate,
        missing_required_requirements=missing_required,
        missing_optional_requirements=missing_optional,
        missing_required_requirements_count=len(missing_required),
        missing_optional_requirements_count=len(missing_optional),
        company_evidence_balance=company_evidence_balance,
        dimension_status_by_id=dict(dimension_contract.get("dimension_status_by_id", {}) or {}),
        dimension_status_map=dict(dimension_contract.get("dimension_status_map", {}) or {}),
        satisfied_dimensions=list(dimension_contract.get("satisfied_dimensions", []) or []),
        covered_dimensions=list(dimension_contract.get("covered_dimensions", []) or []),
        partial_dimensions=list(dimension_contract.get("partial_dimensions", []) or []),
        missing_dimensions=list(dimension_contract.get("missing_dimensions", []) or []),
        dimension_coverage_rate=float(dimension_contract.get("dimension_coverage_rate", 1.0) or 0.0),
        weighted_dimension_coverage_rate=float(dimension_contract.get("weighted_dimension_coverage_rate", 1.0) or 0.0),
        framework_sufficiency_status=str(dimension_contract.get("framework_sufficiency_status", "sufficient")),
        answer_part_status_by_id=dict(answer_part_summary.get("answer_part_status_by_id", {}) or {}),
        evidence_gap_by_answer_part=dict(answer_part_summary.get("evidence_gap_by_answer_part", {}) or {}),
        missing_required_answer_parts=missing_answer_parts,
        partial_required_answer_parts=partial_answer_parts,
        missing_but_analyzable_answer_parts=missing_but_analyzable_parts,
        missing_and_unanswerable_answer_parts=missing_unanswerable_parts,
        answer_parts_fully_satisfied=not missing_answer_parts and not partial_answer_parts and not missing_but_analyzable_parts,
        answer_parts_clean_pass=not missing_answer_parts and not partial_answer_parts and not missing_but_analyzable_parts,
        evidence_health=health,
        tool_error_context=tool_context,
    )
