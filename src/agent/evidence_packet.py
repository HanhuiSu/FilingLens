"""Build a validated evidence packet for analysis drafting."""

from __future__ import annotations

import re
from typing import Any

from src.agent.analysis_framework import summarize_selected_analysis_framework
from src.agent.driver_evidence import apply_scope_aware_summary
from src.agent.evidence import _select_comparison_evidence_rows
from src.agent.evidence_sufficiency import normalize_dimension_status_contract
from src.agent.metric_availability import (
    DIMENSION_CORE_METRICS,
    DIMENSION_ENHANCED_METRICS,
    normalize_metric_name,
)
from src.agent.metric_display import format_metric_value, metric_display_name, period_category
from src.agent.red_flags import detect_red_flags, serialize_red_flags
from src.agent.types import EvidencePacket, EvidencePacketTheme

_RISK_THEME_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("demand_macro", "Demand / Macro Pressure", ("demand", "consumer", "macro", "softness", "weakness", "spending")),
    ("competition", "Competition", ("competition", "competitive", "pricing", "market share", "rival")),
    ("regulation_legal", "Regulation / Legal", ("regulation", "regulatory", "legal", "litigation", "compliance", "privacy", "tariff")),
    ("operations_supply_chain", "Operations / Supply Chain", ("supply chain", "manufacturing", "inventory", "logistics", "operations", "capacity")),
    ("execution_investment", "Execution / Investment", ("execution", "reinvestment", "investment", "margin pressure", "headwind", "operating leverage")),
)

_BUSINESS_THEME_RULES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("profitability_margin", "Profitability / Margin", ("margin", "profitability", "net income", "operating income", "operating margin")),
    ("revenue_mix_demand", "Revenue Mix / Demand", ("revenue", "demand", "sales", "mix", "customer", "consumption")),
    ("operating_efficiency", "Operating Efficiency", ("efficiency", "cost", "expense", "productivity", "discipline")),
    ("segment_growth", "Segment Growth", ("segment", "growth", "cloud", "services", "advertising", "device")),
    ("business_model_positioning", "Business Model Positioning", ("business model", "positioning", "platform", "ecosystem", "strategy", "leverage")),
)

_DIMENSION_EXPECTED_METRICS: dict[str, list[str]] = {
    dimension_id: list(dict.fromkeys([*DIMENSION_CORE_METRICS.get(dimension_id, ()), *DIMENSION_ENHANCED_METRICS.get(dimension_id, ())]))
    for dimension_id in sorted(set(DIMENSION_CORE_METRICS) | set(DIMENSION_ENHANCED_METRICS))
}

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _display_value(row: dict[str, Any]) -> str:
    value = row.get("value")
    metric = str(row.get("metric", ""))
    display = str(row.get("display_value", "")).strip()
    if value is not None:
        canonical = normalize_metric_name(metric)
        if display:
            if canonical in {"pe_ratio", "ps_ratio"} and "%" in display:
                return format_metric_value(metric, value, unit=str(row.get("unit", "")))
            if canonical in {"market_cap", "net_debt"} and "$" not in display:
                return format_metric_value(metric, value, unit=str(row.get("unit", "")))
            return display
        return format_metric_value(metric, value, unit=str(row.get("unit", "")))
    if display:
        return display
    return "N/A"


def _numeric_rows(
    numeric_evidence: list[dict[str, Any]],
    requirement_status_map: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    requirement_status_map = dict(requirement_status_map or {})
    rows: list[dict[str, Any]] = []
    for item in numeric_evidence:
        req_ids = _requirement_ids(item)
        source_requirement_id = str(item.get("source_requirement_id") or (req_ids[0] if req_ids else "")).strip()
        req_status = dict(requirement_status_map.get(source_requirement_id, {}) or {})
        role = str(item.get("role") or item.get("evidence_role") or req_status.get("evidence_role") or "").strip()
        quality_status = str(
            item.get("quality_status")
            or req_status.get("quality_status")
            or ("valid" if str(req_status.get("status") or "") == "satisfied" else req_status.get("failure_reason") or "")
            or "valid"
        )
        row = {
            "evidence_id": str(item.get("evidence_id", "")),
            "ticker": str(item.get("ticker", "")),
            "metric": str(item.get("metric", "")),
            "role": role,
            "evidence_role": role,
            "source_requirement_id": source_requirement_id,
            "requirement_ids": req_ids,
            "period_scope": str(item.get("period_scope") or item.get("period_type") or ""),
            "quality_status": quality_status,
            "display_label": metric_display_name(str(item.get("metric", "")), "zh"),
            "period_type": str(item.get("period_type", "")),
            "period_category": period_category(item.get("period_type")),
            "period_end": str(item.get("period_end", "")),
            "value": item.get("value"),
            "display_value": _display_value(item),
            "unit": str(item.get("unit", "")),
            "provenance": str(item.get("provenance", "")),
            "source_provider": str(item.get("source_provider", "")),
            "confidence": str(item.get("confidence", "")),
            "extraction_method": str(item.get("extraction_method", "")),
            "source_tag": str(item.get("source_tag", "")),
            "reconciliation_warning": str(item.get("reconciliation_warning", "")),
            "requirement_id": str(item.get("requirement_id", "")),
        }
        for trace_key in (
            "share_price",
            "price_date",
            "shares_outstanding",
            "shares_period",
            "market_cap",
            "market_cap_period",
            "statement_period",
            "revenue_period",
            "net_income_period",
            "free_cash_flow_period",
            "period_basis",
            "dependencies",
            "numerator_metric",
            "denominator_metric",
            "numerator_requirement_id",
            "denominator_requirement_id",
        ):
            if trace_key in item:
                row[trace_key] = item.get(trace_key)
        rows.append(row)
    return rows


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def _requirement_ids(item: dict[str, Any]) -> list[str]:
    req_ids = [
        str(req_id).strip()
        for req_id in item.get("requirement_ids", []) or []
        if str(req_id).strip()
    ]
    for key in ("requirement_id", "source_requirement_id"):
        rid = str(item.get(key, "")).strip()
        if rid and rid not in req_ids:
            req_ids.append(rid)
    return req_ids


def _dimension_names(selected_framework: dict[str, Any], dimension_status_map: dict[str, dict[str, Any]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for item in selected_framework.get("dimensions", []) or []:
        if not isinstance(item, dict):
            continue
        dimension_id = str(item.get("id") or "").strip()
        if dimension_id:
            names[dimension_id] = str(item.get("name") or dimension_id)
    for dimension_id, item in dimension_status_map.items():
        if isinstance(item, dict):
            names.setdefault(str(dimension_id), str(item.get("dimension_name") or dimension_id))
    return names


def _active_dimensions(selected_framework: dict[str, Any], dimension_status_map: dict[str, dict[str, Any]]) -> list[str]:
    active = _ordered_unique([str(x) for x in selected_framework.get("active_dimension_ids", []) or []])
    if active:
        return active
    return _ordered_unique([str(x) for x in dimension_status_map.keys()])


def _refs_by_dimension(
    rows: list[dict[str, Any]],
    requirement_status_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        if not ref:
            continue
        row_dimension = str(row.get("dimension_id") or "").strip()
        if row_dimension:
            refs = out.setdefault(row_dimension, [])
            if ref not in refs:
                refs.append(ref)
        for req_id in _requirement_ids(row):
            dimension_id = str(requirement_status_map.get(req_id, {}).get("dimension_id") or "").strip()
            if not dimension_id:
                continue
            refs = out.setdefault(dimension_id, [])
            if ref not in refs:
                refs.append(ref)
    return out


def _metrics_by_dimension(
    numeric_rows: list[dict[str, Any]],
    requirement_status_map: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in numeric_rows:
        metric = normalize_metric_name(str(row.get("metric") or "").strip())
        if not metric:
            continue
        for req_id in _requirement_ids(row):
            status = requirement_status_map.get(req_id, {})
            dimension_id = str(status.get("dimension_id") or "").strip()
            if dimension_id:
                out.setdefault(dimension_id, [])
                if metric not in out[dimension_id]:
                    out[dimension_id].append(metric)
    return out


def _missing_metrics_by_dimension(active_dimensions: list[str], available: dict[str, list[str]]) -> dict[str, list[str]]:
    missing: dict[str, list[str]] = {}
    for dimension_id in active_dimensions:
        expected = _DIMENSION_EXPECTED_METRICS.get(dimension_id, [])
        if not expected:
            continue
        present = set(available.get(dimension_id, []))
        missing[dimension_id] = [metric for metric in expected if metric not in present]
    return missing


def _computed_metric_dependencies(numeric_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in numeric_rows:
        if str(row.get("source_provider") or "") != "computed":
            continue
        metric = str(row.get("metric") or "").strip()
        if not metric:
            continue
        dependency = {
            "evidence_id": row.get("evidence_id"),
            "requirement_id": row.get("requirement_id"),
            "source_tag": row.get("source_tag"),
            "period_end": row.get("period_end"),
            "provenance": row.get("provenance"),
            "dependencies": row.get("dependencies", []),
            "share_price": row.get("share_price"),
            "price_date": row.get("price_date"),
            "shares_outstanding": row.get("shares_outstanding"),
            "shares_period": row.get("shares_period"),
            "market_cap": row.get("market_cap"),
            "market_cap_period": row.get("market_cap_period"),
            "statement_period": row.get("statement_period"),
            "revenue_period": row.get("revenue_period"),
            "net_income_period": row.get("net_income_period"),
            "free_cash_flow_period": row.get("free_cash_flow_period"),
            "source_provider": row.get("source_provider"),
            "confidence": row.get("confidence"),
            "reconciliation_warning": row.get("reconciliation_warning"),
        }
        out.setdefault(metric, []).append(dependency)
    return out


def _rows_by_dimension(
    rows: list[dict[str, Any]],
    requirement_status_map: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ref = str(row.get("evidence_id") or "").strip()
        row_dimension = str(row.get("dimension_id") or "").strip()
        if row_dimension:
            bucket = out.setdefault(row_dimension, [])
            if not any(str(existing.get("evidence_id") or "") == ref for existing in bucket):
                bucket.append(row)
        for req_id in _requirement_ids(row):
            dimension_id = str(requirement_status_map.get(req_id, {}).get("dimension_id") or "").strip()
            if not dimension_id:
                continue
            bucket = out.setdefault(dimension_id, [])
            if not any(str(existing.get("evidence_id") or "") == ref for existing in bucket):
                bucket.append(row)
    return out


def _flatten_claims(dimension_status_map: dict[str, dict[str, Any]], active_dimensions: list[str], key: str) -> list[str]:
    claims: list[str] = []
    for dimension_id in active_dimensions:
        item = dimension_status_map.get(dimension_id, {})
        if not isinstance(item, dict):
            continue
        for claim in item.get(key, []) or []:
            text = str(claim or "").strip()
            if text and text not in claims:
                claims.append(text)
    return claims


def _dimension_summary(
    *,
    active_dimensions: list[str],
    dimension_status_map: dict[str, dict[str, Any]],
    requirement_status_map: dict[str, dict[str, Any]],
    numeric_refs_by_dimension: dict[str, list[str]],
    text_refs_by_dimension: dict[str, list[str]],
    dimension_names: dict[str, str],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for dimension_id in active_dimensions:
        status_item = dict(dimension_status_map.get(dimension_id, {}) or {})
        req_items = [
            dict(item)
            for item in requirement_status_map.values()
            if str(item.get("dimension_id") or "") == dimension_id and bool(item.get("required", True))
        ]
        satisfied_requirements = list(status_item.get("satisfied_requirements", []) or [])
        missing_requirements = list(status_item.get("missing_requirements", []) or [])
        if not satisfied_requirements and req_items:
            satisfied_requirements = sorted(
                str(item.get("requirement_id") or "")
                for item in req_items
                if str(item.get("status") or "") == "satisfied"
            )
        if not missing_requirements and req_items:
            missing_requirements = sorted(
                str(item.get("requirement_id") or "")
                for item in req_items
                if str(item.get("status") or "") in {"missing", "partial", "rejected"}
            )
        numeric_refs = list(numeric_refs_by_dimension.get(dimension_id, []) or [])
        text_refs = list(text_refs_by_dimension.get(dimension_id, []) or [])
        evidence_refs = _ordered_unique(numeric_refs + text_refs)
        status = str(status_item.get("status") or ("missing" if missing_requirements else "satisfied"))
        summary.append(
            {
                "dimension_id": dimension_id,
                "name": str(status_item.get("dimension_name") or dimension_names.get(dimension_id) or dimension_id),
                "status": status,
                "limitation": status_item.get("limitation"),
                "satisfied_requirements": satisfied_requirements,
                "missing_requirements": missing_requirements,
                "numeric_evidence_refs": numeric_refs,
                "text_evidence_refs": text_refs,
                "evidence_refs": evidence_refs,
                "allowed_claims": list(status_item.get("allowed_claims", []) or []),
                "forbidden_claims": list(status_item.get("forbidden_claims", []) or []),
            }
        )
    return summary


def _dimension_limitations(dimension_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in dimension_summary:
        if str(item.get("status") or "") != "missing":
            continue
        message = str(item.get("limitation") or "").strip()
        if not message:
            continue
        out.append(
            {
                "code": f"missing_dimension_{item.get('dimension_id')}",
                "severity": "medium",
                "message": message,
            }
        )
    return out


def _dedupe_limitations(limitations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in limitations:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("code") or ""), str(item.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(dict(item))
    return out


def _comparison_rows(
    *,
    companies: list[str],
    comparison_target: str | None,
    requested_metrics: list[str],
    period_query: dict[str, Any] | None,
    resolved_period_context: dict[str, Any] | None,
    numeric_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len({x for x in companies if x}) + (1 if comparison_target else 0) < 2:
        return []
    selection = _select_comparison_evidence_rows(
        {
            "companies": companies,
            "comparison_target": comparison_target,
            "requested_metrics": requested_metrics,
            "period_query": period_query or {},
            "resolved_period_context": resolved_period_context or {},
        },
        numeric_rows,
    )
    table: list[dict[str, Any]] = []
    for pair in selection.get("comparable_pairs", []) or []:
        left = dict(pair.get("left", {}) or {})
        right = dict(pair.get("right", {}) or {})
        table.append(
            {
                "metric": str(pair.get("metric", "")),
                "period_type": str(pair.get("period_type", "")),
                "period_end": str(left.get("period_end", right.get("period_end", ""))),
                "left_ticker": str(left.get("ticker", "")),
                "left_value": left.get("value"),
                "left_display_value": _display_value(left),
                "right_ticker": str(right.get("ticker", "")),
                "right_value": right.get("value"),
                "right_display_value": _display_value(right),
                "evidence_refs": [str(left.get("evidence_id", "")), str(right.get("evidence_id", ""))],
            }
        )
    return table


def _text_rows(text_evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in text_evidence:
        scoped_item = apply_scope_aware_summary(item)
        rows.append(
            {
                "evidence_id": str(scoped_item.get("evidence_id", "")),
                "ticker": str(scoped_item.get("ticker", "")),
                "form_type": str(scoped_item.get("form_type", "")),
                "fiscal_period": str(scoped_item.get("fiscal_period", "")),
                "section": str(scoped_item.get("section", "")),
                "text_snippet": str(scoped_item.get("text_snippet", "")),
                "supporting_snippet": str(scoped_item.get("supporting_snippet", "")),
                "requirement_id": str(scoped_item.get("requirement_id", "")),
                "dimension_id": str(scoped_item.get("dimension_id", "")),
                "citation_ref": str(scoped_item.get("citation_ref", "") or scoped_item.get("evidence_id", "")),
                "claim": str(scoped_item.get("claim", "")),
                "claim_source": str(scoped_item.get("claim_source", "")),
                "original_claim": str(scoped_item.get("original_claim", "")),
                "driver_level": str(scoped_item.get("driver_level", "")),
                "driver_levels": list(scoped_item.get("driver_levels", []) or []),
                "claim_scope": str(scoped_item.get("claim_scope", "")),
                "allowed_claim_strength": str(scoped_item.get("allowed_claim_strength", "")),
                "scope_reason": str(scoped_item.get("scope_reason", "")),
                "evidence_summary_scope_overclaim": bool(scoped_item.get("evidence_summary_scope_overclaim", False)),
                "summary_scope_warning": str(scoped_item.get("summary_scope_warning", "")),
            }
        )
    return rows


def _rule_match(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _theme_bucket(
    snippets: list[dict[str, Any]],
    *,
    rules: tuple[tuple[str, str, tuple[str, ...]], ...],
    fallback_code: str,
    fallback_label: str,
    risk_mode: bool,
) -> list[EvidencePacketTheme]:
    buckets: dict[str, dict[str, Any]] = {}
    for item in snippets:
        section = str(item.get("section", "")).upper()
        combined_text = _normalize_text(
            f"{item.get('supporting_snippet', '')} {item.get('text_snippet', '')}"
        )
        matched = False
        for code, label, keywords in rules:
            section_match = risk_mode and section == "ITEM_1A"
            section_match = section_match or (not risk_mode and section in {"ITEM_7", "ITEM_1", "ITEM_2"})
            if not section_match and not _rule_match(combined_text, keywords):
                continue
            bucket = buckets.setdefault(
                code,
                {"theme_code": code, "label": label, "evidence_refs": [], "companies": set(), "snippet_count": 0},
            )
            eid = str(item.get("evidence_id", ""))
            if eid and eid not in bucket["evidence_refs"]:
                bucket["evidence_refs"].append(eid)
            company = str(item.get("ticker", ""))
            if company:
                bucket["companies"].add(company)
            bucket["snippet_count"] += 1
            matched = True
            break
        if matched:
            continue
        if risk_mode and section not in {"ITEM_1A", "ITEM_7", "ITEM_1", "ITEM_2"}:
            continue
        if not risk_mode and section not in {"ITEM_7", "ITEM_1", "ITEM_2", "ITEM_1A"}:
            continue
        bucket = buckets.setdefault(
            fallback_code,
            {
                "theme_code": fallback_code,
                "label": fallback_label,
                "evidence_refs": [],
                "companies": set(),
                "snippet_count": 0,
            },
        )
        eid = str(item.get("evidence_id", ""))
        if eid and eid not in bucket["evidence_refs"]:
            bucket["evidence_refs"].append(eid)
        company = str(item.get("ticker", ""))
        if company:
            bucket["companies"].add(company)
        bucket["snippet_count"] += 1
    out: list[EvidencePacketTheme] = []
    for item in buckets.values():
        out.append(
            EvidencePacketTheme(
                theme_code=str(item.get("theme_code", "")),
                label=str(item.get("label", "")),
                evidence_refs=list(item.get("evidence_refs", []) or []),
                companies=sorted(str(x) for x in item.get("companies", set()) if str(x).strip()),
                snippet_count=int(item.get("snippet_count", 0) or 0),
            )
        )
    return sorted(out, key=lambda theme: (-theme.snippet_count, theme.theme_code))


def _provenance_notes(
    numeric_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
) -> list[str]:
    notes: list[str] = []
    providers = sorted({str(row.get("source_provider", "")) for row in numeric_rows if str(row.get("source_provider", "")).strip()})
    if providers:
        notes.append(f"Validated structured metrics sourced from: {', '.join(providers)}.")
    sections = sorted({str(row.get("section", "")) for row in text_rows if str(row.get("section", "")).strip()})
    if sections:
        notes.append(f"Validated filing text snippets sourced from sections: {', '.join(sections)}.")
    companies = sorted({str(row.get("ticker", "")) for row in numeric_rows + text_rows if str(row.get("ticker", "")).strip()})
    if companies:
        notes.append(f"Validated packet covers companies: {', '.join(companies)}.")
    return notes


def build_evidence_packet(
    *,
    user_query: str,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    analysis_scope: str = "",
    time_policy: str = "",
    period_scope: str = "",
    companies: list[str],
    comparison_target: str | None,
    requested_metrics: list[str],
    period_query: dict[str, Any] | None,
    resolved_period_context: dict[str, Any] | None,
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    evidence_sufficiency: dict[str, Any],
    requirement_limitations: list[dict[str, Any]],
    safety_limitations: list[dict[str, Any]],
    selected_framework: dict[str, Any] | None = None,
    requirement_status_map: dict[str, dict[str, Any]] | None = None,
) -> EvidencePacket:
    requirement_status_map = dict(requirement_status_map or {})
    numeric_rows = _numeric_rows(numeric_evidence, requirement_status_map=requirement_status_map)
    text_rows = _text_rows(text_evidence)
    selected_framework = dict(selected_framework or {})
    dimension_contract = normalize_dimension_status_contract(
        dict(evidence_sufficiency.get("dimension_status_by_id", evidence_sufficiency.get("dimension_status_map", {})) or {}),
        satisfied_dimensions=list(evidence_sufficiency.get("satisfied_dimensions", evidence_sufficiency.get("covered_dimensions", [])) or []),
        partial_dimensions=list(evidence_sufficiency.get("partial_dimensions", []) or []),
        missing_dimensions=list(evidence_sufficiency.get("missing_dimensions", []) or []),
        dimension_coverage_rate=evidence_sufficiency.get("dimension_coverage_rate"),
        weighted_dimension_coverage_rate=evidence_sufficiency.get("weighted_dimension_coverage_rate"),
        framework_sufficiency_status=str(evidence_sufficiency.get("framework_sufficiency_status", "") or ""),
    )
    dimension_status_by_id = dict(dimension_contract["dimension_status_by_id"])
    dimension_status_map = dict(dimension_contract["dimension_status_map"])
    active_dimensions = _active_dimensions(selected_framework, dimension_status_map)
    dimension_names = _dimension_names(selected_framework, dimension_status_map)
    numeric_refs_by_dimension = _refs_by_dimension(numeric_rows, requirement_status_map)
    text_refs_by_dimension = _refs_by_dimension(text_rows, requirement_status_map)
    dimension_summary = _dimension_summary(
        active_dimensions=active_dimensions,
        dimension_status_map=dimension_status_map,
        requirement_status_map=requirement_status_map,
        numeric_refs_by_dimension=numeric_refs_by_dimension,
        text_refs_by_dimension=text_refs_by_dimension,
        dimension_names=dimension_names,
    )
    available_metrics_by_dimension = _metrics_by_dimension(numeric_rows, requirement_status_map)
    missing_metrics_by_dimension = _missing_metrics_by_dimension(active_dimensions, available_metrics_by_dimension)
    text_evidence_by_dimension_refs = {
        dimension_id: list(refs)
        for dimension_id, refs in text_refs_by_dimension.items()
    }
    all_limitations = _dedupe_limitations(
        list(requirement_limitations) + list(safety_limitations) + _dimension_limitations(dimension_summary)
    )
    risk_themes = _theme_bucket(
        text_rows,
        rules=_RISK_THEME_RULES,
        fallback_code="other_validated_risk",
        fallback_label="Other Validated Risk",
        risk_mode=True,
    )
    business_themes = _theme_bucket(
        text_rows,
        rules=_BUSINESS_THEME_RULES,
        fallback_code="other_validated_business",
        fallback_label="Other Validated Business",
        risk_mode=False,
    )
    packet_payload: dict[str, Any] = {
        "packet_kind": "canonical_validated_evidence_packet",
        "canonical_source": True,
        "user_query": user_query,
        "task_type": task_type,
        "answer_mode": answer_mode,
        "safety_intent": safety_intent,
        "analysis_scope": analysis_scope,
        "time_policy": time_policy,
        "period_scope": period_scope,
        "selected_framework": selected_framework,
        "active_dimensions": active_dimensions,
        "numeric_table": numeric_rows,
        "comparison_table": _comparison_rows(
            companies=companies,
            comparison_target=comparison_target,
            requested_metrics=requested_metrics,
            period_query=period_query,
            resolved_period_context=resolved_period_context,
            numeric_rows=numeric_rows,
        ),
        "text_snippets": text_rows,
        "numeric_evidence_by_dimension": _rows_by_dimension(numeric_rows, requirement_status_map),
        "text_evidence_by_dimension": _rows_by_dimension(text_rows, requirement_status_map),
        "available_metrics_by_dimension": available_metrics_by_dimension,
        "missing_metrics_by_dimension": missing_metrics_by_dimension,
        "text_evidence_by_dimension_refs": text_evidence_by_dimension_refs,
        "computed_metric_dependencies": _computed_metric_dependencies(numeric_rows),
        "valuation_period_basis": {
            str(row.get("metric")): str(row.get("period_type") or row.get("period_end") or "")
            for row in numeric_rows
            if str(row.get("metric") or "") in {"market_cap", "pe_ratio", "ps_ratio", "fcf_yield"}
        },
        "grouped_risk_themes": risk_themes,
        "grouped_business_themes": business_themes,
        "provenance_notes": _provenance_notes(numeric_rows, text_rows),
        "missing_evidence_summary": {
            "overall_status": str(evidence_sufficiency.get("overall_status", "")),
            "missing_requirements": list(evidence_sufficiency.get("missing_requirements", []) or []),
            "partial_requirements": list(evidence_sufficiency.get("partial_requirements", []) or []),
            "degradation_reason": evidence_sufficiency.get("degradation_reason"),
            "required_numeric_satisfied_rate": evidence_sufficiency.get("required_numeric_satisfied_rate"),
            "required_text_satisfied_rate": evidence_sufficiency.get("required_text_satisfied_rate"),
            "company_evidence_balance": evidence_sufficiency.get("company_evidence_balance"),
            "dimension_status_by_id": dimension_status_by_id,
            "dimension_status_map": dimension_status_map,
            "satisfied_dimensions": list(dimension_contract["satisfied_dimensions"]),
            "covered_dimensions": list(dimension_contract["covered_dimensions"]),
            "partial_dimensions": list(dimension_contract["partial_dimensions"]),
            "missing_dimensions": list(dimension_contract["missing_dimensions"]),
            "dimension_coverage_rate": dimension_contract["dimension_coverage_rate"],
            "weighted_dimension_coverage_rate": dimension_contract["weighted_dimension_coverage_rate"],
            "framework_sufficiency_status": dimension_contract["framework_sufficiency_status"],
        },
        "dimension_sufficiency": {
            "dimension_status_by_id": dimension_status_by_id,
            "dimension_status_map": dimension_status_map,
            "satisfied_dimensions": list(dimension_contract["satisfied_dimensions"]),
            "covered_dimensions": list(dimension_contract["covered_dimensions"]),
            "partial_dimensions": list(dimension_contract["partial_dimensions"]),
            "missing_dimensions": list(dimension_contract["missing_dimensions"]),
            "dimension_coverage_rate": dimension_contract["dimension_coverage_rate"],
            "weighted_dimension_coverage_rate": dimension_contract["weighted_dimension_coverage_rate"],
            "framework_sufficiency_status": dimension_contract["framework_sufficiency_status"],
        },
        "dimension_status_by_id": dimension_status_by_id,
        "dimension_status_map": dimension_status_map,
        "dimension_summary": dimension_summary,
        "evidence_status_by_dimension": {
            str(dimension_id): str(dict(item or {}).get("status") or "missing")
            for dimension_id, item in dimension_status_map.items()
        },
        "red_flags": [],
        "missing_evidence_flags": [],
        "allowed_claims": _flatten_claims(dimension_status_map, active_dimensions, "allowed_claims"),
        "forbidden_claims": _flatten_claims(dimension_status_map, active_dimensions, "forbidden_claims"),
        "limitations": all_limitations,
        "citations": list(citations),
    }
    red_flags = serialize_red_flags(detect_red_flags(packet_payload, dimension_status_map))
    packet_payload["red_flags"] = red_flags
    packet_payload["missing_evidence_flags"] = [
        dict(flag)
        for flag in red_flags
        if str(flag.get("category") or "") == "missing_evidence"
    ]
    if selected_framework:
        packet_payload["selected_framework_summary"] = summarize_selected_analysis_framework(selected_framework)
    packet = EvidencePacket(
        **packet_payload,
    )
    return packet


def summarize_evidence_packet(packet: dict[str, Any]) -> dict[str, Any]:
    packet = dict(packet or {})
    text_flow = packet.get("text_evidence_flow_summary")
    text_flow = dict(text_flow or {}) if isinstance(text_flow, dict) else {}
    summary = {
        "numeric_row_count": len(packet.get("numeric_table", []) or []),
        "comparison_row_count": len(packet.get("comparison_table", []) or []),
        "text_snippet_count": len(packet.get("text_snippets", []) or []),
        "packet_kind": str(packet.get("packet_kind") or ""),
        "canonical_source": bool(packet.get("canonical_source", False)),
        "risk_theme_codes": [str(item.get("theme_code", "")) for item in packet.get("grouped_risk_themes", []) or []],
        "business_theme_codes": [str(item.get("theme_code", "")) for item in packet.get("grouped_business_themes", []) or []],
        "citation_count": len(packet.get("citations", []) or []),
        "overall_status": str(((packet.get("missing_evidence_summary", {}) or {}).get("overall_status", ""))),
        "degradation_reason": (packet.get("missing_evidence_summary", {}) or {}).get("degradation_reason"),
        "active_dimension_count": len(packet.get("active_dimensions", []) or []),
        "dimension_summary_count": len(packet.get("dimension_summary", []) or []),
        "red_flag_count": len(packet.get("red_flags", []) or []),
    }
    if text_flow:
        summary.update(text_flow)
    else:
        summary.update(
            {
                "text_candidate_count": len(packet.get("text_snippets", []) or []),
                "text_pre_citation_validated_count": len(packet.get("text_snippets", []) or []),
                "text_citable_count": len(packet.get("text_snippets", []) or []),
                "text_final_packet_count": len(packet.get("text_snippets", []) or []),
                "text_drop_stage_counts": {},
            }
        )
    return summary
