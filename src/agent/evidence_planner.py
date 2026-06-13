"""Evidence-requirement planning and sufficiency checks."""

from __future__ import annotations

from typing import Any, Mapping

from config import settings
from src.agent.analysis_framework import (
    BALANCE_SHEET_AND_CAPITAL_INTENSITY,
    BUSINESS_MODEL,
    CASH_FLOW_QUALITY,
    FRAMEWORK_ID,
    MOAT_AND_COMPETITIVE_RISK,
    PROFITABILITY_QUALITY,
    REVENUE_QUALITY,
    VALUATION_AND_RISK_BOUNDARY,
    get_fundamental_quality_analysis,
)
from src.agent.constants import ALLOWED_ANALYSIS_METRICS, KNOWN_SEC_SECTIONS
from src.agent.types import (
    CoverageDecision,
    EvidencePlan,
    EvidenceRequirement,
    PlanExecutionStrategy,
    RequirementMergeSummary,
)
from src.agent.evidence_sufficiency import (
    collection_result as collection_result,
    evaluate_evidence_sufficiency as evaluate_evidence_sufficiency,
)


ALLOWED_REQUIREMENT_TYPES = {"numeric", "text", "event", "calculation"}
ALLOWED_REQUIREMENT_SCOPES = {"core", "optional_context", "diagnostic"}
ALLOWED_REQUIREMENT_PERIOD_TYPES = {"quarterly", "annual", "ttm", "latest", None, ""}
PLANNER_METRIC_ALIASES = {
    "revenue_numeric": "revenue",
    "latest_revenue": "revenue",
    "total_revenue": "revenue",
    "sales": "revenue",
    "net_sales": "revenue",
    "net_income_numeric": "net_income",
    "profit": "net_income",
    "earnings": "net_income",
    "aws_operating_income": "aws_operating_income",
    "aws_revenue": "aws_revenue",
    "consolidated_operating_income": "consolidated_operating_income",
    "segment_profit_contribution": "segment_profit_contribution",
    "operating_cash_flow": "operating_cash_flow",
    "cash_from_operations": "operating_cash_flow",
    "free_cash_flow": "free_cash_flow",
    "capital_expenditure": "capital_expenditure",
    "capital_expenditures": "capital_expenditure",
    "capex": "capital_expenditure",
    "cash": "cash_and_equivalents",
    "cash_equivalents": "cash_and_equivalents",
    "cash_and_cash_equivalents": "cash_and_equivalents",
    "debt": "total_debt",
    "total_debt": "total_debt",
    "shares": "shares_outstanding",
    "shares_outstanding": "shares_outstanding",
    "price": "adjusted_close",
    "share_price": "adjusted_close",
    "latest_price": "adjusted_close",
    "adjusted_close": "adjusted_close",
    "market_cap": "market_cap",
    "pe": "pe_ratio",
    "p_e": "pe_ratio",
    "pe_ratio": "pe_ratio",
    "ps": "ps_ratio",
    "p_s": "ps_ratio",
    "ps_ratio": "ps_ratio",
    "fcf_yield": "fcf_yield",
}
OVERVIEW_MINIMUM_DIMENSIONS = {
    BUSINESS_MODEL,
    REVENUE_QUALITY,
    PROFITABILITY_QUALITY,
    CASH_FLOW_QUALITY,
    BALANCE_SHEET_AND_CAPITAL_INTENSITY,
    MOAT_AND_COMPETITIVE_RISK,
    VALUATION_AND_RISK_BOUNDARY,
}
TEXT_INTENT_CONFIG: dict[str, dict[str, Any]] = {
    "biggest_problem": {
        "profile": "risk_summary",
        "risk_terms": "operating challenges demand weakness margin pressure competitive pressure",
        "mda_terms": "management discussion operating challenges demand weakness margin pressure",
        "comparison_terms": "operating challenges demand weakness margin pressure competitive pressure",
    },
    "business_pressure": {
        "profile": "risk_summary",
        "risk_terms": "operating challenges demand weakness margin pressure competitive pressure",
        "mda_terms": "management discussion operating challenges demand weakness execution headwinds",
        "comparison_terms": "operating challenges demand weakness margin pressure competitive pressure",
    },
    "major_risks": {
        "profile": "risk_summary",
        "risk_terms": "risk factors competition regulation supply chain demand uncertainty",
        "mda_terms": "management discussion operating challenges competition regulation demand uncertainty",
        "comparison_terms": "risk factors operating challenges competitive pressure",
    },
    "management_concern": {
        "profile": "summary",
        "risk_terms": "risk factors operating challenges demand softness execution headwinds",
        "mda_terms": "management discussion operating challenges demand softness execution headwinds",
        "comparison_terms": "management discussion operating challenges execution headwinds",
    },
    "comparison_risk": {
        "profile": "risk_summary",
        "risk_terms": "risk factors operating challenges competitive pressure",
        "mda_terms": "management discussion operating challenges competitive pressure demand weakness",
        "comparison_terms": "risk factors operating challenges competitive pressure",
    },
    "comparison_risk_context": {
        "profile": "risk_summary",
        "risk_terms": "competition risk factors",
        "mda_terms": "business risks competitive pressure",
        "comparison_terms": "competition risk factors",
    },
    "comparison_key_difference": {
        "profile": "summary",
        "risk_terms": "business model operating leverage margin profile demand drivers",
        "mda_terms": "management discussion operating leverage margin profile demand drivers",
        "comparison_terms": "business model operating leverage margin profile demand drivers",
    },
    "comparison_context": {
        "profile": "comparison_support",
        "risk_terms": "business context competitive position operating leverage",
        "mda_terms": "management discussion business context operating leverage",
        "comparison_terms": "business context competitive position operating leverage",
    },
}


def _target_tickers() -> set[str]:
    return {str(t).upper() for t in settings.target_tickers}


def _ordered_unique(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        clean = str(item or "").strip()
        if clean and clean not in out:
            out.append(clean)
    return out


def normalize_planner_metric(metric: str) -> str:
    """Normalize LLM planner metric aliases into supported tool metrics."""
    raw = str(metric or "").strip()
    if not raw:
        return ""
    key = raw.lower().strip().replace("-", "_").replace(" ", "_")
    normalized = PLANNER_METRIC_ALIASES.get(key, key)
    if normalized in ALLOWED_ANALYSIS_METRICS:
        return normalized
    return ""


def _normalize_planner_metrics(metrics: list[Any]) -> tuple[list[str], list[dict[str, Any]]]:
    out: list[str] = []
    rejected: list[dict[str, Any]] = []
    for item in metrics:
        raw = str(item or "").strip()
        if not raw:
            continue
        normalized = normalize_planner_metric(raw)
        if normalized:
            if normalized not in out:
                out.append(normalized)
        else:
            rejected.append({"type": "metric", "value": raw, "reason": "unsupported_planner_metric"})
    return out, rejected


def _req_id(prefix: str, company: str | None = None, suffix: str = "") -> str:
    parts = [prefix]
    if company:
        parts.append(str(company).upper())
    if suffix:
        parts.append(str(suffix).upper().replace(" ", "_"))
    return "REQ-" + "-".join(parts)


def _period_type_from_state(state: Mapping[str, Any]) -> str | None:
    period_query = dict(state.get("period_query", {}) or {})
    raw = str(period_query.get("period_type") or "")
    if raw == "trailing":
        return "ttm"
    if raw in {"quarterly", "annual", "latest"}:
        return raw
    target = str(dict(state.get("resolved_period_context", {}) or {}).get("target_period_type") or "")
    if target in {"quarterly", "annual"}:
        return target
    return "latest"


def _analysis_goal(user_query: str, analysis_plan: Mapping[str, Any]) -> str:
    return str(analysis_plan.get("user_intent") or user_query).strip()[:500]


def _base_req(
    *,
    requirement_id: str,
    requirement_type: str,
    company: str | None,
    purpose: str,
    required: bool = True,
    requirement_scope: str | None = None,
    min_results: int = 1,
    metric: str | None = None,
    metrics: list[str] | None = None,
    period_type: str | None = None,
    section_preferences: list[str] | None = None,
    retrieval_query: str | None = None,
    fallback_strategy: list[str] | None = None,
    framework_id: str | None = None,
    dimension_id: str | None = None,
    dimension_name: str | None = None,
    analysis_purpose: str | None = None,
    answer_part_ids: list[str] | None = None,
    evidence_request_id: str | None = None,
    evidence_role: str | None = None,
    alternative_group: str | None = None,
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "requirement_type": requirement_type,
        "company": company,
        "framework_id": framework_id,
        "dimension_id": dimension_id,
        "dimension_name": dimension_name,
        "analysis_purpose": analysis_purpose,
        "metric": metric,
        "metrics": list(metrics or ([] if metric is None else [metric])),
        "period_type": period_type,
        "period_end": None,
        "section_preferences": list(section_preferences or []),
        "retrieval_query": retrieval_query,
        "purpose": purpose,
        "required": required,
        "requirement_scope": requirement_scope or ("core" if required else "optional_context"),
        "min_results": min_results,
        "fallback_strategy": list(fallback_strategy or []),
        "answer_part_ids": list(answer_part_ids or []),
        "evidence_request_id": evidence_request_id,
        "evidence_role": str(evidence_role or ""),
        "alternative_group": str(alternative_group or ""),
    }


def validate_evidence_requirement(raw: Mapping[str, Any]) -> tuple[EvidenceRequirement | None, list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    requirement_id = str(raw.get("requirement_id") or "").strip()
    requirement_type = str(raw.get("requirement_type") or "").strip()
    company_raw = raw.get("company")
    company = str(company_raw).upper().strip() if company_raw else None
    metric = str(raw.get("metric") or "").strip()
    metrics = [str(x).strip() for x in raw.get("metrics", []) if str(x).strip()] if isinstance(raw.get("metrics"), list) else []
    period_type = raw.get("period_type")
    period_type = str(period_type).strip() if period_type is not None else None
    sections = [str(x).upper().strip() for x in raw.get("section_preferences", []) if str(x).strip()] if isinstance(raw.get("section_preferences"), list) else []
    retrieval_query = str(raw.get("retrieval_query") or "").strip() or None
    retrieval_intent = str(raw.get("retrieval_intent") or "").strip()
    retrieval_profile = str(raw.get("retrieval_profile") or "").strip()
    requirement_scope = str(raw.get("requirement_scope") or ("core" if bool(raw.get("required", True)) else "optional_context")).strip()
    if not bool(raw.get("required", True)) and requirement_scope == "core":
        requirement_scope = "optional_context"
    framework_id = str(raw.get("framework_id") or "").strip() or None
    dimension_id = str(raw.get("dimension_id") or "").strip() or None
    dimension_name = str(raw.get("dimension_name") or "").strip() or None
    analysis_purpose = str(raw.get("analysis_purpose") or "").strip() or None
    primary_sections_raw = [str(x).upper().strip() for x in raw.get("primary_sections", []) if str(x).strip()] if isinstance(raw.get("primary_sections"), list) else []
    fallback_sections_raw = [str(x).upper().strip() for x in raw.get("fallback_sections", []) if str(x).strip()] if isinstance(raw.get("fallback_sections"), list) else []
    broadened_queries = [
        str(x).strip()
        for x in raw.get("broadened_queries", [])
        if str(x).strip()
    ] if isinstance(raw.get("broadened_queries"), list) else []
    answer_part_ids = [
        str(x).strip()
        for x in raw.get("answer_part_ids", [])
        if str(x).strip()
    ] if isinstance(raw.get("answer_part_ids"), list) else []
    evidence_request_id = str(raw.get("evidence_request_id") or "").strip() or None
    evidence_role = str(raw.get("evidence_role") or "").strip()
    alternative_group = str(raw.get("alternative_group") or "").strip()
    segment_or_product_scope = str(raw.get("segment_or_product_scope") or "").strip()
    min_results_raw = raw.get("min_results", 1)

    if not requirement_id:
        rejected.append({"type": "requirement", "value": raw, "reason": "missing_requirement_id"})
    if requirement_type not in ALLOWED_REQUIREMENT_TYPES:
        rejected.append({"type": "requirement_type", "value": requirement_type, "reason": "requirement_type_not_allowed"})
    if requirement_scope not in ALLOWED_REQUIREMENT_SCOPES:
        rejected.append({"type": "requirement_scope", "value": requirement_scope, "reason": "requirement_scope_not_allowed"})
    if company and company not in _target_tickers():
        rejected.append({"type": "company", "value": company_raw, "reason": "unknown_or_unsupported_ticker"})
    if metric and metric not in ALLOWED_ANALYSIS_METRICS:
        rejected.append({"type": "metric", "value": metric, "reason": "metric_not_allowed"})
    for item in metrics:
        if item not in ALLOWED_ANALYSIS_METRICS:
            rejected.append({"type": "metric", "value": item, "reason": "metric_not_allowed"})
    if period_type not in ALLOWED_REQUIREMENT_PERIOD_TYPES:
        rejected.append({"type": "period_type", "value": period_type, "reason": "period_type_not_allowed"})
    valid_sections: list[str] = []
    for section in sections:
        if section not in KNOWN_SEC_SECTIONS:
            rejected.append({"type": "section", "value": section, "reason": "section_not_allowed"})
        elif section not in valid_sections:
            valid_sections.append(section)
    valid_primary_sections: list[str] = []
    for section in primary_sections_raw:
        if section not in KNOWN_SEC_SECTIONS:
            rejected.append({"type": "section", "value": section, "reason": "section_not_allowed"})
        elif section not in valid_primary_sections:
            valid_primary_sections.append(section)
    valid_fallback_sections: list[str] = []
    for section in fallback_sections_raw:
        if section not in KNOWN_SEC_SECTIONS:
            rejected.append({"type": "section", "value": section, "reason": "section_not_allowed"})
        elif section not in valid_fallback_sections and section not in valid_primary_sections:
            valid_fallback_sections.append(section)
    try:
        min_results = int(min_results_raw)
    except (TypeError, ValueError):
        min_results = 0
    if min_results < 1:
        rejected.append({"type": "min_results", "value": min_results_raw, "reason": "min_results_must_be_at_least_1"})
    if requirement_type == "text" and not retrieval_query:
        rejected.append({"type": "retrieval_query", "value": retrieval_query, "reason": "text_requirement_requires_retrieval_query"})

    if rejected:
        return None, rejected

    if not metrics and metric:
        metrics = [metric]
    return (
        EvidenceRequirement(
            requirement_id=requirement_id,
            requirement_type=requirement_type,
            company=company,
            framework_id=framework_id,
            dimension_id=dimension_id,
            dimension_name=dimension_name,
            analysis_purpose=analysis_purpose,
            metric=metric or None,
            metrics=_ordered_unique(metrics),
            period_type=period_type or None,
            period_end=str(raw.get("period_end") or "").strip() or None,
            section_preferences=valid_sections,
            retrieval_query=retrieval_query,
            purpose=str(raw.get("purpose") or "").strip(),
            required=bool(raw.get("required", True)),
            requirement_scope=requirement_scope,  # type: ignore[arg-type]
            min_results=min_results,
            fallback_strategy=[
                str(x).strip()
                for x in raw.get("fallback_strategy", [])
                if str(x).strip()
            ]
            if isinstance(raw.get("fallback_strategy"), list)
            else [],
            retrieval_intent=retrieval_intent,
            retrieval_profile=retrieval_profile,
            primary_sections=valid_primary_sections,
            fallback_sections=valid_fallback_sections,
            broadened_queries=broadened_queries,
            answer_part_ids=_ordered_unique(answer_part_ids),
            evidence_request_id=evidence_request_id,
            evidence_role=evidence_role,
            alternative_group=alternative_group,
            segment_or_product_scope=segment_or_product_scope,
        ),
        [],
    )


def _append_valid(
    raw_requirements: list[dict[str, Any]],
    raw: dict[str, Any],
) -> None:
    raw_requirements.append(raw)


def _requirement_scope(raw: Mapping[str, Any]) -> str:
    scope = str(raw.get("requirement_scope") or "").strip()
    if not bool(raw.get("required", True)) and scope == "core":
        return "optional_context"
    if scope in ALLOWED_REQUIREMENT_SCOPES:
        return scope
    return "core" if bool(raw.get("required", True)) else "optional_context"


def _apply_policy_requirement_scopes(
    raw_requirements: list[dict[str, Any]],
    evidence_policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    policy_id = str(evidence_policy.get("policy_id") or "").strip()
    core_tokens = {str(item) for item in evidence_policy.get("core_requirements", []) or [] if str(item)}
    optional_tokens = {str(item) for item in evidence_policy.get("optional_context_requirements", []) or [] if str(item)}
    out: list[dict[str, Any]] = []
    for raw in raw_requirements:
        req = dict(raw)
        rid = str(req.get("requirement_id") or "")
        dimension_id = str(req.get("dimension_id") or "")
        token_values = {
            rid,
            f"dimension:{dimension_id}" if dimension_id else "",
            str(req.get("retrieval_intent") or ""),
            str(req.get("metric") or ""),
        }
        if policy_id:
            req["evidence_policy_id"] = policy_id
        if (token_values & core_tokens) and bool(req.get("required", True)):
            req["requirement_scope"] = "core"
            req["required"] = True
        elif token_values & optional_tokens:
            req["requirement_scope"] = "optional_context"
            req["required"] = False
        else:
            req["requirement_scope"] = _requirement_scope(req)
        if req["requirement_scope"] != "core":
            req["required"] = False
        out.append(req)
    return out


def _numeric_metrics(state: Mapping[str, Any], analysis_plan: Mapping[str, Any], fallback: list[str]) -> list[str]:
    metrics = [
        str(x).strip()
        for x in analysis_plan.get("metric_requirements", []) or state.get("requested_metrics", []) or []
        if str(x).strip() in ALLOWED_ANALYSIS_METRICS
    ]
    return _ordered_unique(metrics or fallback)


def _companies(state: Mapping[str, Any], analysis_plan: Mapping[str, Any]) -> list[str]:
    values = list(analysis_plan.get("companies", []) or []) + list(state.get("companies", []) or [])
    target = state.get("comparison_target")
    if target:
        values.append(str(target))
    return _ordered_unique([str(x).upper() for x in values if str(x).strip()])


def _text_query(user_query: str, company: str | None, purpose: str) -> str:
    prefix = f"{company} " if company else ""
    return f"{prefix}{purpose}: {user_query}".strip()


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


def _ensure_profit_decline_history_requirements(
    raw_requirements: list[dict[str, Any]],
    *,
    user_query: str,
    companies: list[str],
    selected_framework: Mapping[str, Any],
) -> None:
    if not _is_profit_decline_query(user_query):
        return
    meta = _dimension_meta(selected_framework, PROFITABILITY_QUALITY)
    existing = {
        (
            str(req.get("company") or "").upper(),
            str(req.get("metric") or ""),
            str(req.get("period_type") or ""),
        )
        for req in raw_requirements
        if str(req.get("requirement_type") or "") == "numeric"
        and int(req.get("min_results") or 1) >= 2
    }
    for company in companies:
        for metric in ("net_income", "operating_income"):
            key = (str(company).upper(), metric, "ttm")
            if key in existing:
                continue
            req = _dimension_numeric_req(
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                metric=metric,
                meta=meta,
                period_type="ttm",
                required=False,
                min_results=2,
                suffix=f"profit_decline_{metric}_history",
            )
            req["purpose"] = f"Recent {metric} history to verify whether the user's profit-decline premise is supported."
            req["evidence_role"] = "profit_decline_premise_check"
            raw_requirements.append(req)


def _is_change_query(user_query: str) -> bool:
    query = str(user_query or "").lower()
    return any(term in query for term in ("变化", "变动", "趋势", "change", "trend"))


def _change_history_metric_specs(user_query: str, selected_framework: Mapping[str, Any]) -> list[tuple[str, str]]:
    query = str(user_query or "").lower()
    specs: list[tuple[str, str]] = []
    if any(term in query for term in ("毛利率", "gross margin")):
        specs.append((PROFITABILITY_QUALITY, "gross_margin"))
    if any(term in query for term in ("自由现金流", "现金流", "fcf", "cash flow")):
        specs.extend([(CASH_FLOW_QUALITY, "operating_cash_flow"), (CASH_FLOW_QUALITY, "free_cash_flow")])
    if any(term in query for term in ("收入", "营收", "revenue", "sales")):
        specs.append((REVENUE_QUALITY, "revenue"))
    if any(term in query for term in ("利润", "盈利", "profit", "income", "margin")):
        specs.extend([(PROFITABILITY_QUALITY, "net_income"), (PROFITABILITY_QUALITY, "operating_income")])
    if any(term in query for term in ("估值", "市值", "valuation", "market cap")):
        specs.append((VALUATION_AND_RISK_BOUNDARY, "market_cap"))
    if specs:
        return list(dict.fromkeys(specs))
    active = set(_active_dimension_ids(selected_framework))
    if CASH_FLOW_QUALITY in active:
        specs.extend([(CASH_FLOW_QUALITY, "operating_cash_flow"), (CASH_FLOW_QUALITY, "free_cash_flow")])
    if REVENUE_QUALITY in active:
        specs.append((REVENUE_QUALITY, "revenue"))
    if PROFITABILITY_QUALITY in active:
        specs.append((PROFITABILITY_QUALITY, "net_income"))
    return list(dict.fromkeys(specs or [(REVENUE_QUALITY, "revenue")]))


def _ensure_change_history_requirements(
    raw_requirements: list[dict[str, Any]],
    *,
    user_query: str,
    companies: list[str],
    selected_framework: Mapping[str, Any],
) -> None:
    if not _is_change_query(user_query):
        return
    existing = {
        (
            str(req.get("company") or "").upper(),
            str(req.get("metric") or ""),
            str(req.get("period_type") or ""),
        )
        for req in raw_requirements
        if str(req.get("requirement_type") or "") == "numeric"
        and int(req.get("min_results") or 1) >= 2
    }
    for company in companies:
        for dimension_id, metric in _change_history_metric_specs(user_query, selected_framework):
            key = (str(company).upper(), metric, "ttm")
            if key in existing:
                continue
            meta = _dimension_meta(selected_framework, dimension_id)
            req = _dimension_numeric_req(
                company=company,
                dimension_id=dimension_id,
                metric=metric,
                meta=meta,
                period_type="ttm",
                required=False,
                min_results=2,
                suffix=f"change_{metric}_history",
            )
            req["purpose"] = f"Recent {metric} history to answer a change/trend question with at least two periods."
            req["evidence_role"] = "change_history_check"
            raw_requirements.append(req)


def _intent_query(user_query: str, company: str | None, terms: str) -> str:
    parts = [str(company or "").strip().upper(), str(terms or "").strip(), str(user_query or "").strip()]
    return " ".join(part for part in parts if part).strip()


def _normalize_text_sections(primary: list[str], fallback: list[str] | None = None) -> tuple[list[str], list[str]]:
    primary_sections = _ordered_unique([section for section in primary if section in KNOWN_SEC_SECTIONS])
    fallback_sections = _ordered_unique(
        [section for section in list(fallback or []) if section in KNOWN_SEC_SECTIONS and section not in primary_sections]
    )
    return primary_sections, fallback_sections


def _text_requirement_intent(
    *,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    methodology_intent: str = "",
    analysis_scope: str = "",
    primary_dimension: str = "",
    required_dimensions: list[str] | None = None,
    active_dimensions: list[str] | None = None,
    intent_reasons: list[str] | None = None,
) -> str:
    is_comparison = task_type == "company_comparison" or answer_mode == "comparison_brief" or safety_intent == "investment_advice_like"
    reason_set = {str(item) for item in list(intent_reasons or []) if str(item)}
    required_set = {str(item) for item in list(required_dimensions or []) if str(item)}
    active_set = {str(item) for item in list(active_dimensions or []) if str(item)}
    if is_comparison:
        risk_only_focus = (
            primary_dimension == MOAT_AND_COMPETITIVE_RISK
            or (bool(active_set) and active_set <= {MOAT_AND_COMPETITIVE_RISK})
            or (bool(required_set) and required_set <= {MOAT_AND_COMPETITIVE_RISK})
        )
        if "comparison_risk_family" in reason_set or risk_only_focus:
            return "comparison_risk"
        if "comparison_difference_family" in reason_set:
            return "comparison_key_difference"
        return "comparison_context"

    if (
        methodology_intent == "risk_focused_analysis"
        or answer_mode == "risk_focused_analysis"
        or (analysis_scope == "single_company" and primary_dimension == MOAT_AND_COMPETITIVE_RISK)
    ):
        return "major_risks"
    if methodology_intent == "single_company_overview":
        return "biggest_problem"
    return "biggest_problem"


def _text_intent_config(intent: str) -> dict[str, Any]:
    return dict(TEXT_INTENT_CONFIG.get(intent, TEXT_INTENT_CONFIG["biggest_problem"]))


def _broadened_queries(user_query: str, company: str | None, terms: str, section_hint: str) -> list[str]:
    hints = {
        "ITEM_1A": "risk factors disclosed risks competition regulation demand uncertainty",
        "ITEM_7": "management discussion operating results operating challenges demand margin",
        "ITEM_1": "business overview products segments customers competition",
        "ITEM_2": "operating results recent quarter business pressure demand",
    }
    return _ordered_unique(
        [
            _intent_query(user_query, company, terms),
            _intent_query(user_query, company, hints.get(section_hint, terms)),
            _intent_query(user_query, company, f"{terms} filing discussion"),
        ]
    )


def _text_requirement(
    *,
    requirement_id: str,
    company: str,
    purpose: str,
    primary_sections: list[str],
    fallback_sections: list[str],
    retrieval_query: str,
    retrieval_intent: str,
    retrieval_profile: str,
    broadened_queries: list[str],
    required: bool = True,
    min_results: int = 1,
    framework_id: str | None = None,
    dimension_id: str | None = None,
    dimension_name: str | None = None,
    analysis_purpose: str | None = None,
) -> dict[str, Any]:
    raw = _base_req(
        requirement_id=requirement_id,
        requirement_type="text",
        company=company,
        section_preferences=primary_sections,
        retrieval_query=retrieval_query,
        purpose=purpose,
        required=required,
        min_results=min_results,
        fallback_strategy=[
            "strict_broadened_query",
            "relaxed_sections_intent_query",
            "generic_query",
        ],
        framework_id=framework_id,
        dimension_id=dimension_id,
        dimension_name=dimension_name,
        analysis_purpose=analysis_purpose,
    )
    raw["retrieval_intent"] = retrieval_intent
    raw["retrieval_profile"] = retrieval_profile
    raw["primary_sections"] = list(primary_sections)
    raw["fallback_sections"] = list(fallback_sections)
    raw["broadened_queries"] = list(broadened_queries)
    return raw


def _selected_framework(state: Mapping[str, Any]) -> dict[str, Any]:
    selected = state.get("selected_analysis_framework")
    return dict(selected or {}) if isinstance(selected, Mapping) else {}


def _active_dimension_ids(selected: Mapping[str, Any]) -> list[str]:
    return _ordered_unique([str(x) for x in selected.get("active_dimension_ids", []) or [] if str(x).strip()])


def _dimension_definitions() -> dict[str, dict[str, Any]]:
    return {dimension.id: dimension.__dict__ for dimension in get_fundamental_quality_analysis()}


def _dimension_meta(selected: Mapping[str, Any], dimension_id: str) -> dict[str, str]:
    dimensions = _dimension_definitions()
    for raw in selected.get("dimensions", []) or []:
        if isinstance(raw, Mapping) and str(raw.get("id", "")) == dimension_id:
            return {
                "framework_id": str(selected.get("framework_id") or FRAMEWORK_ID),
                "dimension_id": dimension_id,
                "dimension_name": str(raw.get("name") or dimensions.get(dimension_id, {}).get("name") or dimension_id),
                "analysis_purpose": str(raw.get("evidence_purpose") or dimensions.get(dimension_id, {}).get("evidence_purpose") or ""),
            }
    return {
        "framework_id": str(selected.get("framework_id") or FRAMEWORK_ID),
        "dimension_id": dimension_id,
        "dimension_name": str(dimensions.get(dimension_id, {}).get("name") or dimension_id),
        "analysis_purpose": str(dimensions.get(dimension_id, {}).get("evidence_purpose") or ""),
    }


def _dimension_req_id(company: str, dimension_id: str, suffix: str) -> str:
    return _req_id("METH", company, f"{dimension_id}_{suffix}")


def _dimension_numeric_req(
    *,
    company: str,
    dimension_id: str,
    metric: str,
    meta: Mapping[str, str],
    period_type: str | None,
    required: bool,
    min_results: int = 1,
    suffix: str | None = None,
) -> dict[str, Any]:
    return _base_req(
        requirement_id=_dimension_req_id(company, dimension_id, suffix or metric),
        requirement_type="numeric",
        company=company,
        metric=metric,
        metrics=[metric],
        period_type=period_type,
        purpose=f"{meta.get('dimension_name', dimension_id)} numeric evidence: {metric}.",
        required=required,
        min_results=min_results,
        fallback_strategy=["latest_period", "relax_period"] if required else ["skip_optional_numeric"],
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _dimension_calc_req(
    *,
    company: str,
    dimension_id: str,
    metric: str,
    meta: Mapping[str, str],
    period_type: str | None,
    required: bool,
    suffix: str | None = None,
) -> dict[str, Any]:
    return _base_req(
        requirement_id=_dimension_req_id(company, dimension_id, suffix or metric),
        requirement_type="calculation",
        company=company,
        metric=metric,
        metrics=[metric],
        period_type=period_type,
        purpose=f"{meta.get('dimension_name', dimension_id)} computed evidence: {metric}.",
        required=required,
        min_results=1,
        fallback_strategy=["numeric_only"],
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _dimension_text_req(
    *,
    user_query: str,
    company: str,
    dimension_id: str,
    meta: Mapping[str, str],
    primary_sections: list[str],
    fallback_sections: list[str],
    query_terms: list[str],
    profile: str = "summary",
    required: bool = True,
) -> dict[str, Any]:
    primary, fallback = _normalize_text_sections(primary_sections, fallback_sections)
    retrieval_query = _intent_query(user_query, company, query_terms[0])
    broadened = _ordered_unique([_intent_query(user_query, company, terms) for terms in query_terms])
    return _text_requirement(
        requirement_id=_dimension_req_id(company, dimension_id, "TEXT"),
        company=company,
        purpose=f"{meta.get('dimension_name', dimension_id)} filing-text evidence.",
        primary_sections=primary,
        fallback_sections=fallback,
        retrieval_query=retrieval_query,
        retrieval_intent=dimension_id,
        retrieval_profile=profile,
        broadened_queries=broadened,
        required=required,
        min_results=1,
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _compact_comparison_risk_text_req(
    *,
    company: str,
    dimension_id: str,
    meta: Mapping[str, str],
) -> dict[str, Any]:
    primary, fallback = _normalize_text_sections(["ITEM_1A", "ITEM_1", "BUSINESS"], ["ITEM_7", "MD&A"])
    query = f"{company} competition risk factors"
    return _text_requirement(
        requirement_id=_dimension_req_id(company, dimension_id, "TEXT"),
        company=company,
        purpose=f"Company-specific competitive risk context for comparing {company} under the methodology framework.",
        primary_sections=primary,
        fallback_sections=fallback,
        retrieval_query=query,
        retrieval_intent="comparison_risk_context",
        retrieval_profile="risk_summary",
        broadened_queries=[
            f"{company} competition risk factors",
            f"{company} business risks competitive pressure",
            f"{company} risk factors competitive pressure",
        ],
        required=True,
        min_results=1,
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _valuation_missing_check_req(meta: Mapping[str, str]) -> dict[str, Any]:
    return _base_req(
        requirement_id=_req_id("METH", None, "valuation_and_risk_boundary_valuation_evidence_missing"),
        requirement_type="calculation",
        company=None,
        metric="price",
        metrics=["price"],
        period_type="latest",
        purpose="Record that valuation evidence is outside the current compact methodology comparison packet.",
        required=True,
        min_results=1,
        fallback_strategy=["valuation_evidence_missing"],
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _single_company_valuation_requirements(
    *,
    company: str,
    meta: Mapping[str, str],
    period_type: str | None,
) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for metric, req_period, required in (
        ("price", "latest", True),
        ("shares_outstanding", "latest", True),
        ("revenue", period_type, True),
        ("net_income", period_type, True),
        ("free_cash_flow", period_type, False),
    ):
        requirements.append(
            _dimension_numeric_req(
                company=company,
                dimension_id=VALUATION_AND_RISK_BOUNDARY,
                metric=metric,
                meta=meta,
                period_type=req_period,
                required=required,
                suffix=f"valuation_{metric}",
            )
        )
    for metric, required in (
        ("market_cap", True),
        ("pe_ratio", False),
        ("ps_ratio", False),
        ("fcf_yield", False),
    ):
        requirements.append(
            _dimension_calc_req(
                company=company,
                dimension_id=VALUATION_AND_RISK_BOUNDARY,
                metric=metric,
                meta=meta,
                period_type=period_type,
                required=required,
                suffix=f"valuation_{metric}",
            )
        )
    return requirements


def _is_compact_methodology_comparison(
    *,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    active_dimensions: list[str],
) -> bool:
    if not active_dimensions:
        return False
    return (
        task_type == "company_comparison"
        or answer_mode == "comparison_brief"
        or safety_intent == "investment_advice_like"
    )


def _compact_methodology_comparison_requirements(
    *,
    user_query: str,
    companies: list[str],
    selected_framework: Mapping[str, Any],
    period_type: str | None,
) -> list[dict[str, Any]]:
    raw_requirements: list[dict[str, Any]] = []
    active_dimensions = _active_dimension_ids(selected_framework)
    for company in companies:
        if REVENUE_QUALITY in active_dimensions:
            meta = _dimension_meta(selected_framework, REVENUE_QUALITY)
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue",
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue",
                    meta=meta,
                    period_type="ttm",
                    required=False,
                    min_results=3,
                    suffix="revenue_history",
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue_growth",
                    meta=meta,
                    period_type="ttm",
                    required=False,
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_text_req(
                    user_query=user_query,
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    meta=meta,
                    primary_sections=["ITEM_7", "MD&A"],
                    fallback_sections=["ITEM_1", "ITEM_2"],
                    query_terms=[
                        f"{company} revenue growth revenue quality",
                        f"{company} net sales discussion revenue growth",
                        f"{company} revenue trend demand segment revenue",
                    ],
                    required=False,
                ),
            )
        if PROFITABILITY_QUALITY in active_dimensions:
            meta = _dimension_meta(selected_framework, PROFITABILITY_QUALITY)
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=PROFITABILITY_QUALITY,
                    metric="net_income",
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=PROFITABILITY_QUALITY,
                    metric="net_margin",
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
        if CASH_FLOW_QUALITY in active_dimensions:
            meta = _dimension_meta(selected_framework, CASH_FLOW_QUALITY)
            for metric in ("operating_cash_flow", "free_cash_flow", "capital_expenditure"):
                _append_valid(
                    raw_requirements,
                    _dimension_numeric_req(
                        company=company,
                        dimension_id=CASH_FLOW_QUALITY,
                        metric=metric,
                        meta=meta,
                        period_type=period_type,
                        required=True,
                    ),
                )
            for metric in ("cfo_to_net_income", "fcf_margin"):
                _append_valid(
                    raw_requirements,
                    _dimension_calc_req(
                        company=company,
                        dimension_id=CASH_FLOW_QUALITY,
                        metric=metric,
                        meta=meta,
                        period_type=period_type,
                        required=False,
                    ),
                )
        if VALUATION_AND_RISK_BOUNDARY in active_dimensions:
            meta = _dimension_meta(selected_framework, VALUATION_AND_RISK_BOUNDARY)
            for requirement in _single_company_valuation_requirements(
                company=company,
                meta=meta,
                period_type=period_type,
            ):
                _append_valid(raw_requirements, requirement)
        if MOAT_AND_COMPETITIVE_RISK in active_dimensions:
            meta = _dimension_meta(selected_framework, MOAT_AND_COMPETITIVE_RISK)
            _append_valid(
                raw_requirements,
                _compact_comparison_risk_text_req(
                    company=company,
                    dimension_id=MOAT_AND_COMPETITIVE_RISK,
                    meta=meta,
                ),
            )
    return raw_requirements


def _single_company_mda_text_req(
    *,
    user_query: str,
    company: str,
    dimension_id: str,
    meta: Mapping[str, str],
) -> dict[str, Any]:
    primary, fallback = _normalize_text_sections(["ITEM_7", "MD&A"], ["ITEM_1", "ITEM_1A"])
    return _text_requirement(
        requirement_id=_dimension_req_id(company, dimension_id, "MDA_TEXT"),
        company=company,
        purpose=f"{meta.get('dimension_name', dimension_id)} MD&A operating-results context.",
        primary_sections=primary,
        fallback_sections=fallback,
        retrieval_query=_intent_query(user_query, company, "management discussion operating results revenue margin"),
        retrieval_intent="single_company_operating_context",
        retrieval_profile="summary",
        broadened_queries=[
            _intent_query(user_query, company, "management discussion operating results"),
            _intent_query(user_query, company, "revenue net income margin operating results"),
            _intent_query(user_query, company, "MD&A operating performance"),
        ],
        required=False,
        min_results=1,
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _single_company_text_req(
    *,
    requirement_id: str,
    company: str,
    dimension_id: str,
    meta: Mapping[str, str],
    purpose: str,
    primary_sections: list[str],
    fallback_sections: list[str],
    retrieval_query: str,
    fallback_queries: list[str],
    retrieval_intent: str,
    retrieval_profile: str = "summary",
    required: bool = True,
) -> dict[str, Any]:
    primary, fallback = _normalize_text_sections(primary_sections, fallback_sections)
    return _text_requirement(
        requirement_id=requirement_id,
        company=company,
        purpose=purpose,
        primary_sections=primary,
        fallback_sections=fallback,
        retrieval_query=retrieval_query,
        retrieval_intent=retrieval_intent,
        retrieval_profile=retrieval_profile,
        broadened_queries=_ordered_unique([retrieval_query] + fallback_queries),
        required=required,
        min_results=1,
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _segment_or_product_scope_from_query(user_query: str) -> str:
    query = str(user_query or "").lower()
    checks = (
        ("Compute & Networking", ("compute & networking", "compute and networking")),
        ("networking", ("networking", "网络业务", "网络产品", "网络互连", "互连产品", "infiniband", "ethernet", "nvlink", "spectrum-x")),
        ("data center", ("data center", "数据中心")),
        ("AWS", ("aws", "amazon web services")),
        ("Azure", ("azure",)),
        ("iPhone", ("iphone", "苹果手机")),
    )
    for label, terms in checks:
        if any(term in query for term in terms):
            return label
    return ""


def _business_model_query_terms(company: str) -> tuple[str, list[str]]:
    ticker = str(company or "").upper().strip()
    if ticker == "AMZN":
        primary = f"{ticker} business overview products services revenue sources segments net sales AWS Prime marketplace advertising fulfillment reportable segments"
        fallback = [
            f"{ticker} products and services",
            f"{ticker} business segments revenue sources",
            f"{ticker} net sales AWS Prime marketplace advertising fulfillment",
            f"{ticker} reportable segments segment revenue",
            f"{ticker} customers markets platforms",
        ]
        return primary, fallback
    if ticker == "AAPL":
        primary = f"{ticker} business overview products services revenue sources iPhone Mac iPad Services App Store geographic net sales"
        fallback = [
            f"{ticker} products services iPhone Mac iPad",
            f"{ticker} services net sales App Store",
            f"{ticker} net sales by category geographic markets",
            f"{ticker} customers markets platforms",
        ]
        return primary, fallback
    if ticker == "MSFT":
        primary = f"{ticker} business overview products services revenue sources cloud Azure Office LinkedIn Windows gaming segments"
        fallback = [
            f"{ticker} cloud Azure Office revenue sources",
            f"{ticker} business segments productivity cloud personal computing",
            f"{ticker} products services customers markets",
        ]
        return primary, fallback
    if ticker == "NVDA":
        primary = f"{ticker} business overview products services revenue sources Data Center Compute & Networking Graphics Gaming Automotive networking"
        fallback = [
            f"{ticker} Data Center Compute & Networking segment revenue",
            f"{ticker} networking InfiniBand Ethernet NVLink Spectrum-X",
            f"{ticker} products services customers markets",
        ]
        return primary, fallback
    primary = f"{ticker} business overview products services revenue sources segments net sales reportable segments"
    fallback = [
        f"{ticker} products and services",
        f"{ticker} business segments revenue sources",
        f"{ticker} reportable segments segment revenue",
        f"{ticker} customers markets platforms",
    ]
    return primary, fallback


def _segment_product_requirement_specs(company: str, scope: str) -> list[tuple[str, str, str, list[str], bool]]:
    ticker = str(company or "").upper().strip()
    label = str(scope or "").strip()
    if ticker == "NVDA" and label.lower() in {"networking", "compute & networking"}:
        terms = [
            "Networking revenue",
            "Compute & Networking",
            "InfiniBand",
            "Ethernet",
            "NVLink",
            "Spectrum-X",
            "Blackwell",
            "GB200 GB300",
        ]
    else:
        terms = [label, f"{label} revenue", f"{label} growth", f"{label} driver"]
    base = f"{ticker} {label}".strip()
    return [
        ("SEGMENT_PRODUCT_CURRENT_REVENUE", "segment/product current revenue evidence.", f"{base} current revenue", terms, True),
        ("SEGMENT_PRODUCT_COMPARATOR_REVENUE", "segment/product comparator-period revenue evidence.", f"{base} prior period revenue comparable revenue", terms, True),
        ("SEGMENT_PRODUCT_GROWTH_RATE", "segment/product growth-rate evidence.", f"{base} revenue growth rate increase", terms, True),
        ("SEGMENT_PRODUCT_DRIVER_TEXT", "segment/product driver text.", f"{base} revenue growth driven by due to demand product", terms, True),
        ("SEGMENT_PRODUCT_KEYWORD_TEXT", "product keyword text for segment/product scope.", f"{base} {' '.join(terms)}", terms, True),
    ]


def _segment_product_driver_requirements(
    *,
    user_query: str,
    company: str,
    dimension_id: str,
    meta: Mapping[str, str],
    scope: str,
) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    for suffix, purpose, query_terms, fallback_terms, required in _segment_product_requirement_specs(company, scope):
        retrieval_query = _intent_query(user_query, company, query_terms)
        fallback_queries = [_intent_query(user_query, company, term) for term in fallback_terms]
        req = _single_company_text_req(
            requirement_id=_req_id("TEXT", company, suffix),
            company=company,
            dimension_id=dimension_id,
            meta=meta,
            purpose=f"{scope} {purpose}",
            primary_sections=["ITEM_7", "ITEM_2", "ITEM_1"],
            fallback_sections=["BUSINESS", "ITEM_1A"],
            retrieval_query=retrieval_query,
            fallback_queries=fallback_queries,
            retrieval_intent="segment_product_driver",
            retrieval_profile="summary",
            required=required,
        )
        req["segment_or_product_scope"] = scope
        req["answer_part_ids"] = ["segment_product_driver"]
        requirements.append(req)
    return requirements


def _single_company_methodology_requirements(
    *,
    user_query: str,
    companies: list[str],
    selected_framework: Mapping[str, Any],
    period_type: str | None,
) -> list[dict[str, Any]]:
    raw_requirements: list[dict[str, Any]] = []
    company = companies[0] if companies else ""
    if not company:
        return raw_requirements
    active_dimensions = _active_dimension_ids(selected_framework)
    segment_or_product_scope = _segment_or_product_scope_from_query(user_query)
    aws_segment_profit_focus = "aws" in str(user_query or "").lower() and any(
        term in str(user_query or "").lower()
        for term in ("利润", "盈利", "operating income", "整体利润", "贡献", "重要", "profit")
    )
    if BUSINESS_MODEL in active_dimensions:
        meta = _dimension_meta(selected_framework, BUSINESS_MODEL)
        business_query, business_fallbacks = _business_model_query_terms(company)
        _append_valid(
            raw_requirements,
            _single_company_text_req(
                requirement_id=_req_id("TEXT", company, "BUSINESS_MODEL"),
                company=company,
                dimension_id=BUSINESS_MODEL,
                meta=meta,
                purpose="Business model, products, services, customers, markets, segments, net sales, and revenue-source context.",
                primary_sections=["ITEM_1", "BUSINESS"],
                fallback_sections=["ITEM_7", "MD&A"],
                retrieval_query=business_query,
                fallback_queries=business_fallbacks,
                retrieval_intent="single_company_business_model",
                retrieval_profile="summary",
            ),
        )
    if REVENUE_QUALITY in active_dimensions:
        meta = _dimension_meta(selected_framework, REVENUE_QUALITY)
        if segment_or_product_scope:
            for req in _segment_product_driver_requirements(
                user_query=user_query,
                company=company,
                dimension_id=REVENUE_QUALITY,
                meta=meta,
                scope=segment_or_product_scope,
            ):
                _append_valid(raw_requirements, req)
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue",
                    meta=meta,
                    period_type=period_type,
                    required=False,
                    suffix="company_revenue_context",
                ),
            )
        else:
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue",
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue",
                    meta=meta,
                    period_type="ttm",
                    required=False,
                    min_results=3,
                    suffix="revenue_history",
                ),
            )
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=REVENUE_QUALITY,
                    metric="revenue_growth",
                    meta=meta,
                    period_type="ttm",
                    required=False,
                ),
            )
    if PROFITABILITY_QUALITY in active_dimensions:
        meta = _dimension_meta(selected_framework, PROFITABILITY_QUALITY)
        _append_valid(
            raw_requirements,
            _dimension_numeric_req(
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                metric="net_income",
                meta=meta,
                period_type=period_type,
                required=True,
            ),
        )
        _append_valid(
            raw_requirements,
            _dimension_numeric_req(
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                metric="net_income",
                meta=meta,
                period_type="ttm",
                required=False,
                min_results=3,
                suffix="net_income_history",
            ),
        )
        _append_valid(
            raw_requirements,
            _dimension_calc_req(
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                metric="net_margin",
                meta=meta,
                period_type=period_type,
                required=True,
            ),
        )
        _append_valid(
            raw_requirements,
            _dimension_calc_req(
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                metric="net_margin",
                meta=meta,
                period_type="ttm",
                required=False,
                suffix="net_margin_trend",
            ),
        )
        for metric in ("gross_margin", "operating_margin", "eps"):
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=PROFITABILITY_QUALITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=False,
                ),
            )
        if aws_segment_profit_focus:
            for metric, suffix in (
                ("aws_operating_income", "aws_operating_income"),
                ("consolidated_operating_income", "consolidated_operating_income"),
                ("aws_revenue", "aws_revenue"),
            ):
                _append_valid(
                    raw_requirements,
                    _dimension_numeric_req(
                        company=company,
                        dimension_id=PROFITABILITY_QUALITY,
                        metric=metric,
                        meta=meta,
                        period_type=period_type,
                        required=False,
                        suffix=suffix,
                    ),
                )
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=PROFITABILITY_QUALITY,
                    metric="segment_profit_contribution",
                    meta=meta,
                    period_type=period_type,
                    required=False,
                    suffix="aws_profit_contribution",
                ),
            )
            _append_valid(
                raw_requirements,
                _single_company_text_req(
                    requirement_id=_req_id("TEXT", company, "AWS_SEGMENT_PROFIT"),
                    company=company,
                    dimension_id=PROFITABILITY_QUALITY,
                    meta=meta,
                    purpose="AWS segment operating income and same-basis consolidated profit context.",
                    primary_sections=["ITEM_7", "ITEM_2"],
                    fallback_sections=["ITEM_1", "BUSINESS"],
                    retrieval_query=f"{company} AWS operating income Amazon Web Services segment operating income",
                    fallback_queries=[
                        f"{company} Amazon Web Services segment operating income",
                        f"{company} AWS segment net sales operating income",
                        f"{company} AWS operating income consolidated operating income",
                    ],
                    retrieval_intent="aws_segment_profitability",
                    retrieval_profile="summary",
                    required=False,
                ),
            )
        _append_valid(
            raw_requirements,
            _single_company_mda_text_req(
                user_query=user_query,
                company=company,
                dimension_id=PROFITABILITY_QUALITY,
                meta=meta,
            ),
        )
    if CASH_FLOW_QUALITY in active_dimensions:
        meta = _dimension_meta(selected_framework, CASH_FLOW_QUALITY)
        for metric in ("operating_cash_flow", "free_cash_flow"):
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=CASH_FLOW_QUALITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
        _append_valid(
            raw_requirements,
            _dimension_numeric_req(
                company=company,
                dimension_id=CASH_FLOW_QUALITY,
                metric="capital_expenditure",
                meta=meta,
                period_type=period_type,
                required=False,
            ),
        )
        _append_valid(
            raw_requirements,
            _dimension_calc_req(
                company=company,
                dimension_id=CASH_FLOW_QUALITY,
                metric="free_cash_flow",
                meta=meta,
                period_type=period_type,
                required=False,
                suffix="computed_free_cash_flow",
            ),
        )
        for metric in ("cfo_to_net_income", "fcf_margin"):
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=CASH_FLOW_QUALITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=False,
                ),
            )
    if BALANCE_SHEET_AND_CAPITAL_INTENSITY in active_dimensions:
        meta = _dimension_meta(selected_framework, BALANCE_SHEET_AND_CAPITAL_INTENSITY)
        for metric in (
            "cash_and_equivalents",
            "total_debt",
            "total_assets",
            "total_liabilities",
            "shareholders_equity",
            "capital_expenditure",
        ):
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=True,
                ),
            )
        _append_valid(
            raw_requirements,
            _dimension_calc_req(
                company=company,
                dimension_id=BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                metric="net_debt",
                meta=meta,
                period_type=period_type,
                required=False,
            ),
        )
        for metric in ("debt_to_equity", "capex_to_revenue", "receivables_to_revenue", "inventory_to_revenue"):
            _append_valid(
                raw_requirements,
                _dimension_calc_req(
                    company=company,
                    dimension_id=BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=False,
                ),
            )
        for metric in ("inventory", "receivables"):
            _append_valid(
                raw_requirements,
                _dimension_numeric_req(
                    company=company,
                    dimension_id=BALANCE_SHEET_AND_CAPITAL_INTENSITY,
                    metric=metric,
                    meta=meta,
                    period_type=period_type,
                    required=False,
                ),
            )
    if MOAT_AND_COMPETITIVE_RISK in active_dimensions:
        meta = _dimension_meta(selected_framework, MOAT_AND_COMPETITIVE_RISK)
        _append_valid(
            raw_requirements,
            _single_company_text_req(
                requirement_id=_req_id("TEXT", company, "RISK"),
                company=company,
                dimension_id=MOAT_AND_COMPETITIVE_RISK,
                meta=meta,
                purpose="Risk-factor evidence for competition, demand, supply chain, regulation, and customer concentration.",
                primary_sections=["ITEM_1A"],
                fallback_sections=["ITEM_7", "MD&A", "ITEM_1", "BUSINESS"],
                retrieval_query=f"{company} risk factors competition demand supply chain regulation customer concentration",
                fallback_queries=[
                    f"{company} competition risks",
                    f"{company} demand supply chain risks",
                    f"{company} regulatory customer concentration risks",
                ],
                retrieval_intent="single_company_risk_context",
                retrieval_profile="risk_summary",
            ),
        )
        _append_valid(
            raw_requirements,
            _single_company_text_req(
                requirement_id=_req_id("TEXT", company, "COMPETITION"),
                company=company,
                dimension_id=MOAT_AND_COMPETITIVE_RISK,
                meta=meta,
                purpose="Competitive position, market position, products, and customer context.",
                primary_sections=["ITEM_1", "ITEM_7", "ITEM_1A"],
                fallback_sections=["BUSINESS", "MD&A"],
                retrieval_query=f"{company} competitive advantage competition market position products customers",
                fallback_queries=[
                    f"{company} competitive position",
                    f"{company} market position products customers",
                    f"{company} industry competition",
                ],
                retrieval_intent="single_company_competition_context",
                retrieval_profile="risk_summary",
            ),
        )
    if VALUATION_AND_RISK_BOUNDARY in active_dimensions:
        meta = _dimension_meta(selected_framework, VALUATION_AND_RISK_BOUNDARY)
        for req in _single_company_valuation_requirements(company=company, meta=meta, period_type=period_type):
            _append_valid(raw_requirements, req)
    return raw_requirements


def _risk_focused_mda_text_req(
    *,
    company: str,
    meta: Mapping[str, str],
) -> dict[str, Any]:
    primary, fallback = _normalize_text_sections(["ITEM_7", "MD&A"], ["ITEM_2", "ITEM_1", "ITEM_1A"])
    return _text_requirement(
        requirement_id=_req_id("TEXT", company, "RISK_MDA"),
        company=company,
        purpose="Management discussion evidence connecting risks to operating results, revenue, margins, or demand.",
        primary_sections=primary,
        fallback_sections=fallback,
        retrieval_query=f"{company} management discussion operating results demand margin revenue risk challenges",
        retrieval_intent="risk_focused_management_context",
        retrieval_profile="risk_summary",
        broadened_queries=[
            f"{company} management discussion operating results demand margin revenue risk challenges",
            f"{company} MD&A demand margin operating results",
            f"{company} operating challenges revenue margin management discussion",
        ],
        required=True,
        min_results=1,
        framework_id=meta.get("framework_id"),
        dimension_id=meta.get("dimension_id"),
        dimension_name=meta.get("dimension_name"),
        analysis_purpose=meta.get("analysis_purpose"),
    )


def _risk_focused_single_company_requirements(
    *,
    companies: list[str],
    selected_framework: Mapping[str, Any],
    period_type: str | None,
) -> list[dict[str, Any]]:
    raw_requirements: list[dict[str, Any]] = []
    company = companies[0] if companies else ""
    if not company:
        return raw_requirements

    business_meta = _dimension_meta(selected_framework, BUSINESS_MODEL)
    risk_meta = _dimension_meta(selected_framework, MOAT_AND_COMPETITIVE_RISK)
    revenue_meta = _dimension_meta(selected_framework, REVENUE_QUALITY)
    profitability_meta = _dimension_meta(selected_framework, PROFITABILITY_QUALITY)

    _append_valid(
        raw_requirements,
        _single_company_text_req(
            requirement_id=_req_id("TEXT", company, "RISK_BUSINESS_MODEL"),
            company=company,
            dimension_id=BUSINESS_MODEL,
            meta=business_meta,
            purpose="Business context for risk materiality and core business exposure.",
            primary_sections=["ITEM_1", "BUSINESS"],
            fallback_sections=["ITEM_7", "MD&A"],
            retrieval_query=f"{company} business overview products services markets customers revenue sources",
            fallback_queries=[
                f"{company} products services markets customers",
                f"{company} business segments revenue sources",
            ],
            retrieval_intent="single_company_business_model",
            retrieval_profile="summary",
            required=False,
        ),
    )
    _append_valid(
        raw_requirements,
        _single_company_text_req(
            requirement_id=_req_id("TEXT", company, "RISK_FACTORS"),
            company=company,
            dimension_id=MOAT_AND_COMPETITIVE_RISK,
            meta=risk_meta,
            purpose="Risk-factor evidence for the most material company-specific risks.",
            primary_sections=["ITEM_1A"],
            fallback_sections=["ITEM_7", "MD&A", "ITEM_1"],
            retrieval_query=f"{company} risk factors competition demand supply chain margin customer regulation",
            fallback_queries=[
                f"{company} biggest risks competition demand supply chain",
                f"{company} risk factors demand margin customer concentration",
                f"{company} competition supply regulation risks",
            ],
            retrieval_intent="risk_focused_risk_factors",
            retrieval_profile="risk_summary",
        ),
    )
    mda_req = _risk_focused_mda_text_req(company=company, meta=risk_meta)
    mda_req["required"] = False
    _append_valid(raw_requirements, mda_req)

    for metric, meta, dimension_id in (
        ("revenue", revenue_meta, REVENUE_QUALITY),
        ("net_income", profitability_meta, PROFITABILITY_QUALITY),
    ):
        _append_valid(
            raw_requirements,
            _dimension_numeric_req(
                company=company,
                dimension_id=dimension_id,
                metric=metric,
                meta=meta,
                period_type=period_type,
                required=False,
            ),
        )
    _append_valid(
        raw_requirements,
        _dimension_calc_req(
            company=company,
            dimension_id=PROFITABILITY_QUALITY,
            metric="net_margin",
            meta=profitability_meta,
            period_type=period_type,
            required=False,
        ),
    )
    _append_valid(
        raw_requirements,
        _dimension_numeric_req(
            company=company,
            dimension_id=REVENUE_QUALITY,
            metric="revenue",
            meta=revenue_meta,
            period_type="ttm",
            required=False,
            min_results=3,
            suffix="risk_revenue_history",
        ),
    )
    _append_valid(
        raw_requirements,
        _dimension_calc_req(
            company=company,
            dimension_id=PROFITABILITY_QUALITY,
            metric="net_margin",
            meta=profitability_meta,
            period_type="ttm",
            required=False,
            suffix="risk_net_margin_trend",
        ),
    )
    return raw_requirements


def _methodology_requirements(
    *,
    user_query: str,
    companies: list[str],
    selected_framework: Mapping[str, Any],
    period_type: str | None,
    task_type: str = "",
    answer_mode: str = "",
    safety_intent: str = "",
    analysis_scope: str = "",
    methodology_intent: str = "",
) -> list[dict[str, Any]]:
    raw_requirements: list[dict[str, Any]] = []
    active_dimensions = _active_dimension_ids(selected_framework)
    if analysis_scope == "single_company" and answer_mode == "risk_focused_analysis":
        return _risk_focused_single_company_requirements(
            companies=companies,
            selected_framework=selected_framework,
            period_type=period_type,
        )
    if analysis_scope == "single_company" and active_dimensions:
        return _single_company_methodology_requirements(
            user_query=user_query,
            companies=companies,
            selected_framework=selected_framework,
            period_type=period_type,
        )
    if _is_compact_methodology_comparison(
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        active_dimensions=active_dimensions,
    ):
        return _compact_methodology_comparison_requirements(
            user_query=user_query,
            companies=companies,
            selected_framework=selected_framework,
            period_type=period_type,
        )
    for company in companies:
        for dimension_id in active_dimensions:
            meta = _dimension_meta(selected_framework, dimension_id)
            if dimension_id == BUSINESS_MODEL:
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_1"],
                        fallback_sections=["ITEM_7"],
                        query_terms=[
                            "business model",
                            "products services customers",
                            "revenue sources segments",
                        ],
                        profile="summary",
                    ),
                )
            elif dimension_id == REVENUE_QUALITY:
                _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric="revenue", meta=meta, period_type=period_type, required=True))
                _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric="revenue", meta=meta, period_type="ttm", required=False, min_results=2, suffix="revenue_trend"))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="revenue_growth", meta=meta, period_type=period_type, required=False))
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_7"],
                        fallback_sections=["ITEM_1"],
                        query_terms=[
                            "revenue growth demand product sales",
                            "net sales discussion",
                            "segment revenue",
                        ],
                    ),
                )
            elif dimension_id == PROFITABILITY_QUALITY:
                for metric in ("revenue", "net_income"):
                    _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=True))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="net_margin", meta=meta, period_type=period_type, required=True))
                for metric in ("gross_margin", "operating_margin"):
                    _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=False))
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_7"],
                        fallback_sections=["ITEM_1A"],
                        query_terms=["margin profitability cost pressure operating income"],
                    ),
                )
            elif dimension_id == CASH_FLOW_QUALITY:
                for metric in ("operating_cash_flow", "free_cash_flow", "capital_expenditure"):
                    _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=True))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="free_cash_flow", meta=meta, period_type=period_type, required=False, suffix="computed_free_cash_flow"))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="cfo_to_net_income", meta=meta, period_type=period_type, required=False))
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_7"],
                        fallback_sections=["ITEM_8"],
                        query_terms=["cash flow operating cash capital expenditures"],
                    ),
                )
            elif dimension_id == BALANCE_SHEET_AND_CAPITAL_INTENSITY:
                for metric in (
                    "cash_and_equivalents",
                    "total_debt",
                    "total_assets",
                    "total_liabilities",
                    "shareholders_equity",
                    "capital_expenditure",
                ):
                    _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=True))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="net_debt", meta=meta, period_type=period_type, required=True))
                for metric in ("debt_to_equity", "capex_to_revenue", "receivables_to_revenue", "inventory_to_revenue"):
                    _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=False))
                for metric in ("inventory", "receivables"):
                    _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric=metric, meta=meta, period_type=period_type, required=False))
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_7", "ITEM_1A"],
                        fallback_sections=["ITEM_1"],
                        query_terms=["liquidity debt capital expenditures working capital"],
                        profile="risk_summary",
                    ),
                )
            elif dimension_id == MOAT_AND_COMPETITIVE_RISK:
                _append_valid(
                    raw_requirements,
                    _dimension_text_req(
                        user_query=user_query,
                        company=company,
                        dimension_id=dimension_id,
                        meta=meta,
                        primary_sections=["ITEM_1A", "ITEM_1"],
                        fallback_sections=["ITEM_7"],
                        query_terms=[
                            "competition risk factors",
                            "competitive pressure regulatory risk",
                            "product demand supply chain risk",
                        ],
                        profile="risk_summary",
                    ),
                )
                _append_valid(raw_requirements, _dimension_numeric_req(company=company, dimension_id=dimension_id, metric="revenue", meta=meta, period_type=period_type, required=False))
                _append_valid(raw_requirements, _dimension_calc_req(company=company, dimension_id=dimension_id, metric="net_margin", meta=meta, period_type=period_type, required=False))
            elif dimension_id == VALUATION_AND_RISK_BOUNDARY:
                for req in _single_company_valuation_requirements(company=company, meta=meta, period_type=period_type):
                    _append_valid(raw_requirements, req)
    return raw_requirements


def _analytical_text_requirements(user_query: str, company: str, intent: str) -> list[dict[str, Any]]:
    config = _text_intent_config(intent)
    risk_primary, risk_fallback = _normalize_text_sections(["ITEM_1A"], ["ITEM_7", "ITEM_1", "ITEM_2"])
    mda_primary, mda_fallback = _normalize_text_sections(["ITEM_7"], ["ITEM_1A", "ITEM_1", "ITEM_2"])
    return [
        _text_requirement(
            requirement_id=_req_id("TEXT", company, f"{intent}_RISK"),
            company=company,
            purpose=f"Risk-factor evidence for {intent.replace('_', ' ')} analysis.",
            primary_sections=risk_primary,
            fallback_sections=risk_fallback,
            retrieval_query=_intent_query(user_query, company, str(config.get("risk_terms", ""))),
            retrieval_intent=intent,
            retrieval_profile=str(config.get("profile", "risk_summary")),
            broadened_queries=_broadened_queries(
                user_query,
                company,
                str(config.get("risk_terms", "")),
                "ITEM_1A",
            ),
        ),
        _text_requirement(
            requirement_id=_req_id("TEXT", company, f"{intent}_MDA"),
            company=company,
            purpose=f"MD&A evidence for {intent.replace('_', ' ')} analysis.",
            primary_sections=mda_primary,
            fallback_sections=mda_fallback,
            retrieval_query=_intent_query(user_query, company, str(config.get("mda_terms", ""))),
            retrieval_intent=intent,
            retrieval_profile=str(config.get("profile", "summary")),
            broadened_queries=_broadened_queries(
                user_query,
                company,
                str(config.get("mda_terms", "")),
                "ITEM_7",
            ),
        ),
    ]


def _comparison_text_requirement(user_query: str, company: str, intent: str, *, advisory: bool) -> dict[str, Any]:
    config = _text_intent_config(intent)
    primary_sections, fallback_sections = _normalize_text_sections(["ITEM_7", "ITEM_1A"], ["ITEM_1", "ITEM_2"])
    suffix = intent if intent.endswith("context") else f"{intent}_context"
    if intent == "comparison_risk":
        purpose = f"Company-specific risk context for comparing {company} without giving investment advice."
        profile = "risk_summary"
    elif intent == "comparison_key_difference":
        purpose = f"Company-specific business and operating context for comparing {company}."
        profile = "summary"
    else:
        purpose = (
            f"Company-specific balanced comparison context for {company} without making a recommendation."
            if advisory
            else f"Company-specific comparison context for {company}."
        )
        profile = str(config.get("profile", "comparison_support"))
    return _text_requirement(
        requirement_id=_req_id("TEXT", company, suffix),
        company=company,
        purpose=purpose,
        primary_sections=primary_sections,
        fallback_sections=fallback_sections,
        retrieval_query=_intent_query(user_query, company, str(config.get("comparison_terms", ""))),
        retrieval_intent=intent,
        retrieval_profile=profile,
        broadened_queries=_broadened_queries(
            user_query,
            company,
            str(config.get("comparison_terms", "")),
            "ITEM_7",
        ),
    )


_COMPUTED_METRICS = {
    "revenue_growth",
    "net_margin",
    "gross_margin",
    "operating_margin",
    "free_cash_flow",
    "cfo_to_net_income",
    "cash_conversion",
    "fcf_margin",
    "debt_to_equity",
    "capex_to_revenue",
    "receivables_to_revenue",
    "inventory_to_revenue",
    "market_cap",
    "pe_ratio",
    "ps_ratio",
    "fcf_yield",
    "segment_profit_contribution",
    "net_debt",
}


def _research_plan_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _research_plan_parts(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in plan.get("required_answer_parts", []) or [] if isinstance(item, Mapping)]


def _research_plan_requests(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in plan.get("evidence_requests", []) or [] if isinstance(item, Mapping)]


def _research_req_id(company: str | None, request_id: str, suffix: str = "") -> str:
    clean = str(request_id or "REQUEST").upper().replace(" ", "_").replace("-", "_")
    return _req_id("RP", company, f"{clean}_{suffix}" if suffix else clean)


def _request_answer_part_ids(request: Mapping[str, Any]) -> list[str]:
    return _ordered_unique([str(item) for item in request.get("answer_part_ids", []) or [] if str(item).strip()])


def _append_research_raw(raw_requirements: list[dict[str, Any]], raw: dict[str, Any], request: Mapping[str, Any]) -> None:
    req = dict(raw)
    req["answer_part_ids"] = _request_answer_part_ids(request)
    req["evidence_request_id"] = str(request.get("id") or "")
    req["evidence_role"] = str(request.get("evidence_role") or req.get("evidence_role") or "")
    req["alternative_group"] = str(request.get("alternative_group") or req.get("alternative_group") or "")
    _append_valid(raw_requirements, req)


def build_requirements_from_research_plan(state: Mapping[str, Any], research_plan: Mapping[str, Any]) -> EvidencePlan:
    """Build an EvidencePlan from a validated ResearchPlan."""
    plan = _research_plan_dict(research_plan)
    user_query = str(state.get("user_query") or plan.get("user_goal") or "")
    task_type = str(state.get("task_type", "fact_qa"))
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    safety_intent = str(state.get("safety_intent", "normal"))
    methodology_intent = str(state.get("methodology_intent", ""))
    analysis_scope = str(state.get("analysis_scope", ""))
    time_policy = str(state.get("time_policy", ""))
    period_scope = str(state.get("period_scope", ""))
    canonical_intent = dict(state.get("canonical_intent", {}) or {})
    evidence_policy = dict(state.get("evidence_policy", {}) or {})
    evidence_policy_id = str(state.get("evidence_policy_id") or evidence_policy.get("policy_id") or "")
    companies = _ordered_unique(
        [str(item).upper() for item in plan.get("companies", []) or [] if str(item).strip()]
        or _companies(state, dict(state.get("analysis_plan", {}) or {}))
    )
    period_type = _period_type_from_state(state)
    raw_requirements: list[dict[str, Any]] = []
    metric_rejections: list[dict[str, Any]] = []

    for request in _research_plan_requests(plan):
        request_id = str(request.get("id") or "").strip() or "request"
        req_type = str(request.get("type") or "numeric").strip()
        scope = str(request.get("scope") or "core").strip()
        required = bool(request.get("required", scope == "core")) and scope == "core"
        company_values = [str(request.get("company") or "").upper().strip()] if request.get("company") else companies
        company_values = _ordered_unique([company for company in company_values if company] or companies)
        if not company_values and req_type != "calculation":
            company_values = [None]  # type: ignore[list-item]
        raw_metrics = [str(metric) for metric in request.get("metrics", []) or [] if str(metric)]
        metrics, rejected_metrics = _normalize_planner_metrics(raw_metrics)
        for item in rejected_metrics:
            metric_rejections.append(
                {
                    **item,
                    "request_id": request_id,
                    "evidence_request_id": request_id,
                    "company": ",".join(company_values),
                }
            )
        base_metrics = [metric for metric in metrics if metric not in _COMPUTED_METRICS]
        computed_metrics = [metric for metric in metrics if metric in _COMPUTED_METRICS]
        if req_type == "numeric" and not base_metrics and not computed_metrics and not raw_metrics:
            base_metrics = ["revenue"]
        if req_type in {"numeric", "calculation"} and raw_metrics and rejected_metrics and not base_metrics and not computed_metrics:
            continue
        queries = _ordered_unique([str(query) for query in request.get("queries", []) or [] if str(query).strip()])
        sections = _ordered_unique([str(section).upper() for section in request.get("sections", []) or [] if str(section).strip()])
        evidence_role = str(request.get("evidence_role") or "").strip()
        alternative_group = str(request.get("alternative_group") or "").strip()

        for company in company_values:
            company_text = str(company).upper() if company else None
            if req_type == "text":
                retrieval_query = queries[0] if queries else _text_query(user_query, company_text, str(request.get("purpose") or request_id))
                raw = _base_req(
                    requirement_id=_research_req_id(company_text, request_id),
                    requirement_type="text",
                    company=company_text,
                    section_preferences=sections,
                    retrieval_query=retrieval_query,
                    purpose=str(request.get("purpose") or "Research-plan text evidence."),
                    required=required,
                    requirement_scope=scope,
                    min_results=max(1, int(request.get("min_results", 1) or 1)),
                    fallback_strategy=list(request.get("fallback_strategy", []) or ["strict_broadened_query", "relaxed_sections_intent_query"]),
                    answer_part_ids=_request_answer_part_ids(request),
                    evidence_request_id=request_id,
                    evidence_role=evidence_role,
                    alternative_group=alternative_group,
                )
                raw["retrieval_intent"] = request_id
                raw["retrieval_profile"] = "summary" if request_id != "growth_driver_text" else "risk_summary"
                raw["primary_sections"] = sections
                raw["fallback_sections"] = [section for section in ["ITEM_7", "ITEM_2", "ITEM_1", "ITEM_1A"] if section not in sections]
                raw["broadened_queries"] = queries or [retrieval_query]
                _append_research_raw(raw_requirements, raw, request)
                continue

            if req_type == "numeric":
                if base_metrics:
                    _append_research_raw(
                        raw_requirements,
                        _base_req(
                            requirement_id=_research_req_id(company_text, request_id, "NUM"),
                            requirement_type="numeric",
                            company=company_text,
                            metrics=base_metrics,
                            period_type=period_type,
                            purpose=str(request.get("purpose") or "Research-plan numeric evidence."),
                            required=required,
                            requirement_scope=scope,
                            min_results=max(1, int(request.get("min_results", 1) or 1)),
                            fallback_strategy=["latest_period", "relax_period"],
                            answer_part_ids=_request_answer_part_ids(request),
                            evidence_request_id=request_id,
                            evidence_role=evidence_role,
                            alternative_group=alternative_group,
                        ),
                        request,
                    )
                for metric in computed_metrics:
                    required_calc = required
                    calc_scope = scope
                    if metric in {"revenue_growth", "net_margin"} and required and evidence_role != "revenue_growth_calculation":
                        required_calc = False
                        calc_scope = "optional_context"
                    _append_research_raw(
                        raw_requirements,
                        _base_req(
                            requirement_id=_research_req_id(company_text, request_id, metric),
                            requirement_type="calculation",
                            company=company_text,
                            metric=metric,
                            metrics=[metric],
                            period_type=period_type,
                            purpose=str(request.get("purpose") or f"Research-plan computed evidence: {metric}."),
                            required=required_calc,
                            requirement_scope=calc_scope,
                            min_results=1,
                            fallback_strategy=["numeric_only"],
                            answer_part_ids=_request_answer_part_ids(request),
                            evidence_request_id=request_id,
                            evidence_role=evidence_role or ("revenue_growth_calculation" if metric == "revenue_growth" else ""),
                            alternative_group=alternative_group,
                        ),
                        request,
                    )
                continue

            if req_type == "calculation":
                for metric in computed_metrics or metrics:
                    _append_research_raw(
                        raw_requirements,
                        _base_req(
                            requirement_id=_research_req_id(company_text, request_id, metric),
                            requirement_type="calculation",
                            company=company_text,
                            metric=metric,
                            metrics=[metric],
                            period_type=period_type,
                            purpose=str(request.get("purpose") or f"Research-plan computed evidence: {metric}."),
                            required=required,
                            requirement_scope=scope,
                            min_results=max(1, int(request.get("min_results", 1) or 1)),
                            fallback_strategy=["numeric_only"],
                            answer_part_ids=_request_answer_part_ids(request),
                            evidence_request_id=request_id,
                            evidence_role=evidence_role or ("revenue_growth_calculation" if metric == "revenue_growth" else ""),
                            alternative_group=alternative_group,
                        ),
                        request,
                    )
                continue

            if req_type == "event":
                _append_research_raw(
                    raw_requirements,
                    _base_req(
                        requirement_id=_research_req_id(company_text, request_id),
                        requirement_type="event",
                        company=company_text,
                        purpose=str(request.get("purpose") or "Research-plan event evidence."),
                        required=required,
                        requirement_scope=scope,
                        min_results=max(1, int(request.get("min_results", 1) or 1)),
                        fallback_strategy=["skip_optional_event"],
                        answer_part_ids=_request_answer_part_ids(request),
                        evidence_request_id=request_id,
                        evidence_role=evidence_role,
                        alternative_group=alternative_group,
                    ),
                    request,
                )

    rejected: list[dict[str, Any]] = list(metric_rejections)
    requirements: list[EvidenceRequirement] = []
    for raw in raw_requirements:
        req, req_rejected = validate_evidence_requirement(raw)
        rejected.extend(req_rejected)
        if req is not None:
            requirements.append(req)

    core_ids = [r.requirement_id for r in requirements if r.requirement_scope == "core"]
    optional_context_ids = [r.requirement_id for r in requirements if r.requirement_scope == "optional_context"]
    diagnostic_ids = [r.requirement_id for r in requirements if r.requirement_scope == "diagnostic"]
    required_ids = [r.requirement_id for r in requirements if r.required and r.requirement_scope == "core"]
    return EvidencePlan(
        user_query=user_query,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        primary_dimension=str(state.get("primary_dimension", "")),
        required_dimensions=[str(item) for item in state.get("required_dimensions", []) or [] if str(item)],
        optional_dimensions=[str(item) for item in state.get("optional_dimensions", []) or [] if str(item)],
        supporting_context_dimensions=[str(item) for item in state.get("supporting_context_dimensions", []) or [] if str(item)],
        evidence_policy_id=evidence_policy_id,
        evidence_policy=evidence_policy,
        canonical_intent=canonical_intent,
        time_policy=time_policy,
        period_scope=period_scope,
        analysis_goal=str(plan.get("user_goal") or user_query)[:500],
        evidence_requirements=requirements,
        sufficiency_criteria={
            "required_requirement_ids": required_ids,
            "required_count": len(required_ids),
            "core_requirement_ids": core_ids,
            "optional_context_requirement_ids": optional_context_ids,
            "diagnostic_requirement_ids": diagnostic_ids,
            "required_answer_parts": _research_plan_parts(plan),
            "allow_partial_synthesis": True,
        },
        core_requirement_ids=core_ids,
        optional_context_requirement_ids=optional_context_ids,
        diagnostic_requirement_ids=diagnostic_ids,
        expected_synthesis_style="causal_explanation" if str(plan.get("question_type")) == "causal_explanation" else "research_plan",
        rejected_requirements=rejected,
        research_plan=plan,
        required_answer_parts=_research_plan_parts(plan),
        evidence_request_map={str(item.get("id")): dict(item) for item in _research_plan_requests(plan) if str(item.get("id"))},
        plan_source=str(plan.get("planner_source") or "research_plan"),
    )


def _plan_requirements(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(item) for item in plan.get("evidence_requirements", []) or [] if isinstance(item, Mapping)]


def _core_requirement_ids(plan: Mapping[str, Any]) -> list[str]:
    explicit = [str(item) for item in plan.get("core_requirement_ids", []) or [] if str(item)]
    if explicit:
        return explicit
    return [
        str(req.get("requirement_id") or "")
        for req in _plan_requirements(plan)
        if str(req.get("requirement_scope") or ("core" if bool(req.get("required", True)) else "optional_context")) == "core"
        and str(req.get("requirement_id") or "")
    ]


def _required_dimensions_from_state(state: Mapping[str, Any]) -> set[str]:
    values: list[Any] = []
    for key in ("required_dimensions", "optional_dimensions", "supporting_context_dimensions"):
        values.extend(list(state.get(key, []) or []))
    canonical = dict(state.get("canonical_intent", {}) or {})
    values.extend(list(canonical.get("requested_dimensions", []) or []))
    return {str(item) for item in values if str(item)}


def _is_composite_request(state: Mapping[str, Any]) -> bool:
    dims = _required_dimensions_from_state(state)
    if len(dims) > 1:
        return True
    query = str(state.get("user_query") or "").lower()
    composite_terms = ("现金流", "估值", "风险", "cash flow", "valuation", "risk")
    return sum(1 for term in composite_terms if term in query) >= 2


def evaluate_plan_coverage(
    *,
    research_plan: Mapping[str, Any],
    research_evidence_plan: Mapping[str, Any],
    legacy_evidence_plan: Mapping[str, Any],
    state: Mapping[str, Any],
    planner_valid: bool = True,
    mode: str = "expanded",
) -> CoverageDecision:
    """Decide whether planner requirements may replace legacy coverage."""
    legacy_core = _core_requirement_ids(legacy_evidence_plan)
    research_core = _core_requirement_ids(research_evidence_plan)
    legacy_core_count = len(legacy_core)
    research_core_count = len(research_core)
    coverage_ratio = 1.0 if legacy_core_count == 0 else round(research_core_count / legacy_core_count, 4)
    question_type = str(research_plan.get("question_type") or "")
    warnings: list[str] = []
    reason = ""

    if mode == "shadow":
        strategy = PlanExecutionStrategy.LEGACY_ONLY
        reason = "shadow_mode_uses_legacy_evidence_plan"
    elif not planner_valid:
        strategy = PlanExecutionStrategy.LEGACY_ONLY
        reason = "planner_invalid"
    elif question_type == "causal_explanation":
        strategy = PlanExecutionStrategy.REPLACE
        reason = "causal_planner_controls_driver_evidence"
    elif question_type in {"overview", "risk_analysis", "valuation_boundary", "cash_flow_quality", "comparison"} or _is_composite_request(state):
        strategy = PlanExecutionStrategy.MERGE
        reason = f"{question_type or 'composite'}_requires_legacy_coverage"
    elif question_type == "direct_fact":
        strategy = PlanExecutionStrategy.REPLACE
        reason = "direct_fact_allows_lightweight_research_plan"
    else:
        strategy = PlanExecutionStrategy.MERGE if legacy_core_count else PlanExecutionStrategy.REPLACE
        reason = "default_preserve_legacy_coverage" if legacy_core_count else "no_legacy_core_to_preserve"

    if legacy_core_count and coverage_ratio < 0.8 and strategy != PlanExecutionStrategy.LEGACY_ONLY and question_type != "causal_explanation":
        strategy = PlanExecutionStrategy.MERGE
        warnings.append("research_plan_under_covered_legacy_core")
    if question_type == "overview":
        legacy_dims = {str(req.get("dimension_id") or "") for req in _plan_requirements(legacy_evidence_plan) if str(req.get("dimension_id") or "")}
        missing_dims = sorted(OVERVIEW_MINIMUM_DIMENSIONS - legacy_dims)
        if missing_dims:
            warnings.append("legacy_overview_minimum_dimensions_incomplete")
        strategy = PlanExecutionStrategy.MERGE
        reason = "overview_research_plan_augments_legacy_coverage"

    retained = legacy_core_count if strategy in {PlanExecutionStrategy.MERGE, PlanExecutionStrategy.AUGMENT_ONLY, PlanExecutionStrategy.LEGACY_ONLY} else 0
    dropped = [] if retained else list(legacy_core)
    return CoverageDecision(
        strategy=strategy,
        legacy_core_count=legacy_core_count,
        research_core_count=research_core_count,
        retained_legacy_core_count=retained,
        dropped_legacy_core_ids=dropped,
        added_research_requirement_ids=list(research_core),
        coverage_ratio=coverage_ratio,
        warnings=warnings,
        reason=reason,
    )


def _scope_rank(scope: str) -> int:
    return {"diagnostic": 0, "optional_context": 1, "core": 2}.get(str(scope or ""), 2)


def _stronger_scope(left: str, right: str) -> str:
    return left if _scope_rank(left) >= _scope_rank(right) else right


def _semantic_requirement_key(req: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = _ordered_unique(
        [normalize_planner_metric(str(item)) or str(item) for item in list(req.get("metrics", []) or []) + [req.get("metric")] if str(item or "").strip()]
    )
    sections = _ordered_unique([str(item).upper().strip() for item in req.get("section_preferences", []) or [] if str(item).strip()])
    return (
        str(req.get("company") or "").upper(),
        str(req.get("requirement_type") or ""),
        tuple(sorted(metrics)),
        tuple(sorted(sections)),
        str(req.get("evidence_role") or ""),
        str(req.get("dimension_id") or ""),
    )


def _requirement_source(req: Mapping[str, Any], source: str) -> dict[str, Any]:
    item = dict(req)
    existing = [str(value) for value in item.get("merged_from", []) or [] if str(value)]
    if source not in existing:
        existing.append(source)
    item["merged_from"] = existing
    return item


def _merge_requirement_dict(base: Mapping[str, Any], incoming: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    incoming_dict = dict(incoming)
    merged["merged_from"] = _ordered_unique(list(merged.get("merged_from", []) or []) + list(incoming_dict.get("merged_from", []) or []))
    merged["answer_part_ids"] = _ordered_unique(list(merged.get("answer_part_ids", []) or []) + list(incoming_dict.get("answer_part_ids", []) or []))
    merged["fallback_strategy"] = _ordered_unique(list(merged.get("fallback_strategy", []) or []) + list(incoming_dict.get("fallback_strategy", []) or []))
    merged["required"] = bool(merged.get("required", True)) or bool(incoming_dict.get("required", True))
    merged["requirement_scope"] = _stronger_scope(str(merged.get("requirement_scope") or ""), str(incoming_dict.get("requirement_scope") or ""))
    merged["metrics"] = _ordered_unique(list(merged.get("metrics", []) or []) + list(incoming_dict.get("metrics", []) or []))
    merged["section_preferences"] = _ordered_unique(list(merged.get("section_preferences", []) or []) + list(incoming_dict.get("section_preferences", []) or []))
    merged["merged_requirement_ids"] = _ordered_unique(
        list(merged.get("merged_requirement_ids", []) or [])
        + [str(merged.get("requirement_id") or ""), str(incoming_dict.get("requirement_id") or "")]
    )
    if incoming_dict.get("evidence_request_id") and not merged.get("evidence_request_id"):
        merged["evidence_request_id"] = incoming_dict.get("evidence_request_id")
    return merged


def merge_evidence_requirements(
    *,
    legacy_evidence_plan: Mapping[str, Any],
    research_evidence_plan: Mapping[str, Any],
    coverage_decision: CoverageDecision | Mapping[str, Any],
) -> EvidencePlan:
    decision = coverage_decision if isinstance(coverage_decision, CoverageDecision) else CoverageDecision(**dict(coverage_decision or {}))
    strategy = decision.strategy if isinstance(decision.strategy, PlanExecutionStrategy) else PlanExecutionStrategy(str(decision.strategy))
    if strategy == PlanExecutionStrategy.REPLACE:
        plan = dict(research_evidence_plan)
        plan["plan_source"] = "research_plan"
        plan["plan_coverage_decision"] = decision.model_dump(mode="json")
        plan["requirement_merge_summary"] = RequirementMergeSummary(strategy=strategy, merged_total_requirements=len(_plan_requirements(plan))).model_dump(mode="json")
        return EvidencePlan(**plan)
    if strategy == PlanExecutionStrategy.LEGACY_ONLY:
        plan = dict(legacy_evidence_plan)
        plan["plan_source"] = "legacy_evidence_plan"
        plan["plan_coverage_decision"] = decision.model_dump(mode="json")
        plan["requirement_merge_summary"] = RequirementMergeSummary(
            strategy=strategy,
            merged_total_requirements=len(_plan_requirements(plan)),
            legacy_only_count=len(_plan_requirements(plan)),
            retained_legacy_core_count=decision.legacy_core_count,
        ).model_dump(mode="json")
        return EvidencePlan(**plan)

    merged_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    deduped = 0
    legacy_only = 0
    research_only = 0
    legacy_research = 0
    for req in _plan_requirements(legacy_evidence_plan):
        item = _requirement_source(req, "legacy")
        merged_by_key[_semantic_requirement_key(item)] = item
    for req in _plan_requirements(research_evidence_plan):
        item = _requirement_source(req, "research_plan")
        key = _semantic_requirement_key(item)
        if key in merged_by_key:
            merged_by_key[key] = _merge_requirement_dict(merged_by_key[key], item)
            deduped += 1
        else:
            merged_by_key[key] = item

    requirements = [EvidenceRequirement(**item) for item in merged_by_key.values()]
    for req in requirements:
        sources = set(str(item) for item in getattr(req, "merged_from", []) or [])
        if sources == {"legacy"}:
            legacy_only += 1
        elif sources == {"research_plan"}:
            research_only += 1
        elif "legacy" in sources and "research_plan" in sources:
            legacy_research += 1

    core_ids = [r.requirement_id for r in requirements if r.requirement_scope == "core"]
    optional_context_ids = [r.requirement_id for r in requirements if r.requirement_scope == "optional_context"]
    diagnostic_ids = [r.requirement_id for r in requirements if r.requirement_scope == "diagnostic"]
    required_ids = [r.requirement_id for r in requirements if r.required and r.requirement_scope == "core"]
    legacy_plan = dict(legacy_evidence_plan)
    research_plan_dict = dict(research_evidence_plan.get("research_plan", {}) or {})
    rejected = list(legacy_plan.get("rejected_requirements", []) or []) + list(research_evidence_plan.get("rejected_requirements", []) or [])
    merge_summary = RequirementMergeSummary(
        strategy=strategy,
        merged_total_requirements=len(requirements),
        deduped_requirements=deduped,
        legacy_only_count=legacy_only,
        research_only_count=research_only,
        legacy_research_count=legacy_research,
        retained_legacy_core_count=decision.retained_legacy_core_count,
        added_research_requirement_ids=[r.requirement_id for r in requirements if "research_plan" in set(getattr(r, "merged_from", []) or [])],
        dropped_legacy_core_ids=list(decision.dropped_legacy_core_ids),
    )
    legacy_plan.update(
        {
            "evidence_requirements": requirements,
            "sufficiency_criteria": {
                **dict(legacy_plan.get("sufficiency_criteria", {}) or {}),
                "required_requirement_ids": required_ids,
                "required_count": len(required_ids),
                "core_requirement_ids": core_ids,
                "optional_context_requirement_ids": optional_context_ids,
                "diagnostic_requirement_ids": diagnostic_ids,
                "required_answer_parts": list(research_evidence_plan.get("required_answer_parts", []) or []),
            },
            "core_requirement_ids": core_ids,
            "optional_context_requirement_ids": optional_context_ids,
            "diagnostic_requirement_ids": diagnostic_ids,
            "research_plan": research_plan_dict,
            "required_answer_parts": list(research_evidence_plan.get("required_answer_parts", []) or []),
            "evidence_request_map": dict(research_evidence_plan.get("evidence_request_map", {}) or {}),
            "plan_source": "merged",
            "plan_coverage_decision": decision.model_dump(mode="json"),
            "requirement_merge_summary": merge_summary.model_dump(mode="json"),
            "rejected_requirements": rejected,
        }
    )
    return EvidencePlan(**legacy_plan)


def build_evidence_plan(state: Mapping[str, Any]) -> EvidencePlan:
    user_query = str(state.get("user_query", ""))
    query_understanding = dict(state.get("query_understanding") or state.get("query_understanding_summary") or {})
    task_type = str(state.get("task_type", "fact_qa"))
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    safety_intent = str(state.get("safety_intent", "normal"))
    methodology_intent = str(
        state.get("methodology_intent")
        or dict(state.get("analysis_plan", {}) or {}).get("methodology_intent", "")
        or query_understanding.get("legacy_methodology_intent", "")
        or ""
    )
    analysis_scope = str(state.get("analysis_scope") or dict(state.get("analysis_plan", {}) or {}).get("analysis_scope", "") or "")
    if not analysis_scope and query_understanding.get("analysis_scope") in {"single_company", "comparison"}:
        analysis_scope = str(query_understanding.get("analysis_scope"))
    primary_dimension = str(state.get("primary_dimension") or dict(state.get("analysis_plan", {}) or {}).get("primary_dimension", "") or "")
    required_dimensions = [
        str(item)
        for item in (
            state.get("required_dimensions")
            or dict(state.get("analysis_plan", {}) or {}).get("required_dimensions", [])
            or []
        )
        if str(item).strip()
    ]
    optional_dimensions = [
        str(item)
        for item in (
            state.get("optional_dimensions")
            or dict(state.get("analysis_plan", {}) or {}).get("optional_dimensions", [])
            or []
        )
        if str(item).strip()
    ]
    time_policy = str(state.get("time_policy") or dict(state.get("analysis_plan", {}) or {}).get("time_policy", "") or "")
    period_scope = str(state.get("period_scope") or dict(state.get("analysis_plan", {}) or {}).get("period_scope", "") or "")
    needs_tools = bool(state.get("needs_tools", True))
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    canonical_intent = dict(state.get("canonical_intent") or analysis_plan.get("canonical_intent") or {})
    evidence_policy = dict(state.get("evidence_policy") or analysis_plan.get("evidence_policy") or {})
    evidence_policy_id = str(state.get("evidence_policy_id") or analysis_plan.get("evidence_policy_id") or evidence_policy.get("policy_id") or "")
    companies = _companies(state, analysis_plan)
    period_type = _period_type_from_state(state)
    raw_requirements: list[dict[str, Any]] = []
    selected_framework = _selected_framework(state)
    active_dimensions = _active_dimension_ids(selected_framework)
    intent_reasons = [
        str(item)
        for item in (
            state.get("intent_reasons")
            or query_understanding.get("intent_reasons", [])
            or []
        )
        if str(item)
    ]

    if needs_tools and answer_mode not in {"meta", "clarification", "refusal_or_redirect"}:
        metrics = _numeric_metrics(state, analysis_plan, ["revenue", "net_income"])
        text_intent = _text_requirement_intent(
            task_type=task_type,
            answer_mode=answer_mode,
            safety_intent=safety_intent,
            methodology_intent=methodology_intent,
            analysis_scope=analysis_scope,
            primary_dimension=primary_dimension,
            required_dimensions=required_dimensions,
            active_dimensions=active_dimensions,
            intent_reasons=intent_reasons,
        )
        if safety_intent == "investment_advice_like":
            task_type = "company_comparison"
            answer_mode = "comparison_brief"
            text_intent = _text_requirement_intent(
                task_type=task_type,
                answer_mode=answer_mode,
                safety_intent=safety_intent,
                methodology_intent=methodology_intent,
                analysis_scope=analysis_scope,
                primary_dimension=primary_dimension,
                required_dimensions=required_dimensions,
                active_dimensions=active_dimensions,
                intent_reasons=intent_reasons,
            )

        if active_dimensions:
            raw_requirements.extend(
                _methodology_requirements(
                    user_query=user_query,
                    companies=companies,
                    selected_framework=selected_framework,
                    period_type=period_type,
                    task_type=task_type,
                    answer_mode=answer_mode,
                    safety_intent=safety_intent,
                    analysis_scope=analysis_scope,
                    methodology_intent=methodology_intent,
                )
            )
        elif answer_mode == "direct_fact":
            for company in companies:
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("NUM", company, "FACT"),
                        requirement_type="numeric",
                        company=company,
                        metrics=metrics[: max(1, len(metrics))],
                        period_type=period_type,
                        purpose="Answer the direct factual metric request.",
                        required=True,
                        min_results=1,
                        fallback_strategy=["latest_period", "relax_period"],
                    ),
                )
        elif task_type == "trend_analysis":
            for company in companies:
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("NUM", company, "TREND"),
                        requirement_type="numeric",
                        company=company,
                        metrics=_numeric_metrics(state, analysis_plan, ["revenue", "net_income"]),
                        period_type=period_type,
                        purpose="Collect period series for trend analysis.",
                        required=True,
                        min_results=2,
                        fallback_strategy=["latest_period", "relax_period"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("CALC", company, "GROWTH"),
                        requirement_type="calculation",
                        company=company,
                        metrics=["revenue", "net_income"],
                        period_type=period_type,
                        purpose="Compute growth or period-over-period change.",
                        required=True,
                        min_results=1,
                        fallback_strategy=["numeric_only"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("TEXT", company, "MDA"),
                        requirement_type="text",
                        company=company,
                        section_preferences=["ITEM_7"],
                        retrieval_query=_text_query(user_query, company, "MD&A discussion supporting trend interpretation"),
                        purpose="Find MD&A snippets for trend context.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["relax_sections", "fallback_user_query"],
                    ),
                )
        elif answer_mode == "comparison_brief" and safety_intent == "investment_advice_like":
            for company in companies:
                for metric in ("revenue", "net_income"):
                    _append_valid(
                        raw_requirements,
                        _base_req(
                            requirement_id=_req_id("NUM", company, metric),
                            requirement_type="numeric",
                            company=company,
                            metric=metric,
                            metrics=[metric],
                            period_type=period_type,
                            purpose=f"Collect comparable {metric} evidence for an opinionated but non-advisory company comparison.",
                            required=True,
                            min_results=1,
                            fallback_strategy=["latest_common_period", "relax_period"],
                        ),
                    )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("CALC", company, "OPERATING_MARGIN"),
                        requirement_type="calculation",
                        company=company,
                        metrics=["operating_margin"],
                        period_type=period_type,
                        purpose="Prefer a profitability-derived metric for balanced comparison.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["numeric_only"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("CALC", company, "GROWTH"),
                        requirement_type="calculation",
                        company=company,
                        metrics=["revenue", "net_income"],
                        period_type=period_type,
                        purpose="Prefer growth comparison if sufficient numeric evidence is available.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["numeric_only"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _comparison_text_requirement(
                        user_query,
                        company,
                        text_intent,
                        advisory=True,
                    ),
                )
        elif task_type == "company_comparison" or safety_intent == "investment_advice_like":
            for company in companies:
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("NUM", company, "COMPARISON"),
                        requirement_type="numeric",
                        company=company,
                        metrics=_numeric_metrics(state, analysis_plan, ["revenue", "net_income"]),
                        period_type=period_type,
                        purpose="Collect same-basis numeric evidence for company comparison.",
                        required=True,
                        min_results=1,
                        fallback_strategy=["latest_common_period", "relax_period"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("CALC", company, "COMPARISON"),
                        requirement_type="calculation",
                        company=company,
                        metrics=["revenue", "net_income", "operating_margin"],
                        period_type=period_type,
                        purpose="Compute comparable growth or margin signals when possible.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["numeric_only"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _comparison_text_requirement(
                        user_query,
                        company,
                        text_intent,
                        advisory=False,
                    ),
                )
        elif answer_mode == "cautious_outlook":
            for company in companies:
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("NUM", company, "OUTLOOK"),
                        requirement_type="numeric",
                        company=company,
                        metrics=_numeric_metrics(state, analysis_plan, ["revenue", "net_income", "operating_margin"]),
                        period_type=period_type,
                        purpose="Collect disclosed historical trend evidence for cautious outlook.",
                        required=True,
                        min_results=2,
                        fallback_strategy=["latest_period", "relax_period"],
                    ),
                )
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("CALC", company, "OUTLOOK"),
                        requirement_type="calculation",
                        company=company,
                        metrics=["revenue", "net_income", "operating_margin"],
                        period_type=period_type,
                        purpose="Compute disclosed trend signals for cautious outlook.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["numeric_only"],
                    ),
                )
                for suffix, section, purpose in (
                    ("MDA", "ITEM_7", "MD&A discussion for cautious outlook"),
                    ("RISK", "ITEM_1A", "Risk factors for cautious outlook"),
                ):
                    req_sections = _ordered_unique(list(analysis_plan.get("section_preferences", []) or []) or [section])
                    _append_valid(
                        raw_requirements,
                        _base_req(
                            requirement_id=_req_id("TEXT", company, suffix),
                            requirement_type="text",
                            company=company,
                            section_preferences=req_sections,
                            retrieval_query=_text_query(user_query, company, purpose),
                            purpose=purpose,
                            required=True,
                            min_results=1,
                            fallback_strategy=["relax_sections", "fallback_user_query"],
                        ),
                    )
                if str(state.get("event_intent", "none")) != "none":
                    _append_valid(
                        raw_requirements,
                        _base_req(
                            requirement_id=_req_id("EVENT", company, "FILING_REACTION"),
                            requirement_type="event",
                            company=company,
                            purpose="Optional filing market reaction context.",
                            required=False,
                            min_results=1,
                            fallback_strategy=["skip_optional_event"],
                        ),
                    )
        elif answer_mode == "analytical" or task_type == "report_summary":
            for company in companies:
                for requirement in _analytical_text_requirements(user_query, company, text_intent):
                    _append_valid(raw_requirements, requirement)
                _append_valid(
                    raw_requirements,
                    _base_req(
                        requirement_id=_req_id("NUM", company, "LATEST"),
                        requirement_type="numeric",
                        company=company,
                        metrics=["revenue", "net_income"],
                        period_type="latest",
                        purpose="Optional latest numeric context for open-ended analysis.",
                        required=False,
                        min_results=1,
                        fallback_strategy=["skip_optional_numeric"],
                    ),
                )

    _ensure_profit_decline_history_requirements(
        raw_requirements,
        user_query=user_query,
        companies=companies,
        selected_framework=selected_framework,
    )
    _ensure_change_history_requirements(
        raw_requirements,
        user_query=user_query,
        companies=companies,
        selected_framework=selected_framework,
    )
    raw_requirements = _apply_policy_requirement_scopes(raw_requirements, evidence_policy)
    rejected: list[dict[str, Any]] = []
    requirements: list[EvidenceRequirement] = []
    for raw in raw_requirements:
        req, req_rejected = validate_evidence_requirement(raw)
        rejected.extend(req_rejected)
        if req is not None:
            requirements.append(req)

    core_ids = [r.requirement_id for r in requirements if r.requirement_scope == "core"]
    optional_context_ids = [r.requirement_id for r in requirements if r.requirement_scope == "optional_context"]
    diagnostic_ids = [r.requirement_id for r in requirements if r.requirement_scope == "diagnostic"]
    required_ids = [r.requirement_id for r in requirements if r.required and r.requirement_scope == "core"]
    return EvidencePlan(
        user_query=user_query,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        methodology_intent=methodology_intent,
        analysis_scope=analysis_scope,
        primary_dimension=primary_dimension,
        required_dimensions=required_dimensions,
        optional_dimensions=optional_dimensions,
        supporting_context_dimensions=[
            str(item)
            for item in (
                state.get("supporting_context_dimensions")
                or analysis_plan.get("supporting_context_dimensions", [])
                or []
            )
            if str(item).strip()
        ],
        evidence_policy_id=evidence_policy_id,
        evidence_policy=evidence_policy,
        canonical_intent=canonical_intent,
        time_policy=time_policy,
        period_scope=period_scope,
        analysis_goal=_analysis_goal(user_query, analysis_plan),
        evidence_requirements=requirements,
        sufficiency_criteria={
            "required_requirement_ids": required_ids,
            "required_count": len(required_ids),
            "core_requirement_ids": core_ids,
            "optional_context_requirement_ids": optional_context_ids,
            "diagnostic_requirement_ids": diagnostic_ids,
            "allow_partial_synthesis": True,
        },
        core_requirement_ids=core_ids,
        optional_context_requirement_ids=optional_context_ids,
        diagnostic_requirement_ids=diagnostic_ids,
        expected_synthesis_style={
            "direct_fact": "direct_fact",
            "cautious_outlook": "cautious_outlook",
            "comparison_brief": "balanced_comparison"
            if safety_intent == "investment_advice_like"
            else "comparison_brief",
            "risk_focused_analysis": "risk_focused_analysis",
            "analytical": "validated_analysis",
        }.get(answer_mode, "analytical_brief"),
        rejected_requirements=rejected,
    )
