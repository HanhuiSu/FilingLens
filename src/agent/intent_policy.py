"""Evidence policy registry keyed by CanonicalIntent."""

from __future__ import annotations

from typing import Any, Mapping

from src.agent.analysis_framework import (
    BALANCE_SHEET_AND_CAPITAL_INTENSITY,
    BUSINESS_MODEL,
    CASH_FLOW_QUALITY,
    MOAT_AND_COMPETITIVE_RISK,
    PROFITABILITY_QUALITY,
    REVENUE_QUALITY,
    VALUATION_AND_RISK_BOUNDARY,
)
from src.agent.types import EvidencePolicy


_FAMILY_DIMENSION = {
    "cash_flow": CASH_FLOW_QUALITY,
    "valuation": VALUATION_AND_RISK_BOUNDARY,
    "profitability": PROFITABILITY_QUALITY,
    "revenue": REVENUE_QUALITY,
    "balance_sheet": BALANCE_SHEET_AND_CAPITAL_INTENSITY,
}


def _as_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def _policy(
    *,
    policy_id: str,
    intent_family: str,
    answer_mode: str,
    primary_dimension: str = "",
    required_dimensions: list[str] | None = None,
    optional_dimensions: list[str] | None = None,
    core_requirements: list[str] | None = None,
    optional_context_requirements: list[str] | None = None,
    diagnostic_requirements: list[str] | None = None,
    sufficiency_rule: str = "all_core_requirements",
    allowed_degradation: list[str] | None = None,
) -> EvidencePolicy:
    return EvidencePolicy(
        policy_id=policy_id,
        intent_family=intent_family,
        answer_mode=answer_mode,
        primary_dimension=primary_dimension,
        required_dimensions=list(required_dimensions or []),
        optional_dimensions=list(optional_dimensions or []),
        core_requirements=list(core_requirements or []),
        optional_context_requirements=list(optional_context_requirements or []),
        diagnostic_requirements=list(diagnostic_requirements or []),
        sufficiency_rule=sufficiency_rule,
        allowed_degradation=list(allowed_degradation or []),
    )


def resolve_evidence_policy(canonical_intent: Mapping[str, Any] | Any) -> EvidencePolicy:
    """Return the central evidence policy for a canonical intent."""
    intent = _as_dict(canonical_intent)
    family = str(intent.get("intent_family") or "overview").strip()
    scope = str(intent.get("analysis_scope") or "unknown").strip()
    requested_dimensions = _string_list(intent.get("requested_dimensions"))

    if family == "refusal":
        return _policy(
            policy_id="refusal_v1",
            intent_family=family,
            answer_mode="refusal_or_redirect",
            sufficiency_rule="no_evidence_required",
        )

    if scope == "single_company" and len(requested_dimensions) > 1:
        return _policy(
            policy_id="single_company_composite_v1",
            intent_family=family,
            answer_mode="analytical",
            required_dimensions=requested_dimensions,
            core_requirements=[f"dimension:{dimension_id}" for dimension_id in requested_dimensions],
            sufficiency_rule="all_explicit_dimensions_have_core_signal",
            allowed_degradation=["partial_dimension_caveat"],
        )

    if scope == "single_company" and family == "risk":
        return _policy(
            policy_id="single_company_risk_v1",
            intent_family=family,
            answer_mode="risk_focused_analysis",
            primary_dimension=MOAT_AND_COMPETITIVE_RISK,
            required_dimensions=[MOAT_AND_COMPETITIVE_RISK],
            optional_dimensions=[BUSINESS_MODEL, REVENUE_QUALITY, PROFITABILITY_QUALITY],
            core_requirements=["risk_factors_text"],
            optional_context_requirements=[
                "business_model_text",
                "mda_text",
                "revenue_latest",
                "net_income_latest",
                "margin_trend",
            ],
            sufficiency_rule="risk_text_validated",
            allowed_degradation=["missing_optional_context", "medium_numeric_context"],
        )

    if scope == "single_company" and requested_dimensions == [BUSINESS_MODEL]:
        return _policy(
            policy_id="single_company_business_model_v1",
            intent_family=family,
            answer_mode="analytical",
            primary_dimension=BUSINESS_MODEL,
            required_dimensions=[BUSINESS_MODEL],
            core_requirements=[f"dimension:{BUSINESS_MODEL}"],
            sufficiency_rule="dimension_core_requirements",
            allowed_degradation=["partial_dimension_caveat"],
        )

    if scope == "single_company" and family in _FAMILY_DIMENSION:
        dimension_id = _FAMILY_DIMENSION[family]
        return _policy(
            policy_id=f"single_company_{family}_v1",
            intent_family=family,
            answer_mode="analytical",
            primary_dimension=dimension_id,
            required_dimensions=[dimension_id],
            core_requirements=[f"dimension:{dimension_id}"],
            sufficiency_rule="dimension_core_requirements",
            allowed_degradation=["partial_dimension_caveat"],
        )

    if scope == "comparison" or family == "comparison":
        default_dimensions = [REVENUE_QUALITY, PROFITABILITY_QUALITY, MOAT_AND_COMPETITIVE_RISK]
        dimensions = requested_dimensions or default_dimensions
        return _policy(
            policy_id="comparison_dimension_specific_v1" if requested_dimensions else "comparison_existing_boundary_v1",
            intent_family="comparison",
            answer_mode="comparison_brief",
            primary_dimension=dimensions[0] if len(dimensions) == 1 else "",
            required_dimensions=dimensions,
            optional_dimensions=[] if requested_dimensions else [VALUATION_AND_RISK_BOUNDARY],
            core_requirements=[f"dimension:{dimension_id}" for dimension_id in dimensions]
            if requested_dimensions
            else ["balanced_numeric_evidence"],
            optional_context_requirements=[] if requested_dimensions else ["valuation_context"],
            sufficiency_rule="dimension_specific_comparison_policy" if requested_dimensions else "existing_comparison_policy",
            allowed_degradation=["numeric_only_comparison"],
        )

    return _policy(
        policy_id="single_company_overview_v1",
        intent_family=family,
        answer_mode="analytical",
        required_dimensions=[REVENUE_QUALITY, PROFITABILITY_QUALITY, MOAT_AND_COMPETITIVE_RISK],
        optional_dimensions=[BUSINESS_MODEL, CASH_FLOW_QUALITY, BALANCE_SHEET_AND_CAPITAL_INTENSITY, VALUATION_AND_RISK_BOUNDARY],
        core_requirements=["business_or_risk_text", "core_numeric_context"],
        optional_context_requirements=[
            f"dimension:{BUSINESS_MODEL}",
            "business_model_text",
            "cash_flow_context",
            "balance_sheet_context",
            "valuation_context",
        ],
        sufficiency_rule="overview_core_context",
        allowed_degradation=["partial_dimension_caveat"],
    )
