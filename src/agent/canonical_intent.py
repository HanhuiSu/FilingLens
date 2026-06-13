"""Canonical intent contract for query routing and evidence policy selection."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.methodology_intent import legacy_methodology_intent
from src.agent.output_language import detect_output_language
from src.agent.query_ontology import SUPPORTED_DIMENSIONS
from src.agent.types import CanonicalIntent


_LEGACY_TO_FAMILY = {
    "single_company_overview": "overview",
    "risk_focused_analysis": "risk",
    "cash_flow_quality_analysis": "cash_flow",
    "profitability_quality_analysis": "profitability",
    "revenue_quality_analysis": "revenue",
    "balance_sheet_analysis": "balance_sheet",
    "valuation_boundary_analysis": "valuation",
    "company_comparison": "comparison",
    "investment_advice_like": "comparison",
    "unsupported_prediction": "refusal",
    "": "overview",
}

_FAMILY_TO_LEGACY = {
    "overview": "single_company_overview",
    "risk": "risk_focused_analysis",
    "cash_flow": "cash_flow_quality_analysis",
    "profitability": "profitability_quality_analysis",
    "revenue": "revenue_quality_analysis",
    "balance_sheet": "balance_sheet_analysis",
    "valuation": "valuation_boundary_analysis",
    "comparison": "company_comparison",
    "refusal": "unsupported_prediction",
}

_RISK_GENERAL_TERMS = (
    "有什么风险",
    "有哪些风险",
    "风险点",
    "风险在哪里",
    "风险在哪",
    "经营风险",
    "担心什么",
    "最担心",
    "最需要担心",
    "隐患",
    "最大隐患",
    "主要风险",
    "最大风险",
    "主要压力",
    "出什么问题",
    "可能出问题",
    "值得警惕",
    "最值得警惕",
    "biggest risk",
    "main risk",
    "key risk",
    "key risks",
    "what risk",
    "worry about",
)

_AWS_PROFIT_TERMS = (
    "利润",
    "盈利",
    "operating income",
    "整体利润",
    "贡献",
    "重要",
    "profit",
    "profitability",
)

_COMPARISON_RISK_TERMS = (
    "危险",
    "更危险",
    "风险更大",
    "哪个风险",
    "哪个更有风险",
    "riskier",
    "more risky",
    "greater risk",
    "higher risk",
)


def _as_dict(value: Any) -> dict[str, Any]:
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


def _family_from_methodology(value: Any) -> str:
    raw = str(value or "").strip()
    if raw in {
        "overview",
        "risk",
        "cash_flow",
        "profitability",
        "revenue",
        "balance_sheet",
        "valuation",
        "comparison",
        "refusal",
    }:
        return raw
    return _LEGACY_TO_FAMILY.get(raw, "overview")


def _legacy_from_family(family: str, safety_intent: str) -> str:
    if str(safety_intent or "") in {"unsupported_or_out_of_scope", "prediction", "unsupported"}:
        return "unsupported_prediction"
    if str(safety_intent or "") == "investment_advice_like" and family == "comparison":
        return "investment_advice_like"
    return _FAMILY_TO_LEGACY.get(str(family or ""), "")


def _is_general_risk_query(user_query: str) -> bool:
    q = re.sub(r"\s+", " ", str(user_query or "").lower()).strip()
    return any(term in q for term in _RISK_GENERAL_TERMS)


def _aws_profit_segment_query(user_query: str) -> bool:
    q = re.sub(r"\s+", " ", str(user_query or "").lower()).strip()
    return "aws" in q and any(term in q for term in _AWS_PROFIT_TERMS)


def _segment_or_product_scope(user_query: str) -> str:
    q = re.sub(r"\s+", " ", str(user_query or "").lower()).strip()
    if not q:
        return ""
    checks = (
        ("Compute & Networking", ("compute & networking", "compute and networking")),
        ("networking", ("networking", "网络业务", "网络产品", "网络互连", "互连产品", "infiniband", "ethernet", "nvlink", "spectrum-x")),
        ("data center", ("data center", "数据中心")),
        ("AWS", ("aws", "amazon web services")),
        ("Azure", ("azure",)),
        ("iPhone", ("iphone", "苹果手机")),
    )
    for label, terms in checks:
        if any(term in q for term in terms):
            return label
    return ""


def _comparison_risk_query(user_query: str) -> bool:
    q = re.sub(r"\s+", " ", str(user_query or "").lower()).strip()
    return any(term in q for term in _COMPARISON_RISK_TERMS)


def _time_focus(time_scope: Mapping[str, Any], period_query: Mapping[str, Any]) -> str:
    period_type = str(period_query.get("period_type") or "").strip()
    if period_query.get("is_explicit"):
        return "explicit_range"
    if period_type == "quarterly":
        return "latest_quarter"
    if period_type == "annual":
        return "annual"
    if period_type == "trailing":
        return "latest"
    scope_text = " ".join(str(v) for v in dict(time_scope or {}).values()).lower()
    if "next quarter" in scope_text or "下季度" in scope_text or "下个季度" in scope_text:
        return "next_quarter"
    return "latest"


def _answer_mode_for(
    *,
    intent_family: str,
    analysis_scope: str,
    requested_dimensions: list[str],
    safety_intent: str,
) -> str:
    if safety_intent in {"unsupported_or_out_of_scope", "prediction", "unsupported"} or intent_family == "refusal":
        return "refusal_or_redirect"
    if analysis_scope == "comparison" or intent_family == "comparison":
        return "comparison_brief"
    if analysis_scope == "single_company" and len(requested_dimensions) > 1:
        return "analytical"
    if analysis_scope == "single_company" and intent_family == "risk":
        return "risk_focused_analysis"
    if analysis_scope == "single_company" and intent_family in {
        "cash_flow",
        "valuation",
        "profitability",
        "revenue",
        "balance_sheet",
        "overview",
    }:
        return "analytical"
    return "direct_fact" if intent_family == "overview" else "analytical"


def build_canonical_intent(
    *,
    user_query: str,
    query_understanding: Mapping[str, Any],
    companies: list[str],
    comparison_target: str | None = None,
    methodology_intent: str = "",
    analysis_scope: str = "",
    safety_intent: str = "normal",
    period_query: Mapping[str, Any] | None = None,
    answer_mode_override: str | None = None,
    output_language: str | None = None,
) -> CanonicalIntent:
    """Merge rule and semantic query signals into a stable internal contract."""
    qu = _as_dict(query_understanding)
    semantic = _as_dict(qu.get("semantic_proposal"))
    understood_methodology = str(qu.get("methodology_intent") or "").strip()
    raw_methodology = understood_methodology if understood_methodology not in {"", "none"} else str(methodology_intent or "").strip()
    requested_dimensions = [
        item
        for item in _string_list(qu.get("requested_dimensions"))
        if item in SUPPORTED_DIMENSIONS
    ]
    family = _family_from_methodology(raw_methodology)
    rule_family = _family_from_methodology(qu.get("rule_methodology_intent"))
    proposed_family = _family_from_methodology(qu.get("proposed_methodology_intent") or semantic.get("methodology_intent"))

    safety = str(safety_intent or qu.get("legacy_safety_intent") or "normal").strip() or "normal"
    segment_or_product_scope = _segment_or_product_scope(user_query)
    if safety in {"unsupported_or_out_of_scope", "prediction", "unsupported"}:
        family = "refusal"
    elif _aws_profit_segment_query(user_query):
        family = "profitability"
        requested_dimensions = [
            dimension
            for dimension in ["profitability_quality", "business_model"]
            if dimension not in requested_dimensions
        ] + requested_dimensions
    elif segment_or_product_scope:
        family = "revenue"
        requested_dimensions = [
            dimension
            for dimension in ["revenue_quality", "business_model"]
            if dimension not in requested_dimensions
        ] + requested_dimensions
    elif _is_general_risk_query(user_query) and family in {"overview", "risk"}:
        family = "risk"

    resolved_companies = [str(item).upper().strip() for item in companies if str(item).strip()]
    if comparison_target:
        target = str(comparison_target).upper().strip()
        if target and target not in resolved_companies:
            resolved_companies.append(target)

    if _comparison_risk_query(user_query) and len(resolved_companies) >= 2:
        family = "comparison"
        requested_dimensions = ["moat_and_competitive_risk"]

    scope = str(analysis_scope or qu.get("analysis_scope") or "unknown").strip() or "unknown"
    if family == "comparison" or len(resolved_companies) >= 2:
        scope = "comparison"
    elif family == "refusal":
        scope = "unsupported"
    elif len(resolved_companies) == 1 and family not in {"overview", ""}:
        scope = "single_company"

    legacy = _legacy_from_family(family, safety)
    if raw_methodology in {"", "none"} and family == "overview" and scope != "single_company":
        legacy = ""
    answer_mode = str(answer_mode_override or "").strip() or _answer_mode_for(
        intent_family=family,
        analysis_scope=scope,
        requested_dimensions=requested_dimensions,
        safety_intent=safety,
    )
    confidence = float(qu.get("confidence") or 0.0)
    merge_decision = {
        "rule_intent_family": rule_family,
        "semantic_intent_family": proposed_family if semantic or qu.get("proposed_methodology_intent") else "",
        "final_intent_family": family,
        "source": str(qu.get("intent_source") or "fallback_rules"),
        "safety_override": family == "refusal",
        "explicit_dimensions_preserved": requested_dimensions,
        "reason": list(qu.get("intent_reasons", []) or []),
    }
    source_signals = [
        {
            "source": "rule",
            "intent_family": rule_family,
            "reasons": list(qu.get("intent_reasons", []) or []),
        }
    ]
    if semantic or qu.get("proposed_methodology_intent"):
        source_signals.append(
            {
                "source": "semantic",
                "intent_family": proposed_family,
                "confidence": semantic.get("confidence"),
                "accepted": str(qu.get("intent_source") or "").startswith("semantic"),
            }
        )

    return CanonicalIntent(
        intent_family=family,
        analysis_scope=scope,
        output_language=output_language or detect_output_language(user_query),
        companies=resolved_companies,
        requested_dimensions=requested_dimensions,
        segment_focus=segment_or_product_scope or ("AWS" if _aws_profit_segment_query(user_query) else ""),
        segment_or_product_scope=segment_or_product_scope,
        time_focus=_time_focus(_as_dict(qu.get("time_scope")), _as_dict(period_query)),
        user_expectation=str(qu.get("user_expectation") or "quick_answer"),
        safety_intent=safety,
        confidence=round(max(0.0, min(confidence, 1.0)), 3),
        source_signals=source_signals,
        legacy_methodology_intent=legacy,
        answer_mode=answer_mode,
        intent_merge_decision=merge_decision,
        time_scope=_as_dict(qu.get("time_scope")),
    )
