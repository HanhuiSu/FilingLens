"""Unified semantic contract for incoming user queries."""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

from pydantic import Field

from src.agent.entity_resolution import ResolvedCompany, resolve_companies
from src.agent.methodology_intent import (
    classify_methodology_intent,
    infer_safety_intent,
    infer_user_expectation,
    legacy_methodology_intent,
    legacy_safety_intent,
    normalize_query_text,
)
from src.agent.query_ontology import (
    ALLOWED_ANALYSIS_METRICS,
    ALLOWED_METHODOLOGY_INTENTS,
    ALLOWED_SAFETY_INTENTS,
    DETERMINISTIC_DIMENSION_TERMS,
    SUPPORTED_DIMENSIONS,
    normalize_dimension_label,
    normalize_metric_label,
    normalize_safety_intent_label,
)
from src.agent.types import AgentDomainModel

AnalysisScope = Literal["single_company", "comparison", "meta", "unsupported", "unknown"]
MethodologyIntent = Literal[
    "overview",
    "risk",
    "cash_flow",
    "profitability",
    "revenue",
    "balance_sheet",
    "valuation",
    "comparison",
    "none",
]
UserExpectation = Literal["quick_answer", "deep_analysis", "recommendation_like", "diagnostic", "clarification"]
SafetyIntent = Literal["normal", "investment_advice_like", "prediction", "unsupported"]

_PROPOSAL_CONFIDENCE_FLOOR = 0.55


class QueryUnderstandingProposal(AgentDomainModel):
    company_mentions: list[str] = Field(default_factory=list)
    analysis_scope: str = "unknown"
    methodology_intent: str = "none"
    requested_dimensions: list[str] = Field(default_factory=list)
    requested_metrics: list[str] = Field(default_factory=list)
    user_expectation: str = "quick_answer"
    safety_intent: str = "normal"
    time_scope: dict[str, Any] = Field(default_factory=dict)
    ambiguity: bool = False
    needs_clarification: bool = False
    confidence: float = 0.0
    reasons: list[str] = Field(default_factory=list)


class QueryUnderstanding(AgentDomainModel):
    raw_query: str = ""
    normalized_query: str = ""
    companies: list[ResolvedCompany] = Field(default_factory=list)
    unresolved_mentions: list[str] = Field(default_factory=list)
    ambiguity: bool = False
    analysis_scope: AnalysisScope = "unknown"
    methodology_intent: MethodologyIntent = "none"
    legacy_methodology_intent: str = ""
    user_expectation: UserExpectation = "quick_answer"
    safety_intent: SafetyIntent = "normal"
    legacy_safety_intent: str = "normal"
    time_scope: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    needs_clarification: bool = False
    clarification_reason: str | None = None
    intent_source: str = "fallback_rules"
    intent_reasons: list[str] = Field(default_factory=list)
    requested_dimensions: list[str] = Field(default_factory=list)
    requested_metrics: list[str] = Field(default_factory=list)
    semantic_proposal: dict[str, Any] = Field(default_factory=dict)
    rule_methodology_intent: MethodologyIntent = "none"
    proposed_methodology_intent: str = ""
    proposal_validation_warnings: list[dict[str, Any]] = Field(default_factory=list)
    intent_conflict: bool = False


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _is_meta_query(normalized_query: str) -> bool:
    q = normalized_query
    compact = re.sub(r"[\s?？!！。,.，]+", "", q)
    zh_terms = ("你是谁", "你是什么", "你能做什么", "你可以做什么", "你的能力", "介绍一下你自己")
    en_terms = ("who are you", "what are you", "what can you do", "your capabilities")
    return any(term in compact for term in zh_terms) or any(term in q for term in en_terms)


def _has_explicit_time(normalized_query: str) -> bool:
    q = normalized_query
    return bool(
        re.search(r"(?:^|[^0-9])(20\d{2})(?:年|q[1-4]|[^0-9]|$)", q)
        or re.search(r"\b(?:fy|fiscal year)\s*20\d{2}\b", q)
        or re.search(r"\b20\d{2}\s*q[1-4]\b", q)
    )


def _analysis_scope(
    *,
    normalized_query: str,
    company_count: int,
    methodology_intent: str,
    safety_intent: str,
) -> AnalysisScope:
    if safety_intent in {"prediction", "unsupported"}:
        return "unsupported"
    if _is_meta_query(normalized_query):
        return "meta"
    if methodology_intent == "comparison" or company_count >= 2:
        return "comparison"
    if company_count == 1 and methodology_intent != "none":
        return "single_company"
    return "unknown"


def _needs_clarification(scope: str, methodology_intent: str, company_count: int, ambiguity: bool) -> tuple[bool, str | None]:
    if ambiguity and company_count <= 0:
        return True, "company_mention_ambiguous"
    if scope == "unknown" and methodology_intent != "none" and company_count <= 0:
        return True, "company_required_for_methodology_intent"
    return False, None


def _time_scope(scope: str, explicit_time: bool) -> dict[str, Any]:
    if scope == "single_company" and not explicit_time:
        return {
            "policy": "latest_available",
            "period_scope": "latest annual + latest quarterly",
            "fallback": "latest 4 quarters if annual unavailable",
            "is_explicit": False,
        }
    return {"policy": "", "period_scope": "", "fallback": "", "is_explicit": explicit_time}


def _string_list(value: Any, *, max_items: int = 12) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if text and text not in out:
            out.append(text)
        if len(out) >= max_items:
            break
    return out


def _proposal_payload(parsed: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(parsed, Mapping):
        return None
    raw = parsed.get("query_understanding_proposal")
    return raw if isinstance(raw, Mapping) else None


def _proposal_model(raw: Mapping[str, Any] | None) -> tuple[QueryUnderstandingProposal | None, dict[str, Any], list[dict[str, Any]]]:
    if raw is None:
        return None, {}, []
    try:
        proposal = QueryUnderstandingProposal(**dict(raw))
    except Exception as exc:  # pragma: no cover - defensive against malformed LLM JSON
        return None, dict(raw), [{"field": "query_understanding_proposal", "reason": "proposal_schema_invalid", "detail": str(exc)}]
    payload = proposal.model_dump(exclude_none=True)
    return proposal, payload, []


def _clamp_confidence(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    return round(max(0.0, min(raw, 1.0)), 3)


def _normalize_proposed_safety(value: Any) -> str:
    normalized = normalize_safety_intent_label(value)
    return normalized if normalized in ALLOWED_SAFETY_INTENTS else ""


def _more_conservative_safety(rule_safety: str, proposed_safety: str) -> str:
    if rule_safety in {"prediction", "unsupported"}:
        return rule_safety
    if proposed_safety in {"prediction", "unsupported"}:
        return proposed_safety
    if rule_safety == "investment_advice_like" or proposed_safety == "investment_advice_like":
        return "investment_advice_like"
    return "normal"


def _validated_dimensions(value: Any, warnings: list[dict[str, Any]]) -> list[str]:
    dimensions: list[str] = []
    for item in _string_list(value, max_items=8):
        normalized = normalize_dimension_label(item)
        if normalized in SUPPORTED_DIMENSIONS and normalized not in dimensions:
            dimensions.append(normalized)
        else:
            warnings.append({"field": "requested_dimensions", "value": item, "reason": "dimension_not_supported"})
    return dimensions


def _normalize_metric(value: Any) -> str:
    return normalize_metric_label(value)


def _validated_metrics(value: Any, warnings: list[dict[str, Any]]) -> list[str]:
    metrics: list[str] = []
    for item in _string_list(value, max_items=16):
        normalized = _normalize_metric(item)
        if normalized in ALLOWED_ANALYSIS_METRICS and normalized not in metrics:
            metrics.append(normalized)
        else:
            warnings.append({"field": "requested_metrics", "value": item, "reason": "metric_not_supported"})
    return metrics


def _explicit_requested_dimensions(normalized_query: str) -> list[str]:
    q = str(normalized_query or "").lower()
    matches: list[tuple[int, int, str]] = []
    for order, (dimension_id, terms) in enumerate(DETERMINISTIC_DIMENSION_TERMS):
        positions = [q.find(term) for term in terms if term and q.find(term) >= 0]
        if positions:
            matches.append((min(positions), order, dimension_id))
    return [dimension_id for _, _, dimension_id in sorted(matches)]


def _merge_dimensions(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    for group in groups:
        for item in group:
            dimension_id = str(item or "").strip()
            if dimension_id in SUPPORTED_DIMENSIONS and dimension_id not in merged:
                merged.append(dimension_id)
    return merged


def _direct_fact_forces_none(rule_reasons: list[str]) -> bool:
    return "direct_fact_metric_question" in set(rule_reasons or [])


def _proposal_intent_result(
    *,
    proposal: QueryUnderstandingProposal | None,
    semantic_payload: dict[str, Any],
    rule_methodology_intent: str,
    rule_reasons: list[str],
    rule_safety_intent: str,
    company_count: int,
    base_warnings: list[dict[str, Any]],
) -> dict[str, Any]:
    warnings = list(base_warnings)
    if proposal is None:
        return {
            "accepted": False,
            "methodology_intent": rule_methodology_intent,
            "safety_intent": rule_safety_intent,
            "source": "fallback_rules",
            "reasons": list(rule_reasons),
            "requested_dimensions": [],
            "requested_metrics": [],
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": "",
            "warnings": warnings,
            "intent_conflict": False,
        }

    confidence = _clamp_confidence(proposal.confidence)
    proposed_intent = str(proposal.methodology_intent or "").strip()
    proposed_safety = _normalize_proposed_safety(proposal.safety_intent)
    requested_dimensions = _validated_dimensions(proposal.requested_dimensions, warnings)
    requested_metrics = _validated_metrics(proposal.requested_metrics, warnings)
    if proposed_safety == "" and proposal.safety_intent:
        warnings.append({"field": "safety_intent", "value": proposal.safety_intent, "reason": "safety_intent_not_allowed"})

    if confidence < _PROPOSAL_CONFIDENCE_FLOOR:
        warnings.append({"field": "confidence", "value": confidence, "reason": "proposal_confidence_too_low"})
        return {
            "accepted": False,
            "methodology_intent": rule_methodology_intent,
            "safety_intent": rule_safety_intent,
            "source": "fallback_rules",
            "reasons": list(rule_reasons),
            "requested_dimensions": [],
            "requested_metrics": [],
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": proposed_intent,
            "warnings": warnings,
            "intent_conflict": bool(proposed_intent and proposed_intent != rule_methodology_intent),
        }
    if proposed_intent not in ALLOWED_METHODOLOGY_INTENTS:
        warnings.append({"field": "methodology_intent", "value": proposed_intent, "reason": "methodology_intent_not_allowed"})
        return {
            "accepted": False,
            "methodology_intent": rule_methodology_intent,
            "safety_intent": rule_safety_intent,
            "source": "fallback_rules",
            "reasons": list(rule_reasons),
            "requested_dimensions": [],
            "requested_metrics": [],
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": proposed_intent,
            "warnings": warnings,
            "intent_conflict": False,
        }

    final_safety = _more_conservative_safety(rule_safety_intent, proposed_safety or "normal")
    intent_conflict = proposed_intent != rule_methodology_intent
    if final_safety in {"prediction", "unsupported"}:
        if proposed_intent != "none":
            warnings.append({"field": "methodology_intent", "value": proposed_intent, "reason": "safety_forces_none"})
        return {
            "accepted": True,
            "methodology_intent": "none",
            "safety_intent": final_safety,
            "source": "program_safety_override",
            "reasons": list(proposal.reasons or []) or list(rule_reasons),
            "requested_dimensions": requested_dimensions,
            "requested_metrics": [],
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": proposed_intent,
            "warnings": warnings,
            "intent_conflict": intent_conflict,
        }
    if _direct_fact_forces_none(rule_reasons) and proposed_intent != "none":
        warnings.append({"field": "methodology_intent", "value": proposed_intent, "reason": "direct_fact_forces_none"})
        return {
            "accepted": False,
            "methodology_intent": rule_methodology_intent,
            "safety_intent": final_safety,
            "source": "fallback_rules",
            "reasons": list(rule_reasons),
            "requested_dimensions": [],
            "requested_metrics": [],
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": proposed_intent,
            "warnings": warnings,
            "intent_conflict": intent_conflict,
        }
    if company_count >= 2 and proposed_intent not in {"comparison", "none"}:
        warnings.append({"field": "methodology_intent", "value": proposed_intent, "reason": "multi_company_forces_comparison"})
        return {
            "accepted": True,
            "methodology_intent": "comparison",
            "safety_intent": final_safety,
            "source": "program_validated_override",
            "reasons": list(proposal.reasons or []) or list(rule_reasons),
            "requested_dimensions": requested_dimensions,
            "requested_metrics": requested_metrics,
            "semantic_proposal": semantic_payload,
            "proposed_methodology_intent": proposed_intent,
            "warnings": warnings,
            "intent_conflict": intent_conflict,
        }

    return {
        "accepted": True,
        "methodology_intent": proposed_intent,
        "safety_intent": final_safety,
        "source": "semantic_proposal_validated",
        "reasons": list(proposal.reasons or []) or list(rule_reasons),
        "requested_dimensions": requested_dimensions,
        "requested_metrics": requested_metrics,
        "semantic_proposal": semantic_payload,
        "proposed_methodology_intent": proposed_intent,
        "warnings": warnings,
        "intent_conflict": intent_conflict,
    }


def _coerce_parsed(parsed: Mapping[str, Any] | None) -> tuple[list[Any], Any | None]:
    if not isinstance(parsed, Mapping):
        return [], None
    companies = []
    for values in (
        parsed.get("companies", []),
        (_proposal_payload(parsed) or {}).get("company_mentions", []),
        (parsed.get("analysis_plan") if isinstance(parsed.get("analysis_plan"), Mapping) else {}).get("companies", []),
    ):
        for item in _string_list(values):
            if item not in companies:
                companies.append(item)
    return companies, parsed.get("comparison_target")


def build_query_understanding(
    raw_query: str,
    optional_llm_client: Any | None = None,
    *,
    parsed: Mapping[str, Any] | None = None,
) -> QueryUnderstanding:
    normalized = normalize_query_text(raw_query)
    parsed_companies, comparison_target = _coerce_parsed(parsed)
    entity_result = resolve_companies(
        raw_query,
        parsed_companies=parsed_companies,
        comparison_target=comparison_target,
    )
    rule_intent_result = classify_methodology_intent(
        normalized,
        entity_result.resolved_companies,
    )
    proposal_model, semantic_proposal, proposal_warnings = _proposal_model(_proposal_payload(parsed))
    proposal_result = _proposal_intent_result(
        proposal=proposal_model,
        semantic_payload=semantic_proposal,
        rule_methodology_intent=rule_intent_result.methodology_intent,
        rule_reasons=list(rule_intent_result.reasons or []),
        rule_safety_intent=infer_safety_intent(normalized),
        company_count=len({company.ticker for company in entity_result.resolved_companies}),
        base_warnings=proposal_warnings,
    )
    if proposal_result["accepted"]:
        methodology_intent = proposal_result["methodology_intent"]
        safety_intent = proposal_result["safety_intent"]
        intent_source = proposal_result["source"]
        intent_reasons = proposal_result["reasons"]
        requested_dimensions = proposal_result["requested_dimensions"]
        requested_metrics = proposal_result["requested_metrics"]
        intent_conflict = bool(proposal_result["intent_conflict"])
    elif optional_llm_client is not None:
        intent_result = classify_methodology_intent(
            normalized,
            entity_result.resolved_companies,
            optional_llm_client=optional_llm_client,
        )
        methodology_intent = intent_result.methodology_intent
        safety_intent = infer_safety_intent(normalized)
        intent_source = intent_result.source
        intent_reasons = list(intent_result.reasons or [])
        requested_dimensions = []
        requested_metrics = []
        intent_conflict = False
    else:
        methodology_intent = rule_intent_result.methodology_intent
        safety_intent = proposal_result["safety_intent"]
        intent_source = proposal_result["source"]
        intent_reasons = proposal_result["reasons"]
        requested_dimensions = proposal_result["requested_dimensions"]
        requested_metrics = proposal_result["requested_metrics"]
        intent_conflict = bool(proposal_result["intent_conflict"])
    if safety_intent in {"prediction", "unsupported"} or _direct_fact_forces_none(list(rule_intent_result.reasons or [])):
        requested_dimensions = []
        requested_metrics = []
    else:
        requested_dimensions = _merge_dimensions(
            list(requested_dimensions or []),
            _explicit_requested_dimensions(normalized),
        )
    company_count = len({company.ticker for company in entity_result.resolved_companies})
    scope = _analysis_scope(
        normalized_query=normalized,
        company_count=company_count,
        methodology_intent=methodology_intent,
        safety_intent=safety_intent,
    )
    needs_clarification, clarification_reason = _needs_clarification(
        scope,
        methodology_intent,
        company_count,
        entity_result.ambiguity,
    )
    expectation = infer_user_expectation(
        normalized,
        methodology_intent=methodology_intent,
        safety_intent=safety_intent,
        needs_clarification=needs_clarification,
    )
    intent_confidence = (
        _clamp_confidence(proposal_model.confidence)
        if proposal_model is not None and proposal_result["accepted"]
        else rule_intent_result.confidence
    )
    confidence_parts = [intent_confidence]
    if entity_result.resolved_companies:
        confidence_parts.append(entity_result.confidence)
    if needs_clarification:
        confidence_parts.append(0.35)
    confidence = round(sum(confidence_parts) / len(confidence_parts), 3) if confidence_parts else 0.0
    return QueryUnderstanding(
        raw_query=str(raw_query or ""),
        normalized_query=normalized,
        companies=entity_result.resolved_companies,
        unresolved_mentions=entity_result.unresolved_mentions,
        ambiguity=entity_result.ambiguity,
        analysis_scope=scope,
        methodology_intent=methodology_intent,
        legacy_methodology_intent=legacy_methodology_intent(methodology_intent),
        user_expectation=expectation,
        safety_intent=safety_intent,
        legacy_safety_intent=legacy_safety_intent(safety_intent),
        time_scope=_time_scope(scope, _has_explicit_time(normalized)),
        confidence=confidence,
        needs_clarification=needs_clarification,
        clarification_reason=clarification_reason,
        intent_source=intent_source,
        intent_reasons=list(intent_reasons or []),
        requested_dimensions=requested_dimensions,
        requested_metrics=requested_metrics,
        semantic_proposal=proposal_result["semantic_proposal"],
        rule_methodology_intent=rule_intent_result.methodology_intent,
        proposed_methodology_intent=str(proposal_result["proposed_methodology_intent"] or ""),
        proposal_validation_warnings=list(proposal_result["warnings"] or []),
        intent_conflict=intent_conflict,
    )


def query_understanding_summary(understanding: QueryUnderstanding) -> dict[str, Any]:
    return {
        "normalized_query": understanding.normalized_query,
        "companies": [company.model_dump(exclude_none=True) for company in understanding.companies],
        "unresolved_mentions": list(understanding.unresolved_mentions),
        "ambiguity": understanding.ambiguity,
        "analysis_scope": understanding.analysis_scope,
        "methodology_intent": understanding.methodology_intent,
        "legacy_methodology_intent": understanding.legacy_methodology_intent,
        "user_expectation": understanding.user_expectation,
        "safety_intent": understanding.safety_intent,
        "legacy_safety_intent": understanding.legacy_safety_intent,
        "time_scope": dict(understanding.time_scope or {}),
        "confidence": understanding.confidence,
        "needs_clarification": understanding.needs_clarification,
        "clarification_reason": understanding.clarification_reason,
        "intent_source": understanding.intent_source,
        "intent_reasons": list(understanding.intent_reasons),
        "requested_dimensions": list(understanding.requested_dimensions),
        "requested_metrics": list(understanding.requested_metrics),
        "semantic_proposal": dict(understanding.semantic_proposal or {}),
        "rule_methodology_intent": understanding.rule_methodology_intent,
        "proposed_methodology_intent": understanding.proposed_methodology_intent,
        "proposal_validation_warnings": list(understanding.proposal_validation_warnings),
        "intent_conflict": understanding.intent_conflict,
    }
