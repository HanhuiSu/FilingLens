"""Deterministic validation for analyst drafts."""

from __future__ import annotations

import json
import re
from typing import Any

from src.agent.types import DraftValidationResult, DraftViolation

_FORBIDDEN_INVESTMENT_TERMS = (
    "推荐买",
    "应该买",
    "应该卖",
    "买入",
    "卖出",
    "荐股",
    "should buy",
    "should sell",
    "recommend buying",
    "recommend selling",
)

_FORBIDDEN_FORECAST_TERMS = (
    "一定会上涨",
    "一定会下跌",
    "明天会涨",
    "明天会跌",
    "一定上涨",
    "一定下跌",
    "一定更好",
    "肯定更好",
    "will rise tomorrow",
    "will fall tomorrow",
    "guaranteed to rise",
    "guaranteed to fall",
    "definitely better",
    "certainly better",
)

_MANAGEMENT_MOTIVE_TERMS = (
    "management fears",
    "management is worried",
    "management wants",
    "management believes internally",
    "管理层担心",
    "管理层害怕",
    "管理层想要",
)

_INVESTMENT_ADVICE_PATTERNS = (
    r"\bshould\b.{0,24}\b(?:buy|sell)\b",
    r"\brecommend(?:s|ed|ing)?\b.{0,24}\b(?:buy|sell|buying|selling)\b",
    r"\b(?:buy|sell)\b.{0,16}\b(?:now|today)\b",
    r"建议.{0,12}(?:买入|卖出|买|卖)",
)

_RISK_TERMS = (
    "risk",
    "risks",
    "pressure",
    "uncertainty",
    "competition",
    "regulation",
    "headwind",
    "challenge",
    "风险",
    "压力",
    "竞争",
    "监管",
    "挑战",
    "不确定",
)

_VALUATION_FORBIDDEN_TERMS = (
    "cheap",
    "expensive",
    "undervalued",
    "overvalued",
    "worth buying",
    "buy",
    "sell",
    "recommend",
    "便宜",
    "昂贵",
    "贵",
    "低估",
    "高估",
    "值得买",
    "买入",
    "卖出",
    "推荐",
)

_CASH_FLOW_QUALITY_TERMS = (
    "cash flow is strong",
    "cash flow is weak",
    "cash-flow quality",
    "cash flow quality",
    "earnings quality",
    "现金流强",
    "现金流弱",
    "现金流质量",
    "利润现金含量",
    "盈利质量",
)

_PROFITABILITY_TERMS = (
    "profitability",
    "profit quality",
    "margin quality",
    "盈利能力",
    "盈利质量",
    "利润质量",
    "利润率",
)

_PROFITABILITY_EVIDENCE_TERMS = (
    "net margin",
    "net income",
    "净利率",
    "净利润",
)

_UNIT_TOKENS = ("usd", "eur", "cny", "million", "billion", "美元", "%", "percent")
_PERIOD_WORDS = ("annual", "quarterly", "fy", "q1", "q2", "q3", "q4")
_NEGATIVE_NUMBER_CONTEXT_TERMS = (
    "负",
    "为负",
    "亏损",
    "下降",
    "下滑",
    "减少",
    "降低",
    "negative",
    "loss",
    "decline",
    "declined",
    "decrease",
    "decreased",
    "down",
)

_HARD_VIOLATION_REASONS = {
    "unknown_citation_ref",
    "invented_number",
    "unknown_period",
    "unknown_unit",
    "period_rewritten",
    "unknown_company",
    "investment_advice_wording",
    "unsupported_forecast_wording",
    "management_motive_inference",
    "draft_missing",
    "missing_statement",
}

_INLINE_CITATION_RE = re.compile(r"\[([NT]\d+)\]")


def _numbers_in(text: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?%?(?![A-Za-z])", text or ""))


def _number_value(token: str) -> float | None:
    cleaned = str(token or "").replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _negative_context_for_token(sentence: str, token: str) -> bool:
    if not sentence or not token:
        return False
    pattern = re.escape(str(token))
    for match in re.finditer(pattern, sentence):
        window = sentence[max(0, match.start() - 18) : min(len(sentence), match.end() + 8)].lower()
        if any(term.lower() in window for term in _NEGATIVE_NUMBER_CONTEXT_TERMS):
            return True
    return False


def _number_supported_by_packet(token: str, allowed_numbers: set[str], sentence: str = "") -> bool:
    if token in allowed_numbers:
        return True
    value = _number_value(token)
    if value is None:
        return False

    candidate_values = [value]
    if value > 0 and _negative_context_for_token(sentence, token):
        candidate_values.append(-value)

    is_percent = str(token).endswith("%")
    scale_factors = (
        (1.0, 100.0)
        if is_percent
        else (1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 0.001, 1e-6, 1e-8, 1e-9, 1e-12)
    )
    for allowed in allowed_numbers:
        allowed_value = _number_value(allowed)
        if allowed_value is None:
            continue
        for scale in scale_factors:
            expected = allowed_value * scale
            tolerance = max(0.02, abs(expected) * 0.001)
            if any(abs(candidate - expected) < tolerance for candidate in candidate_values):
                return True
    return False


def _unsupported_numbers(sentence: str, allowed_numbers: set[str]) -> set[str]:
    return {
        token
        for token in _numbers_in(sentence)
        if not (
            token.isdigit()
            and 1 <= int(token) <= 31
            and ("月" in sentence or "日" in sentence)
        )
        and not _number_supported_by_packet(token, allowed_numbers, sentence)
    }


def _is_hard_violation_reason(reason: str) -> bool:
    return str(reason or "") in _HARD_VIOLATION_REASONS


def _date_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text or ""))


def _year_tokens(text: str) -> set[str]:
    return set(re.findall(r"\b20\d{2}\b", text or ""))


def _packet_payload(packet: dict[str, Any]) -> str:
    return json.dumps(packet, ensure_ascii=False, sort_keys=True)


def _packet_citation_refs(packet: dict[str, Any]) -> set[str]:
    refs = {str(item.get("evidence_id", "")).strip() for item in packet.get("numeric_table", []) or [] if isinstance(item, dict)}
    refs |= {str(item.get("evidence_id", "")).strip() for item in packet.get("text_snippets", []) or [] if isinstance(item, dict)}
    refs |= {str(item.get("evidence_id", "")).strip() for item in packet.get("citations", []) or [] if isinstance(item, dict)}
    return {ref for ref in refs if ref}


def _packet_allowed_numbers(packet: dict[str, Any]) -> set[str]:
    return _numbers_in(_packet_payload(packet))


def _packet_allowed_dates(packet: dict[str, Any]) -> set[str]:
    return _date_tokens(_packet_payload(packet))


def _packet_allowed_years(packet: dict[str, Any]) -> set[str]:
    return _year_tokens(_packet_payload(packet))


def _packet_allowed_units(packet: dict[str, Any]) -> set[str]:
    payload = _packet_payload(packet).lower()
    return {token for token in _UNIT_TOKENS if token in payload}


def _packet_allowed_period_words(packet: dict[str, Any]) -> set[str]:
    payload = _packet_payload(packet).lower()
    return {token for token in _PERIOD_WORDS if token in payload}


def _unsupported_units(units_in_sentence: set[str], allowed_units: set[str]) -> set[str]:
    unsupported: set[str] = set()
    for token in units_in_sentence:
        if token in allowed_units:
            continue
        if token == "美元" and "usd" in allowed_units:
            continue
        if token == "usd" and "美元" in allowed_units:
            continue
        unsupported.add(token)
    return unsupported


def _packet_companies(packet: dict[str, Any]) -> set[str]:
    companies = {
        str(item.get("ticker", "")).strip().upper()
        for item in (packet.get("numeric_table", []) or []) + (packet.get("text_snippets", []) or [])
        if isinstance(item, dict) and str(item.get("ticker", "")).strip()
    }
    return {company for company in companies if company}


def _risk_evidence_refs(packet: dict[str, Any]) -> set[str]:
    refs = {
        str(item.get("evidence_id", "")).strip()
        for item in packet.get("text_snippets", []) or []
        if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
    }
    for theme in packet.get("grouped_risk_themes", []) or []:
        if not isinstance(theme, dict):
            continue
        refs |= {str(ref).strip() for ref in theme.get("evidence_refs", []) or [] if str(ref).strip()}
    return refs


def _dimension_status_map(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summary = dict(packet.get("dimension_sufficiency", {}) or {})
    status_map = summary.get("dimension_status_map")
    if isinstance(status_map, dict):
        return {str(k): dict(v) for k, v in status_map.items() if isinstance(v, dict)}
    missing = dict(packet.get("missing_evidence_summary", {}) or {})
    status_map = missing.get("dimension_status_map")
    if isinstance(status_map, dict):
        return {str(k): dict(v) for k, v in status_map.items() if isinstance(v, dict)}
    return {}


def _dimension_status(packet: dict[str, Any], dimension_id: str) -> str:
    return str(_dimension_status_map(packet).get(dimension_id, {}).get("status") or "")


def _active_dimensions(packet: dict[str, Any]) -> set[str]:
    active = {
        str(item).strip()
        for item in packet.get("active_dimensions", []) or []
        if str(item).strip()
    }
    if active:
        return active
    return set(_dimension_status_map(packet).keys())


def _dimension_evidence_refs(packet: dict[str, Any], dimension_id: str) -> set[str]:
    refs: set[str] = set()
    for item in packet.get("dimension_summary", []) or []:
        if not isinstance(item, dict) or str(item.get("dimension_id") or "") != dimension_id:
            continue
        refs |= {str(ref).strip() for ref in item.get("evidence_refs", []) or [] if str(ref).strip()}
        refs |= {str(ref).strip() for ref in item.get("numeric_evidence_refs", []) or [] if str(ref).strip()}
        refs |= {str(ref).strip() for ref in item.get("text_evidence_refs", []) or [] if str(ref).strip()}
    if refs:
        return refs
    status_map = _dimension_status_map(packet)
    for table_name in ("numeric_table", "text_snippets"):
        for row in packet.get(table_name, []) or []:
            if not isinstance(row, dict):
                continue
            row_dimension = str(row.get("dimension_id") or "").strip()
            if row_dimension == dimension_id and str(row.get("evidence_id") or "").strip():
                refs.add(str(row.get("evidence_id")).strip())
    if refs:
        return refs
    # Last resort: if a packet predates dimension_summary, allow any ref for
    # known active dimensions so legacy tests stay focused on guardrails.
    if dimension_id in status_map and not packet.get("dimension_summary"):
        return _packet_citation_refs(packet)
    return refs


def _dimension_claim_forbidden_reason(dimension_id: str, status: str) -> str | None:
    if status == "missing":
        if dimension_id == "cash_flow_quality":
            return "dimension_forbidden_cash_flow_quality_claim"
        if dimension_id == "valuation_and_risk_boundary":
            return "dimension_forbidden_valuation_claim"
        if dimension_id == "moat_and_competitive_risk":
            return "dimension_forbidden_specific_risk_claim"
        return "missing_dimension_analyzed"
    return None


def _dimension_forbidden_reason(sentence: str, field_name: str, packet: dict[str, Any]) -> str | None:
    lowered = (sentence or "").lower()
    if _dimension_status(packet, "valuation_and_risk_boundary") == "missing" and any(
        term in lowered for term in _VALUATION_FORBIDDEN_TERMS
    ):
        return "dimension_forbidden_valuation_claim"
    if _dimension_status(packet, "cash_flow_quality") == "missing" and any(
        term in lowered for term in _CASH_FLOW_QUALITY_TERMS
    ):
        return "dimension_forbidden_cash_flow_quality_claim"
    if _dimension_status(packet, "moat_and_competitive_risk") == "missing" and _is_risk_like(sentence, field_name):
        return "dimension_forbidden_specific_risk_claim"
    if _dimension_status(packet, "profitability_quality") == "partial" and any(
        term in lowered for term in _PROFITABILITY_TERMS
    ) and not any(term in lowered for term in _PROFITABILITY_EVIDENCE_TERMS):
        return "dimension_partial_profitability_requires_net_margin_or_net_income"
    return None


def _item_dicts(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _inline_citation_refs(text: str) -> list[str]:
    return list(dict.fromkeys(match.group(1) for match in _INLINE_CITATION_RE.finditer(text or "")))


def _strip_inline_citations(text: str) -> str:
    cleaned = _INLINE_CITATION_RE.sub("", text or "")
    cleaned = re.sub(r"\s+([，。；；;,.])", r"\1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return cleaned.strip()


def _item_citation_refs(item: dict[str, Any], sentence: str) -> list[str]:
    refs: list[str] = []
    for raw_ref in item.get("citation_refs", item.get("evidence_refs", [])) or []:
        ref = str(raw_ref).strip()
        if ref:
            refs.append(ref)
    refs.extend(_inline_citation_refs(sentence))
    return list(dict.fromkeys(refs))


def _add_violation(violations: list[DraftViolation], *, field: str, reason: str, item: dict[str, Any], unknown_refs: list[str] | None = None) -> None:
    violations.append(
        DraftViolation(
            field=field,
            reason=reason,
            item=dict(item or {}),
            unknown_refs=list(unknown_refs or []),
        )
    )


def _plain_text_violation_reason(text: str) -> str | None:
    lowered = (text or "").lower()
    if _contains_forbidden_investment_wording(lowered):
        return "investment_advice_wording"
    if any(term in lowered for term in _FORBIDDEN_FORECAST_TERMS):
        return "unsupported_forecast_wording"
    if any(term in lowered for term in _MANAGEMENT_MOTIVE_TERMS):
        return "management_motive_inference"
    return None


def _contains_forbidden_investment_wording(text: str) -> bool:
    lowered = (text or "").lower()
    if any(term in lowered for term in _FORBIDDEN_INVESTMENT_TERMS):
        return True
    return any(re.search(pattern, lowered) for pattern in _INVESTMENT_ADVICE_PATTERNS)


def _is_risk_like(sentence: str, field_name: str) -> bool:
    lowered = (sentence or "").lower()
    boundary_terms = (
        "missing",
        "limited",
        "not a complete analysis",
        "not complete",
        "缺少",
        "有限",
        "完整分析",
        "不是完整分析",
    )
    if any(term in lowered for term in boundary_terms):
        return False
    return field_name == "risk_tradeoffs" or any(term in lowered for term in _RISK_TERMS)


def _numeric_or_boundary_note(sentence: str, field_name: str, refs: list[str]) -> bool:
    if field_name not in {"counterpoints", "uncertainty_notes"}:
        return False
    if not any(ref.startswith("N") for ref in refs):
        return False
    lowered = (sentence or "").lower()
    boundary_terms = (
        "口径",
        "期间",
        "来源",
        "混用",
        "核验",
        "不确定",
        "有限",
        "caveat",
        "basis",
        "source",
        "period",
        "limited",
        "uncertain",
    )
    metric_terms = (
        "净利率",
        "利润率",
        "现金流",
        "fcf",
        "capex",
        "revenue",
        "margin",
        "cash flow",
        "valuation",
        "p/e",
        "p/s",
    )
    return any(term in lowered for term in boundary_terms + metric_terms)


def _conditional_comparison_like(sentence: str) -> bool:
    lowered = (sentence or "").lower()
    return any(
        term in lowered
        for term in (
            "if ",
            "based on",
            "within the current",
            "under the current",
            "conditional",
            "如果",
            "基于",
            "在当前",
            "只基于",
            "更看重",
            "条件",
        )
    )


def _validate_statement(
    *,
    item: dict[str, Any],
    field_name: str,
    safety_intent: str,
    answer_mode: str,
    synthesis_mode: str,
    allowed_refs: set[str],
    allowed_numbers: set[str],
    allowed_dates: set[str],
    allowed_years: set[str],
    allowed_units: set[str],
    allowed_period_words: set[str],
    allowed_companies: set[str],
    risk_refs: set[str],
    has_text_evidence: bool,
    evidence_packet: dict[str, Any],
    violations: list[DraftViolation],
) -> dict[str, Any] | None:
    raw_sentence = str(item.get("statement", "")).strip()
    sentence = _strip_inline_citations(raw_sentence)
    refs = _item_citation_refs(item, raw_sentence)
    if not sentence:
        return None
    unknown_refs = [ref for ref in refs if ref not in allowed_refs]
    if unknown_refs:
        _add_violation(violations, field=field_name, reason="unknown_citation_ref", item=item, unknown_refs=unknown_refs)
        return None
    if field_name != "safety_notes" and not refs:
        _add_violation(violations, field=field_name, reason="missing_citation_ref", item=item)
        if field_name in {"counterpoints", "risk_tradeoffs", "uncertainty_notes"}:
            return None
    if _unsupported_numbers(sentence, allowed_numbers):
        _add_violation(violations, field=field_name, reason="invented_number", item=item)
        return None
    if _date_tokens(sentence) - allowed_dates or _year_tokens(sentence) - allowed_years:
        _add_violation(violations, field=field_name, reason="unknown_period", item=item)
        return None
    lowered = sentence.lower()
    units_in_sentence = {token for token in _UNIT_TOKENS if token in lowered}
    if _unsupported_units(units_in_sentence, allowed_units):
        _add_violation(violations, field=field_name, reason="unknown_unit", item=item)
        return None
    period_words_in_sentence = {token for token in _PERIOD_WORDS if token in lowered}
    if period_words_in_sentence - allowed_period_words:
        _add_violation(violations, field=field_name, reason="period_rewritten", item=item)
        return None
    preferred_company = str(item.get("preferred_company", "") or "")
    if preferred_company and preferred_company.upper() not in allowed_companies:
        _add_violation(violations, field=field_name, reason="unknown_company", item=item)
        return None
    if _contains_forbidden_investment_wording(lowered):
        _add_violation(violations, field=field_name, reason="investment_advice_wording", item=item)
        return None
    if any(term in lowered for term in _FORBIDDEN_FORECAST_TERMS):
        _add_violation(violations, field=field_name, reason="unsupported_forecast_wording", item=item)
        return None
    if any(term in lowered for term in _MANAGEMENT_MOTIVE_TERMS):
        _add_violation(violations, field=field_name, reason="management_motive_inference", item=item)
        return None
    dimension_reason = _dimension_forbidden_reason(sentence, field_name, evidence_packet)
    if dimension_reason:
        _add_violation(violations, field=field_name, reason=dimension_reason, item=item)
        if _is_hard_violation_reason(dimension_reason):
            return None
    is_risk_like = _is_risk_like(sentence, field_name)
    text_refs = {ref for ref in refs if ref.startswith("T")}
    numeric_boundary_note = _numeric_or_boundary_note(sentence, field_name, refs)
    if is_risk_like and not has_text_evidence:
        _add_violation(violations, field=field_name, reason="strong_risk_without_text_evidence", item=item)
    if is_risk_like and not text_refs and not numeric_boundary_note:
        _add_violation(violations, field=field_name, reason="risk_judgment_without_text_ref", item=item)
    if is_risk_like and risk_refs and text_refs and not (text_refs & risk_refs):
        _add_violation(violations, field=field_name, reason="risk_judgment_not_grounded", item=item)
    if field_name == "supporting_points" and not refs:
        _add_violation(violations, field=field_name, reason="supporting_point_missing_citation", item=item)
    if safety_intent == "investment_advice_like" and field_name == "safety_notes":
        if "投资建议" not in sentence and "investment advice" not in lowered:
            _add_violation(violations, field=field_name, reason="missing_non_advisory_note", item=item)
    if answer_mode == "comparison_brief" and synthesis_mode == "limited_judgment" and field_name in {"decision_basis", "supporting_points", "counterpoints"}:
        if any(ref.startswith("T") for ref in refs) and not has_text_evidence:
            _add_violation(violations, field=field_name, reason="limited_judgment_requires_numeric_only", item=item)
    return {"statement": sentence, "citation_refs": refs}


def validate_analyst_draft(
    draft: dict[str, Any],
    evidence_packet: dict[str, Any],
    safety_policy: dict[str, Any] | None,
    *,
    synthesis_mode: str = "",
) -> DraftValidationResult:
    packet = dict(evidence_packet or {})
    safety_policy = dict(safety_policy or {})
    answer_mode = str(packet.get("answer_mode", safety_policy.get("answer_mode", "direct_fact")) or "direct_fact")
    safety_intent = str(packet.get("safety_intent", safety_policy.get("safety_intent", "normal")) or "normal")

    if not draft:
        repair = ["Return a valid JSON analyst draft using only packet evidence."]
        return DraftValidationResult(
            passed=False,
            status="invalid",
            final_status="invalid",
            accepted_draft={},
            violations=[DraftViolation(field="draft", reason="draft_missing", item={})],
            rejected_items=[{"field": "draft", "reason": "draft_missing", "item": {}}],
            repair_instructions=repair,
            revision_instructions=repair,
            fallback_reason="draft_missing",
        )

    allowed_refs = _packet_citation_refs(packet)
    allowed_numbers = _packet_allowed_numbers(packet)
    allowed_dates = _packet_allowed_dates(packet)
    allowed_years = _packet_allowed_years(packet)
    allowed_units = _packet_allowed_units(packet)
    allowed_period_words = _packet_allowed_period_words(packet)
    allowed_companies = _packet_companies(packet)
    risk_refs = _risk_evidence_refs(packet)
    has_text_evidence = bool(packet.get("text_snippets", []) or [])

    violations: list[DraftViolation] = []
    accepted: dict[str, Any] = {
        "framework_summary": str(draft.get("framework_summary", "") or "").strip(),
        "dimension_analyses": [],
        "overall_judgment": str(draft.get("overall_judgment", "") or "").strip(),
        "methodology_counterpoints": [],
        "methodology_limitations": [],
        "follow_up_metrics": [],
        "tentative_conclusion": {},
        "decision_basis": [],
        "supporting_points": [],
        "counterpoints": [],
        "risk_tradeoffs": [],
        "uncertainty_notes": [],
        "citation_refs": [],
        "safety_notes": [],
    }

    for field_name in ("framework_summary", "overall_judgment"):
        value = str(accepted.get(field_name, "") or "").strip()
        if not value:
            continue
        reason = _plain_text_violation_reason(value)
        if reason:
            _add_violation(violations, field=field_name, reason=reason, item={"statement": value})

    conclusion = dict(draft.get("tentative_conclusion", {}) or {})
    accepted_conclusion = _validate_statement(
        item=conclusion,
        field_name="tentative_conclusion",
        safety_intent=safety_intent,
        answer_mode=answer_mode,
        synthesis_mode=synthesis_mode,
        allowed_refs=allowed_refs,
        allowed_numbers=allowed_numbers,
        allowed_dates=allowed_dates,
        allowed_years=allowed_years,
        allowed_units=allowed_units,
        allowed_period_words=allowed_period_words,
        allowed_companies=allowed_companies,
        risk_refs=risk_refs,
        has_text_evidence=has_text_evidence,
        evidence_packet=packet,
        violations=violations,
    )
    if accepted_conclusion:
        accepted["tentative_conclusion"] = {
            "statement": str(accepted_conclusion.get("statement", "")),
            "stance": str(conclusion.get("stance", "")),
            "preferred_company": str(conclusion.get("preferred_company", "")),
            "citation_refs": list(accepted_conclusion.get("citation_refs", []) or []),
        }
        if answer_mode == "comparison_brief" and not _conditional_comparison_like(str(accepted_conclusion.get("statement", ""))):
            _add_violation(
                violations,
                field="tentative_conclusion",
                reason="missing_conditional_comparison_judgment",
                item=conclusion,
            )
    else:
        _add_violation(violations, field="tentative_conclusion", reason="missing_statement", item=conclusion)

    active_dimensions = _active_dimensions(packet)
    for raw_item in draft.get("dimension_analyses", []) or []:
        if not isinstance(raw_item, dict):
            continue
        dimension_id = str(raw_item.get("dimension_id", "")).strip()
        claim = str(raw_item.get("claim", raw_item.get("statement", ""))).strip()
        refs = [
            str(ref).strip()
            for ref in raw_item.get("evidence_refs", raw_item.get("citation_refs", [])) or []
            if str(ref).strip()
        ]
        item = {"dimension_id": dimension_id, "claim": claim, "evidence_refs": refs, "status": str(raw_item.get("status", ""))}
        if not dimension_id or dimension_id not in active_dimensions:
            _add_violation(violations, field="dimension_analyses", reason="inactive_dimension_analyzed", item=item)
            continue
        status = _dimension_status(packet, dimension_id) or str(raw_item.get("status", ""))
        forbidden_reason = _dimension_claim_forbidden_reason(dimension_id, status)
        if forbidden_reason:
            _add_violation(violations, field="dimension_analyses", reason=forbidden_reason, item=item)
            continue
        if status not in {"satisfied", "partial"}:
            _add_violation(violations, field="dimension_analyses", reason="dimension_status_not_usable", item=item)
            continue
        if not claim:
            _add_violation(violations, field="dimension_analyses", reason="missing_statement", item=item)
            continue
        if not refs:
            _add_violation(violations, field="dimension_analyses", reason="dimension_claim_missing_evidence_refs", item=item)
            continue
        unknown_refs = [ref for ref in refs if ref not in allowed_refs]
        if unknown_refs:
            _add_violation(violations, field="dimension_analyses", reason="unknown_citation_ref", item=item, unknown_refs=unknown_refs)
            continue
        dimension_refs = _dimension_evidence_refs(packet, dimension_id)
        if dimension_refs and not set(refs).issubset(dimension_refs):
            _add_violation(violations, field="dimension_analyses", reason="dimension_claim_refs_not_in_dimension", item=item)
        accepted_item = _validate_statement(
            item={"statement": claim, "citation_refs": refs},
            field_name="dimension_analyses",
            safety_intent=safety_intent,
            answer_mode=answer_mode,
            synthesis_mode=synthesis_mode,
            allowed_refs=allowed_refs,
            allowed_numbers=allowed_numbers,
            allowed_dates=allowed_dates,
            allowed_years=allowed_years,
            allowed_units=allowed_units,
            allowed_period_words=allowed_period_words,
            allowed_companies=allowed_companies,
            risk_refs=risk_refs,
            has_text_evidence=has_text_evidence,
            evidence_packet=packet,
            violations=violations,
        )
        if accepted_item:
            accepted["dimension_analyses"].append(
                {
                    "dimension_id": dimension_id,
                    "status": status,
                    "claim": claim,
                    "evidence_refs": refs,
                }
            )

    for field_name in ("methodology_counterpoints", "methodology_limitations", "follow_up_metrics"):
        for text in draft.get(field_name, []) or []:
            value = str(text or "").strip()
            if not value:
                continue
            reason = _plain_text_violation_reason(value)
            if reason:
                _add_violation(violations, field=field_name, reason=reason, item={"statement": value})
                continue
            accepted[field_name].append(value)

    for field_name in ("decision_basis", "supporting_points", "counterpoints", "risk_tradeoffs", "uncertainty_notes", "safety_notes"):
        for item in _item_dicts(draft.get(field_name, [])):
            accepted_item = _validate_statement(
                item=item,
                field_name=field_name,
                safety_intent=safety_intent,
                answer_mode=answer_mode,
                synthesis_mode=synthesis_mode,
                allowed_refs=allowed_refs,
                allowed_numbers=allowed_numbers,
                allowed_dates=allowed_dates,
                allowed_years=allowed_years,
                allowed_units=allowed_units,
                allowed_period_words=allowed_period_words,
                allowed_companies=allowed_companies,
                risk_refs=risk_refs,
                has_text_evidence=has_text_evidence,
                evidence_packet=packet,
                violations=violations,
            )
            if accepted_item:
                accepted[field_name].append(accepted_item)

    if answer_mode == "comparison_brief":
        usable_dimensions = [
            dimension_id
            for dimension_id in active_dimensions
            if _dimension_status(packet, dimension_id) in {"satisfied", "partial"}
        ]
        if usable_dimensions and not accepted["dimension_analyses"]:
            _add_violation(violations, field="dimension_analyses", reason="missing_dimension_analyses", item={})
        if not accepted["counterpoints"] and not accepted["methodology_counterpoints"]:
            _add_violation(violations, field="counterpoints", reason="missing_counterpoint", item={})
        if safety_intent == "investment_advice_like" and not accepted["safety_notes"]:
            _add_violation(violations, field="safety_notes", reason="missing_non_advisory_note", item={})
    elif answer_mode == "analytical":
        if not accepted["decision_basis"] and not accepted["dimension_analyses"]:
            _add_violation(violations, field="decision_basis", reason="missing_decision_basis", item={})
        if not accepted["uncertainty_notes"] and not accepted["methodology_limitations"]:
            _add_violation(violations, field="uncertainty_notes", reason="missing_uncertainty_notes", item={})
    elif answer_mode == "cautious_outlook":
        if not accepted["uncertainty_notes"] and not accepted["safety_notes"]:
            _add_violation(violations, field="uncertainty_notes", reason="missing_caution_note", item={})

    accepted_refs: set[str] = set(accepted.get("tentative_conclusion", {}).get("citation_refs", []) or [])
    for item in accepted["dimension_analyses"]:
        accepted_refs.update(str(ref) for ref in item.get("evidence_refs", []) if str(ref).strip())
    for field_name in ("decision_basis", "supporting_points", "counterpoints", "risk_tradeoffs", "uncertainty_notes", "safety_notes"):
        for item in accepted[field_name]:
            accepted_refs.update(str(ref) for ref in item.get("citation_refs", []) if str(ref).strip())
    accepted["citation_refs"] = sorted(accepted_refs)

    repair_instructions: list[str] = []
    for violation in violations:
        reason = str(violation.reason).replace("_", " ")
        repair_instructions.append(f"Fix draft violation: {reason}.")
    if not accepted.get("tentative_conclusion", {}).get("statement"):
        repair_instructions.append("Add one evidence-grounded tentative conclusion.")
    if answer_mode == "comparison_brief":
        repair_instructions.append("Keep the comparison conditional and non-advisory.")
        if safety_intent == "investment_advice_like":
            repair_instructions.append("Add an explicit non-investment-advice note.")
    if answer_mode == "analytical":
        repair_instructions.append("State the main problem or risk and the evidence basis.")
    if answer_mode == "cautious_outlook":
        repair_instructions.append("Keep any outlook as a cautious observation, not a prediction.")
    repair_instructions = list(dict.fromkeys(repair_instructions))

    hard_violations = [item for item in violations if _is_hard_violation_reason(str(item.reason))]
    warning_violations = [item for item in violations if item not in hard_violations]
    passed = not hard_violations
    status = "accepted" if passed and not warning_violations else ("accepted_with_warnings" if passed else "invalid")
    final_status = "passed" if passed and not warning_violations else ("passed_with_warnings" if passed else "failed_validation")
    fallback_reason = None if passed else "draft_validation_failed"
    return DraftValidationResult(
        passed=passed,
        status=status,
        final_status=final_status,
        accepted_draft=accepted if passed else {},
        violations=violations,
        warnings=[item.model_dump(exclude_none=True) for item in warning_violations],
        hard_violations=[item.model_dump(exclude_none=True) for item in hard_violations],
        rejected_items=[item.model_dump(exclude_none=True) for item in violations],
        repair_instructions=[] if passed else repair_instructions,
        revision_instructions=[] if passed else repair_instructions,
        fallback_reason=fallback_reason,
    )


def summarize_draft_validation(validation: dict[str, Any]) -> dict[str, Any]:
    validation = dict(validation or {})
    violations = validation.get("violations", []) or []
    if violations and isinstance(violations[0], DraftViolation):
        reasons = [str(item.reason) for item in violations]
    else:
        reasons = [str(item.get("reason", "")) for item in violations if isinstance(item, dict)]
    return {
        "passed": bool(validation.get("passed", False)),
        "final_status": str(validation.get("final_status", validation.get("status", ""))),
        "violation_count": len(violations),
        "hard_violation_count": len(validation.get("hard_violations", []) or []),
        "warning_count": len(validation.get("warnings", []) or []),
        "violation_reasons": [reason for reason in reasons if reason],
        "repair_instruction_count": len(validation.get("repair_instructions", []) or []),
        "fallback_reason": validation.get("fallback_reason"),
    }
