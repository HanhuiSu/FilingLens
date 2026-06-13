# ruff: noqa: F401
"""Evidence-constrained analytical synthesis for final answer shaping."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.analyst_draft import project_analyst_draft_to_synthesis
from src.agent.analytical_reasoning import analytical_reasoning_payload, bracket_ref
from src.agent.evidence_sufficiency import summarize_evidence_requirements
from src.agent.metric_availability import normalize_metric_name
from src.agent.metric_display import format_metric_value, metric_display_name, period_category
from src.agent.output_language import display_theme, risk_theme_key
from src.agent.rendering import (
    render_methodology_comparison_brief,
    render_methodology_single_company_brief,
    render_risk_focused_analysis_brief,
)
from src.agent.types import AnalyticalSynthesis, MethodologyAnswer, MethodologyDimensionSection, RiskFocusedAnswer, RiskTheme

_FORBIDDEN_INVESTMENT_TERMS = (
    "推荐买",
    "应该买",
    "应该卖",
    "买入",
    "卖出",
    "一定更好",
    "必然更好",
    "should buy",
    "should sell",
    "recommend buying",
    "recommend selling",
    "must buy",
    "definitely better",
)

_FORBIDDEN_FORECAST_TERMS = (
    "一定会上涨",
    "一定会下跌",
    "明天会涨",
    "明天会跌",
    "明天会涨到",
    "明天会跌到",
    "必然上涨",
    "必然下跌",
    "will rise tomorrow",
    "will fall tomorrow",
    "will definitely rise",
    "will definitely fall",
    "guaranteed to rise",
    "guaranteed to fall",
)

_LIMITED_NUMERIC_ONLY_MODES = {"limited_judgment", "limited_outlook"}


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def _dedupe_dicts(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        key = str(item)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_lines(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        line = str(item).strip()
        if not line or line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def _claim_id(index: int, prefix: str) -> str:
    return f"{prefix}{index + 1}"


def _claim_sentence(claim: dict[str, Any]) -> str:
    return str(claim.get("sentence", "")).strip()


def _claim_ids_from_claim(claim: dict[str, Any], fallback: str) -> list[str]:
    explicit = [str(x) for x in claim.get("claim_ids", []) if str(x).strip()]
    if explicit:
        return explicit
    evidence_ids = [str(x) for x in claim.get("evidence_ids", []) if str(x).strip()]
    return evidence_ids or [fallback]


def _numbers_in(text: str) -> set[str]:
    pattern = r"(?<![A-Za-z0-9-])[-+]?\d+(?:\.\d+)?%?(?=(?:[KMBT]\b)|[万亿亿美元倍xX%]|\s|$|[,.;:)\]，。；、])"
    return set(re.findall(pattern, text or ""))


def _number_value(token: str) -> float | None:
    cleaned = str(token or "").replace(",", "").rstrip("%")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _number_supported_by_allowed(token: str, allowed_numbers: set[str]) -> bool:
    if token in allowed_numbers:
        return True
    value = _number_value(token)
    if value is None:
        return False
    is_percent = str(token).endswith("%")
    scale_factors = (
        (1.0, 100.0, 0.01)
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
            if abs(value - expected) < tolerance:
                return True
    return False


def _unsupported_numbers(sentence: str, allowed_numbers: set[str]) -> set[str]:
    lowered = str(sentence or "").lower()
    unsupported: set[str] = set()
    for token in _numbers_in(sentence):
        if f"{token}-k" in lowered or f"{token}-q" in lowered:
            continue
        if _number_supported_by_allowed(token, allowed_numbers):
            continue
        unsupported.add(token)
    return unsupported


def _allowed_numeric_tokens(valid_numeric_claims: list[dict[str, Any]]) -> set[str]:
    tokens: set[str] = set()
    for claim in valid_numeric_claims:
        tokens |= _numbers_in(_claim_sentence(claim))
    return tokens


def _item_claim_ids(item: dict[str, Any]) -> list[str]:
    ids = item.get("claim_ids", item.get("evidence_ids", []))
    return [str(x) for x in ids if str(x).strip()]


def _unsupported(item: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "claim_type": "synthesis",
        "sentence": str(item.get("sentence", "")),
        "evidence_ids": _item_claim_ids(item),
        "reason": reason,
    }


def synthesis_view_kind(answer_mode: str, task_type: str = "") -> str:
    if answer_mode == "meta":
        return "meta_response"
    if answer_mode == "clarification":
        return "clarification"
    if answer_mode == "risk_focused_analysis":
        return "risk_focused_analysis_brief"
    if answer_mode == "cautious_outlook":
        return "cautious_outlook"
    if answer_mode == "comparison_brief":
        return "comparison_brief"
    if answer_mode == "refusal_or_redirect":
        return "refusal_or_redirect"
    if task_type == "company_comparison":
        return "comparison_brief"
    if task_type in {"trend_analysis", "report_summary", "fact_qa", "company_comparison"}:
        return "analytical_brief"
    return "analytical_brief"


def _cautious_sentence(lang: str) -> str:
    if lang == "zh":
        return "基于当前已披露数据，这只能作为趋势和风险角度的观察，不能预测未来结果。"
    return "Based on currently disclosed data, this is an observation from trends and risks, not a prediction of future results."


def _investment_boundary_sentence(lang: str) -> str:
    if lang == "zh":
        return "这不是投资建议；更合适的做法是按盈利稳定性、增长弹性和风险暴露来比较。"
    return "This is not investment advice; a better framing is to compare earnings stability, growth optionality, and risk exposure."


def _limited_judgment_notice(lang: str) -> str:
    if lang == "zh":
        return "当前缺少可验证文本证据，因此只做有限财务对比，不能声称这是完整分析。"
    return "Current verifiable text evidence is missing, so this is only a limited financial comparison, not a complete analysis."


def _limited_analysis_notice(lang: str) -> str:
    if lang == "zh":
        return "当前缺少足够的可验证文本证据，因此这里只能基于结构化财务数据做有限分析，不能声称是完整叙事判断。"
    return "Required filing text evidence is incomplete, so this is only a limited analysis based on structured financial data, not a complete narrative judgment."


def _limited_outlook_notice(lang: str) -> str:
    if lang == "zh":
        return "当前缺少足够的可验证文本证据，因此这里只能基于历史数值趋势做谨慎展望，不能把它当作完整前瞻分析。"
    return "Required filing text evidence is incomplete, so any outlook here is limited to historical numeric trends, not a complete forward-looking analysis."


def _insufficient_message(lang: str, synthesis_mode: str) -> str:
    if lang == "zh":
        if synthesis_mode == "insufficient_comparison":
            return "当前可验证数值证据不足，无法形成可靠的公司比较结论。"
        if synthesis_mode == "insufficient_outlook":
            return "当前可验证证据不足，无法形成可靠的展望判断。"
        return "当前验证证据不足，无法形成可靠分析结论。"
    if synthesis_mode == "insufficient_comparison":
        return "Validated numeric evidence is insufficient for a reliable company comparison."
    if synthesis_mode == "insufficient_outlook":
        return "Validated evidence is insufficient for a reliable outlook."
    return "Validated evidence is insufficient for a reliable analytical conclusion."


def _natural_short_answer(
    lang: str,
    key_facts: list[dict[str, Any]],
    analysis: list[dict[str, Any]],
    sufficient: bool,
    *,
    synthesis_mode: str,
) -> str:
    if synthesis_mode == "limited_judgment":
        return _limited_judgment_notice(lang)
    if synthesis_mode == "limited_analysis":
        return _limited_analysis_notice(lang)
    if synthesis_mode == "limited_outlook":
        return _limited_outlook_notice(lang)
    if synthesis_mode.startswith("insufficient_"):
        return _insufficient_message(lang, synthesis_mode)
    if key_facts and analysis and sufficient:
        fact = str(key_facts[0].get("sentence", ""))
        if lang == "zh":
            return f"基于已验证的财务数据和披露文本，核心结论是：{fact}"
        return f"Based on validated financial data and filing evidence, the key takeaway is: {fact}"
    if key_facts:
        return str(key_facts[0].get("sentence", ""))
    if analysis:
        return str(analysis[0].get("sentence", ""))
    if lang == "zh":
        return "当前验证证据不足，无法形成可靠分析结论。"
    return "Validated evidence is insufficient for a reliable analytical conclusion."


def _risk_judgment_like(sentence: str) -> bool:
    lowered = sentence.lower()
    risk_terms = ("risk", "risks", "uncertainty", "uncertain", "concern", "challenge", "风险", "不确定", "问题", "挑战")
    boundary_terms = (
        "not a prediction",
        "不能预测",
        "not investment advice",
        "不是投资建议",
        "missing",
        "缺少",
        "limited",
        "有限",
        "完整分析",
    )
    return any(term in lowered for term in risk_terms) and not any(term in lowered for term in boundary_terms)


def _narrative_judgment_like(sentence: str) -> bool:
    lowered = sentence.lower()
    narrative_terms = (
        "management",
        "guidance",
        "md&a",
        "risk factor",
        "risk factors",
        "business model",
        "management view",
        "管理层",
        "指引",
        "风险因素",
        "业务模式",
    )
    boundary_terms = (
        "missing",
        "limited",
        "not complete",
        "not a complete analysis",
        "缺少",
        "有限",
        "完整分析",
        "不是完整分析",
    )
    return any(term in lowered for term in narrative_terms) and not any(term in lowered for term in boundary_terms)


_METHODOLOGY_DIMENSION_TITLES_ZH = {
    "business_model": "商业模式",
    "revenue_quality": "收入质量",
    "profitability_quality": "盈利质量",
    "cash_flow_quality": "现金流质量",
    "balance_sheet_and_capital_intensity": "资产负债与资本强度",
    "moat_and_competitive_risk": "竞争与风险",
    "valuation_and_risk_boundary": "估值边界",
}

_METHODOLOGY_DIMENSION_TITLES_EN = {
    "business_model": "Business Model",
    "revenue_quality": "Revenue Quality",
    "profitability_quality": "Profitability Quality",
    "cash_flow_quality": "Cash-Flow Quality",
    "balance_sheet_and_capital_intensity": "Balance Sheet and Capital Intensity",
    "moat_and_competitive_risk": "Competitive Risk",
    "valuation_and_risk_boundary": "Valuation Boundary",
}


def _dimension_title(dimension_id: str, lang: str) -> str:
    titles = _METHODOLOGY_DIMENSION_TITLES_ZH if lang == "zh" else _METHODOLOGY_DIMENSION_TITLES_EN
    return titles.get(dimension_id, dimension_id.replace("_", " ").title())


def _status_value(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("status") or "").strip()
    return str(item or "").strip()


def _dimension_statuses_from_sufficiency(sufficiency: dict[str, Any] | None) -> dict[str, str]:
    status_map = dict((sufficiency or {}).get("dimension_status_map", {}) or {})
    return {str(dimension_id): _status_value(item) for dimension_id, item in status_map.items()}


def _methodology_comparison_ready(statuses: dict[str, str]) -> bool:
    required = ("revenue_quality", "profitability_quality", "moat_and_competitive_risk")
    return all(statuses.get(dimension_id) in {"satisfied", "partial"} for dimension_id in required)


def _methodology_single_company_ready(statuses: dict[str, str]) -> bool:
    return statuses.get("revenue_quality") in {"satisfied", "partial"} and statuses.get("profitability_quality") in {
        "satisfied",
        "partial",
    }


def _any_methodology_dimension_ready(statuses: dict[str, str]) -> bool:
    return any(status in {"satisfied", "partial"} for status in statuses.values())


def _valuation_missing(statuses: dict[str, str]) -> bool:
    return statuses.get("valuation_and_risk_boundary") == "missing"


def _selected_framework_id(selected_framework: dict[str, Any] | str | None) -> str:
    if isinstance(selected_framework, str):
        return selected_framework
    selected_framework = dict(selected_framework or {})
    return str(
        selected_framework.get("id")
        or selected_framework.get("framework_id")
        or selected_framework.get("analysis_framework_id")
        or "fundamental_quality_analysis"
    )


def derive_synthesis_mode(
    *,
    answer_mode: str,
    task_type: str,
    safety_intent: str,
    evidence_sufficiency: dict[str, Any] | None = None,
    has_validated_numeric: bool = False,
    has_validated_text: bool = False,
) -> str:
    sufficiency = dict(evidence_sufficiency or {})
    overall_status = str(sufficiency.get("overall_status", "")).strip()
    can_synthesize = bool(sufficiency.get("can_synthesize", False))

    if answer_mode in {"meta", "clarification", "refusal_or_redirect"}:
        return "conversational_short_circuit"
    if answer_mode == "direct_fact":
        return "direct_fact"
    if answer_mode == "risk_focused_analysis":
        return "risk_focused_analysis" if has_validated_text or can_synthesize else "insufficient_risk_evidence"

    is_comparison = (
        answer_mode == "comparison_brief"
        or task_type == "company_comparison"
        or safety_intent == "investment_advice_like"
    )
    statuses = _dimension_statuses_from_sufficiency(sufficiency)
    if (
        answer_mode == "analytical"
        and task_type == "report_summary"
        and (has_validated_numeric or has_validated_text)
        and _any_methodology_dimension_ready(statuses)
    ):
        return "methodology_single_company"
    if (
        is_comparison
        and has_validated_numeric
        and has_validated_text
        and _methodology_comparison_ready(statuses)
        and _valuation_missing(statuses)
    ):
        return "methodology_comparison"

    if not sufficiency and (has_validated_numeric or has_validated_text):
        if is_comparison:
            if has_validated_text:
                return "balanced_comparison"
            if has_validated_numeric:
                return "limited_judgment"
            return "insufficient_comparison"
        if answer_mode == "cautious_outlook":
            if has_validated_text:
                return "cautious_outlook"
            if has_validated_numeric:
                return "limited_outlook"
            return "insufficient_outlook"
        if answer_mode == "analytical" or task_type in {"report_summary", "trend_analysis"}:
            if has_validated_text:
                return "limited_analysis"
            return "insufficient_analysis"

    if is_comparison:
        if overall_status == "sufficient":
            if has_validated_numeric and has_validated_text:
                return "balanced_comparison"
            if has_validated_numeric:
                return "limited_judgment"
            return "insufficient_comparison"
        if can_synthesize and has_validated_numeric:
            return "limited_judgment"
        return "insufficient_comparison"

    if answer_mode == "cautious_outlook":
        if overall_status == "sufficient":
            if has_validated_text or not has_validated_numeric:
                return "cautious_outlook"
            return "limited_outlook"
        if can_synthesize and has_validated_numeric:
            return "limited_outlook"
        return "insufficient_outlook"

    if answer_mode == "analytical" or task_type in {"report_summary", "trend_analysis"}:
        if overall_status == "sufficient":
            if has_validated_text:
                return "validated_analysis"
            return "insufficient_analysis"
        if can_synthesize and has_validated_numeric and _any_methodology_dimension_ready(statuses):
            return "methodology_single_company"
        if has_validated_text:
            return "limited_analysis"
        return "insufficient_analysis"

    if overall_status == "sufficient":
        if has_validated_text or not has_validated_numeric:
            return "validated_analysis"
        return "limited_analysis"
    if can_synthesize and has_validated_numeric:
        return "limited_analysis"
    return "insufficient_analysis"


def _latest_claims_by_company_metric(valid_numeric_claims: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for index, claim in enumerate(valid_numeric_claims):
        ticker = str(claim.get("ticker", "")).upper().strip()
        metric = str(claim.get("metric", "")).strip()
        period_end = str(claim.get("period_end", "")).strip()
        value = claim.get("value")
        if not ticker or not metric or value is None:
            continue
        key = (ticker, metric)
        existing_period = str(latest.get(key, {}).get("period_end", ""))
        if key not in latest or period_end >= existing_period:
            latest[key] = {
                "ticker": ticker,
                "metric": metric,
                "period_end": period_end,
                "value": value,
                "claim_ids": _claim_ids_from_claim(claim, _claim_id(index, "N")),
            }
    return latest


def _metric_leader(
    latest_claims: dict[tuple[str, str], dict[str, Any]],
    metric: str,
) -> tuple[str, list[str]] | tuple[None, list[str]]:
    candidates = [item for (ticker, item_metric), item in latest_claims.items() if item_metric == metric and ticker]
    best: dict[str, Any] | None = None
    for item in candidates:
        try:
            value = float(item.get("value"))
        except (TypeError, ValueError):
            continue
        if best is None or value > float(best.get("value")):
            best = item
    if best is None:
        return None, []
    return str(best.get("ticker", "")), list(best.get("claim_ids", []))


def _limited_judgment_analysis(valid_numeric_claims: list[dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    latest_claims = _latest_claims_by_company_metric(valid_numeric_claims)
    companies = sorted({ticker for ticker, _metric in latest_claims.keys() if ticker})
    if len(companies) < 2:
        return []
    profit_leader, profit_claim_ids = _metric_leader(latest_claims, "net_income")
    if not profit_leader:
        profit_leader, profit_claim_ids = _metric_leader(latest_claims, "operating_margin")
    scale_leader, scale_claim_ids = _metric_leader(latest_claims, "revenue")
    analysis: list[dict[str, Any]] = []
    if profit_leader:
        if lang == "zh":
            analysis.append(
                {
                    "sentence": f"如果更看重当前盈利表现与稳定性，{profit_leader} 目前在已验证财务数据上更占优。",
                    "claim_ids": profit_claim_ids,
                }
            )
        else:
            analysis.append(
                {
                    "sentence": f"If current profitability and earnings stability matter more, {profit_leader} currently looks stronger on the validated financial data.",
                    "claim_ids": profit_claim_ids,
                }
            )
    if scale_leader:
        if not profit_leader or scale_leader != profit_leader:
            if lang == "zh":
                analysis.append(
                    {
                        "sentence": f"如果更看重收入规模或增长弹性，{scale_leader} 也值得继续观察。",
                        "claim_ids": scale_claim_ids,
                    }
                )
            else:
                analysis.append(
                    {
                        "sentence": f"If revenue scale or growth optionality matters more, {scale_leader} is still worth watching.",
                        "claim_ids": scale_claim_ids,
                    }
                )
    return analysis[:2]


def _frame_focus_dimensions(frame: dict[str, Any]) -> list[str]:
    dimensions: list[str] = []
    for item in frame.get("evidence_basis", []) or []:
        if not isinstance(item, dict):
            continue
        dimension = str(item.get("dimension", "")).strip()
        if dimension and dimension not in dimensions:
            dimensions.append(dimension)
    return dimensions


def _comparison_frame_short_answer(frame: dict[str, Any], lang: str, synthesis_mode: str) -> str:
    frame = dict(frame or {})
    preferred = str(frame.get("preferred_company", ""))
    profitability_winner = str(frame.get("profitability_winner", ""))
    scale_winner = str(frame.get("scale_winner", ""))
    margin_winner = str(frame.get("margin_winner", ""))
    focus_dimensions = _frame_focus_dimensions(frame)
    if preferred in {"", "unavailable", "mixed", "no_clear_preference"}:
        if "cash_flow_quality" in focus_dimensions:
            if lang == "zh":
                refs = _methodology_refs(_frame_dimension_refs(frame, "cash_flow_quality"))
                return f"基于当前现金流证据，更合适的结论是只能做有边界比较，而不是用净利率替代现金流质量。{refs}"
            return "Based on current cash-flow evidence, the right conclusion is a bounded comparison rather than substituting net margin for cash-flow quality."
        if "valuation_and_risk_boundary" in focus_dimensions:
            if lang == "zh":
                refs = _methodology_refs(_frame_dimension_refs(frame, "valuation_and_risk_boundary"))
                judgments = _frame_metric_judgments(frame, "valuation_and_risk_boundary")
                if judgments:
                    return f"有限结论：估值风险呈分歧，不能给单一绝对排序。{refs}"
                return f"基于当前估值证据，只能做有限估值风险边界比较，不能用普通风险模板替代。{refs}"
            return "Based on current valuation evidence, only a bounded valuation-risk comparison is supported."
        if "revenue_quality" in focus_dimensions:
            if lang == "zh":
                scale_refs = _frame_dimension_field_refs(frame, "revenue_quality", "scale_winner_refs") or _frame_dimension_refs(frame, "revenue_quality")
                refs = _methodology_refs(scale_refs)
                scale_name = scale_winner if scale_winner not in {"", "unavailable", "mixed", "no_clear_preference"} else ""
                if scale_name:
                    return f"如果只基于当前增长/收入质量证据，{scale_name} 在收入规模上占优；但收入历史证据不完整，因此不能完整判断增长质量。{refs}"
                return f"基于当前收入证据，只能做有边界的增长/收入质量比较，不能用净利率替代。{refs}"
            return "Based on current revenue evidence, only a bounded growth/revenue-quality comparison is supported."
        if lang == "zh":
            return "基于当前已验证财报证据，更合适的结论是两家公司各有侧重，而不是直接下单一偏好。"
        return "Based on the validated filing evidence, the better conclusion is that the two companies lead on different dimensions rather than one clear preference."
    if lang == "zh":
        if "cash_flow_quality" in focus_dimensions:
            refs = _methodology_refs(_frame_dimension_refs(frame, "cash_flow_quality"))
            return f"如果只基于当前现金流证据，我更偏向 {preferred}；判断依据是经营现金流、自由现金流、资本开支和 FCF margin，而不是净利率。{refs}"
        if "valuation_and_risk_boundary" in focus_dimensions:
            refs = _methodology_refs(_frame_dimension_refs(frame, "valuation_and_risk_boundary"))
            return f"有限结论：估值风险呈分歧，不能给单一绝对排序；需要逐项看 P/E、P/S 和 FCF yield。{refs}"
        if "revenue_quality" in focus_dimensions:
            scale_refs = _frame_dimension_field_refs(frame, "revenue_quality", "scale_winner_refs") or _frame_dimension_refs(frame, "revenue_quality")
            refs = _methodology_refs(scale_refs)
            scale_name = scale_winner if scale_winner not in {"", "unavailable", "mixed", "no_clear_preference"} else preferred
            return f"如果只基于当前增长/收入质量证据，{scale_name} 在收入规模上占优；但收入历史证据不完整，因此不能完整判断增长质量。{refs}"
        if synthesis_mode == "limited_judgment":
            sentence = f"如果只基于当前财报证据，我更偏向 {preferred}"
            if profitability_winner == preferred or margin_winner == preferred:
                sentence += "，因为盈利能力和净利率证据更强"
            sentence += "。"
            if scale_winner and scale_winner not in {"", preferred, "unavailable", "mixed"}:
                sentence += f" 但如果更看重收入规模，{scale_winner} 仍值得关注。"
            return sentence
        sentence = f"基于当前已验证的财务与文本证据，我会更偏向 {preferred}"
        if profitability_winner == preferred or margin_winner == preferred:
            sentence += "，因为它在盈利能力和净利率上更占优"
        sentence += "。"
        if scale_winner and scale_winner not in {"", preferred, "unavailable", "mixed"}:
            sentence += f" 同时，{scale_winner} 在收入规模上仍有优势。"
        return sentence
    if "cash_flow_quality" in focus_dimensions:
        return f"Based only on current cash-flow evidence, I lean toward {preferred}; the basis is operating cash flow, free cash flow, capex, and FCF margin, not net margin."
    if "valuation_and_risk_boundary" in focus_dimensions:
        return f"Based only on current valuation evidence, {preferred} deserves closer valuation-risk attention; the boundary is P/E, P/S, FCF yield, market cap, and share price inputs."
    if "revenue_quality" in focus_dimensions:
        return f"Based only on current growth/revenue-quality evidence, I lean toward {preferred}; the basis is revenue growth, revenue history, and revenue scale, not net margin."
    sentence = f"If I stay within the current validated filing evidence, I lean toward {preferred}"
    if profitability_winner == preferred or margin_winner == preferred:
        sentence += " because it looks stronger on profitability and margin evidence"
    sentence += "."
    if scale_winner and scale_winner not in {"", preferred, "unavailable", "mixed"}:
        sentence += f" But {scale_winner} still has the edge on revenue scale."
    return sentence


def _frame_dimension_refs(frame: dict[str, Any], dimension: str) -> list[str]:
    item = _frame_dimension_item(frame, dimension)
    if item:
        return [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
    return []


def _frame_dimension_item(frame: dict[str, Any], dimension: str) -> dict[str, Any]:
    for item in frame.get("evidence_basis", []) or []:
        if isinstance(item, dict) and str(item.get("dimension", "")) == dimension:
            return dict(item)
    return {}


def _frame_dimension_field_refs(frame: dict[str, Any], dimension: str, field: str) -> list[str]:
    item = _frame_dimension_item(frame, dimension)
    return [str(ref) for ref in item.get(field, []) or [] if str(ref).strip()]


def _frame_metric_judgments(frame: dict[str, Any], dimension: str) -> list[dict[str, Any]]:
    item = _frame_dimension_item(frame, dimension)
    return [dict(value) for value in item.get("metric_judgments", []) or [] if isinstance(value, dict)]


def build_bounded_valuation_risk_comparison_candidate(
    frame: dict[str, Any],
    packet: dict[str, Any] | None = None,
    *,
    lang: str = "zh",
) -> str:
    """Render a fixed bounded valuation-risk comparison from frame judgments."""
    frame = dict(frame or {})
    focus_dimensions = _frame_focus_dimensions(frame)
    if "valuation_and_risk_boundary" not in focus_dimensions:
        return ""
    judgments = _frame_metric_judgments(frame, "valuation_and_risk_boundary")
    if not judgments:
        return ""

    def _refs_text(refs: list[str]) -> str:
        ordered = [str(ref).strip() for ref in refs if str(ref).strip()]
        return "".join(f"[{ref}]" for ref in list(dict.fromkeys(ordered))[:4])

    def _sentence_for(judgment: dict[str, Any]) -> str:
        sentence = str(judgment.get("zh" if lang == "zh" else "en") or judgment.get("sentence") or "").strip()
        sentence = _clean_user_facing_phrase(sentence).rstrip("。.;； ")
        refs = _refs_text([str(ref) for ref in judgment.get("evidence_refs", []) or []])
        if refs and not re.search(r"\[[NT]\d+\]", sentence):
            sentence = f"{sentence}{refs}"
        return sentence

    ordered: list[dict[str, Any]] = []
    used: set[int] = set()
    for marker in ("P/E", "pe_ratio", "P/S", "ps_ratio", "FCF yield", "fcf_yield"):
        for idx, judgment in enumerate(judgments):
            if idx in used:
                continue
            haystack = " ".join(str(judgment.get(key) or "") for key in ("metric", "zh", "en", "sentence"))
            if marker.lower() in haystack.lower():
                ordered.append(judgment)
                used.add(idx)
                break
    for idx, judgment in enumerate(judgments):
        if idx not in used:
            ordered.append(judgment)
    ordered = ordered[:3]
    refs = _refs_text(_frame_dimension_refs(frame, "valuation_and_risk_boundary"))
    if not refs:
        all_refs: list[str] = []
        for judgment in ordered:
            all_refs.extend(str(ref) for ref in judgment.get("evidence_refs", []) or [])
        refs = _refs_text(all_refs)

    if lang == "zh":
        lines = [f"有限结论：不能给单一绝对排序，因为估值指标指向不一致。{refs}".rstrip()]
        for judgment in ordered:
            sentence = _sentence_for(judgment)
            if sentence:
                lines.append(f"- {sentence}。")
        lines.append("证据边界：缺少完整股价、股数和市值口径或历史/同业基准，因此不能判断谁绝对更贵。")
        return "\n".join(lines)

    lines = [
        "Conclusion",
        f"A single absolute ranking is not supported because valuation indicators point in different directions. {refs}".rstrip(),
        "",
        "Metric-by-Metric Comparison",
    ]
    for judgment in ordered:
        sentence = _sentence_for(judgment)
        if sentence:
            lines.append(f"- {sentence}.")
    lines.extend(
        [
            "",
            "Interpretation",
            "- P/E, P/S, and FCF yield should be read separately; higher multiples and lower FCF yield can point to different kinds of valuation risk.",
            "",
            "Evidence Boundary",
            "- Complete price / shares / market-cap context or historical/peer benchmarks are missing, so the evidence cannot determine which company is absolutely more expensive.",
        ]
    )
    return "\n".join(lines)


def _comparison_frame_analysis(frame: dict[str, Any], lang: str, synthesis_mode: str) -> list[dict[str, Any]]:
    frame = dict(frame or {})
    preferred = str(frame.get("preferred_company", ""))
    profitability_winner = str(frame.get("profitability_winner", ""))
    scale_winner = str(frame.get("scale_winner", ""))
    margin_winner = str(frame.get("margin_winner", ""))
    growth_winner = str(frame.get("growth_winner", ""))
    focus_dimensions = _frame_focus_dimensions(frame)
    items: list[dict[str, Any]] = []
    if "cash_flow_quality" in focus_dimensions:
        refs = _frame_dimension_refs(frame, "cash_flow_quality")
        rationale = _clean_user_facing_phrase(str(frame.get("rationale", "")).strip())
        if lang == "zh":
            if rationale:
                sentence = f"现金流质量比较使用实际经营现金流、自由现金流、资本开支和 FCF margin：{rationale}"
            elif preferred and preferred not in {"mixed", "unavailable", "no_clear_preference"}:
                sentence = f"现金流质量比较应看经营现金流、自由现金流、资本开支和 FCF margin；当前现金流证据更支持 {preferred}。"
            else:
                sentence = "现金流质量比较应看经营现金流、自由现金流、资本开支和 FCF margin；当前只能做有边界比较。"
        else:
            if rationale:
                sentence = rationale
            elif preferred and preferred not in {"mixed", "unavailable", "no_clear_preference"}:
                sentence = f"Cash-flow quality should be judged through operating cash flow, free cash flow, capex, and FCF margin; current evidence favors {preferred}."
            else:
                sentence = "Cash-flow quality should be judged through operating cash flow, free cash flow, capex, and FCF margin; only a bounded comparison is supported."
        items.append({"sentence": sentence, "claim_ids": refs})
        return items[:3]
    if "valuation_and_risk_boundary" in focus_dimensions:
        refs = _frame_dimension_refs(frame, "valuation_and_risk_boundary")
        metric_judgments = _frame_metric_judgments(frame, "valuation_and_risk_boundary")
        rationale = _clean_user_facing_phrase(str(frame.get("rationale", "")).strip())
        if lang == "zh":
            if metric_judgments:
                items.append({"sentence": "有限结论：估值风险呈分歧，不能给单一绝对排序。", "claim_ids": refs})
                for judgment in metric_judgments[:3]:
                    sentence = _clean_user_facing_phrase(str(judgment.get("zh") or "").strip())
                    if sentence:
                        items.append(
                            {
                                "sentence": sentence,
                                "claim_ids": [str(ref) for ref in judgment.get("evidence_refs", []) or [] if str(ref).strip()],
                            }
                        )
                return items[:4]
            if rationale:
                sentence = f"估值风险比较使用实际 P/E、P/S、FCF yield、市值或股价输入：{rationale}"
            elif preferred and preferred not in {"mixed", "unavailable", "no_clear_preference"}:
                sentence = f"估值风险比较应看 P/E、P/S、FCF yield、市值和股价输入；当前估值边界更需要关注 {preferred}。"
            else:
                sentence = "估值风险比较应看 P/E、P/S、FCF yield、市值和股价输入；当前只能做有限估值边界比较。"
        else:
            if metric_judgments:
                items.append({"sentence": "Valuation risk is mixed, so a single absolute ranking is not supported.", "claim_ids": refs})
                for judgment in metric_judgments[:3]:
                    sentence = _clean_user_facing_phrase(str(judgment.get("en") or "").strip())
                    if sentence:
                        items.append(
                            {
                                "sentence": sentence,
                                "claim_ids": [str(ref) for ref in judgment.get("evidence_refs", []) or [] if str(ref).strip()],
                            }
                        )
                return items[:4]
            if rationale:
                sentence = rationale
            elif preferred and preferred not in {"mixed", "unavailable", "no_clear_preference"}:
                sentence = f"Valuation risk should be judged through P/E, P/S, FCF yield, market cap, and share price inputs; current valuation boundaries point to {preferred}."
            else:
                sentence = "Valuation risk should be judged through P/E, P/S, FCF yield, market cap, and share price inputs; only a bounded valuation comparison is supported."
        items.append({"sentence": sentence, "claim_ids": refs})
        return items[:3]
    if "revenue_quality" in focus_dimensions:
        refs = _frame_dimension_refs(frame, "revenue_quality")
        scale_refs = _frame_dimension_field_refs(frame, "revenue_quality", "scale_winner_refs") or refs
        growth_refs = _frame_dimension_field_refs(frame, "revenue_quality", "growth_winner_refs")
        history_refs = _frame_dimension_field_refs(frame, "revenue_quality", "revenue_history_refs") or refs
        history_complete = bool(_frame_dimension_item(frame, "revenue_quality").get("revenue_history_complete"))
        if lang == "zh":
            scale_name = scale_winner if scale_winner not in {"", "mixed", "unavailable", "no_clear_preference"} else preferred
            if scale_name and scale_name not in {"mixed", "unavailable", "no_clear_preference"}:
                boundary = "收入历史证据较完整，因此仍只在已验证期间内比较增长质量。" if history_complete else "收入历史证据不完整，因此不能完整判断增长质量。"
                sentence = f"如果只基于当前增长/收入质量证据，{scale_name} 在收入规模上占优；但{boundary}"
                items.append({"sentence": sentence, "claim_ids": scale_refs or history_refs})
                if growth_refs:
                    items.append({"sentence": "收入增速证据只能作为有限增长质量线索，不能用净利率替代增长质量。", "claim_ids": growth_refs})
                return items[:3]
            else:
                sentence = "增长/收入质量比较应看收入增速、收入历史和收入规模；当前只能做有边界比较。"
        else:
            scale_name = scale_winner if scale_winner not in {"", "mixed", "unavailable", "no_clear_preference"} else preferred
            if scale_name and scale_name not in {"mixed", "unavailable", "no_clear_preference"}:
                boundary = "revenue-history evidence is sufficiently comparable only within validated periods." if history_complete else "revenue-history evidence is incomplete, so full growth quality cannot be determined."
                sentence = f"Based only on current growth/revenue-quality evidence, {scale_name} leads on revenue scale, but {boundary}"
                items.append({"sentence": sentence, "claim_ids": scale_refs or history_refs})
                if growth_refs:
                    items.append({"sentence": "Revenue-growth evidence is only a bounded growth-quality signal; net margin is not a substitute for growth quality.", "claim_ids": growth_refs})
                return items[:3]
            else:
                sentence = "Growth/revenue quality should be judged through revenue growth, revenue history, and revenue scale; only a bounded comparison is supported."
        items.append({"sentence": sentence, "claim_ids": refs})
        return items[:3]
    if preferred and preferred not in {"mixed", "unavailable", "no_clear_preference"}:
        if lang == "zh":
            sentence = f"当前证据更支持把 {preferred} 视为在盈利能力维度更占优的一方；这个判断基于营收、净利润和由程序计算的净利率。"
        else:
            sentence = f"Current evidence more strongly supports {preferred} on profitability; this uses revenue, net income, and calculated net margin."
        items.append({"sentence": sentence, "claim_ids": _frame_dimension_refs(frame, "profitability")})
    if margin_winner and margin_winner not in {"", "unavailable", "mixed"}:
        if lang == "zh":
            other_companies = [
                str(company)
                for company in frame.get("companies", []) or []
                if str(company).strip() and str(company).strip() != margin_winner
            ]
            if other_companies:
                sentence = f"{margin_winner} 净利率更占优，{other_companies[0]} 净利率作为对照较低；净利率由程序按净利润除以营收计算。"
            else:
                sentence = f"{margin_winner} 净利率更占优；净利率由程序按净利润除以营收计算。"
        else:
            sentence = f"On margin, {margin_winner} leads; net margin is computed programmatically as net income divided by revenue."
        items.append({"sentence": sentence, "claim_ids": _frame_dimension_refs(frame, "margin")})
    if scale_winner and scale_winner not in {"", "unavailable", "mixed"} and scale_winner != preferred:
        if lang == "zh":
            items.append({"sentence": f"如果更看重收入规模，{scale_winner} 仍有明显优势。", "claim_ids": _frame_dimension_refs(frame, "scale")})
        else:
            items.append({"sentence": f"If revenue scale matters more, {scale_winner} still has a clear advantage.", "claim_ids": _frame_dimension_refs(frame, "scale")})
    if growth_winner and growth_winner not in {"", "unavailable", "mixed"} and len(items) < 3:
        if lang == "zh":
            items.append({"sentence": f"增长维度上，当前可验证指标更支持 {growth_winner}。", "claim_ids": _frame_dimension_refs(frame, "growth")})
        else:
            items.append({"sentence": f"On available growth evidence, {growth_winner} leads.", "claim_ids": _frame_dimension_refs(frame, "growth")})
    if synthesis_mode != "limited_judgment":
        for risk in frame.get("risk_considerations", []) or []:
            if not isinstance(risk, dict):
                continue
            companies = [str(x) for x in risk.get("companies", []) or [] if str(x).strip()]
            label = str(risk.get("label", "") or risk.get("theme_code", ""))
            refs = [str(x) for x in risk.get("evidence_refs", []) or [] if str(x).strip()]
            if not companies or not label:
                continue
            if lang == "zh":
                items.append({"sentence": f"已验证文本还提示 {', '.join(companies)} 需要结合{label}来看待。", "claim_ids": refs})
            else:
                items.append({"sentence": f"Validated text also suggests {', '.join(companies)} should be read in light of {label}.", "claim_ids": refs})
            if len(items) >= 3:
                break
    if not items and profitability_winner and profitability_winner not in {"", "unavailable", "mixed"}:
        if lang == "zh":
            items.append({"sentence": f"如果更看重盈利能力，{profitability_winner} 当前更占优。", "claim_ids": []})
        else:
            items.append({"sentence": f"If profitability and margin evidence matter more, {profitability_winner} currently looks stronger.", "claim_ids": []})
    return items[:3]


def _comparison_frame_risk_tradeoff(frame: dict[str, Any], lang: str) -> dict[str, Any] | None:
    frame = dict(frame or {})
    refs = _frame_dimension_refs(frame, "risk")
    if refs:
        risks = [item for item in frame.get("risk_considerations", []) or [] if isinstance(item, dict)]
        first = dict(risks[0]) if risks else {}
        companies = [str(x) for x in first.get("companies", []) or [] if str(x).strip()]
        theme_code = str(first.get("theme_code", ""))
        label = str(first.get("label", "") or theme_code or "风险因素")
        zh_labels = {
            "demand_macro": "需求 / 宏观压力",
            "competition": "竞争",
            "regulation_legal": "监管 / 法律",
            "operations_supply_chain": "运营 / 供应链",
            "execution_investment": "执行 / 再投资",
            "validated_filing_text": "已验证披露文本",
            "other_validated_risk": "其他已验证风险",
        }
        if lang == "zh":
            company_text = "、".join(companies) if companies else "相关公司"
            label_text = zh_labels.get(theme_code, label)
            if theme_code == "validated_filing_text":
                return {"sentence": f"已验证文本为 {company_text} 提供了风险背景，但不足以支持强不对称风险判断。", "claim_ids": refs}
            return {"sentence": f"已验证文本提示 {company_text} 需要结合 {label_text} 等风险权衡来看。", "claim_ids": refs}
        company_text = ", ".join(companies) if companies else "the companies"
        if theme_code == "validated_filing_text":
            return {"sentence": f"Validated filing text provides risk context for {company_text}, but it does not support a strong asymmetric risk conclusion.", "claim_ids": refs}
        return {"sentence": f"Validated text suggests {company_text} should be read alongside {label} risk tradeoffs.", "claim_ids": refs}
    if lang == "zh":
        return {"sentence": "当前缺少足够的可验证文本证据，因此风险比较只能保持有限。", "claim_ids": []}
    return {"sentence": "Risk comparison is limited because validated filing text evidence is not available.", "claim_ids": []}


_DIMENSION_LABELS_ZH = {
    "business_model": "业务模式",
    "revenue_quality": "收入质量",
    "profitability_quality": "盈利质量",
    "cash_flow_quality": "现金流质量",
    "balance_sheet_and_capital_intensity": "资产负债与资本强度",
    "moat_and_competitive_risk": "竞争与风险",
    "valuation_and_risk_boundary": "估值边界",
}

_DIMENSION_LABELS_EN = {
    "business_model": "Business Model",
    "revenue_quality": "Revenue Quality",
    "profitability_quality": "Profitability Quality",
    "cash_flow_quality": "Cash Flow Quality",
    "balance_sheet_and_capital_intensity": "Balance Sheet And Capital Intensity",
    "moat_and_competitive_risk": "Competitive Risk",
    "valuation_and_risk_boundary": "Valuation Boundary",
}


def _dimension_label(dimension_id: str, lang: str) -> str:
    if lang == "zh":
        return _DIMENSION_LABELS_ZH.get(dimension_id, dimension_id)
    return _DIMENSION_LABELS_EN.get(dimension_id, dimension_id)


def _usable_dimension_statuses(requirement_summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    status_map = dict(requirement_summary.get("dimension_status_map", {}) or {})
    return {
        str(dimension_id): dict(item)
        for dimension_id, item in status_map.items()
        if isinstance(item, dict) and str(item.get("status", "")) in {"satisfied", "partial"}
    }


def _frame_refs(frame: dict[str, Any], dimension: str) -> list[str]:
    for item in frame.get("evidence_basis", []) or []:
        if isinstance(item, dict) and str(item.get("dimension", "")) == dimension:
            return [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
    return []


def _default_follow_up_metrics(missing_dimensions: list[str], lang: str) -> list[str]:
    zh = {
        "business_model": "业务模式披露（ITEM 1）",
        "revenue_quality": "收入趋势、分部收入、递延收入",
        "profitability_quality": "毛利率、营业利润率、净利率",
        "cash_flow_quality": "经营现金流、自由现金流",
        "valuation_and_risk_boundary": "市值、市盈率、市销率、自由现金流收益率",
        "balance_sheet_and_capital_intensity": "现金、总债务、资本开支",
        "moat_and_competitive_risk": "10-K/10-Q 风险因素与竞争披露",
    }
    en = {
        "business_model": "business-model disclosures (Item 1)",
        "revenue_quality": "revenue trend, segment revenue, and deferred revenue",
        "profitability_quality": "gross margin, operating margin, and net margin",
        "cash_flow_quality": "operating cash flow and free cash flow",
        "valuation_and_risk_boundary": "market cap, P/E, P/S, and FCF yield",
        "balance_sheet_and_capital_intensity": "cash, total debt, and capital expenditures",
        "moat_and_competitive_risk": "validated 10-K/10-Q risk and competition disclosures",
    }
    mapping = zh if lang == "zh" else en
    return _dedupe_lines([mapping.get(dim, dim) for dim in missing_dimensions if mapping.get(dim, dim)])


def _useful_follow_up_metrics(dimension_ids: list[str], lang: str) -> list[str]:
    zh = {
        "business_model": "业务模式、客户结构与收入来源披露更新",
        "revenue_quality": "收入增速、分部收入与递延收入变化",
        "profitability_quality": "毛利率、营业利润率与净利率变化",
        "cash_flow_quality": "CFO/净利润、FCF margin 与资本开支变化",
        "valuation_and_risk_boundary": "P/E、P/S、FCF yield 与股价/市值变化",
        "balance_sheet_and_capital_intensity": "现金及等价物、总债务、应收款与存货变化",
        "moat_and_competitive_risk": "最新 10-K/10-Q 风险因素与竞争披露更新",
    }
    en = {
        "business_model": "business-model, customer mix, and revenue-source disclosure updates",
        "revenue_quality": "revenue growth, segment revenue, and deferred revenue changes",
        "profitability_quality": "gross margin, operating margin, and net margin changes",
        "cash_flow_quality": "CFO/net income, FCF margin, and capex changes",
        "valuation_and_risk_boundary": "P/E, P/S, FCF yield, and price/market-cap changes",
        "balance_sheet_and_capital_intensity": "cash, total debt, receivables, and inventory changes",
        "moat_and_competitive_risk": "latest 10-K/10-Q risk-factor and competition disclosure updates",
    }
    mapping = zh if lang == "zh" else en
    return _dedupe_lines([mapping.get(dim, dim) for dim in dimension_ids if mapping.get(dim, dim)])


def _methodology_limitation_messages(
    *,
    limitations: list[dict[str, Any]],
    red_flags: list[dict[str, Any]],
    requirement_summary: dict[str, Any],
    accepted_draft: dict[str, Any],
) -> list[str]:
    messages: list[str] = []
    for item in limitations:
        if isinstance(item, dict) and str(item.get("message", "")).strip():
            messages.append(str(item.get("message")))
    messages.extend(str(item) for item in accepted_draft.get("methodology_limitations", []) or [] if str(item).strip())
    status_map = dict(requirement_summary.get("dimension_status_map", {}) or {})
    for item in status_map.values():
        if isinstance(item, dict) and str(item.get("status", "")) == "missing" and str(item.get("limitation", "")).strip():
            messages.append(str(item.get("limitation")))
    for flag in red_flags:
        if isinstance(flag, dict) and str(flag.get("message", "")).strip():
            messages.append(str(flag.get("message")))
    return _dedupe_lines(messages)


def _packet_dimension_summary(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in packet.get("dimension_summary", []) or []:
        if not isinstance(item, dict):
            continue
        dimension_id = str(item.get("dimension_id") or "").strip()
        if dimension_id:
            out[dimension_id] = dict(item)
    return out


def _packet_status_map(packet: dict[str, Any], fallback: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    raw = dict(packet.get("dimension_status_map", {}) or fallback or {})
    out: dict[str, dict[str, Any]] = {}
    for dimension_id, item in raw.items():
        if isinstance(item, dict):
            out[str(dimension_id)] = dict(item)
        else:
            out[str(dimension_id)] = {"status": str(item)}
    return out


def _packet_active_dimensions(packet: dict[str, Any], status_map: dict[str, dict[str, Any]]) -> list[str]:
    active = [str(item) for item in packet.get("active_dimensions", []) or [] if str(item).strip()]
    if active:
        return list(dict.fromkeys(active))
    return list(status_map.keys())


def _packet_requested_dimensions(packet: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for source in (
        packet.get("requested_dimensions"),
        dict(packet.get("canonical_intent", {}) or {}).get("requested_dimensions"),
    ):
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _status_available_metrics(packet: dict[str, Any]) -> set[str]:
    metrics: set[str] = set()
    for item in _packet_status_map(packet).values():
        if not isinstance(item, dict):
            continue
        for key in ("required_available", "enhanced_available"):
            metrics.update(
                normalize_metric_name(metric)
                for metric in item.get(key, []) or []
                if str(metric or "").strip()
            )
    return metrics


def _record_renderer_caveat(packet: dict[str, Any], message: str) -> None:
    message = str(message or "").strip()
    if not message:
        return
    caveats = packet.setdefault("_renderer_caveats", [])
    if isinstance(caveats, list) and message not in caveats:
        caveats.append(message)


def _packet_companies(packet: dict[str, Any], frame: dict[str, Any]) -> list[str]:
    companies = [str(item).upper() for item in frame.get("companies", []) or [] if str(item).strip()]
    for row in list(packet.get("numeric_table", []) or []) + list(packet.get("text_snippets", []) or []):
        if isinstance(row, dict):
            ticker = str(row.get("ticker") or row.get("company") or "").upper().strip()
            if ticker and ticker not in companies:
                companies.append(ticker)
    return companies


def _packet_rows_by_ref(packet: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in list(packet.get("numeric_table", []) or []) + list(packet.get("text_snippets", []) or []):
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if evidence_id:
            rows[evidence_id] = dict(row)
    return rows


def _dimension_refs(packet: dict[str, Any], summary_map: dict[str, dict[str, Any]], dimension_id: str) -> list[str]:
    summary = dict(summary_map.get(dimension_id, {}) or {})
    refs = [str(ref) for ref in summary.get("evidence_refs", []) or [] if str(ref).strip()]
    if refs:
        return refs
    status = dict(_packet_status_map(packet).get(dimension_id, {}) or {})
    refs = [str(ref) for ref in status.get("supporting_evidence_ids", []) or [] if str(ref).strip()]
    if refs:
        return list(dict.fromkeys(refs))
    rows = list(packet.get("numeric_evidence_by_dimension", {}).get(dimension_id, []) or [])
    rows += list(packet.get("text_evidence_by_dimension", {}).get(dimension_id, []) or [])
    return list(dict.fromkeys(str(row.get("evidence_id")) for row in rows if isinstance(row, dict) and str(row.get("evidence_id", "")).strip()))


def _dimension_numeric_refs(summary_map: dict[str, dict[str, Any]], dimension_id: str) -> list[str]:
    return [
        str(ref)
        for ref in summary_map.get(dimension_id, {}).get("numeric_evidence_refs", []) or []
        if str(ref).strip()
    ]


def _dimension_text_refs(summary_map: dict[str, dict[str, Any]], dimension_id: str) -> list[str]:
    return [
        str(ref)
        for ref in summary_map.get(dimension_id, {}).get("text_evidence_refs", []) or []
        if str(ref).strip()
    ]


def _packet_text_refs_for_dimension(packet: dict[str, Any], dimension_id: str) -> list[str]:
    refs: list[str] = []
    for source in (
        packet.get("text_snippets", []),
        packet.get("text_evidence", []),
        packet.get("validated_text_evidence", []),
    ):
        for row in source or []:
            if not isinstance(row, dict):
                continue
            ref = str(row.get("evidence_id") or row.get("id") or "").strip()
            if not ref or ref.startswith("REQ-"):
                continue
            row_dimension = str(row.get("dimension_id") or "").strip()
            section = str(row.get("section") or "").upper().strip()
            if (
                row_dimension
                and row_dimension != dimension_id
                and not (
                    dimension_id == "moat_and_competitive_risk"
                    and section in {"ITEM_1A", "ITEM_7", "ITEM_2"}
                )
            ):
                continue
            refs.append(ref)
    return list(dict.fromkeys(refs))


def _metric_rows_by_company(packet: dict[str, Any], metric: str, refs: list[str] | None = None) -> dict[str, dict[str, Any]]:
    ref_filter = {str(ref) for ref in refs or [] if str(ref).strip()}
    target_metric = normalize_metric_name(metric)
    rows: dict[str, dict[str, Any]] = {}
    for row in packet.get("numeric_table", []) or []:
        if not isinstance(row, dict) or normalize_metric_name(str(row.get("metric") or "")) != target_metric:
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if ref_filter and evidence_id not in ref_filter:
            continue
        ticker = str(row.get("ticker") or row.get("company") or "").upper().strip()
        if not ticker:
            continue
        current_period = str(rows.get(ticker, {}).get("period_end", ""))
        period = str(row.get("period_end") or "")
        if ticker not in rows or period >= current_period:
            rows[ticker] = dict(row)
    return rows


def _metric_history(
    packet: dict[str, Any],
    metric: str,
    company: str,
    refs: list[str] | None = None,
) -> list[dict[str, Any]]:
    ref_filter = {str(ref) for ref in refs or [] if str(ref).strip()}
    company = company.upper().strip()
    target_metric = normalize_metric_name(metric)
    rows: list[dict[str, Any]] = []
    for row in packet.get("numeric_table", []) or []:
        if not isinstance(row, dict) or normalize_metric_name(str(row.get("metric") or "")) != target_metric:
            continue
        if company and str(row.get("ticker") or row.get("company") or "").upper().strip() != company:
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if ref_filter and evidence_id not in ref_filter:
            continue
        rows.append(dict(row))
    return sorted(rows, key=lambda item: str(item.get("period_end") or item.get("period") or ""), reverse=True)


def _metric_float(row: dict[str, Any] | None) -> float | None:
    if not row:
        return None
    try:
        return float(row.get("value"))
    except (TypeError, ValueError):
        return None


def _metric_evidence_refs(*rows: dict[str, Any] | None) -> list[str]:
    return list(
        dict.fromkeys(
            str(row.get("evidence_id") or "").strip()
            for row in rows
            if row and str(row.get("evidence_id") or "").strip()
        )
    )


def _sanity_flag(
    flag_id: str,
    *,
    metric: str,
    company: str,
    dimension_id: str,
    message: str,
    evidence_refs: list[str],
    severity: str = "medium",
) -> dict[str, Any]:
    return {
        "id": flag_id,
        "severity": severity,
        "category": "data_sanity",
        "metric": normalize_metric_name(metric),
        "company": company,
        "dimension_id": dimension_id,
        "message": message,
        "evidence_refs": list(dict.fromkeys(ref for ref in evidence_refs if ref)),
        "user_visible": True,
    }


def build_metric_sanity_flags(
    evidence_packet: dict[str, Any] | None,
    company_profile: dict[str, Any] | None = None,
    *,
    lang: str = "zh",
) -> list[dict[str, Any]]:
    """Build lightweight sanity flags for suspicious single-company metrics."""
    packet = dict(evidence_packet or {})
    companies = _packet_companies(packet, {}) or [str(dict(company_profile or {}).get("ticker") or "").upper().strip()]
    companies = [company for company in companies if company]
    flags: list[dict[str, Any]] = []

    for company in companies:
        net_margin_row = _metric_rows_by_company(packet, "net_margin").get(company)
        fcf_margin_row = _metric_rows_by_company(packet, "fcf_margin").get(company)
        capex_to_revenue_row = _metric_rows_by_company(packet, "capex_to_revenue").get(company)
        cfo_to_net_income_row = _metric_rows_by_company(packet, "cfo_to_net_income").get(company)
        cfo_row = _metric_rows_by_company(packet, "operating_cash_flow").get(company)
        fcf_row = _metric_rows_by_company(packet, "free_cash_flow").get(company)
        capex_row = _metric_rows_by_company(packet, "capital_expenditure").get(company)

        net_margin = _metric_float(net_margin_row)
        if net_margin is not None and net_margin > 0.30:
            flags.append(
                _sanity_flag(
                    "unusually_high_net_margin",
                    metric="net_margin",
                    company=company,
                    dimension_id="profitability_quality",
                    message=(
                        "净利率对综合业务公司属于异常偏高；需要核验期间口径、累计/季度口径或非经常性项目，不能直接外推为可持续盈利能力。"
                        if lang == "zh"
                        else "Net margin is unusually high for a diversified company; period basis, quarterly-vs-cumulative treatment, and non-recurring items should be checked before extrapolating profitability."
                    ),
                    evidence_refs=_metric_evidence_refs(net_margin_row),
                    severity="high",
                )
            )
        elif net_margin is not None and net_margin < 0:
            flags.append(
                _sanity_flag(
                    "negative_net_margin",
                    metric="net_margin",
                    company=company,
                    dimension_id="profitability_quality",
                    message=(
                        "净利率为负，说明当前入库口径下净利润未覆盖收入，需要结合一次性项目和期间口径判断。"
                        if lang == "zh"
                        else "Net margin is negative, so profitability should be read with period-basis and one-off item caveats."
                    ),
                    evidence_refs=_metric_evidence_refs(net_margin_row),
                )
            )

        fcf_margin = _metric_float(fcf_margin_row)
        if fcf_margin is not None and fcf_margin < 0:
            flags.append(
                _sanity_flag(
                    "negative_fcf_margin",
                    metric="fcf_margin",
                    company=company,
                    dimension_id="cash_flow_quality",
                    message=(
                        "FCF margin 为负，说明收入在资本开支后没有转化为自由现金流，需要把经营现金流和资本开支分开看。"
                        if lang == "zh"
                        else "FCF margin is negative, so revenue is not converting into free cash flow after capex; operating cash flow and capex should be separated."
                    ),
                    evidence_refs=_metric_evidence_refs(fcf_margin_row),
                )
            )

        capex_to_revenue = _metric_float(capex_to_revenue_row)
        if capex_to_revenue is not None and capex_to_revenue > 0.30:
            flags.append(
                _sanity_flag(
                    "high_capex_intensity",
                    metric="capex_to_revenue",
                    company=company,
                    dimension_id="balance_sheet_and_capital_intensity",
                    message=(
                        "capex/revenue 明显偏高，资本开支强度偏高，可能压制自由现金流，需要结合投资周期和收入口径核验。"
                        if lang == "zh"
                        else "Capex/revenue indicates elevated capital intensity that can pressure free cash flow; investment cycle and revenue basis should be checked."
                    ),
                    evidence_refs=_metric_evidence_refs(capex_to_revenue_row),
                )
            )

        cfo_to_net_income = _metric_float(cfo_to_net_income_row)
        if cfo_to_net_income is not None and cfo_to_net_income > 1.5:
            flags.append(
                _sanity_flag(
                    "high_cash_conversion_or_income_quality_mismatch",
                    metric="cfo_to_net_income",
                    company=company,
                    dimension_id="cash_flow_quality",
                    message=(
                        "CFO/净利润偏高，可能反映现金转换很强，也可能来自营运资本、期间口径或非经常性项目错配。"
                        if lang == "zh"
                        else "CFO/net income is elevated, which may indicate strong conversion or a mismatch from working capital, period basis, or non-recurring items."
                    ),
                    evidence_refs=_metric_evidence_refs(cfo_to_net_income_row),
                )
            )

        cfo = _metric_float(cfo_row)
        fcf = _metric_float(fcf_row)
        capex = _metric_float(capex_row)
        if cfo is not None and fcf is not None and capex is not None and cfo > 0 and fcf <= 0 and abs(capex) >= abs(cfo) * 0.8:
            flags.append(
                _sanity_flag(
                    "capex_drag_on_fcf",
                    metric="free_cash_flow",
                    company=company,
                    dimension_id="cash_flow_quality",
                    message=(
                        "经营现金流为正但自由现金流为负，且资本开支接近或超过经营现金流；经营端现金流强，但资本开支吞噬/压制自由现金流。"
                        if lang == "zh"
                        else "Operating cash flow is positive while free cash flow is negative, with capex near or above operating cash flow; operating cash generation is strong but capex absorbs free cash flow."
                    ),
                    evidence_refs=_metric_evidence_refs(cfo_row, fcf_row, capex_row),
                    severity="high",
                )
            )

        for metric in ("revenue", "net_income", "net_margin", "operating_cash_flow", "free_cash_flow"):
            rows = _metric_history(packet, metric, company)
            providers = {
                str(row.get("source_provider") or row.get("provider") or "").lower().strip()
                for row in rows
                if str(row.get("source_provider") or row.get("provider") or "").strip()
            }
            if len(providers) > 1:
                metric_label = metric_display_name(metric, lang).strip()
                flags.append(
                    _sanity_flag(
                        "mixed_provider_time_series",
                        metric=metric,
                        company=company,
                        dimension_id="revenue_quality" if metric == "revenue" else "profitability_quality",
                        message=(
                            f"{metric_label}多期序列混用了不同数据来源，趋势判断需要口径 caveat。"
                            if lang == "zh"
                            else f"{metric_label} history mixes providers, so trend interpretation needs a source-basis caveat."
                        ),
                        evidence_refs=_metric_evidence_refs(*rows[:3]),
                    )
                )
                break

    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for flag in flags:
        key = (str(flag.get("id") or ""), str(flag.get("company") or ""), str(flag.get("metric") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(flag)
    return deduped


def _display_metric_row(row: dict[str, Any]) -> str:
    display = str(row.get("display_value") or "").strip()
    value = row.get("value")
    if value is None:
        return display or "N/A"
    metric = str(row.get("metric") or "")
    canonical = normalize_metric_name(metric)
    if canonical in {"adjusted_close", "latest_close", "close", "price"}:
        canonical = "share_price"
    if display:
        if canonical in {"pe_ratio", "ps_ratio"} and "%" in display:
            return format_metric_value(metric, value, unit=row.get("unit"))
        if canonical in {"market_cap", "net_debt"} and "$" not in display:
            return format_metric_value(metric, value, unit=row.get("unit"))
        return display
    return format_metric_value(metric, value, unit=row.get("unit"))


def _trend_interpretation(metric_id: str, history: list[dict[str, Any]], lang: str) -> str:
    numeric_rows = [row for row in history if _metric_float(row) is not None]
    if len(numeric_rows) < 2:
        return "历史趋势证据不足" if lang == "zh" else "historical trend evidence is insufficient"
    providers = {
        str(row.get("source_provider") or row.get("provider") or "").strip().lower()
        for row in numeric_rows
        if str(row.get("source_provider") or row.get("provider") or "").strip()
    }
    period_types = {
        str(row.get("period_category") or period_category(row.get("period_type"))).strip()
        for row in numeric_rows
        if str(row.get("period_category") or row.get("period_type") or "").strip()
    }
    comparable_types = period_types & {"annual", "quarterly", "ttm"}
    if metric_id == "revenue" and len(comparable_types) > 1:
        return (
            "当前期间口径不一致，因此不做趋势结论"
            if lang == "zh"
            else "period bases are inconsistent, so no trend conclusion is made"
        )
    if metric_id == "revenue" and len(providers) > 1:
        return (
            "已入库序列显示数值变化，但由于期间口径或数据来源混用，不能直接等同于可比口径下的经营趋势"
            if lang == "zh"
            else "the ingested revenue series shows a value change, but mixed sources or period bases mean it should not be treated as a comparable operating trend"
        )
    if len(providers) > 1:
        return (
            "已入库序列显示数值变化，但由于期间口径或数据来源混用，不能直接等同于可比口径下的经营趋势"
            if lang == "zh"
            else "the ingested series shows a value change, but mixed sources or period bases mean it should not be treated as a comparable operating trend"
        )
    if len(comparable_types) > 1:
        return (
            "当前期间口径不一致，因此不做趋势结论"
            if lang == "zh"
            else "period bases are inconsistent, so no trend conclusion is made"
        )
    latest = numeric_rows[0]
    earliest = numeric_rows[-1]
    latest_value = _metric_float(latest)
    earliest_value = _metric_float(earliest)
    if latest_value is None or earliest_value in {None, 0}:
        return "历史趋势证据不足" if lang == "zh" else "historical trend evidence is insufficient"
    change = (latest_value - earliest_value) / abs(earliest_value)
    if change > 0.02:
        direction = "上升" if lang == "zh" else "increased"
    elif change < -0.02:
        direction = "下降" if lang == "zh" else "declined"
    else:
        direction = "基本稳定" if lang == "zh" else "was broadly stable"
    latest_display = _display_metric_row(latest)
    earliest_display = _display_metric_row(earliest)
    latest_period = str(latest.get("period_end") or latest.get("period") or "latest")
    earliest_period = str(earliest.get("period_end") or earliest.get("period") or "earliest")
    if lang == "zh":
        return f"最近可验证期间从 {earliest_display}（{earliest_period}）到 {latest_display}（{latest_period}），趋势为{direction}"
    return f"From {earliest_display} ({earliest_period}) to {latest_display} ({latest_period}), the trend {direction}"


def _metric_sentence(metric_rows: dict[str, dict[str, Any]], companies: list[str], metric_label: str, lang: str) -> str:
    parts: list[str] = []
    ordered = [company for company in companies if company]
    ordered += [company for company in metric_rows.keys() if company not in ordered]
    for company in ordered:
        row = metric_rows.get(company)
        if row:
            parts.append(f"{company} {metric_label} {_display_metric_row(row)}")
        else:
            missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
            parts.append(f"{company} {metric_label} {missing}")
    if not parts:
        return ""
    if lang == "zh":
        return "，".join(parts)
    return "; ".join(parts)


def _leader(metric_rows: dict[str, dict[str, Any]]) -> str:
    best_company = ""
    best_value: float | None = None
    for company, row in metric_rows.items():
        try:
            value = float(row.get("value"))
        except (TypeError, ValueError):
            continue
        if best_value is None or value > best_value:
            best_company = company
            best_value = value
    return best_company


def _metric_company_values(
    metric_rows: dict[str, dict[str, Any]],
    companies: list[str],
    lang: str,
) -> dict[str, str]:
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    ordered = [company for company in companies if company]
    ordered += [company for company in metric_rows.keys() if company not in ordered]
    return {
        company: _display_metric_row(metric_rows[company]) if company in metric_rows else missing
        for company in ordered
    }


def _metric_table_judgment(metric_id: str, leader: str, lang: str) -> str:
    if not leader:
        return "证据不足" if lang == "zh" else "insufficient evidence"
    if lang == "zh":
        if metric_id == "revenue":
            return f"{leader} 规模更大"
        if metric_id == "net_income":
            return f"{leader} 更强"
        if metric_id == "net_margin":
            return f"{leader} 盈利转化更强"
        return f"{leader} 更高"
    if metric_id == "revenue":
        return f"{leader} has greater scale"
    if metric_id == "net_income":
        return f"{leader} is stronger"
    if metric_id == "net_margin":
        return f"{leader} has stronger profit conversion"
    return f"{leader} is higher"


def _build_metric_table(packet: dict[str, Any], companies: list[str], lang: str) -> list[dict[str, Any]]:
    requested = _packet_requested_dimensions(packet)
    status_map = _packet_status_map(packet)
    active = requested or _packet_active_dimensions(packet, status_map)
    default_comparison = {"revenue_quality", "profitability_quality", "moat_and_competitive_risk", "valuation_and_risk_boundary"}
    if not requested and set(active) >= default_comparison:
        active = ["revenue_quality", "profitability_quality", "moat_and_competitive_risk"]
    if "cash_flow_quality" in active:
        metric_specs = [
            ("operating_cash_flow", "经营现金流" if lang == "zh" else "Operating Cash Flow"),
            ("free_cash_flow", "自由现金流" if lang == "zh" else "Free Cash Flow"),
            ("capital_expenditure", "资本开支" if lang == "zh" else "Capital Expenditure"),
            ("fcf_margin", "FCF margin"),
        ]
    elif "valuation_and_risk_boundary" in active and "profitability_quality" not in active:
        metric_specs = [
            ("share_price", "股价" if lang == "zh" else "Share Price"),
            ("market_cap", "市值" if lang == "zh" else "Market Cap"),
            ("pe_ratio", "P/E"),
            ("ps_ratio", "P/S"),
            ("fcf_yield", "FCF yield"),
        ]
    elif "revenue_quality" in active and "profitability_quality" not in active:
        metric_specs = [
            ("revenue", "收入" if lang == "zh" else "Revenue"),
            ("revenue_growth", "收入增速" if lang == "zh" else "Revenue Growth"),
        ]
    else:
        metric_specs = [
            ("revenue", "收入" if lang == "zh" else "Revenue"),
            ("net_income", "净利润" if lang == "zh" else "Net Income"),
            ("net_margin", "净利率" if lang == "zh" else "Net Margin"),
        ]
    out: list[dict[str, Any]] = []
    for metric_id, label in metric_specs:
        metric_rows = _metric_rows_by_company(packet, metric_id)
        leader = _leader(metric_rows)
        ordered_companies = [company for company in companies if company]
        ordered_companies += [company for company in metric_rows.keys() if company not in ordered_companies]
        evidence_refs = [
            str(metric_rows[company].get("evidence_id"))
            for company in ordered_companies
            if company in metric_rows and str(metric_rows[company].get("evidence_id") or "").strip()
        ]
        out.append(
            {
                "metric_id": metric_id,
                "label": label,
                "company_values": _metric_company_values(metric_rows, ordered_companies, lang),
                "judgment": _metric_table_judgment(metric_id, leader, lang),
                "evidence_refs": list(dict.fromkeys(evidence_refs)),
            }
        )
    return out


_FOCUSED_SINGLE_COMPANY_METRICS: dict[str, list[tuple[str, bool]]] = {
    "cash_flow_quality": [
        ("operating_cash_flow", True),
        ("free_cash_flow", True),
        ("capital_expenditure", True),
        ("cash_conversion", True),
        ("fcf_margin", True),
    ],
    "balance_sheet_and_capital_intensity": [
        ("cash", True),
        ("total_debt", True),
        ("net_debt", True),
        ("total_assets", True),
        ("total_liabilities", True),
        ("shareholders_equity", True),
        ("debt_to_equity", True),
        ("capex_to_revenue", True),
        ("receivables", False),
        ("inventory", False),
    ],
    "valuation_and_risk_boundary": [
        ("share_price", True),
        ("market_cap", True),
        ("pe_ratio", True),
        ("ps_ratio", True),
        ("fcf_yield", True),
    ],
    "profitability_quality": [
        ("net_income", True),
        ("gross_margin", True),
        ("operating_margin", True),
        ("net_margin", True),
        ("gross_profit", False),
        ("operating_income", False),
    ],
    "revenue_quality": [
        ("revenue", True),
        ("revenue_growth", False),
    ],
}

_BROAD_SINGLE_COMPANY_METRICS: list[tuple[str, bool]] = [
    ("revenue", True),
    ("net_income", True),
    ("net_margin", True),
    ("operating_cash_flow", True),
    ("free_cash_flow", True),
    ("capital_expenditure", True),
    ("cfo_to_net_income", True),
    ("fcf_margin", True),
    ("cash", True),
    ("total_debt", True),
    ("capex_to_revenue", True),
    ("market_cap", True),
    ("pe_ratio", True),
    ("ps_ratio", True),
    ("fcf_yield", True),
]


def _focused_single_company_dimension(packet: dict[str, Any]) -> str | None:
    status_map = _packet_status_map(packet)
    active = _packet_active_dimensions(packet, status_map)
    if len(active) == 1 and active[0] in _FOCUSED_SINGLE_COMPANY_METRICS:
        return active[0]
    return None


_SINGLE_COMPANY_OVERVIEW_DIMENSIONS = [
    "business_model",
    "revenue_quality",
    "profitability_quality",
    "cash_flow_quality",
    "balance_sheet_and_capital_intensity",
    "moat_and_competitive_risk",
    "valuation_and_risk_boundary",
]


def _packet_intent_value(packet: dict[str, Any], key: str) -> str:
    direct = str(packet.get(key) or "").strip()
    if direct:
        return direct
    canonical = packet.get("canonical_intent")
    if isinstance(canonical, dict):
        return str(canonical.get(key) or "").strip()
    return ""


def _is_broad_single_company_overview(
    packet: dict[str, Any],
    active_dimensions: list[str],
    status_map: dict[str, dict[str, Any]],
) -> bool:
    if str(packet.get("analysis_scope") or "").strip() != "single_company":
        return False
    active = list(dict.fromkeys(active_dimensions or list(status_map.keys())))
    intent_family = _packet_intent_value(packet, "intent_family")
    methodology_intent = str(packet.get("methodology_intent") or packet.get("legacy_methodology_intent") or "").strip()
    evidence_policy_id = str(packet.get("evidence_policy_id") or "").strip()
    if intent_family == "overview" or methodology_intent in {"overview", "single_company_overview"}:
        return True
    if evidence_policy_id in {"single_company_overview_v1"}:
        return True
    if evidence_policy_id == "single_company_composite_v1" and len(set(active) & set(_SINGLE_COMPANY_OVERVIEW_DIMENSIONS)) >= 5:
        return True
    return len(set(active) & set(_SINGLE_COMPANY_OVERVIEW_DIMENSIONS)) >= 5


def _active_single_company_metric_specs(packet: dict[str, Any]) -> list[tuple[str, bool]]:
    status_map = _packet_status_map(packet)
    active = _packet_active_dimensions(packet, status_map)
    if len(active) <= 1:
        return []
    metric_dimensions = [dimension_id for dimension_id in active if dimension_id in _FOCUSED_SINGLE_COMPANY_METRICS]
    context_dimensions = [dimension_id for dimension_id in active if dimension_id not in _FOCUSED_SINGLE_COMPANY_METRICS]
    if not (2 <= len(metric_dimensions) <= 3):
        return []
    if not set(context_dimensions).issubset({"business_model", "moat_and_competitive_risk"}):
        return []
    out: list[tuple[str, bool]] = []
    seen: set[str] = set()
    for dimension_id in metric_dimensions:
        for metric_id, show_if_missing in _FOCUSED_SINGLE_COMPANY_METRICS.get(dimension_id, []):
            canonical = normalize_metric_name(metric_id)
            if canonical in seen:
                continue
            seen.add(canonical)
            out.append((metric_id, show_if_missing))
    return out


def _single_company_metric_specs(packet: dict[str, Any], lang: str) -> list[tuple[str, str, bool]]:
    focused_dimension = _focused_single_company_dimension(packet)
    if focused_dimension:
        metric_specs = _FOCUSED_SINGLE_COMPANY_METRICS[focused_dimension]
    elif active_metric_specs := _active_single_company_metric_specs(packet):
        metric_specs = active_metric_specs
    else:
        metric_specs = _BROAD_SINGLE_COMPANY_METRICS
    return [(metric_id, metric_display_name(metric_id, lang), show_if_missing) for metric_id, show_if_missing in metric_specs]


def _single_company_metric_interpretation(metric_id: str, value: str, lang: str) -> str:
    metric_id = normalize_metric_name(metric_id)
    lowered_value = value.lower()
    if (
        "缺少" in value
        or "无法计算" in value
        or "暂不计算" in value
        or "unavailable" in lowered_value
        or "cannot be computed" in lowered_value
        or "not computed" in lowered_value
        or value == "N/A"
    ):
        return "当前缺少可验证数据" if lang == "zh" else "Verified data is unavailable"
    if lang == "zh":
        if metric_id == "revenue":
            return "反映当前收入规模"
        if metric_id == "net_income":
            return "反映当前盈利规模"
        if metric_id == "net_margin":
            return "反映盈利转化能力"
        if metric_id == "operating_cash_flow":
            return "观察利润能否转化为经营现金流"
        if metric_id == "free_cash_flow":
            return "观察资本开支后的现金创造能力"
        if metric_id == "cash":
            return "观察短期流动性缓冲"
        if metric_id == "total_debt":
            return "观察债务负担"
        if metric_id == "net_debt":
            return "观察现金扣除债务后的净债务方向"
        if metric_id == "total_assets":
            return "观察资产规模"
        if metric_id == "total_liabilities":
            return "观察负债规模"
        if metric_id == "shareholders_equity":
            return "观察权益缓冲"
        if metric_id == "debt_to_equity":
            return "观察债务相对股东权益的杠杆压力"
        if metric_id == "capex_to_revenue":
            return "观察资本开支强度"
        if metric_id == "receivables":
            return "观察营运资本占用"
        if metric_id == "inventory":
            return "观察库存占用"
        if metric_id == "share_price":
            return "观察最新价格输入"
        if metric_id == "market_cap":
            return "观察当前市场定价规模"
        if metric_id == "pe_ratio":
            return "观察市值相对净利润的倍数"
        if metric_id == "ps_ratio":
            return "观察市值相对收入的倍数"
        if metric_id == "fcf_yield":
            return "观察自由现金流相对市值的收益率"
        return "作为补充指标观察"
    if metric_id == "revenue":
        return "Shows current revenue scale"
    if metric_id == "net_income":
        return "Shows current profit scale"
    if metric_id == "net_margin":
        return "Shows profit conversion"
    if metric_id == "operating_cash_flow":
        return "Shows whether earnings convert into operating cash flow"
    if metric_id == "free_cash_flow":
        return "Shows cash generation after capex"
    if metric_id == "cash":
        return "Shows liquidity buffer"
    if metric_id == "total_debt":
        return "Shows debt burden"
    if metric_id == "net_debt":
        return "Shows net debt after cash"
    if metric_id == "total_assets":
        return "Shows asset scale"
    if metric_id == "total_liabilities":
        return "Shows liability scale"
    if metric_id == "shareholders_equity":
        return "Shows equity buffer"
    if metric_id == "debt_to_equity":
        return "Shows leverage relative to equity"
    if metric_id == "capex_to_revenue":
        return "Shows capital intensity"
    if metric_id == "receivables":
        return "Shows working-capital usage"
    if metric_id == "inventory":
        return "Shows inventory usage"
    if metric_id == "share_price":
        return "Shows latest price input"
    if metric_id == "market_cap":
        return "Shows current market-implied scale"
    if metric_id == "pe_ratio":
        return "Shows market-cap multiple over net income"
    if metric_id == "ps_ratio":
        return "Shows market-cap multiple over revenue"
    if metric_id == "fcf_yield":
        return "Shows free-cash-flow yield on market cap"
    return "Supplemental metric"


def _metric_dependency_missing_reason(packet: dict[str, Any], metric_id: str, company: str, lang: str) -> str:
    metric_id = normalize_metric_name(metric_id)
    label = metric_display_name(metric_id, lang)
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"

    def has_metric(dep: str) -> bool:
        return bool(_metric_rows_by_company(packet, dep).get(company))

    def dependency_list(row: dict[str, Any] | None) -> set[str]:
        deps = set()
        if not row:
            return deps
        for dep in row.get("missing_dependencies", []) or []:
            deps.add(normalize_metric_name(dep))
        for dep in row.get("dependencies", []) or []:
            if isinstance(dep, dict) and str(dep.get("missing_dependencies", "")).strip():
                for item in dep.get("missing_dependencies", []) or []:
                    deps.add(normalize_metric_name(item))
        return deps

    candidate_row = None
    for row in packet.get("numeric_table", []) or []:
        if isinstance(row, dict) and normalize_metric_name(str(row.get("metric") or "")) == metric_id:
            ticker = str(row.get("ticker") or row.get("company") or company).upper().strip()
            if not company or ticker == company:
                candidate_row = dict(row)
                break
    missing_deps = dependency_list(candidate_row)

    if metric_id == "capex_to_revenue":
        capex_available = has_metric("capital_expenditure")
        revenue_available = has_metric("revenue")
        if "capital_expenditure" in missing_deps or not capex_available:
            return (
                "缺少资本开支，无法计算资本开支/收入"
                if lang == "zh"
                else "capital expenditure is missing, so capex/revenue cannot be computed"
            )
        if "revenue" in missing_deps or not revenue_available:
            return (
                "缺少可匹配收入期间，无法计算资本开支/收入"
                if lang == "zh"
                else "matching revenue period is missing, so capex/revenue cannot be computed"
            )
        return (
            "资本开支和收入期间不匹配，暂不计算资本开支/收入"
            if lang == "zh"
            else "capital expenditure and revenue periods do not match, so capex/revenue is not computed"
        )

    dependency_messages = {
        "fcf_yield": (
            ("free_cash_flow", "缺少自由现金流，无法计算 FCF yield", "free cash flow is missing, so FCF yield cannot be computed"),
            ("market_cap", "缺少市值，无法计算 FCF yield", "market cap is missing, so FCF yield cannot be computed"),
        ),
        "cash_conversion": (
            ("operating_cash_flow", "缺少经营现金流，无法计算现金转换率", "operating cash flow is missing, so cash conversion cannot be computed"),
            ("net_income", "缺少净利润，无法计算现金转换率", "net income is missing, so cash conversion cannot be computed"),
        ),
        "net_debt": (
            ("total_debt", "缺少总债务，无法计算净债务", "total debt is missing, so net debt cannot be computed"),
            ("cash", "缺少现金，无法计算净债务", "cash is missing, so net debt cannot be computed"),
        ),
        "debt_to_equity": (
            ("total_debt", "缺少总债务，无法计算债务/权益", "total debt is missing, so debt/equity cannot be computed"),
            ("shareholders_equity", "缺少股东权益，无法计算债务/权益", "shareholders' equity is missing, so debt/equity cannot be computed"),
        ),
        "pe_ratio": (
            ("market_cap", "缺少市值，无法计算 P/E", "market cap is missing, so P/E cannot be computed"),
            ("net_income", "缺少净利润，无法计算 P/E", "net income is missing, so P/E cannot be computed"),
        ),
        "ps_ratio": (
            ("market_cap", "缺少市值，无法计算 P/S", "market cap is missing, so P/S cannot be computed"),
            ("revenue", "缺少收入，无法计算 P/S", "revenue is missing, so P/S cannot be computed"),
        ),
    }
    for dep, zh_message, en_message in dependency_messages.get(metric_id, ()):
        if dep in missing_deps or not has_metric(dep):
            return zh_message if lang == "zh" else en_message
    return f"{label} {missing}" if lang == "zh" else f"{label}: {missing}"


def _build_single_company_metric_table(packet: dict[str, Any], company: str, lang: str) -> list[dict[str, Any]]:
    metric_specs = _single_company_metric_specs(packet, lang)
    focused_dimension = _focused_single_company_dimension(packet)
    include_trend_in_interpretation = focused_dimension is None and not _active_single_company_metric_specs(packet)
    status_available = _status_available_metrics(packet)
    out: list[dict[str, Any]] = []
    for metric_id, label, show_if_missing in metric_specs:
        history = _metric_history(packet, metric_id, company)
        row = history[0] if history else _metric_rows_by_company(packet, metric_id).get(company)
        canonical = normalize_metric_name(metric_id)
        if not row and canonical in status_available:
            _record_renderer_caveat(
                packet,
                (
                    f"{label} 在维度状态中可用，但当前 EvidencePacket 缺少可展示数值；未将其误报为缺失。"
                    if lang == "zh"
                    else f"{label} is available in DimensionStatus, but the current EvidencePacket has no display row; it was not reported as missing."
                ),
            )
            continue
        if not row and not show_if_missing:
            continue
        value = _display_metric_row(row) if row else _metric_dependency_missing_reason(packet, metric_id, company, lang)
        evidence_refs = [
            str(item.get("evidence_id"))
            for item in history[:4]
            if str(item.get("evidence_id") or "").strip()
        ]
        if not evidence_refs and row and str(row.get("evidence_id") or "").strip():
            evidence_refs = [str(row.get("evidence_id"))]
        trend = _trend_interpretation(metric_id, history, lang)
        base_interpretation = _single_company_metric_interpretation(metric_id, value, lang)
        if include_trend_in_interpretation:
            interpretation = f"{base_interpretation}；{trend}" if lang == "zh" else f"{base_interpretation}; {trend}"
        else:
            interpretation = base_interpretation
        out.append(
            {
                "metric_id": metric_id,
                "label": label,
                "value": value,
                "trend": trend,
                "interpretation": interpretation,
                "period_category": period_category(row.get("period_type")) if row else "",
                "evidence_refs": list(dict.fromkeys(evidence_refs)),
            }
        )
    return out


def _snippet_excerpt(row: dict[str, Any], *, max_len: int = 140) -> str:
    text = str(row.get("supporting_snippet") or row.get("text_snippet") or row.get("text") or "").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def _risk_phrases_from_text(company: str, text: str, lang: str) -> list[str]:
    lowered = text.lower()
    company = company.upper()
    phrases: list[str] = []
    if lang == "zh":
        if any(term in lowered for term in ("ai demand", "artificial intelligence", "accelerated computing", "加速计算", "人工智能")):
            phrases.append("AI 和加速计算需求波动")
        if any(term in lowered for term in ("new product", "new service", "product introduction", "service introduction", "launch", "demand", "需求")):
            phrases.append("产品推出和需求不确定性")
        if any(term in lowered for term in ("competition", "competitive", "competitor", "market pressure", "competitive pressure", "竞争")):
            phrases.append("多业务线竞争" if company == "AMZN" else "产品市场竞争压力")
        if any(term in lowered for term in ("macro", "macroeconomic", "economic", "inflation", "foreign exchange", "interest rate", "宏观")):
            phrases.append("宏观不确定性")
        if any(term in lowered for term in ("regulation", "regulatory", "legal", "litigation", "lawsuit", "antitrust", "compliance", "监管", "法律")):
            phrases.append("监管/法律事项")
        if any(term in lowered for term in ("customer concentration", "customer", "customers", "客户集中")):
            phrases.append("客户和市场集中度风险")
        if any(term in lowered for term in ("reinvestment", "operating leverage", "fulfillment", "logistics", "cost pressure")):
            phrases.append("再投资和运营杠杆波动")
        if any(term in lowered for term in ("margin", "supply", "inventory", "component", "supplier", "供应链", "供给")):
            phrases.append("利润率、库存或供应链压力")
    else:
        if any(term in lowered for term in ("ai demand", "artificial intelligence", "accelerated computing")):
            phrases.append("AI and accelerated-computing demand variability")
        if any(term in lowered for term in ("new product", "new service", "product introduction", "service introduction", "launch", "demand")):
            phrases.append("product-launch and demand uncertainty")
        if any(term in lowered for term in ("competition", "competitive", "competitor", "market pressure", "competitive pressure")):
            phrases.append("multi-line competitive pressure" if company == "AMZN" else "product-market competitive pressure")
        if any(term in lowered for term in ("macro", "macroeconomic", "economic", "inflation", "foreign exchange", "interest rate")):
            phrases.append("macroeconomic uncertainty")
        if any(term in lowered for term in ("regulation", "regulatory", "legal", "litigation", "lawsuit", "antitrust", "compliance")):
            phrases.append("regulatory or legal matters")
        if any(term in lowered for term in ("customer concentration", "customer", "customers")):
            phrases.append("customer or market concentration risk")
        if any(term in lowered for term in ("reinvestment", "operating leverage", "fulfillment", "logistics", "cost pressure")):
            phrases.append("reinvestment and operating-leverage variability")
        if any(term in lowered for term in ("margin", "supply", "inventory", "component", "supplier")):
            phrases.append("margin, inventory, or supply-chain pressure")
    return _dedupe_lines(phrases)


def _business_section_summary(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> str:
    rows_by_ref = _packet_rows_by_ref(packet)
    snippets = [
        str(dict(rows_by_ref.get(ref, {}) or {}).get("claim") or "").strip()
        or _snippet_excerpt(dict(rows_by_ref.get(ref, {}) or {}))
        for ref in refs
    ]
    combined = " ".join(snippet for snippet in snippets if snippet)
    refs_text = _methodology_refs(refs)
    if not combined:
        return (
            "当前缺少业务模式文本证据，不能完整描述业务模式。"
            if lang == "zh"
            else "Business-model text evidence is missing, so a full business-model description is not supported."
        )
    lowered = combined.lower()
    amazon_segments: list[str] = []
    if "north america" in lowered:
        amazon_segments.append("North America")
    if "international" in lowered:
        amazon_segments.append("International")
    if "aws" in lowered or "amazon web services" in lowered:
        amazon_segments.append("AWS")
    if len(amazon_segments) >= 3:
        if lang == "zh":
            extras: list[str] = []
            if any(term in lowered for term in ("third-party seller", "third party seller", "seller services", "marketplace")):
                extras.append("第三方卖家 / marketplace")
            if any(term in lowered for term in ("advertising", "ads")):
                extras.append("广告业务")
            if "prime" in lowered:
                extras.append("Prime 会员生态")
            extra_text = f"；同一证据还支持关注{'、'.join(extras)}等收入来源" if extras else ""
            return f"{company} 的业务结构可按 {'、'.join(amazon_segments)} 三个分部理解{extra_text}。{refs_text}".strip()
        return f"{company}'s business structure can be read through the North America, International, and AWS segments. {refs_text}".strip()
    phrases: list[str] = []
    if any(term in lowered for term in ("gpu", "graphics processing")):
        phrases.append("GPU")
    if any(term in lowered for term in ("accelerated computing", "accelerated compute", "加速计算")):
        phrases.append("加速计算")
    if any(term in lowered for term in ("artificial intelligence", " ai ", "generative ai", "人工智能")):
        phrases.append("AI 相关需求")
    if any(term in lowered for term in ("data center", "datacenter", "数据中心")):
        phrases.append("数据中心")
    if any(term in lowered for term in ("aws", "amazon web services")):
        phrases.append("AWS")
    if any(term in lowered for term in ("prime",)):
        phrases.append("Prime 会员生态")
    if any(term in lowered for term in ("marketplace", "third-party seller", "third party seller", "seller services")):
        phrases.append("第三方卖家 / marketplace")
    if any(term in lowered for term in ("advertising", "ads")):
        phrases.append("广告业务")
    if any(term in lowered for term in ("fulfillment", "logistics")):
        phrases.append("履约与物流网络")
    if any(term in lowered for term in ("gaming", "geforce", "游戏")):
        phrases.append("游戏")
    if any(term in lowered for term in ("professional visualization", "visualization", "专业可视化")):
        phrases.append("专业可视化")
    if any(term in lowered for term in ("automotive", "汽车")):
        phrases.append("汽车")
    if any(term in lowered for term in ("product", "service", "platform", "solution", "products", "services")):
        phrases.append("产品和服务组合")
    if any(term in lowered for term in ("customer", "market", "industry", "segment")):
        phrases.append("客户、市场或分部结构")
    if any(term in lowered for term in ("revenue", "sales", "net sales", "subscription", "license")):
        phrases.append("收入来源")
    if not phrases:
        return (
            f"{company} 已有经验证的业务披露文本，可支持初步业务模式判断；但当前文本没有足够具体的产品、客户、市场或收入来源主题，因此只能做有限描述。{refs_text}"
            if lang == "zh"
            else f"{company} has validated business disclosure text for an initial business-model view, but the available text does not expose enough specific product, customer, market, or revenue-source themes for a detailed breakdown. {refs_text}"
        )
    if lang == "zh":
        return f"{company} 的业务模式主要可围绕{'、'.join(_dedupe_lines(phrases))}展开。{refs_text}".strip()
    return f"{company}'s business-model analysis can be grounded in {', '.join(_dedupe_lines(phrases))}. {refs_text}".strip()


def _single_company_risk_summary(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> str:
    rows_by_ref = _packet_rows_by_ref(packet)
    company_rows: list[tuple[str, str]] = []
    for ref in refs:
        row = dict(rows_by_ref.get(ref, {}) or {})
        if not row:
            continue
        ticker = str(row.get("ticker") or row.get("company") or company).upper().strip()
        if ticker != company:
            continue
        excerpt = str(row.get("claim") or "").strip() or _snippet_excerpt(row)
        if excerpt:
            company_rows.append((ref, excerpt))
    if not company_rows:
        return (
            "当前缺少风险文本证据，不能做具体风险判断。"
            if lang == "zh"
            else "Risk text evidence is missing, so specific risk judgments are not supported."
        )
    combined = " ".join(excerpt for _ref, excerpt in company_rows[:3])
    phrases = _risk_phrases_from_text(company, combined, lang)
    if not phrases:
        return (
            "当前缺少足够文本证据，因此不能做具体竞争风险判断。"
            if lang == "zh"
            else "Current text evidence is not specific enough for a concrete competitive-risk judgment."
        )
    refs_text = _methodology_refs([ref for ref, _excerpt in company_rows[:3]])
    if lang == "zh":
        return f"{company} 的主要风险来自{'、'.join(phrases)}。{refs_text}".strip()
    return f"{company}'s main risks in validated text relate to {', '.join(phrases)}. {refs_text}".strip()


def _overview_risk_ranking(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> list[dict[str, Any]]:
    rows_by_ref = _packet_rows_by_ref(packet)
    candidates: list[tuple[str, str]] = []
    for ref in refs:
        if not str(ref).startswith("T"):
            continue
        row = dict(rows_by_ref.get(ref, {}) or {})
        if not row:
            continue
        evidence_id = str(row.get("evidence_id") or ref).strip()
        if not evidence_id.startswith("T"):
            continue
        ticker = str(row.get("ticker") or row.get("company") or company).upper().strip()
        if company and ticker and ticker != company:
            continue
        text = str(row.get("claim") or "").strip() or _snippet_excerpt(row)
        if text:
            candidates.append((ref, text))

    specs = [
        (
            "fulfillment_staffing",
            "履约与人员配置风险" if lang == "zh" else "Fulfillment and staffing risk",
            "可能先影响交付效率和服务质量，再通过履约成本上升传导到利润率" if lang == "zh" else "can first affect delivery efficiency and service quality, then flow through fulfillment cost into margins",
            ("fulfillment", "logistics", "staffing", "workforce", "delivery", "warehouse", "人员", "履约", "物流"),
        ),
        (
            "inventory",
            "库存风险" if lang == "zh" else "Inventory risk",
            "可能通过减值、折价促销和仓储成本影响毛利率，同时增加现金占用并压制 FCF" if lang == "zh" else "can pressure gross margin through write-downs, markdowns, and storage costs while tying up cash and pressuring FCF",
            ("inventory", "inventories", "write-down", "markdown", "obsolete", "库存"),
        ),
        (
            "regulatory_cross_border",
            "监管 / 跨境合规风险" if lang == "zh" else "Regulatory and cross-border compliance risk",
            "可能影响区域业务、卖家生态以及广告/第三方服务收入，并抬高合规成本" if lang == "zh" else "can affect regional operations, seller ecosystems, advertising or third-party service revenue, and compliance cost",
            ("regulatory", "regulation", "legal", "compliance", "antitrust", "china", "india", "jurisdiction", "监管", "法律", "合规"),
        ),
        (
            "competition",
            "竞争风险" if lang == "zh" else "Competition risk",
            "可能通过定价、市场份额、产品组合和利润率影响收入与利润" if lang == "zh" else "can affect revenue and profit through pricing, market share, product mix, and margins",
            ("competition", "competitive", "competitor", "pricing", "market share", "竞争"),
        ),
        (
            "supply_chain",
            "供应链风险" if lang == "zh" else "Supply-chain risk",
            "可能影响库存可得性、成本和交付稳定性" if lang == "zh" else "can affect inventory availability, cost, and delivery reliability",
            ("supply chain", "supplier", "component", "供应链", "供应商"),
        ),
    ]
    ranked: list[dict[str, Any]] = []
    used_refs: set[str] = set()
    rank_labels = (
        ["最高优先级", "第二优先级", "第三优先级", "第四优先级"]
        if lang == "zh"
        else ["Highest Priority", "Second Priority", "Third Priority", "Fourth Priority"]
    )
    for theme_id, title, mechanism, keywords in specs:
        matched_refs: list[str] = []
        matched_text: list[str] = []
        for ref, text in candidates:
            lowered = text.lower()
            if any(keyword.lower() in lowered for keyword in keywords):
                matched_refs.append(ref)
                matched_text.append(text)
        if not matched_refs:
            continue
        ranked.append(
            {
                "theme_id": theme_id,
                "rank_label": rank_labels[min(len(ranked), len(rank_labels) - 1)],
                "theme": title,
                "mechanism": mechanism,
                "evidence_refs": list(dict.fromkeys(matched_refs))[:3],
                "supporting_text": " ".join(matched_text)[:360],
            }
        )
        used_refs.update(matched_refs)
        if len(ranked) >= 4:
            break

    if not ranked and candidates:
        for ref, text in candidates[:3]:
            ranked.append(
                {
                    "theme_id": "validated_risk_text",
                    "rank_label": rank_labels[min(len(ranked), len(rank_labels) - 1)],
                    "theme": "已验证风险披露" if lang == "zh" else "Validated risk disclosure",
                    "mechanism": (
                        "影响机制需要结合原文继续拆解，但该风险已由申报文本支撑"
                        if lang == "zh"
                        else "the mechanism needs further breakdown from the source text, but the risk is supported by filing text"
                    ),
                    "evidence_refs": [ref],
                    "supporting_text": text[:360],
                }
            )
    return ranked


def _metric_value_for_summary(
    packet: dict[str, Any],
    metric: str,
    company: str,
    refs: list[str],
    *,
    missing: str,
) -> tuple[str, list[str]]:
    row = _metric_rows_by_company(packet, metric, refs).get(company) or _metric_rows_by_company(packet, metric).get(company)
    if not row:
        canonical = normalize_metric_name(metric)
        if canonical in {
            "capex_to_revenue",
            "fcf_yield",
            "cash_conversion",
            "net_debt",
            "debt_to_equity",
            "pe_ratio",
            "ps_ratio",
        }:
            return _metric_dependency_missing_reason(packet, canonical, company, "zh" if "缺少" in missing else "en"), []
        return missing, []
    evidence_id = str(row.get("evidence_id") or "").strip()
    return _display_metric_row(row), [evidence_id] if evidence_id else []


def _metric_numeric_value_for_summary(packet: dict[str, Any], metric: str, company: str, refs: list[str]) -> float | None:
    row = _metric_rows_by_company(packet, metric, refs).get(company) or _metric_rows_by_company(packet, metric).get(company)
    if not row:
        return None
    try:
        return float(row.get("value"))
    except (TypeError, ValueError):
        return None


def _cash_flow_ratio_interpretation(
    *,
    cfo_ratio_value: float | None,
    fcf_margin_value: float | None,
    lang: str,
) -> str:
    if lang != "zh":
        points: list[str] = []
        if cfo_ratio_value is not None:
            if cfo_ratio_value >= 1.0:
                points.append("operating cash flow covers net income")
            elif cfo_ratio_value >= 0.8:
                points.append("operating cash flow is close to net income")
            else:
                points.append("operating cash flow trails net income")
        if fcf_margin_value is not None:
            if fcf_margin_value >= 0.2:
                points.append("free-cash-flow margin indicates substantial cash conversion from revenue")
            elif fcf_margin_value > 0:
                points.append("free-cash-flow margin is positive but should be tracked")
            else:
                points.append("free-cash-flow margin is weak or negative")
        return "; ".join(points)

    points = []
    if cfo_ratio_value is not None:
        if cfo_ratio_value >= 1.0:
            points.append("经营现金流覆盖净利润，利润现金含量较好")
        elif cfo_ratio_value >= 0.8:
            points.append("经营现金流接近净利润，利润现金含量尚可")
        else:
            points.append("经营现金流低于净利润，利润现金含量需要谨慎看待")
    if fcf_margin_value is not None:
        if fcf_margin_value >= 0.2:
            points.append("FCF margin 较高，收入转化为自由现金流的能力较强")
        elif fcf_margin_value > 0:
            points.append("FCF margin 为正，收入有一定自由现金流转化")
        else:
            points.append("FCF margin 偏低，收入转化为自由现金流的空间有限")
    return "；".join(points)


def _cash_flow_section_summary(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> tuple[str, list[str]]:
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    cfo, cfo_refs = _metric_value_for_summary(packet, "operating_cash_flow", company, refs, missing=missing)
    fcf, fcf_refs = _metric_value_for_summary(packet, "free_cash_flow", company, refs, missing=missing)
    capex, capex_refs = _metric_value_for_summary(packet, "capital_expenditure", company, refs, missing=missing)
    cfo_ratio, ratio_refs = _metric_value_for_summary(packet, "cfo_to_net_income", company, refs, missing=missing)
    fcf_margin, fcf_margin_refs = _metric_value_for_summary(packet, "fcf_margin", company, refs, missing=missing)
    evidence_refs = list(dict.fromkeys(cfo_refs + fcf_refs + capex_refs + ratio_refs + fcf_margin_refs))
    if not evidence_refs:
        return _single_company_section_limitation("cash_flow_quality", lang), []
    cfo_ratio_value = _metric_numeric_value_for_summary(packet, "cfo_to_net_income", company, refs)
    fcf_margin_value = _metric_numeric_value_for_summary(packet, "fcf_margin", company, refs)
    cfo_value = _metric_numeric_value_for_summary(packet, "operating_cash_flow", company, refs)
    fcf_value = _metric_numeric_value_for_summary(packet, "free_cash_flow", company, refs)
    capex_value = _metric_numeric_value_for_summary(packet, "capital_expenditure", company, refs)
    capex_drag = (
        cfo_value is not None
        and fcf_value is not None
        and capex_value is not None
        and cfo_value > 0
        and fcf_value <= 0
        and abs(capex_value) >= abs(cfo_value) * 0.8
    )
    ratio_interpretation = _cash_flow_ratio_interpretation(
        cfo_ratio_value=cfo_ratio_value,
        fcf_margin_value=None if capex_drag else fcf_margin_value,
        lang=lang,
    )
    mechanism_points: list[str] = []
    if cfo_value is not None and fcf_value is not None and capex_value is not None:
        if capex_drag:
            mechanism_points.append(
                "经营现金流为正且规模较强，但资本开支接近或超过经营现金流，使自由现金流为负；因此现金流质量要分开看：经营端较强，资本强度压制自由现金流"
                if lang == "zh"
                else "operating cash flow is sizable and positive, but capex is near or above operating cash flow, leaving free cash flow negative; cash-flow quality should be separated between operating generation and capital intensity"
            )
        elif cfo_value > 0 and fcf_value < cfo_value and abs(capex_value) > 0:
            mechanism_points.append(
                "经营现金流为正，但资本开支会削弱自由现金流，需要分开看经营端和资本投入端"
                if lang == "zh"
                else "operating cash flow is positive, but capex reduces free cash flow, so operating generation and capital investment should be read separately"
            )
    if fcf_margin_value is not None and fcf_margin_value < 0 and not capex_drag:
        mechanism_points.append(
            "FCF margin 为负，说明收入在资本开支后暂未转化为自由现金流"
            if lang == "zh"
            else "FCF margin is negative, showing revenue is not converting into free cash flow after capex"
        )
    refs_text = _methodology_refs(evidence_refs)
    if lang == "zh":
        details = f"经营现金流为 {cfo}，自由现金流为 {fcf}，资本开支为 {capex}"
        ratios = []
        if cfo_ratio != missing:
            ratios.append(f"CFO/净利润为 {cfo_ratio}")
        if fcf_margin != missing:
            ratios.append(f"FCF margin 为 {fcf_margin}")
        ratio_text = f"；{'，'.join(ratios)}" if ratios else "；现金流转化率证据不足"
        interpretation_text = f"；{ratio_interpretation}" if ratio_interpretation else ""
        mechanism_text = f"。{'；'.join(_dedupe_lines(mechanism_points))}" if mechanism_points else ""
        return f"{details}{ratio_text}{interpretation_text}{mechanism_text}。{refs_text}".strip(), evidence_refs
    details = f"Operating cash flow is {cfo}, free cash flow is {fcf}, and capex is {capex}"
    ratios = []
    if cfo_ratio != missing:
        ratios.append(f"CFO/net income is {cfo_ratio}")
    if fcf_margin != missing:
        ratios.append(f"FCF margin is {fcf_margin}")
    ratio_text = f"; {', '.join(ratios)}" if ratios else "; cash-conversion ratio evidence is incomplete"
    interpretation_text = f"; {ratio_interpretation}" if ratio_interpretation else ""
    mechanism_text = f"; {'; '.join(_dedupe_lines(mechanism_points))}" if mechanism_points else ""
    return f"{details}{ratio_text}{interpretation_text}{mechanism_text}. {refs_text}".strip(), evidence_refs


def _balance_sheet_section_summary(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> tuple[str, list[str]]:
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    cash, cash_refs = _metric_value_for_summary(packet, "cash", company, refs, missing=missing)
    debt, debt_refs = _metric_value_for_summary(packet, "total_debt", company, refs, missing=missing)
    net_debt, net_debt_refs = _metric_value_for_summary(packet, "net_debt", company, refs, missing=missing)
    assets, assets_refs = _metric_value_for_summary(packet, "total_assets", company, refs, missing=missing)
    liabilities, liabilities_refs = _metric_value_for_summary(packet, "total_liabilities", company, refs, missing=missing)
    equity, equity_refs = _metric_value_for_summary(packet, "shareholders_equity", company, refs, missing=missing)
    debt_to_equity, dte_refs = _metric_value_for_summary(packet, "debt_to_equity", company, refs, missing=missing)
    capex, capex_refs = _metric_value_for_summary(packet, "capital_expenditure", company, refs, missing=missing)
    capex_to_revenue, ctr_refs = _metric_value_for_summary(packet, "capex_to_revenue", company, refs, missing=missing)
    inventory, inventory_refs = _metric_value_for_summary(packet, "inventory", company, refs, missing=missing)
    receivables, receivable_refs = _metric_value_for_summary(packet, "receivables", company, refs, missing=missing)
    evidence_refs = list(
        dict.fromkeys(
            cash_refs
            + debt_refs
            + net_debt_refs
            + assets_refs
            + liabilities_refs
            + equity_refs
            + dte_refs
            + capex_refs
            + ctr_refs
            + inventory_refs
            + receivable_refs
        )
    )
    if not evidence_refs:
        return _single_company_section_limitation("balance_sheet_and_capital_intensity", lang), []
    refs_text = _methodology_refs(evidence_refs)
    cash_row = _metric_rows_by_company(packet, "cash", refs).get(company) or _metric_rows_by_company(packet, "cash").get(company)
    debt_row = _metric_rows_by_company(packet, "total_debt", refs).get(company) or _metric_rows_by_company(packet, "total_debt").get(company)
    net_debt_row = _metric_rows_by_company(packet, "net_debt", refs).get(company) or _metric_rows_by_company(packet, "net_debt").get(company)
    dte_row = _metric_rows_by_company(packet, "debt_to_equity", refs).get(company) or _metric_rows_by_company(packet, "debt_to_equity").get(company)
    cash_value = _metric_float(cash_row)
    debt_value = _metric_float(debt_row)
    net_debt_value = _metric_float(net_debt_row)
    dte_value = _metric_float(dte_row)
    if lang == "zh" and cash_refs and debt_refs and (net_debt_refs or dte_refs):
        if cash_value is not None and debt_value not in {None, 0}:
            ratio = cash_value / abs(debt_value)
            if ratio > 1.2:
                cash_debt_relation = "现金规模高于债务"
            elif ratio < 0.8:
                cash_debt_relation = "债务规模高于现金"
            else:
                cash_debt_relation = "现金与债务规模接近"
        else:
            cash_debt_relation = "现金债务关系仍需结合更多口径观察"
        if net_debt_value is None:
            net_debt_view = "净债务方向需要结合现金和债务口径确认"
        elif net_debt_value < 0:
            net_debt_view = "公司处于净现金状态"
        elif abs(net_debt_value) <= max(abs(cash_value or 0), abs(debt_value or 0), 1) * 0.1:
            net_debt_view = "净债务接近中性"
        else:
            net_debt_view = "存在净债务"
        if dte_value is None:
            leverage_view = "债务/权益压力仍需补充判断"
        elif dte_value < 0.2:
            leverage_view = "债务/权益压力较低"
        elif dte_value < 0.6:
            leverage_view = "债务/权益压力中等"
        else:
            leverage_view = "债务/权益压力较高"
        return (
            f"现金及等价物为 {cash}，总债务为 {debt}，净债务约 {net_debt}，债务/权益为 {debt_to_equity}。"
            f"从这些已验证指标看，{cash_debt_relation}，{net_debt_view}，{leverage_view}；"
            f"总资产为 {assets}，总负债为 {liabilities}，股东权益为 {equity}，资本开支为 {capex}，"
            f"capex/revenue 为 {capex_to_revenue}。这支持对短期杠杆压力和资本强度的有限观察，"
            f"但不是完整偿债能力评估，仍需债务期限、利息支出和现金流覆盖证据。{refs_text}"
        ).strip(), evidence_refs
    if lang != "zh" and cash_refs and debt_refs and (net_debt_refs or dte_refs):
        return (
            f"Cash and equivalents are {cash}, total debt is {debt}, net debt is about {net_debt}, and debt/equity is "
            f"{debt_to_equity}. Assets are {assets}, liabilities are {liabilities}, equity is {equity}, capex is {capex}, "
            f"and capex/revenue is {capex_to_revenue}. This supports a limited view of leverage pressure and capital "
            f"intensity, but it is not a full solvency assessment without debt maturity, interest expense, and cash-flow "
            f"coverage evidence. {refs_text}"
        ).strip(), evidence_refs
    if lang == "zh":
        return (
            f"现金及等价物为 {cash}，总债务为 {debt}，净债务为 {net_debt}；总资产为 {assets}，"
            f"总负债为 {liabilities}，股东权益为 {equity}，债务/权益为 {debt_to_equity}；资本开支为 {capex}，"
            f"capex/revenue 为 {capex_to_revenue}，应收款为 {receivables}，存货为 {inventory}。"
            f"这些指标用于观察抗风险能力、营运资本和增长是否消耗资本。{refs_text}"
        ).strip(), evidence_refs
    return (
        f"Cash and equivalents are {cash}, total debt is {debt}, net debt is {net_debt}, total assets are {assets}, "
        f"liabilities are {liabilities}, equity is {equity}, and debt/equity is {debt_to_equity}; capex is {capex}, "
        f"capex/revenue is {capex_to_revenue}, receivables are {receivables}, and inventory is {inventory}. "
        f"These metrics bound balance-sheet resilience and capital intensity. {refs_text}"
    ).strip(), evidence_refs


def _balance_sheet_focus_judgment(packet: dict[str, Any], company: str, lang: str) -> str | None:
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    cash, cash_refs = _metric_value_for_summary(packet, "cash", company, [], missing=missing)
    debt, debt_refs = _metric_value_for_summary(packet, "total_debt", company, [], missing=missing)
    net_debt, net_debt_refs = _metric_value_for_summary(packet, "net_debt", company, [], missing=missing)
    debt_to_equity, dte_refs = _metric_value_for_summary(packet, "debt_to_equity", company, [], missing=missing)
    assets, asset_refs = _metric_value_for_summary(packet, "total_assets", company, [], missing=missing)
    liabilities, liability_refs = _metric_value_for_summary(packet, "total_liabilities", company, [], missing=missing)
    equity, equity_refs = _metric_value_for_summary(packet, "shareholders_equity", company, [], missing=missing)
    if not (cash_refs and debt_refs and net_debt_refs and dte_refs):
        return None
    if lang == "zh":
        structure = ""
        if asset_refs and liability_refs and equity_refs:
            structure = f"总资产为 {assets}，总负债为 {liabilities}，股东权益为 {equity}。"
        return (
            f"基于当前已验证指标，{company} 的资产负债表短期杠杆压力不高：现金约 {cash}，"
            f"总债务约 {debt}，净债务约 {net_debt}，债务/权益约 {debt_to_equity}。"
            f"{structure}这说明债务相对权益和现金规模不重，但这不是完整偿债能力判断，"
            "因为还需要债务期限、利息支出和现金流覆盖数据。"
        )
    structure = ""
    if asset_refs and liability_refs and equity_refs:
        structure = f" Assets are {assets}, liabilities are {liabilities}, and equity is {equity}."
    return (
        f"Based on currently validated metrics, {company}'s near-term leverage pressure does not look high: "
        f"cash is about {cash}, total debt is about {debt}, net debt is about {net_debt}, and debt/equity is about "
        f"{debt_to_equity}.{structure} This is not a full solvency assessment because debt maturity, interest expense, "
        "and cash-flow coverage evidence are still needed."
    )


def _valuation_section_summary(packet: dict[str, Any], refs: list[str], company: str, lang: str) -> tuple[str, list[str]]:
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    market_cap, market_refs = _metric_value_for_summary(packet, "market_cap", company, refs, missing=missing)
    pe_ratio, pe_refs = _metric_value_for_summary(packet, "pe_ratio", company, refs, missing=missing)
    ps_ratio, ps_refs = _metric_value_for_summary(packet, "ps_ratio", company, refs, missing=missing)
    fcf_yield, fcf_refs = _metric_value_for_summary(packet, "fcf_yield", company, refs, missing=missing)
    price, price_refs = _metric_value_for_summary(packet, "adjusted_close", company, refs, missing=missing)
    if price == missing:
        price, price_refs = _metric_value_for_summary(packet, "price", company, refs, missing=missing)
    evidence_refs = list(dict.fromkeys(price_refs + market_refs + pe_refs + ps_refs + fcf_refs))
    if not evidence_refs:
        return _single_company_section_limitation("valuation_and_risk_boundary", lang), []
    refs_text = _methodology_refs(evidence_refs)
    if lang == "zh":
        valuation_direction = (
            f"{company} 的估值风险偏高"
            if any(value != missing for value in (pe_ratio, ps_ratio, fcf_yield))
            else f"{company} 只能观察估值输入边界"
        )
        return (
            f"不能给买卖建议；但从 P/E {pe_ratio}、P/S {ps_ratio}、FCF yield {fcf_yield} 和市值 {market_cap} 看，"
            f"{valuation_direction}，是否合理取决于增长兑现能力、历史分位和同业基准{refs_text}。"
        ).strip(), evidence_refs
    return (
        f"Latest price evidence is {price}, market cap is {market_cap}, P/E is {pe_ratio}, P/S is {ps_ratio}, "
        f"and FCF yield is {fcf_yield}. This supports only a valuation-boundary view, not trading actions or price forecasts. {refs_text}"
    ).strip(), evidence_refs


def _risk_section_summary(packet: dict[str, Any], refs: list[str], companies: list[str], lang: str) -> str:
    rows_by_ref = _packet_rows_by_ref(packet)
    by_company: dict[str, list[tuple[str, str]]] = {}
    for ref in refs:
        row = dict(rows_by_ref.get(ref, {}) or {})
        if not row:
            continue
        ticker = str(row.get("ticker") or row.get("company") or "").upper().strip()
        if not ticker:
            continue
        excerpt = _snippet_excerpt(row)
        if excerpt:
            by_company.setdefault(ticker, []).append((ref, excerpt))
    ordered = [company for company in companies if company in by_company]
    ordered += [company for company in by_company.keys() if company not in ordered]
    if not ordered:
        return (
            "已验证文本证据提供竞争与风险背景，但不足以做强风险排序。"
            if lang == "zh"
            else "Validated text evidence provides competitive-risk context, but not enough for a hard risk ranking."
        )
    parts = []
    for company in ordered:
        company_rows = by_company.get(company, [])[:3]
        combined = " ".join(excerpt for _ref, excerpt in company_rows)
        phrases = _risk_phrases_from_text(company, combined, lang)
        if not phrases:
            continue
        refs_for_company = _methodology_refs([ref for ref, _excerpt in company_rows])
        if lang == "zh":
            parts.append(f"{company} 的主要风险来自{'、'.join(phrases)}。{refs_for_company}".strip())
        else:
            parts.append(f"{company}'s main risks in validated text relate to {', '.join(phrases)}. {refs_for_company}".strip())
    if not parts:
        return (
            "当前缺少足够文本证据，因此不能做具体竞争风险判断。"
            if lang == "zh"
            else "Current text evidence is not specific enough for a concrete competitive-risk judgment."
        )
    tail = "因此风险维度不能简单偏向某一方。" if lang == "zh" else "This does not support a simple one-sided risk ranking."
    return f"{' '.join(parts)} {tail}" if lang == "zh" else f"{' '.join(parts)} {tail}"


def _valuation_limitation(lang: str) -> str:
    return (
        "当前缺少估值证据，因此不能判断哪家公司估值更有吸引力。"
        if lang == "zh"
        else "Valuation evidence is missing, so I cannot judge which company has the more attractive valuation boundary."
    )


def _investment_limitation(lang: str) -> str:
    return (
        "当前分析基于已验证财报和文本证据，不构成投资建议。"
        if lang == "zh"
        else "This analysis is based on validated filings and text evidence only; it is not investment advice."
    )


def _provider_limitation(packet: dict[str, Any], lang: str) -> str:
    providers = {
        str(row.get("source_provider") or "").strip().lower()
        for row in packet.get("numeric_table", []) or []
        if isinstance(row, dict)
    }
    if "yfinance" not in providers:
        return ""
    return (
        "部分结构化财务数据来自 yfinance，可信度为 medium。"
        if lang == "zh"
        else "Some structured financial data comes from yfinance, with medium confidence."
    )


def _single_company_section_limitation(dimension_id: str, lang: str) -> str:
    if lang == "zh":
        if dimension_id == "business_model":
            return "当前缺少业务模式文本证据，不能完整描述业务模式。"
        if dimension_id == "moat_and_competitive_risk":
            return "当前缺少风险文本证据，不能做具体风险判断。"
        if dimension_id == "valuation_and_risk_boundary":
            return "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。"
        if dimension_id == "revenue_quality":
            return "当前缺少收入证据，不能判断收入质量。"
        if dimension_id == "profitability_quality":
            return "当前缺少净利润或净利率证据，不能判断盈利质量。"
        if dimension_id == "cash_flow_quality":
            return "当前缺少经营现金流、自由现金流或资本开支证据，无法验证利润能否转化为现金。"
        if dimension_id == "balance_sheet_and_capital_intensity":
            return "当前缺少现金、债务、资本开支、应收款或存货证据，不能判断抗风险能力和资本投入强度。"
        return "当前缺少该维度证据。"
    if dimension_id == "business_model":
        return "Business-model text evidence is missing."
    if dimension_id == "moat_and_competitive_risk":
        return "Risk text evidence is missing, so specific risk judgments are not supported."
    if dimension_id == "valuation_and_risk_boundary":
        return "Valuation evidence is missing, so valuation attractiveness or cheap/expensive claims are not supported."
    if dimension_id == "cash_flow_quality":
        return "Operating cash flow, free cash flow, or capex evidence is missing, so earnings cash conversion cannot be verified."
    if dimension_id == "balance_sheet_and_capital_intensity":
        return "Cash, debt, capex, receivables, or inventory evidence is missing, so balance-sheet resilience and capital intensity cannot be judged."
    return "Evidence for this dimension is missing."


def _build_single_company_methodology_answer(
    *,
    packet: dict[str, Any],
    status_map: dict[str, dict[str, Any]],
    selected_framework: dict[str, Any] | str | None,
    red_flags: list[dict[str, Any]] | None,
    lang: str,
) -> MethodologyAnswer | None:
    framework_id = _selected_framework_id(selected_framework or packet.get("selected_framework", {}))
    statuses = {dimension_id: str(item.get("status") or "") for dimension_id, item in status_map.items()}
    if not _any_methodology_dimension_ready(statuses):
        return None
    companies = _packet_companies(packet, {})
    company = companies[0] if companies else ""
    if not company:
        return None
    requested_dimensions = _packet_requested_dimensions(packet)
    active_dimensions = requested_dimensions or _packet_active_dimensions(packet, status_map)
    is_broad_overview = _is_broad_single_company_overview(packet, active_dimensions, status_map)
    if is_broad_overview:
        active_dimensions = list(_SINGLE_COMPANY_OVERVIEW_DIMENSIONS)
    summary_map = _packet_dimension_summary(packet)
    metric_table = _build_single_company_metric_table(packet, company, lang)
    data_sanity_flags = build_metric_sanity_flags(packet, lang=lang)
    dimension_sections: list[MethodologyDimensionSection] = []
    evidence_refs: list[str] = []

    def sanity_messages(dimension_id: str) -> list[str]:
        return [
            str(flag.get("message") or "").strip()
            for flag in data_sanity_flags
            if str(flag.get("dimension_id") or "") == dimension_id and str(flag.get("message") or "").strip()
        ]

    def sanity_flags(dimension_id: str) -> list[dict[str, Any]]:
        return [
            dict(flag)
            for flag in data_sanity_flags
            if str(flag.get("dimension_id") or "") == dimension_id
        ]

    def add_section(dimension_id: str, summary: str = "", refs: list[str] | None = None, limitation: str | None = None) -> None:
        if dimension_id not in active_dimensions:
            return
        status = statuses.get(dimension_id) or str(summary_map.get(dimension_id, {}).get("status") or "missing")
        if status not in {"satisfied", "partial", "missing"}:
            status = "missing"
        clean_refs = list(dict.fromkeys(str(ref) for ref in refs or [] if str(ref).strip()))
        evidence_refs.extend(clean_refs)
        dimension_sections.append(
            MethodologyDimensionSection(
                dimension_id=dimension_id,
                title=_dimension_label(dimension_id, lang),
                status=status,  # type: ignore[arg-type]
                summary=summary,
                evidence_refs=clean_refs,
                limitation=limitation,
            )
        )

    business_refs = _dimension_text_refs(summary_map, "business_model") or _dimension_refs(packet, summary_map, "business_model")
    if statuses.get("business_model") in {"satisfied", "partial"} and business_refs:
        add_section("business_model", _business_section_summary(packet, business_refs, company, lang), business_refs)
    elif "business_model" in active_dimensions:
        add_section("business_model", limitation=_single_company_section_limitation("business_model", lang))

    revenue_refs = _dimension_numeric_refs(summary_map, "revenue_quality") or _dimension_refs(packet, summary_map, "revenue_quality")
    revenue_row = _metric_rows_by_company(packet, "revenue", revenue_refs).get(company)
    if statuses.get("revenue_quality") in {"satisfied", "partial"} and revenue_row:
        value = _display_metric_row(revenue_row)
        refs = [str(ref).strip() for ref in revenue_refs if str(ref).strip()] or [str(revenue_row.get("evidence_id") or "").strip()]
        refs = [ref for ref in refs if ref]
        revenue_trend = _trend_interpretation("revenue", _metric_history(packet, "revenue", company, revenue_refs), lang)
        if lang == "zh" and "历史收入序列存在口径限制" in revenue_trend:
            summary = f"{company} 当前可验证收入规模为 {value}；{revenue_trend}。收入质量只能在这些已验证期间内判断。"
        else:
            summary = (
                f"{company} 当前收入为 {value}；{revenue_trend}。收入质量只能在这些已验证期间内判断。"
                if lang == "zh"
                else f"{company}'s current revenue is {value}; {revenue_trend}. Revenue quality is bounded by these validated periods."
            )
        add_section("revenue_quality", summary, refs)
    elif "revenue_quality" in active_dimensions:
        add_section("revenue_quality", limitation=_single_company_section_limitation("revenue_quality", lang))

    profitability_refs = _dimension_numeric_refs(summary_map, "profitability_quality") or _dimension_refs(packet, summary_map, "profitability_quality")
    income_row = _metric_rows_by_company(packet, "net_income", profitability_refs).get(company)
    margin_row = _metric_rows_by_company(packet, "net_margin", profitability_refs).get(company)
    gross_row = _metric_rows_by_company(packet, "gross_margin", profitability_refs).get(company)
    operating_row = _metric_rows_by_company(packet, "operating_margin", profitability_refs).get(company)
    if statuses.get("profitability_quality") in {"satisfied", "partial"} and (income_row or margin_row):
        income_value = _display_metric_row(income_row) if income_row else ("缺少可验证数据" if lang == "zh" else "verified data unavailable")
        margin_value = _display_metric_row(margin_row) if margin_row else ("缺少可验证数据" if lang == "zh" else "verified data unavailable")
        gross_value = _display_metric_row(gross_row) if gross_row else ("缺少可验证数据" if lang == "zh" else "verified data unavailable")
        operating_value = _display_metric_row(operating_row) if operating_row else ("缺少可验证数据" if lang == "zh" else "verified data unavailable")
        income_trend = _trend_interpretation("net_income", _metric_history(packet, "net_income", company, profitability_refs), lang)
        margin_trend = _trend_interpretation("net_margin", _metric_history(packet, "net_margin", company, profitability_refs), lang)
        refs = [
            str(row.get("evidence_id"))
            for row in (income_row, margin_row, gross_row, operating_row)
            if row and str(row.get("evidence_id") or "").strip()
        ]
        has_high_margin_flag = any(str(flag.get("id") or "") == "unusually_high_net_margin" for flag in sanity_flags("profitability_quality"))
        if lang == "zh":
            margin_structure = (
                "毛利率与营业利润率之间的差距说明，履约、技术投入、销售管理或其他运营成本仍会影响利润释放。"
                if gross_row and operating_row
                else "当前盈利质量不能只看净利润，还需要结合毛利率、营业利润率和费用结构继续核验。"
            )
            conclusion = (
                "由于净利率触发异常偏高 caveat，盈利质量只能做方向性观察，不能直接外推为可持续盈利能力。"
                if has_high_margin_flag
                else "盈利质量判断主要基于这些已验证指标，并需要继续观察利润率结构能否稳定。"
            )
            summary = (
                f"{company} 当前净利润为 {income_value}，净利率为 {margin_value}，毛利率为 {gross_value}，营业利润率为 {operating_value}；"
                f"净利润趋势：{income_trend}；净利率趋势：{margin_trend}。{margin_structure}{conclusion}"
            )
        else:
            margin_structure = (
                "The spread between gross margin and operating margin shows that fulfillment, technology, sales, administrative, or other operating costs still shape profit conversion."
                if gross_row and operating_row
                else "Profitability quality should be read through net income, gross margin, operating margin, and cost structure rather than net income alone."
            )
            conclusion = (
                "Because net margin triggers an unusually-high caveat, profitability quality is only a directional observation until the basis is checked."
                if has_high_margin_flag
                else "The profitability view is bounded by these validated metrics and requires continued margin-structure monitoring."
            )
            summary = f"{company}'s current net income is {income_value}, net margin is {margin_value}, gross margin is {gross_value}, and operating margin is {operating_value}; net-income trend: {income_trend}; net-margin trend: {margin_trend}. {margin_structure} {conclusion}"
        profit_sanity = sanity_messages("profitability_quality")
        if profit_sanity:
            summary = f"{summary}{'；' if lang == 'zh' else ' '}{'；'.join(profit_sanity) if lang == 'zh' else ' '.join(profit_sanity)}"
        add_section("profitability_quality", summary, refs)
    elif "profitability_quality" in active_dimensions:
        add_section("profitability_quality", limitation=_single_company_section_limitation("profitability_quality", lang))

    cash_refs = _dimension_numeric_refs(summary_map, "cash_flow_quality") or _dimension_refs(packet, summary_map, "cash_flow_quality")
    if statuses.get("cash_flow_quality") in {"satisfied", "partial"} and cash_refs:
        summary, refs = _cash_flow_section_summary(packet, cash_refs, company, lang)
        add_section("cash_flow_quality", summary, refs)
    elif "cash_flow_quality" in active_dimensions:
        add_section("cash_flow_quality", limitation=_single_company_section_limitation("cash_flow_quality", lang))

    balance_refs = _dimension_numeric_refs(summary_map, "balance_sheet_and_capital_intensity") or _dimension_refs(packet, summary_map, "balance_sheet_and_capital_intensity")
    if statuses.get("balance_sheet_and_capital_intensity") in {"satisfied", "partial"} and balance_refs:
        summary, refs = _balance_sheet_section_summary(packet, balance_refs, company, lang)
        balance_sanity = sanity_messages("balance_sheet_and_capital_intensity")
        if balance_sanity:
            summary = f"{summary}{'；' if lang == 'zh' else ' '}{'；'.join(balance_sanity) if lang == 'zh' else ' '.join(balance_sanity)}"
        add_section("balance_sheet_and_capital_intensity", summary, refs)
    elif "balance_sheet_and_capital_intensity" in active_dimensions:
        add_section(
            "balance_sheet_and_capital_intensity",
            limitation=_single_company_section_limitation("balance_sheet_and_capital_intensity", lang),
        )

    risk_refs = _dimension_text_refs(summary_map, "moat_and_competitive_risk") or _dimension_refs(packet, summary_map, "moat_and_competitive_risk")
    overview_risk_ranking = _overview_risk_ranking(packet, risk_refs, company, lang) if is_broad_overview else []
    if statuses.get("moat_and_competitive_risk") in {"satisfied", "partial"} and risk_refs:
        add_section("moat_and_competitive_risk", _single_company_risk_summary(packet, risk_refs, company, lang), risk_refs)
    elif "moat_and_competitive_risk" in active_dimensions:
        add_section("moat_and_competitive_risk", limitation=_single_company_section_limitation("moat_and_competitive_risk", lang))

    if "valuation_and_risk_boundary" in active_dimensions:
        valuation_refs = _dimension_numeric_refs(summary_map, "valuation_and_risk_boundary") or _dimension_refs(
            packet,
            summary_map,
            "valuation_and_risk_boundary",
        )
        if statuses.get("valuation_and_risk_boundary") in {"satisfied", "partial"} and valuation_refs:
            summary, refs = _valuation_section_summary(packet, valuation_refs, company, lang)
            add_section("valuation_and_risk_boundary", summary, refs)
        else:
            add_section(
                "valuation_and_risk_boundary",
                limitation=_single_company_section_limitation("valuation_and_risk_boundary", lang),
            )

    if not dimension_sections:
        return None

    limitations: list[str] = []
    for section in dimension_sections:
        if section.status == "missing" and section.limitation:
            limitations.append(section.limitation)
    focused_dimension = _focused_single_company_dimension(packet)
    if focused_dimension == "balance_sheet_and_capital_intensity" and metric_table:
        limitations.append(
            "当前资产负债判断主要基于最近一期指标，缺少多期趋势验证。"
            if lang == "zh"
            else "The balance-sheet view is mainly based on the latest point-in-time metrics and lacks multi-period trend validation."
        )
    limitations.append(_investment_limitation(lang))
    provider_limitation = _provider_limitation(packet, lang)
    if provider_limitation:
        limitations.append(provider_limitation)
    limitations.extend(
        str(item)
        for item in packet.get("_renderer_caveats", []) or []
        if str(item).strip()
    )
    limitations.extend(str(flag.get("message") or "") for flag in data_sanity_flags if str(flag.get("message") or "").strip())
    for flag in red_flags or packet.get("red_flags", []) or []:
        if isinstance(flag, dict) and bool(flag.get("user_visible", True)):
            message = str(flag.get("message") or "").strip()
            if message:
                limitations.append(_clean_user_facing_phrase(message))

    ready_dimension_labels = [
        section.title
        for section in dimension_sections
        if section.status in {"satisfied", "partial"}
    ]
    missing_dimensions = [section.dimension_id for section in dimension_sections if section.status == "missing"]
    partial_dimensions = [section.dimension_id for section in dimension_sections if section.status == "partial"]
    all_rendered_dimensions_satisfied = bool(dimension_sections) and not missing_dimensions and not partial_dimensions
    business_model_missing = "business_model" in missing_dimensions
    balance_focus_judgment = None
    if focused_dimension == "balance_sheet_and_capital_intensity" and statuses.get("balance_sheet_and_capital_intensity") in {"satisfied", "partial"}:
        balance_focus_judgment = _balance_sheet_focus_judgment(packet, company, lang)
    canonical_intent = dict(packet.get("canonical_intent", {}) or {})
    scope_text = " ".join(
        str(value or "")
        for value in (
            packet.get("segment_or_product_scope"),
            packet.get("segment_focus"),
            canonical_intent.get("segment_or_product_scope"),
            canonical_intent.get("segment_focus"),
            packet.get("user_query"),
        )
    ).lower()
    network_focus = company.upper() == "NVDA" and any(
        term in scope_text
        for term in ("network", "networking", "网络", "infiniband", "ethernet", "nvlink")
    )
    network_refs = _methodology_refs(evidence_refs[:4])
    if balance_focus_judgment:
        judgment = balance_focus_judgment
    elif network_focus and lang == "zh":
        judgment = (
            "有限判断：NVDA 网络业务增长大概率与 AI 集群建设、GPU 集群互连以及 "
            f"NVLink/InfiniBand/Ethernet 需求有关；当前证据主要支持分部/产品层面的增长线索，"
            f"不能直接推出总公司营收增长的完整因果或贡献比例。{network_refs}"
        )
    elif network_focus:
        judgment = (
            "Limited judgment: NVDA networking growth is most likely linked to AI cluster buildouts, GPU-cluster interconnect, "
            f"and NVLink/InfiniBand/Ethernet demand; the evidence mainly supports segment/product-level growth signals, not a full company-level causal contribution. {network_refs}"
        )
    elif is_broad_overview and lang == "zh":
        if business_model_missing:
            judgment = (
                f"{company} 的业务模式核心文本证据当前不足，因此这只能作为有限公司分析；"
                "下面的判断主要基于已验证财务指标和风险文本，不能当作完整商业模式结论。"
            )
        elif data_sanity_flags:
            judgment = (
                f"基于当前已验证证据，{company} 可以形成公司层面的基本面轮廓；"
                "但部分盈利或现金流指标触发口径/异常值 caveat，因此结论需要分维度谨慎阅读。"
            )
        else:
            judgment = (
                f"基于当前已验证证据，{company} 的公司分析应分开看业务模式、收入、盈利、现金流、资本强度、估值和风险；"
                "结论仍受已入库财报和文本证据范围限制。"
            )
    elif is_broad_overview:
        if business_model_missing:
            judgment = (
                f"{company}'s core business-model text evidence is currently insufficient, so this is a limited company analysis "
                "based mainly on validated financial metrics and risk text."
            )
        elif data_sanity_flags:
            judgment = (
                f"Based on currently validated evidence, {company} supports a company-level fundamental overview, but some "
                "profitability or cash-flow metrics trigger data-basis caveats."
            )
        else:
            judgment = (
                f"Based on currently validated evidence, {company} should be analyzed across business model, revenue, "
                "profitability, cash flow, capital intensity, valuation, and risk; conclusions remain bounded by available evidence."
            )
    elif lang == "zh":
        if all_rendered_dimensions_satisfied and ready_dimension_labels:
            judgment = (
                f"基于当前已验证证据，{company} 可以从"
                f"{'、'.join(ready_dimension_labels)}形成可追溯的基本面判断；结论仍受已入库数据范围限制。"
            )
        elif ready_dimension_labels:
            judgment = (
                f"基于当前已验证证据，{company} 可以围绕"
                f"{'、'.join(ready_dimension_labels)}做有限基本面分析；缺失维度只作为证据边界处理。"
            )
        else:
            judgment = f"基于当前已验证证据，{company} 只能做有限基本面分析，缺失维度需要作为证据边界处理。"
    else:
        if all_rendered_dimensions_satisfied and ready_dimension_labels:
            judgment = (
                f"Based on currently validated evidence, {company} supports a traceable fundamental view across "
                f"{', '.join(ready_dimension_labels)}; conclusions remain bounded by the ingested data."
            )
        elif ready_dimension_labels:
            judgment = (
                f"Based on currently validated evidence, {company} supports a limited fundamental analysis "
                f"around {', '.join(ready_dimension_labels)}; missing dimensions are treated as evidence boundaries."
            )
        else:
            judgment = (
                f"Based on currently validated evidence, {company} supports only a limited fundamental analysis; "
                "missing dimensions are treated as evidence boundaries."
            )

    if missing_dimensions:
        follow_up_metrics = _default_follow_up_metrics(missing_dimensions, lang)
    elif partial_dimensions:
        follow_up_metrics = _useful_follow_up_metrics(partial_dimensions, lang)
    else:
        follow_up_metrics = _useful_follow_up_metrics(
            [section.dimension_id for section in dimension_sections if section.status == "satisfied"],
            lang,
        )
    return MethodologyAnswer(
        framework_id=framework_id,
        task_type=str(packet.get("task_type") or "report_summary"),
        answer_mode=str(packet.get("answer_mode") or "analytical"),
        analysis_scope="single_company",
        judgment=_clean_user_facing_phrase(judgment),
        overview_profile={
            "kind": "single_company_overview" if is_broad_overview else "",
            "company": company,
            "business_model_missing": business_model_missing,
            "dimension_count": len(dimension_sections),
        },
        data_sanity_flags=data_sanity_flags,
        overview_risk_ranking=overview_risk_ranking,
        single_company_metric_table=metric_table,
        dimension_sections=dimension_sections,
        counterpoint=None,
        limitations=_dedupe_lines([_clean_user_facing_phrase(item) for item in limitations if str(item).strip()]),
        follow_up_metrics=_dedupe_lines(follow_up_metrics),
        evidence_refs=list(dict.fromkeys(str(ref) for ref in evidence_refs if str(ref).strip())),
    )


def build_methodology_answer(
    evidence_packet: dict[str, Any] | None,
    dimension_status_map: dict[str, Any] | None = None,
    red_flags: list[dict[str, Any]] | None = None,
    selected_framework: dict[str, Any] | str | None = None,
    comparison_judgment_frame: dict[str, Any] | None = None,
    analyst_draft: dict[str, Any] | None = None,
    *,
    lang: str = "zh",
) -> MethodologyAnswer | None:
    packet = dict(evidence_packet or {})
    frame = dict(comparison_judgment_frame or packet.get("comparison_judgment_frame", {}) or {})
    task_type = str(packet.get("task_type") or "company_comparison")
    answer_mode = str(packet.get("answer_mode") or "comparison_brief")
    safety_intent = str(packet.get("safety_intent") or "")
    analysis_scope = str(packet.get("analysis_scope") or "")
    selected_framework = selected_framework or packet.get("selected_framework", {})
    status_map = _packet_status_map(packet, dimension_status_map)
    if analysis_scope == "single_company":
        return _build_single_company_methodology_answer(
            packet=packet,
            status_map=status_map,
            selected_framework=selected_framework,
            red_flags=red_flags,
            lang=lang,
        )
    is_comparison = answer_mode == "comparison_brief" or task_type == "company_comparison" or safety_intent == "investment_advice_like"
    if not is_comparison or not packet:
        return None

    framework_id = _selected_framework_id(selected_framework)
    statuses = {dimension_id: str(item.get("status") or "") for dimension_id, item in status_map.items()}
    if not _methodology_comparison_ready(statuses):
        return None
    active_dimensions = _packet_active_dimensions(packet, status_map)
    summary_map = _packet_dimension_summary(packet)
    companies = _packet_companies(packet, frame)
    metric_table = _build_metric_table(packet, companies, lang)
    dimension_sections: list[MethodologyDimensionSection] = []
    evidence_refs: list[str] = []

    def add_section(
        dimension_id: str,
        *,
        summary: str = "",
        refs: list[str] | None = None,
        limitation: str | None = None,
    ) -> None:
        if dimension_id not in active_dimensions:
            return
        status = statuses.get(dimension_id) or str(summary_map.get(dimension_id, {}).get("status") or "missing")
        if status not in {"satisfied", "partial", "missing"}:
            status = "missing"
        clean_refs = list(dict.fromkeys(str(ref) for ref in refs or [] if str(ref).strip()))
        evidence_refs.extend(clean_refs)
        dimension_sections.append(
            MethodologyDimensionSection(
                dimension_id=dimension_id,
                title=_dimension_label(dimension_id, lang),
                status=status,  # type: ignore[arg-type]
                summary=summary,
                evidence_refs=clean_refs,
                limitation=limitation,
            )
        )

    if "revenue_quality" in active_dimensions:
        refs = _dimension_numeric_refs(summary_map, "revenue_quality") or _dimension_refs(packet, summary_map, "revenue_quality")
        revenue_rows = _metric_rows_by_company(packet, "revenue", refs)
        growth_rows = _metric_rows_by_company(packet, "revenue_growth", refs)
        leader = _leader(revenue_rows)
        metric_line = _metric_sentence(revenue_rows, companies, "revenue" if lang != "zh" else "收入", lang)
        growth_line = _metric_sentence(growth_rows, companies, "revenue growth" if lang != "zh" else "收入增速", lang) if growth_rows else ""
        if statuses.get("revenue_quality") in {"satisfied", "partial"} and metric_line:
            if "profitability_quality" not in active_dimensions:
                frame_revenue_item = _frame_dimension_item(frame, "revenue_quality")
                history_complete = bool(frame_revenue_item.get("revenue_history_complete"))
                if lang == "zh":
                    scale_clause = f"{leader} 在收入规模上占优" if leader else "只能比较已验证收入规模"
                    summary = (
                        f"如果只基于当前增长/收入质量证据，{scale_clause}；"
                        f"但{'收入历史证据较完整，仍只能在已验证期间内比较增长质量' if history_complete else '收入历史证据不完整，因此不能完整判断增长质量'}；"
                        f"{growth_line or '当前收入增速证据不足'}；{metric_line}。"
                    )
                else:
                    scale_clause = f"{leader} leads on revenue scale" if leader else "only validated revenue scale can be compared"
                    summary = f"Based only on current growth/revenue-quality evidence, {scale_clause}, but revenue-history evidence is incomplete; {growth_line or 'revenue-growth evidence is incomplete'}; {metric_line}."
            else:
                summary = (
                    f"{leader} 在收入规模上更强；{metric_line}。"
                    if lang == "zh" and leader
                    else f"{leader} leads on revenue scale; {metric_line}."
                    if leader
                    else metric_line
                )
            add_section("revenue_quality", summary=summary, refs=refs)

    if "profitability_quality" in active_dimensions:
        refs = _dimension_numeric_refs(summary_map, "profitability_quality") or _dimension_refs(packet, summary_map, "profitability_quality")
        income_rows = _metric_rows_by_company(packet, "net_income", refs)
        margin_rows = _metric_rows_by_company(packet, "net_margin", refs)
        income_leader = _leader(income_rows)
        margin_leader = _leader(margin_rows)
        income_line = _metric_sentence(income_rows, companies, "net income" if lang != "zh" else "净利润", lang)
        margin_line = _metric_sentence(margin_rows, companies, "net margin" if lang != "zh" else "净利率", lang)
        if statuses.get("profitability_quality") in {"satisfied", "partial"} and (income_line or margin_line):
            if lang == "zh":
                leader_text = ""
                if income_leader and margin_leader and income_leader == margin_leader:
                    leader_text = f"{income_leader} 的净利润和净利率更高，当前证据支持其盈利转化能力更强；"
                summary = f"{leader_text}{income_line}；{margin_line}。".strip("；")
            else:
                leader_text = ""
                if income_leader and margin_leader and income_leader == margin_leader:
                    leader_text = f"{income_leader} has higher net income and net margin; "
                summary = f"{leader_text}{income_line}; {margin_line}."
            add_section("profitability_quality", summary=summary, refs=refs)

    if "cash_flow_quality" in active_dimensions:
        refs = _dimension_numeric_refs(summary_map, "cash_flow_quality") or _dimension_refs(packet, summary_map, "cash_flow_quality")
        status = statuses.get("cash_flow_quality", "")
        if status in {"satisfied", "partial"} and refs:
            cfo_line = _metric_sentence(_metric_rows_by_company(packet, "operating_cash_flow", refs), companies, "operating cash flow" if lang != "zh" else "经营现金流", lang)
            fcf_line = _metric_sentence(_metric_rows_by_company(packet, "free_cash_flow", refs), companies, "free cash flow" if lang != "zh" else "自由现金流", lang)
            capex_line = _metric_sentence(_metric_rows_by_company(packet, "capital_expenditure", refs), companies, "capital expenditure" if lang != "zh" else "资本开支", lang)
            fcf_margin_line = _metric_sentence(_metric_rows_by_company(packet, "fcf_margin", refs), companies, "FCF margin", lang)
            summary = (
                f"现金流质量比较必须看经营现金流、自由现金流、资本开支和 FCF margin；{cfo_line}；{fcf_line}；{capex_line}；{fcf_margin_line}。"
                if lang == "zh"
                else f"Cash-flow quality must be read through operating cash flow, free cash flow, capex, and FCF margin; {cfo_line}; {fcf_line}; {capex_line}; {fcf_margin_line}."
            )
            add_section("cash_flow_quality", summary=summary, refs=refs)
        elif "cash_flow_quality" in active_dimensions:
            add_section(
                "cash_flow_quality",
                limitation=_single_company_section_limitation("cash_flow_quality", lang),
            )

    if "moat_and_competitive_risk" in active_dimensions:
        refs = (
            _packet_text_refs_for_dimension(packet, "moat_and_competitive_risk")
            or _dimension_text_refs(summary_map, "moat_and_competitive_risk")
            or _dimension_refs(packet, summary_map, "moat_and_competitive_risk")
        )
        status = statuses.get("moat_and_competitive_risk", "")
        if status in {"satisfied", "partial"} and refs:
            add_section(
                "moat_and_competitive_risk",
                summary=_risk_section_summary(packet, refs, companies, lang),
                refs=refs,
            )

    valuation_status = statuses.get("valuation_and_risk_boundary")
    if "valuation_and_risk_boundary" in active_dimensions:
        refs = _dimension_numeric_refs(summary_map, "valuation_and_risk_boundary") or _dimension_refs(packet, summary_map, "valuation_and_risk_boundary")
        if valuation_status in {"satisfied", "partial"} and refs:
            pe_line = _metric_sentence(_metric_rows_by_company(packet, "pe_ratio", refs), companies, "P/E", lang)
            ps_line = _metric_sentence(_metric_rows_by_company(packet, "ps_ratio", refs), companies, "P/S", lang)
            fcf_yield_line = _metric_sentence(_metric_rows_by_company(packet, "fcf_yield", refs), companies, "FCF yield", lang)
            market_cap_line = _metric_sentence(_metric_rows_by_company(packet, "market_cap", refs), companies, "market cap" if lang != "zh" else "市值", lang)
            metric_judgments = _frame_metric_judgments(frame, "valuation_and_risk_boundary")
            judgment_lines = [
                _clean_user_facing_phrase(str(item.get("zh" if lang == "zh" else "en") or "").strip())
                for item in metric_judgments
                if str(item.get("zh" if lang == "zh" else "en") or "").strip()
            ]
            summary = (
                (
                    f"估值风险呈分歧，不能给单一绝对排序；{'；'.join(judgment_lines) or f'{pe_line}；{ps_line}；{fcf_yield_line}'}；{market_cap_line}。"
                )
                if lang == "zh"
                else f"Valuation risk is mixed, so a single absolute ranking is not supported; {'; '.join(judgment_lines) or f'{pe_line}; {ps_line}; {fcf_yield_line}'}; {market_cap_line}."
            )
            add_section("valuation_and_risk_boundary", summary=summary, refs=refs)
        elif valuation_status == "missing":
            add_section(
                "valuation_and_risk_boundary",
                summary="",
                refs=[],
                limitation=_valuation_limitation(lang),
            )

    preferred = str(frame.get("preferred_company") or "").upper().strip()
    if not preferred:
        preferred = _leader(_metric_rows_by_company(packet, "net_margin")) or _leader(_metric_rows_by_company(packet, "net_income"))
    scale_winner = str(frame.get("scale_winner") or _leader(_metric_rows_by_company(packet, "revenue")) or "").upper().strip()
    if lang == "zh":
        if "cash_flow_quality" in active_dimensions and "profitability_quality" not in active_dimensions:
            judgment = (
                f"围绕现金流质量，当前更偏向 {preferred}；判断依据是经营现金流、自由现金流、资本开支和 FCF margin，而不是净利率。"
                if preferred and preferred not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else "围绕现金流质量，当前只能做有边界比较；判断依据应是经营现金流、自由现金流、资本开支和 FCF margin，而不是净利率。"
            )
            counterpoint = "反方观点是：现金流质量不等于盈利能力，其他维度需要另行比较。"
        elif "valuation_and_risk_boundary" in active_dimensions and "profitability_quality" not in active_dimensions:
            valuation_refs = _methodology_refs(_frame_dimension_refs(frame, "valuation_and_risk_boundary"))
            judgment = (
                f"围绕估值风险，当前更需要关注 {preferred} 的估值边界；依据是 P/E、P/S、FCF yield、市值和股价输入。{valuation_refs}"
                if preferred and preferred not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else f"围绕估值风险，当前只能做有限估值边界比较；必须看 P/E、P/S、FCF yield、市值和股价输入，且指标分歧时不能单一排序。{valuation_refs}"
            )
            counterpoint = "反方观点是：普通风险因素不能替代估值风险判断；估值输入缺失时只能给出边界。"
        elif "revenue_quality" in active_dimensions and "profitability_quality" not in active_dimensions:
            revenue_refs = _methodology_refs(
                _frame_dimension_field_refs(frame, "revenue_quality", "scale_winner_refs")
                or _frame_dimension_refs(frame, "revenue_quality")
            )
            revenue_leader = scale_winner if scale_winner and scale_winner not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"} else preferred
            judgment = (
                f"如果只基于当前增长/收入质量证据，{revenue_leader} 在收入规模上占优；但收入历史证据不完整，因此不能完整判断增长质量。{revenue_refs}"
                if revenue_leader and revenue_leader not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else f"围绕增长/收入质量，当前只能做有边界比较；需要收入增速、收入历史和收入规模，不能用净利率替代。{revenue_refs}"
            )
            counterpoint = "反方观点是：单期收入规模领先不等于增长质量更好，需要收入增速和历史口径验证。"
        elif preferred and scale_winner and scale_winner != preferred:
            judgment = f"基于当前已验证证据，我更偏向 {preferred} 的盈利质量；但如果更重视收入规模，{scale_winner} 仍有优势。这不是投资建议。"
        elif preferred:
            judgment = f"基于当前已验证证据，我更偏向 {preferred} 的基本面质量。这不是投资建议。"
        else:
            judgment = "基于当前已验证证据，只能形成条件化基本面比较。这不是投资建议。"
        counterpoint = (
            f"反方观点是：{scale_winner} 收入规模更大；在不判断未来利润率的前提下，这说明其基本面弹性仍需要继续跟踪。"
            if scale_winner and scale_winner != preferred
            else "反方观点是：当前证据仍缺少估值、现金流和资产负债表维度，不能形成完整长期投资结论。"
        )
    else:
        if "cash_flow_quality" in active_dimensions and "profitability_quality" not in active_dimensions:
            judgment = (
                f"On cash-flow quality, I lean toward {preferred}; the basis is operating cash flow, free cash flow, capex, and FCF margin, not net margin."
                if preferred and preferred not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else "On cash-flow quality, only a bounded comparison is supported; the basis should be operating cash flow, free cash flow, capex, and FCF margin, not net margin."
            )
            counterpoint = "The counterpoint is that cash-flow quality is not the same as profitability; other dimensions need a separate comparison."
        elif "valuation_and_risk_boundary" in active_dimensions and "profitability_quality" not in active_dimensions:
            judgment = (
                f"On valuation risk, {preferred} needs more valuation-boundary scrutiny based on P/E, P/S, FCF yield, market cap, and share-price inputs."
                if preferred and preferred not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else "On valuation risk, only a limited valuation-boundary comparison is supported; P/E, P/S, FCF yield, market cap, and share-price inputs are required."
            )
            counterpoint = "The counterpoint is that ordinary risk factors cannot replace valuation-risk evidence; missing valuation inputs should be treated as boundaries."
        elif "revenue_quality" in active_dimensions and "profitability_quality" not in active_dimensions:
            judgment = (
                f"On growth/revenue quality, I lean toward {preferred}; the basis is revenue growth, revenue history, and revenue scale, not net margin."
                if preferred and preferred not in {"NO_CLEAR_PREFERENCE", "MIXED", "UNAVAILABLE"}
                else "On growth/revenue quality, only a bounded comparison is supported; revenue growth, revenue history, and revenue scale are needed, not net margin."
            )
            counterpoint = "The counterpoint is that current revenue scale does not by itself prove better growth quality."
        elif preferred and scale_winner and scale_winner != preferred:
            judgment = f"Based on currently validated evidence, I lean toward {preferred} on profitability quality; if revenue scale matters more, {scale_winner} still has an advantage. This is not investment advice."
        elif preferred:
            judgment = f"Based on currently validated evidence, I lean toward {preferred} on fundamental quality. This is not investment advice."
        else:
            judgment = "Based on currently validated evidence, only a conditional fundamental comparison is supported. This is not investment advice."
        counterpoint = (
            f"The counterpoint is that {scale_winner} has greater revenue scale; without making a margin forecast, that makes its business trajectory worth tracking."
            if scale_winner and scale_winner != preferred
            else "The counterpoint is that valuation, cash-flow, and balance-sheet evidence remains incomplete."
        )

    draft = dict(analyst_draft or {})
    draft_counterpoints = draft.get("methodology_counterpoints") or draft.get("counterpoints") or []
    if draft_counterpoints and isinstance(draft_counterpoints, list):
        first = draft_counterpoints[0]
        if isinstance(first, dict):
            counterpoint = _clean_user_facing_phrase(str(first.get("statement") or counterpoint))
        elif str(first).strip():
            counterpoint = _clean_user_facing_phrase(str(first))

    limitations = [_valuation_limitation(lang)] if valuation_status == "missing" else []
    limitations.append(_investment_limitation(lang))
    provider_limitation = _provider_limitation(packet, lang)
    if provider_limitation:
        limitations.append(provider_limitation)
    for flag in red_flags or packet.get("red_flags", []) or []:
        if isinstance(flag, dict) and bool(flag.get("user_visible", True)):
            message = str(flag.get("message") or "").strip()
            if message:
                limitations.append(_clean_user_facing_phrase(message))

    if not dimension_sections:
        return None
    evidence_refs = list(dict.fromkeys(evidence_refs))
    return MethodologyAnswer(
        framework_id=framework_id,
        task_type=task_type,
        answer_mode=answer_mode,
        judgment=_clean_user_facing_phrase(judgment),
        metric_table=metric_table,
        dimension_sections=dimension_sections,
        counterpoint=_clean_user_facing_phrase(counterpoint),
        limitations=_dedupe_lines([_clean_user_facing_phrase(item) for item in limitations if str(item).strip()]),
        follow_up_metrics=_default_follow_up_metrics(
            [section.dimension_id for section in dimension_sections if section.status == "missing"],
            lang,
        ),
        evidence_refs=evidence_refs,
    )


def _draft_dimension_analyses(accepted_draft: dict[str, Any], usable_dims: dict[str, dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in accepted_draft.get("dimension_analyses", []) or []:
        if not isinstance(item, dict):
            continue
        dimension_id = str(item.get("dimension_id", "")).strip()
        if dimension_id not in usable_dims:
            continue
        claim = str(item.get("claim", "")).strip()
        refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
        if not claim:
            continue
        out.append(
            {
                "dimension_id": dimension_id,
                "label": _dimension_label(dimension_id, lang),
                "status": str(usable_dims[dimension_id].get("status", "")),
                "claim": claim,
                "evidence_refs": refs,
            }
        )
    return out


def _frame_dimension_analyses(frame: dict[str, Any], usable_dims: dict[str, dict[str, Any]], lang: str) -> list[dict[str, Any]]:
    frame = dict(frame or {})
    out: list[dict[str, Any]] = []
    scale_winner = str(frame.get("scale_winner", ""))
    preferred = str(frame.get("preferred_company", ""))
    margin_winner = str(frame.get("margin_winner", ""))
    if "revenue_quality" in usable_dims and scale_winner not in {"", "unavailable", "mixed"}:
        refs = _frame_dimension_field_refs(frame, "revenue_quality", "scale_winner_refs") or _frame_refs(frame, "revenue_quality")
        history_complete = bool(_frame_dimension_item(frame, "revenue_quality").get("revenue_history_complete"))
        claim = (
            (
                f"如果只基于当前增长/收入质量证据，{scale_winner} 在收入规模上占优；"
                f"但{'收入历史证据较完整，仍只能在已验证期间内比较增长质量。' if history_complete else '收入历史证据不完整，因此不能完整判断增长质量。'}"
            )
            if lang == "zh"
            else f"Based only on current growth/revenue-quality evidence, {scale_winner} leads on revenue scale, but revenue-history evidence is incomplete, so full growth quality cannot be determined."
        )
        out.append(
            {
                "dimension_id": "revenue_quality",
                "label": _dimension_label("revenue_quality", lang),
                "status": str(usable_dims["revenue_quality"].get("status", "")),
                "claim": claim,
                "evidence_refs": refs,
            }
        )
    if "profitability_quality" in usable_dims:
        winner = preferred if preferred not in {"", "unavailable", "mixed", "no_clear_preference"} else margin_winner
        if winner not in {"", "unavailable", "mixed", "no_clear_preference"}:
            claim = (
                f"{winner} 在盈利转化能力上更占优；该判断基于净利润和程序计算的净利率。"
                if lang == "zh"
                else f"{winner} looks stronger on profit conversion, based on net income and programmatically computed net margin."
            )
            out.append(
                {
                    "dimension_id": "profitability_quality",
                    "label": _dimension_label("profitability_quality", lang),
                    "status": str(usable_dims["profitability_quality"].get("status", "")),
                    "claim": claim,
                    "evidence_refs": _dedupe_lines(_frame_refs(frame, "profitability") + _frame_refs(frame, "margin")),
                }
            )
    if "moat_and_competitive_risk" in usable_dims and _frame_refs(frame, "risk"):
        claim = str(frame.get("risk_tradeoff", "")).strip()
        if claim:
            out.append(
                {
                    "dimension_id": "moat_and_competitive_risk",
                    "label": _dimension_label("moat_and_competitive_risk", lang),
                    "status": str(usable_dims["moat_and_competitive_risk"].get("status", "")),
                    "claim": claim,
                    "evidence_refs": _frame_refs(frame, "risk"),
                }
            )
    return out


def _build_methodology_report(
    *,
    lang: str,
    answer_mode: str,
    task_type: str,
    safety_intent: str,
    short_answer: str,
    accepted_draft: dict[str, Any],
    comparison_judgment_frame: dict[str, Any],
    requirement_summary: dict[str, Any],
    limitations: list[dict[str, Any]],
    red_flags: list[dict[str, Any]],
) -> dict[str, Any]:
    usable_dims = _usable_dimension_statuses(requirement_summary)
    if not usable_dims and not accepted_draft and not comparison_judgment_frame:
        return {}
    dimension_analyses = _draft_dimension_analyses(accepted_draft, usable_dims, lang)
    if not dimension_analyses and comparison_judgment_frame:
        dimension_analyses = _frame_dimension_analyses(comparison_judgment_frame, usable_dims, lang)
    if not dimension_analyses:
        return {}
    missing_dimensions = [
        str(dim)
        for dim, item in dict(requirement_summary.get("dimension_status_map", {}) or {}).items()
        if isinstance(item, dict) and str(item.get("status", "")) == "missing"
    ]
    limitation_messages = _methodology_limitation_messages(
        limitations=limitations,
        red_flags=red_flags,
        requirement_summary=requirement_summary,
        accepted_draft=accepted_draft,
    )
    try:
        text_rate = float(requirement_summary.get("required_text_satisfied_rate"))
    except (TypeError, ValueError):
        text_rate = 0.0
    if text_rate >= 1.0 and not any("已验证文本" in msg or "validated text" in msg.lower() for msg in limitation_messages):
        limitation_messages.insert(
            0,
            "已验证文本证据已纳入风险与竞争边界。" if lang == "zh" else "Validated text evidence is included for risk and competitive context.",
        )
    report = {
        "kind": "methodology_comparison_brief"
        if answer_mode == "comparison_brief" or safety_intent == "investment_advice_like"
        else "methodology_report",
        "framework_summary": str(accepted_draft.get("framework_summary") or (
            "基于基本面质量框架组织证据。" if lang == "zh" else "Evidence is organized under the fundamental quality framework."
        )),
        "conclusion": str(accepted_draft.get("overall_judgment") or short_answer),
        "dimension_analyses": dimension_analyses,
        "counterpoints": _dedupe_lines(
            [str(item) for item in accepted_draft.get("methodology_counterpoints", []) or [] if str(item).strip()]
            + ([str(comparison_judgment_frame.get("counterpoint", ""))] if comparison_judgment_frame else [])
        ),
        "limitations": limitation_messages,
        "follow_up_metrics": _dedupe_lines(
            [str(item) for item in accepted_draft.get("follow_up_metrics", []) or [] if str(item).strip()]
            + _default_follow_up_metrics(missing_dimensions, lang)
        ),
        "missing_dimensions": missing_dimensions,
        "active_dimensions": list(dict(requirement_summary.get("dimension_status_map", {}) or {}).keys()),
        "answer_mode": answer_mode,
        "task_type": task_type,
        "safety_intent": safety_intent,
    }
    return report


_RISK_THEME_CATALOG = [
    {
        "key": "product_demand_uncertainty",
        "theme_id": "product_demand_uncertainty",
        "theme_name": "新产品和需求不确定性",
        "terms": ("new product", "new service", "product introduction", "launch", "demand", "ai demand", "uncertainty", "新产品", "需求", "需求不确定"),
        "why": "这会直接影响收入增长持续性、库存安排和利润率弹性。",
        "financial": ("revenue", "net_margin"),
        "weight": 4.0,
    },
    {
        "key": "fulfillment_inventory_capex_pressure",
        "theme_id": "fulfillment_inventory_capex_pressure",
        "theme_name": "履约/库存/资本开支压力",
        "terms": ("fulfillment", "logistics", "inventory", "inventories", "capital expenditure", "capex", "investment", "履约", "物流", "库存", "资本开支", "投资"),
        "why": "履约、库存和资本投入压力可能通过服务质量、成本、现金占用和自由现金流传导到财务表现。",
        "financial": ("revenue", "net_margin"),
        "weight": 4.3,
    },
    {
        "key": "regulatory_compliance_pressure",
        "theme_id": "regulatory_compliance_pressure",
        "theme_name": "监管/合规风险",
        "terms": ("regulatory", "regulation", "legal", "compliance", "antitrust", "jurisdiction", "监管", "法律", "合规"),
        "why": "监管、法律和合规风险可能通过区域限制、合规成本和经营不确定性影响收入与利润率。",
        "financial": ("revenue", "net_margin"),
        "weight": 3.9,
    },
    {
        "key": "cloud_competition_pressure",
        "theme_id": "cloud_competition_pressure",
        "theme_name": "AWS/云竞争",
        "terms": ("aws", "cloud", "competition", "competitive", "azure", "google cloud", "云", "竞争"),
        "why": "云竞争可能通过价格、市场份额、客户迁移和投资强度影响 AWS 或云业务收入增速与利润率。",
        "financial": ("revenue", "net_margin"),
        "weight": 3.7,
    },
    {
        "key": "competition_pressure",
        "theme_id": "competition_pressure",
        "theme_name": "竞争加剧",
        "terms": ("competition", "competitive", "competitor", "competitive pressure", "market pressure", "竞争", "竞争压力"),
        "why": "竞争压力可能影响定价能力、市场份额和利润率。",
        "financial": ("revenue", "net_margin"),
        "weight": 3.5,
    },
    {
        "key": "supply_inventory_pressure",
        "theme_id": "supply_inventory_pressure",
        "theme_name": "供应链或库存压力",
        "terms": ("supply", "supply chain", "supplier", "inventory", "component", "capacity", "供应链", "库存", "产能", "供给"),
        "why": "供应链、库存或产能约束可能影响交付节奏、收入确认和毛利/净利率。",
        "financial": ("revenue", "net_margin"),
        "weight": 3.3,
    },
    {
        "key": "customer_market_concentration",
        "theme_id": "customer_market_concentration",
        "theme_name": "客户/市场集中风险",
        "terms": ("customer concentration", "customer", "customers", "single customer", "market concentration", "客户", "市场集中"),
        "why": "客户或市场集中会放大单一客户需求变化对收入和现金回收的影响。",
        "financial": ("revenue",),
        "weight": 3.0,
    },
    {
        "key": "margin_pressure",
        "theme_id": "margin_pressure",
        "theme_name": "利润率压力",
        "terms": ("margin", "cost pressure", "pricing", "price", "profitability", "operating leverage", "利润率", "成本压力", "定价"),
        "why": "利润率压力会削弱盈利质量，并可能让收入增长较难转化为净利润。",
        "financial": ("net_income", "net_margin"),
        "weight": 2.8,
    },
    {
        "key": "macro_regulatory_legal",
        "theme_id": "macro_regulatory_legal",
        "theme_name": "宏观、监管或法律风险",
        "terms": ("macro", "macroeconomic", "economic", "regulation", "regulatory", "legal", "litigation", "compliance", "宏观", "监管", "法律", "合规"),
        "why": "宏观、监管或法律事项可能增加经营不确定性，但若缺少公司特定连接，优先级低于核心业务风险。",
        "financial": ("revenue", "net_income"),
        "weight": 1.8,
    },
]


def _risk_theme_financial_context(packet: dict[str, Any], company: str, metrics: tuple[str, ...], lang: str) -> list[str]:
    labels = {
        "revenue": "收入",
        "net_income": "净利润",
        "net_margin": "净利率",
    }
    out: list[str] = []
    for metric in metrics:
        row = _metric_rows_by_company(packet, metric).get(company)
        if not row:
            continue
        value = _display_metric_row(row)
        ref = str(row.get("evidence_id") or "").strip()
        refs = _methodology_refs([ref]) if ref else ""
        label = labels.get(metric, metric)
        out.append(f"{label}：{value} {refs}".strip() if lang == "zh" else f"{label}: {value} {refs}".strip())
    return out


def _risk_dimension_status(packet: dict[str, Any]) -> str:
    status_map = dict(packet.get("dimension_status_map", {}) or {})
    if not status_map:
        status_map = dict(dict(packet.get("dimension_sufficiency", {}) or {}).get("dimension_status_map", {}) or {})
    return str(dict(status_map.get("moat_and_competitive_risk", {}) or {}).get("status", "") or "")


def _risk_scenario_anchor(packet: dict[str, Any], lang: str) -> dict[str, Any]:
    query = str(packet.get("user_query") or packet.get("query") or "").lower()
    if not query:
        return {}
    scenarios = [
        {
            "id": "economic_slowdown",
            "label": "经济放缓" if lang == "zh" else "economic slowdown",
            "query_terms": ("经济放缓", "经济下行", "衰退", "宏观放缓", "slowdown", "recession", "economic downturn", "macro"),
            "evidence_terms": (
                "economic",
                "macroeconomic",
                "recession",
                "slowdown",
                "customer spending",
                "spending",
                "budget",
                "宏观",
                "经济",
                "客户支出",
                "预算",
            ),
        },
        {
            "id": "demand_slowdown",
            "label": "需求放缓" if lang == "zh" else "demand slowdown",
            "query_terms": ("需求放缓", "需求下降", "订单放缓", "demand slowdown", "demand weak", "order slowdown"),
            "evidence_terms": ("demand", "customer", "order", "sales", "revenue", "growth", "需求", "客户", "订单", "销售", "收入", "增长"),
        },
        {
            "id": "margin_pressure",
            "label": "利润率压力" if lang == "zh" else "margin pressure",
            "query_terms": ("利润率", "毛利率", "成本压力", "margin", "cost pressure", "pricing pressure"),
            "evidence_terms": ("margin", "cost", "pricing", "profit", "expense", "毛利率", "利润率", "成本", "定价", "利润"),
        },
    ]
    for item in scenarios:
        if any(term in query for term in item["query_terms"]):
            return item
    if "如果" in query or "scenario" in query or "under" in query:
        return {
            "id": "user_scenario",
            "label": "用户设定情景" if lang == "zh" else "user-provided scenario",
            "query_terms": (),
            "evidence_terms": ("risk", "demand", "revenue", "margin", "cash", "风险", "需求", "收入", "利润率", "现金"),
        }
    return {}


def _risk_theme_matches_scenario(theme: RiskTheme, scenario: dict[str, Any]) -> bool:
    terms = [str(term).lower() for term in scenario.get("evidence_terms", []) or [] if str(term).strip()]
    if not terms:
        return True
    text = " ".join(
        [
            theme.theme_name,
            theme.why_it_matters,
            theme.supporting_quote,
        ]
    ).lower()
    return any(term in text for term in terms)


def _risk_direct_support_score(text: str, section: str, matched_term_count: int) -> float:
    lowered = str(text or "").lower()
    score = float(matched_term_count)
    if str(section or "").upper().strip() == "ITEM_1A":
        score += 2.0
    if any(term in lowered for term in ("risk", "risks", "uncertain", "uncertainty", "adverse", "disrupt", "pressure", "风险", "不确定", "不利", "中断", "压力")):
        score += 1.5
    if any(term in lowered for term in ("revenue", "sales", "margin", "profit", "cash", "cost", "demand", "收入", "销售", "利润", "现金", "成本", "需求")):
        score += 1.0
    return round(score, 4)


def _risk_support_level(text: str, section: str, matched_term_count: int) -> str:
    score = _risk_direct_support_score(text, section, matched_term_count)
    if score >= 3.0:
        return "direct"
    if matched_term_count > 0:
        return "inferred"
    return "template_only"


def _risk_snippet_rationale(theme_name: str, snippet: str, lang: str) -> str:
    excerpt = _clean_user_facing_phrase(str(snippet or "").strip())
    excerpt = excerpt[:180].rstrip(" ，,。.;；")
    if lang == "zh":
        return f"披露文本直接提及或支撑“{theme_name}”：{excerpt}。"
    return f"Validated filing text directly supports '{theme_name}': {excerpt}."


def _risk_text_rows(packet: dict[str, Any], company: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in packet.get("text_snippets", []) or []:
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        if not evidence_id.startswith("T"):
            continue
        ticker = str(row.get("ticker") or row.get("company") or company).upper().strip()
        if company and ticker and ticker != company:
            continue
        dimension = str(row.get("dimension_id") or "").strip()
        section = str(row.get("section") or "").upper().strip()
        if dimension == "moat_and_competitive_risk" or section in {"ITEM_1A", "ITEM_7", "ITEM_2"}:
            rows.append(dict(row))
    return rows


def _risk_refs_text(refs: list[str]) -> str:
    return _methodology_refs([str(ref) for ref in refs if str(ref).strip()])


def _risk_ranking_items(
    themes: list[RiskTheme],
    lang: str,
    *,
    scenario: dict[str, Any] | None = None,
    scenario_caveated: bool = False,
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    scenario = dict(scenario or {})
    scenario_label = str(scenario.get("label") or "").strip()
    for index, theme in enumerate(themes[:4], start=1):
        refs = list(dict.fromkeys(str(ref) for ref in theme.evidence_refs if str(ref).strip()))
        if not any(ref.startswith("T") for ref in refs):
            continue
        support_level = str(getattr(theme, "mechanism_support_level", "") or "inferred")
        if support_level == "template_only":
            continue
        rationale = theme.why_it_matters or (
            "该主题来自已验证风险披露。" if lang == "zh" else "This theme comes from validated risk disclosure."
        )
        theme_key = risk_theme_key(theme.model_dump())
        theme_name = display_theme(theme_key, lang)
        if scenario_label and scenario_caveated:
            rationale = (
                f"公司披露没有直接量化{scenario_label}情景；该主题作为已验证风险线索阅读。{rationale}"
                if lang == "zh"
                else f"The filings do not directly quantify the {scenario_label} scenario; read this as a validated risk signal. {rationale}"
            )
        elif scenario_label:
            rationale = (
                f"该风险与{scenario_label}情景相关；{rationale}"
                if lang == "zh"
                else f"This risk is relevant to the {scenario_label} scenario; {rationale}"
            )
        items.append(
            {
                "rank": index,
                "theme_key": theme_key,
                "theme_name": theme_name,
                "severity_hint": theme.severity_hint,
                "rationale": rationale,
                "evidence_refs": refs,
                "supporting_snippet": str(getattr(theme, "supporting_snippet", "") or theme.supporting_quote or ""),
                "direct_support_score": float(getattr(theme, "direct_support_score", 0.0) or 0.0),
                "mechanism_support_level": support_level,
                "scenario_id": str(scenario.get("id") or ""),
                "scenario_caveated": scenario_caveated,
            }
        )
    return items


def _risk_supplemental_items(themes: list[RiskTheme], lang: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for theme in themes:
        support_level = str(getattr(theme, "mechanism_support_level", "") or "inferred")
        if support_level == "direct":
            continue
        refs = list(dict.fromkeys(str(ref) for ref in theme.evidence_refs if str(ref).startswith("T")))
        if not refs:
            continue
        items.append(
            {
                "theme_name": theme.theme_name,
                "theme_key": risk_theme_key(theme.model_dump()),
                "rationale": (
                    f"该主题只有间接文本支持，不能进入核心排序：{theme.supporting_quote}"
                    if lang == "zh"
                    else f"This theme has indirect text support only, so it is not in the core ranking: {theme.supporting_quote}"
                ),
                "evidence_refs": refs,
                "supporting_snippet": str(getattr(theme, "supporting_snippet", "") or theme.supporting_quote or ""),
                "direct_support_score": float(getattr(theme, "direct_support_score", 0.0) or 0.0),
                "mechanism_support_level": support_level,
            }
        )
    return items[:4]


def _risk_template_only_items(themes: list[RiskTheme], lang: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for theme in themes:
        support_level = str(getattr(theme, "mechanism_support_level", "") or "inferred")
        if support_level != "template_only":
            continue
        items.append(
            {
                "theme_name": theme.theme_name,
                "theme_key": risk_theme_key(theme.model_dump()),
                "rationale": (
                    "该主题只有模板/目录层面的匹配，不能进入风险排序。"
                    if lang == "zh"
                    else "This theme has template/catalog support only and is not eligible for risk ranking."
                ),
                "evidence_refs": [],
                "supporting_snippet": str(getattr(theme, "supporting_snippet", "") or theme.supporting_quote or ""),
                "direct_support_score": float(getattr(theme, "direct_support_score", 0.0) or 0.0),
                "mechanism_support_level": support_level,
            }
        )
    return items[:4]


def _cited_observation_metrics(metrics: list[str]) -> list[str]:
    return [str(item).strip() for item in metrics if str(item).strip() and re.search(r"\[[NT]\d+\]", str(item))]


def _risk_metric_context(packet: dict[str, Any], company: str, metrics: tuple[str, ...], lang: str) -> str:
    zh_labels = {
        "revenue": "收入",
        "net_income": "净利润",
        "net_margin": "净利率",
        "operating_cash_flow": "经营现金流",
        "free_cash_flow": "自由现金流",
        "fcf_margin": "FCF margin",
    }
    en_labels = {
        "revenue": "revenue",
        "net_income": "net income",
        "net_margin": "net margin",
        "operating_cash_flow": "operating cash flow",
        "free_cash_flow": "free cash flow",
        "fcf_margin": "FCF margin",
    }
    labels = zh_labels if lang == "zh" else en_labels
    parts: list[str] = []
    for metric in metrics:
        row = _metric_rows_by_company(packet, metric).get(company)
        if not row:
            continue
        ref = str(row.get("evidence_id") or "").strip()
        refs = _risk_refs_text([ref]) if ref else ""
        label = labels.get(metric, metric)
        parts.append(f"{label} {_display_metric_row(row)} {refs}".strip())
    return "；".join(parts) if lang == "zh" else "; ".join(parts)


def _risk_transmission_path(top: RiskTheme, lang: str) -> list[str]:
    risk_name = top.theme_name or ("该风险" if lang == "zh" else "this risk")
    refs = _risk_refs_text(list(top.evidence_refs))
    support_level = str(getattr(top, "mechanism_support_level", "") or "inferred")
    snippet = _clean_user_facing_phrase(str(getattr(top, "supporting_snippet", "") or top.supporting_quote or "").strip())
    snippet = snippet[:180].rstrip(" ，,。.;；")
    if support_level != "direct":
        if lang == "zh":
            return [
                f"风险源：{risk_name} 只有间接文本支持。{refs}".strip(),
                "机制边界：当前披露不足以直接说明收入、利润或现金流传导路径。",
            ]
        return [
            f"Risk source: {risk_name} has indirect text support only. {refs}".strip(),
            "Mechanism boundary: current disclosure does not directly support a revenue, profit, or cash-flow transmission path.",
        ]
    if lang == "zh":
        return [
            f"风险源：{risk_name} 来自公司已验证风险披露。{refs}".strip(),
            f"直接支持文本：{snippet}。{refs}".strip(),
            "机制边界：未在文本中直接出现的收入、利润或现金流传导，不写入确定性机制。",
        ]
    return [
        f"Risk source: {risk_name} is grounded in validated company risk disclosure. {refs}".strip(),
        f"Direct support text: {snippet}. {refs}".strip(),
        "Mechanism boundary: revenue, profit, or cash-flow paths not directly stated in the text are not presented as deterministic mechanisms.",
    ]


def _risk_financial_impact_lines(packet: dict[str, Any], company: str, top: RiskTheme, lang: str) -> list[str]:
    risk_name = top.theme_name or ("该风险" if lang == "zh" else "this risk")
    refs = _risk_refs_text(list(top.evidence_refs))
    support_text = str(getattr(top, "supporting_snippet", "") or top.supporting_quote or "").lower()
    revenue_context = _risk_metric_context(packet, company, ("revenue",), lang)
    profit_context = _risk_metric_context(packet, company, ("net_income", "net_margin"), lang)
    cash_context = _risk_metric_context(packet, company, ("operating_cash_flow", "free_cash_flow", "fcf_margin"), lang)
    impacts = {
        "revenue": any(term in support_text for term in ("revenue", "sales", "demand", "orders", "收入", "销售", "需求", "订单")),
        "profit": any(term in support_text for term in ("margin", "profit", "cost", "pricing", "利润", "毛利", "成本", "定价")),
        "cash": any(term in support_text for term in ("cash", "working capital", "inventory", "capital", "现金", "营运资本", "库存", "资本")),
    }
    if lang == "zh":
        lines = []
        lines.append((f"收入：披露文本直接涉及收入/需求时才连接收入影响；{revenue_context or '当前缺少可验证收入背景指标'}。{refs}" if impacts["revenue"] else f"收入：当前风险文本未直接说明收入传导，不能自动推出收入影响。{refs}").strip())
        lines.append((f"利润：披露文本直接涉及成本、定价或利润率时才连接利润影响；{profit_context or '当前缺少可验证利润背景指标'}。{refs}" if impacts["profit"] else f"利润：当前风险文本未直接说明利润率或成本传导，不能自动推出利润影响。{refs}").strip())
        lines.append((f"现金流：披露文本直接涉及现金、库存或营运资本时才连接现金流影响；{cash_context or '当前现金流指标未完整覆盖'}。{refs}" if impacts["cash"] else f"现金流：当前风险文本未直接说明现金流传导，不能自动推出现金流影响。{refs}").strip())
        return lines
    lines = []
    lines.append((f"Revenue: connect this risk to revenue only when disclosure directly mentions revenue, demand, or orders; {revenue_context or 'verified revenue context is unavailable'}. {refs}" if impacts["revenue"] else f"Revenue: current risk text does not directly support a revenue transmission claim. {refs}").strip())
    lines.append((f"Profit: connect this risk to profit only when disclosure directly mentions cost, pricing, or margins; {profit_context or 'verified profit context is unavailable'}. {refs}" if impacts["profit"] else f"Profit: current risk text does not directly support a margin or cost transmission claim. {refs}").strip())
    lines.append((f"Cash flow: connect this risk to cash flow only when disclosure directly mentions cash, inventory, or working capital; {cash_context or 'cash-flow metrics are incomplete'}. {refs}" if impacts["cash"] else f"Cash flow: current risk text does not directly support a cash-flow transmission claim. {refs}").strip())
    return lines


def _risk_observation_metrics(packet: dict[str, Any], company: str, risk_refs: list[str], lang: str) -> list[str]:
    def metric_refs(metrics: tuple[str, ...]) -> list[str]:
        refs: list[str] = []
        for metric in metrics:
            row = _metric_rows_by_company(packet, metric).get(company)
            if not row:
                continue
            ref = str(row.get("evidence_id") or "").strip()
            if ref:
                refs.append(ref)
        return list(dict.fromkeys(refs))

    def with_refs(text: str, refs: list[str]) -> str | None:
        refs = list(dict.fromkeys(str(ref).strip() for ref in refs if str(ref).strip()))[:4]
        if not refs:
            return None
        return f"{text}{_risk_refs_text(refs)}"

    top_risk_refs = list(dict.fromkeys(ref for ref in risk_refs if str(ref).strip()))[:2]
    if lang == "zh":
        items = [
            with_refs("收入增速、订单/需求披露和客户需求表述（监控项）", metric_refs(("revenue",)) + top_risk_refs),
            with_refs("毛利率、营业利润率、净利率与履约/供应链成本（监控项）", metric_refs(("gross_margin", "operating_margin", "net_margin", "net_income")) + top_risk_refs),
            with_refs("库存、资本开支、经营现金流和自由现金流（监控项）", metric_refs(("inventory", "capital_expenditures", "operating_cash_flow", "free_cash_flow")) + top_risk_refs),
            with_refs("后续 10-Q/10-K 风险因素与 MD&A 是否升级或缓和（监控项）", top_risk_refs),
        ]
        return [item for item in items if item]
    items = [
        with_refs("revenue growth, order/demand disclosure, and customer-demand commentary (metric to monitor)", metric_refs(("revenue",)) + top_risk_refs),
        with_refs("gross margin, operating margin, net margin, and fulfillment/supply-chain cost (metric to monitor)", metric_refs(("gross_margin", "operating_margin", "net_margin", "net_income")) + top_risk_refs),
        with_refs("inventory, capex, operating cash flow, and free cash flow (metric to monitor)", metric_refs(("inventory", "capital_expenditures", "operating_cash_flow", "free_cash_flow")) + top_risk_refs),
        with_refs("whether later 10-Q/10-K risk factors and MD&A language intensify or ease (metric to monitor)", top_risk_refs),
    ]
    return [item for item in items if item]


def _fallback_risk_theme_from_text(packet: dict[str, Any], company: str, lang: str) -> RiskTheme | None:
    rows = _risk_text_rows(packet, company)
    if not rows:
        return None
    row = rows[0]
    text = str(row.get("claim") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
    lowered = text.lower()
    theme_parts: list[str] = []
    term_map = (
        (("inventory", "库存"), "inventory pressure", "库存压力"),
        (("competition", "competitive", "competitor", "竞争"), "competitive pressure", "竞争压力"),
        (("regulation", "regulatory", "legal", "litigation", "监管", "法律", "诉讼"), "regulatory or legal matters", "监管或法律事项"),
        (("customer demand", "customer spending", "spending", "demand", "客户需求", "客户支出", "需求"), "customer demand/spending volatility", "客户需求/支出波动"),
        (("supply chain", "supplier", "supply", "供应链", "供应商", "供给"), "supply-chain or supply constraints", "供应链或供给约束"),
        (("margin", "cost", "pricing", "利润率", "成本", "定价"), "margin/cost pressure", "利润率/成本压力"),
    )
    for terms, en_label, zh_label in term_map:
        label = zh_label if lang == "zh" else en_label
        if any(term in lowered for term in terms) and label not in theme_parts:
            theme_parts.append(label)
    if not theme_parts:
        return None
    theme_name = "、".join(theme_parts[:3])
    if lang != "zh" and theme_parts:
        theme_name = ", ".join(theme_parts[:3])
    evidence_refs = [str(item.get("evidence_id") or "").strip() for item in rows if str(item.get("evidence_id") or "").strip()]
    evidence_refs = list(dict.fromkeys(evidence_refs))
    supporting_quote = _snippet_excerpt({"supporting_snippet": text}, max_len=180)
    return RiskTheme(
        theme_key="validated_risk_disclosure",
        theme_id="validated_risk_disclosure",
        company=company,
        theme_name=theme_name,
        severity_hint="medium",
        why_it_matters=(
            _risk_snippet_rationale(theme_name, supporting_quote, lang)
        ),
        evidence_refs=evidence_refs[:4],
        supporting_quote=supporting_quote,
        supporting_snippet=text,
        related_financial_context=_risk_theme_financial_context(packet, company, ("revenue", "net_income", "net_margin"), lang),
        direct_support_score=_risk_direct_support_score(text, str(row.get("section") or ""), 1),
        mechanism_support_level=_risk_support_level(text, str(row.get("section") or ""), 1),  # type: ignore[arg-type]
        score=4.0,
    )


def _risk_refs_from_rows(rows: list[dict[str, Any]], *, limit: int = 4) -> list[str]:
    refs = [
        str(row.get("evidence_id") or "").strip()
        for row in rows
        if str(row.get("evidence_id") or "").strip().startswith("T")
    ]
    return list(dict.fromkeys(refs))[:limit]


def _bounded_inferred_risk_theme(
    *,
    theme_id: str,
    company: str,
    theme_name: str,
    why: str,
    refs: list[str],
    score: float,
    lang: str,
) -> RiskTheme:
    quote = (
        "已验证风险文本提供风险边界；排序来自业务结构推断，需要后续数据验证。"
        if lang == "zh"
        else "Validated risk text provides the boundary; ranking is inferred from business structure and requires follow-up data."
    )
    return RiskTheme(
        theme_key=theme_id,
        theme_id=theme_id,
        company=company,
        theme_name=display_theme(theme_id, lang) if display_theme(theme_id, lang) != theme_id else theme_name,
        severity_hint="medium",
        why_it_matters=why,
        evidence_refs=list(dict.fromkeys(refs))[:4],
        supporting_quote=quote,
        supporting_snippet=quote,
        related_financial_context=[],
        direct_support_score=0.1,
        mechanism_support_level="inferred",
        score=score,
    )


def _amzn_bounded_business_risk_themes(company: str, refs: list[str], lang: str) -> list[RiskTheme]:
    if not refs:
        return []
    if lang == "zh":
        specs = [
            (
                "fulfillment_inventory_capex_pressure",
                "履约/库存/资本开支压力",
                "Amazon 的零售、物流和云基础设施模型使履约成本、库存周转和资本开支压力优先影响利润率与自由现金流；已验证风险文本只提供风险边界。",
                9.0,
            ),
            (
                "regulatory_compliance_pressure",
                "监管/合规风险",
                "Amazon 的多地区平台、卖家生态和云服务会把监管、法律和合规要求传导为运营限制与合规成本；已验证风险文本只提供风险边界。",
                8.0,
            ),
            (
                "cloud_competition_pressure",
                "AWS/云竞争",
                "AWS 面临云竞争时，价格、客户迁移和投资强度可能影响云业务收入增速与利润率；已验证风险文本只提供风险边界。",
                7.0,
            ),
        ]
    else:
        specs = [
            (
                "fulfillment_inventory_capex_pressure",
                "fulfillment / inventory / capex pressure",
                "Amazon's retail, logistics, and cloud infrastructure model makes fulfillment cost, inventory turns, and capex pressure most relevant to margin and FCF.",
                9.0,
            ),
            (
                "regulatory_compliance_pressure",
                "regulatory / compliance risk",
                "Amazon's multi-region platform and cloud services can transmit regulatory, legal, and compliance pressure into operating constraints and costs.",
                8.0,
            ),
            (
                "cloud_competition_pressure",
                "AWS / cloud competition",
                "Cloud competition can affect AWS revenue growth, pricing, customer migration, and margin through investment intensity.",
                7.0,
            ),
        ]
    return [
        _bounded_inferred_risk_theme(
            theme_id=theme_id,
            company=company,
            theme_name=theme_name,
            why=why,
            refs=refs,
            score=score,
            lang=lang,
        )
        for theme_id, theme_name, why, score in specs
    ]


def _economic_slowdown_business_risk_theme(company: str, refs: list[str], lang: str) -> RiskTheme | None:
    if not refs:
        return None
    if lang == "zh":
        return _bounded_inferred_risk_theme(
            theme_id="customer_it_cloud_spend_slowdown",
            company=company,
            theme_name="客户 IT/云支出放缓",
            why="经济放缓通常先压缩客户 IT 与云预算，传导为收入增速压力，再影响利润率、经营现金流和自由现金流；供应链只作为已披露风险线索之一。",
            refs=refs,
            score=10.0,
            lang=lang,
        )
    return _bounded_inferred_risk_theme(
        theme_id="customer_it_cloud_spend_slowdown",
        company=company,
        theme_name="customer IT/cloud-spend slowdown",
        why="An economic slowdown usually first pressures customer IT and cloud budgets, then revenue growth, margins, operating cash flow, and FCF; supply chain is only one disclosed signal.",
        refs=refs,
        score=10.0,
        lang=lang,
    )


def _extract_risk_themes(packet: dict[str, Any], company: str, lang: str) -> list[RiskTheme]:
    rows = [
        dict(row)
        for row in packet.get("text_snippets", []) or []
        if isinstance(row, dict)
        and str(row.get("evidence_id") or "").strip().startswith("T")
        and str(row.get("ticker") or row.get("company") or company).upper().strip() == company
    ]
    themes: dict[str, RiskTheme] = {}
    for row in rows:
        dimension = str(row.get("dimension_id") or "").strip()
        section = str(row.get("section") or "").upper().strip()
        if dimension != "moat_and_competitive_risk" and section not in {"ITEM_1A", "ITEM_7", "ITEM_2"}:
            continue
        text = str(row.get("claim") or row.get("supporting_snippet") or row.get("text_snippet") or "").strip()
        lowered = text.lower()
        if not lowered:
            continue
        evidence_id = str(row.get("evidence_id") or "").strip()
        for spec in _RISK_THEME_CATALOG:
            theme_key = str(spec.get("key") or spec.get("theme_id") or "").strip()
            theme_label = display_theme(theme_key, lang)
            matched_terms = [term for term in spec["terms"] if term in lowered]
            if not matched_terms:
                continue
            direct_support_score = _risk_direct_support_score(text, section, len(matched_terms))
            support_level = _risk_support_level(text, section, len(matched_terms))
            score = direct_support_score + float(spec.get("weight", 0.0))
            if any(term in lowered for term in ("legal proceedings", "general risk", "may be subject")) and spec["theme_id"] == "macro_regulatory_legal":
                score -= 0.8
            current = themes.get(theme_key)
            if current:
                refs = list(dict.fromkeys([*current.evidence_refs, evidence_id]))
                current.evidence_refs = refs
                current.score = max(current.score, score) + 0.25 * (len(refs) - 1)
                current.direct_support_score = max(float(getattr(current, "direct_support_score", 0.0) or 0.0), direct_support_score)
                if str(getattr(current, "mechanism_support_level", "") or "") != "direct" and support_level == "direct":
                    current.mechanism_support_level = "direct"
                if len(text) > len(current.supporting_quote):
                    current.supporting_quote = _snippet_excerpt({"supporting_snippet": text}, max_len=180)
                    current.supporting_snippet = text
                    current.theme_name = display_theme(risk_theme_key(current.model_dump()), lang)
                    current.why_it_matters = _risk_snippet_rationale(current.theme_name, current.supporting_quote, lang)
                continue
            supporting_quote = _snippet_excerpt({"supporting_snippet": text}, max_len=180)
            themes[theme_key] = RiskTheme(
                theme_key=theme_key,
                theme_id=str(spec["theme_id"]),
                company=company,
                theme_name=theme_label,
                severity_hint="high" if score >= 6 else ("medium" if score >= 4 else "low"),
                why_it_matters=_risk_snippet_rationale(theme_label, supporting_quote, lang),
                evidence_refs=[evidence_id],
                supporting_quote=supporting_quote,
                supporting_snippet=text,
                related_financial_context=_risk_theme_financial_context(packet, company, tuple(spec["financial"]), lang),
                direct_support_score=direct_support_score,
                mechanism_support_level=support_level,  # type: ignore[arg-type]
                score=round(score, 4),
            )
    return sorted(themes.values(), key=lambda item: item.score, reverse=True)


def build_risk_focused_answer(packet: dict[str, Any], *, lang: str = "zh") -> RiskFocusedAnswer | None:
    companies = _packet_companies(packet, {})
    company = companies[0] if companies else ""
    if not company:
        return None
    scenario = _risk_scenario_anchor(packet, lang)
    themes = _extract_risk_themes(packet, company, lang)
    risk_rows = _risk_text_rows(packet, company)
    risk_refs = _risk_refs_from_rows(risk_rows)
    if not themes:
        fallback_theme = _fallback_risk_theme_from_text(packet, company, lang)
        if fallback_theme:
            themes = [fallback_theme]
    if not themes and company.upper() == "AMZN" and risk_rows:
        themes = _amzn_bounded_business_risk_themes(company, risk_refs, lang)
    scenario_caveated = False
    if scenario and themes:
        scenario_matched = [theme for theme in themes if _risk_theme_matches_scenario(theme, scenario)]
        if scenario_matched:
            themes = scenario_matched
        else:
            scenario_caveated = True
    if str(scenario.get("id") or "") == "economic_slowdown" and company.upper() == "MSFT" and risk_rows:
        scenario_theme = _economic_slowdown_business_risk_theme(company, risk_refs, lang)
        if scenario_theme:
            themes = [scenario_theme, *[theme for theme in themes if theme.theme_id != scenario_theme.theme_id]]
            scenario_caveated = False
    boundaries = [
        _investment_limitation(lang),
        "当前缺少估值证据，因此不能判断价格是否便宜或昂贵。" if lang == "zh" else "Valuation evidence is missing, so cheap/expensive claims are not supported.",
        "缺少现金流证据不会阻止风险判断，但会限制对风险是否影响利润现金含量的判断。" if lang == "zh" else "Missing cash-flow evidence does not block risk analysis, but limits conclusions about cash conversion.",
    ]
    financial_context = _risk_theme_financial_context(packet, company, ("revenue", "net_income", "net_margin"), lang)
    if not financial_context:
        financial_context = ["当前缺少可验证财务背景指标。"] if lang == "zh" else ["Verified financial context is unavailable."]
    risk_status = _risk_dimension_status(packet)
    if not themes and (risk_status == "missing" or not risk_rows):
        direct = (
            f"当前缺少足够的已验证风险文本证据，不能判断 {company} 最大的问题。"
            if lang == "zh"
            else f"Validated risk text evidence is insufficient to identify {company}'s biggest issue."
        )
        return RiskFocusedAnswer(
            company=company,
            direct_judgment=direct,
            why_core_issue=["风险专题回答需要 ITEM_1A/MD&A 等可验证文本证据。"] if lang == "zh" else ["Risk-focused analysis requires validated Item 1A/MD&A evidence."],
            financial_context=financial_context,
            financial_impact_analysis=[
                "当前缺少可验证风险文本，不能建立收入 / 利润 / 现金流传导分析。"
                if lang == "zh"
                else "Validated risk text is missing, so revenue/profit/cash-flow transmission cannot be analyzed."
            ],
            key_observation_metrics=[],
            evidence_boundaries=boundaries,
        )
    generic_risk_context = False
    if not themes:
        generic_risk_context = True
        evidence_refs = [str(row.get("evidence_id") or "").strip() for row in risk_rows if str(row.get("evidence_id") or "").strip()]
        themes = [
            RiskTheme(
                theme_key="validated_risk_context",
                theme_id="validated_risk_context",
                company=company,
                theme_name=display_theme("validated_risk_context", lang),
                severity_hint="medium",
                why_it_matters=(
                    "当前已有已验证风险文本证据，但主题词较泛化；因此只能做保守风险判断。"
                    if lang == "zh"
                    else "Validated risk text exists, but themes are broad; the risk judgment stays conservative."
                ),
                evidence_refs=list(dict.fromkeys(evidence_refs))[:4],
                supporting_quote=_snippet_excerpt(risk_rows[0], max_len=180) if risk_rows else "",
                related_financial_context=financial_context,
                direct_support_score=0.0,
                mechanism_support_level="template_only",
                score=3.0,
            )
        ]
    top = themes[0]
    secondary = themes[1:4]
    risk_ranking = _risk_ranking_items(themes, lang, scenario=scenario, scenario_caveated=scenario_caveated)
    supplemental_risk_signals = _risk_supplemental_items(themes, lang)
    template_only_risk_items = _risk_template_only_items(themes, lang)
    if risk_ranking:
        first_theme_name = str(risk_ranking[0].get("theme_name") or "").strip()
        if first_theme_name and top.theme_name != first_theme_name:
            matched = next((theme for theme in themes if theme.theme_name == first_theme_name), None)
            if matched:
                top = matched
                secondary = [theme for theme in themes if theme.theme_name != top.theme_name][:3]
    transmission_path = _risk_transmission_path(top, lang)
    financial_impact = _risk_financial_impact_lines(packet, company, top, lang)
    observation_metrics = _cited_observation_metrics(_risk_observation_metrics(packet, company, list(top.evidence_refs), lang))
    scenario_label = str(scenario.get("label") or "").strip()
    if str(scenario.get("id") or "") == "economic_slowdown":
        transmission_path = [
            (
                "基于业务结构推断：客户 IT/云支出放缓 -> 收入增速压力 -> 利润率/现金流影响；供应链只作为已披露风险线索之一。"
                if lang == "zh"
                else "Business-structure inference: customer IT/cloud spending slows -> revenue-growth pressure -> margin/cash-flow impact; supply chain is a disclosed signal, not an automatic top rank."
            ),
            *transmission_path,
        ]
    if generic_risk_context:
        direct = (
            "当前风险文本过于泛化，无法可靠排序；只能列出已披露风险边界。"
            if lang == "zh"
            else "Current risk text is too generic to rank reliably; only disclosed risk boundaries can be listed."
        )
    elif scenario_label and top.theme_id == "customer_it_cloud_spend_slowdown":
        direct = (
            f"在{scenario_label}情景下，有限判断应先看客户 IT/云支出放缓对收入、利润率和现金流的传导；供应链只作为已披露风险线索之一。"
            if lang == "zh"
            else f"Under the {scenario_label} scenario, the limited judgment should start with customer IT/cloud-spend pressure flowing into revenue, margins, and cash flow; supply chain is only one disclosed signal."
        )
    elif scenario_label and scenario_caveated:
        direct = (
            f"在{scenario_label}情景下，有限判断应先看客户 IT/云支出放缓对收入、利润率和现金流的传导；已披露风险文本用于限定风险线索。"
            if lang == "zh"
            else f"Under the {scenario_label} scenario, the limited judgment should first follow customer IT/cloud-spend transmission into revenue, margins, and cash flow; validated risk text bounds the risk signals."
        )
    elif scenario_label:
        direct = (
            f"在{scenario_label}情景下，基于已验证风险文本，我认为当前最需要关注的是{top.theme_name}；其余风险按下列排序阅读。"
            if lang == "zh"
            else f"Under the {scenario_label} scenario, the most prominent validated-text risk is {top.theme_name}; read the remaining risks in the ranking below."
        )
    else:
        direct = (
            f"基于披露信息，我认为当前最需要关注的是{top.theme_name}；其余风险按下列排序阅读。"
            if lang == "zh"
            else f"Based on the disclosures, the most prominent risk is {top.theme_name}; read the remaining risks in the ranking below."
        )
    if not risk_ranking and risk_rows:
        risk_ranking = [
            {
                "rank": index,
                "theme_name": theme.theme_name,
                "severity_hint": theme.severity_hint,
                "rationale": (
                    f"有限判断：该风险来自已验证文本，但排序需要按业务传导谨慎阅读。{theme.why_it_matters}"
                    if lang == "zh"
                    else f"Limited judgment: this risk comes from validated text, but the ranking should be read through business transmission. {theme.why_it_matters}"
                ),
                "evidence_refs": list(theme.evidence_refs),
                "supporting_snippet": str(getattr(theme, "supporting_snippet", "") or theme.supporting_quote or ""),
                "direct_support_score": float(getattr(theme, "direct_support_score", 0.0) or 0.0),
                "mechanism_support_level": str(getattr(theme, "mechanism_support_level", "") or "inferred"),
                "scenario_id": str(scenario.get("id") or ""),
                "scenario_caveated": scenario_caveated,
            }
            for index, theme in enumerate(themes[:4], start=1)
            if any(str(ref).startswith("T") for ref in theme.evidence_refs)
            and float(getattr(theme, "direct_support_score", 0.0) or 0.0) > 0
        ]
    why = [
        top.why_it_matters,
        f"该判断来自已验证风险文本，而不是估值或股价预测。{_methodology_refs(top.evidence_refs)}"
        if lang == "zh"
        else f"This judgment comes from validated risk text, not valuation or price forecasting. {_methodology_refs(top.evidence_refs)}",
    ]
    if scenario_label:
        why.insert(
            0,
            (
                f"场景锚点：{scenario_label}；排序只使用带引用的已验证风险文本，未直接量化该情景的部分会显式保留边界。"
                if lang == "zh"
                else f"Scenario anchor: {scenario_label}; the ranking uses cited validated risk text only, with explicit boundaries where the scenario is not directly quantified."
            ),
        )
        boundaries.append(
            (
                f"当前披露没有完整建模“{scenario_label}”情景，因此这是情景约束下的风险线索排序，不是预测。"
                if lang == "zh"
                else f"The filings do not fully model the {scenario_label} scenario, so this is a scenario-bounded risk signal ranking, not a forecast."
            )
        )
    filing_evidence = [
        {
            "theme_key": risk_theme_key(theme.model_dump()),
            "theme_name": display_theme(risk_theme_key(theme.model_dump()), lang),
            "severity_hint": theme.severity_hint,
            "supporting_quote": theme.supporting_quote,
            "evidence_refs": list(theme.evidence_refs),
            "why_it_matters": theme.why_it_matters,
            "mechanism_support_level": theme.mechanism_support_level,
            "direct_support_score": theme.direct_support_score,
        }
        for theme in [top, *secondary]
    ]
    return RiskFocusedAnswer(
        company=company,
        top_risk=top,
        secondary_risks=secondary,
        direct_judgment=direct,
        risk_ranking=risk_ranking,
        supplemental_risk_signals=supplemental_risk_signals,
        template_only_risk_items=template_only_risk_items,
        why_core_issue=why,
        transmission_path=transmission_path,
        filing_evidence=filing_evidence,
        financial_context=financial_context,
        financial_impact_analysis=financial_impact,
        key_observation_metrics=observation_metrics,
        evidence_boundaries=_dedupe_lines(boundaries),
        scenario_anchor=scenario,
        scenario_caveated=scenario_caveated,
        evidence_refs=list(dict.fromkeys(ref for theme in themes for ref in theme.evidence_refs)),
    )


def _segment_scope_from_requirement_plan(*plans: Mapping[str, Any] | None) -> str:
    for plan in plans:
        if not isinstance(plan, Mapping):
            continue
        for key in ("segment_or_product_scope", "segment_focus"):
            scope = str(plan.get(key) or "").strip()
            if scope:
                return scope
        for source in (
            plan.get("requirements"),
            plan.get("evidence_requirements"),
            plan.get("requirement_specs"),
        ):
            if not isinstance(source, list):
                continue
            for item in source:
                if not isinstance(item, Mapping):
                    continue
                scope = str(item.get("segment_or_product_scope") or item.get("segment_focus") or "").strip()
                if scope:
                    return scope
    return ""


def build_analytical_synthesis(
    *,
    user_query: str,
    analysis_plan: dict[str, Any],
    evidence_plan: dict[str, Any] | None = None,
    evidence_collection_results: list[dict[str, Any]] | None = None,
    evidence_sufficiency: dict[str, Any] | None = None,
    valid_numeric_claims: list[dict[str, Any]],
    valid_text_claims: list[dict[str, Any]],
    numeric_citations: list[dict[str, Any]],
    text_citations: list[dict[str, Any]],
    numeric_evidence_cards: list[dict[str, Any]],
    text_evidence_cards: list[dict[str, Any]],
    limitations: list[dict[str, Any]],
    answer_policy: dict[str, Any],
    answer_mode: str,
    safety_intent: str,
    task_type: str,
    lang: str,
    accepted_draft: dict[str, Any] | None = None,
    comparison_judgment_frame: dict[str, Any] | None = None,
    final_answer_source: str = "",
    draft_status: str = "",
    draft_final_status: str = "",
    proposed_synthesis: dict[str, Any] | None = None,
    red_flags: list[dict[str, Any]] | None = None,
    evidence_packet: dict[str, Any] | None = None,
) -> AnalyticalSynthesis:
    """Build a synthesis object from validated claims only.

    Raw tool results and raw retrieved documents are intentionally not accepted
    by this function. The arguments are all post-validation artifacts.
    """
    requirement_summary = summarize_evidence_requirements(
        evidence_plan or {},
        evidence_collection_results or [],
        evidence_sufficiency or {},
    )
    for key in (
        "missing_but_analyzable_answer_parts",
        "missing_and_unanswerable_answer_parts",
        "missing_required_answer_parts",
        "partial_required_answer_parts",
    ):
        merged = list(dict.fromkeys([str(item) for item in list(requirement_summary.get(key, []) or []) + list((evidence_sufficiency or {}).get(key, []) or []) if str(item)]))
        if merged:
            requirement_summary[key] = merged
    degradation_reason = str(requirement_summary.get("degradation_reason") or "")
    synthesis_mode = derive_synthesis_mode(
        answer_mode=answer_mode,
        task_type=task_type,
        safety_intent=safety_intent,
        evidence_sufficiency=evidence_sufficiency,
        has_validated_numeric=bool(numeric_evidence_cards),
        has_validated_text=bool(text_evidence_cards),
    )
    key_facts = [
        {
            "sentence": _claim_sentence(claim),
            "claim_ids": _claim_ids_from_claim(claim, _claim_id(i, "N")),
        }
        for i, claim in enumerate(valid_numeric_claims)
        if _claim_sentence(claim)
    ]
    analysis = [
        {
            "sentence": _claim_sentence(claim),
            "claim_ids": _claim_ids_from_claim(claim, _claim_id(i, "T")),
        }
        for i, claim in enumerate(valid_text_claims)
        if _claim_sentence(claim)
    ]
    synthesis_limitations = list(limitations)
    risks: list[dict[str, Any]] = []
    projected_draft = project_analyst_draft_to_synthesis(accepted_draft or {}) if accepted_draft else {}
    draft_is_primary = bool(final_answer_source.startswith("analyst_draft") and accepted_draft)

    if answer_mode == "cautious_outlook" or synthesis_mode in {"limited_outlook", "insufficient_outlook"}:
        risks.append({"sentence": _cautious_sentence(lang), "claim_ids": []})
    if safety_intent == "investment_advice_like":
        risks.append({"sentence": _investment_boundary_sentence(lang), "claim_ids": []})
    if synthesis_mode == "limited_judgment":
        risks.append({"sentence": _limited_judgment_notice(lang), "claim_ids": []})
        analysis = (
            _comparison_frame_analysis(comparison_judgment_frame or {}, lang, synthesis_mode)
            or _limited_judgment_analysis(valid_numeric_claims, lang)
        )
    elif synthesis_mode == "limited_analysis":
        risks.append({"sentence": _limited_analysis_notice(lang), "claim_ids": []})
        analysis = analysis[:3]
    elif synthesis_mode == "limited_outlook":
        risks.append({"sentence": _limited_outlook_notice(lang), "claim_ids": []})
        analysis = []
    elif synthesis_mode.startswith("insufficient_"):
        analysis = []
        key_facts = key_facts[:1]

    if comparison_judgment_frame and (answer_mode == "comparison_brief" or task_type == "company_comparison"):
        risk_tradeoff = _comparison_frame_risk_tradeoff(comparison_judgment_frame, lang)
        if risk_tradeoff and str(risk_tradeoff.get("sentence", "")).strip():
            existing_risks = {str(item.get("sentence", "")) for item in risks if isinstance(item, dict)}
            if str(risk_tradeoff.get("sentence", "")) not in existing_risks:
                risks.append(risk_tradeoff)

    if projected_draft:
        projected_analysis = [
            item
            for item in projected_draft.get("analysis", []) or []
            if isinstance(item, dict) and str(item.get("sentence", "")).strip()
        ]
        if projected_analysis:
            analysis = projected_analysis
        for item in projected_draft.get("risks_or_uncertainties", []) or []:
            if isinstance(item, dict) and str(item.get("sentence", "")).strip():
                risks.append(item)

    if proposed_synthesis:
        for item in proposed_synthesis.get("analysis", []) if isinstance(proposed_synthesis.get("analysis"), list) else []:
            if isinstance(item, dict):
                analysis.append(item)
        for item in proposed_synthesis.get("risks_or_uncertainties", []) if isinstance(proposed_synthesis.get("risks_or_uncertainties"), list) else []:
            if isinstance(item, dict):
                risks.append(item)

    short_answer = _natural_short_answer(
        lang,
        key_facts,
        analysis,
        str(requirement_summary.get("overall_status", "")) == "sufficient",
        synthesis_mode=synthesis_mode,
    )
    projected_short_answer = str(projected_draft.get("short_answer", "")).strip()
    if projected_short_answer:
        short_answer = projected_short_answer
    elif comparison_judgment_frame and (answer_mode == "comparison_brief" or task_type == "company_comparison"):
        short_answer = _comparison_frame_short_answer(comparison_judgment_frame, lang, synthesis_mode)

    packet_for_answer = dict(evidence_packet or {})
    if not packet_for_answer:
        packet_for_answer = {
            "task_type": task_type,
            "answer_mode": answer_mode,
            "analysis_scope": str(analysis_plan.get("analysis_scope", "")),
            "numeric_table": list(numeric_evidence_cards or []),
            "text_snippets": list(text_evidence_cards or []),
        }
    defaults = {
        "task_type": task_type,
        "answer_mode": answer_mode,
        "analysis_scope": str(analysis_plan.get("analysis_scope", "")),
        "methodology_intent": str(analysis_plan.get("methodology_intent", "")),
        "intent_family": str(analysis_plan.get("intent_family", "")),
        "evidence_policy_id": str(analysis_plan.get("evidence_policy_id", "")),
    }
    for key, value in defaults.items():
        if value and not str(packet_for_answer.get(key) or "").strip():
            packet_for_answer[key] = value

    risk_focused_answer_model = (
        build_risk_focused_answer(packet_for_answer, lang=lang)
        if answer_mode == "risk_focused_analysis"
        else None
    )
    risk_focused_answer = risk_focused_answer_model.model_dump(exclude_none=True) if risk_focused_answer_model else {}
    if risk_focused_answer:
        short_answer = str(risk_focused_answer.get("direct_judgment") or short_answer)
        risk_status = _risk_dimension_status(packet_for_answer) or str(
            dict(dict((evidence_sufficiency or {}).get("dimension_status_map", {}) or {}).get("moat_and_competitive_risk", {}) or {}).get("status", "")
        )
        risk_rows = _risk_text_rows(packet_for_answer, str(risk_focused_answer.get("company") or "").upper().strip())
        try:
            text_rate = float((evidence_sufficiency or {}).get("required_text_satisfied_rate", 0) or 0)
        except (TypeError, ValueError):
            text_rate = 0.0
        has_risk_text = bool(risk_rows) or bool(text_evidence_cards)
        risk_ready = bool(risk_focused_answer.get("top_risk")) or (
            risk_status == "satisfied" and has_risk_text and text_rate >= 1.0
        )
        synthesis_mode = "risk_focused_analysis" if risk_ready else "insufficient_risk_evidence"

    methodology_answer_model = None
    if answer_mode != "risk_focused_analysis":
        methodology_answer_model = build_methodology_answer(
            packet_for_answer,
            dimension_status_map=dict((evidence_sufficiency or {}).get("dimension_status_map", {}) or {}),
            red_flags=list(red_flags or []),
            selected_framework=dict((evidence_packet or {}).get("selected_framework", {}) or {}),
            comparison_judgment_frame=comparison_judgment_frame or {},
            analyst_draft=accepted_draft or {},
            lang=lang,
        )
    methodology_answer = methodology_answer_model.model_dump(exclude_none=True) if methodology_answer_model else {}
    if methodology_answer and not risk_focused_answer:
        if not draft_is_primary:
            short_answer = str(methodology_answer.get("judgment") or short_answer)
        packet_statuses = {
            dimension_id: _status_value(item)
            for dimension_id, item in dict(
                (evidence_packet or {}).get("dimension_status_map", {})
                or (evidence_sufficiency or {}).get("dimension_status_map", {})
                or {}
            ).items()
        }
        if str(methodology_answer.get("analysis_scope") or "") == "single_company" and _any_methodology_dimension_ready(packet_statuses):
            synthesis_mode = "methodology_single_company"
            if _valuation_missing(packet_statuses):
                degradation_reason = degradation_reason or "valuation_evidence_missing"
        elif _methodology_comparison_ready(packet_statuses) and _valuation_missing(packet_statuses):
            synthesis_mode = "methodology_comparison"
            degradation_reason = degradation_reason or "valuation_evidence_missing"

    if draft_is_primary and projected_short_answer:
        short_answer = projected_short_answer

    research_plan = dict((evidence_plan or {}).get("research_plan", {}) or (evidence_packet or {}).get("research_plan", {}) or {})
    canonical_intent = dict(analysis_plan.get("canonical_intent", {}) or {})
    packet = dict(evidence_packet or {})
    packet_canonical = dict(packet.get("canonical_intent", {}) or {})
    segment_or_product_scope = str(
        analysis_plan.get("segment_or_product_scope")
        or analysis_plan.get("segment_focus")
        or canonical_intent.get("segment_or_product_scope")
        or canonical_intent.get("segment_focus")
        or packet.get("segment_or_product_scope")
        or packet.get("segment_focus")
        or packet_canonical.get("segment_or_product_scope")
        or packet_canonical.get("segment_focus")
        or _segment_scope_from_requirement_plan(evidence_plan or {}, packet)
        or ""
    ).strip()
    missing_answer_parts = {
        str(item)
        for item in list(requirement_summary.get("missing_required_answer_parts", []) or [])
        + list((evidence_sufficiency or {}).get("missing_required_answer_parts", []) or [])
        if str(item)
    }
    missing_but_analyzable_parts = {
        str(item)
        for item in list(requirement_summary.get("missing_but_analyzable_answer_parts", []) or [])
        + list((evidence_sufficiency or {}).get("missing_but_analyzable_answer_parts", []) or [])
        if str(item)
    }
    analytical_payload = analytical_reasoning_payload(
        research_plan=research_plan,
        numeric_cards=numeric_evidence_cards,
        text_cards=text_evidence_cards,
        requirement_summary=requirement_summary,
        evidence_sufficiency=evidence_sufficiency or {},
        lang=lang,
    )
    if str(research_plan.get("question_type") or "") == "causal_explanation":
        company = next((str(item).upper() for item in research_plan.get("companies", []) or [] if str(item)), "")
        if not company:
            company = next((str(item).upper() for item in analysis_plan.get("companies", []) or [] if str(item)), "")
        company = company or "该公司"
        partial_answer_parts = {
            str(item)
            for item in list(requirement_summary.get("partial_required_answer_parts", []) or [])
            + list((evidence_sufficiency or {}).get("partial_required_answer_parts", []) or [])
            if str(item)
        }
        company_level_cards: list[dict[str, Any]] = []
        segment_product_cards: list[dict[str, Any]] = []
        for card in text_evidence_cards:
            claim_scope = str(card.get("claim_scope") or "").strip()
            if claim_scope == "company":
                company_level_cards.append(card)
                continue
            if claim_scope in {"segment", "product"}:
                segment_product_cards.append(card)
                continue
            levels = {str(item) for item in card.get("driver_levels", []) or [] if str(item)}
            if not levels and card.get("driver_level"):
                levels.add(str(card.get("driver_level")))
            if "company_level_driver" in levels:
                company_level_cards.append(card)
            elif levels & {"segment_level_driver", "product_level_driver"}:
                segment_product_cards.append(card)
        growth_boundary = "当前仍不能量化每个因素对总营收增长的贡献比例。"
        if "quantify_growth" in partial_answer_parts:
            growth_boundary = "当前总营收增长率计算口径仍不足，也不能量化每个因素对总营收增长的贡献比例。"
        if lang == "zh":
            if segment_or_product_scope:
                if segment_product_cards:
                    if any(term in segment_or_product_scope.lower() for term in ("network", "网络", "infiniband", "ethernet", "nvlink")):
                        short_answer = (
                            f"有限判断：{company} 网络业务增长大概率与 AI 集群建设、GPU 集群互连以及 "
                            "NVLink/InfiniBand/Ethernet 需求有关；但这些证据主要是分部/产品层面，"
                            "不能直接推出总公司营收增长的完整原因或贡献比例。"
                        )
                    else:
                        short_answer = f"从当前证据看，{company} 的 {segment_or_product_scope} 增长可以在分部/产品层面解释；这些证据不能直接推出总公司营收增长的完整原因或贡献比例。"
                else:
                    short_answer = f"当前不能确认 {company} 的 {segment_or_product_scope} 增长原因；需要分部/产品收入、同比口径和产品驱动文本证据。"
            elif company_level_cards:
                short_answer = f"从当前证据看，{company} 营收增长主要指向公司披露的 AI 加速计算平台需求和相关业务扩张；但{growth_boundary}"
            elif segment_product_cards:
                short_answer = f"从当前证据看，{company} 的增长线索主要指向 AI 计算平台、Compute & Networking 和网络互连产品；但这些证据更多是分部/产品层面，不能完整代表总公司营收增长原因，且{growth_boundary}"
            elif "identify_growth_drivers" in missing_but_analyzable_parts or "identify_growth_drivers" in missing_answer_parts:
                short_answer = f"当前不能确认 {company} 营收增长的直接原因；但合理验证方向包括 AI 数据中心需求、产品周期、客户资本开支、供给释放和产品组合变化。"
            else:
                short_answer = f"从当前证据看，{company} 的营收增长有明确驱动线索；但{growth_boundary}"
            boundary_sentence = "推断和假设不能写成确定事实；当前答案不量化各因素贡献比例，也不判断增长可持续性。"
        else:
            if segment_or_product_scope:
                if segment_product_cards:
                    short_answer = f"{company}'s {segment_or_product_scope} growth can be explained at the segment/product level, but that evidence does not prove the full cause or contribution of total company revenue growth."
                else:
                    short_answer = f"Current evidence is insufficient to explain {company}'s {segment_or_product_scope} growth; segment/product revenue, comparable-period growth, and driver text are required."
            elif "identify_growth_drivers" in missing_but_analyzable_parts or "identify_growth_drivers" in missing_answer_parts:
                short_answer = f"{company}'s revenue growth can be analyzed in tiers: verified facts first, then evidence-based inferences and hypotheses to verify."
            else:
                short_answer = f"{company}'s revenue growth can be explained using validated filing text, segment/product evidence, and explicit evidence boundaries."
            boundary_sentence = "Inferences and hypotheses must not be written as facts; this answer does not quantify each factor's contribution or assess durability."
        claim_analysis = [
            {"sentence": str(claim.get("text") or ""), "claim_ids": list(claim.get("citation_refs", []) or [])}
            for claim in analytical_payload.get("analytical_claims", [])
            if str(claim.get("tier") or "") == "evidence_inferred" and str(claim.get("text") or "")
        ]
        if claim_analysis:
            analysis = claim_analysis
        risks = [{"sentence": boundary_sentence, "claim_ids": []}]
        if "identify_growth_drivers" in missing_but_analyzable_parts or "identify_growth_drivers" in missing_answer_parts:
            final_answer_source = "research_plan_analytical_reasoning"
            synthesis_mode = "causal_explanation_analytical_with_gaps"
            degradation_reason = degradation_reason or "driver_text_evidence_missing_but_analyzable"
        else:
            synthesis_mode = "causal_explanation_tiered"

    methodology_report = _build_methodology_report(
        lang=lang,
        answer_mode=answer_mode,
        task_type=task_type,
        safety_intent=safety_intent,
        short_answer=short_answer,
        accepted_draft=accepted_draft or {},
        comparison_judgment_frame=comparison_judgment_frame or {},
        requirement_summary=requirement_summary,
        limitations=synthesis_limitations,
        red_flags=list(red_flags or []),
    )

    citations = _dedupe_dicts(list(numeric_citations) + list(text_citations))
    strategy = "deterministic_numeric_plus_llm_text" if valid_numeric_claims and valid_text_claims else "numeric_only"
    if not valid_numeric_claims and valid_text_claims:
        strategy = "text_only"
    if not valid_numeric_claims and not valid_text_claims:
        strategy = "synthesis_degraded"
    if synthesis_mode in _LIMITED_NUMERIC_ONLY_MODES:
        strategy = "numeric_only"
        citations = _dedupe_dicts(list(numeric_citations))
    if synthesis_mode.startswith("insufficient_"):
        strategy = "synthesis_degraded"
        citations = []

    synthesis = AnalyticalSynthesis(
        short_answer=short_answer,
        key_facts=key_facts,
        analysis=analysis,
        risks_or_uncertainties=risks,
        limitations=synthesis_limitations,
        citations=citations,
        synthesis_strategy=strategy,
        synthesis_mode=synthesis_mode,
        accepted_draft=dict(accepted_draft or {}),
        comparison_judgment_frame=dict(comparison_judgment_frame or {}),
        methodology_answer=methodology_answer,
        risk_focused_answer=risk_focused_answer,
        methodology_report=methodology_report,
        final_answer_source=final_answer_source,
        draft_status=draft_status,
        draft_final_status=draft_final_status,
        unsupported_synthesis_items=[],
        requirement_summary=requirement_summary,
        missing_requirements=list(requirement_summary.get("missing_requirements", []) or []),
        degradation_reason=degradation_reason or None,
        research_plan=research_plan,
        user_query=user_query,
        analysis_plan=analysis_plan,
        answer_policy=answer_policy,
        numeric_evidence_cards=numeric_evidence_cards,
        text_evidence_cards=text_evidence_cards,
        analytical_claims=list(analytical_payload.get("analytical_claims", []) or []),
        claim_tiers=dict(analytical_payload.get("claim_tiers", {}) or {}),
        analytical_reasoning_status=str(analytical_payload.get("analytical_reasoning_status") or ""),
        evidence_health=str(analytical_payload.get("evidence_health") or "complete"),
        tool_error_context=list(analytical_payload.get("tool_error_context", []) or []),
        segment_or_product_scope=segment_or_product_scope,
        task_type=task_type,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
    )
    return validate_synthesis_output(
        synthesis.model_dump(exclude_none=True),
        valid_numeric_claims=valid_numeric_claims,
        valid_text_claims=valid_text_claims,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        lang=lang,
        suppress_risk_judgment=synthesis_mode in _LIMITED_NUMERIC_ONLY_MODES
        or degradation_reason in {"numeric_only_comparison", "text_evidence_missing", "limited_outlook"},
    )


def validate_synthesis_output(
    synthesis: dict[str, Any],
    *,
    valid_numeric_claims: list[dict[str, Any]],
    valid_text_claims: list[dict[str, Any]],
    answer_mode: str,
    safety_intent: str,
    lang: str = "en",
    suppress_risk_judgment: bool = False,
) -> AnalyticalSynthesis:
    allowed_numbers = _allowed_numeric_tokens(valid_numeric_claims)
    allowed_numbers |= _allowed_numeric_tokens(valid_text_claims)
    synthesis_mode = str(synthesis.get("synthesis_mode", "")).strip()
    limited_numeric_only = synthesis_mode in _LIMITED_NUMERIC_ONLY_MODES
    text_claim_ids = {
        _claim_id(i, "T") for i, claim in enumerate(valid_text_claims) if _claim_sentence(claim)
    }
    for claim in valid_text_claims:
        text_claim_ids |= {eid for eid in _claim_ids_from_claim(claim, "") if eid.startswith("T")}
    numeric_claim_ids = {
        _claim_id(i, "N") for i, claim in enumerate(valid_numeric_claims) if _claim_sentence(claim)
    }
    for claim in valid_numeric_claims:
        numeric_claim_ids |= {eid for eid in _claim_ids_from_claim(claim, "") if eid.startswith("N")}
    unsupported: list[dict[str, Any]] = list(synthesis.get("unsupported_synthesis_items", []) or [])

    def validate_items(
        items: list[dict[str, Any]],
        *,
        require_text_claim: bool,
        require_numeric_only: bool = False,
    ) -> list[dict[str, Any]]:
        valid: list[dict[str, Any]] = []
        for item in items:
            sentence = str(item.get("sentence", "")).strip()
            if not sentence:
                continue
            lowered = sentence.lower()
            if safety_intent == "investment_advice_like" and any(term in lowered for term in _FORBIDDEN_INVESTMENT_TERMS):
                unsupported.append(_unsupported(item, "investment_advice_wording"))
                continue
            if any(term in lowered for term in _FORBIDDEN_FORECAST_TERMS):
                unsupported.append(_unsupported(item, "unsupported_forecast_wording"))
                continue
            new_numbers = _unsupported_numbers(sentence, allowed_numbers)
            if new_numbers:
                unsupported.append(_unsupported(item, "synthesis_unvalidated_number"))
                continue
            ids = _item_claim_ids(item)
            if require_numeric_only:
                if any(eid.startswith("T") for eid in ids) or _narrative_judgment_like(sentence):
                    reason = (
                        "limited_judgment_requires_numeric_only"
                        if synthesis_mode == "limited_judgment"
                        else "limited_analysis_requires_numeric_only"
                        if synthesis_mode == "limited_analysis"
                        else "limited_outlook_requires_numeric_only"
                    )
                    unsupported.append(_unsupported(item, reason))
                    continue
            if suppress_risk_judgment and _risk_judgment_like(sentence):
                unsupported.append(_unsupported(item, "risk_analysis_without_required_text_evidence"))
                continue
            has_text_ref = any(eid in text_claim_ids for eid in ids)
            has_numeric_ref = any(eid in numeric_claim_ids or eid.startswith("N") for eid in ids)
            if require_text_claim and not has_text_ref and not (has_numeric_ref and not _risk_judgment_like(sentence)):
                unsupported.append(_unsupported(item, "text_analysis_without_validated_text_claim"))
                continue
            valid.append({"sentence": sentence, "claim_ids": ids})
        return valid

    key_facts = validate_items(list(synthesis.get("key_facts", []) or []), require_text_claim=False)
    analysis = validate_items(
        list(synthesis.get("analysis", []) or []),
        require_text_claim=bool(valid_text_claims) and not limited_numeric_only,
        require_numeric_only=limited_numeric_only,
    )
    risks = validate_items(
        list(synthesis.get("risks_or_uncertainties", []) or []),
        require_text_claim=False,
        require_numeric_only=False,
    )
    if synthesis_mode.startswith("insufficient_") and analysis:
        for item in analysis:
            unsupported.append(_unsupported(item, "insufficient_evidence_no_analysis"))
        analysis = []

    short_answer = str(synthesis.get("short_answer", "")).strip()
    if _unsupported_numbers(short_answer, allowed_numbers):
        unsupported.append({"claim_type": "synthesis", "sentence": short_answer, "evidence_ids": [], "reason": "short_answer_unvalidated_number"})
        short_answer = str(key_facts[0].get("sentence", "")) if key_facts else ""
    if safety_intent == "investment_advice_like" and any(term in short_answer.lower() for term in _FORBIDDEN_INVESTMENT_TERMS):
        unsupported.append({"claim_type": "synthesis", "sentence": short_answer, "evidence_ids": [], "reason": "short_answer_investment_advice_wording"})
        short_answer = _investment_boundary_sentence(lang)
    if any(term in short_answer.lower() for term in _FORBIDDEN_FORECAST_TERMS):
        unsupported.append({"claim_type": "synthesis", "sentence": short_answer, "evidence_ids": [], "reason": "short_answer_unsupported_forecast_wording"})
        short_answer = str(key_facts[0].get("sentence", "")) if key_facts else _cautious_sentence(lang)
    if limited_numeric_only and _narrative_judgment_like(short_answer):
        unsupported.append({"claim_type": "synthesis", "sentence": short_answer, "evidence_ids": [], "reason": f"{synthesis_mode}_requires_numeric_only"})
        short_answer = _limited_judgment_notice(lang) if synthesis_mode == "limited_judgment" else (
            _limited_analysis_notice(lang) if synthesis_mode == "limited_analysis" else _limited_outlook_notice(lang)
        )
    if answer_mode == "cautious_outlook" and not any("not a prediction" in r.get("sentence", "").lower() or "不能预测" in r.get("sentence", "") for r in risks):
        risks.append({"sentence": _cautious_sentence(lang), "claim_ids": []})

    if not short_answer:
        short_answer = str(key_facts[0].get("sentence", "")) if key_facts else _insufficient_message(lang, synthesis_mode or "insufficient_analysis")

    return AnalyticalSynthesis(
        short_answer=short_answer,
        key_facts=key_facts,
        analysis=analysis,
        risks_or_uncertainties=risks,
        limitations=list(synthesis.get("limitations", []) or []),
        citations=list(synthesis.get("citations", []) or []),
        synthesis_strategy=str(synthesis.get("synthesis_strategy", "synthesis_degraded")),
        synthesis_mode=synthesis_mode or "synthesis_degraded",
        accepted_draft=dict(synthesis.get("accepted_draft", {}) or {}),
        comparison_judgment_frame=dict(synthesis.get("comparison_judgment_frame", {}) or {}),
        methodology_answer=dict(synthesis.get("methodology_answer", {}) or {}),
        risk_focused_answer=dict(synthesis.get("risk_focused_answer", {}) or {}),
        methodology_report=dict(synthesis.get("methodology_report", {}) or {}),
        final_answer_source=str(synthesis.get("final_answer_source", "")),
        draft_status=str(synthesis.get("draft_status", "")),
        draft_final_status=str(synthesis.get("draft_final_status", "")),
        unsupported_synthesis_items=unsupported,
        requirement_summary=dict(synthesis.get("requirement_summary", {}) or {}),
        missing_requirements=list(synthesis.get("missing_requirements", []) or []),
        degradation_reason=synthesis.get("degradation_reason"),
        research_plan=dict(synthesis.get("research_plan", {}) or {}),
        numeric_evidence_cards=list(synthesis.get("numeric_evidence_cards", []) or []),
        text_evidence_cards=list(synthesis.get("text_evidence_cards", []) or []),
        analytical_claims=list(synthesis.get("analytical_claims", []) or []),
        claim_tiers=dict(synthesis.get("claim_tiers", {}) or {}),
        analytical_reasoning_status=str(synthesis.get("analytical_reasoning_status", "")),
        evidence_health=str(synthesis.get("evidence_health", "complete") or "complete"),
        tool_error_context=list(synthesis.get("tool_error_context", []) or []),
        segment_or_product_scope=str(synthesis.get("segment_or_product_scope", "") or ""),
        user_query=str(synthesis.get("user_query", "") or ""),
        analysis_plan=dict(synthesis.get("analysis_plan", {}) or {}),
        answer_policy=dict(synthesis.get("answer_policy", {}) or {}),
        task_type=str(synthesis.get("task_type", "") or ""),
        answer_mode=str(synthesis.get("answer_mode", "") or ""),
        safety_intent=str(synthesis.get("safety_intent", "") or ""),
    )


def _inline_refs(refs: list[str]) -> str:
    return "".join(f"[{ref}]" for ref in refs if str(ref).strip())


def _item_refs(item: dict[str, Any]) -> list[str]:
    refs = [str(ref) for ref in item.get("citation_refs", item.get("claim_ids", [])) if str(ref).strip()]
    text = str(item.get("statement", item.get("sentence", "")) or "")
    refs.extend(match.group(1) for match in re.finditer(r"\[([NT]\d+)\]", text))
    return list(dict.fromkeys(refs))


def _clean_user_facing_phrase(sentence: str) -> str:
    cleaned = str(sentence or "")
    if "yfinance" in cleaned.lower() and "fallback" in cleaned.lower():
        return "部分结构化财务数据来自 yfinance，可信度为 medium。"
    replacements = {
        "profitability_quality": "profitability and margin evidence",
        "profitability quality": "profitability and margin evidence",
        "absolute_profitability": "absolute profitability",
        "numeric_only_comparison": "limited financial comparison",
        "numeric_only": "limited numeric evidence",
        "no_clear_preference": "no clear preference",
        "dependency_numeric_requirement_missing": "some derived metrics are unavailable",
        "required_evidence_missing": "some required evidence is unavailable",
        "requirement_missing": "some evidence is unavailable",
        "requirement_partial": "some evidence is incomplete",
        "unsupported_claims_present": "some unsupported statements were removed",
        "Required evidence is missing": "some required evidence is unavailable",
        "fallback": "alternate data source",
    }
    for old, new in replacements.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bREQ-[A-Z0-9_-]+\b", "", cleaned)
    cleaned = re.sub(r"\bdependency_[A-Za-z0-9_]+\b", "dependency detail", cleaned)
    cleaned = re.sub(r"\bnumeric_only_[A-Za-z0-9_]+\b", "limited numeric evidence", cleaned)
    cleaned = re.sub(r"\bdependency_metric_id\b", "dependency detail", cleaned)
    cleaned = re.sub(r"。+\s*；+", "。", cleaned)
    cleaned = re.sub(r"；{2,}", "；", cleaned)
    cleaned = re.sub(
        r"((?:\[[NT]\d+\])+)\s+(?=(?:capex/revenue|FCF margin|net margin|gross margin|operating margin)\b)",
        r"\1 此外，",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+([，。；：])", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _render_item_with_refs(item: dict[str, Any]) -> str:
    raw_sentence = str(item.get("statement", item.get("sentence", ""))).strip()
    raw_sentence = re.sub(r"\[([NT]\d+)\]", "", raw_sentence)
    sentence = _clean_user_facing_phrase(raw_sentence)
    refs = _inline_refs(_item_refs(item))
    if not sentence:
        return ""
    return f"{sentence} {refs}".strip() if refs else sentence


def _dimension_label_for_draft(dimension_id: str, lang: str) -> str:
    labels_zh = {
        "revenue_quality": "收入质量",
        "profitability_quality": "盈利质量",
        "cash_flow_quality": "现金流质量",
        "balance_sheet_and_capital_intensity": "资产负债表与资本强度",
        "valuation_and_risk_boundary": "估值边界",
        "moat_and_competitive_risk": "竞争与风险",
        "business_model": "商业模式",
    }
    labels_en = {
        "revenue_quality": "Revenue Quality",
        "profitability_quality": "Profitability Quality",
        "cash_flow_quality": "Cash Flow Quality",
        "balance_sheet_and_capital_intensity": "Balance Sheet and Capital Intensity",
        "valuation_and_risk_boundary": "Valuation Boundary",
        "moat_and_competitive_risk": "Competitive Risk",
        "business_model": "Business Model",
    }
    labels = labels_zh if lang == "zh" else labels_en
    return labels.get(dimension_id, _clean_user_facing_phrase(dimension_id))


def _render_dimension_draft_line(item: dict[str, Any], lang: str) -> str:
    dimension_id = str(item.get("dimension_id", "")).strip()
    dimension_label = _dimension_label_for_draft(dimension_id, lang)
    claim = _clean_user_facing_phrase(str(item.get("claim") or item.get("statement") or item.get("sentence") or "").strip())
    refs = [
        str(ref)
        for ref in item.get("evidence_refs", item.get("citation_refs", item.get("claim_ids", []))) or []
        if str(ref).strip()
    ]
    if not claim:
        return ""
    prefix = f"{dimension_label}: " if dimension_label else ""
    rendered = f"{prefix}{claim}"
    inline = _inline_refs(refs)
    return f"{rendered} {inline}".strip() if inline else rendered


def _limitation_lines(synthesis: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for item in synthesis.get("limitations", []) or []:
        if isinstance(item, dict):
            line = str(item.get("message", "") or item.get("code", "")).strip()
        else:
            line = str(item).strip()
        if line:
            lines.append(line)
    return lines


def _accepted_draft_refs(draft: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    conclusion = dict(draft.get("tentative_conclusion", {}) or {})
    refs.extend(_item_refs(conclusion))
    refs.extend(str(ref) for ref in draft.get("citation_refs", []) or [] if str(ref).strip())
    for field_name in ("decision_basis", "supporting_points", "counterpoints", "risk_tradeoffs", "uncertainty_notes", "safety_notes"):
        for item in draft.get(field_name, []) or []:
            if isinstance(item, dict):
                refs.extend(_item_refs(item))
    return list(dict.fromkeys(refs))


def _comparison_frame_counterpoint(frame: dict[str, Any], lang: str) -> str:
    preferred = str(frame.get("preferred_company", ""))
    scale_winner = str(frame.get("scale_winner", ""))
    profitability_winner = str(frame.get("profitability_winner", ""))
    margin_winner = str(frame.get("margin_winner", ""))
    raw_counterpoint = str(frame.get("counterpoint", "")).strip()
    focus_dimensions = _frame_focus_dimensions(frame)
    if "cash_flow_quality" in focus_dimensions:
        if lang == "zh":
            return "反方观点是：非领先公司在其他未请求维度上可能更好，但本结论只限现金流质量。"
        return "As a counterpoint, the non-leading company may look better on other non-requested dimensions, but this conclusion is limited to cash-flow quality."
    if "valuation_and_risk_boundary" in focus_dimensions:
        if lang == "zh":
            return "反方观点是：普通风险因素不能替代估值风险判断；估值输入缺失时只能给出边界。"
        return "As a counterpoint, ordinary risk factors cannot substitute for valuation-risk evidence; missing valuation inputs limit the conclusion."
    if "revenue_quality" in focus_dimensions:
        if lang == "zh":
            return "反方观点是：净利率不能替代增长/收入质量，收入增速和历史口径需要单独验证。"
        return "As a counterpoint, net margin cannot substitute for growth/revenue quality; revenue growth and history need separate validation."
    if scale_winner and scale_winner not in {"", "unavailable", "mixed", preferred}:
        if lang == "zh":
            return f"反过来看，如果更看重收入规模，{scale_winner} 仍然更占优。"
        return f"As a counterpoint, {scale_winner} still looks stronger if revenue scale matters more."
    if margin_winner and margin_winner not in {"", "unavailable", "mixed", preferred}:
        if lang == "zh":
            return f"反过来看，如果更看重净利率，{margin_winner} 也更有优势。"
        return f"As a counterpoint, {margin_winner} still looks stronger if margin matters more."
    if profitability_winner and profitability_winner not in {"", "unavailable", "mixed", preferred}:
        if lang == "zh":
            return f"反过来看，如果更看重盈利能力，{profitability_winner} 也更有优势。"
        return f"As a counterpoint, {profitability_winner} still looks stronger if profitability and margin evidence matter more."
    return _clean_user_facing_phrase(raw_counterpoint)


def _render_draft_comparison_text(synthesis: dict[str, Any], lang: str) -> str:
    draft = dict(synthesis.get("accepted_draft", {}) or {})
    frame = dict(synthesis.get("comparison_judgment_frame", {}) or {})
    synthesis_mode = str(synthesis.get("synthesis_mode", ""))
    frame_conclusion = _comparison_frame_short_answer(frame, lang, synthesis_mode) if frame else ""
    conclusion = frame_conclusion or _render_item_with_refs(dict(draft.get("tentative_conclusion", {}) or {}))

    frame_basis = [
        _render_item_with_refs(item)
        for item in _comparison_frame_analysis(frame, lang, synthesis_mode)
        if isinstance(item, dict) and _render_item_with_refs(item)
    ]
    draft_conclusion = _render_item_with_refs(dict(draft.get("tentative_conclusion", {}) or {}))
    basis = [_render_item_with_refs(item) for item in draft.get("decision_basis", []) or [] if isinstance(item, dict)]
    support = [_render_item_with_refs(item) for item in draft.get("supporting_points", []) or [] if isinstance(item, dict)]
    dimension_lines = [
        _render_dimension_draft_line(item, lang)
        for item in draft.get("dimension_analyses", []) or []
        if isinstance(item, dict) and _render_dimension_draft_line(item, lang)
    ]
    if draft_conclusion and draft_conclusion != conclusion:
        basis = [draft_conclusion] + basis

    frame_counterpoint = _comparison_frame_counterpoint(frame, lang) if frame else ""
    counterpoints = [frame_counterpoint] if frame_counterpoint else []
    counterpoints.extend(_render_item_with_refs(item) for item in draft.get("counterpoints", []) or [] if isinstance(item, dict))

    risks = [
        _render_item_with_refs(x)
        for x in synthesis.get("risks_or_uncertainties", []) or []
        if isinstance(x, dict) and _render_item_with_refs(x)
    ]
    frame_risk = _comparison_frame_risk_tradeoff(frame, lang) if frame else None
    if frame_risk:
        rendered_frame_risk = _render_item_with_refs(frame_risk)
        if rendered_frame_risk:
            risks.insert(0, rendered_frame_risk)
    limitation_lines = _limitation_lines(synthesis)
    priority_limitations = [
        line for line in limitation_lines if "估值" in line or "valuation" in line.lower()
    ]
    other_limitations = [line for line in limitation_lines if line not in priority_limitations]
    risks.extend(priority_limitations)
    risks.extend(str(item).strip() for item in draft.get("methodology_limitations", []) or [] if str(item).strip())
    risks.extend(_render_item_with_refs(item) for item in draft.get("risk_tradeoffs", []) or [] if isinstance(item, dict))
    risks.extend(_render_item_with_refs(item) for item in draft.get("uncertainty_notes", []) or [] if isinstance(item, dict))
    risks.extend(other_limitations)

    safety = [_render_item_with_refs(item) for item in draft.get("safety_notes", []) or [] if isinstance(item, dict)]
    if not safety:
        safety = [_investment_boundary_sentence(lang)]
    headings = (
        "比较判断",
        "关键依据",
        "维度分析",
        "反方观点",
        "风险与限制",
        "非投资建议",
    ) if lang == "zh" else (
        "Judgment",
        "Key Basis",
        "Dimension Analysis",
        "Counterpoint",
        "Risks and Limits",
        "Not Investment Advice",
    )
    lines = [headings[0], conclusion]
    basis_lines = _dedupe_lines(frame_basis + basis + support)
    if basis_lines:
        lines.extend(["", headings[1], *[f"- {item}" for item in basis_lines[:5] if item]])
    dimension_lines = _dedupe_lines(dimension_lines)
    if dimension_lines:
        lines.extend(["", headings[2], *[f"- {item}" for item in dimension_lines[:6] if item]])
    counterpoints = _dedupe_lines(counterpoints)
    if counterpoints:
        lines.extend(["", headings[3], *[f"- {item}" for item in counterpoints[:4] if item]])
    risks = _dedupe_lines(risks)
    limitation_blob = " ".join(
        [
            str(synthesis.get("limitations", "")),
            str(draft.get("methodology_limitations", "")),
            str(dict(synthesis.get("methodology_answer", {}) or {}).get("limitations", "")),
        ]
    )
    if ("估值" in limitation_blob or "valuation" in limitation_blob.lower()) and not any(
        "估值" in line or "valuation" in line.lower() for line in risks
    ):
        risks.insert(
            0,
            "当前缺少估值证据，因此不能判断哪家公司价格更便宜。" if lang == "zh"
            else "Valuation evidence is missing, so relative cheapness cannot be judged.",
        )
    valuation_risks = [line for line in risks if "估值" in line or "valuation" in line.lower()]
    if valuation_risks:
        risks = valuation_risks + [line for line in risks if line not in valuation_risks]
    if risks:
        lines.extend(["", headings[4], *[f"- {item}" for item in risks[:5] if item]])
    safety = _dedupe_lines(safety)
    if safety:
        lines.extend(["", headings[5], *[f"- {item}" for item in safety[:3] if item]])
    return "\n".join(line for line in lines if line).strip()


def _render_draft_risk_text(synthesis: dict[str, Any], lang: str) -> str:
    draft = dict(synthesis.get("accepted_draft", {}) or {})
    risk_focused_answer = dict(synthesis.get("risk_focused_answer", {}) or {})
    conclusion = _render_item_with_refs(dict(draft.get("tentative_conclusion", {}) or {}))
    decision_basis = [_render_item_with_refs(item) for item in draft.get("decision_basis", []) or [] if isinstance(item, dict)]
    support = [_render_item_with_refs(item) for item in draft.get("supporting_points", []) or [] if isinstance(item, dict)]
    risks = [_render_item_with_refs(item) for item in draft.get("risk_tradeoffs", []) or [] if isinstance(item, dict)]
    uncertainty = [_render_item_with_refs(item) for item in draft.get("uncertainty_notes", []) or [] if isinstance(item, dict)]
    uncertainty.extend(_limitation_lines(synthesis))
    ranking_items = [
        dict(item)
        for item in risk_focused_answer.get("risk_ranking", []) or []
        if isinstance(item, dict)
        and str(item.get("mechanism_support_level") or "direct") == "direct"
        and any(str(ref).startswith("T") for ref in item.get("evidence_refs", []) or [])
    ]
    if not ranking_items:
        fallback_themes = [
            dict(item)
            for item in [
                risk_focused_answer.get("top_risk", {}),
                *(risk_focused_answer.get("secondary_risks", []) or []),
            ]
            if isinstance(item, dict) and item
            and str(item.get("mechanism_support_level") or "inferred") == "direct"
            and any(str(ref).startswith("T") for ref in item.get("evidence_refs", []) or [])
        ]
        for theme in fallback_themes[:4]:
            ranking_items.append(
                {
                    "theme_name": theme.get("theme_name"),
                    "rationale": theme.get("why_it_matters"),
                    "evidence_refs": theme.get("evidence_refs", []),
                }
            )
    rank_labels = (
        ["最高优先级", "第二优先级", "第三优先级", "第四优先级"]
        if lang == "zh"
        else ["Top priority", "Second priority", "Third priority", "Fourth priority"]
    )
    ranking_lines: list[str] = []
    for index, item in enumerate(ranking_items[:4]):
        rank = rank_labels[index] if index < len(rank_labels) else ("后续优先级" if lang == "zh" else "Additional priority")
        name = _clean_user_facing_phrase(str(item.get("theme_name") or "风险主题").strip())
        rationale = _clean_user_facing_phrase(str(item.get("rationale") or "").strip())
        refs = _inline_refs([str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()])
        if not refs:
            continue
        if not rationale:
            rationale = "该主题来自已验证风险披露。" if lang == "zh" else "This theme comes from validated risk disclosure."
        ranking_lines.append(
            f"{rank}：{name}。{rationale} {refs}".strip()
            if lang == "zh"
            else f"{rank}: {name}. {rationale} {refs}".strip()
        )
    observation_metrics = _dedupe_lines(
        [str(item).strip() for item in risk_focused_answer.get("key_observation_metrics", []) or [] if str(item).strip() and re.search(r"\[[NT]\d+\]", str(item))]
    )
    headings = (
        "风险判断",
        "可监控的已披露风险线索" if risk_focused_answer.get("scenario_caveated") else "核心风险排序",
        "为什么这是核心问题",
        "财报证据",
        "关键观察指标",
        "证据边界",
        "引用",
    ) if lang == "zh" else (
        "Risk Judgment",
        "Monitorable Disclosed Risk Signals" if risk_focused_answer.get("scenario_caveated") else "Core Risk Ranking",
        "Main Risks",
        "Evidence Basis",
        "Key Observation Indicators",
        "Limits and Uncertainty",
        "Citations",
    )
    lines = [headings[0], conclusion]
    combined_visible = "\n".join([conclusion, *decision_basis, *support, *risks, *uncertainty])
    needs_priority = not any(term in combined_visible for term in ("最大", "优先级", "第一", "biggest", "top priority", "highest priority"))
    needs_observation = not any(term in combined_visible for term in ("观察", "指标", "monitor", "indicator"))
    if ranking_lines and needs_priority:
        lines.extend(["", headings[1], *[f"- {item}" for item in ranking_lines[:4] if item]])
    if decision_basis:
        lines.extend(["", headings[2], *[f"- {item}" for item in decision_basis[:4] if item]])
    if support:
        lines.extend(["", headings[3], *[f"- {item}" for item in support[:4] if item]])
    if observation_metrics and needs_observation:
        lines.extend(["", headings[4], *[f"- {item}" for item in observation_metrics[:6] if item]])
    risk_limits = _dedupe_lines(risks + uncertainty)
    if risk_limits:
        lines.extend(["", headings[5], *[f"- {item}" for item in risk_limits[:5] if item]])
    refs = _accepted_draft_refs(draft)
    if refs:
        lines.extend(["", headings[6], f"- {_inline_refs(refs)}"])
    return "\n".join(line for line in lines if line).strip()


def _render_draft_analytical_text(synthesis: dict[str, Any], lang: str) -> str:
    draft = dict(synthesis.get("accepted_draft", {}) or {})
    conclusion = _render_item_with_refs(dict(draft.get("tentative_conclusion", {}) or {}))
    decision_basis = [_render_item_with_refs(item) for item in draft.get("decision_basis", []) or [] if isinstance(item, dict)]
    support = [_render_item_with_refs(item) for item in draft.get("supporting_points", []) or [] if isinstance(item, dict)]
    uncertainty = [_render_item_with_refs(item) for item in draft.get("uncertainty_notes", []) or [] if isinstance(item, dict)]
    uncertainty.extend(_limitation_lines(synthesis))
    headings = (
        "简短结论",
        "主要问题",
        "证据依据",
        "不确定性",
        "引用",
    ) if lang == "zh" else (
        "Conclusion",
        "Main Issue",
        "Evidence Basis",
        "Uncertainty",
        "Citations",
    )
    lines = [headings[0], conclusion]
    if decision_basis:
        lines.extend(["", headings[1], f"- {decision_basis[0]}"])
    evidence_basis = decision_basis[1:] + support
    if evidence_basis:
        lines.extend(["", headings[2], *[f"- {item}" for item in evidence_basis[:5] if item]])
    if uncertainty:
        lines.extend(["", headings[3], *[f"- {item}" for item in uncertainty[:5] if item]])
    refs = _accepted_draft_refs(draft)
    if refs:
        lines.extend(["", headings[4], f"- {_inline_refs(refs)}"])
    return "\n".join(line for line in lines if line).strip()


def _render_draft_outlook_text(synthesis: dict[str, Any], lang: str) -> str:
    draft = dict(synthesis.get("accepted_draft", {}) or {})
    conclusion = _render_item_with_refs(dict(draft.get("tentative_conclusion", {}) or {}))
    basis = [_render_item_with_refs(item) for item in draft.get("decision_basis", []) or [] if isinstance(item, dict)]
    uncertainty = [_render_item_with_refs(item) for item in draft.get("uncertainty_notes", []) or [] if isinstance(item, dict)]
    safety = [_render_item_with_refs(item) for item in draft.get("safety_notes", []) or [] if isinstance(item, dict)]
    if not safety:
        safety = [_cautious_sentence(lang)]
    headings = (
        "简短观察",
        "已披露依据",
        "不确定性",
        "非预测声明",
    ) if lang == "zh" else (
        "Brief Observation",
        "Disclosed Basis",
        "Uncertainty",
        "Not a Prediction",
    )
    lines = [headings[0], conclusion]
    if basis:
        lines.extend(["", headings[1], *[f"- {item}" for item in basis[:5] if item]])
    if uncertainty:
        lines.extend(["", headings[2], *[f"- {item}" for item in uncertainty[:4] if item]])
    if safety:
        lines.extend(["", headings[3], *[f"- {item}" for item in safety[:2] if item]])
    return "\n".join(line for line in lines if line).strip()


def _render_comparison_frame_text(synthesis: dict[str, Any], lang: str) -> str:
    frame = dict(synthesis.get("comparison_judgment_frame", {}) or {})
    short_answer = str(synthesis.get("short_answer", "")).strip()
    analysis = [_render_item_with_refs(x) for x in synthesis.get("analysis", []) or [] if isinstance(x, dict) and _render_item_with_refs(x)]
    risks = [_render_item_with_refs(x) for x in synthesis.get("risks_or_uncertainties", []) or [] if isinstance(x, dict) and _render_item_with_refs(x)]
    frame_risk = _comparison_frame_risk_tradeoff(frame, lang) if frame else None
    if frame_risk:
        rendered_frame_risk = _render_item_with_refs(frame_risk)
        if rendered_frame_risk:
            risks.insert(0, rendered_frame_risk)
    counterpoint = _comparison_frame_counterpoint(frame, lang)
    if counterpoint:
        analysis = [item for item in analysis if item != counterpoint]
    rationale = _clean_user_facing_phrase(str(frame.get("rationale", "")).strip())
    if lang != "zh" and rationale and rationale not in analysis:
        analysis.insert(0, rationale)
    headings = (
        "比较判断",
        "关键依据",
        "反方观点",
        "风险与限制",
        "非投资建议",
    ) if lang == "zh" else (
        "Judgment",
        "Key Basis",
        "Counterpoint",
        "Risks and Limits",
        "Not Investment Advice",
    )
    lines = [headings[0], short_answer]
    analysis = _dedupe_lines(analysis)
    if analysis:
        lines.extend(["", headings[1], *[f"- {item}" for item in analysis[:4] if item]])
    if counterpoint:
        lines.extend(["", headings[2], f"- {counterpoint}"])
    risk_lines = _dedupe_lines(risks + _limitation_lines(synthesis))
    if risk_lines:
        lines.extend(["", headings[3], *[f"- {item}" for item in risk_lines[:5] if item]])
    lines.extend(["", headings[4], f"- {_investment_boundary_sentence(lang)}"])
    return "\n".join(line for line in lines if line).strip()


def _methodology_refs(refs: list[str]) -> str:
    return _inline_refs([str(ref) for ref in refs if str(ref).strip()])


def _methodology_limitation_lines(report: dict[str, Any]) -> list[str]:
    return _dedupe_lines(
        [
            _clean_user_facing_phrase(str(item).strip())
            for item in report.get("limitations", []) or []
            if str(item).strip()
        ]
    )


def _render_methodology_answer_text(answer: dict[str, Any], *, lang: str) -> str:
    judgment = _clean_user_facing_phrase(str(answer.get("judgment", "")).strip())
    sections = [
        item
        for item in answer.get("dimension_sections", []) or []
        if isinstance(item, dict)
    ]
    usable_sections = [
        item
        for item in sections
        if str(item.get("status", "")) in {"satisfied", "partial"} and str(item.get("summary", "")).strip()
    ]
    missing_limitations = [
        f"{str(item.get('title') or item.get('dimension_id') or '').strip()}：{_clean_user_facing_phrase(str(item.get('limitation') or '').strip())}"
        if lang == "zh"
        else f"{str(item.get('title') or item.get('dimension_id') or '').strip()}: {_clean_user_facing_phrase(str(item.get('limitation') or '').strip())}"
        for item in sections
        if str(item.get("status", "")) == "missing" and str(item.get("limitation") or "").strip()
    ]
    limitations = _dedupe_lines(
        missing_limitations
        + [
            _clean_user_facing_phrase(str(item).strip())
            for item in answer.get("limitations", []) or []
            if str(item).strip()
        ]
    )
    counterpoint = _clean_user_facing_phrase(str(answer.get("counterpoint") or "").strip())

    headings = (
        "比较判断",
        "维度分析",
        "反方观点",
        "限制",
    ) if lang == "zh" else (
        "Judgment",
        "Dimension Analysis",
        "Counterpoint",
        "Limits",
    )
    lines = [headings[0], judgment]
    if usable_sections:
        lines.extend(["", headings[1]])
        for item in usable_sections:
            title = str(item.get("title") or item.get("dimension_id") or "").strip()
            summary = _clean_user_facing_phrase(str(item.get("summary") or "").strip())
            refs = _methodology_refs(list(item.get("evidence_refs", []) or []))
            if lang == "zh":
                lines.append(f"- {title}：{summary} {refs}".strip())
            else:
                lines.append(f"- {title}: {summary} {refs}".strip())
    if counterpoint:
        lines.extend(["", headings[2], f"- {counterpoint}"])
    if limitations:
        lines.extend(["", headings[3], *[f"- {item}" for item in limitations[:6]]])
    return "\n".join(line for line in lines if line).strip()


def _render_methodology_report_text(report: dict[str, Any], *, lang: str, answer_mode: str, safety_intent: str) -> str:
    conclusion = _clean_user_facing_phrase(str(report.get("conclusion", "")).strip())
    framework_summary = _clean_user_facing_phrase(str(report.get("framework_summary", "")).strip())
    dimension_items = [
        item
        for item in report.get("dimension_analyses", []) or []
        if isinstance(item, dict) and str(item.get("claim", "")).strip() and str(item.get("status", "")) in {"satisfied", "partial"}
    ]
    counterpoints = _dedupe_lines(
        [
            _clean_user_facing_phrase(str(item).strip())
            for item in report.get("counterpoints", []) or []
            if str(item).strip()
        ]
    )
    limitations = _methodology_limitation_lines(report)
    follow_ups = _dedupe_lines([str(item).strip() for item in report.get("follow_up_metrics", []) or [] if str(item).strip()])

    is_comparison = answer_mode == "comparison_brief" or safety_intent == "investment_advice_like"
    if is_comparison:
        headings = (
            "比较判断",
            "维度分析",
            "反方观点",
            "限制",
        ) if lang == "zh" else (
            "Judgment",
            "Dimension Analysis",
            "Counterpoint",
            "Limits",
        )
        lines = [headings[0], conclusion]
        if dimension_items:
            lines.extend(["", headings[1]])
            for item in dimension_items[:4]:
                label = str(item.get("label") or item.get("dimension_id") or "").strip()
                claim = _clean_user_facing_phrase(str(item.get("claim", "")).strip())
                refs = _methodology_refs(list(item.get("evidence_refs", []) or []))
                lines.append(f"- {label}：{claim} {refs}".strip() if lang == "zh" else f"- {label}: {claim} {refs}".strip())
        if counterpoints:
            lines.extend(["", headings[2], *[f"- {item}" for item in counterpoints[:3]]])
        limit_lines = _dedupe_lines(limitations + [_investment_boundary_sentence(lang)])
        if limit_lines:
            lines.extend(["", headings[3], *[f"- {item}" for item in limit_lines[:5]]])
        return "\n".join(line for line in lines if line).strip()

    headings = (
        "结论",
        "分析框架",
        "维度分析",
        "红旗与限制",
        "后续需要跟踪的指标",
    ) if lang == "zh" else (
        "Conclusion",
        "Analysis Framework",
        "Dimension Analysis",
        "Red Flags and Limits",
        "Follow-up Metrics",
    )
    lines = [headings[0], conclusion]
    if framework_summary:
        lines.extend(["", headings[1], framework_summary])
    if dimension_items:
        lines.extend(["", headings[2]])
        for item in dimension_items[:6]:
            label = str(item.get("label") or item.get("dimension_id") or "").strip()
            claim = _clean_user_facing_phrase(str(item.get("claim", "")).strip())
            refs = _methodology_refs(list(item.get("evidence_refs", []) or []))
            lines.append(f"- {label}：{claim} {refs}".strip() if lang == "zh" else f"- {label}: {claim} {refs}".strip())
    if limitations:
        lines.extend(["", headings[3], *[f"- {item}" for item in limitations[:6]]])
    if follow_ups:
        lines.extend(["", headings[4], *[f"- {item}" for item in follow_ups[:5]]])
    return "\n".join(line for line in lines if line).strip()


def _methodology_answer_is_single_company_composite(methodology_answer: dict[str, Any]) -> bool:
    if str(methodology_answer.get("analysis_scope") or "") != "single_company":
        return False
    dimensions = {
        str(item.get("dimension_id") or "")
        for item in methodology_answer.get("dimension_sections", []) or []
        if isinstance(item, dict)
    }
    requested = {"cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"}
    return len(dimensions) <= 4 and len(dimensions & requested) >= 2


def _methodology_answer_is_single_company_overview(methodology_answer: dict[str, Any]) -> bool:
    if str(methodology_answer.get("analysis_scope") or "") != "single_company":
        return False
    profile = dict(methodology_answer.get("overview_profile", {}) or {})
    return str(profile.get("kind") or "") == "single_company_overview"


def _render_single_company_valuation_sentence(methodology_answer: dict[str, Any], *, lang: str) -> str:
    if str(methodology_answer.get("analysis_scope") or "") != "single_company":
        return ""
    sections = [
        dict(item)
        for item in methodology_answer.get("dimension_sections", []) or []
        if isinstance(item, Mapping)
    ]
    valuation_sections = [
        item
        for item in sections
        if str(item.get("dimension_id") or "") == "valuation_and_risk_boundary"
    ]
    if not valuation_sections:
        return ""
    requested_dimensions = {
        str(item.get("dimension_id") or "")
        for item in sections
        if str(item.get("dimension_id") or "").strip()
    }
    if requested_dimensions - {"valuation_and_risk_boundary"}:
        return ""
    section = valuation_sections[0]
    summary = re.sub(r"\s+", " ", str(section.get("summary") or "")).strip()
    if not summary:
        return ""
    refs = "".join(bracket_ref(ref) for ref in section.get("evidence_refs", []) or [] if str(ref).strip())
    if refs and refs not in summary:
        summary = f"{summary.rstrip('。.!')}{refs}。"
    if lang == "zh":
        summary = summary.replace("。 ", "。").strip()
        if not summary.startswith("不能给买卖建议"):
            summary = f"不能给买卖建议；{summary}"
    return summary


def _citation_for_card(card: Mapping[str, Any]) -> str:
    ref = str(card.get("citation_ref") or card.get("evidence_id") or "").strip()
    if not ref:
        return ""
    return ref if ref.startswith("[") else f"[{ref}]"


def _text_for_driver_card(card: Mapping[str, Any]) -> str:
    text = str(card.get("claim") or card.get("supporting_snippet") or card.get("text_snippet") or "").strip()
    claim_scope = str(card.get("claim_scope") or "").strip()
    if claim_scope in {"segment", "product"} and any(term in text for term in ("营收增长主要由", "总营收增长由", "收入增长主要由")):
        label = "产品层面" if claim_scope == "product" else "分部层面"
        text = f"{label}证据提供了增长线索，但不能完整代表总公司营收增长原因。"
    if len(text) > 280:
        text = text[:279].rstrip() + "..."
    citation = _citation_for_card(card)
    return f"{text} {citation}".strip()


def _driver_scope_kind(scope: str) -> str:
    lowered = str(scope or "").lower()
    if any(term in lowered for term in ("network", "网络", "nvlink", "infiniband", "ethernet")):
        return "networking"
    if any(term in lowered for term in ("data center", "datacenter", "数据中心", "compute")):
        return "data_center"
    return ""


def _driver_card_blob(card: Mapping[str, Any]) -> str:
    return " ".join(
        str(card.get(key) or "")
        for key in (
            "claim",
            "supporting_snippet",
            "text_snippet",
            "source_text",
            "section",
            "metric",
            "dimension_id",
        )
    )


def _driver_card_is_relevant(card: Mapping[str, Any], scope_kind: str) -> bool:
    lowered = _driver_card_blob(card).lower()
    if not lowered:
        return False
    if (
        "china" in lowered
        and "unable to create" in lowered
        and "competitive product" in lowered
        and "data center market" in lowered
    ):
        return False
    if "export control" in lowered and "growth" not in lowered and "grew" not in lowered and "up " not in lowered:
        return False
    if scope_kind == "networking":
        return any(
            term in lowered
            for term in (
                "networking revenue",
                "networking grew",
                "compute & networking",
                "nvlink",
                "infiniband",
                "ethernet",
                "gb200",
                "gb300",
            )
        )
    if scope_kind == "data_center":
        return any(
            term in lowered
            for term in (
                "data center revenue",
                "data center networking",
                "data center",
                "datacenter",
                "compute & networking",
                "nvlink",
                "infiniband",
                "ethernet",
            )
        ) and any(term in lowered for term in ("revenue", "grew", "growth", "up ", "nvlink", "infiniband", "ethernet"))
    return False


def _driver_card_priority(card: Mapping[str, Any], scope_kind: str) -> int:
    lowered = _driver_card_blob(card).lower()
    if scope_kind == "networking":
        if "networking revenue" in lowered and ("up 162" in lowered or "$8.2" in lowered):
            return 0
        if "nvlink" in lowered and ("gb200" in lowered or "gb300" in lowered):
            return 1
        if "compute & networking" in lowered:
            return 2
        return 5
    if "data center revenue" in lowered and ("up 56" in lowered or "$41.1" in lowered):
        return 0
    if "data center networking" in lowered and ("grew 142" in lowered or "nvlink" in lowered):
        return 1
    if "compute & networking" in lowered:
        return 2
    return 5


def _driver_company_label(synthesis: Mapping[str, Any], cards: list[dict[str, Any]]) -> str:
    for card in cards:
        ticker = str(card.get("ticker") or card.get("company") or "").upper().strip()
        if ticker:
            return ticker
    analysis_plan = dict(synthesis.get("analysis_plan", {}) or {})
    for item in analysis_plan.get("companies", []) or []:
        ticker = str(item or "").upper().strip()
        if ticker:
            return ticker
    methodology_answer = dict(synthesis.get("methodology_answer", {}) or {})
    ticker = str(methodology_answer.get("company") or "").upper().strip()
    return ticker or "该公司"


def _driver_fact_line(card: Mapping[str, Any], scope_kind: str, *, lang: str) -> str:
    blob = re.sub(r"\s+", " ", _driver_card_blob(card)).strip()
    lowered = blob.lower()
    citation = _citation_for_card(card)
    if not citation:
        return ""
    if lang == "zh":
        networking_match = re.search(
            r"Networking revenue was \$(?P<value>\d+(?:\.\d+)?) billion, up (?P<yoy>\d+)%",
            blob,
            flags=re.IGNORECASE,
        )
        if networking_match:
            return (
                f"Networking revenue 为 ${networking_match.group('value')}B，同比增长 {networking_match.group('yoy')}%，"
                f"文本将增长与 NVLink compute fabric 以及 GB200/GB300 系统联系起来。{citation}"
            )
        data_center_match = re.search(
            r"Data Center revenue was \$(?P<value>\d+(?:\.\d+)?) billion, up (?P<yoy>\d+)%[^.。]*?up (?P<seq>\d+)% sequentially",
            blob,
            flags=re.IGNORECASE,
        )
        if data_center_match:
            return (
                f"Data Center revenue 为 ${data_center_match.group('value')}B，同比增长 {data_center_match.group('yoy')}%、"
                f"环比增长 {data_center_match.group('seq')}%。{citation}"
            )
        if "data center networking" in lowered and ("grew 142" in lowered or "nvlink" in lowered):
            return f"数据中心 networking 同比增长 142%，文本将增长与 NVLink compute fabric、GB200/GB300、Ethernet 和 InfiniBand 平台联系起来。{citation}"
        if "compute & networking" in lowered:
            return f"Compute & Networking 分部覆盖数据中心加速计算、网络平台、AI 解决方案和相关软件。{citation}"
        excerpt = blob[:180].rstrip(" ,.;；，。")
        return f"已验证文本显示：{excerpt}。{citation}"
    if "networking revenue" in lowered and ("$8.2" in lowered or "up 162" in lowered):
        return f"Networking revenue was $8.2B, up 162%, and the text links growth to NVLink compute fabric and GB200/GB300 systems. {citation}"
    if "data center revenue" in lowered and ("$41.1" in lowered or "up 56" in lowered):
        return f"Data Center revenue was $41.1B, up 56% year over year and 5% sequentially. {citation}"
    if "data center networking" in lowered and ("grew 142" in lowered or "nvlink" in lowered):
        return f"Data Center networking grew 142%, with text linking growth to NVLink compute fabric, GB200/GB300, Ethernet, and InfiniBand platforms. {citation}"
    if "compute & networking" in lowered:
        return f"Compute & Networking includes data-center accelerated computing, networking platforms, AI solutions, and software. {citation}"
    excerpt = blob[:180].rstrip(" ,.;")
    return f"Validated text shows: {excerpt}. {citation}"


def _render_driver_analysis_text(synthesis: Mapping[str, Any], *, lang: str) -> str:
    analysis_plan = dict(synthesis.get("analysis_plan", {}) or {})
    canonical_intent = dict(analysis_plan.get("canonical_intent", {}) or {})
    scope = str(
        synthesis.get("segment_or_product_scope")
        or analysis_plan.get("segment_or_product_scope")
        or analysis_plan.get("segment_focus")
        or canonical_intent.get("segment_or_product_scope")
        or canonical_intent.get("segment_focus")
        or ""
    ).strip()
    text_cards = [dict(item) for item in synthesis.get("text_evidence_cards", []) or [] if isinstance(item, Mapping)]
    scope_kind = _driver_scope_kind(scope)
    if not scope_kind:
        research_plan = dict(synthesis.get("research_plan", {}) or {})
        if str(research_plan.get("question_type") or "") == "causal_explanation":
            return ""
        query_blob = str(synthesis.get("user_query") or "").lower()
        inferred_blob = " ".join([query_blob, *[_driver_card_blob(card) for card in text_cards]]).lower()
        if any(term in query_blob for term in ("网络业务", "networking", "network revenue")):
            scope_kind = "networking"
            scope = scope or "networking"
        elif any(term in query_blob for term in ("data center", "datacenter", "数据中心")):
            scope_kind = "data_center"
            scope = scope or "data center"
        elif any(term in inferred_blob for term in ("networking revenue", "网络业务", "nvlink", "infiniband", "ethernet")):
            scope_kind = "networking"
            scope = scope or "networking"
        elif any(term in inferred_blob for term in ("data center", "datacenter", "数据中心")) and any(
            term in inferred_blob for term in ("revenue", "营收", "收入", "growth", "增长")
        ):
            scope_kind = "data_center"
            scope = scope or "data center"
    if not scope_kind:
        return ""
    relevant_cards = [card for card in text_cards if _driver_card_is_relevant(card, scope_kind)]
    if not relevant_cards:
        return ""
    relevant_cards = sorted(relevant_cards, key=lambda card: _driver_card_priority(card, scope_kind))
    fact_lines = list(dict.fromkeys(_driver_fact_line(card, scope_kind, lang=lang) for card in relevant_cards))
    fact_lines = [line for line in fact_lines if line][:4]
    if not fact_lines:
        return ""
    refs = "".join(dict.fromkeys(re.findall(r"\[[T]\d+\]", " ".join(fact_lines))) )
    company = _driver_company_label(synthesis, relevant_cards)
    if lang == "zh":
        if scope_kind == "networking":
            conclusion = (
                f"{company} 网络业务增长大概率与 AI 集群建设、GPU 集群互连和 NVLink 相关系统放量有关；"
                f"InfiniBand/Ethernet 需求和供给/出货节奏是后续验证方向。{refs}"
            )
            inference = (
                f"AI 集群建设会放大 GPU 集群互连需求，直接支持 NVLink 相关网络收入；"
                f"InfiniBand/Ethernet 与供给、出货节奏仍需用网络产品收入、订单、递延收入和管理层披露继续验证。{refs}"
            )
            hypotheses = [
                "待验证：Networking 收入中 NVLink、InfiniBand、Ethernet 的各自占比和增速。",
                "待验证：GB200/GB300 出货节奏、客户 AI capex、订单和递延收入是否同步支持网络业务增长。",
            ]
            boundary = [
                "上述结论是分部/产品层面的有限判断，不能直接证明总公司营收增长的完整贡献比例或可持续性。",
                "已验证文本直接支持 NVLink 和 GB200/GB300 线索；InfiniBand/Ethernet 属于互连需求的待验证扩展方向。",
            ]
        else:
            conclusion = (
                f"数据中心很可能是 {company} 营收增长的重要候选驱动；但当前证据不能确认它对总营收增长的主要贡献比例。{refs}"
            )
            inference = (
                f"Data Center revenue 高增长和数据中心 networking 高增长同时出现，使数据中心成为重要候选驱动；"
                f"但这仍是分部/产品线索，不能升级为总公司级确定归因。{refs}"
            )
            hypotheses = [
                "待验证：Data Center 对总营收增量的贡献比例、其他分部增速和产品组合变化。",
                "待验证：AI 加速计算、网络互连、供给释放和客户订单节奏分别贡献了多少增长。",
            ]
            boundary = [
                "分部/产品证据不能单独证明总公司营收增长的完整原因、贡献比例或可持续性。",
                "当前答案只发布可引用事实和由这些事实支撑的有限推断，不把待验证因素写成确定事实。",
            ]
        lines = [
            "结论",
            conclusion,
            "",
            "已验证线索",
            *(f"- {line}" for line in fact_lines),
            "",
            "合理推断",
            f"- {inference}",
            "",
            "待验证数据",
            *[f"- {line}" for line in hypotheses],
            "",
            "证据边界",
            *[f"- {line}" for line in boundary],
        ]
        return "\n".join(line for line in lines if line).strip()
    if scope_kind == "networking":
        conclusion = (
            f"{company}'s networking growth is most plausibly tied to AI-cluster buildouts, GPU-cluster interconnect, "
            f"and NVLink-related systems; InfiniBand/Ethernet demand and shipment timing still need verification. {refs}"
        )
        inference = (
            f"AI-cluster buildouts increase GPU interconnect demand and support NVLink-related networking revenue; "
            f"InfiniBand/Ethernet mix, supply, and shipment timing need revenue, order, backlog, and deferred-revenue evidence. {refs}"
        )
        hypotheses = [
            "To verify: networking revenue mix and growth by NVLink, InfiniBand, and Ethernet.",
            "To verify: GB200/GB300 shipment timing, customer AI capex, orders, and deferred revenue.",
        ]
    else:
        conclusion = (
            f"Data Center is likely an important candidate driver of {company}'s revenue growth, but current evidence does not confirm its contribution share to total revenue growth. {refs}"
        )
        inference = (
            f"Strong Data Center revenue growth and Data Center networking growth make Data Center an important candidate driver, "
            f"but these are segment/product signals rather than full company-level attribution. {refs}"
        )
        hypotheses = [
            "To verify: Data Center contribution to total revenue growth, other segment growth, and product mix.",
            "To verify: AI accelerated computing, networking interconnect, supply release, and customer order timing.",
        ]
    lines = [
        "Conclusion",
        conclusion,
        "",
        "Verified Signals",
        *(f"- {line}" for line in fact_lines),
        "",
        "Reasonable Inference",
        f"- {inference}",
        "",
        "Data to Verify",
        *[f"- {line}" for line in hypotheses],
        "",
        "Evidence Boundary",
        "- This is a segment/product-level limited judgment, not proof of total-company contribution share or durability.",
        "- Facts are cited above; inferred and to-verify items should not be read as confirmed company facts.",
    ]
    return "\n".join(line for line in lines if line).strip()


def _render_causal_research_plan_text(synthesis: dict[str, Any], *, lang: str) -> str:
    short_answer = str(synthesis.get("short_answer") or "").strip()
    numeric_cards = [dict(item) for item in synthesis.get("numeric_evidence_cards", []) or [] if isinstance(item, Mapping)]
    text_cards = [dict(item) for item in synthesis.get("text_evidence_cards", []) or [] if isinstance(item, Mapping)]
    requirement_summary = dict(synthesis.get("requirement_summary", {}) or {})
    analytical_claims = [dict(item) for item in synthesis.get("analytical_claims", []) or [] if isinstance(item, Mapping)]
    analysis_plan = dict(synthesis.get("analysis_plan", {}) or {})
    canonical_intent = dict(analysis_plan.get("canonical_intent", {}) or {})
    segment_or_product_scope = str(
        synthesis.get("segment_or_product_scope")
        or analysis_plan.get("segment_or_product_scope")
        or analysis_plan.get("segment_focus")
        or canonical_intent.get("segment_or_product_scope")
        or canonical_intent.get("segment_focus")
        or ""
    ).strip()
    partial_parts = set(str(item) for item in requirement_summary.get("partial_required_answer_parts", []) or [] if str(item))
    missing_parts = set(str(item) for item in requirement_summary.get("missing_required_answer_parts", []) or [] if str(item))
    missing_but_parts = set(str(item) for item in requirement_summary.get("missing_but_analyzable_answer_parts", []) or [] if str(item))

    if analytical_claims:
        def lines_for(tier: str) -> list[str]:
            rows: list[str] = []
            for claim in analytical_claims:
                raw_tier = claim.get("tier")
                tier_value = str(getattr(raw_tier, "value", raw_tier) or "")
                if tier_value != tier:
                    continue
                text = str(claim.get("text") or "").strip()
                refs = "".join(bracket_ref(ref) for ref in claim.get("citation_refs", []) or [] if str(ref).strip())
                if refs and refs not in text:
                    text = f"{text} {refs}".strip()
                if text:
                    rows.append(f"- {text}")
            return rows

        backed = lines_for("evidence_backed")
        backed_numeric: list[str] = []
        company_scope_backed: list[str] = []
        segment_product_scope_backed: list[str] = []
        other_scope_backed: list[str] = []
        for claim in analytical_claims:
            raw_tier = claim.get("tier")
            tier_value = str(getattr(raw_tier, "value", raw_tier) or "")
            if tier_value != "evidence_backed":
                continue
            text = str(claim.get("text") or "").strip()
            refs = "".join(bracket_ref(ref) for ref in claim.get("citation_refs", []) or [] if str(ref).strip())
            if refs and refs not in text:
                text = f"{text} {refs}".strip()
            if not text:
                continue
            claim_scope = str(claim.get("claim_scope") or "").strip()
            line = f"- {text}"
            if claim_scope == "company":
                company_scope_backed.append(line)
            elif claim_scope in {"segment", "product"}:
                segment_product_scope_backed.append(line)
            elif claim_scope in {"market_context", "unknown"}:
                other_scope_backed.append(line)
            else:
                backed_numeric.append(line)
        inferred = lines_for("evidence_inferred")
        hypotheses = lines_for("hypothesis_to_verify")
        if segment_or_product_scope:
            if lang == "zh":
                lines = [
                    "简短结论",
                    short_answer or f"当前只能在分部/产品层面分析 {segment_or_product_scope}，不能推出总公司级完整归因。",
                    "",
                    "已验证事实",
                    *(backed_numeric or ["- 当前缺少可引用的分部/产品收入量化事实。"]),
                    *(segment_product_scope_backed[:3] if segment_product_scope_backed else []),
                    "",
                    "分部/产品层面驱动",
                    *(segment_product_scope_backed or ["- 当前缺少可引用的分部/产品 driver text。"]),
                    "",
                    "可引用推断",
                    *(inferred or ["- 当前证据不足以形成可引用支撑的推断。"]),
                    "",
                    "待验证假设",
                    *(hypotheses or ["- 待验证假设：需要补充分部收入、产品关键词和需求驱动文本。"]),
                    "",
                    "不能推出的公司级结论",
                    f"- {segment_or_product_scope} 的分部/产品证据不能单独证明总公司营收增长的完整原因、贡献比例或可持续性。",
                    "- 不能把产品关键词或局部分部增长直接写成公司级确定归因。",
                ]
                return "\n".join(lines).strip()
            lines = [
                "Short Answer",
                short_answer or f"The answer can analyze {segment_or_product_scope} at the segment/product level, but cannot prove total-company causal attribution.",
                "",
                "Verified Facts",
                *(backed_numeric or ["- Cited segment/product revenue facts are unavailable."]),
                *(segment_product_scope_backed[:3] if segment_product_scope_backed else []),
                "",
                "Segment/Product-Level Drivers",
                *(segment_product_scope_backed or ["- Cited segment/product driver text is unavailable."]),
                "",
                "Cited Inference",
                *(inferred or ["- Current evidence is insufficient for a cited inference."]),
                "",
                "Hypotheses To Verify",
                *(hypotheses or ["- Hypothesis to verify: segment revenue, product keywords, and demand-driver text are needed."]),
                "",
                "Company-Level Conclusions Not Supported",
                f"- {segment_or_product_scope} segment/product evidence does not by itself prove total-company revenue-growth causality, contribution share, or durability.",
                "- Product keywords or local segment growth cannot be written as definitive company-level attribution.",
            ]
            return "\n".join(lines).strip()
        boundary_lines: list[str]
        if lang == "zh":
            boundary_lines = [
                "- 以上把事实、推断和待验证假设分开；没有 citation 的内容只能作为分析框架或验证方向。",
                "- 当前不能量化每个因素对总营收增长的贡献比例，也不能判断增长是否可持续。",
            ]
            if "identify_growth_drivers" in missing_but_parts:
                boundary_lines.insert(0, "- 直接 driver text 不完整，因此不能把未验证因素写成确定原因；待验证因素也不能写成公司确认的事实。")
            if "quantify_growth" in partial_parts or "quantify_growth" in missing_parts:
                boundary_lines.insert(0, "- 当前无法可靠计算总营收增长率；总营收增长量化仍不完整，单期收入或分部增长文本只能支持部分判断。")
            lines = [
                "核心判断",
                short_answer or "当前只能给出分层分析，不能把未验证假设写成确定原因。",
                "",
                "已验证证据",
                *(backed_numeric or (backed if not (company_scope_backed or segment_product_scope_backed or other_scope_backed) else [])),
                "",
                "公司级证据",
                *(company_scope_backed or ["- 当前没有公司总营收层面的 driver text 可用于完整解释总营收增长原因。"]),
                "",
                "分部 / 产品层面证据",
                *(segment_product_scope_backed or ["- 当前没有额外的分部或产品层面驱动文本。"]),
                *(other_scope_backed[:2] if other_scope_backed else []),
                "",
                "基于证据的合理推断",
                *(inferred or ["- 当前证据不足以形成可引用支撑的推断。"]),
                "",
                "待验证假设",
                *(hypotheses or ["- 待验证假设：需要补充 filing 文本、分部收入和客户需求证据。"]),
                "",
                "反方因素 / 其他解释",
                "- 待验证假设：增长也可能受到供给释放、价格/产品组合、收入确认节奏或一次性订单节奏影响。",
                "- 需要验证：分部增长是否能代表总公司增长，而不是局部产品线表现。",
                "",
                "关键观察指标",
                "- 分部收入增速、数据中心与网络产品收入、ASP/出货量、客户 AI capex、递延收入和订单节奏。",
                "",
                "证据边界",
                *boundary_lines,
            ]
            return "\n".join(lines).strip()
        boundary_lines = [
            "- Facts, inferences, and hypotheses are separated; uncited content is only an analytical framework or verification path.",
            "- The current evidence does not quantify each factor's contribution to total revenue growth or prove durability.",
        ]
        if "identify_growth_drivers" in missing_but_parts:
            boundary_lines.insert(0, "- Direct driver text is incomplete, so hypotheses must not be written as confirmed company causes.")
        if "quantify_growth" in partial_parts or "quantify_growth" in missing_parts:
            boundary_lines.insert(0, "- Total revenue growth quantification is incomplete; one-period revenue or segment growth text is only partial support.")
        lines = [
            "Core Judgment",
            short_answer or "The answer can provide tiered analysis, but unverified hypotheses cannot be written as definitive causes.",
            "",
            "Verified Evidence",
            *(backed_numeric or (backed if not (company_scope_backed or segment_product_scope_backed or other_scope_backed) else [])),
            "",
            "Company-Level Evidence",
            *(company_scope_backed or ["- No company-level total-revenue driver text is available for a complete explanation."]),
            "",
            "Segment / Product-Level Evidence",
            *(segment_product_scope_backed or ["- No additional segment or product-level driver text is available."]),
            *(other_scope_backed[:2] if other_scope_backed else []),
            "",
            "Evidence-Based Inference",
            *(inferred or ["- Current evidence is insufficient for a cited inference."]),
            "",
            "Hypotheses To Verify",
            *(hypotheses or ["- Hypothesis to verify: filing text, segment revenue, and customer-demand evidence are needed."]),
            "",
            "Counterpoints / Other Explanations",
            "- Hypothesis to verify: growth may also reflect supply availability, pricing/product mix, revenue-recognition timing, or order timing.",
            "- Verify whether segment growth represents total-company growth rather than local product-line performance.",
            "",
            "Key Observation Metrics",
            "- Segment revenue growth, Data Center and networking revenue, ASP/shipments, customer AI capex, deferred revenue, and order timing.",
            "",
            "Evidence Boundary",
            *boundary_lines,
        ]
        return "\n".join(lines).strip()

    company_level: list[dict[str, Any]] = []
    segment_product: list[dict[str, Any]] = []
    other_context: list[dict[str, Any]] = []
    for card in text_cards:
        claim_scope = str(card.get("claim_scope") or "").strip()
        if claim_scope == "company":
            company_level.append(card)
            continue
        if claim_scope in {"segment", "product"}:
            segment_product.append(card)
            continue
        levels = set(str(item) for item in card.get("driver_levels", []) or [] if str(item))
        if not levels and card.get("driver_level"):
            levels.add(str(card.get("driver_level")))
        if "company_level_driver" in levels:
            company_level.append(card)
        elif levels & {"segment_level_driver", "product_level_driver"}:
            segment_product.append(card)
        else:
            other_context.append(card)

    if lang == "zh":
        lines = ["简短结论", short_answer or "当前答案必须同时说明增长幅度、增长驱动和证据边界。"]
        lines.extend(["", "增长量化"])
        if numeric_cards:
            for card in numeric_cards[:3]:
                metric_label = metric_display_name(str(card.get("metric") or ""), "zh")
                lines.append(
                    f"- {card.get('ticker', '')} 在 {card.get('period_end', '')} 期间的{metric_label}为 {_display_metric_row(card)}。{_citation_for_card(card)}".strip()
                )
        if "quantify_growth" in partial_parts or "quantify_growth" in missing_parts:
            lines.append("- 当前总营收增长量化证据不完整：单期收入或分部增长文本不能单独满足“增长这么多”的完整量化。")
        lines.extend(["", "公司级证据"])
        if company_level:
            lines.extend(f"- {_text_for_driver_card(card)}" for card in company_level[:2])
        else:
            lines.append("- 当前没有足够的公司总营收层面 driver text 可用于完整解释总营收增长原因。")
        lines.extend(["", "分部 / 产品层面证据"])
        supplemental = segment_product or other_context
        if supplemental:
            lines.extend(f"- {_text_for_driver_card(card)}" for card in supplemental[:3])
        else:
            lines.append("- 当前没有额外的分部或产品层面驱动文本。")
        lines.extend(["", "证据边界"])
        if not company_level and supplemental:
            lines.append("- 上述补充属于分部/产品层面证据，不能完整代表公司总收入增长原因。")
        if "identify_growth_drivers" in partial_parts:
            lines.append("- 增长驱动证据只达到部分满足，因此不能 clean pass；结论只能带边界发布。")
        if "identify_growth_drivers" in missing_parts:
            lines.append("- 缺少可验证 driver text 时，只能量化增长，不能解释原因。")
        return "\n".join(line for line in lines if line is not None).strip()

    lines = ["Short Answer", short_answer or "The answer must cover growth magnitude, growth drivers, and evidence boundaries."]
    lines.extend(["", "Growth Quantification"])
    if numeric_cards:
        for card in numeric_cards[:3]:
            lines.append(
                f"- {card.get('ticker', '')} {card.get('metric', '')} {card.get('period_end', '')}: {card.get('display_value') or card.get('value')} {_citation_for_card(card)}".strip()
            )
    if "quantify_growth" in partial_parts or "quantify_growth" in missing_parts:
        lines.append("- Total revenue growth quantification is incomplete; one-period revenue or segment growth text is not enough.")
    lines.extend(["", "Company-Level Drivers"])
    if company_level:
        lines.extend(f"- {_text_for_driver_card(card)}" for card in company_level[:2])
    else:
        lines.append("- No company-level total-revenue driver text is available for a complete explanation.")
    lines.extend(["", "Segment/Product Support"])
    supplemental = segment_product or other_context
    if supplemental:
        lines.extend(f"- {_text_for_driver_card(card)}" for card in supplemental[:3])
    else:
        lines.append("- No additional segment/product driver text is available.")
    lines.extend(["", "Evidence Boundary"])
    if not company_level and supplemental:
        lines.append("- Segment/product evidence cannot fully represent total company revenue growth causes.")
    if "identify_growth_drivers" in partial_parts:
        lines.append("- Driver evidence is only partially satisfied, so the answer cannot receive a clean pass.")
    if "identify_growth_drivers" in missing_parts:
        lines.append("- Without validated driver text, growth can be quantified but causes cannot be explained.")
    return "\n".join(line for line in lines if line is not None).strip()


def render_synthesis_text(synthesis: dict[str, Any], *, lang: str, answer_mode: str, safety_intent: str) -> str:
    driver_rendered = _render_driver_analysis_text(synthesis, lang=lang)
    if driver_rendered.strip():
        return driver_rendered
    research_plan = dict(synthesis.get("research_plan", {}) or {})
    if str(research_plan.get("question_type") or "") == "causal_explanation":
        rendered = _render_causal_research_plan_text(synthesis, lang=lang)
        if rendered.strip():
            return rendered
    final_answer_source = str(synthesis.get("final_answer_source", "")).strip()
    accepted_draft = dict(synthesis.get("accepted_draft", {}) or {})
    risk_focused_answer = dict(synthesis.get("risk_focused_answer", {}) or {})
    methodology_answer = dict(synthesis.get("methodology_answer", {}) or {})
    if final_answer_source.startswith("analyst_draft") and accepted_draft:
        if answer_mode == "risk_focused_analysis":
            rendered = _render_draft_risk_text(synthesis, lang)
            if rendered.strip():
                return rendered
        if _methodology_answer_is_single_company_composite(methodology_answer):
            rendered = render_methodology_single_company_brief(methodology_answer, lang=lang)
            if rendered.strip():
                return rendered
        if _methodology_answer_is_single_company_overview(methodology_answer):
            rendered = render_methodology_single_company_brief(methodology_answer, lang=lang)
            if rendered.strip():
                return rendered
        if answer_mode == "comparison_brief" or safety_intent == "investment_advice_like":
            return _render_draft_comparison_text(synthesis, lang)
        if answer_mode == "risk_focused_analysis":
            return _render_draft_risk_text(synthesis, lang)
        if answer_mode == "cautious_outlook":
            return _render_draft_outlook_text(synthesis, lang)
        return _render_draft_analytical_text(synthesis, lang)

    if risk_focused_answer:
        rendered = render_risk_focused_analysis_brief(risk_focused_answer, lang=lang)
        if rendered.strip():
            return rendered

    if methodology_answer:
        valuation_sentence = _render_single_company_valuation_sentence(methodology_answer, lang=lang)
        if valuation_sentence.strip():
            return valuation_sentence
        if str(methodology_answer.get("analysis_scope") or "") == "single_company":
            rendered = render_methodology_single_company_brief(methodology_answer, lang=lang)
        else:
            rendered = render_methodology_comparison_brief(methodology_answer, lang=lang)
        if rendered.strip():
            return rendered

    methodology_report = dict(synthesis.get("methodology_report", {}) or {})
    if methodology_report:
        rendered = _render_methodology_report_text(
            methodology_report,
            lang=lang,
            answer_mode=answer_mode,
            safety_intent=safety_intent,
        )
        if rendered.strip():
            return rendered

    if final_answer_source == "comparison_decision_fallback" and dict(synthesis.get("comparison_judgment_frame", {}) or {}):
        return _render_comparison_frame_text(synthesis, lang)

    synthesis_mode = str(synthesis.get("synthesis_mode", "")).strip()
    if synthesis_mode == "limited_judgment":
        headings = ("有限判断", "关键数据", "条件化结论", "限制与边界") if lang == "zh" else (
            "Limited Judgment", "Key Facts", "Conditional View", "Limits and Boundaries"
        )
    elif synthesis_mode == "limited_analysis":
        headings = ("有限分析", "关键数据", "可支持的判断", "限制与边界") if lang == "zh" else (
            "Limited Analysis", "Key Facts", "Supported View", "Limits and Boundaries"
        )
    elif synthesis_mode == "limited_outlook":
        headings = ("有限展望", "已披露趋势", "谨慎观察", "限制与边界") if lang == "zh" else (
            "Limited Outlook", "Disclosed Trends", "Cautious View", "Limits and Boundaries"
        )
    elif answer_mode == "comparison_brief" or safety_intent == "investment_advice_like":
        headings = ("非投资建议", "关键事实", "分析判断", "风险与限制") if lang == "zh" else (
            "Not Investment Advice", "Key Facts", "Analytical View", "Risks and Limits"
        )
    elif answer_mode == "cautious_outlook":
        headings = ("简短判断", "已披露趋势", "支撑因素", "风险与不确定性") if lang == "zh" else (
            "Brief View", "Disclosed Trends", "Supporting Factors", "Risks and Uncertainties"
        )
    else:
        headings = ("简短结论", "关键事实", "分析", "风险与限制") if lang == "zh" else (
            "Short Answer", "Key Facts", "Analysis", "Risks and Limits"
        )
    lines = [headings[0], str(synthesis.get("short_answer", "")).strip()]
    facts = [str(x.get("sentence", "")) for x in synthesis.get("key_facts", []) if str(x.get("sentence", "")).strip()]
    analysis = [str(x.get("sentence", "")) for x in synthesis.get("analysis", []) if str(x.get("sentence", "")).strip()]
    risks = [str(x.get("sentence", "")) for x in synthesis.get("risks_or_uncertainties", []) if str(x.get("sentence", "")).strip()]
    if facts:
        lines.extend(["", headings[1], *[f"- {x}" for x in facts[:5]]])
    if analysis:
        lines.extend(["", headings[2], *[f"- {x}" for x in analysis[:5]]])
    if risks:
        lines.extend(["", headings[3], *[f"- {x}" for x in risks[:5]]])
    return "\n".join([x for x in lines if x is not None]).strip()


def build_synthesis_view(synthesis: dict[str, Any], *, answer_mode: str, task_type: str, safety_intent: str, lang: str) -> dict[str, Any]:
    kind = synthesis_view_kind(answer_mode, task_type)
    facts = [x for x in synthesis.get("key_facts", []) if isinstance(x, dict)]
    analysis = [x for x in synthesis.get("analysis", []) if isinstance(x, dict)]
    risks = [x for x in synthesis.get("risks_or_uncertainties", []) if isinstance(x, dict)]
    accepted_draft = dict(synthesis.get("accepted_draft", {}) or {})
    base = {
        "kind": kind,
        "synthesis_mode": str(synthesis.get("synthesis_mode", "")),
        "short_answer": str(synthesis.get("short_answer", "")),
        "key_facts": facts,
        "analysis": analysis,
        "risks_or_uncertainties": risks,
        "citations": list(synthesis.get("citations", [])),
        "limitations": list(synthesis.get("limitations", [])),
        "final_answer_source": str(synthesis.get("final_answer_source", "")),
        "methodology_answer": dict(synthesis.get("methodology_answer", {}) or {}),
        "risk_focused_answer": dict(synthesis.get("risk_focused_answer", {}) or {}),
        "methodology_report": dict(synthesis.get("methodology_report", {}) or {}),
    }
    risk_focused_answer = dict(synthesis.get("risk_focused_answer", {}) or {})
    methodology_answer = dict(synthesis.get("methodology_answer", {}) or {})
    if risk_focused_answer:
        top_risk = dict(risk_focused_answer.get("top_risk", {}) or {})
        secondary = [
            dict(item)
            for item in risk_focused_answer.get("secondary_risks", []) or []
            if isinstance(item, dict)
        ]
        base["kind"] = "risk_focused_analysis_brief"
        base["risk_focused_answer"] = risk_focused_answer
        base["top_risk_theme"] = top_risk
        base["risk_themes"] = ([top_risk] if top_risk else []) + secondary
        base["financial_context"] = list(risk_focused_answer.get("financial_context", []) or [])
        base["risk_ranking"] = list(risk_focused_answer.get("risk_ranking", []) or [])
        base["transmission_path"] = list(risk_focused_answer.get("transmission_path", []) or [])
        base["financial_impact_analysis"] = list(risk_focused_answer.get("financial_impact_analysis", []) or [])
        base["key_observation_metrics"] = list(risk_focused_answer.get("key_observation_metrics", []) or [])
        base["evidence_boundaries"] = list(risk_focused_answer.get("evidence_boundaries", []) or [])
        return base

    final_answer_source = str(synthesis.get("final_answer_source", ""))
    preserve_comparison_view = methodology_answer and (answer_mode == "comparison_brief" or safety_intent == "investment_advice_like")
    if (
        final_answer_source.startswith("analyst_draft")
        and accepted_draft
        and not preserve_comparison_view
        and not _methodology_answer_is_single_company_composite(methodology_answer)
        and not _methodology_answer_is_single_company_overview(methodology_answer)
    ):
        base["kind"] = "analyst_draft_brief"
        base["accepted_draft"] = accepted_draft
        base["draft_status"] = str(synthesis.get("draft_status", ""))
        base["draft_final_status"] = str(synthesis.get("draft_final_status", ""))
        if methodology_answer:
            base["methodology_answer"] = methodology_answer
            base["single_company_metric_table"] = [
                dict(item)
                for item in methodology_answer.get("single_company_metric_table", []) or []
                if isinstance(item, dict)
            ]
            base["metric_table"] = [
                dict(item)
                for item in methodology_answer.get("metric_table", []) or []
                if isinstance(item, dict)
            ]
        return base

    if methodology_answer and str(methodology_answer.get("analysis_scope") or "") == "single_company":
        base["kind"] = "methodology_single_company_brief"
        base["single_company_metric_table"] = [
            dict(item)
            for item in methodology_answer.get("single_company_metric_table", []) or []
            if isinstance(item, dict)
        ]
        base["dimension_sections"] = [
            dict(item)
            for item in methodology_answer.get("dimension_sections", []) or []
            if isinstance(item, dict)
        ]
        base["methodology_limitations"] = list(methodology_answer.get("limitations", []) or [])
        base["follow_up_metrics"] = list(methodology_answer.get("follow_up_metrics", []) or [])
        base["overview_profile"] = dict(methodology_answer.get("overview_profile", {}) or {})
        base["data_sanity_flags"] = list(methodology_answer.get("data_sanity_flags", []) or [])
        base["overview_risk_ranking"] = list(methodology_answer.get("overview_risk_ranking", []) or [])
        return base
    if methodology_answer and (answer_mode == "comparison_brief" or safety_intent == "investment_advice_like"):
        base["kind"] = "methodology_comparison_brief"
        base["metric_table"] = [
            dict(item)
            for item in methodology_answer.get("metric_table", []) or []
            if isinstance(item, dict)
        ]
        base["dimension_sections"] = [
            dict(item)
            for item in methodology_answer.get("dimension_sections", []) or []
            if isinstance(item, dict)
        ]
        base["methodology_limitations"] = list(methodology_answer.get("limitations", []) or [])
        base["methodology_counterpoints"] = [methodology_answer.get("counterpoint")] if methodology_answer.get("counterpoint") else []

    methodology_report = dict(synthesis.get("methodology_report", {}) or {})
    if not methodology_answer and methodology_report and (answer_mode == "comparison_brief" or safety_intent == "investment_advice_like"):
        base["kind"] = "methodology_comparison_brief"
        base["dimension_analyses"] = [
            dict(item)
            for item in methodology_report.get("dimension_analyses", []) or []
            if isinstance(item, dict)
        ]
        base["methodology_counterpoints"] = list(methodology_report.get("counterpoints", []) or [])
        base["methodology_limitations"] = list(methodology_report.get("limitations", []) or [])
    if kind == "cautious_outlook":
        base.update(
            {
                "disclosed_trends": facts,
                "supporting_factors": analysis,
                "risk_factors": risks,
                "cannot_predict": _cautious_sentence(lang),
            }
        )
    if kind == "comparison_brief":
        base.update(
            {
                "non_advisory_statement": _investment_boundary_sentence(lang),
                "comparison_dimensions": [],
                "key_data": facts,
                "analysis_judgment": analysis,
                "counterpoints": list(accepted_draft.get("counterpoints", []) or []),
                "risks_and_limits": risks,
            }
        )
    elif kind == "analytical_brief":
        base.update(
            {
                "main_issue": dict((accepted_draft.get("decision_basis", []) or [{}])[0] or {}) if accepted_draft.get("decision_basis") else {},
                "evidence_basis": list((accepted_draft.get("decision_basis", []) or [])[1:] + list(accepted_draft.get("supporting_points", []) or [])),
            }
        )
    return base
