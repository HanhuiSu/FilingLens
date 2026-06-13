# ruff: noqa: F401,F403,F405
"""Answer text, output payload, view, and market-reaction rendering."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.constants import *
from src.agent.analysis_framework import summarize_selected_analysis_framework
from src.agent.citations import _build_citation_refs, _dedupe_document_citations, _doc_key
from src.agent.evidence import (
    _build_structured_conclusion,
    _collect_event_rows,
    _collect_financial_rows,
    _collect_price_rows,
    _dedupe_rows,
    _format_usd_value,
    _latest_row,
    _metric_label,
    _ordered_unique_tickers,
    _period_label,
    _raw_numeric_text,
    _rows_to_structured_sources,
    _select_comparison_evidence_rows,
    _to_float,
)
from src.agent.evidence_sufficiency import normalize_dimension_status_contract, summarize_evidence_requirements
from src.agent.llm import _get_llm
from src.agent.metric_availability import normalize_metric_name
from src.agent.metric_display import format_metric_value, period_category
from src.agent.output_language import display_theme, repair_language_leakage, risk_theme_key
from src.agent.red_flags import user_visible_red_flags
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

def _clarification_message(lang: str, reason: str | None) -> str:
    reason = reason or "time_period_ambiguous"
    if lang == "zh":
        return (
            "时间口径不够明确，暂时无法给出可靠结论。"
            f"（原因：{reason}）请明确例如：`2025Q1`、`2024财年`、`2025自然年` 或 `最近4个季度`。"
        )
    return (
        "Time period is ambiguous, so I cannot provide a reliable answer right now. "
        f"(reason: {reason}) Please specify a period like `2025Q1`, `FY2024`, "
        "`calendar year 2025`, or `last 4 quarters`."
    )

def _comparison_basis_line(lang: str, basis_label: str) -> str:
    if not basis_label:
        return ""
    if lang == "zh":
        return f"比较口径：{basis_label}"
    return f"Comparison Basis: {basis_label}"

def _annual_year_basis_line(
    lang: str,
    task_type: str,
    period_query: dict[str, Any],
    resolved_period_context: dict[str, Any],
) -> str:
    if task_type == "company_comparison":
        return ""
    period_type = str(period_query.get("period_type") or "")
    target_period_type = str(resolved_period_context.get("target_period_type") or "")
    is_annual_context = period_type == "annual" or (period_type in {"latest", "trailing"} and target_period_type == "annual")
    if not is_annual_context:
        return ""
    year_basis = str(period_query.get("year_basis") or "fiscal")
    if lang == "zh":
        if year_basis == "calendar":
            return "时间口径：自然年（calendar year）"
        return "时间口径：财年（fiscal year）"
    if year_basis == "calendar":
        return "Time Basis: calendar year"
    return "Time Basis: fiscal year"

def _clean_answer_text(text: str, lang: str) -> str:
    cleaned = (text or "").replace("\r\n", "\n")
    cleaned = re.sub(r"```[\w-]*", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"__(.*?)__", r"\1", cleaned)
    cleaned = re.sub(r"^\s*[-*_]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if lang == "zh":
        cleaned = re.sub(r"\bUSD\b", "美元", cleaned, flags=re.IGNORECASE)

    out_lines: list[str] = []
    seen_special: set[str] = set()
    for line in cleaned.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(?:\[[NT]\d+\])+", stripped):
            continue
        lowered = stripped.lower()
        special_key = ""
        if "非投资建议" in stripped or "不构成投资建议" in stripped or "investment advice" in lowered:
            special_key = "investment_boundary"
        elif "缺少可验证文本证据" in stripped or "verifiable filing-text evidence is limited" in lowered:
            special_key = "limited_text_evidence"
        elif "缺少估值证据" in stripped or "valuation evidence is missing" in lowered:
            special_key = "valuation_missing"
        elif stripped.startswith("比较口径：") or lowered.startswith("comparison basis:"):
            special_key = f"comparison_basis:{lowered}"
        if special_key:
            if special_key in seen_special:
                continue
            seen_special.add(special_key)
        out_lines.append(line)
    cleaned = "\n".join(out_lines)
    if lang == "zh":
        top_match = re.search(r"最高优先级：([^，。；;\n]+)", cleaned)
        if top_match:
            top_name = top_match.group(1).strip()
            cleaned = re.sub(
                r"最(?:重要|突出)风险是[^。；;\n]+",
                f"最突出的风险是{top_name}",
                cleaned,
                count=1,
            )
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned.strip()


_INTERNAL_USER_TEXT_PATTERN = re.compile(
    r"\bREQ-[A-Z0-9_-]+\b|\b[a-z]+(?:_[a-z0-9]+){1,}\b"
)
_INTERNAL_ANSWER_TEXT_PATTERN = re.compile(
    r"\bREQ-[A-Z0-9_-]+\b|\b(?:dependency_numeric_requirement_missing|numeric_only_comparison|"
    r"dependency_[A-Za-z0-9_]+|numeric_only_[A-Za-z0-9_]+|dependency_metric_id|"
    r"profitability_quality|absolute_profitability|no_clear_preference|text_evidence_missing|"
    r"text_evidence_partial|required_evidence_missing|limited_judgment|balanced_comparison|"
    r"comparison_decision_fallback|unsupported_claims_present|requirement_partial|"
    r"requirement_missing|fallback|suggested_fix|validator_result|ContractResult)\b|"
    r"Rewrite only|contract repair|primary generation|candidate layer|repair instruction|route="
    ,
    re.IGNORECASE,
)


def _ratio_display_value(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return str(value)
    return f"{num * 100:.2f}%"


def _display_numeric_value(row: dict[str, Any], lang: str) -> str:
    metric = str(row.get("metric", ""))
    display = str(row.get("display_value") or "").strip()
    if row.get("value") is not None:
        canonical = normalize_metric_name(metric)
        if display:
            if canonical in {"pe_ratio", "ps_ratio"} and "%" in display:
                return format_metric_value(metric, row.get("value"), unit=str(row.get("unit", "")))
            if canonical in {"market_cap", "net_debt"} and "$" not in display:
                return format_metric_value(metric, row.get("value"), unit=str(row.get("unit", "")))
            return display
        formatted = format_metric_value(metric, row.get("value"), unit=str(row.get("unit", "")))
        if formatted != "N/A":
            return formatted
    if display:
        return display
    return _raw_numeric_text(row.get("value"))


def _human_limited_text_message(lang: str) -> str:
    if lang == "zh":
        return "当前缺少足够的可验证文本证据，因此只能做有限财务视角判断。"
    return "Current verifiable filing-text evidence is limited, so this can only be a limited financial comparison."


def _human_calc_unavailable_message(lang: str) -> str:
    if lang == "zh":
        return "部分派生指标不可用，因此比较主要基于收入、净利润和可验证披露证据。"
    return "Some derived metrics are unavailable, so the comparison mainly relies on revenue, net income, and verifiable filing evidence."


def _human_imbalanced_text_message(lang: str) -> str:
    if lang == "zh":
        return "两家公司可验证文本证据覆盖不均衡，因此风险比较需要谨慎解读。"
    return "Verifiable text evidence coverage is uneven across the compared companies, so risk comparison should be read cautiously."


def _human_generic_limitation_message(lang: str) -> str:
    if lang == "zh":
        return "部分证据或派生指标不可用，因此结论需要谨慎解读。"
    return "Some evidence or derived metrics are unavailable, so the conclusion should be read cautiously."


def _sanitize_limitation_message(item: dict[str, Any], lang: str, task_type: str) -> str:
    code = str(item.get("code", "")).strip()
    message = str(item.get("message", "") or "").strip()
    requirement_id = str(item.get("requirement_id", "") or "").upper()
    failure_reason = str(item.get("failure_reason", "") or "")
    internal_blob = " ".join([code, message, requirement_id, failure_reason])
    is_comparison = task_type == "company_comparison"

    if code == "investment_advice_boundary":
        return (
            "以下内容仅是基于已验证证据的基本面比较，不构成投资建议。"
            if lang == "zh"
            else "This is a fundamental comparison based only on validated evidence, not investment advice."
        )
    if "yfinance" in internal_blob.lower():
        return (
            "部分结构化财务数据来自 yfinance，可信度为 medium。"
            if lang == "zh"
            else "Some structured financial data comes from yfinance, with medium confidence."
        )
    if "required evidence is missing" in internal_blob.lower() or "required_evidence_missing" in internal_blob:
        return _human_generic_limitation_message(lang)
    if "valuation_evidence_missing" in internal_blob or "valuation_and_risk_boundary" in internal_blob:
        return (
            "当前缺少估值证据，因此不能判断哪家公司价格更便宜。"
            if lang == "zh"
            else "Valuation evidence is missing, so I cannot judge which company is cheaper."
        )
    if "cash_flow_quality" in internal_blob or "balance_sheet_and_capital_intensity" in internal_blob:
        return (
            "当前缺少完整现金流和资产负债表证据，因此不能形成完整长期投资结论。"
            if lang == "zh"
            else "Full cash-flow and balance-sheet evidence is missing, so this cannot support a complete long-term investment conclusion."
        )
    if code == "unsupported_claims_present":
        return (
            "部分未通过证据校验的表述已被移除。"
            if lang == "zh"
            else "Some statements that did not pass evidence validation were removed."
        )
    if code in {"numeric_only_comparison", "text_evidence_missing"}:
        return _human_limited_text_message(lang)
    if code == "text_evidence_partial":
        return (
            "当前只有部分可验证文本证据，因此只能做有限分析。"
            if lang == "zh"
            else "Only partial verifiable filing-text evidence is available, so the analysis is limited."
        )
    if code in {"imbalanced_company_evidence", "text_unbalanced"}:
        return _human_imbalanced_text_message(lang)
    if (
        "dependency_numeric_requirement_missing" in internal_blob
        or "OPERATING_MARGIN" in internal_blob
        or "NET_MARGIN" in internal_blob
        or (code.startswith("requirement_") and "CALC" in requirement_id)
    ):
        return _human_calc_unavailable_message(lang)
    if code.startswith("requirement_") and ("TEXT" in requirement_id or "text" in failure_reason.lower()):
        return _human_limited_text_message(lang) if is_comparison else (
            "当前缺少足够的可验证文本证据，因此只能做有限分析。"
            if lang == "zh"
            else "Current verifiable filing-text evidence is limited, so the analysis is limited."
        )
    if _INTERNAL_USER_TEXT_PATTERN.search(message):
        return _human_generic_limitation_message(lang)
    return message or _human_generic_limitation_message(lang)


def sanitize_user_facing_limitations(
    limitations: list[dict[str, Any]],
    lang: str,
    *,
    task_type: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in limitations or []:
        if not isinstance(item, dict):
            continue
        cleaned = dict(item)
        cleaned["message"] = _sanitize_limitation_message(cleaned, lang, task_type)
        key = (str(cleaned.get("code", "")), str(cleaned.get("message", "")))
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _general_user_facing_answer_cleanup(text: str, lang: str) -> str:
    cleaned = str(text or "")
    if lang == "zh":
        heading_replacements = {
            "Company Overview": "结论",
            "One-Sentence Positioning": "结论",
            "Business And Revenue": "收入和盈利",
            "Profitability And Cash Flow": "现金流与估值",
            "Risk": "主要风险",
            "Valuation Boundary": "现金流与估值",
            "Evidence Boundary": "证据边界",
        }
        for old, new in heading_replacements.items():
            cleaned = re.sub(rf"(?m)^(\s*#+\s*)?{re.escape(old)}\s*$", new, cleaned)
        raw_metric_replacements = {
            "net_income": "净利润",
            "market_cap": "市值",
            "pe_ratio": "P/E",
            "ps_ratio": "P/S",
            "fcf_yield": "FCF yield",
            "operating_cash_flow": "经营现金流",
            "free_cash_flow": "自由现金流",
            "total_debt": "总债务",
            "adjusted_close": "股价",
        }
    else:
        raw_metric_replacements = {
            "net_income": "net income",
            "market_cap": "market cap",
            "pe_ratio": "P/E",
            "ps_ratio": "P/S",
            "fcf_yield": "FCF yield",
            "operating_cash_flow": "operating cash flow",
            "free_cash_flow": "free cash flow",
            "total_debt": "total debt",
            "adjusted_close": "share price",
        }
    for old, new in raw_metric_replacements.items():
        cleaned = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, cleaned)
    if lang == "zh":
        cleaned = re.sub(r"\bcaveat\b", "口径提示", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\bmedium\b", "中等", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"口径/异常值\s+口径提示", "口径/异常值提示", cleaned)
        cleaned = re.sub(r"口径\s+口径提示", "口径提示", cleaned)
        cleaned = re.sub(
            r"(?m)^(结论|业务定位|收入和盈利|现金流与估值|主要风险|已验证线索|合理推断|待验证数据|证据边界)((?:\[[NT]\d+\])+)$",
            r"\1",
            cleaned,
        )

    label_patterns = (
        r"有限判断\s*/\s*基于业务结构推断[:：]\s*",
        r"Limited judgment\s*/\s*business-structure inference[:：]\s*",
    )
    for pattern in label_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    if lang == "zh":
        cleaned = re.sub(r"^\s*证据不足以强行判断[，,]?", "结论\n当前只能给出有限判断，", cleaned)

    lines: list[str] = []
    missing_streak = 0
    previous_heading = ""
    heading_like = {
        "结论",
        "已验证事实",
        "已验证线索",
        "合理推断",
        "待验证假设",
        "待验证数据",
        "证据边界",
        "业务定位",
        "收入和盈利",
        "现金流与估值",
        "主要风险",
    }
    for line in cleaned.splitlines():
        stripped = line.strip()
        if stripped in heading_like:
            if previous_heading == stripped:
                continue
            previous_heading = stripped
        elif stripped:
            previous_heading = ""
        if "缺少可验证数据" in stripped:
            missing_streak += 1
            if missing_streak > 1:
                continue
        else:
            missing_streak = 0
        lines.append(line)
    cleaned = "\n".join(lines)

    advice_pattern = (
        r"不构成投资建议|不是投资建议|不能给买卖建议|不能据此给出买卖建议|不能形成买卖建议|"
        r"not investment advice|buy/sell advice|trading recommendation"
    )
    pieces = re.split(r"(?<=[。.!?])(\s+|\n+)", cleaned)
    rebuilt: list[str] = []
    seen_advice = False
    for i in range(0, len(pieces), 2):
        piece = pieces[i]
        sep = pieces[i + 1] if i + 1 < len(pieces) else ""
        if not piece:
            if sep:
                rebuilt.append(sep)
            continue
        has_advice = re.search(advice_pattern, piece, flags=re.IGNORECASE)
        if has_advice and seen_advice:
            continue
        if has_advice:
            seen_advice = True
        rebuilt.append(piece)
        rebuilt.append(sep)
    cleaned = "".join(rebuilt)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def sanitize_user_facing_answer_text(text: str, lang: str) -> str:
    if lang == "en":
        text = repair_language_leakage(text, lang)
    if not _INTERNAL_ANSWER_TEXT_PATTERN.search(text or ""):
        return _general_user_facing_answer_cleanup(text, lang)
    replacement = _human_generic_limitation_message(lang)
    cleaned = re.sub(r"Evidence requirement REQ-[A-Z0-9_-]+ was [^.。]*[.。]?", replacement, text or "")
    cleaned = re.sub(
        r"[^。.!?\n]*(?:Rewrite only|contract repair|primary generation|candidate layer|suggested_fix|repair instruction|validator_result|ContractResult|route=)[^。.!?\n]*[。.!?]?",
        replacement,
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\bREQ-[A-Z0-9_-]+\b", "", cleaned)
    cleaned = _INTERNAL_ANSWER_TEXT_PATTERN.sub(replacement, cleaned)
    cleaned = re.sub(rf"(?:{re.escape(replacement)}\s*){{2,}}", replacement, cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return _general_user_facing_answer_cleanup(cleaned, lang)


def _methodology_inline_refs(refs: list[str]) -> str:
    return "".join(f"[{ref}]" for ref in refs if str(ref).strip())


def _normalize_methodology_punctuation(text: str, lang: str) -> str:
    cleaned = str(text or "")
    if lang == "zh":
        cleaned = re.sub(r"。+\s*；+", "。", cleaned)
        cleaned = re.sub(r"；{2,}", "；", cleaned)
        cleaned = re.sub(r"([。！？])((?:\[[NT]\d+\])+)[；;]", r"\1\2 ", cleaned)
        cleaned = re.sub(
            r"((?:\[[NT]\d+\])+)\s+(?=(?:capex/revenue|FCF margin|net margin|gross margin|operating margin)\b)",
            r"\1 此外，",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+([，。；：])", r"\1", cleaned)
        cleaned = re.sub(r"([，；：])\s+", r"\1", cleaned)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _clean_methodology_visible_text(text: str, lang: str) -> str:
    cleaned = str(text or "").strip()
    replacements = {
        "Required evidence is missing": (
            "当前缺少必要证据" if lang == "zh" else "Required evidence is unavailable"
        ),
        "required_evidence_missing": (
            "当前缺少必要证据" if lang == "zh" else "required evidence is unavailable"
        ),
        "dependency_numeric_requirement_missing": (
            "部分派生指标不可用" if lang == "zh" else "some derived metrics are unavailable"
        ),
        "numeric_only_comparison": (
            "有限财务视角判断" if lang == "zh" else "limited financial comparison"
        ),
        "numeric_only": (
            "有限数字证据" if lang == "zh" else "limited numeric evidence"
        ),
        "yfinance fallback": "yfinance",
        "fallback": (
            "备用数据来源" if lang == "zh" else "alternate data source"
        ),
    }
    for old, new in replacements.items():
        cleaned = re.sub(re.escape(old), new, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bREQ-[A-Z0-9_-]+\b", "", cleaned)
    cleaned = re.sub(r"\bdependency_[A-Za-z0-9_]+\b", "部分依赖数据不可用" if lang == "zh" else "some dependency data is unavailable", cleaned)
    cleaned = re.sub(r"\bnumeric_only_[A-Za-z0-9_]+\b", "有限数字证据" if lang == "zh" else "limited numeric evidence", cleaned)
    cleaned = sanitize_user_facing_answer_text(cleaned, lang)
    return _normalize_methodology_punctuation(cleaned, lang)


def _section_text_with_refs(summary: str, refs: list[str], lang: str) -> str:
    text = _clean_methodology_visible_text(summary, lang)
    if not text:
        return ""
    clean_refs = [str(ref) for ref in refs if str(ref).strip()]
    if not clean_refs:
        return text
    if any(f"[{ref}]" in text for ref in clean_refs):
        return text
    return _normalize_methodology_punctuation(f"{text} {_methodology_inline_refs(clean_refs)}".strip(), lang)


_METHODOLOGY_COMPARISON_DIMENSION_ORDER = [
    "revenue_quality",
    "profitability_quality",
    "moat_and_competitive_risk",
    "valuation_and_risk_boundary",
]


def _methodology_limitation_key(text: str) -> str:
    normalized = str(text or "").lower()
    if "估值证据" in normalized or "valuation evidence" in normalized or ("估值" in normalized and ("缺少" in normalized or "不足" in normalized)):
        return "valuation_missing"
    if "yfinance" in normalized:
        return "yfinance_provider"
    if "投资建议" in normalized or "不是投资建议" in normalized or "不构成" in normalized and "建议" in normalized or "investment advice" in normalized:
        return "investment_boundary"
    if "净利率" in normalized or "毛利率" in normalized or "营业利润率" in normalized or "margin" in normalized:
        return "profitability_scope"
    if "现金流" in normalized or "资产负债表" in normalized or "cash-flow" in normalized or "balance-sheet" in normalized:
        return "cash_balance_scope"
    normalized = re.sub(r"\[[A-Z]\d+\]", "", normalized)
    normalized = re.sub(r"\bREQ-[A-Z0-9_-]+\b", "", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bdependency_[a-z0-9_]+\b", "", normalized)
    normalized = re.sub(r"\bnumeric_only_[a-z0-9_]+\b", "", normalized)
    normalized = re.sub(r"[\s。；;，,：:、.-]+", "", normalized)
    return normalized or str(text)


def _dedupe_methodology_limitations(
    items: list[str],
    *,
    lang: str,
    exclude_keys: set[str] | None = None,
    limit: int = 5,
) -> list[str]:
    out: list[str] = []
    seen: set[str] = set(exclude_keys or set())
    for item in items:
        clean = _clean_methodology_visible_text(str(item), lang)
        if not clean:
            continue
        key = _methodology_limitation_key(clean)
        if key in seen:
            continue
        seen.add(key)
        out.append(clean)
        if len(out) >= limit:
            break
    return out


def _render_methodology_metric_table(metric_table: list[dict[str, Any]], *, lang: str) -> list[str]:
    rows = [dict(item) for item in metric_table or [] if isinstance(item, dict)]
    if not rows:
        return []
    company_order: list[str] = []
    for row in rows:
        company_values = row.get("company_values", {})
        if not isinstance(company_values, dict):
            continue
        for company in company_values.keys():
            company_text = str(company).strip()
            if company_text and company_text not in company_order:
                company_order.append(company_text)
    if len(company_order) < 2:
        return []
    company_order = company_order[:2]
    heading = "核心指标对比" if lang == "zh" else "Core Metric Comparison"
    judgment_label = "当前判断" if lang == "zh" else "Current View"
    metric_label = "指标" if lang == "zh" else "Metric"
    lines = [
        heading,
        f"| {metric_label} | {' | '.join(company_order)} | {judgment_label} |",
        f"| --- | {' | '.join('---' for _ in company_order)} | --- |",
    ]
    missing = "缺少可验证数据" if lang == "zh" else "verified data unavailable"
    for row in rows:
        label = _clean_methodology_visible_text(str(row.get("label") or row.get("metric_id") or ""), lang)
        company_values = row.get("company_values", {})
        if not isinstance(company_values, dict):
            company_values = {}
        values = [
            _clean_methodology_visible_text(str(company_values.get(company) or missing), lang)
            for company in company_order
        ]
        values = [
            missing
            if lang == "zh" and value.startswith(f"{label} {missing}")
            else value
            for value in values
        ]
        judgment = _clean_methodology_visible_text(str(row.get("judgment") or ""), lang)
        lines.append(f"| {label} | {' | '.join(values)} | {judgment} |")
    return lines


def _render_single_company_metric_table(metric_table: list[dict[str, Any]], *, lang: str) -> list[str]:
    rows = [dict(item) for item in metric_table or [] if isinstance(item, dict)]
    if not rows:
        return []
    heading = "核心指标表" if lang == "zh" else "Core Metric Table"
    metric_label = "指标" if lang == "zh" else "Metric"
    value_label = "数值" if lang == "zh" else "Value"
    interpretation_label = "当前解读" if lang == "zh" else "Current Read"
    lines = [
        heading,
        f"| {metric_label} | {value_label} | {interpretation_label} |",
        "| --- | --- | --- |",
    ]
    for row in rows:
        label = _clean_methodology_visible_text(str(row.get("label") or row.get("metric_id") or ""), lang)
        value = _clean_methodology_visible_text(str(row.get("value") or ("缺少可验证数据" if lang == "zh" else "verified data unavailable")), lang)
        if lang == "zh" and value.startswith(f"{label} 缺少可验证数据"):
            value = "缺少可验证数据"
        elif lang != "zh" and value.lower().startswith(f"{label.lower()} verified data unavailable"):
            value = "verified data unavailable"
        interpretation = _clean_methodology_visible_text(str(row.get("interpretation") or ""), lang)
        refs = _methodology_inline_refs([str(ref) for ref in row.get("evidence_refs", []) or []])
        lines.append(f"| {label} | {value} {refs} | {interpretation} |".strip())
    return lines


def _single_company_composite_dimensions(sections_by_dimension: dict[str, dict[str, Any]]) -> set[str]:
    requested = {"cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"}
    if len(sections_by_dimension) > 4:
        return set()
    present = {dimension_id for dimension_id in requested if dimension_id in sections_by_dimension}
    return present if len(present) >= 2 else set()


def _single_company_section_text(
    sections_by_dimension: dict[str, dict[str, Any]],
    dimension_id: str,
    *,
    lang: str,
) -> str:
    item = dict(sections_by_dimension.get(dimension_id, {}) or {})
    if not item:
        return ""
    status = str(item.get("status") or "").strip()
    refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
    summary = _section_text_with_refs(str(item.get("summary", "")), refs, lang)
    if status == "missing" or not summary:
        summary = _clean_methodology_visible_text(str(item.get("limitation") or ""), lang)
    return summary


def _single_company_overview_profile(answer: dict[str, Any], sections_by_dimension: dict[str, dict[str, Any]]) -> dict[str, Any]:
    profile = dict(answer.get("overview_profile", {}) or {})
    if str(profile.get("kind") or "") == "single_company_overview":
        return profile
    return {}


def _single_company_overview_section(
    sections_by_dimension: dict[str, dict[str, Any]],
    dimension_id: str,
    *,
    lang: str,
) -> str:
    text = _single_company_section_text(sections_by_dimension, dimension_id, lang=lang)
    if text:
        return text
    return "该维度证据不足，因此只能基于已验证证据做有限判断。" if lang == "zh" else "Evidence for this dimension is insufficient, so only a limited view is supported."


def _render_overview_risk_lines(answer: dict[str, Any], sections_by_dimension: dict[str, dict[str, Any]], *, lang: str) -> list[str]:
    ranking = [
        dict(item)
        for item in answer.get("overview_risk_ranking", []) or []
        if isinstance(item, dict) and any(str(ref).startswith("T") for ref in item.get("evidence_refs", []) or [])
    ]
    if not ranking:
        return [_single_company_overview_section(sections_by_dimension, "moat_and_competitive_risk", lang=lang)]
    lines: list[str] = []
    for item in ranking[:4]:
        rank = _clean_methodology_visible_text(str(item.get("rank_label") or ""), lang)
        theme = _clean_methodology_visible_text(str(item.get("theme") or ""), lang)
        mechanism = _clean_methodology_visible_text(str(item.get("mechanism") or ""), lang)
        refs = _methodology_inline_refs([str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).startswith("T")])
        if lang == "zh":
            lines.append(f"- {rank}：{theme}，传导机制是{mechanism}。{refs}".strip())
        else:
            lines.append(f"- {rank}: {theme}; mechanism: {mechanism}. {refs}".strip())
    return lines


def _render_methodology_single_company_overview(
    answer: dict[str, Any],
    sections_by_dimension: dict[str, dict[str, Any]],
    *,
    lang: str,
) -> str:
    headings = (
        (
            "结论",
            "业务定位",
            "收入和盈利",
            "现金流与估值",
            "主要风险",
            "证据边界",
        )
        if lang == "zh"
        else (
            "Conclusion",
            "Business Positioning",
            "Revenue And Profitability",
            "Cash Flow And Valuation",
            "Primary Risks",
            "Evidence Boundary",
        )
    )
    judgment = _clean_methodology_visible_text(str(answer.get("judgment") or ""), lang)
    if not judgment:
        judgment = "当前证据不足以形成完整公司分析。" if lang == "zh" else "Current evidence is insufficient for a complete company analysis."
    business_positioning = _single_company_overview_section(sections_by_dimension, "business_model", lang=lang)
    revenue_profit = [
        _single_company_overview_section(sections_by_dimension, "revenue_quality", lang=lang),
        _single_company_overview_section(sections_by_dimension, "profitability_quality", lang=lang),
    ]
    cash_valuation = [
        _single_company_overview_section(sections_by_dimension, "cash_flow_quality", lang=lang),
        _single_company_overview_section(sections_by_dimension, "balance_sheet_and_capital_intensity", lang=lang),
        _single_company_overview_section(sections_by_dimension, "valuation_and_risk_boundary", lang=lang),
    ]
    follow_up_metrics = [
        _clean_methodology_visible_text(str(item), lang)
        for item in answer.get("follow_up_metrics", []) or []
        if str(item).strip()
    ]
    limitation_lines = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("limitations", []) or [] if str(item).strip()],
        lang=lang,
        limit=5,
    )
    if not limitation_lines:
        limitation_lines = [
            "当前分析基于已验证财报和文本证据，不构成投资建议。"
            if lang == "zh"
            else "This analysis is based on validated filings and text evidence only; it is not investment advice."
        ]
    if follow_up_metrics:
        follow_up_line = (
            "后续仍需统一验证：" + "、".join(list(dict.fromkeys(follow_up_metrics))[:5]) + "。"
            if lang == "zh"
            else "Still to verify: " + ", ".join(list(dict.fromkeys(follow_up_metrics))[:5]) + "."
        )
        limitation_lines = [follow_up_line, *limitation_lines]
    lines = [
        headings[0],
        judgment,
        "",
        headings[1],
        f"- {business_positioning}",
        "",
        headings[2],
        *[f"- {item}" for item in revenue_profit if item],
        "",
        headings[3],
        *[f"- {item}" for item in cash_valuation if item],
        "",
        headings[4],
        *_render_overview_risk_lines(answer, sections_by_dimension, lang=lang),
        "",
        headings[5],
        *[f"- {item}" for item in _dedupe_methodology_limitations(limitation_lines, lang=lang, limit=6)],
    ]
    return _normalize_methodology_punctuation("\n".join(line for line in lines if line).strip(), lang)


def render_methodology_single_company_brief(methodology_answer: dict[str, Any], *, lang: str = "zh") -> str:
    """Render a single-company methodology answer as the primary user-facing answer."""
    answer = dict(methodology_answer or {})
    sections = [item for item in answer.get("dimension_sections", []) or [] if isinstance(item, dict)]
    sections_by_dimension = {
        str(item.get("dimension_id") or ""): dict(item)
        for item in sections
        if str(item.get("dimension_id") or "").strip()
    }
    dimension_order = [
        "business_model",
        "revenue_quality",
        "profitability_quality",
        "cash_flow_quality",
        "balance_sheet_and_capital_intensity",
        "moat_and_competitive_risk",
        "valuation_and_risk_boundary",
    ]
    dimension_lines: list[str] = []
    dimension_limitation_keys: set[str] = set()
    index = 1
    for dimension_id in dimension_order:
        item = sections_by_dimension.get(dimension_id)
        if not item:
            continue
        title = str(item.get("title") or item.get("dimension_id") or "").strip()
        status = str(item.get("status") or "").strip()
        refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
        summary = _section_text_with_refs(str(item.get("summary", "")), refs, lang)
        if status == "missing" or not summary:
            limitation = str(item.get("limitation") or "").strip()
            if not limitation:
                if dimension_id == "moat_and_competitive_risk":
                    limitation = "当前缺少风险文本证据，不能做具体风险判断。" if lang == "zh" else "Risk text evidence is missing, so specific risk judgments are not supported."
                elif dimension_id == "valuation_and_risk_boundary":
                    limitation = "当前缺少估值证据，因此不能判断估值吸引力或是否便宜/昂贵。" if lang == "zh" else "Valuation evidence is missing, so valuation attractiveness cannot be judged."
                elif dimension_id == "cash_flow_quality":
                    limitation = "当前缺少经营现金流、自由现金流或资本开支证据，无法验证利润能否转化为现金。" if lang == "zh" else "Cash-flow evidence is missing, so earnings cash conversion cannot be verified."
                elif dimension_id == "balance_sheet_and_capital_intensity":
                    limitation = "当前缺少现金、债务、资本开支、应收款或存货证据，不能判断抗风险能力和资本投入强度。" if lang == "zh" else "Balance-sheet and capex evidence is missing, so resilience and capital intensity cannot be judged."
                else:
                    limitation = "当前缺少该维度证据。" if lang == "zh" else "Evidence for this dimension is currently missing."
            summary = _clean_methodology_visible_text(limitation, lang)
            dimension_limitation_keys.add(_methodology_limitation_key(summary))
        prefix = f"{index}. {title}：" if lang == "zh" else f"{index}. {title}: "
        dimension_lines.append(f"{prefix}{summary}".strip())
        index += 1

    limitation_lines = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("limitations", []) or [] if str(item).strip()],
        lang=lang,
        exclude_keys=dimension_limitation_keys,
        limit=5,
    )
    follow_up_metrics = [
        _clean_methodology_visible_text(str(item), lang)
        for item in answer.get("follow_up_metrics", []) or []
        if str(item).strip()
    ]
    judgment = _clean_methodology_visible_text(str(answer.get("judgment", "")).strip(), lang)
    composite_dimensions = _single_company_composite_dimensions(sections_by_dimension)
    if composite_dimensions:
        headings = (
            ("单公司组合分析", "总体判断", "现金流质量", "估值边界", "主要风险", "反方因素 / 不确定性", "后续观察指标", "证据边界")
            if lang == "zh"
            else ("Single-Company Composite Analysis", "Overall Judgment", "Cash-Flow Quality", "Valuation Boundary", "Primary Risks", "Counterpoints / Uncertainty", "Follow-up Indicators", "Evidence Boundary")
        )
        lines = [
            headings[0],
            "",
            headings[1],
            judgment or ("当前证据不足以形成完整组合分析。" if lang == "zh" else "Current evidence is not enough for a complete composite analysis."),
        ]
        section_specs = [
            ("cash_flow_quality", headings[2]),
            ("valuation_and_risk_boundary", headings[3]),
            ("moat_and_competitive_risk", headings[4]),
        ]
        for dimension_id, heading in section_specs:
            section_text = _single_company_section_text(sections_by_dimension, dimension_id, lang=lang)
            if section_text:
                lines.extend(["", heading, section_text])

        uncertainty_lines: list[str] = []
        counterpoint = _clean_methodology_visible_text(str(answer.get("counterpoint") or ""), lang)
        if counterpoint:
            uncertainty_lines.append(counterpoint)
        uncertainty_lines.extend(limitation_lines[:3])
        if not uncertainty_lines:
            uncertainty_lines.append(
                "反方因素是：这些判断仍依赖后续披露验证，不能外推为确定预测或投资建议。"
                if lang == "zh"
                else "The counterpoint is that these views still depend on later filings and should not be extrapolated into deterministic forecasts or investment advice."
            )
        lines.extend(["", headings[5], *[f"- {item}" for item in _dedupe_methodology_limitations(uncertainty_lines, lang=lang, limit=4)]])

        if follow_up_metrics:
            lines.extend(["", headings[6], *[f"- {item}" for item in list(dict.fromkeys(follow_up_metrics))[:6]]])
        boundary_lines = limitation_lines or [
            "当前分析基于已验证财报和文本证据，不构成投资建议。"
            if lang == "zh"
            else "This analysis is based on validated filings and text evidence only; it is not investment advice."
        ]
        lines.extend(["", headings[7], *[f"- {item}" for item in _dedupe_methodology_limitations(boundary_lines, lang=lang, limit=5)]])
        return "\n".join(line for line in lines if line).strip()

    overview_profile = _single_company_overview_profile(answer, sections_by_dimension)
    if overview_profile:
        rendered = _render_methodology_single_company_overview(answer, sections_by_dimension, lang=lang)
        if rendered.strip():
            return rendered

    headings = (
        ("基本面快速分析", "初步判断", "维度分析", "证据限制", "后续应关注指标")
        if lang == "zh"
        else ("Fundamental Quick Analysis", "Initial View", "Dimension Analysis", "Evidence Limits", "Follow-up Metrics")
    )
    judgment = _clean_methodology_visible_text(str(answer.get("judgment", "")).strip(), lang)
    lines = [headings[0], "", headings[1], judgment or ("当前证据不足以形成完整基本面判断。" if lang == "zh" else "Current evidence is not enough for a complete fundamental view.")]
    metric_lines = _render_single_company_metric_table(
        [dict(item) for item in answer.get("single_company_metric_table", []) or [] if isinstance(item, dict)],
        lang=lang,
    )
    if metric_lines:
        lines.extend(["", *metric_lines])
    if dimension_lines:
        lines.extend(["", headings[2], *dimension_lines])
    if limitation_lines:
        lines.extend(["", headings[3], *[f"- {item}" for item in limitation_lines]])
    if follow_up_metrics:
        lines.extend(["", headings[4], *[f"- {item}" for item in list(dict.fromkeys(follow_up_metrics))[:5]]])
    return "\n".join(line for line in lines if line).strip()


def render_methodology_comparison_brief(methodology_answer: dict[str, Any], *, lang: str = "zh") -> str:
    """Render the methodology comparison as the primary user-facing answer."""
    answer = dict(methodology_answer or {})
    sections = [item for item in answer.get("dimension_sections", []) or [] if isinstance(item, dict)]
    sections_by_dimension = {
        str(item.get("dimension_id") or ""): dict(item)
        for item in sections
        if str(item.get("dimension_id") or "").strip()
    }

    judgment = _clean_methodology_visible_text(str(answer.get("judgment", "")).strip(), lang)
    counterpoint = _clean_methodology_visible_text(str(answer.get("counterpoint", "")).strip(), lang)
    dimension_lines: list[str] = []
    dimension_limitation_keys: set[str] = set()
    index = 1
    for dimension_id in _METHODOLOGY_COMPARISON_DIMENSION_ORDER:
        item = sections_by_dimension.get(dimension_id)
        if not item:
            continue
        title = str(item.get("title") or item.get("dimension_id") or "").strip()
        status = str(item.get("status") or "").strip()
        refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
        summary = _section_text_with_refs(str(item.get("summary", "")), refs, lang)
        if status == "missing":
            limitation = str(item.get("limitation") or "").strip()
            if not limitation and dimension_id == "valuation_and_risk_boundary":
                limitation = (
                    "当前缺少估值证据，因此不能判断谁的估值更有吸引力。"
                    if lang == "zh"
                    else "Valuation evidence is missing, so I cannot judge which valuation boundary is more attractive."
                )
            if not limitation:
                limitation = (
                    "当前缺少该维度证据。"
                    if lang == "zh"
                    else "Evidence for this dimension is currently missing."
                )
            summary = _clean_methodology_visible_text(limitation, lang)
            dimension_limitation_keys.add(_methodology_limitation_key(summary))
        elif not summary:
            summary = (
                "当前缺少该维度证据。"
                if lang == "zh"
                else "Evidence for this dimension is currently missing."
            )
            dimension_limitation_keys.add(_methodology_limitation_key(summary))
        prefix = f"{index}. {title}：" if lang == "zh" else f"{index}. {title}: "
        dimension_lines.append(f"{prefix}{summary}".strip())
        index += 1
    for item in sections:
        dimension_id = str(item.get("dimension_id") or "")
        if dimension_id in _METHODOLOGY_COMPARISON_DIMENSION_ORDER:
            continue
        status = str(item.get("status") or "").strip()
        if status not in {"satisfied", "partial", "missing"}:
            continue
        title = str(item.get("title") or item.get("dimension_id") or "").strip()
        refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
        summary = _section_text_with_refs(str(item.get("summary", "")), refs, lang)
        if status == "missing":
            summary = _clean_methodology_visible_text(str(item.get("limitation") or ""), lang) or (
                "当前缺少该维度证据。" if lang == "zh" else "Evidence for this dimension is currently missing."
            )
            dimension_limitation_keys.add(_methodology_limitation_key(summary))
        if summary:
            prefix = f"{index}. {title}：" if lang == "zh" else f"{index}. {title}: "
            dimension_lines.append(f"{prefix}{summary}".strip())
            index += 1
    raw_limitations = [str(item) for item in answer.get("limitations", []) or [] if str(item).strip()]
    limitation_lines = _dedupe_methodology_limitations(
        raw_limitations,
        lang=lang,
        exclude_keys=dimension_limitation_keys,
        limit=5,
    )
    if not limitation_lines:
        limitation_lines = [
            "证据限制：当前比较只基于已验证证据，缺失维度不能外推为确定结论。"
            if lang == "zh"
            else "Evidence limit: this comparison is based only on validated evidence, and missing dimensions cannot be extrapolated into certainty."
        ]

    headings = (
        ("基本面方法论比较", "比较判断", "维度分析", "反方观点", "限制")
        if lang == "zh"
        else ("Fundamental Methodology Comparison", "Judgment", "Dimension Analysis", "Counterpoint", "Limits")
    )
    lines = [headings[0], "", headings[1], judgment or ("当前证据不足以形成完整比较判断。" if lang == "zh" else "Current evidence is not enough for a complete comparison judgment.")]
    metric_lines = _render_methodology_metric_table(
        [dict(item) for item in answer.get("metric_table", []) or [] if isinstance(item, dict)],
        lang=lang,
    )
    if metric_lines:
        lines.extend(["", *metric_lines])
    if dimension_lines:
        lines.extend(["", headings[2], *dimension_lines])
    if counterpoint:
        lines.extend(["", headings[3], counterpoint])
    if limitation_lines:
        lines.extend(["", headings[4], *[f"- {item}" for item in limitation_lines]])
    return "\n".join(line for line in lines if line).strip()


def _risk_theme_line(theme: dict[str, Any], *, lang: str) -> str:
    name = _clean_methodology_visible_text(display_theme(risk_theme_key(theme), lang), lang)
    why = _clean_methodology_visible_text(str(theme.get("why_it_matters") or ""), lang)
    refs = _methodology_inline_refs([str(ref) for ref in theme.get("evidence_refs", []) or []])
    if not why:
        why = "该主题来自已验证风险披露。" if lang == "zh" else "This theme comes from validated risk disclosure."
    return f"{name}：{why} {refs}".strip() if lang == "zh" else f"{name}: {why} {refs}".strip()


def render_risk_focused_analysis_brief(risk_answer: dict[str, Any], *, lang: str = "zh") -> str:
    """Render a single-company risk-focused answer as the primary answer.

    The body intentionally summarizes evidence instead of pasting SEC snippets.
    Raw snippets stay in the Evidence panel and trace.
    """
    answer = dict(risk_answer or {})
    top_risk = dict(answer.get("top_risk", {}) or {})
    secondary = [
        dict(item)
        for item in answer.get("secondary_risks", []) or []
        if isinstance(item, dict)
    ]
    direct = _clean_methodology_visible_text(str(answer.get("direct_judgment") or "").strip(), lang)
    if not direct:
        company = str(answer.get("company") or "").strip()
        direct = (
            f"当前缺少足够风险文本证据，不能可靠判断 {company} 最大的问题。"
            if lang == "zh"
            else f"Validated risk text evidence is insufficient to identify {company}'s biggest issue."
        )
    direct_refs = _methodology_inline_refs(
        [
            str(ref)
            for ref in (
                answer.get("evidence_refs", [])
                or top_risk.get("evidence_refs", [])
                or []
            )
            if str(ref).startswith("T")
        ]
    )
    if direct_refs and not re.search(r"\[[NT]\d+\]", direct):
        direct = f"{direct}{direct_refs}"

    scenario_caveated = bool(answer.get("scenario_caveated"))
    ranking_items = [
        dict(item)
        for item in answer.get("risk_ranking", []) or []
        if isinstance(item, dict)
        and str(item.get("mechanism_support_level") or "direct") != "template_only"
        and any(str(ref).startswith("T") for ref in item.get("evidence_refs", []) or [])
    ]
    if not ranking_items:
        fallback_themes = ([top_risk] if top_risk else []) + secondary
        for index, theme in enumerate(fallback_themes[:4], start=1):
            if not theme:
                continue
            if str(theme.get("mechanism_support_level") or "inferred") != "direct":
                continue
            refs = [str(ref) for ref in theme.get("evidence_refs", []) or [] if str(ref).startswith("T")]
            if not refs:
                continue
            ranking_items.append(
                {
                    "rank": index,
                    "theme_key": risk_theme_key(theme),
                    "theme_name": display_theme(risk_theme_key(theme), lang),
                    "rationale": theme.get("why_it_matters"),
                    "evidence_refs": refs,
                }
            )
    if ranking_items and not scenario_caveated:
        first_name = _clean_methodology_visible_text(display_theme(risk_theme_key(ranking_items[0]), lang), lang)
        scenario_anchor = dict(answer.get("scenario_anchor", {}) or {})
        scenario_label = _clean_methodology_visible_text(str(scenario_anchor.get("label") or ""), lang)
        if first_name and scenario_label and lang == "zh" and ("最突出" in direct or "最重要" in direct or "最需要关注" in direct):
            direct = f"在{scenario_label}情景下，基于已验证风险文本，我认为当前最需要关注的是{first_name}；其余风险按下列排序阅读。"
        elif first_name and scenario_label and lang != "zh" and ("most prominent" in direct.lower() or "main issue" in direct.lower()):
            direct = f"Under the {scenario_label} scenario, the most prominent validated-text risk is {first_name}; read the remaining risks in the ranking below."
        elif first_name and lang == "zh" and ("最突出" in direct or "最重要" in direct or "最需要关注" in direct):
            direct = f"基于披露信息，我认为当前最需要关注的是{first_name}；其余风险按下列排序阅读。"
        elif first_name and lang != "zh" and ("most prominent" in direct.lower() or "main issue" in direct.lower()):
            direct = f"Based on the disclosures, the most prominent risk is {first_name}; read the remaining risks in the ranking below."
    elif scenario_caveated:
        scenario_anchor = dict(answer.get("scenario_anchor", {}) or {})
        scenario_label = _clean_methodology_visible_text(str(scenario_anchor.get("label") or ""), lang)
        if scenario_label:
            direct = (
                f"在{scenario_label}情景下，公司披露没有直接量化该情景；以下只能作为可监控的已披露风险线索。"
                if lang == "zh"
                else f"Under the {scenario_label} scenario, the filings do not directly quantify that scenario; the items below are monitorable disclosed risk signals only."
            )

    ranking_lines: list[str] = []
    zh_rank_labels = ["最高优先级", "第二优先级", "第三优先级", "第四优先级"]
    en_rank_labels = ["Top priority", "Second priority", "Third priority", "Fourth priority"]
    rank_labels = zh_rank_labels if lang == "zh" else en_rank_labels
    for item in ranking_items[:4]:
        rank_index = len(ranking_lines)
        rank = rank_labels[rank_index] if rank_index < len(rank_labels) else ("后续优先级" if lang == "zh" else "Additional priority")
        name = _clean_methodology_visible_text(display_theme(risk_theme_key(item), lang), lang)
        rationale = _clean_methodology_visible_text(str(item.get("rationale") or ""), lang)
        refs = _methodology_inline_refs([str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).startswith("T")])
        if not refs:
            continue
        if not rationale:
            rationale = "该主题来自已验证风险披露。" if lang == "zh" else "This theme comes from validated risk disclosure."
        ranking_lines.append(f"{rank}：{name}。{rationale} {refs}".strip() if lang == "zh" else f"{rank}: {name}. {rationale} {refs}".strip())

    supplemental_items = [
        dict(item)
        for item in answer.get("supplemental_risk_signals", []) or []
        if isinstance(item, dict)
        and str(item.get("mechanism_support_level") or "inferred") != "direct"
        and any(str(ref).startswith("T") for ref in item.get("evidence_refs", []) or [])
    ]
    ranked_theme_names = {
        _clean_methodology_visible_text(display_theme(risk_theme_key(item), lang), lang)
        for item in ranking_items
        if risk_theme_key(item)
    }
    supplemental_items = [
        item
        for item in supplemental_items
        if _clean_methodology_visible_text(display_theme(risk_theme_key(item), lang), lang) not in ranked_theme_names
    ]
    if not supplemental_items:
        for theme in ([top_risk] if top_risk else []) + secondary:
            if not theme or str(theme.get("mechanism_support_level") or "inferred") == "direct":
                continue
            if _clean_methodology_visible_text(display_theme(risk_theme_key(theme), lang), lang) in ranked_theme_names:
                continue
            refs = [str(ref) for ref in theme.get("evidence_refs", []) or [] if str(ref).startswith("T")]
            if not refs:
                continue
            supplemental_items.append(
                {
                    "theme_key": risk_theme_key(theme),
                    "theme_name": display_theme(risk_theme_key(theme), lang),
                    "rationale": theme.get("why_it_matters"),
                    "evidence_refs": refs,
                    "mechanism_support_level": theme.get("mechanism_support_level") or "inferred",
                }
            )
    supplemental_lines: list[str] = []
    for item in supplemental_items[:4]:
        name = _clean_methodology_visible_text(display_theme(risk_theme_key(item), lang), lang)
        rationale = _clean_methodology_visible_text(str(item.get("rationale") or ""), lang)
        refs = _methodology_inline_refs([str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).startswith("T")])
        if not refs:
            continue
        if not rationale:
            rationale = "该线索只有间接文本支持，不能进入核心排序。" if lang == "zh" else "This signal has indirect text support only and is not in the core ranking."
        supplemental_lines.append(f"{name}：{rationale} {refs}".strip() if lang == "zh" else f"{name}: {rationale} {refs}".strip())

    verified_lines: list[str] = []
    for item in answer.get("filing_evidence", []) or []:
        if not isinstance(item, dict):
            continue
        theme = _clean_methodology_visible_text(display_theme(risk_theme_key(item), lang), lang)
        support_level = str(item.get("mechanism_support_level") or "direct")
        quote_source = (
            (item.get("supporting_quote") or item.get("why_it_matters"))
            if support_level != "direct"
            else (item.get("why_it_matters") if lang == "zh" else (item.get("supporting_quote") or item.get("why_it_matters")))
        )
        quote = _clean_methodology_visible_text(str(quote_source or ""), lang)
        refs = _methodology_inline_refs([str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).startswith("T")])
        if theme and quote and refs:
            verified_lines.append(f"- {theme}：{quote} {refs}".strip() if lang == "zh" else f"- {theme}: {quote} {refs}".strip())
    if not verified_lines and ranking_lines:
        verified_lines = [f"- {line}" for line in ranking_lines[:4]]

    transmission_lines = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("transmission_path", []) or [] if str(item).strip()],
        lang=lang,
        limit=5,
    )
    if not transmission_lines:
        transmission_lines = _dedupe_methodology_limitations(
            [str(item) for item in answer.get("why_core_issue", []) or [] if str(item).strip()],
            lang=lang,
            limit=4,
        )

    financial_impact = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("financial_impact_analysis", []) or [] if str(item).strip()],
        lang=lang,
        limit=5,
    )
    if not financial_impact:
        financial_impact = _dedupe_methodology_limitations(
            [str(item) for item in answer.get("financial_context", []) or [] if str(item).strip()],
            lang=lang,
            limit=5,
        )

    observation_metrics = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("key_observation_metrics", []) or [] if str(item).strip() and re.search(r"\[[NT]\d+\]", str(item))],
        lang=lang,
        limit=6,
    )
    boundaries = _dedupe_methodology_limitations(
        [str(item) for item in answer.get("evidence_boundaries", []) or [] if str(item).strip()],
        lang=lang,
        limit=5,
    )
    if not boundaries:
        boundaries = [
            "该回答只基于已验证财报文本和结构化财务背景，不构成投资建议。"
            if lang == "zh"
            else "This answer uses only validated filing text and financial context, and is not investment advice."
        ]

    headings = (
        ("结论", "已验证风险文本", "基于业务模型的风险排序", "财务传导路径", "待验证数据", "证据边界")
        if lang == "zh"
        else ("Risk Judgment", "Verified Risk Text", "Business-Model Inference", "Financial Transmission Path", "Data to Verify", "Evidence Boundary")
    )
    lines = [headings[0], direct]
    if verified_lines:
        lines.extend(["", headings[1], *verified_lines])
    if ranking_lines:
        lines.extend(["", headings[2], *ranking_lines])
    if supplemental_lines:
        lines.extend([*[f"- 补充：{item}" if lang == "zh" else f"- Supplemental: {item}" for item in supplemental_lines]])
    path_lines: list[str] = []
    path_lines.extend(transmission_lines)
    path_lines.extend(financial_impact)
    if path_lines:
        lines.extend(["", headings[3], *[f"- {item}" for item in path_lines]])
    data_lines = observation_metrics or (
        ["待验证：收入增速、毛利率/营业利润率、经营现金流、资本开支和 FCF。" if lang == "zh" else "To verify: revenue growth, gross/operating margin, operating cash flow, capex, and FCF."]
    )
    if data_lines:
        lines.extend(["", headings[4], *[f"- {item}" for item in data_lines]])
    if boundaries:
        lines.extend(["", headings[5], *[f"- {item}" for item in boundaries]])
    return "\n".join(line for line in lines if line).strip()

def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?])\s+|\n+", text or "")
    return [p.strip(" -•\t") for p in parts if p.strip(" -•\t")]

def _sample_docs_for_prompt(retrieved_docs: list[dict[str, Any]], max_chunks: int) -> list[dict[str, Any]]:
    if max_chunks <= 0:
        return []
    if len(retrieved_docs) <= max_chunks:
        return list(retrieved_docs)

    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for doc in retrieved_docs:
        ticker = str(doc.get("ticker", "") or "_UNKNOWN")
        if ticker not in buckets:
            buckets[ticker] = []
            order.append(ticker)
        buckets[ticker].append(doc)

    # Single ticker: keep original order.
    if len(order) <= 1:
        return list(retrieved_docs[:max_chunks])

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, str, str, int]] = set()
    idx = 0
    while len(selected) < max_chunks:
        advanced = False
        for ticker in order:
            bucket = buckets[ticker]
            if idx < len(bucket):
                doc = bucket[idx]
                key = _doc_key(doc)
                if key not in selected_keys:
                    selected.append(doc)
                    selected_keys.add(key)
                advanced = True
                if len(selected) >= max_chunks:
                    break
        if not advanced:
            break
        idx += 1

    if len(selected) < max_chunks:
        for doc in retrieved_docs:
            key = _doc_key(doc)
            if key in selected_keys:
                continue
            selected.append(doc)
            selected_keys.add(key)
            if len(selected) >= max_chunks:
                break
    return selected[:max_chunks]

def _build_key_data_lines(state: AgentState, lang: str, limit: int = 6) -> list[str]:
    tool_results = state.get("tool_results", [])
    rows = _collect_financial_rows(tool_results)
    lines: list[str] = []

    rows_sorted = sorted(rows, key=lambda r: (r.get("period_end", ""), r.get("period_type", "")), reverse=True)
    seen: set[tuple[str, str, str]] = set()
    for row in rows_sorted:
        key = (row.get("ticker", ""), row.get("metric", ""), row.get("period_type", ""))
        if key in seen:
            continue
        seen.add(key)
        metric = _metric_label(row.get("metric", ""), lang)
        value = _format_usd_value(row.get("value"), lang)
        period = row.get("period_end", "")
        period_type = _period_label(row.get("period_type"), lang)
        if lang == "zh":
            lines.append(f"- {key[0]} {metric}（{period_type}, {period}）：{value}")
        else:
            lines.append(f"- {key[0]} {metric} ({period_type}, {period}): {value}")
        if len(lines) >= limit:
            return lines

    price_rows = _collect_price_rows(tool_results)
    price_rows_sorted = sorted(price_rows, key=lambda r: r.get("date", ""), reverse=True)
    for row in price_rows_sorted:
        metrics = [k for k in row.keys() if k not in {"ticker", "date"}]
        if not metrics:
            continue
        metric = metrics[0]
        value = row.get(metric)
        if lang == "zh":
            lines.append(f"- {row.get('ticker', '')} {_metric_label(metric, lang)}（{row.get('date', '')}）：{value}")
        else:
            lines.append(f"- {row.get('ticker', '')} {_metric_label(metric, lang)} ({row.get('date', '')}): {value}")
        if len(lines) >= limit:
            break
    return lines

def _build_risk_lines(state: AgentState, lang: str) -> list[str]:
    lines: list[str] = []
    errors = [tr for tr in state.get("tool_results", []) if "error" in tr]
    if errors:
        if lang == "zh":
            lines.append("- 部分工具调用失败，结果可能不完整。")
        else:
            lines.append("- Some tool calls failed; results may be incomplete.")

    retrieved_docs = state.get("retrieved_docs", [])
    if retrieved_docs and all(d.get("section") == "MIXED" for d in retrieved_docs):
        if lang == "zh":
            lines.append("- 文档证据多为 MIXED 章节，叙述信息可读性有限。")
        else:
            lines.append("- Many retrieved chunks are MIXED sections with limited narrative clarity.")

    if lang == "zh":
        lines.append("- 结论仅基于当前已入库数据，可能存在披露时滞。")
    else:
        lines.append("- Conclusions are limited to currently indexed data and may lag latest disclosures.")
    return lines

def _first_sentence(text: str) -> str:
    parts = _split_sentences(text)
    return parts[0] if parts else (text or "").strip()

def _structured_source_lines(structured_sources: list[dict[str, Any]], lang: str) -> list[str]:
    lines: list[str] = []
    for src in structured_sources:
        metric_label = _metric_label(str(src.get("metric", "")), lang)
        ticker = str(src.get("ticker", ""))
        period_type = _period_label(str(src.get("period_type", "")), lang)
        period_end = str(src.get("period_end", ""))
        filing_date = str(src.get("filing_date", ""))
        source_provider = str(src.get("source_provider", "") or "")
        confidence = str(src.get("confidence", "") or "")
        warning = str(src.get("reconciliation_warning", "") or "")
        if lang == "zh":
            line = f"- {ticker} | 指标={metric_label} | 周期={period_type} | 截止={period_end}"
            if source_provider:
                line += f" | 来源={source_provider}"
            if confidence:
                line += f" | 可信度={confidence}"
            if filing_date:
                line += f" | 披露={filing_date}"
            if warning:
                line += f" | 警告={warning}"
        else:
            line = f"- {ticker} | metric={metric_label} | period_type={period_type} | period_end={period_end}"
            if source_provider:
                line += f" | source_provider={source_provider}"
            if confidence:
                line += f" | confidence={confidence}"
            if filing_date:
                line += f" | filing_date={filing_date}"
            if warning:
                line += f" | warning={warning}"
        lines.append(line)
    return lines

def _document_evidence_lines(document_citations: list[dict[str, Any]], lang: str, limit: int = 8) -> list[str]:
    lines: list[str] = []
    for c in document_citations[:limit]:
        ticker = str(c.get("ticker", c.get("source", "")))
        form_type = str(c.get("form_type", c.get("filing_type", "")))
        period = str(c.get("fiscal_period", c.get("period", "")))
        section = str(c.get("section", ""))
        fallback = bool(c.get("section_fallback"))
        if lang == "zh":
            line = f"- {ticker} | {form_type} | {period} | {section}"
            if fallback:
                line += " | 章节回退"
        else:
            line = f"- {ticker} | {form_type} | {period} | {section}"
            if fallback:
                line += " | section_fallback"
        lines.append(line)
    return lines

def _evidence_insufficient_message(lang: str, task_type: str) -> str:
    if lang == "zh":
        if task_type in NUMERIC_REQUIRED_TASK_TYPES:
            return "证据不足：当前缺少可追溯的数值证据，无法输出可靠结论。"
        return "证据不足：当前证据无法支持可追溯结论。"
    if task_type in NUMERIC_REQUIRED_TASK_TYPES:
        return "Insufficient evidence: no traceable numeric evidence is available for a reliable conclusion."
    return "Insufficient evidence: available evidence is not enough for a traceable conclusion."

def _render_claim_answer(
    lang: str,
    task_type: str,
    numeric_claims: list[dict[str, Any]],
    text_claims: list[dict[str, Any]],
    comparison_basis_label: str = "",
) -> str:
    if task_type in NUMERIC_REQUIRED_TASK_TYPES and not numeric_claims:
        msg = _evidence_insufficient_message(lang, task_type)
        if task_type == "company_comparison":
            basis_line = _comparison_basis_line(lang, comparison_basis_label)
            if basis_line:
                return f"{basis_line}\n{msg}"
        return msg
    if not numeric_claims and not text_claims:
        msg = _evidence_insufficient_message(lang, task_type)
        if task_type == "company_comparison":
            basis_line = _comparison_basis_line(lang, comparison_basis_label)
            if basis_line:
                return f"{basis_line}\n{msg}"
        return msg

    if lang == "zh":
        blocks: list[str] = ["结论"]
        if task_type == "company_comparison":
            basis_line = _comparison_basis_line(lang, comparison_basis_label)
            if basis_line:
                blocks.append(basis_line)
        if numeric_claims:
            blocks.extend([f"- {c['sentence']}" for c in numeric_claims])
        if text_claims:
            blocks.extend(["", "文档依据"])
            blocks.extend([f"- {c['sentence']}" for c in text_claims])
        blocks.extend(["", "风险与不足", "- 仅保留通过证据校验的句子。"])
        return "\n".join(blocks).strip()

    blocks = ["Conclusion"]
    if task_type == "company_comparison":
        basis_line = _comparison_basis_line(lang, comparison_basis_label)
        if basis_line:
            blocks.append(basis_line)
    if numeric_claims:
        blocks.extend([f"- {c['sentence']}" for c in numeric_claims])
    if text_claims:
        blocks.extend(["", "Document Evidence"])
        blocks.extend([f"- {c['sentence']}" for c in text_claims])
    blocks.extend(["", "Risks and Limits", "- Only claims that passed evidence validation are kept."])
    return "\n".join(blocks).strip()

def _truncate_text(text: str, max_len: int) -> str:
    raw = str(text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[: max(0, max_len - 1)].rstrip() + "…"

def _task_title(lang: str, task_type: str) -> str:
    zh_titles = {
        "fact_qa": "财报事实问答",
        "trend_analysis": "财务趋势分析",
        "company_comparison": "公司横向对比",
        "report_summary": "财报综合摘要",
    }
    en_titles = {
        "fact_qa": "Financial Fact Answer",
        "trend_analysis": "Financial Trend Analysis",
        "company_comparison": "Company Comparison",
        "report_summary": "Filing Summary",
    }
    return zh_titles.get(task_type, "分析结果") if lang == "zh" else en_titles.get(task_type, "Analysis Result")

def _dedupe_numeric_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for card in cards:
        key = (
            str(card.get("ticker", "")),
            str(card.get("metric", "")),
            str(card.get("period_end", "")),
            str(card.get("source_tool", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(card)
    return out

def _dedupe_text_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for card in cards:
        filing_ref = str(card.get("form_type", "") or card.get("filing_type", "") or card.get("filing_id", ""))
        text_key = str(
            card.get("claim")
            or card.get("supporting_snippet")
            or card.get("text_snippet")
            or card.get("evidence_summary")
            or ""
        ).lower()
        text_key = re.sub(r"\s+", " ", text_key).strip()[:220]
        key = (
            str(card.get("ticker", "")).upper().strip(),
            filing_ref.upper().strip(),
            str(card.get("section", "")).upper().strip(),
            text_key,
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(card)
    return out

def _cap_output_evidence(
    task_type: str,
    numeric_cards: list[dict[str, Any]],
    text_cards: list[dict[str, Any]],
    companies: list[str],
    comparison_target: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task_type == "company_comparison":
        target_companies: list[str] = []
        for t in companies:
            s = str(t or "").upper()
            if s and s not in target_companies:
                target_companies.append(s)
        ct = str(comparison_target or "").upper()
        if ct and ct not in target_companies:
            target_companies.append(ct)
        if not target_companies:
            target_companies = sorted({str(c.get("ticker", "")).upper() for c in numeric_cards + text_cards if str(c.get("ticker", ""))})

        n_cap = int(OUTPUT_EVIDENCE_CAPS["company_comparison"]["numeric_per_company"])
        t_cap = int(OUTPUT_EVIDENCE_CAPS["company_comparison"]["text_per_company"])
        numeric_out: list[dict[str, Any]] = []
        text_out: list[dict[str, Any]] = []
        for ticker in target_companies[:2]:
            n_rows = [r for r in numeric_cards if str(r.get("ticker", "")).upper() == ticker]
            t_rows = [r for r in text_cards if str(r.get("ticker", "")).upper() == ticker]
            numeric_out.extend(n_rows[: max(0, n_cap)])
            text_out.extend(t_rows[: max(0, t_cap)])
        return numeric_out, text_out

    caps = OUTPUT_EVIDENCE_CAPS.get(task_type, {"numeric": 4, "text": 3})
    n_cap = int(caps.get("numeric", 4))
    t_cap = int(caps.get("text", 3))
    return numeric_cards[: max(0, n_cap)], text_cards[: max(0, t_cap)]

def _limitation_item(code: str, severity: str, message: str) -> dict[str, Any]:
    return {
        "code": str(code),
        "severity": str(severity),
        "message": str(message),
    }

def _build_limitations(
    lang: str,
    task_type: str,
    unsupported_claims: list[dict[str, Any]],
    numeric_claims: list[dict[str, Any]],
    text_claims: list[dict[str, Any]],
    requested_metrics: list[str],
    period_error: str | None,
    comparison_text_unbalanced: bool,
    requirement_aware: bool = False,
    market_reaction_limitations: list[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    if period_error:
        if period_error == "no_common_period_for_same_period_comparison":
            out.append(
                _limitation_item(
                    "no_common_period",
                    "high",
                    "当前两家公司没有共同可比期间。" if lang == "zh" else "No common comparable period across companies.",
                )
            )
        else:
            out.append(
                _limitation_item(
                    "period_mismatch",
                    "high",
                    "时间口径与可用证据不一致。" if lang == "zh" else "Requested period is inconsistent with available evidence.",
                )
            )

    if comparison_text_unbalanced and not requirement_aware:
        out.append(
            _limitation_item(
                "text_unbalanced",
                "medium",
                "对比文本证据不均衡，已降级为结构化主证据。"
                if lang == "zh"
                else "Text evidence is unbalanced across compared companies; downgraded to structured-primary evidence.",
            )
        )

    if task_type in NUMERIC_REQUIRED_TASK_TYPES and not numeric_claims and not requirement_aware:
        code = "metric_missing" if requested_metrics else "evidence_insufficient"
        msg = (
            "缺少可用指标或数值证据。"
            if lang == "zh"
            else "Required metrics or numeric evidence are missing."
        )
        out.append(_limitation_item(code, "high", msg))

    if not numeric_claims and not text_claims and not requirement_aware:
        out.append(
            _limitation_item(
                "evidence_insufficient",
                "high",
                "当前证据不足以支持结论。" if lang == "zh" else "Current evidence is insufficient for a reliable conclusion.",
            )
        )

    if unsupported_claims:
        out.append(
            _limitation_item(
                "unsupported_claims_present",
                "medium",
                f"已丢弃 {len(unsupported_claims)} 条未通过校验的句子。"
                if lang == "zh"
                else f"Dropped {len(unsupported_claims)} claims that failed validation.",
            )
        )

    if any(str(c.get("source_provider", "")).lower() == "yfinance" for c in numeric_claims):
        out.append(
            _limitation_item(
                "yfinance_structured_fallback",
                "medium",
                "结构化财务指标来自 yfinance fallback，并非 SEC XBRL 原始事实。"
                if lang == "zh"
                else "Structured financial facts are sourced from yfinance fallback, not SEC XBRL facts.",
            )
        )

    if any(str(c.get("reconciliation_warning", "")).strip() for c in numeric_claims):
        out.append(
            _limitation_item(
                "structured_data_reconciliation_warning",
                "medium",
                "结构化数据存在 SEC 与 fallback 来源数值差异。"
                if lang == "zh"
                else "Structured data includes a value mismatch between SEC and fallback sources.",
            )
        )

    for reason in market_reaction_limitations or []:
        reason_text = str(reason)
        if reason_text.startswith("no_event_window_data_for_"):
            out.append(
                _limitation_item(
                    "event_window_missing",
                    "medium",
                    "部分公司缺少财报事件窗口数据。"
                    if lang == "zh"
                    else "Event-window data is missing for part of the requested companies.",
                )
            )
        if reason_text.startswith("event_query_failed_for_"):
            out.append(
                _limitation_item(
                    "event_query_failed",
                    "medium",
                    "财报事件窗口查询失败，已回退到现有证据。"
                    if lang == "zh"
                    else "Event-window lookup failed and the answer falls back to other available evidence.",
                )
            )

    if task_type == "report_summary" and not out:
        out.append(
            _limitation_item(
                "evidence_insufficient",
                "low",
                "结论仅基于当前已入库数据范围。"
                if lang == "zh"
                else "Conclusions are limited to currently indexed data scope.",
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        code = str(item.get("code", ""))
        if code in seen:
            continue
        seen.add(code)
        deduped.append(item)
    return deduped

def _conversational_limitations(state: AgentState, lang: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    safety_intent = str(state.get("safety_intent", "normal"))
    if safety_intent == "investment_advice_like":
        out.append(
            _limitation_item(
                "investment_advice_boundary",
                "high",
                "以下内容仅是基于证据的分析框架，不构成投资建议、买卖推荐或股价预测。"
                if lang == "zh"
                else "This is an evidence-grounded analysis framework, not investment advice, a trading recommendation, or a price forecast.",
            )
        )
    if answer_mode == "cautious_outlook":
        out.append(
            _limitation_item(
                "forward_looking_uncertainty",
                "medium",
                "涉及未来展望的判断存在不确定性，应以已验证历史事实和披露文本为边界。"
                if lang == "zh"
                else "Forward-looking discussion is uncertain and bounded by validated historical facts and filing evidence.",
            )
        )
    if state.get("needs_tools") is False:
        out.append(
            _limitation_item(
                "no_external_tools",
                "low",
                "这是对话/澄清类回答，未调用外部工具或检索证据。"
                if lang == "zh"
                else "This is a conversational/clarification response; no external tools or evidence retrieval were used.",
            )
        )
    if safety_intent == "unsupported_or_out_of_scope":
        out.append(
            _limitation_item(
                "unsupported_scope",
                "medium",
                "该问题超出当前财报分析系统的支持范围。"
                if lang == "zh"
                else "The question is outside the supported scope of this filings-analysis system.",
            )
        )
    safety_messages = {
        "no_realtime_news_access": {
            "severity": "high",
            "zh": "当前系统没有 web search 或实时行情源，因此不会声称知道实时新闻或实时股价。",
            "en": "This system has no web search or live market-data source, so it will not claim real-time news or live prices.",
        },
        "unsupported_price_prediction": {
            "severity": "high",
            "zh": "短期股价预测超出当前系统支持范围。",
            "en": "Near-term stock-price prediction is outside the supported scope.",
        },
        "insufficient_validated_evidence": {
            "severity": "medium",
            "zh": "当前没有足够的已验证证据支撑更具体的结论。",
            "en": "Current validated evidence is insufficient for a more specific conclusion.",
        },
    }
    for item in state.get("safety_limitations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", ""))
        if not code:
            continue
        known = safety_messages.get(code)
        out.append(
            _limitation_item(
                code,
                str(item.get("severity") or (known or {}).get("severity") or "medium"),
                str((known or {}).get(lang) or item.get("message") or ""),
            )
        )
    for item in state.get("requirement_limitations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", ""))
        if not code:
            continue
        out.append(
            _limitation_item(
                code,
                str(item.get("severity") or "medium"),
                str(item.get("message") or ""),
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in out:
        code = str(item.get("code", ""))
        if code in seen:
            continue
        seen.add(code)
        deduped.append(item)
    return deduped

def _table_from_numeric_cards(
    cards: list[dict[str, Any]],
    max_rows: int = 10,
    include_placeholder: bool = False,
) -> dict[str, Any]:
    columns = ["ticker", "metric", "period_type", "period_end", "value", "unit"]
    rows: list[dict[str, Any]] = []
    for card in cards[: max(0, max_rows)]:
        rows.append(
            {
                "ticker": str(card.get("ticker", "")),
                "metric": str(card.get("metric", "")),
                "period_type": str(card.get("period_type", "")),
                "period_end": str(card.get("period_end", "")),
                "value": card.get("value"),
                "unit": str(card.get("unit", "")),
            }
        )
    if include_placeholder and not rows:
        rows.append(
            {
                "ticker": "N/A",
                "metric": "N/A",
                "period_type": "N/A",
                "period_end": "N/A",
                "value": "N/A",
                "unit": "",
            }
        )
    return {"columns": columns, "rows": rows}

def _build_output_view(
    lang: str,
    task_type: str,
    answer_mode: str,
    safety_intent: str,
    summary: str,
    key_points: list[str],
    numeric_cards: list[dict[str, Any]],
    text_cards: list[dict[str, Any]],
    period_query: dict[str, Any],
    comparison_basis_line: str,
    limitations: list[dict[str, Any]],
) -> dict[str, Any]:
    view_kind = {
        "meta": "meta_response",
        "clarification": "clarification",
        "cautious_outlook": "cautious_outlook",
        "comparison_brief": "comparison_brief",
        "refusal_or_redirect": "refusal_or_redirect",
    }.get(answer_mode, "analytical_brief")
    if task_type == "company_comparison" or safety_intent == "investment_advice_like":
        view_kind = "comparison_brief"

    if task_type == "fact_qa":
        year = period_query.get("year")
        quarter = period_query.get("quarter")
        period_note = ""
        if year and quarter:
            period_note = f"{year}Q{quarter}" if lang != "zh" else f"口径：{year}Q{quarter}"
        elif year:
            period_note = f"FY{year}" if lang != "zh" else f"口径：{year}财年"
        elif period_query.get("period_type"):
            period_note = str(period_query.get("period_type"))
        headline = numeric_cards[0] if numeric_cards else {
            "evidence_id": "N/A",
            "ticker": "N/A",
            "metric": "N/A",
            "value": "N/A",
            "unit": "",
            "period_type": "N/A",
            "period_end": "N/A",
            "source_tool": "placeholder",
            "provenance": "insufficient_numeric_evidence",
        }
        return {
            "kind": view_kind,
            "short_answer": summary,
            "key_facts": [{"sentence": point} for point in key_points[:2]],
            "analysis": [],
            "risks_or_uncertainties": limitations[:3],
            "headline_metric": headline,
            "period_note": period_note,
            "supporting_points": key_points[:2],
        }

    if task_type == "trend_analysis":
        return {
            "kind": view_kind,
            "short_answer": summary,
            "key_facts": [{"sentence": point} for point in key_points[:4]],
            "analysis": [{"sentence": point} for point in key_points[:4]],
            "risks_or_uncertainties": limitations[:3],
            "trend_conclusion": summary,
            "change_points": key_points[:4],
            "trend_table": _table_from_numeric_cards(numeric_cards, max_rows=10),
        }

    if task_type == "company_comparison":
        delta_summary = key_points[0] if key_points else summary
        return {
            "kind": view_kind,
            "short_answer": summary,
            "key_facts": [{"sentence": point} for point in key_points[:4]],
            "analysis": [{"sentence": delta_summary}] if delta_summary else [],
            "risks_or_uncertainties": limitations[:3],
            "non_advisory_statement": (
                "以下内容仅是基于证据的分析框架，不构成投资建议、买卖推荐或股价预测。"
                if lang == "zh"
                else "This is an evidence-grounded analysis framework, not investment advice, a trading recommendation, or a price forecast."
            ),
            "comparison_basis_line": comparison_basis_line,
            "comparison_table": _table_from_numeric_cards(
                numeric_cards,
                max_rows=12,
                include_placeholder=True,
            ),
            "delta_summary": delta_summary,
        }

    risk_lines = [str(x.get("message", "")) for x in limitations if str(x.get("message", "")).strip()]
    if not risk_lines:
        risk_lines = [
            "当前未发现额外限制。" if lang == "zh" else "No additional limitation identified in this run.",
        ]
    return {
        "kind": view_kind,
        "short_answer": summary,
        "key_facts": [{"sentence": point} for point in key_points[:4]],
        "analysis": [{"sentence": c.get("text_snippet", "")} for c in text_cards[:5]],
        "risks_or_uncertainties": limitations[:3],
        "executive_summary": summary,
        "key_data_points": key_points[:4],
        "text_findings": [c.get("text_snippet", "") for c in text_cards[:5]],
        "risk_and_limits": "；".join(risk_lines[:3]) if lang == "zh" else "; ".join(risk_lines[:3]),
    }

def _format_pct_or_na(value: Any) -> str:
    num = _to_float(value)
    if num is None:
        return "N/A"
    return f"{num * 100:.2f}%"

def _build_market_reaction_block(
    state: AgentState,
    lang: str,
    numeric_cards: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not bool(state.get("market_reaction_requested")) and not state.get("event_results"):
        return None

    raw_events = _collect_event_rows(state.get("tool_results", []))
    if not raw_events and state.get("event_results"):
        for item in state.get("event_results", []):
            data = item.get("data", {}) if isinstance(item, dict) else {}
            ticker = str(data.get("ticker", item.get("ticker", ""))) if isinstance(data, dict) else ""
            for ev in data.get("events", []) if isinstance(data, dict) else []:
                if not isinstance(ev, dict):
                    continue
                e = dict(ev)
                e.setdefault("ticker", ticker)
                raw_events.append(e)

    raw_events = sorted(raw_events, key=lambda e: str(e.get("event_date", "")), reverse=True)
    event_rows: list[dict[str, Any]] = []
    for ev in raw_events[:8]:
        returns = ev.get("returns", {}) if isinstance(ev.get("returns", {}), dict) else {}
        event_rows.append(
            {
                "ticker": str(ev.get("ticker", "")),
                "filing_id": str(ev.get("filing_id", "")),
                "form_type": str(ev.get("form_type", "")),
                "fiscal_period": str(ev.get("fiscal_period", "")),
                "event_date": str(ev.get("event_date", "")),
                "trading_anchor_date": str(ev.get("trading_anchor_date", "")),
                "return_1d": _to_float(returns.get("return_1d")),
                "return_3d": _to_float(returns.get("return_3d")),
                "return_5d": _to_float(returns.get("return_5d")),
                "return_10d": _to_float(returns.get("return_10d")),
                "coverage_flag": str(ev.get("coverage_flag", "")),
            }
        )

    highlights: list[str] = []
    for row in event_rows[:4]:
        if lang == "zh":
            highlights.append(
                (
                    f"{row['ticker']} {row['event_date']} 财报后 "
                    f"1日={_format_pct_or_na(row.get('return_1d'))}，"
                    f"5日={_format_pct_or_na(row.get('return_5d'))}。"
                )
            )
        else:
            highlights.append(
                (
                    f"{row['ticker']} post-filing reaction on {row['event_date']}: "
                    f"1D={_format_pct_or_na(row.get('return_1d'))}, "
                    f"5D={_format_pct_or_na(row.get('return_5d'))}."
                )
            )

    if not highlights and numeric_cards:
        top = numeric_cards[0]
        if lang == "zh":
            highlights.append(
                f"{top.get('ticker', '')} {top.get('metric', '')} 为 {top.get('display_value', top.get('value', 'N/A'))}。"
            )
        else:
            highlights.append(
                f"{top.get('ticker', '')} {top.get('metric', '')} = {top.get('display_value', top.get('value', 'N/A'))}."
            )

    limitations = [str(x) for x in state.get("market_reaction_limitations", []) if str(x).strip()]
    if bool(state.get("market_reaction_requested")) and not event_rows:
        limitations.append("event_window_data_unavailable")
    limitations = list(dict.fromkeys(limitations))

    title = "市场反应" if lang == "zh" else "Market Reaction"
    return {
        "title": title,
        "anchor_rule": "filing_date_then_next_trading_day",
        "events": event_rows,
        "highlights": highlights,
        "limitations": limitations,
    }

def _render_answer_from_output(output: dict[str, Any], lang: str) -> str:
    task_type = str(output.get("task_type", "fact_qa"))
    title = str(output.get("title", "")).strip()
    summary = str(output.get("summary", "")).strip()
    key_points = [str(x).strip() for x in output.get("key_points", []) if str(x).strip()]
    packet = output.get("evidence_packet", {}) if isinstance(output.get("evidence_packet", {}), dict) else {}
    if packet.get("canonical_source"):
        numeric_cards = [x for x in packet.get("numeric_table", []) if isinstance(x, dict)]
        text_cards = [x for x in packet.get("text_snippets", []) if isinstance(x, dict)]
    else:
        numeric_cards = []
        text_cards = []
    limitations = [x for x in output.get("limitations", []) if isinstance(x, dict)]
    view = output.get("view", {}) if isinstance(output.get("view", {}), dict) else {}
    market_reaction = output.get("market_reaction", {}) if isinstance(output.get("market_reaction", {}), dict) else {}
    if str(view.get("kind", "")) == "methodology_comparison_brief":
        methodology_answer = dict(view.get("methodology_answer", {}) or {})
        if methodology_answer:
            rendered = render_methodology_comparison_brief(methodology_answer, lang=lang)
            if rendered.strip():
                return rendered
    if str(view.get("kind", "")) == "methodology_single_company_brief":
        methodology_answer = dict(view.get("methodology_answer", {}) or {})
        if methodology_answer:
            rendered = render_methodology_single_company_brief(methodology_answer, lang=lang)
            if rendered.strip():
                return rendered
    if str(view.get("kind", "")) == "risk_focused_analysis_brief":
        risk_answer = dict(view.get("risk_focused_answer", {}) or {})
        if risk_answer:
            rendered = render_risk_focused_analysis_brief(risk_answer, lang=lang)
            if rendered.strip():
                return rendered

    if task_type == "fact_qa":
        lines: list[str] = [title, summary]
        period_note = str(view.get("period_note", "")).strip()
        if period_note:
            lines.append(period_note)
        for point in key_points[:2]:
            lines.append(f"- {point}")
        mr_highlights = market_reaction.get("highlights", []) if isinstance(market_reaction, dict) else []
        if mr_highlights:
            lines.append("")
            lines.append("市场反应" if lang == "zh" else "Market Reaction")
            for h in mr_highlights[:2]:
                lines.append(f"- {h}")
        return "\n".join([x for x in lines if x]).strip()

    if task_type == "company_comparison":
        lines = [title, summary]
        basis = str(view.get("comparison_basis_line", "")).strip() or str(output.get("comparison_basis", "")).strip()
        if basis:
            lines.append(basis)
        for point in key_points[:4]:
            lines.append(f"- {point}")
        mr_highlights = market_reaction.get("highlights", []) if isinstance(market_reaction, dict) else []
        if mr_highlights:
            lines.append("")
            lines.append("市场反应" if lang == "zh" else "Market Reaction")
            for h in mr_highlights[:4]:
                lines.append(f"- {h}")
        if limitations:
            lines.append("")
            lines.append("风险与不足" if lang == "zh" else "Risks and Limits")
            for lim in limitations[:3]:
                lines.append(f"- {lim.get('message', '')}")
        return "\n".join([x for x in lines if x]).strip()

    section_numeric = "数字依据" if lang == "zh" else "Numeric Evidence"
    section_text = "文本依据" if lang == "zh" else "Text Evidence"
    section_limits = "风险与不足" if lang == "zh" else "Risks and Limits"
    lines = [title, summary]
    for point in key_points[:4]:
        lines.append(f"- {point}")
    if numeric_cards:
        lines.extend(["", section_numeric])
        for card in numeric_cards[:4]:
            lines.append(
                f"- {card.get('ticker', '')} {card.get('metric', '')} {card.get('period_end', '')}: {card.get('value', '')}"
            )
    if text_cards:
        lines.extend(["", section_text])
        for card in text_cards[:3]:
            lines.append(
                f"- {card.get('ticker', '')} {card.get('form_type', '')} {card.get('fiscal_period', '')} {card.get('section', '')}"
            )
    mr_highlights = market_reaction.get("highlights", []) if isinstance(market_reaction, dict) else []
    if mr_highlights:
        lines.extend(["", "市场反应" if lang == "zh" else "Market Reaction"])
        for h in mr_highlights[:4]:
            lines.append(f"- {h}")
    if limitations:
        lines.extend(["", section_limits])
        for lim in limitations[:3]:
            lines.append(f"- {lim.get('message', '')}")
    return "\n".join([x for x in lines if x]).strip()

def _build_phase4_output(
    state: AgentState,
    lang: str,
    task_type: str,
    comparison_basis_label: str,
    period_query: dict[str, Any],
    numeric_claims: list[dict[str, Any]],
    text_claims: list[dict[str, Any]],
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
    numeric_citations: list[dict[str, Any]],
    text_citations: list[dict[str, Any]],
    unsupported_claims: list[dict[str, Any]],
    period_error: str | None = None,
    comparison_text_unbalanced: bool = False,
) -> dict[str, Any]:
    numeric_map = {str(x.get("evidence_id", "")): x for x in numeric_evidence}
    text_map = {str(x.get("evidence_id", "")): x for x in text_evidence}

    numeric_cards: list[dict[str, Any]] = []
    for c in numeric_citations:
        eid = str(c.get("evidence_id", ""))
        src = numeric_map.get(eid, {})
        card = {
                "evidence_id": eid,
                "requirement_id": str(src.get("requirement_id", c.get("requirement_id", ""))),
                "ticker": str(src.get("ticker", c.get("ticker", c.get("source", "")))),
                "metric": str(src.get("metric", c.get("metric", ""))),
                "value": src.get("value"),
                "display_value": _display_numeric_value(src, lang),
                "unit": str(src.get("unit", "")),
                "period_type": str(src.get("period_type", c.get("period_type", ""))),
                "period_category": period_category(src.get("period_type", c.get("period_type", ""))),
                "period_end": str(src.get("period_end", c.get("period_end", c.get("period", "")))),
                "source_tool": str(src.get("source_tool", "")),
                "provenance": str(src.get("provenance", "")),
                "source_provider": str(src.get("source_provider", c.get("source_provider", ""))),
                "source_url": str(src.get("source_url", c.get("source_url", ""))),
                "source_filing_id": str(src.get("source_filing_id", c.get("source_filing_id", ""))),
                "confidence": str(src.get("confidence", c.get("confidence", ""))),
                "extraction_method": str(src.get("extraction_method", c.get("extraction_method", ""))),
                "source_tag": str(src.get("source_tag", c.get("source_tag", ""))),
                "reconciliation_warning": str(
                    src.get("reconciliation_warning", c.get("reconciliation_warning", ""))
                ),
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
            if trace_key in src:
                card[trace_key] = src.get(trace_key)
        numeric_cards.append(card)

    text_cards: list[dict[str, Any]] = []
    for c in text_citations:
        eid = str(c.get("evidence_id", ""))
        src = text_map.get(eid, {})
        fiscal_period = str(
            src.get("fiscal_period")
            or c.get("fiscal_period")
            or c.get("period")
            or UNKNOWN_PERIOD
        )
        text_cards.append(
            {
                "evidence_id": eid,
                "requirement_id": str(src.get("requirement_id", c.get("requirement_id", ""))),
                "ticker": str(src.get("ticker", c.get("ticker", c.get("source", "")))),
                "filing_id": str(src.get("filing_id", c.get("filing_id", ""))),
                "form_type": str(src.get("form_type", c.get("form_type", c.get("filing_type", "")))),
                "fiscal_period": fiscal_period,
                "section": str(src.get("section", c.get("section", ""))),
                "chunk_order": int(src.get("chunk_order", c.get("chunk_order", 0)) or 0),
                "text_snippet": _truncate_text(str(src.get("text_snippet", c.get("text_snippet", ""))), 220),
                "supporting_snippet": _truncate_text(
                    str(src.get("supporting_snippet", c.get("supporting_snippet", c.get("text_snippet", "")))),
                    360,
                ),
                "supporting_terms": list(src.get("supporting_terms", c.get("supporting_terms", [])) or []),
                "score_breakdown": dict(src.get("score_breakdown", c.get("score_breakdown", {})) or {}),
                "dimension_id": str(src.get("dimension_id", c.get("dimension_id", ""))),
                "claim": str(src.get("claim", c.get("claim", c.get("sentence", "")))),
                "claim_source": str(src.get("claim_source", c.get("claim_source", ""))),
                "citation_ref": str(src.get("citation_ref", c.get("citation_ref", eid))),
                "risk_theme": str(src.get("risk_theme", c.get("risk_theme", ""))),
                "theme_name": str(src.get("theme_name", c.get("theme_name", ""))),
                "evidence_summary": str(src.get("evidence_summary", c.get("evidence_summary", ""))),
                "driver_level": str(src.get("driver_level", c.get("driver_level", ""))),
                "driver_levels": list(src.get("driver_levels", c.get("driver_levels", [])) or []),
            }
        )

    numeric_cards = _dedupe_numeric_cards(numeric_cards)
    text_cards = _dedupe_text_cards(text_cards)
    numeric_cards, text_cards = _cap_output_evidence(
        task_type=task_type,
        numeric_cards=numeric_cards,
        text_cards=text_cards,
        companies=state.get("companies", []),
        comparison_target=state.get("comparison_target"),
    )

    claim_points = [str(c.get("sentence", "")).strip() for c in numeric_claims + text_claims if str(c.get("sentence", "")).strip()]
    claim_points = list(dict.fromkeys(claim_points))
    default_summary = _evidence_insufficient_message(lang, task_type)
    if claim_points:
        summary = claim_points[0]
    else:
        summary = default_summary
    summary = _truncate_text(summary, 180 if task_type == "fact_qa" else 240)

    limitations = _build_limitations(
        lang=lang,
        task_type=task_type,
        unsupported_claims=unsupported_claims,
        numeric_claims=numeric_claims,
        text_claims=text_claims,
        requested_metrics=state.get("requested_metrics", []),
        period_error=period_error,
        comparison_text_unbalanced=comparison_text_unbalanced,
        requirement_aware=bool(state.get("evidence_plan")),
        market_reaction_limitations=state.get("market_reaction_limitations", []),
    )
    limitation_by_code = {str(item.get("code", "")): item for item in limitations}
    for item in _conversational_limitations(state, lang):
        limitation_by_code.setdefault(str(item.get("code", "")), item)
    limitations = list(limitation_by_code.values())
    limitations = sanitize_user_facing_limitations(limitations, lang, task_type=task_type)
    comparison_basis_line = _comparison_basis_line(lang, comparison_basis_label) if task_type == "company_comparison" else ""
    output = {
        "protocol_version": OUTPUT_PROTOCOL_VERSION,
        "output_language": str(state.get("output_language") or lang),
        "task_type": task_type,
        "answer_mode": str(state.get("answer_mode", "direct_fact")),
        "safety_intent": str(state.get("safety_intent", "normal")),
        "analysis_scope": str(state.get("analysis_scope", "")),
        "time_policy": str(state.get("time_policy", "")),
        "period_scope": str(state.get("period_scope", "")),
        "synthesis_mode": str(state.get("synthesis_mode", "")),
        "needs_tools": bool(state.get("needs_tools", True)),
        "needs_clarification": bool(state.get("needs_clarification", False)),
        "clarification_question": state.get("clarification_question"),
        "title": _task_title(lang, task_type),
        "summary": summary,
        "comparison_basis": comparison_basis_label if task_type == "company_comparison" else "",
        "key_points": claim_points[: (2 if task_type == "fact_qa" else 5)],
        "numeric_evidence": numeric_cards,
        "text_evidence": text_cards,
        "limitations": limitations,
        "draft_validation": {},
        "draft_revision_attempts": [],
        "final_answer_source": "",
        "used_tools": list(state.get("selected_tools", [])),
        "trace_id": str(state.get("trace_id", "")),
    }
    analysis_plan = dict(state.get("analysis_plan", {}) or {})
    if analysis_plan:
        output["analysis_plan_summary"] = {
            "analysis_dimensions": list(analysis_plan.get("analysis_dimensions", [])),
            "needed_evidence": list(analysis_plan.get("needed_evidence", [])),
            "validated_tools": list(state.get("validated_tools", analysis_plan.get("validated_tools", []))),
        }
    selected_analysis_framework = dict(state.get("selected_analysis_framework", {}) or {})
    if selected_analysis_framework:
        output["analysis_framework"] = summarize_selected_analysis_framework(selected_analysis_framework)
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    evidence_sufficiency = dict(state.get("evidence_sufficiency", {}) or {})
    if evidence_plan:
        output["evidence_plan_summary"] = summarize_evidence_requirements(
            evidence_plan,
            list(state.get("evidence_collection_results", []) or []),
            evidence_sufficiency,
        )
        output["sufficiency_status"] = str(evidence_sufficiency.get("overall_status", ""))
        output["degradation_reason"] = evidence_sufficiency.get("degradation_reason")
    dimension_sufficiency = normalize_dimension_status_contract(
        dict(
            state.get("dimension_status_by_id")
            or evidence_sufficiency.get("dimension_status_by_id")
            or state.get("dimension_status_map")
            or evidence_sufficiency.get("dimension_status_map", {})
            or {}
        ),
        satisfied_dimensions=list(
            state.get("satisfied_dimensions")
            or evidence_sufficiency.get("satisfied_dimensions")
            or state.get("covered_dimensions")
            or evidence_sufficiency.get("covered_dimensions", [])
            or []
        ),
        partial_dimensions=list(state.get("partial_dimensions") or evidence_sufficiency.get("partial_dimensions", []) or []),
        missing_dimensions=list(state.get("missing_dimensions") or evidence_sufficiency.get("missing_dimensions", []) or []),
        dimension_coverage_rate=state.get("dimension_coverage_rate", evidence_sufficiency.get("dimension_coverage_rate")),
        weighted_dimension_coverage_rate=state.get(
            "weighted_dimension_coverage_rate",
            evidence_sufficiency.get("weighted_dimension_coverage_rate"),
        ),
        framework_sufficiency_status=str(
            state.get("framework_sufficiency_status", evidence_sufficiency.get("framework_sufficiency_status", ""))
            or ""
        ),
    )
    if dimension_sufficiency["dimension_status_map"]:
        output["dimension_sufficiency"] = dimension_sufficiency
    if state.get("red_flags"):
        output["red_flags"] = user_visible_red_flags(list(state.get("red_flags", []) or []))
    market_reaction_block = _build_market_reaction_block(
        state=state,
        lang=lang,
        numeric_cards=numeric_cards,
    )
    if market_reaction_block is not None:
        output["market_reaction"] = market_reaction_block
    output["view"] = _build_output_view(
        lang=lang,
        task_type=task_type,
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        safety_intent=str(state.get("safety_intent", "normal")),
        summary=summary,
        key_points=output["key_points"],
        numeric_cards=numeric_cards,
        text_cards=text_cards,
        period_query=period_query,
        comparison_basis_line=comparison_basis_line,
        limitations=limitations,
    )
    return output

def _comparison_answer_content(state: AgentState, rows: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    tickers = _ordered_unique_tickers(state, rows)
    if len(tickers) < 2:
        return {
            "conclusion": "缺少可比较公司数据。" if lang == "zh" else "Insufficient companies for comparison.",
            "key_lines": [],
            "structured_rows": [],
        }

    t1, t2 = tickers[0], tickers[1]
    selected = _select_comparison_evidence_rows(state, rows)
    pairs = selected.get("comparable_pairs", [])
    supplement_rows = selected.get("supplement_rows", [])
    key_lines: list[str] = []
    structured_rows: list[dict[str, Any]] = []

    if pairs:
        for pair in pairs[:4]:
            metric = _metric_label(str(pair.get("metric", "")), lang)
            period_type = _period_label(str(pair.get("period_type", "")), lang)
            left = pair.get("left", {})
            right = pair.get("right", {})
            lv = _display_numeric_value(left, lang)
            rv = _display_numeric_value(right, lang)
            period_end = str(left.get("period_end", ""))
            if lang == "zh":
                key_lines.append(
                    f"- {metric}（{period_type}, {period_end}）：{t1}={lv}，{t2}={rv}"
                )
            else:
                key_lines.append(
                    f"- {metric} ({period_type}, {period_end}): {t1}={lv}, {t2}={rv}"
                )
            structured_rows.extend([left, right])

        first = pairs[0]
        left = first.get("left", {})
        right = first.get("right", {})
        metric = _metric_label(str(first.get("metric", "")), lang)
        period_type = _period_label(str(first.get("period_type", "")), lang)
        period_end = str(left.get("period_end", ""))
        v1 = _to_float(left.get("value"))
        v2 = _to_float(right.get("value"))
        if v1 is not None and v2 is not None:
            higher = t1 if v1 >= v2 else t2
            lower = t2 if higher == t1 else t1
            if lang == "zh":
                conclusion = (
                    f"在最新可比{period_type}（{period_end}）的{metric}上，"
                    f"{higher}高于{lower}。"
                )
            else:
                conclusion = (
                    f"In the latest comparable {period_type} ({period_end}), "
                    f"{higher} is higher than {lower} on {metric}."
                )
        else:
            conclusion = (
                f"{t1} 与 {t2} 存在可比口径，但关键值缺失。"
                if lang == "zh"
                else f"{t1} and {t2} have comparable periods, but key values are missing."
            )
    else:
        conclusion = (
            f"{t1} 与 {t2} 当前无共同可比口径（年度同年/季度同截止日），无法形成严格同口径比较。"
            if lang == "zh"
            else (
                f"{t1} and {t2} currently do not share a strict comparable basis "
                "(same annual year / same quarterly period_end), so no strict comparison can be concluded."
            )
        )

    for row in supplement_rows[:6]:
        metric = _metric_label(str(row.get("metric", "")), lang)
        period_type = _period_label(str(row.get("period_type", "")), lang)
        value = _display_numeric_value(row, lang)
        ticker = str(row.get("ticker", ""))
        period_end = str(row.get("period_end", ""))
        if lang == "zh":
            key_lines.append(
                f"- 补充（不可直接比较）：{ticker} {metric}（{period_type}, {period_end}）={value}"
            )
        else:
            key_lines.append(
                f"- Supplement (not directly comparable): {ticker} {metric} ({period_type}, {period_end}) = {value}"
            )
        structured_rows.append(row)

    return {
        "conclusion": conclusion,
        "key_lines": key_lines,
        "structured_rows": _dedupe_rows(structured_rows),
    }

def _summary_key_rows(state: AgentState, limit: int = 6) -> list[dict[str, Any]]:
    rows = _collect_financial_rows(state.get("tool_results", []))
    rows_sorted = sorted(rows, key=lambda r: (r.get("period_end", ""), r.get("period_type", "")), reverse=True)
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for row in rows_sorted:
        key = (str(row.get("ticker", "")), str(row.get("metric", "")), str(row.get("period_type", "")))
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected

def _render_answer_blocks(
    lang: str,
    conclusion: str,
    key_lines: list[str],
    structured_lines: list[str],
    document_lines: list[str],
    risk_lines: list[str],
) -> str:
    if lang == "zh":
        blocks = [
            "结论",
            conclusion or "暂无明确结论。",
            "",
            "关键数据",
            *(key_lines or ["- 当前查询未返回结构化关键数据。"]),
            "",
            "数据来源（结构化）",
            *(structured_lines or ["- 当前无结构化来源。"]),
            "",
            "文档依据",
            *(document_lines or ["- 当前无文档依据。"]),
            "",
            "风险与不足",
            *(risk_lines or ["- 暂无。"]),
        ]
        return "\n".join(blocks).strip()

    blocks = [
        "Conclusion",
        conclusion or "No clear conclusion.",
        "",
        "Key Data",
        *(key_lines or ["- No structured key data returned."]),
        "",
        "Structured Sources",
        *(structured_lines or ["- No structured source used."]),
        "",
        "Document Evidence",
        *(document_lines or ["- No document citation returned."]),
        "",
        "Risks and Limits",
        *(risk_lines or ["- None."]),
    ]
    return "\n".join(blocks).strip()

def _compose_answer_payload(
    state: AgentState,
    cleaned_answer: str,
    lang: str,
    document_citations: list[dict[str, Any]],
) -> dict[str, Any]:
    task_type = state.get("task_type", "fact_qa")
    rows = _collect_financial_rows(state.get("tool_results", []))
    structured_rows: list[dict[str, Any]] = []
    key_lines: list[str] = []

    if task_type == "company_comparison":
        cmp = _comparison_answer_content(state, rows, lang)
        conclusion = str(cmp.get("conclusion", ""))
        key_lines = list(cmp.get("key_lines", []))
        structured_rows = list(cmp.get("structured_rows", []))
    elif task_type == "fact_qa":
        tickers = _ordered_unique_tickers(state, rows)
        if len(tickers) >= 2:
            cmp = _comparison_answer_content(state, rows, lang)
            conclusion = str(cmp.get("conclusion", ""))
            key_lines = list(cmp.get("key_lines", []))
            structured_rows = list(cmp.get("structured_rows", []))
        else:
            preferred = set(state.get("requested_metrics", []))
            rows_sorted = sorted(rows, key=lambda r: r.get("period_end", ""), reverse=True)
            selected_row: dict[str, Any] | None = None
            for row in rows_sorted:
                if preferred and row.get("metric") not in preferred:
                    continue
                selected_row = row
                break
            if selected_row is None and rows_sorted:
                selected_row = rows_sorted[0]

            if selected_row:
                ticker = str(selected_row.get("ticker", ""))
                metric_name = _metric_label(str(selected_row.get("metric", "")), lang)
                period = str(selected_row.get("period_end", ""))
                period_type = _period_label(str(selected_row.get("period_type", "")), lang)
                value = _format_usd_value(selected_row.get("value"), lang)
                if lang == "zh":
                    conclusion = f"{ticker} 在 {period}（{period_type}）的{metric_name}为 {value}。"
                else:
                    conclusion = f"{ticker} {metric_name} at {period} ({period_type}) is {value}."
                key_lines = [f"- {ticker} {metric_name}（{period_type}, {period}）：{value}" if lang == "zh" else f"- {ticker} {metric_name} ({period_type}, {period}): {value}"]
                structured_rows = [selected_row]
            else:
                conclusion = (
                    "当前未检索到可用于该问题的结构化数值，无法给出可追溯结论。"
                    if lang == "zh"
                    else "No structured numeric evidence was retrieved for this question, so no traceable numeric conclusion can be made."
                )
    else:
        data_conclusion = _build_structured_conclusion(state, lang)
        model_summary = _first_sentence(cleaned_answer)
        conclusion = data_conclusion or model_summary or (
            "基于现有证据，结论见下方。"
            if lang == "zh"
            else "Based on current evidence, see details below."
        )
        key_rows = _summary_key_rows(state, limit=6)
        structured_rows = list(key_rows)
        for row in key_rows:
            metric_name = _metric_label(str(row.get("metric", "")), lang)
            value = _format_usd_value(row.get("value"), lang)
            period = str(row.get("period_end", ""))
            period_type = _period_label(str(row.get("period_type", "")), lang)
            if lang == "zh":
                key_lines.append(f"- {row.get('ticker', '')} {metric_name}（{period_type}, {period}）：{value}")
            else:
                key_lines.append(f"- {row.get('ticker', '')} {metric_name} ({period_type}, {period}): {value}")

        if not structured_rows and re.search(r"\d", conclusion):
            conclusion = (
                "基于文档证据可见公司存在关键变化或风险，具体依据见下方文档依据。"
                if lang == "zh"
                else "Document evidence indicates key changes or risks; see document evidence below for details."
            )

    structured_sources = _rows_to_structured_sources(_dedupe_rows(structured_rows))
    structured_lines = _structured_source_lines(structured_sources, lang)
    document_citations = _dedupe_document_citations(document_citations)
    document_lines = _document_evidence_lines(document_citations, lang, limit=8)
    risk_lines = _build_risk_lines(state, lang)

    answer = _render_answer_blocks(
        lang=lang,
        conclusion=conclusion,
        key_lines=key_lines,
        structured_lines=structured_lines,
        document_lines=document_lines,
        risk_lines=risk_lines,
    )
    return {
        "answer": answer,
        "structured_sources": structured_sources,
        "document_citations": document_citations,
    }

def _format_fact_qa_answer(
    state: AgentState,
    cleaned_answer: str,
    lang: str,
    citations: list[dict[str, Any]],
) -> str:
    preferred = set(state.get("requested_metrics", []))
    rows = _collect_financial_rows(state.get("tool_results", []))
    rows = sorted(rows, key=lambda r: r.get("period_end", ""), reverse=True)
    tickers = _ordered_unique_tickers(state, rows)

    # If multiple tickers are present, return a concise side-by-side snapshot.
    if len(tickers) >= 2:
        t1, t2 = tickers[0], tickers[1]
        metric_priority = list(dict.fromkeys((state.get("requested_metrics", []) or []) + ["revenue", "net_income"]))
        selected_metric = ""
        r1: dict[str, Any] | None = None
        r2: dict[str, Any] | None = None
        for metric in metric_priority:
            for period_type in ("quarterly", "annual", None):
                a = _latest_row(rows, t1, metric, period_type if period_type else None)
                b = _latest_row(rows, t2, metric, period_type if period_type else None)
                if a and b:
                    selected_metric = metric
                    r1, r2 = a, b
                    break
            if r1 and r2:
                break

        refs = _build_citation_refs(citations, lang, limit=4)
        if r1 and r2:
            metric_name = _metric_label(selected_metric, lang)
            value1 = _format_usd_value(r1.get("value"), lang)
            value2 = _format_usd_value(r2.get("value"), lang)
            same_period = (
                r1.get("period_end") == r2.get("period_end")
                and r1.get("period_type") == r2.get("period_type")
            )

            v1 = _to_float(r1.get("value"))
            v2 = _to_float(r2.get("value"))
            ratio_text = ""
            if v1 is not None and v2 is not None and min(v1, v2) > 0:
                hi = max(v1, v2)
                lo = min(v1, v2)
                ratio_text = f"{hi / lo:.2f}"

            if lang == "zh":
                if same_period:
                    higher = t1 if (v1 or 0) >= (v2 or 0) else t2
                    lower = t2 if higher == t1 else t1
                    conclusion = (
                        f"结论：在最新可比{_period_label(r1.get('period_type'), lang)}（{r1.get('period_end', '')}），"
                        f"{higher} 的{metric_name}高于 {lower}"
                    )
                    if ratio_text:
                        conclusion += f"（约 {ratio_text} 倍）"
                    conclusion += "。"
                else:
                    conclusion = "结论：这是多公司查询，以下为各公司最新口径数据（日期可能不同）。"
                line_a = (
                    f"{t1} {metric_name}（{_period_label(r1.get('period_type'), lang)}, "
                    f"{r1.get('period_end', '')}）：{value1}。"
                )
                line_b = (
                    f"{t2} {metric_name}（{_period_label(r2.get('period_type'), lang)}, "
                    f"{r2.get('period_end', '')}）：{value2}。"
                )
                source_parts = [
                    f"{t1} {_period_label(r1.get('period_type'), lang)} {r1.get('period_end', '')} {metric_name}",
                    f"{t2} {_period_label(r2.get('period_type'), lang)} {r2.get('period_end', '')} {metric_name}",
                ]
                if refs:
                    source_parts.append(f"文档参考：{refs[0]}")
                source = f"来源：{'；'.join(source_parts)}"
                return "\n".join([conclusion, line_a, line_b, source]).strip()

            if same_period:
                higher = t1 if (v1 or 0) >= (v2 or 0) else t2
                lower = t2 if higher == t1 else t1
                conclusion = (
                    f"Conclusion: In the latest comparable {_period_label(r1.get('period_type'), lang)} "
                    f"({r1.get('period_end', '')}), {higher} has higher {metric_name} than {lower}"
                )
                if ratio_text:
                    conclusion += f" (about {ratio_text}x)"
                conclusion += "."
            else:
                conclusion = "Conclusion: This is a multi-company query; latest-disclosure dates may differ."
            line_a = (
                f"{t1} {metric_name} ({_period_label(r1.get('period_type'), lang)}, "
                f"{r1.get('period_end', '')}): {value1}."
            )
            line_b = (
                f"{t2} {metric_name} ({_period_label(r2.get('period_type'), lang)}, "
                f"{r2.get('period_end', '')}): {value2}."
            )
            source_parts = [
                f"{t1} {_period_label(r1.get('period_type'), lang)} {r1.get('period_end', '')} {metric_name}",
                f"{t2} {_period_label(r2.get('period_type'), lang)} {r2.get('period_end', '')} {metric_name}",
            ]
            if refs:
                source_parts.append(f"filing reference: {refs[0]}")
            source = f"Source: {'; '.join(source_parts)}"
            return "\n".join([conclusion, line_a, line_b, source]).strip()

    selected_row: dict[str, Any] | None = None
    if preferred:
        for row in rows:
            if row.get("metric") in preferred:
                selected_row = row
                break
    if selected_row is None and rows:
        selected_row = rows[0]

    refs = _build_citation_refs(citations, lang, limit=2)
    if selected_row:
        ticker = selected_row.get("ticker", "")
        metric_name = _metric_label(selected_row.get("metric", ""), lang)
        period = selected_row.get("period_end", "")
        period_type = _period_label(selected_row.get("period_type"), lang)
        value = _format_usd_value(selected_row.get("value"), lang)
        if lang == "zh":
            line1 = f"结论：{ticker} 在 {period}（{period_type}）的{metric_name}为 {value}。"
            source = f"{ticker} {period_type} {period} {metric_name}"
            if refs:
                source = f"{source}；文档参考：{refs[0]}"
            line2 = f"来源：{source}"
            return f"{line1}\n{line2}"
        line1 = f"Conclusion: {ticker} {metric_name} at {period} ({period_type}) is {value}."
        source = f"{ticker} {period_type} {period} {metric_name}"
        if refs:
            source = f"{source}; filing reference: {refs[0]}"
        line2 = f"Source: {source}"
        return f"{line1}\n{line2}"

    fallback = _split_sentences(cleaned_answer)
    if lang == "zh":
        line1 = f"结论：{fallback[0] if fallback else '未获取到可用数值。'}"
        line2 = f"来源：{refs[0]}" if refs else "来源：当前工具未返回可追溯引用。"
        return f"{line1}\n{line2}"
    line1 = f"Conclusion: {fallback[0] if fallback else 'No usable numeric data was returned.'}"
    line2 = f"Source: {refs[0]}" if refs else "Source: No traceable citation returned by tools."
    return f"{line1}\n{line2}"

def _format_non_fact_answer(
    state: AgentState,
    cleaned_answer: str,
    lang: str,
    citations: list[dict[str, Any]],
) -> str:
    sentences = _split_sentences(cleaned_answer)
    model_summary = " ".join(sentences[:2]) if sentences else cleaned_answer
    data_conclusion = _build_structured_conclusion(state, lang)
    conclusion = data_conclusion or model_summary
    if lang == "zh" and _is_language_mismatch(conclusion, "zh"):
        conclusion = "基于当前证据，结论如下：公司表现可参考下方关键数据与依据。"
    if lang == "en" and _is_language_mismatch(conclusion, "en"):
        conclusion = "Based on current evidence, please refer to the key data and evidence sections below."
    key_lines = _build_key_data_lines(state, lang, limit=6)
    evidence_lines = [f"- {r}" for r in _build_citation_refs(citations, lang, limit=6)]
    risk_lines = _build_risk_lines(state, lang)

    if lang == "zh":
        blocks = [
            "结论",
            conclusion or "暂无明确结论。",
            "",
            "关键数据",
            *(key_lines or ["- 当前查询未返回结构化关键数据。"]),
            "",
            "依据",
            *(evidence_lines or ["- 当前未返回可引用的文档证据。"]),
            "",
            "风险与不足",
            *risk_lines,
        ]
        return "\n".join(blocks).strip()

    blocks = [
        "Conclusion",
        conclusion or "No clear conclusion.",
        "",
        "Key Data",
        *(key_lines or ["- No structured key data returned in this run."]),
        "",
        "Evidence",
        *(evidence_lines or ["- No document citations returned in this run."]),
        "",
        "Risks and Limits",
        *risk_lines,
    ]
    return "\n".join(blocks).strip()

def _is_language_mismatch(answer_text: str, lang: str) -> bool:
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", answer_text or ""))
    alpha_count = len(re.findall(r"[A-Za-z]", answer_text or ""))
    if lang == "zh":
        total = max(len(answer_text or ""), 1)
        return cjk_count < 10 or (cjk_count / total) < 0.08
    return alpha_count < max(20, cjk_count)

def _enforce_answer_language(answer_text: str, user_query: str, lang: str) -> str:
    if not answer_text or not _is_language_mismatch(answer_text, lang):
        return answer_text

    target_name = "Simplified Chinese" if lang == "zh" else "English"
    prompt = (
        f"Rewrite the following answer into {target_name} only.\n"
        "Do not add or remove facts. Keep all numbers and source labels unchanged.\n\n"
        f"{answer_text}"
    )
    try:
        llm = _get_llm(reasoning=False, temperature=0.0, max_tokens=min(max(len(answer_text) // 2, 300), 1800))
        rewritten = llm.invoke([
            SystemMessage(content="You are a strict financial answer rewriter."),
            HumanMessage(content=prompt),
        ])
        text = re.sub(r"<think>.*?</think>", "", rewritten.content or "", flags=re.DOTALL).strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("language rewrite failed: %s", exc)
    return answer_text


build_phase4_output = _build_phase4_output
render_answer_from_output = _render_answer_from_output
compose_answer_payload = _compose_answer_payload
format_fact_qa_answer = _format_fact_qa_answer
format_non_fact_answer = _format_non_fact_answer
build_market_reaction_block = _build_market_reaction_block
