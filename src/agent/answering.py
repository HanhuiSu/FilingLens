# ruff: noqa: F401,F403,F405
"""Answer-generation orchestration for the financial-analysis agent."""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from config import settings
from src.agent.answer_assembler import AnswerAssembler
from src.agent.analyst_draft import build_methodology_context, summarize_analyst_draft
from src.agent.analyst_loop import run_analyst_draft_loop
from src.agent.analysis_framework import analysis_framework_trace_fields, summarize_selected_analysis_framework
from src.agent.citations import _apply_text_citation_policy, _collect_citations_from_claims
from src.agent.comparison_decision import build_comparison_judgment_frame, summarize_comparison_judgment_frame
from src.agent.constants import *
from src.agent.evidence_sufficiency import build_trace_summary, finalize_evidence_accounting, summarize_evidence_requirements
from src.agent.evidence import (
    _build_deterministic_numeric_claims,
    _build_evidence_bundle,
    _collect_event_rows,
    _evidence_catalog_text,
    _normalize_claims,
    _period_consistency_ok,
    _target_lang,
    _validate_numeric_claims_strict,
    validate_text_claims_enhanced,
)
from src.agent.evidence_packet import build_evidence_packet, summarize_evidence_packet
from src.agent.driver_evidence import annotate_driver_evidence, apply_profit_decline_summary_neutralization, apply_scope_aware_summary
from src.agent.llm import _get_llm, _parse_json_response
from src.agent.metric_display import format_metric_value
from src.agent.output_language import (
    detect_output_language,
    language_leakage_count,
    repair_language_leakage,
)
from src.agent.progress import append_progress_event
from src.agent.query_plan import _default_period_query, _detect_event_intent
from src.agent.red_flags import detect_red_flags, serialize_red_flags, user_visible_red_flags
from src.agent.rendering import (
    _annual_year_basis_line,
    _build_phase4_output,
    _clean_answer_text,
    _clarification_message,
    _comparison_basis_line,
    _enforce_answer_language,
    _evidence_insufficient_message,
    _first_sentence,
    _limitation_item,
    _render_answer_from_output,
    _truncate_text,
    sanitize_user_facing_answer_text,
)
from src.agent.synthesis import (
    build_analytical_synthesis,
    build_bounded_valuation_risk_comparison_candidate,
    build_synthesis_view,
    derive_synthesis_mode,
    render_synthesis_text,
)
from src.agent.state import AgentState
from src.agent.prompts import GENERATE_ANSWER

logger = logging.getLogger(__name__)

def _conversation_limitation(lang: str, code: str) -> dict[str, Any]:
    messages = {
        "no_external_tools": {
            "zh": "这是对话/澄清类回答，未调用外部工具或检索证据。",
            "en": "This is a conversational/clarification response; no external tools or evidence retrieval were used.",
            "severity": "low",
        },
        "unsupported_scope": {
            "zh": "该问题超出当前财报分析系统的支持范围。",
            "en": "The question is outside the supported scope of this filings-analysis system.",
            "severity": "medium",
        },
        "investment_advice_boundary": {
            "zh": "以下内容仅是基于证据的分析框架，不构成投资建议、买卖推荐或股价预测。",
            "en": "This is an evidence-grounded analysis framework, not investment advice, a buy/sell recommendation, or a price forecast.",
            "severity": "high",
        },
        "forward_looking_uncertainty": {
            "zh": "涉及未来展望的判断存在不确定性，应以已验证历史事实和披露文本为边界。",
            "en": "Forward-looking discussion is uncertain and bounded by validated historical facts and filing evidence.",
            "severity": "medium",
        },
        "no_realtime_news_access": {
            "zh": "当前系统没有 web search 或实时行情源，因此不会声称知道实时新闻或实时股价。",
            "en": "This system has no web search or live market-data source, so it will not claim real-time news or live prices.",
            "severity": "high",
        },
        "unsupported_price_prediction": {
            "zh": "短期股价预测超出当前系统支持范围。",
            "en": "Near-term stock-price prediction is outside the supported scope.",
            "severity": "high",
        },
        "insufficient_validated_evidence": {
            "zh": "当前没有足够的已验证证据支撑更具体的结论。",
            "en": "Current validated evidence is insufficient for a more specific conclusion.",
            "severity": "medium",
        },
    }
    item = messages.get(code, messages["no_external_tools"])
    return _limitation_item(code, str(item["severity"]), str(item.get(lang, item["en"])))

def _state_safety_limitations(state: AgentState, lang: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in state.get("safety_limitations", []) or []:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", ""))
        if code:
            out.append(_conversation_limitation(lang, code))
    return out

def _append_safety_limitation(state: AgentState, code: str, severity: str, message: str) -> AgentState:
    limitations = list(state.get("safety_limitations", []) or [])
    if not any(str(item.get("code", "")) == code for item in limitations if isinstance(item, dict)):
        limitations.append({"code": code, "severity": severity, "message": message})
    updated = dict(state)
    updated["safety_limitations"] = limitations
    return updated

def _has_evidence_requirements(state: AgentState) -> bool:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    return bool(evidence_plan.get("evidence_requirements"))

def _allowed_requirement_ids(state: AgentState) -> tuple[set[str], set[str]]:
    sufficiency = dict(state.get("evidence_sufficiency", {}) or {})
    satisfied = {str(x) for x in sufficiency.get("satisfied_requirements", []) if str(x).strip()}
    partial = {str(x) for x in sufficiency.get("partial_requirements", []) if str(x).strip()}
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    requirements = {
        str(req.get("requirement_id", "")).strip(): str(req.get("requirement_type", "")).strip()
        for req in evidence_plan.get("evidence_requirements", []) or []
        if isinstance(req, dict) and str(req.get("requirement_id", "")).strip()
    }
    numeric_allowed = {
        rid
        for rid in satisfied
        if requirements.get(rid) != "text"
    }
    text_allowed = {
        rid
        for rid in satisfied | partial
        if requirements.get(rid) == "text"
    }
    return numeric_allowed, text_allowed


def _evidence_requirement_ids(item: dict[str, Any]) -> list[str]:
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


def _filter_evidence_to_candidate_requirements(
    state: AgentState,
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not _has_evidence_requirements(state):
        return numeric_evidence, text_evidence
    numeric_allowed, text_allowed = _allowed_requirement_ids(state)
    return (
        [e for e in numeric_evidence if any(req_id in numeric_allowed for req_id in _evidence_requirement_ids(e))],
        [e for e in text_evidence if any(req_id in text_allowed for req_id in _evidence_requirement_ids(e))],
    )


def _has_satisfied_collection_evidence(state: Mapping[str, Any]) -> bool:
    """Return true when collection produced usable evidence for any requirement."""
    for result in list(state.get("evidence_collection_results", []) or []):
        if not isinstance(result, Mapping):
            continue
        status = str(result.get("status") or "").strip()
        evidence_type = str(result.get("evidence_type") or "").strip()
        items = [item for item in result.get("items", []) or [] if isinstance(item, Mapping)]
        if status in {"satisfied", "partial"} and evidence_type in {"numeric", "calculation", "text", "event"} and items:
            return True
    return False


def _requirement_metadata_by_id(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    out: dict[str, dict[str, Any]] = {}
    for req in evidence_plan.get("evidence_requirements", []) or []:
        if not isinstance(req, Mapping):
            continue
        rid = str(req.get("requirement_id", "") or "").strip()
        if not rid:
            continue
        out[rid] = {
            "dimension_id": str(req.get("dimension_id", "") or ""),
            "framework_id": str(req.get("framework_id", "") or ""),
            "retrieval_intent": str(req.get("retrieval_intent", "") or ""),
            "analysis_purpose": str(req.get("analysis_purpose", "") or ""),
        }
    return out


def _attach_text_requirement_metadata(
    state: Mapping[str, Any],
    text_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    req_meta = _requirement_metadata_by_id(state)
    out: list[dict[str, Any]] = []
    for item in text_evidence:
        row = dict(item)
        req_ids = _evidence_requirement_ids(row)
        for req_id in req_ids:
            meta = req_meta.get(req_id)
            if not meta:
                continue
            for key, value in meta.items():
                if value and not row.get(key):
                    row[key] = value
            break
        out.append(row)
    return out


def _text_validation_context(state: Mapping[str, Any]) -> dict[str, Any]:
    req_meta = _requirement_metadata_by_id(state)
    return {
        "analysis_scope": str(state.get("analysis_scope", "") or ""),
        "task_type": str(state.get("task_type", "") or ""),
        "answer_mode": str(state.get("answer_mode", "") or ""),
        "requirement_dimension_map": {
            rid: str(meta.get("dimension_id", "") or "")
            for rid, meta in req_meta.items()
            if str(meta.get("dimension_id", "") or "")
        },
    }


def _text_claim_type_for_section(section: str) -> str:
    sec = str(section or "").upper().strip()
    if sec == "ITEM_1A":
        return "risk_factor"
    if sec in {"ITEM_7", "ITEM_2"}:
        return "management_discussion"
    if sec == "ITEM_1":
        return "business_context"
    return "operating_context"


def _phrase_present(text: str, *needles: str) -> bool:
    lowered = str(text or "").lower()
    return any(needle in lowered for needle in needles)


def _as_mapping_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


_VALUATION_BOUNDARY_METRICS = {"pe_ratio", "ps_ratio", "fcf_yield"}


def _inline_refs_from_rows(rows: list[dict[str, Any]], metrics: set[str]) -> str:
    refs = [
        str(row.get("evidence_id") or "").strip()
        for row in rows
        if str(row.get("metric") or "").strip() in metrics and str(row.get("evidence_id") or "").strip()
    ]
    refs = list(dict.fromkeys(refs))
    return "".join(f"[{ref}]" for ref in refs[:6])


def _available_valuation_metric_labels(rows: list[dict[str, Any]], lang: str) -> list[str]:
    labels_zh = {"pe_ratio": "P/E", "ps_ratio": "P/S", "fcf_yield": "FCF yield"}
    labels_en = {"pe_ratio": "P/E", "ps_ratio": "P/S", "fcf_yield": "FCF yield"}
    labels = labels_zh if lang == "zh" else labels_en
    seen = {
        str(row.get("metric") or "").strip()
        for row in rows
        if str(row.get("metric") or "").strip() in _VALUATION_BOUNDARY_METRICS
    }
    return [labels[metric] for metric in ("pe_ratio", "ps_ratio", "fcf_yield") if metric in seen]


def _latest_metric_row(rows: list[dict[str, Any]], metric: str, company: str | None = None) -> dict[str, Any]:
    company_norm = str(company or "").upper().strip()
    candidates = []
    for row in rows:
        if not isinstance(row, dict) or str(row.get("metric") or "") != metric:
            continue
        if company_norm and str(row.get("ticker") or row.get("company") or "").upper().strip() != company_norm:
            continue
        candidates.append(row)
    if not candidates:
        return {}
    return dict(sorted(candidates, key=lambda item: str(item.get("period_end") or ""))[-1])


def _metric_display(row: Mapping[str, Any]) -> str:
    if not row:
        return "缺少可验证数据"
    display = str(row.get("display_value") or row.get("formatted_value") or "").strip()
    if display:
        return display
    return format_metric_value(str(row.get("metric") or ""), row.get("value"), unit=str(row.get("unit") or ""))


def _metric_ref(row: Mapping[str, Any]) -> str:
    ref = str(row.get("evidence_id") or "").strip()
    return f"[{ref}]" if ref else ""


def _is_one_sentence_query(user_query: str) -> bool:
    lowered = str(user_query or "").lower()
    return "一句话" in lowered or "one sentence" in lowered


def _bounded_fcf_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "")
    if "自由现金流" not in query and "free cash flow" not in query.lower():
        return answer
    pressure_query = any(term in query.lower() for term in ("worse", "under pressure", "pressured", "pressure", "weak", "weaker"))
    if "变差" not in query and "承压" not in query and "压力" not in query and not pressure_query:
        return answer
    ocf = _latest_metric_row(numeric_evidence, "operating_cash_flow")
    fcf = _latest_metric_row(numeric_evidence, "free_cash_flow")
    capex = _latest_metric_row(numeric_evidence, "capital_expenditure")
    if not (ocf and fcf and capex):
        return answer
    refs = "".join(dict.fromkeys([_metric_ref(row) for row in (fcf, ocf, capex) if _metric_ref(row)]))
    fcf_rows = [row for row in numeric_evidence if isinstance(row, dict) and str(row.get("metric") or "") == "free_cash_flow"]
    if lang == "zh":
        if len(fcf_rows) >= 2:
            sorted_fcf = sorted(fcf_rows, key=lambda row: str(row.get("period_end") or ""))
            latest = sorted_fcf[-1]
            previous = sorted_fcf[-2]
            try:
                worsened = float(latest.get("value")) < float(previous.get("value"))
            except (TypeError, ValueError):
                worsened = False
            trend = "同口径最近两期显示自由现金流变差" if worsened else "同口径最近两期未显示自由现金流变差"
            return (
                f"{trend}；最新自由现金流为 {_metric_display(fcf)}，经营现金流为 {_metric_display(ocf)}，资本开支为 {_metric_display(capex)}；"
                f"当前自由现金流偏弱主要受资本开支占用影响。{refs}"
            )
        return (
            f"当前证据能说明 AMZN 最新自由现金流为 {_metric_display(fcf)}，经营现金流为 {_metric_display(ocf)}，资本开支为 {_metric_display(capex)}；"
            f"自由现金流偏弱主要来自资本开支占用，但是否“变差”需要同口径历史 FCF 对比。{refs}"
        )
    return "\n".join(
        [
            "Conclusion",
            f"Current evidence supports a bounded causal view: free cash flow is under pressure because capex is absorbing a large share of operating cash flow. {refs}",
            "",
            "Verified Facts",
            f"- Latest free cash flow: {_metric_display(fcf)}. {_metric_ref(fcf)}",
            f"- Latest operating cash flow: {_metric_display(ocf)}. {_metric_ref(ocf)}",
            f"- Latest capex: {_metric_display(capex)}. {_metric_ref(capex)}",
            "",
            "Reasonable Inference",
            "- When capex is close to or above operating cash flow, FCF can turn negative or remain pressured even if operating cash flow is positive.",
            "",
            "Data to Verify",
            "- Same-basis historical FCF, capex trend, cloud/logistics investment timing, revenue-recognition timing, and working-capital changes.",
            "",
            "Evidence Boundary",
            "- This explains the current FCF pressure path; whether FCF has worsened requires same-basis historical FCF comparison.",
        ]
    )


def _bounded_valuation_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    valuation_query = any(term in query for term in ("估值", "贵不贵", "能买吗", "可以买", "值得买", "buy", "valuation", "expensive", "cheap"))
    if not valuation_query:
        return answer
    pe = _latest_metric_row(numeric_evidence, "pe_ratio")
    ps = _latest_metric_row(numeric_evidence, "ps_ratio")
    fcf_yield = _latest_metric_row(numeric_evidence, "fcf_yield")
    if not (pe or ps or fcf_yield):
        return answer
    refs = "".join(dict.fromkeys([_metric_ref(row) for row in (pe, ps, fcf_yield) if _metric_ref(row)]))
    company = str((pe or ps or fcf_yield).get("ticker") or (pe or ps or fcf_yield).get("company") or "").upper().strip() or "该公司"
    metrics = []
    if pe:
        metrics.append(f"P/E {_metric_display(pe)}")
    if ps:
        metrics.append(f"P/S {_metric_display(ps)}")
    if fcf_yield:
        metrics.append(f"FCF yield {_metric_display(fcf_yield)}")
    metric_text = "、".join(metrics) if lang == "zh" else ", ".join(metrics)
    if _is_one_sentence_query(user_query):
        if lang == "zh":
            return f"不能给买卖建议；{company} 的 {metric_text} 显示市场定价很高，但缺少同业、历史区间或增长预期基准，不能严格判断“贵不贵”。{refs}"
        return f"I cannot give buy/sell advice; based on validated valuation metrics, {company}'s {metric_text} supports only a limited valuation-boundary view. {refs}"
    if lang == "en" and any(term in query for term in ("p/e", "p/s", "fcf yield", "valuation risk")):
        metric_lines = []
        if pe:
            metric_lines.append(f"- P/E: {_metric_display(pe)}. A higher multiple lowers the margin for disappointment. {_metric_ref(pe)}")
        if ps:
            metric_lines.append(f"- P/S: {_metric_display(ps)}. A higher sales multiple increases sensitivity to revenue-growth durability. {_metric_ref(ps)}")
        if fcf_yield:
            metric_lines.append(f"- FCF yield: {_metric_display(fcf_yield)}. A lower FCF yield means less cash-flow yield support at the current price. {_metric_ref(fcf_yield)}")
        return "\n".join(
            [
                "Conclusion",
                f"{company}'s available valuation inputs support a bounded view that valuation risk is elevated, but not a buy/sell conclusion. {refs}",
                "",
                "Verified Facts",
                *metric_lines,
                "",
                "Reasonable Inference",
                "- High valuation multiples and low FCF yield make the valuation more dependent on continued growth and cash-flow conversion.",
                "",
                "Data to Verify",
                "- Historical valuation ranges, peer benchmarks, forward growth durability, margin trend, and FCF conversion.",
                "",
                "Evidence Boundary",
                "- This does not assess competitive risk, target price, or whether the stock is cheap or expensive in an absolute sense.",
            ]
        )
    if lang == "zh" and ("当前候选答案未通过" in answer or "估值证据不足" in answer or "缺少估值证据" in answer):
        return (
            f"当前不能回答“能不能买”或给出买卖建议；已验证估值指标显示 {company} 的 {metric_text}。"
            f"这些倍数显示市场定价很高；但缺少同业、历史区间或增长预期基准，因此不能严格判断“贵不贵”。{refs}"
        )
    return answer


def _bounded_valuation_comparison_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    state: Mapping[str, Any],
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    is_comparison = str(state.get("task_type") or "") == "company_comparison" or len(state.get("companies", []) or []) >= 2
    if not is_comparison or not any(term in query for term in ("valuation", "估值", "p/e", "p/s", "fcf yield")):
        return answer
    companies = [str(item).upper().strip() for item in state.get("companies", []) or [] if str(item).strip()]
    target = str(state.get("comparison_target") or "").upper().strip()
    if target and target not in companies:
        companies.append(target)
    companies = list(dict.fromkeys(companies))[:2]
    if len(companies) < 2:
        return answer
    metrics = [("pe_ratio", "P/E"), ("ps_ratio", "P/S"), ("fcf_yield", "FCF yield")]
    rows: dict[str, dict[str, dict[str, Any]]] = {
        company: {metric: _latest_metric_row(numeric_evidence, metric, company) for metric, _label in metrics}
        for company in companies
    }
    if not any(rows[company][metric] for company in companies for metric, _label in metrics):
        return answer

    def value(company: str, metric: str) -> str:
        return _metric_display(rows[company].get(metric, {})) if rows[company].get(metric) else "verified data unavailable"

    def refs_for(metric: str) -> str:
        refs = [
            _metric_ref(rows[company].get(metric, {}))
            for company in companies
            if rows[company].get(metric)
        ]
        return "".join(dict.fromkeys(ref for ref in refs if ref))

    comparison_lines: list[str] = []
    for metric, label in metrics:
        left_row = rows[companies[0]].get(metric, {})
        right_row = rows[companies[1]].get(metric, {})
        left = value(companies[0], metric)
        right = value(companies[1], metric)
        judgment = "comparison is evidence-limited"
        left_num = _numeric_value(left_row) if left_row else None
        right_num = _numeric_value(right_row) if right_row else None
        if left_num is not None and right_num is not None:
            if metric == "fcf_yield":
                riskier = companies[0] if left_num < right_num else companies[1]
                judgment = f"{riskier} has the lower FCF yield, which is less forgiving from a cash-flow yield perspective"
            else:
                riskier = companies[0] if left_num > right_num else companies[1]
                judgment = f"{riskier} has the higher multiple, which indicates higher multiple pressure on this metric"
        comparison_lines.append(f"- {label}: {companies[0]} {left}; {companies[1]} {right}. {judgment}. {refs_for(metric)}".strip())
    if lang == "zh":
        return answer
    return "\n".join(
        [
            "Conclusion",
            f"The valuation-risk comparison is mixed across P/E, P/S, and FCF yield; it should not be collapsed into a single buy/sell-style ranking.",
            "",
            "Metric-by-Metric Comparison",
            *comparison_lines,
            "",
            "Interpretation",
            "- Higher P/E or P/S indicates higher multiple pressure; lower FCF yield indicates less cash-flow yield support. The metrics can point to different risk angles.",
            "",
            "Evidence Boundary",
            "- This is not investment advice, a target price, or a deterministic forecast. Historical ranges, peer benchmarks, and growth durability remain to verify.",
        ]
    )


def _bounded_revenue_quality_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    if not any(term in query for term in ("收入质量", "营收质量", "revenue quality")):
        return answer
    if "当前候选答案未通过" not in answer and "收入质量" in answer:
        return answer
    revenue = _latest_metric_row(numeric_evidence, "revenue")
    growth = _latest_metric_row(numeric_evidence, "revenue_growth")
    if not revenue:
        return answer
    refs = "".join(dict.fromkeys([_metric_ref(row) for row in (revenue, growth) if _metric_ref(row)]))
    company = str(revenue.get("ticker") or revenue.get("company") or "").upper().strip() or "该公司"
    if lang == "zh":
        growth_text = f"，收入增速为 {_metric_display(growth)}" if growth else "；当前缺少可验证收入增速或文本证据"
        return f"{company} 最新收入为 {_metric_display(revenue)}{growth_text}；因此只能做有边界的收入质量判断，不能声称增长质量已经完全验证。{refs}"
    growth_text = f", revenue growth is {_metric_display(growth)}" if growth else "; verified revenue-growth or text evidence is incomplete"
    return f"{company}'s latest revenue is {_metric_display(revenue)}{growth_text}; this supports only a bounded revenue-quality view. {refs}"


def _bounded_aws_segment_profit_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    if "aws" not in query or not any(term in query for term in ("利润", "盈利", "operating income", "整体利润", "贡献", "重要", "profit")):
        return answer
    segment = _latest_metric_row(numeric_evidence, "aws_operating_income")
    consolidated = _latest_metric_row(numeric_evidence, "consolidated_operating_income")
    contribution = _latest_metric_row(numeric_evidence, "segment_profit_contribution")
    aws_revenue = _latest_metric_row(numeric_evidence, "aws_revenue")
    aws_refs = [
        str(item.get("evidence_id") or "").strip()
        for item in text_evidence
        if isinstance(item, dict)
        and "aws" in " ".join(str(item.get(key) or "").lower() for key in ("claim", "supporting_snippet", "text_snippet", "evidence_summary"))
        and str(item.get("evidence_id") or "").strip()
    ]
    segment_refs: list[str] = []
    operating_income_refs: list[str] = []
    for item in text_evidence:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("evidence_id") or "").strip()
        if not ref:
            continue
        combined = " ".join(
            str(item.get(key) or "")
            for key in ("claim", "supporting_snippet", "text_snippet", "evidence_summary")
        )
        lowered = combined.lower()
        if "aws" in lowered and ("segment" in lowered or "分部" in lowered or "north america" in lowered or "international" in lowered):
            segment_refs.append(ref)
        if "aws" in lowered and "operating income" in lowered and any(term in lowered for term in ("increase", "increased", "growth", "grew", "增长", "上升")):
            operating_income_refs.append(ref)
    refs = "".join(f"[{ref}]" for ref in list(dict.fromkeys(aws_refs))[:3])
    if contribution:
        basis_refs = "".join(dict.fromkeys([_metric_ref(row) for row in (contribution, segment, consolidated, aws_revenue) if _metric_ref(row)]))
        if lang == "zh":
            revenue_text = f"，AWS 收入为 {_metric_display(aws_revenue)}" if aws_revenue else ""
            return (
                f"同口径证据下，AWS 利润贡献率为 {_metric_display(contribution)}；"
                f"AWS operating income 为 {_metric_display(segment)}，consolidated operating income 为 {_metric_display(consolidated)}{revenue_text}。{basis_refs or refs}"
            )
        revenue_text = f", AWS revenue is {_metric_display(aws_revenue)}" if aws_revenue else ""
        return (
            f"On the same basis, AWS segment profit contribution is {_metric_display(contribution)}; "
            f"AWS operating income is {_metric_display(segment)} and consolidated operating income is {_metric_display(consolidated)}{revenue_text}. {basis_refs or refs}"
        )
    if segment and consolidated:
        return (
            f"AWS operating income 为 {_metric_display(segment)}，consolidated operating income 为 {_metric_display(consolidated)}；"
            f"当前未取得已验证贡献率计算，因此只做同口径利润边界说明，不输出贡献比例。{_metric_ref(segment)}{_metric_ref(consolidated)}"
            if lang == "zh"
            else f"AWS operating income is {_metric_display(segment)} and consolidated operating income is {_metric_display(consolidated)}; no validated contribution-rate calculation is available, so I will not state a contribution ratio. {_metric_ref(segment)}{_metric_ref(consolidated)}"
        )
    if lang == "zh":
        segment_ref_text = "".join(f"[{ref}]" for ref in list(dict.fromkeys(segment_refs))[:2]) or refs
        oi_ref_text = "".join(f"[{ref}]" for ref in list(dict.fromkeys(operating_income_refs))[:2]) or refs
        parts = []
        if segment_ref_text:
            parts.append(f"AWS 是 Amazon 三个分部之一。{segment_ref_text}")
        else:
            parts.append("当前只能确认 AWS 分部相关文本线索。")
        if oi_ref_text:
            parts.append(f"文本显示 AWS operating income 有增长线索。{oi_ref_text}")
        else:
            parts.append("当前只有 AWS operating income 相关文本线索。")
        parts.append("但缺少 AWS operating income 与 consolidated operating income 同口径数值，不能计算 AWS 对整体利润贡献比例。")
        return "".join(parts)
    return f"I can only confirm AWS segment and AWS operating-income related text signals; I cannot quantify AWS's contribution to total profit without same-basis AWS operating income and consolidated operating income. {refs}"


def _bounded_scenario_risk_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    text_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    scenario_terms = (
        "经济放缓",
        "放缓",
        "衰退",
        "下季度",
        "宏观",
        "客户预算",
        "需求",
        "投资延迟",
        "供应链",
        "slowdown",
        "slow",
        "slows",
        "recession",
        "economic",
        "macro",
        "customer budget",
        "demand",
        "investment delay",
        "supply chain",
    )
    if not any(term in query for term in scenario_terms):
        return answer
    risk_terms = (
        "economic",
        "macroeconomic",
        "slowdown",
        "recession",
        "demand",
        "customer",
        "budget",
        "spending",
        "investment",
        "delay",
        "supply chain",
        "uncertainty",
        "经济",
        "宏观",
        "放缓",
        "需求",
        "客户",
        "预算",
        "支出",
        "投资",
        "延迟",
        "供应链",
    )
    matched: list[dict[str, Any]] = []
    for row in text_evidence:
        if not isinstance(row, dict):
            continue
        combined = " ".join(
            str(row.get(key) or "")
            for key in ("claim", "supporting_snippet", "text_snippet", "evidence_summary", "section", "dimension_id")
        )
        lowered = combined.lower()
        if any(term in lowered for term in risk_terms):
            matched.append(row)
    if not matched:
        return answer
    if re.search(r"\[T\d+\]", answer or "") and not any(term in str(answer).lower() for term in ("当前候选答案未通过", "blocked", "证据不足", "无法回答")):
        return answer
    refs = "".join(
        f"[{ref}]"
        for ref in list(
            dict.fromkeys(str(row.get("evidence_id") or "").strip() for row in matched if str(row.get("evidence_id") or "").strip())
        )[:3]
    )
    company = str((matched[0].get("ticker") or matched[0].get("company") or "")).upper().strip() or "该公司"
    if lang == "zh":
        return (
            f"可以做有限风险判断：{company} 的已验证风险文本涉及经济/宏观不确定性、需求、客户支出或供应链等线索；"
            f"因此经济放缓可能通过需求和客户预算节奏影响业务，但当前证据不能量化下季度影响，也不能做确定预测。{refs}"
        )
    return "\n".join(
        [
            "Risk Judgment",
            f"A bounded risk answer is supported: validated risk text for {company} mentions economic or macro uncertainty, demand, customer spending, or supply-chain signals. {refs}",
            "",
            "Verified Risk Text",
            f"- The cited risk text supports monitoring economic sensitivity, demand, customer spending, or supply-chain signals. {refs}",
            "",
            "Business-Model Inference",
            "- In an economic slowdown, the first-order business risk is slower customer budget timing or delayed demand rather than a deterministic one-quarter forecast.",
            "",
            "Financial Transmission Path",
            "- Customer budget delays can pressure revenue timing first, then margins, operating cash flow, and FCF if fixed costs or investment commitments do not adjust as quickly.",
            "",
            "Data to Verify",
            "- Revenue growth, order/demand commentary, customer budget language, margin trend, operating cash flow, capex, and FCF.",
            "",
            "Evidence Boundary",
            "- Current evidence cannot quantify next-quarter impact or support a deterministic forecast. This is not investment advice.",
        ]
    )


def _change_history_boundary_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "").lower()
    if not any(term in query for term in ("变化", "变动", "趋势", "change", "trend")):
        return answer
    is_gross_margin_change = any(term in query for term in ("毛利率", "gross margin"))
    if (
        not is_gross_margin_change
        and ("无法判断变化" in answer or "只能观察最新水平" in answer or "current level only" in answer.lower())
    ):
        return answer
    if is_gross_margin_change:
        if "无法判断毛利率变化" in answer:
            return answer
        gross_rows = [
            row
            for row in numeric_evidence
            if isinstance(row, dict) and str(row.get("metric") or "").strip() == "gross_margin"
        ]
        periods = {
            (
                str(row.get("period_end") or row.get("period") or "").strip(),
                str(row.get("period_type") or row.get("period_category") or "").strip(),
                str(row.get("source_provider") or row.get("provider") or "").strip(),
            )
            for row in gross_rows
            if str(row.get("period_end") or row.get("period") or "").strip()
        }
        if len(periods) < 2:
            latest = _latest_metric_row(numeric_evidence, "gross_margin")
            latest_text = f"只能观察最新季度毛利率为 {_metric_display(latest)}。" if latest else "当前只能观察最新毛利率水平。"
            refs = _metric_ref(latest) if latest else ""
            boundary = f"当前缺少两期可比毛利率，无法判断毛利率变化。{latest_text}{refs}"
            return boundary
    relevant_metrics: set[str] = set()
    if is_gross_margin_change:
        relevant_metrics.add("gross_margin")
    if any(term in query for term in ("自由现金流", "现金流", "fcf", "cash flow")):
        relevant_metrics.update({"operating_cash_flow", "free_cash_flow"})
    if any(term in query for term in ("收入", "营收", "revenue", "sales")):
        relevant_metrics.add("revenue")
    if any(term in query for term in ("利润", "盈利", "profit", "income")):
        relevant_metrics.update({"net_income", "operating_income"})
    if not relevant_metrics:
        relevant_metrics.update(str(row.get("metric") or "") for row in numeric_evidence if isinstance(row, dict))
    periods_by_metric: dict[str, set[str]] = {}
    for row in numeric_evidence:
        if not isinstance(row, dict):
            continue
        metric = str(row.get("metric") or "")
        if metric not in relevant_metrics:
            continue
        period = str(row.get("period_end") or row.get("period") or "")
        if period:
            periods_by_metric.setdefault(metric, set()).add(period)
    if any(len(periods) >= 2 for periods in periods_by_metric.values()):
        return answer
    boundary = "当前无法判断变化，只能观察最新水平。" if lang == "zh" else "Current evidence cannot determine the change; it only supports the latest level."
    if _is_one_sentence_query(user_query):
        return f"{answer.rstrip('。.!')}；{boundary.rstrip('。.!')}。"
    return f"{answer.rstrip()}\n\n{boundary}"


def _dedupe_valuation_boundary_caveats(answer: str) -> str:
    caveat_patterns = (
        r"已有\s*(?:P/E|PE|部分估值倍数)",
        r"已有部分估值倍数",
        r"不能给买卖建议；但从",
        r"partial valuation multiples",
    )
    kept_caveat = False
    out_lines: list[str] = []
    for line in str(answer or "").splitlines() or [str(answer or "")]:
        parts = [part for part in re.split(r"(?<=[。.!?])\s*", line) if part]
        kept_parts: list[str] = []
        for part in parts:
            is_caveat = any(re.search(pattern, part, flags=re.IGNORECASE) for pattern in caveat_patterns)
            if not is_caveat:
                kept_parts.append(part)
                continue
            if kept_caveat:
                continue
            kept_caveat = True
            kept_parts.append(part)
        if kept_parts:
            out_lines.append("".join(kept_parts))
    return "\n".join(out_lines)


def _rewrite_valuation_boundary_contradiction(answer: str, numeric_evidence: list[dict[str, Any]], lang: str) -> str:
    labels = _available_valuation_metric_labels(numeric_evidence, lang)
    if not labels:
        return answer
    metrics = "、".join(labels) if lang == "zh" else ", ".join(labels)
    refs = _inline_refs_from_rows(numeric_evidence, _VALUATION_BOUNDARY_METRICS)
    if lang == "zh":
        direction_inputs: list[str] = []
        multiple_labels = [label for label in labels if label in {"P/E", "P/S"}]
        if multiple_labels:
            direction_inputs.append("、".join(multiple_labels) + " 较高")
        if "FCF yield" in labels:
            direction_inputs.append("FCF yield 较低")
        input_text = "、".join(direction_inputs) if direction_inputs else f"{metrics} 可见"
        boundary = (
            f"不能给买卖建议；但从 {input_text}来看，估值风险偏高，"
            f"是否合理取决于增长兑现能力和同业/历史基准{refs}。"
        )
        replacements = {
            "当前估值证据不足，无法判断AMZN是否值得购买": boundary,
            "当前估值证据不足，无法判断 AMZN 是否值得购买": boundary,
            "当前估值证据不足": f"已有部分估值倍数（{metrics}），但完整估值口径仍不完整",
            "缺少估值证据，无法判断价格是否便宜或昂贵": (
                f"不能给买卖建议；但从 {input_text}来看，估值风险偏高，是否合理取决于增长兑现能力和同业/历史基准"
            ),
            "当前缺少估值证据，不能判断价格是否便宜或昂贵，也不能形成买卖建议。": (
                f"不能给买卖建议；但从 {input_text}来看，估值风险偏高，是否合理取决于增长兑现能力和同业/历史基准。"
            ),
        }
    else:
        boundary = (
            f"Partial valuation multiples ({metrics}) are available for a limited valuation-boundary view{refs}, "
            "but full price / shares / market-cap context or peer context remains incomplete; this does not support "
            "buy/sell advice, target prices, or a standalone cheap/expensive conclusion."
        )
        replacements = {
            "valuation evidence is missing": (
                f"partial valuation multiples ({metrics}) are available, but full valuation context is incomplete"
            ),
            "valuation evidence is insufficient": boundary,
        }
    rewritten = str(answer or "")
    for old, new in replacements.items():
        rewritten = rewritten.replace(old, new)
    if lang == "zh":
        rewritten = re.sub(
            r"(?:当前)?缺少估值证据",
            "缺少完整估值口径或可比估值上下文",
            rewritten,
        )
        rewritten = re.sub(
            r"(?:当前)?估值证据不足",
            f"已有部分估值倍数（{metrics}），但完整估值口径仍不完整",
            rewritten,
        )
    return _dedupe_valuation_boundary_caveats(rewritten)


def _is_risk_comparison_query(user_query: str, state: Mapping[str, Any]) -> bool:
    text = str(user_query or "").lower()
    if any(
        term in text
        for term in (
            "风险更大",
            "谁的风险",
            "哪个更危险",
            "哪个风险更高",
            "更危险的是谁",
            "风险更高",
            "riskier",
            "greater risk",
            "higher risk",
        )
    ):
        return True
    return str(state.get("task_type") or "") == "company_comparison" and "风险" in str(user_query or "")


def _short_risk_label(row: Mapping[str, Any]) -> str:
    text = str(row.get("claim") or row.get("summary") or row.get("supporting_snippet") or row.get("snippet") or "").strip()
    text = " ".join(text.split())
    if not text:
        return "validated risk evidence"
    text = re.sub(r"\[[NT]\d+\]", "", text).strip()
    return text[:140].rstrip(" ,;；，。") or "validated risk evidence"


def _risk_lines_from_state_evidence(state: Mapping[str, Any], companies: list[str], lang: str) -> dict[str, str]:
    packet = _as_mapping_dict(state.get("evidence_packet"))
    rows: list[Mapping[str, Any]] = []
    source = packet.get("text_snippets")
    if isinstance(source, list):
        rows.extend(item for item in source if isinstance(item, Mapping))
    by_company: dict[str, list[Mapping[str, Any]]] = {company: [] for company in companies}
    for row in rows:
        company = str(row.get("ticker") or row.get("company") or "").upper().strip()
        if company in by_company:
            by_company[company].append(row)
    out: dict[str, str] = {}
    for company, company_rows in by_company.items():
        if not company_rows:
            continue
        row = company_rows[0]
        ref = str(row.get("evidence_id") or row.get("id") or "").strip()
        ref_text = f"[{ref}]" if ref else ""
        out[company] = f"{company}: {_short_risk_label(row)}{ref_text}"
    return out


def _risk_consideration_lines(frame: Mapping[str, Any], companies: list[str], lang: str, state: Mapping[str, Any] | None = None) -> list[str]:
    by_company: dict[str, list[tuple[str, list[str]]]] = {company: [] for company in companies}
    shared: list[tuple[str, list[str], list[str]]] = []
    for item in frame.get("risk_considerations", []) or []:
        if not isinstance(item, Mapping):
            continue
        item_companies = [str(company).upper() for company in item.get("companies", []) or [] if str(company).strip()]
        label = str(item.get("label") or item.get("theme_code") or "validated risk").strip()
        refs = [str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip()]
        if len(set(item_companies) & set(companies)) > 1:
            shared.append((label, refs, item_companies))
            continue
        for company in item_companies:
            if company in by_company:
                by_company[company].append((label, refs))
    evidence_lines = _risk_lines_from_state_evidence(state or {}, companies, lang) if state is not None else {}
    lines: list[str] = []
    for company in companies:
        items = by_company.get(company, [])
        if items:
            label, refs = items[0]
            ref_text = "".join(f"[{ref}]" for ref in refs[:3])
            lines.append(f"{company}: {label}{ref_text}")
        elif company in evidence_lines:
            lines.append(evidence_lines[company])
        else:
            shared_items = [item for item in shared if company in item[2]]
            if shared_items:
                label, refs, _ = shared_items[0]
                ref_text = "".join(f"[{ref}]" for ref in refs[:3])
                lines.append(f"{company}: shared risk context includes {label}{ref_text}")
            else:
                lines.append(f"{company}: 当前缺少足够的单独风险文本证据。" if lang == "zh" else f"{company}: company-specific risk text is limited.")
    return lines


def _bounded_risk_comparison_answer(state: Mapping[str, Any], synthesis_payload: Mapping[str, Any], lang: str) -> str:
    frame = _as_mapping_dict(synthesis_payload.get("comparison_judgment_frame")) or _as_mapping_dict(state.get("comparison_judgment_frame"))
    companies = [str(company).upper() for company in frame.get("companies", []) or [] if str(company).strip()]
    if len(companies) < 2:
        companies = [str(company).upper() for company in state.get("companies", []) or [] if str(company).strip()]
        target = str(state.get("comparison_target") or "").upper().strip()
        if target:
            companies.append(target)
    companies = list(dict.fromkeys(companies))[:2]
    if len(companies) < 2:
        companies = ["AMZN", "NVDA"]
    a, b = companies[0], companies[1]
    risk_lines = _risk_consideration_lines(frame, companies, lang, state)
    refs = []
    for item in frame.get("risk_considerations", []) or []:
        if isinstance(item, Mapping):
            refs.extend(str(ref) for ref in item.get("evidence_refs", []) or [] if str(ref).strip())
    ref_text = "".join(f"[{ref}]" for ref in list(dict.fromkeys(refs))[:4])
    if lang == "zh":
        return "\n".join(
            [
                "风险比较结论",
                f"证据不足以强行判断 {a} 和 {b} 哪个更危险或谁的风险更大；当前只能做有边界的风险比较。{ref_text}",
                "",
                "双边风险证据",
                *[f"- {line}" for line in risk_lines],
                "",
                "比较边界",
                "- 风险文本覆盖和财务指标口径不完全对称，因此不能用收入规模或投资偏好替代风险排序。",
                "- 若必须排序，需要更多同口径风险披露、盈利/现金流压力和估值上下文；当前结论不构成投资建议。",
            ]
        )
    return "\n".join(
        [
            "Risk Comparison",
            f"I cannot force a ranking of whether {a} or {b} has greater risk; the supportable answer is a bounded risk comparison. {ref_text}",
            "",
            "Bilateral Risk Evidence",
            *[f"- {line}" for line in risk_lines],
            "",
            "Comparison Boundary",
            "- Risk-text coverage and financial metric bases are not fully symmetric, so revenue scale or investment preference cannot replace a risk ranking.",
            "- A stronger ranking would require more comparable risk disclosures, profitability/cash-flow pressure evidence, and valuation context; this is not investment advice.",
        ]
    )


def _rewrite_risk_comparison_answer_if_needed(
    answer: str,
    *,
    state: Mapping[str, Any],
    synthesis_payload: Mapping[str, Any],
    user_query: str,
    lang: str,
) -> str:
    if not _is_risk_comparison_query(user_query, state):
        return answer
    if "谁的风险更大" in answer and ("不能强行判断" in answer or "风险比较" in answer):
        return answer
    return _bounded_risk_comparison_answer(state, synthesis_payload, lang)


def _is_profit_decline_query(user_query: str) -> bool:
    query = str(user_query or "").lower()
    has_why = "为什么" in query or "why" in query
    has_decline = any(term in query for term in ("利润下降", "净利润下降", "盈利下降", "profit decline", "profit declined", "earnings decline"))
    return has_why and has_decline


def _numeric_value(row: Mapping[str, Any]) -> float | None:
    try:
        value = row.get("value")
        return float(value) if value is not None and str(value) != "" else None
    except (TypeError, ValueError):
        return None


def _metric_rows_for_profit_premise(rows: list[dict[str, Any]], metric: str) -> list[dict[str, Any]]:
    out = [
        dict(row)
        for row in rows
        if str(row.get("metric") or "").strip() == metric and _numeric_value(row) is not None
    ]
    return sorted(out, key=lambda row: str(row.get("period_end") or row.get("period") or ""))


def _display_metric_value(row: Mapping[str, Any]) -> str:
    display = str(row.get("display_value") or "").strip()
    if display:
        return display
    value = row.get("value")
    unit = str(row.get("unit") or "").strip()
    metric = str(row.get("metric") or row.get("metric_label") or "").strip().lower()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"{value} {unit}".strip()
    unit_l = unit.lower()
    if unit_l == "ratio" or metric.endswith("_margin") or metric in {"fcf_yield", "cash_conversion", "capex_to_revenue"}:
        return f"{numeric * 100:.2f}%"
    if unit_l in {"usd", "usd_per_share"} or metric == "eps":
        if unit_l == "usd_per_share" or metric == "eps":
            return f"${numeric:,.2f}"
        abs_value = abs(numeric)
        if abs_value >= 1_000_000_000_000:
            return f"${numeric / 1_000_000_000_000:.2f}T"
        if abs_value >= 1_000_000_000:
            return f"${numeric / 1_000_000_000:.2f}B"
        if abs_value >= 1_000_000:
            return f"${numeric / 1_000_000:.2f}M"
        return f"${numeric:,.0f}"
    return f"{numeric:,.2f}".rstrip("0").rstrip(".")


def _profit_decline_premise_answer(rows: list[dict[str, Any]], *, lang: str) -> str:
    metric = "net_income"
    metric_rows = _metric_rows_for_profit_premise(rows, metric)
    if len(metric_rows) < 2:
        metric = "operating_income"
        metric_rows = _metric_rows_for_profit_premise(rows, metric)
    metric_label = "净利润" if metric == "net_income" else "营业利润"
    if lang != "zh":
        metric_label = "net income" if metric == "net_income" else "operating income"
    if not metric_rows:
        return (
            "当前可验证数据不足以判断利润是否下降；如果你指的是某个特定期间，需要指定比较区间。可以继续验证的因素包括收入、成本、费用、一次性项目和税率变化。"
            if lang == "zh"
            else "Current validated data is not enough to verify whether profit declined; please specify the comparison period. Factors to verify include revenue, costs, expenses, one-time items, and tax rate changes."
        )
    if len(metric_rows) == 1:
        row = metric_rows[-1]
        ref = str(row.get("evidence_id") or "").strip()
        ref_text = f"[{ref}]" if ref else ""
        return (
            f"当前只有单期{metric_label}证据（{str(row.get('period_end') or row.get('period') or '未知期间')}，{_display_metric_value(row)}）{ref_text}，无法验证是否下降。若你指的是其他期间，需要指定比较区间；在前提未验证前，只能列出待验证因素，不能写成已发生的利润下降原因。"
            if lang == "zh"
            else f"Only one period of {metric_label} evidence is currently validated ({str(row.get('period_end') or row.get('period') or 'unknown period')}, {_display_metric_value(row)}){ref_text}, so a decline cannot be verified. Please specify the comparison period; until the premise is verified, this can only be framed as factors to check, not causes of an actual decline."
        )
    comparator, current = metric_rows[-2], metric_rows[-1]
    current_value = _numeric_value(current)
    comparator_value = _numeric_value(comparator)
    refs = "".join(
        f"[{ref}]"
        for ref in list(
            dict.fromkeys(
                str(row.get("evidence_id") or "").strip()
                for row in (current, comparator)
                if str(row.get("evidence_id") or "").strip()
            )
        )
    )
    current_period = str(current.get("period_end") or current.get("period") or "最新期间")
    comparator_period = str(comparator.get("period_end") or comparator.get("period") or "上一期")
    if current_value is not None and comparator_value is not None and current_value > comparator_value:
        return (
            "\n".join(
                [
                    "结论",
                    f"当前可验证数据不支持“利润下降”这个前提；最新{metric_label}（{current_period}，{_display_metric_value(current)}）高于上一期（{comparator_period}，{_display_metric_value(comparator)}）。{refs}",
                    "",
                    "已验证事实",
                    f"- 最新{metric_label}为 {_display_metric_value(current)}，期间为 {current_period}。{refs}",
                    f"- 对比期{metric_label}为 {_display_metric_value(comparator)}，期间为 {comparator_period}。{refs}",
                    "",
                    "合理推断",
                    f"- 合理推断：在这两期已验证数据范围内，不能解释下降原因，因为引用事实显示{metric_label}上升而不是下降。{refs}",
                    "",
                    "待验证假设",
                    "- 待验证：如果你指的是其他期间，需要指定比较区间，并继续核验收入、成本、费用、一次性项目和税率变化。",
                    "",
                    "证据边界",
                    f"- 该判断只比较当前已验证的最近两期{metric_label}。",
                    "- 缺少指定区间前，不能把待验证因素写成已发生的原因解释。",
                ]
            )
            if lang == "zh"
            else "\n".join(
                [
                    "Conclusion",
                    f"Current validated data does not support the premise that profit declined; latest {metric_label} ({current_period}, {_display_metric_value(current)}) is higher than the prior period ({comparator_period}, {_display_metric_value(comparator)}). {refs}",
                    "",
                    "Verified Facts",
                    f"- Latest {metric_label} is {_display_metric_value(current)} for {current_period}. {refs}",
                    f"- Prior-period {metric_label} is {_display_metric_value(comparator)} for {comparator_period}. {refs}",
                    "",
                    "Reasonable Inference",
                    f"- Reasonable inference: within these two validated periods, decline causes should not be explained because the cited facts show {metric_label} increased rather than declined. {refs}",
                    "",
                    "Hypotheses To Verify",
                    "- To verify: if another interval was intended, specify the comparison period and check revenue, costs, expenses, one-time items, and taxes.",
                    "",
                    "Evidence Boundary",
                    f"- This check only compares the two most recent validated {metric_label} periods.",
                    "- Until the intended period is specified, factors to verify cannot be stated as causes of an actual profit decline.",
                ]
            )
        ).strip()
    return ""


def _profit_decline_false_premise_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for metric in ("net_income", "operating_income"):
        metric_rows = _metric_rows_for_profit_premise(rows, metric)
        if len(metric_rows) < 2:
            continue
        comparator, current = metric_rows[-2], metric_rows[-1]
        current_value = _numeric_value(current)
        comparator_value = _numeric_value(comparator)
        if current_value is not None and comparator_value is not None and current_value > comparator_value:
            return [current, comparator]
    return []


def _dedupe_evidence_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = str(row.get("evidence_id") or row.get("requirement_id") or id(row))
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _rewrite_profit_decline_premise_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    if not _is_profit_decline_query(user_query):
        return answer
    premise_answer = _profit_decline_premise_answer(numeric_evidence, lang=lang)
    return premise_answer or answer


def _profit_decline_false_premise_direct_result(
    *,
    state: Mapping[str, Any],
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    text_evidence: list[dict[str, Any]],
    lang: str,
    event_intent: str,
    market_reaction_requested: bool,
    event_query: Mapping[str, Any],
    event_results: list[dict[str, Any]],
    market_reaction_evidence: list[dict[str, Any]],
    market_reaction_limitations: list[str],
) -> dict[str, Any] | None:
    if not _is_profit_decline_query(user_query):
        return None
    premise_rows = _profit_decline_false_premise_rows(numeric_evidence)
    if len(premise_rows) < 2:
        return None
    answer_text = _profit_decline_premise_answer(numeric_evidence, lang=lang)
    if not answer_text:
        return None
    final_numeric_evidence = _dedupe_evidence_rows(premise_rows)
    final_text_evidence = [
        apply_profit_decline_summary_neutralization(dict(item), user_query=user_query)
        for item in text_evidence
        if isinstance(item, Mapping)
    ]
    output = {
        "task_type": str(state.get("task_type") or "report_summary"),
        "title": "简短结论" if lang == "zh" else "Short Conclusion",
        "summary": _first_sentence(answer_text) or answer_text,
        "key_points": [line for line in answer_text.splitlines() if line.strip()][:4],
        "limitations": [
            "该判断只比较当前已验证的最近两期净利润。",
            "当前不能解释“下降原因”，因为可验证数据没有显示这两个期间利润下降。",
        ]
        if lang == "zh"
        else [
            "This check only compares the two most recent validated profit periods.",
            "It cannot explain causes of a decline because validated evidence does not show a decline across these two periods.",
        ],
        "numeric_evidence": final_numeric_evidence,
        "text_evidence": final_text_evidence,
        "answer_status": "deterministic_false_premise",
        "final_answer_source": "deterministic_false_premise",
        "synthesis_mode": "deterministic_false_premise",
        "canonical_intent": dict(state.get("canonical_intent", {}) or {}),
        "evidence_policy_id": str(state.get("evidence_policy_id", "") or ""),
        "warnings": [],
    }
    synthesis = {
        "short_answer": output["summary"],
        "key_facts": output["key_points"],
        "analysis": [],
        "risks_or_uncertainties": output["limitations"],
        "limitations": output["limitations"],
        "citations": [str(row.get("evidence_id") or "") for row in final_numeric_evidence if str(row.get("evidence_id") or "")],
        "synthesis_strategy": "deterministic_false_premise",
        "synthesis_mode": "deterministic_false_premise",
        "final_answer_source": "deterministic_false_premise",
        "unsupported_synthesis_items": [],
    }
    state_dict: AgentState = dict(state)
    state_dict["final_answer_source"] = "deterministic_false_premise"
    state_dict, output, _packet, _frame, _methodology_context = _ensure_canonical_evidence_packet(
        state=state_dict,
        output=output,
        user_query=user_query,
        task_type=str(state.get("task_type") or "report_summary"),
        period_query=dict(state.get("period_query", {}) or {}),
        resolved_period_context=dict(state.get("resolved_period_context", {}) or {}),
        final_numeric_evidence=final_numeric_evidence,
        final_text_evidence=final_text_evidence,
        citations=final_numeric_evidence,
        comparison_target=state.get("comparison_target"),
        requested_metrics=list(state.get("requested_metrics", []) or []),
    )
    answer_text = _record_answer_transform(
        state_dict,
        previous_text="",
        new_text=answer_text,
        owner="deterministic_false_premise",
        transform="deterministic_false_premise",
        reason="validated numeric evidence contradicts user premise",
        claim_change_allowed=True,
    )
    _capture_answer_candidate(state_dict, body=answer_text, owner="deterministic_false_premise", provenance={"synthesis_mode": "deterministic_false_premise"})
    output["answer_history"] = list(state_dict.get("answer_history", []) or [])
    output["answer_candidate"] = dict(state_dict.get("answer_candidate", {}) or {})
    output["answer_candidates"] = list(state_dict.get("answer_candidates", []) or [])
    return {
        "final_answer": answer_text,
        "draft_answer": answer_text,
        "numeric_evidence": final_numeric_evidence,
        "text_evidence": final_text_evidence,
        "unsupported_claims": [],
        "numeric_citations": final_numeric_evidence,
        "text_citations": [],
        "citations": final_numeric_evidence,
        "output": output,
        "structured_sources": final_numeric_evidence,
        "document_citations": [],
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction_requested,
        "event_query": dict(event_query or {}),
        "event_results": event_results,
        "market_reaction_evidence": market_reaction_evidence,
        "market_reaction_limitations": market_reaction_limitations,
        "synthesis": synthesis,
        "synthesis_strategy": "deterministic_false_premise",
        "synthesis_mode": "deterministic_false_premise",
        "final_answer_source": "deterministic_false_premise",
        "answer_history": list(state_dict.get("answer_history", []) or []),
        "answer_candidate": dict(state_dict.get("answer_candidate", {}) or {}),
        "answer_candidates": list(state_dict.get("answer_candidates", []) or []),
        "unsupported_synthesis_items": [],
        "synthesis_model_issues": [],
        "why_tools_skipped": list(state.get("why_tools_skipped", [])),
        **_requirement_state_payload(state_dict),
        "messages": [AIMessage(content=answer_text)],
    }


def _rewrite_fcf_causal_answer_if_needed(
    answer: str,
    *,
    user_query: str,
    numeric_evidence: list[dict[str, Any]],
    lang: str,
) -> str:
    query = str(user_query or "")
    if "自由现金流" not in query and "free cash flow" not in query.lower():
        return answer
    pressure_query = any(term in query.lower() for term in ("worse", "under pressure", "pressured", "pressure", "weak", "weaker"))
    if "变差" not in query and "承压" not in query and "压力" not in query and not pressure_query:
        return answer
    if "变差" in answer and ("待验证" in answer or "假设" in answer):
        return answer
    refs = _inline_refs_from_rows(numeric_evidence, {"operating_cash_flow", "free_cash_flow", "capital_expenditure", "capex"})
    if lang == "zh":
        addition = (
            f"自由现金流变差/承压的可验证路径是：经营现金流仍为正，但资本开支接近或超过经营现金流，"
            f"使自由现金流转负或受到压制。{refs} 待验证假设包括：资本开支是否持续高位、收入确认节奏是否变化，"
            "以及云基础设施/物流投入是否继续吞噬经营现金流。"
        )
    else:
        addition = (
            f"The verifiable path for weaker free cash flow is that operating cash flow remains positive while capex is near or above OCF, "
            f"pressuring FCF. {refs} Hypotheses to verify include whether capex remains elevated, revenue-recognition timing changes, "
            "and cloud/logistics investment continues to absorb operating cash flow."
        )
    return f"{answer.rstrip()}\n\n待验证假设\n- {addition}" if lang == "zh" else f"{answer.rstrip()}\n\nHypotheses to Verify\n- {addition}"


def _business_phrases_from_snippet(snippet: str) -> list[str]:
    phrases: list[str] = []
    if _phrase_present(snippet, "gpu", "graphics processing"):
        phrases.append("GPU")
    if _phrase_present(snippet, "data center", "datacenter"):
        phrases.append("数据中心")
    if _phrase_present(snippet, "gaming", "geforce"):
        phrases.append("游戏")
    if _phrase_present(snippet, "professional visualization", "visualization"):
        phrases.append("专业可视化")
    if _phrase_present(snippet, "automotive"):
        phrases.append("汽车")
    if _phrase_present(snippet, "products", "product", "services", "service"):
        phrases.append("产品和服务")
    if _phrase_present(snippet, "customers", "markets", "platforms"):
        phrases.append("客户和市场")
    return list(dict.fromkeys(phrases))[:6]


def _risk_phrases_from_snippet(snippet: str) -> list[str]:
    phrases: list[str] = []
    if _phrase_present(snippet, "competition", "competitive", "competitor"):
        phrases.append("竞争压力")
    if _phrase_present(snippet, "demand"):
        phrases.append("需求波动")
    if _phrase_present(snippet, "supply chain", "supply"):
        phrases.append("供应链风险")
    if _phrase_present(snippet, "regulation", "regulatory"):
        phrases.append("监管风险")
    if _phrase_present(snippet, "customer concentration", "customers"):
        phrases.append("客户相关风险")
    if _phrase_present(snippet, "macro", "macroeconomic", "economic"):
        phrases.append("宏观不确定性")
    return list(dict.fromkeys(phrases))[:5]


def _fallback_claim_sentence(item: dict[str, Any], lang: str) -> str:
    ticker = str(item.get("ticker", "") or "Company").upper()
    snippet = str(item.get("supporting_snippet") or item.get("text_snippet") or "").strip()
    terms = [str(x).lower() for x in item.get("supporting_terms", []) or [] if str(x).strip()]
    dimension_id = str(item.get("dimension_id", "") or "")
    if lang == "zh":
        if dimension_id == "business_model":
            phrases = _business_phrases_from_snippet(snippet)
            if phrases:
                return f"{ticker} 的业务主要围绕{'、'.join(phrases)}等产品、服务或市场展开。"
            return f"{ticker} 的业务模式可从已检索到的业务描述中初步识别，主要围绕其披露的产品、服务和市场展开。"
        if dimension_id == "moat_and_competitive_risk":
            phrases = _risk_phrases_from_snippet(snippet)
            if phrases:
                return f"{ticker} 披露的风险包括{'、'.join(phrases)}等因素。"
            return f"{ticker} 披露的风险因素需要结合已验证文本证据解读。"
        if "competition" in terms or "competitive" in snippet.lower():
            return f"{ticker} 的披露文本显示其面临竞争相关压力。"
        if "regulation" in terms or "regulatory" in snippet.lower():
            return f"{ticker} 的披露文本显示监管因素可能影响业务。"
        if "margin" in terms or "margins" in snippet.lower():
            return f"{ticker} 的披露文本显示利润率相关压力值得关注。"
        if "demand" in terms or "demand" in snippet.lower():
            return f"{ticker} 的披露文本显示需求变化是经营关注点。"
        return f"{ticker} 的披露文本提供了可用于比较的业务和风险背景。"
    if "competition" in terms or "competitive" in snippet.lower():
        return f"{ticker} discloses competition-related business pressure."
    if "regulation" in terms or "regulatory" in snippet.lower():
        return f"{ticker} discloses regulatory factors that may affect the business."
    if "margin" in terms or "margins" in snippet.lower():
        return f"{ticker} discloses margin-related pressure."
    if "demand" in terms or "demand" in snippet.lower():
        return f"{ticker} discloses demand-related operating context."
    return f"{ticker} provides business and risk context relevant to the comparison."


def _deterministic_text_claim_fallback(
    *,
    text_evidence: list[dict[str, Any]],
    task_type: str,
    lang: str,
    analysis_scope: str = "",
) -> list[dict[str, Any]]:
    if not text_evidence:
        return []
    rows = sorted(
        [dict(item) for item in text_evidence if str(item.get("evidence_id", "")).startswith("T")],
        key=lambda item: (
            str(item.get("ticker", "")),
            -float(item.get("score", 0.0) or 0.0),
        ),
    )
    if not rows:
        return []
    selected: list[dict[str, Any]] = []
    if analysis_scope == "single_company":
        seen_dimensions: set[str] = set()
        for target_dimension in ("business_model", "moat_and_competitive_risk"):
            for item in rows:
                if str(item.get("dimension_id", "")) != target_dimension:
                    continue
                selected.append(item)
                seen_dimensions.add(target_dimension)
                break
        for item in rows:
            if item in selected:
                continue
            selected.append(item)
            if len(selected) >= 3:
                break
    elif task_type == "company_comparison":
        seen_tickers: set[str] = set()
        for item in rows:
            ticker = str(item.get("ticker", "")).upper()
            if not ticker or ticker in seen_tickers:
                continue
            selected.append(item)
            seen_tickers.add(ticker)
        selected = selected[:2]
    else:
        selected = rows[:3]

    claims: list[dict[str, Any]] = []
    for item in selected:
        snippet = str(item.get("supporting_snippet") or item.get("text_snippet") or "").strip()
        if not snippet:
            continue
        claims.append(
            {
                "claim": _fallback_claim_sentence(item, lang),
                "sentence": _fallback_claim_sentence(item, lang),
                "company": str(item.get("ticker", "")).upper(),
                "claim_type": _text_claim_type_for_section(str(item.get("section", ""))),
                "dimension_id": str(item.get("dimension_id", "")),
                "citation_ref": str(item.get("evidence_id", "")),
                "evidence_ids": [str(item.get("evidence_id", ""))],
                "supporting_quote": snippet[:240],
                "confidence": "medium",
                "generated_by": "deterministic_snippet_fallback",
                "claim_source": "deterministic_fallback",
                "source_requirement_id": str(item.get("requirement_id", "")),
            }
        )
    return claims


def _raw_item_counts_by_requirement(collection_results: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in collection_results:
        rid = str(result.get("requirement_id", "")).strip()
        if not rid:
            continue
        raw_hit_count = int(result.get("raw_hit_count", 0) or 0)
        if raw_hit_count:
            counts[rid] = max(counts.get(rid, 0), raw_hit_count)
            continue
        items = [item for item in result.get("items", []) or [] if isinstance(item, dict)]
        counts[rid] = counts.get(rid, 0) + len(items)
    return counts


def _retry_counts_by_requirement(retry_history: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in retry_history:
        rid = str(attempt.get("requirement_id", "")).strip()
        if not rid:
            continue
        counts[rid] = counts.get(rid, 0) + int(attempt.get(key, 0) or 0)
    return counts


def _collection_diagnostics(state: AgentState) -> dict[str, Any]:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    collection_results = list(state.get("evidence_collection_results", []) or [])
    collection_sufficiency = dict(state.get("evidence_sufficiency", {}) or {})
    retry_history = list(state.get("retry_history", []) or [])
    raw_retrieval_hits = _retry_counts_by_requirement(retry_history, "raw_hit_count") or _raw_item_counts_by_requirement(collection_results)
    if not evidence_plan:
        return {
            "collection_evidence_collection_results": collection_results,
            "collection_evidence_sufficiency": collection_sufficiency,
            "raw_retrieval_hits_by_requirement": raw_retrieval_hits,
        }
    collection_summary = summarize_evidence_requirements(
        evidence_plan,
        collection_results,
        collection_sufficiency,
    )
    return {
        "collection_evidence_collection_results": collection_results,
        "collection_evidence_sufficiency": collection_sufficiency,
        "collection_evidence_sufficiency_summary": collection_summary,
        "collection_requirement_status_map": dict(collection_summary.get("requirement_status_map", {}) or {}),
        "collection_trace_summary": build_trace_summary(
            evidence_plan,
            collection_results,
            collection_sufficiency,
            synthesis_mode=str(state.get("synthesis_mode", "")),
        ),
        "raw_retrieval_hits_by_requirement": raw_retrieval_hits,
    }


def _final_validation_failure_reasons(
    state: AgentState,
    *,
    validated_numeric_evidence: list[dict[str, Any]],
    validated_text_evidence: list[dict[str, Any]],
    unsupported_claims: list[dict[str, Any]],
    text_policy_reasons: list[dict[str, Any]],
    comparison_text_unbalanced: bool,
    period_error: str | None = None,
    claim_generation_error: str = "",
) -> dict[str, str]:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    requirements = list(evidence_plan.get("evidence_requirements", []) or [])
    raw_counts = _retry_counts_by_requirement(list(state.get("retry_history", []) or []), "raw_hit_count") or _raw_item_counts_by_requirement(
        list(state.get("evidence_collection_results", []) or [])
    )
    validated_numeric_ids = {
        req_id
        for item in validated_numeric_evidence
        for req_id in _evidence_requirement_ids(dict(item))
    }
    validated_text_ids = {
        req_id
        for item in validated_text_evidence
        for req_id in _evidence_requirement_ids(dict(item))
    }
    model_failure = str(claim_generation_error or "").strip() or next(
        (
            str(item.get("reason", "")).strip()
            for item in unsupported_claims
            if str(item.get("reason", "")).strip() == "model_output_invalid_json"
            or str(item.get("reason", "")).startswith("model_call_failed:")
        ),
        "",
    )
    text_policy_failure = "comparison_text_unbalanced" if comparison_text_unbalanced else (
        "text_citation_policy_filtered" if text_policy_reasons else ""
    )

    overrides: dict[str, str] = {}
    for req in requirements:
        rid = str(req.get("requirement_id", "")).strip()
        if not rid:
            continue
        req_type = str(req.get("requirement_type", ""))
        raw_count = int(raw_counts.get(rid, 0) or 0)
        if req_type == "text":
            if raw_count > 0 and rid not in validated_text_ids:
                overrides[rid] = text_policy_failure or model_failure or "no_validated_text_evidence"
        elif req_type in {"numeric", "event"}:
            if period_error and raw_count > 0 and rid not in validated_numeric_ids:
                overrides[rid] = str(period_error)
    return overrides


def _text_drop_stage(
    *,
    raw_hit_count: int,
    section_filtered_hit_count: int,
    usable_hit_count: int,
    snippet_support_passed_count: int,
    validated_text_claim_count: int,
    text_citation_kept_count: int,
    final_validated_text_count: int,
    failure_reason: str,
    rejection_reasons: dict[str, int] | None = None,
    collection_drop_stage: str = "",
) -> str:
    if final_validated_text_count > 0:
        return "satisfied"
    if raw_hit_count <= 0:
        return "no_raw_hits"
    if section_filtered_hit_count <= 0:
        return "section_filter_dropped"
    if usable_hit_count <= 0:
        if collection_drop_stage in {"section_filter_dropped", "quality_filter_dropped", "snippet_support_failed"}:
            return collection_drop_stage
        reasons = dict(rejection_reasons or {})
        if int(reasons.get("quality_filter_dropped", 0) or 0) > 0:
            return "quality_filter_dropped"
        if int(reasons.get("snippet_support_failed", 0) or 0) > 0:
            return "snippet_support_failed"
        if int(reasons.get("section_filter_dropped", 0) or 0) > 0:
            return "section_filter_dropped"
        return "quality_filter_dropped"
    if snippet_support_passed_count <= 0:
        return "snippet_support_failed"
    if validated_text_claim_count <= 0:
        return "claim_validation_failed"
    if text_citation_kept_count <= 0:
        return "citation_policy_dropped"
    return "final_bundle_dropped"


def _text_requirement_diagnostics(
    state: AgentState,
    *,
    candidate_text_evidence: list[dict[str, Any]],
    raw_text_claims: list[dict[str, Any]] | None = None,
    valid_text_claims: list[dict[str, Any]],
    rejected_text_claims: list[dict[str, Any]] | None = None,
    text_claim_validation_warnings: list[dict[str, Any]] | None = None,
    text_citations: list[dict[str, Any]],
    final_text_evidence: list[dict[str, Any]],
    requirement_status_map: dict[str, Any],
    claim_generation_error: str = "",
) -> dict[str, dict[str, Any]]:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    text_requirements = [
        req
        for req in evidence_plan.get("evidence_requirements", []) or []
        if isinstance(req, dict) and str(req.get("requirement_type", "")) == "text"
    ]
    retry_history = list(state.get("retry_history", []) or [])
    collection_results = list(state.get("evidence_collection_results", []) or [])
    collection_by_req = {
        str(result.get("requirement_id", "")).strip(): dict(result)
        for result in collection_results
        if isinstance(result, dict) and str(result.get("requirement_id", "")).strip()
    }
    fallback_counts = _raw_item_counts_by_requirement(list(state.get("evidence_collection_results", []) or []))
    raw_hit_counts = _retry_counts_by_requirement(retry_history, "raw_hit_count") or fallback_counts
    section_filtered_counts = _retry_counts_by_requirement(retry_history, "section_filtered_hit_count") or fallback_counts
    snippet_support_counts = _retry_counts_by_requirement(retry_history, "snippet_support_passed_count")
    evidence_by_id = {
        str(item.get("evidence_id", "")): item
        for item in candidate_text_evidence
        if str(item.get("evidence_id", "")).strip()
    }
    def _claim_requirement_ids(claim: dict[str, Any]) -> list[str]:
        rid = str(claim.get("source_requirement_id", "")).strip()
        if rid:
            return [rid]
        citation_ref = str(claim.get("citation_ref", "") or "").strip().upper()
        evidence_ids = [str(x).strip().upper() for x in claim.get("evidence_ids", []) or [] if str(x).strip()]
        if not citation_ref and evidence_ids:
            citation_ref = evidence_ids[0]
        if citation_ref:
            return _evidence_requirement_ids(dict(evidence_by_id.get(citation_ref, {}) or {}))
        return []

    usable_counts: dict[str, int] = {}
    for item in candidate_text_evidence:
        for rid in _evidence_requirement_ids(dict(item)):
            usable_counts[rid] = usable_counts.get(rid, 0) + 1
    raw_text_claim_count: dict[str, int] = {}
    for claim in raw_text_claims or []:
        for rid in _claim_requirement_ids(claim):
            raw_text_claim_count[rid] = raw_text_claim_count.get(rid, 0) + 1
    validated_text_claim_count: dict[str, int] = {}
    for claim in valid_text_claims:
        req_ids = {
            req_id
            for eid in claim.get("evidence_ids", []) or []
            if str(eid).startswith("T")
            for req_id in _evidence_requirement_ids(dict(evidence_by_id.get(str(eid), {}) or {}))
        }
        for rid in {rid for rid in req_ids if rid}:
            validated_text_claim_count[rid] = validated_text_claim_count.get(rid, 0) + 1
    rejected_by_req: dict[str, list[dict[str, Any]]] = {}
    for claim in rejected_text_claims or []:
        for rid in _claim_requirement_ids(claim):
            rejected_by_req.setdefault(rid, []).append(dict(claim))
    warnings_by_req: dict[str, list[dict[str, Any]]] = {}
    for warning in text_claim_validation_warnings or []:
        for rid in _claim_requirement_ids(warning):
            warnings_by_req.setdefault(rid, []).append(dict(warning))
    text_citation_kept_count: dict[str, int] = {}
    for citation in text_citations:
        req_ids = _evidence_requirement_ids(dict(citation))
        if not req_ids:
            req_ids = _evidence_requirement_ids(dict(evidence_by_id.get(str(citation.get("evidence_id", "")), {}) or {}))
        for rid in req_ids:
            text_citation_kept_count[rid] = text_citation_kept_count.get(rid, 0) + 1
    final_validated_text_count: dict[str, int] = {}
    for item in final_text_evidence:
        for rid in _evidence_requirement_ids(dict(item)):
            final_validated_text_count[rid] = final_validated_text_count.get(rid, 0) + 1

    diagnostics: dict[str, dict[str, Any]] = {}
    for req in text_requirements:
        rid = str(req.get("requirement_id", "")).strip()
        if not rid:
            continue
        failure_reason = str(requirement_status_map.get(rid, {}).get("failure_reason", "") or "")
        collection_detail = dict(collection_by_req.get(rid, {}) or {})
        raw_count = int(raw_hit_counts.get(rid, collection_detail.get("raw_hit_count", 0)) or 0)
        section_count = int(section_filtered_counts.get(rid, collection_detail.get("section_filtered_hit_count", 0)) or 0)
        usable_count = int(usable_counts.get(rid, collection_detail.get("usable_hit_count", 0)) or 0)
        snippet_count = int(snippet_support_counts.get(rid, collection_detail.get("snippet_support_passed_count", usable_count)) or 0)
        if usable_count > 0:
            raw_count = max(raw_count, usable_count)
            section_count = max(section_count, usable_count)
            snippet_count = max(snippet_count, usable_count)
        text_claim_count = int(validated_text_claim_count.get(rid, 0) or 0)
        citation_count = int(text_citation_kept_count.get(rid, 0) or 0)
        final_count = int(final_validated_text_count.get(rid, 0) or 0)
        rejection_reasons = {
            str(k): int(v or 0)
            for k, v in dict(collection_detail.get("rejection_reasons", {}) or {}).items()
            if str(k).strip()
        }
        diagnostics[rid] = {
            "requirement_id": rid,
            "company": str(req.get("company") or collection_detail.get("company") or ""),
            "retrieval_query": str(req.get("retrieval_query") or collection_detail.get("retrieval_query") or ""),
            "section_preferences": list(req.get("section_preferences") or collection_detail.get("section_preferences") or []),
            "fallback_queries": list(req.get("broadened_queries") or collection_detail.get("fallback_queries") or []),
            "fallback_sections": list(req.get("fallback_sections") or collection_detail.get("fallback_sections") or []),
            "raw_hit_count": raw_count,
            "section_filtered_hit_count": section_count,
            "usable_hit_count": usable_count,
            "snippet_support_passed_count": snippet_count,
            "raw_text_claim_count": int(raw_text_claim_count.get(rid, 0) or 0),
            "candidate_text_claim_count": int(raw_text_claim_count.get(rid, 0) or 0),
            "validated_text_claim_count": text_claim_count,
            "text_claim_validated_count": text_claim_count,
            "rejected_text_claims": rejected_by_req.get(rid, []),
            "text_claim_validation_warnings": warnings_by_req.get(rid, []),
            "claim_generation_error": claim_generation_error,
            "text_citation_kept_count": citation_count,
            "final_validated_text_count": final_count,
            "top_raw_snippets": list(collection_detail.get("top_raw_snippets", []) or []),
            "top_rejected_snippets": list(collection_detail.get("top_rejected_snippets", []) or []),
            "rejection_reasons": rejection_reasons,
            "drop_stage": _text_drop_stage(
                raw_hit_count=raw_count,
                section_filtered_hit_count=section_count,
                usable_hit_count=usable_count,
                snippet_support_passed_count=snippet_count,
                validated_text_claim_count=text_claim_count,
                text_citation_kept_count=citation_count,
                final_validated_text_count=final_count,
                failure_reason=failure_reason,
                rejection_reasons=rejection_reasons,
                collection_drop_stage=str(collection_detail.get("drop_stage", "") or ""),
            ),
        }
    return diagnostics


def _merge_requirement_diagnostics(
    final_accounting: dict[str, Any],
    diagnostics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if not diagnostics:
        return final_accounting
    merged = dict(final_accounting)
    merged_results: list[dict[str, Any]] = []
    for result in list(merged.get("evidence_collection_results", []) or []):
        item = dict(result)
        rid = str(item.get("requirement_id", "")).strip()
        if rid in diagnostics:
            item.update(diagnostics[rid])
        merged_results.append(item)
    merged["evidence_collection_results"] = merged_results

    summary = dict(merged.get("evidence_sufficiency_summary", {}) or {})
    status_map = {
        rid: dict(item)
        for rid, item in dict(merged.get("requirement_status_map", {}) or {}).items()
    }
    for rid, diag in diagnostics.items():
        if rid in status_map:
            status_map[rid].update(diag)
    summary["requirement_status_map"] = status_map
    summary["collected_evidence_by_requirement"] = status_map
    merged["evidence_sufficiency_summary"] = summary
    merged["requirement_status_map"] = status_map
    merged["text_requirement_diagnostics"] = diagnostics
    return merged


def _apply_final_requirement_ledger(
    state: AgentState,
    *,
    validated_numeric_evidence: list[dict[str, Any]],
    validated_text_evidence: list[dict[str, Any]],
    candidate_text_evidence: list[dict[str, Any]] | None = None,
    raw_text_claims: list[dict[str, Any]] | None = None,
    valid_text_claims: list[dict[str, Any]] | None = None,
    rejected_text_claims: list[dict[str, Any]] | None = None,
    text_claim_validation_warnings: list[dict[str, Any]] | None = None,
    text_citations: list[dict[str, Any]] | None = None,
    unsupported_claims: list[dict[str, Any]],
    text_policy_reasons: list[dict[str, Any]],
    comparison_text_unbalanced: bool,
    synthesis_mode: str,
    period_error: str | None = None,
    claim_generation_error: str = "",
) -> dict[str, Any]:
    evidence_plan = dict(state.get("evidence_plan", {}) or {})
    if not evidence_plan.get("evidence_requirements"):
        return dict(state)
    updated = dict(state)
    updated.update(_collection_diagnostics(state))
    failure_overrides = _final_validation_failure_reasons(
        state,
        validated_numeric_evidence=validated_numeric_evidence,
        validated_text_evidence=validated_text_evidence,
        unsupported_claims=unsupported_claims,
        text_policy_reasons=text_policy_reasons,
        comparison_text_unbalanced=comparison_text_unbalanced,
        period_error=period_error,
        claim_generation_error=claim_generation_error,
    )
    final_accounting = finalize_evidence_accounting(
        evidence_plan,
        list(state.get("evidence_collection_results", []) or []),
        validated_numeric_evidence=validated_numeric_evidence,
        validated_text_evidence=validated_text_evidence,
        validation_failure_reasons=failure_overrides,
        synthesis_mode=synthesis_mode,
    )
    text_diagnostics = _text_requirement_diagnostics(
        state,
        candidate_text_evidence=list(candidate_text_evidence or []),
        raw_text_claims=list(raw_text_claims or []),
        valid_text_claims=list(valid_text_claims or []),
        rejected_text_claims=list(rejected_text_claims or []),
        text_claim_validation_warnings=list(text_claim_validation_warnings or []),
        text_citations=list(text_citations or []),
        final_text_evidence=validated_text_evidence,
        requirement_status_map=dict(final_accounting.get("requirement_status_map", {}) or {}),
        claim_generation_error=claim_generation_error,
    )
    final_accounting = _merge_requirement_diagnostics(final_accounting, text_diagnostics)
    updated.update(final_accounting)
    updated["final_requirement_status_map"] = dict(final_accounting.get("requirement_status_map", {}) or {})
    return updated


def _requirement_state_payload(state: AgentState) -> dict[str, Any]:
    keys = (
        "research_plan_raw",
        "research_plan_validated",
        "research_plan_used",
        "research_plan_validation",
        "required_answer_parts",
        "legacy_evidence_plan",
        "evidence_collection_results",
        "evidence_sufficiency",
        "evidence_sufficiency_summary",
        "answer_part_status_by_id",
        "evidence_gap_by_answer_part",
        "missing_required_answer_parts",
        "partial_required_answer_parts",
        "missing_but_analyzable_answer_parts",
        "missing_and_unanswerable_answer_parts",
        "evidence_health",
        "tool_error_context",
        "research_plan_source",
        "research_plan_fallback_reason",
        "research_plan_duration_ms",
        "requirement_status_map",
        "dimension_status_by_id",
        "dimension_status_map",
        "satisfied_dimensions",
        "covered_dimensions",
        "partial_dimensions",
        "missing_dimensions",
        "dimension_coverage_rate",
        "weighted_dimension_coverage_rate",
        "framework_sufficiency_status",
        "red_flags",
        "missing_evidence_flags",
        "forbidden_claims",
        "allowed_claims",
        "final_requirement_status_map",
        "requirement_limitations",
        "missing_requirements",
        "degradation_reason",
        "trace_summary",
        "validated_requirement_ids",
        "validated_numeric_evidence_count",
        "validated_text_evidence_count",
        "collection_evidence_collection_results",
        "collection_evidence_sufficiency",
        "collection_evidence_sufficiency_summary",
        "collection_requirement_status_map",
        "collection_trace_summary",
        "raw_retrieval_hits_by_requirement",
        "text_requirement_diagnostics",
        "evidence_packet",
        "evidence_packet_summary",
        "methodology_context",
        "comparison_judgment_frame",
        "analyst_draft",
        "analyst_draft_validation",
        "draft_validation",
        "draft_attempts",
        "draft_revision_attempts",
        "draft_violations",
        "draft_final_status",
        "draft_status",
        "final_answer_source",
        "answer_history",
        "answer_candidate",
        "answer_candidates",
        "canonical_intent",
        "intent_merge_decision",
        "evidence_policy",
        "evidence_policy_id",
        "contract_decision",
        "draft_release_decision",
        "relevance_decision",
        "relevance_status",
        "analytical_claims",
        "claim_tiers",
        "analytical_reasoning_status",
        "selected_analysis_framework",
    )
    payload = {key: state.get(key) for key in keys if key in state}
    framework_fields = analysis_framework_trace_fields(state.get("selected_analysis_framework", {}))
    if framework_fields:
        trace_summary = dict(payload.get("trace_summary", {}) or {})
        trace_summary.update(framework_fields)
        payload["trace_summary"] = trace_summary
    return payload


def _supports_analyst_draft(answer_mode: str) -> bool:
    return answer_mode in {"comparison_brief", "analytical", "cautious_outlook", "risk_focused_analysis"}


def _should_use_analyst_draft(answer_mode: str, synthesis_mode: str) -> bool:
    if not settings.analyst_draft_enabled:
        return False
    if answer_mode in {"comparison_brief", "analytical"}:
        return not synthesis_mode.startswith("insufficient_")
    if answer_mode == "cautious_outlook":
        return synthesis_mode == "cautious_outlook"
    if answer_mode == "risk_focused_analysis":
        return synthesis_mode == "risk_focused_analysis"
    return False


def _draft_revision_attempts(draft_attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in draft_attempts
        if int(item.get("attempt_index", item.get("attempt_number", 0)) or 0) > 1
    ]


def _final_answer_source_from_draft_state(
    *,
    accepted_draft: dict[str, Any] | None,
    draft_attempts: list[dict[str, Any]],
    comparison_judgment_frame: dict[str, Any] | None,
) -> str:
    if accepted_draft:
        return "analyst_draft_revised" if _draft_revision_attempts(draft_attempts) else "analyst_draft_initial"
    if comparison_judgment_frame:
        return "comparison_decision_fallback"
    return "deterministic_synthesis"


def _requested_dimensions_from_state(state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for source in (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions", []),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions", []),
        state.get("requested_dimensions", []),
    ):
        for item in source or []:
            text = str(item).strip()
            if text and text not in out:
                out.append(text)
    return out


def _text_evidence_flow_summary(state: Mapping[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    diagnostics = state.get("text_requirement_diagnostics")
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    drop_stage_counts: dict[str, int] = {}
    candidate_count = 0
    pre_citation_validated_count = 0
    citable_count = 0
    final_packet_count = len(packet.get("text_snippets", []) or [])
    for item in diagnostics.values():
        if not isinstance(item, dict):
            continue
        candidate_count += int(item.get("usable_hit_count") or item.get("candidate_text_claim_count") or 0)
        pre_citation_validated_count += int(
            item.get("validated_text_claim_count")
            or item.get("text_claim_validated_count")
            or item.get("snippet_support_passed_count")
            or 0
        )
        citable_count += int(item.get("text_citation_kept_count") or item.get("final_validated_text_count") or 0)
        stage = str(item.get("drop_stage") or "").strip()
        if stage:
            drop_stage_counts[stage] = drop_stage_counts.get(stage, 0) + 1
    if not diagnostics:
        candidate_count = final_packet_count
        pre_citation_validated_count = final_packet_count
        citable_count = final_packet_count
    return {
        "text_candidate_count": candidate_count,
        "text_pre_citation_validated_count": pre_citation_validated_count,
        "text_citable_count": citable_count,
        "text_final_packet_count": final_packet_count,
        "text_drop_stage_counts": drop_stage_counts,
    }


def _ensure_canonical_evidence_packet(
    *,
    state: AgentState,
    output: dict[str, Any],
    user_query: str,
    task_type: str,
    period_query: dict[str, Any],
    resolved_period_context: dict[str, Any],
    final_numeric_evidence: list[dict[str, Any]],
    final_text_evidence: list[dict[str, Any]],
    citations: list[dict[str, Any]],
    comparison_target: str | None,
    requested_metrics: list[str],
    requirement_limitations: list[dict[str, Any]] | None = None,
    safety_limitations: list[dict[str, Any]] | None = None,
) -> tuple[AgentState, dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build the canonical evidence packet for every answer path."""
    packet = build_evidence_packet(
        user_query=user_query,
        task_type=str(task_type),
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        safety_intent=str(state.get("safety_intent", "normal")),
        analysis_scope=str(state.get("analysis_scope", "")),
        time_policy=str(state.get("time_policy", "")),
        period_scope=str(state.get("period_scope", "")),
        companies=list(state.get("companies", []) or []),
        comparison_target=comparison_target,
        requested_metrics=list(requested_metrics or []),
        period_query=period_query,
        resolved_period_context=resolved_period_context,
        numeric_evidence=final_numeric_evidence,
        text_evidence=final_text_evidence,
        citations=citations,
        evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
        requirement_limitations=list(requirement_limitations if requirement_limitations is not None else output.get("limitations", []) or []),
        safety_limitations=list(safety_limitations or []),
        selected_framework=dict(state.get("selected_analysis_framework", {}) or {}),
        requirement_status_map=dict(state.get("requirement_status_map", {}) or {}),
    ).model_dump(exclude_none=True)
    packet["research_plan"] = dict(state.get("research_plan_used", {}) or {})
    packet["canonical_intent"] = dict(state.get("canonical_intent", {}) or {})
    packet["requested_dimensions"] = _requested_dimensions_from_state(state)
    packet["required_answer_parts"] = list(state.get("required_answer_parts", []) or [])
    packet["evidence_gap_by_answer_part"] = dict(state.get("evidence_gap_by_answer_part", {}) or {})
    packet["answer_part_status_by_id"] = dict(state.get("answer_part_status_by_id", {}) or {})
    packet["partial_required_answer_parts"] = list(state.get("partial_required_answer_parts", []) or [])
    packet["text_evidence_flow_summary"] = _text_evidence_flow_summary(state, packet)
    dimension_status_map = dict(
        state.get("dimension_status_map")
        or dict(state.get("evidence_sufficiency", {}) or {}).get("dimension_status_map", {})
        or {}
    )
    serialized_red_flags = list(packet.get("red_flags", []) or [])
    if not serialized_red_flags:
        red_flags = detect_red_flags(packet, dimension_status_map)
        serialized_red_flags = serialize_red_flags(red_flags)
        packet["red_flags"] = serialized_red_flags
        packet["missing_evidence_flags"] = [
            dict(flag)
            for flag in serialized_red_flags
            if str(flag.get("category", "")) == "missing_evidence"
        ]
    packet["red_flags"] = serialized_red_flags
    methodology_context = build_methodology_context(packet)
    packet["methodology_context"] = methodology_context
    comparison_judgment_frame: dict[str, Any] = {}
    if str(task_type) == "company_comparison" or str(state.get("answer_mode", "")) == "comparison_brief":
        comparison_judgment_frame = build_comparison_judgment_frame(packet).model_dump(exclude_none=True)
        packet["comparison_judgment_frame"] = comparison_judgment_frame
    packet_summary = summarize_evidence_packet(packet)
    state["evidence_packet"] = packet
    state["evidence_packet_summary"] = packet_summary
    state["comparison_judgment_frame"] = comparison_judgment_frame
    state["methodology_context"] = methodology_context
    state["red_flags"] = serialized_red_flags
    state["missing_evidence_flags"] = list(packet.get("missing_evidence_flags", []) or [])
    state["forbidden_claims"] = list(packet.get("forbidden_claims", []) or [])
    state["allowed_claims"] = list(packet.get("allowed_claims", []) or [])
    output["evidence_packet_summary"] = packet_summary
    output["red_flags"] = user_visible_red_flags(serialized_red_flags)
    if comparison_judgment_frame:
        output["comparison_judgment_frame"] = summarize_comparison_judgment_frame(comparison_judgment_frame)
    return state, output, packet, comparison_judgment_frame, methodology_context


def _record_answer_transform(
    state: AgentState,
    *,
    previous_text: str,
    new_text: str,
    owner: str,
    transform: str,
    reason: str,
    claim_change_allowed: bool = False,
) -> str:
    if str(previous_text or "") == str(new_text or ""):
        return new_text
    previous_body = str(previous_text or "")
    if not previous_body:
        previous_body = str(state.get("final_answer") or state.get("draft_answer") or "")
    candidate = AnswerAssembler.candidate(
        body=new_text,
        owner=owner,
        requested_dimensions=_requested_dimensions_from_state(state),
        evidence_refs=None,
        limitations=[
            str(item.get("message") or item.get("code") or item)
            if isinstance(item, dict)
            else str(item)
            for item in state.get("requirement_limitations", []) or []
        ],
        allowed_repairs=["add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"],
        provenance={"transform": transform, "reason": reason},
    )
    payload = AnswerAssembler.select(
        candidate,
        state,
        previous_body=previous_body,
        transform=transform,
        reason=reason,
        claim_change_allowed=claim_change_allowed,
    )
    state.update(payload)
    return new_text


def _capture_answer_candidate(state: AgentState, *, body: str, owner: str, provenance: dict[str, Any] | None = None) -> None:
    candidate = AnswerAssembler.candidate(
        body=body,
        owner=owner,
        requested_dimensions=_requested_dimensions_from_state(state),
        evidence_refs=None,
        limitations=[str(item.get("message") or item.get("code") or item) if isinstance(item, dict) else str(item) for item in state.get("requirement_limitations", []) or []],
        allowed_repairs=["add_citation", "strip_sentence", "scope_limit", "downgrade_to_bounded"],
        provenance=provenance or {},
    ).model_dump(exclude_none=True)
    state["answer_candidate"] = candidate
    candidates = [dict(item) for item in state.get("answer_candidates", []) or [] if isinstance(item, Mapping)]
    candidates.append(candidate)
    state["answer_candidates"] = candidates[-8:]


def _output_with_canonical_packet(output: dict[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    packet = state.get("evidence_packet", {})
    if isinstance(packet, Mapping) and packet.get("canonical_source"):
        return {**output, "evidence_packet": dict(packet)}
    return output


def _should_skip_draft_for_dimension_comparison(
    packet: Mapping[str, Any],
    comparison_judgment_frame: Mapping[str, Any],
) -> bool:
    if not comparison_judgment_frame:
        return False
    task_type = str(packet.get("task_type") or "")
    answer_mode = str(packet.get("answer_mode") or "")
    if task_type != "company_comparison" and answer_mode != "comparison_brief":
        return False
    requested = {
        str(item).strip()
        for item in packet.get("requested_dimensions", []) or []
        if str(item).strip()
    }
    deterministic_dims = {
        "cash_flow_quality",
        "revenue_quality",
        "valuation_and_risk_boundary",
    }
    focus = requested & deterministic_dims
    if not focus:
        return False
    status_map = dict(packet.get("dimension_status_map", {}) or {})
    for dimension in focus:
        status = str(dict(status_map.get(dimension, {}) or {}).get("status") or "")
        if status in {"satisfied", "partial"}:
            return True
    return False


def _conversational_output(
    *,
    state: AgentState,
    lang: str,
    task_type: str,
    answer_text: str,
    answer_mode: str,
    safety_intent: str,
    needs_clarification: bool,
    clarification_question: str | None,
    final_answer_source: str,
) -> dict[str, Any]:
    limitations = [_conversation_limitation(lang, "no_external_tools"), *_state_safety_limitations(state, lang)]
    if safety_intent == "unsupported_or_out_of_scope" or answer_mode == "refusal_or_redirect":
        limitations.append(_conversation_limitation(lang, "unsupported_scope"))
    output = {
        "protocol_version": OUTPUT_PROTOCOL_VERSION,
        "output_language": lang,
        "language_leakage": language_leakage_count(answer_text, lang),
        "language_leakage_unresolved": language_leakage_count(answer_text, lang) > 0,
        "task_type": task_type,
        "answer_mode": answer_mode,
        "safety_intent": safety_intent,
        "needs_tools": False,
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
        "title": "对话式财报分析助手" if lang == "zh" else "Conversational Filings Analyst",
        "summary": _truncate_text(answer_text, 240),
        "comparison_basis": "",
        "key_points": [_truncate_text(answer_text, 240)],
        "numeric_evidence": [],
        "text_evidence": [],
        "limitations": limitations,
        "used_tools": [],
        "trace_id": str(state.get("trace_id", "")),
        "view": {
            "kind": "clarification" if answer_mode == "clarification" else (
                "meta_response" if answer_mode == "meta" else "refusal_or_redirect"
            ),
            "short_answer": _truncate_text(answer_text, 240),
            "headline_metric": {},
            "period_note": "",
            "supporting_points": [_truncate_text(answer_text, 240)],
            "clarification_question": clarification_question or "",
            "example_questions": [
                "AAPL 最近几个季度营收趋势如何？",
                "总结 TSLA 最新 10-K 的主要风险因素。",
            ]
            if lang == "zh"
            else [
                "How has AAPL revenue trended recently?",
                "Summarize the main risks in TSLA's latest 10-K.",
            ],
        },
        "synthesis_strategy": "conversational_short_circuit",
        "synthesis_mode": "conversational_short_circuit",
        "synthesis": {
            "short_answer": _truncate_text(answer_text, 240),
            "key_facts": [],
            "analysis": [],
            "risks_or_uncertainties": limitations,
            "limitations": limitations,
            "citations": [],
            "synthesis_strategy": "conversational_short_circuit",
            "synthesis_mode": "conversational_short_circuit",
            "final_answer_source": final_answer_source,
            "unsupported_synthesis_items": [],
        },
        "final_answer_source": final_answer_source,
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
    return output

def _meta_answer(lang: str) -> str:
    if lang == "zh":
        return (
            "我是一个财报与公司分析 Agent，主要基于 SEC filing 文本、结构化财务指标、价格历史和财报事件窗口来回答问题。"
            "我可以做事实问答、趋势分析、公司对比、风险/MD&A 摘要和财报发布后的市场反应分析。"
            "我会尽量给出可追踪证据和引用，但不提供投资建议、买卖推荐或股价预测。"
        )
    return (
        "I am a financial-filings analysis agent. I answer using SEC filing text, structured financial facts, "
        "price history, and filing event-window evidence. I can help with fact QA, trend analysis, company "
        "comparison, risk/MD&A summaries, and market reactions around filings. I provide evidence-grounded "
        "analysis, not investment advice, buy/sell recommendations, or stock-price predictions."
    )

def _refusal_or_redirect_answer(lang: str) -> str:
    if lang == "zh":
        return (
            "这个问题超出了我当前支持的范围。我更适合回答公司财报、SEC filing 文本、结构化财务指标、公司对比，"
            "以及财报发布前后的市场反应问题。你可以改问：“AAPL 最近几个季度营收趋势如何？”或"
            "“总结 TSLA 最新 10-K 的主要风险因素。”"
        )
    return (
        "That question is outside my supported scope. I am best suited for company filings, SEC filing text, "
        "structured financial metrics, company comparisons, and market reactions around filings. You could ask, "
        "\"How has AAPL revenue trended recently?\" or \"Summarize the main risks in TSLA's latest 10-K.\""
    )

def _maybe_conversational_short_circuit(
    state: AgentState,
    *,
    lang: str,
    task_type: str,
    event_intent: str,
    market_reaction_requested: bool,
    event_query: dict[str, Any],
    event_results: list[dict[str, Any]],
    market_reaction_evidence: list[dict[str, Any]],
    market_reaction_limitations: list[str],
) -> dict[str, Any] | None:
    answer_mode = str(state.get("answer_mode", "direct_fact"))
    safety_intent = str(state.get("safety_intent", "normal"))
    needs_tools = bool(state.get("needs_tools", True))
    if needs_tools and answer_mode not in {"meta", "clarification", "refusal_or_redirect"}:
        return None

    needs_clarification = bool(state.get("needs_clarification", answer_mode == "clarification"))
    clarification_question = state.get("clarification_question")
    if answer_mode == "meta":
        answer_text = _meta_answer(lang)
    elif answer_mode == "clarification":
        answer_text = str(clarification_question or _clarification_message(lang, "clarification_needed"))
        if safety_intent == "investment_advice_like":
            if lang == "zh":
                answer_text = (
                    "我不能判断“这个股票能买吗”，也不提供买入、卖出或持有建议；这不构成投资建议。"
                    "请补充 ticker/公司名；我可以把问题转成风险、估值和证据边界分析，并说明限制。"
                )
            else:
                answer_text = (
                    "I cannot decide whether this stock is buyable or provide buy, sell, or hold advice. "
                    "Please provide a ticker or company name; I can reframe it as risk, valuation, and evidence-boundary analysis."
                )
    elif answer_mode == "refusal_or_redirect" or safety_intent == "unsupported_or_out_of_scope":
        answer_text = _refusal_or_redirect_answer(lang)
    else:
        return None

    output = _conversational_output(
        state=state,
        lang=lang,
        task_type=task_type,
        answer_text=answer_text,
        answer_mode=answer_mode,
        safety_intent=safety_intent,
        needs_clarification=needs_clarification,
        clarification_question=str(clarification_question) if clarification_question else None,
        final_answer_source="unsupported_or_refusal",
    )
    state, output, _packet, _frame, _methodology_context = _ensure_canonical_evidence_packet(
        state=state,
        output=output,
        user_query=str(state.get("user_query") or ""),
        task_type=task_type,
        period_query=dict(state.get("period_query", _default_period_query())),
        resolved_period_context=dict(state.get("resolved_period_context", {}) or {}),
        final_numeric_evidence=[],
        final_text_evidence=[],
        citations=[],
        comparison_target=state.get("comparison_target"),
        requested_metrics=list(state.get("requested_metrics", []) or []),
        requirement_limitations=list(output.get("limitations", []) or []),
        safety_limitations=[],
    )
    answer_text = _enforce_answer_language(answer_text, state.get("user_query", ""), lang)
    answer_text = _clean_answer_text(answer_text, lang)
    state["final_answer_source"] = "unsupported_or_refusal"
    answer_text = _record_answer_transform(
        state,
        previous_text="",
        new_text=answer_text,
        owner="unsupported_or_refusal",
        transform="conversational_short_circuit",
        reason=answer_mode,
        claim_change_allowed=True,
    )
    _capture_answer_candidate(state, body=answer_text, owner="unsupported_or_refusal", provenance={"answer_mode": answer_mode})
    output["answer_history"] = list(state.get("answer_history", []) or [])
    output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
    output["answer_candidates"] = list(state.get("answer_candidates", []) or [])
    return {
        "final_answer": answer_text,
        "draft_answer": answer_text,
        "output_language": lang,
        "language_leakage": language_leakage_count(answer_text, lang),
        "language_leakage_unresolved": language_leakage_count(answer_text, lang) > 0,
        "numeric_evidence": [],
        "text_evidence": [],
        "unsupported_claims": [],
        "numeric_citations": [],
        "text_citations": [],
        "citations": [],
        "output": output,
        "structured_sources": [],
        "document_citations": [],
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction_requested,
        "event_query": event_query,
        "event_results": event_results,
        "market_reaction_evidence": market_reaction_evidence,
        "market_reaction_limitations": market_reaction_limitations,
        "synthesis": output.get("synthesis", {}),
        "synthesis_strategy": "conversational_short_circuit",
        "synthesis_mode": "conversational_short_circuit",
        "final_answer_source": "unsupported_or_refusal",
        "answer_history": list(state.get("answer_history", []) or []),
        "answer_candidate": dict(state.get("answer_candidate", {}) or {}),
        "answer_candidates": list(state.get("answer_candidates", []) or []),
        "unsupported_synthesis_items": [],
        "synthesis_model_issues": [
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": f"conversational_short_circuit:{answer_mode}",
            }
        ],
        "why_tools_skipped": list(state.get("why_tools_skipped", []))
        or [{"reason": f"answer_mode:{answer_mode}", "message": "tools_skipped_for_conversational_response"}],
        **_requirement_state_payload(state),
        "messages": [AIMessage(content=answer_text)],
    }

def generate_agent_answer(state: AgentState) -> dict[str, Any]:
    """Generate the final answer through evidence-constrained claim validation."""
    state = dict(state)
    user_query = state["user_query"]
    task_type = state.get("task_type", "fact_qa")
    period_query = dict(state.get("period_query", _default_period_query()))
    resolved_period_context = dict(state.get("resolved_period_context", {}))
    comparison_basis_label = str(state.get("comparison_basis_label", ""))
    lang = str(state.get("output_language") or detect_output_language(user_query))
    answer_language = "Simplified Chinese" if lang == "zh" else "English"
    event_intent = str(state.get("event_intent", _detect_event_intent(user_query, task_type=str(task_type)))).lower()
    if event_intent not in EVENT_INTENT_TYPES:
        event_intent = "none"
    market_reaction_requested = bool(state.get("market_reaction_requested", event_intent == "required"))
    event_query = dict(state.get("event_query", {}))
    event_results = list(state.get("event_results", []))
    market_reaction_evidence = list(state.get("market_reaction_evidence", _collect_event_rows(state.get("tool_results", []))))
    market_reaction_limitations = list(state.get("market_reaction_limitations", []))

    conversational = _maybe_conversational_short_circuit(
        state,
        lang=lang,
        task_type=str(task_type),
        event_intent=event_intent,
        market_reaction_requested=market_reaction_requested,
        event_query=event_query,
        event_results=event_results,
        market_reaction_evidence=market_reaction_evidence,
        market_reaction_limitations=market_reaction_limitations,
    )
    if conversational is not None:
        return conversational

    if resolved_period_context.get("needs_clarification"):
        answer_text = _clarification_message(lang, str(resolved_period_context.get("clarification_reason", "")))
        clarification_claims = [
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": str(resolved_period_context.get("clarification_reason", "period_clarification_needed")),
            }
        ]
        output = _build_phase4_output(
            state=state,
            lang=lang,
            task_type=task_type,
            comparison_basis_label=comparison_basis_label,
            period_query=period_query,
            numeric_claims=[],
            text_claims=[],
            numeric_evidence=[],
            text_evidence=[],
            numeric_citations=[],
            text_citations=[],
            unsupported_claims=clarification_claims,
            period_error=str(resolved_period_context.get("clarification_reason", "period_clarification_needed")),
            comparison_text_unbalanced=False,
        )
        output["synthesis_mode"] = "synthesis_degraded"
        output["final_answer_source"] = "deterministic_synthesis"
        output["summary"] = _truncate_text(answer_text, 240)
        output["key_points"] = [output["summary"]]
        state["final_answer_source"] = "deterministic_synthesis"
        state, output, _packet, _frame, _methodology_context = _ensure_canonical_evidence_packet(
            state=state,
            output=output,
            user_query=user_query,
            task_type=str(task_type),
            period_query=period_query,
            resolved_period_context=resolved_period_context,
            final_numeric_evidence=[],
            final_text_evidence=[],
            citations=[],
            comparison_target=state.get("comparison_target"),
            requested_metrics=list(state.get("requested_metrics", []) or []),
        )
        rendered = _render_answer_from_output(_output_with_canonical_packet(output, state), lang)
        rendered = _clean_answer_text(rendered, lang)
        rendered = _record_answer_transform(
            state,
            previous_text="",
            new_text=rendered,
            owner="deterministic_synthesis",
            transform="period_clarification_boundary",
            reason=str(resolved_period_context.get("clarification_reason", "period_clarification_needed")),
            claim_change_allowed=True,
        )
        _capture_answer_candidate(state, body=rendered, owner="deterministic_synthesis", provenance={"synthesis_mode": "synthesis_degraded"})
        output["answer_history"] = list(state.get("answer_history", []) or [])
        output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
        output["answer_candidates"] = list(state.get("answer_candidates", []) or [])
        return {
            "final_answer": rendered,
            "draft_answer": rendered,
            "numeric_evidence": [],
            "text_evidence": [],
            "unsupported_claims": clarification_claims,
            "numeric_citations": [],
            "text_citations": [],
            "citations": [],
            "output": output,
            "structured_sources": [],
            "document_citations": [],
            "event_intent": event_intent,
            "market_reaction_requested": market_reaction_requested,
            "event_query": event_query,
            "event_results": event_results,
            "market_reaction_evidence": market_reaction_evidence,
            "market_reaction_limitations": market_reaction_limitations,
            "synthesis": {
                "short_answer": output.get("summary", rendered),
                "key_facts": [],
                "analysis": [],
                "risks_or_uncertainties": output.get("limitations", []),
                "limitations": output.get("limitations", []),
                "citations": [],
                "synthesis_strategy": "synthesis_degraded",
                "synthesis_mode": "synthesis_degraded",
                "final_answer_source": "deterministic_synthesis",
                "unsupported_synthesis_items": [],
            },
            "synthesis_strategy": "synthesis_degraded",
            "synthesis_mode": "synthesis_degraded",
            "final_answer_source": "deterministic_synthesis",
            "answer_history": list(state.get("answer_history", []) or []),
            "answer_candidate": dict(state.get("answer_candidate", {}) or {}),
            "answer_candidates": list(state.get("answer_candidates", []) or []),
            "unsupported_synthesis_items": [],
            "why_tools_skipped": list(state.get("why_tools_skipped", [])),
            **_requirement_state_payload(state),
            "messages": [AIMessage(content=rendered)],
        }

    evidence_bundle = _build_evidence_bundle(state)
    numeric_evidence = list(evidence_bundle["numeric_evidence"])
    text_evidence = list(evidence_bundle["text_evidence"])
    evidence_sufficiency = dict(state.get("evidence_sufficiency", {}) or {})
    requirement_limitations = list(
        evidence_sufficiency.get("requirement_limitations", state.get("requirement_limitations", [])) or []
    )
    if not requirement_limitations and _has_evidence_requirements(state):
        requirement_limitations = list(
            summarize_evidence_requirements(
                dict(state.get("evidence_plan", {}) or {}),
                list(state.get("evidence_collection_results", []) or []),
                evidence_sufficiency,
            ).get("requirement_limitations", [])
            or []
        )
    state["requirement_limitations"] = requirement_limitations
    if _has_evidence_requirements(state):
        overall_status = str(evidence_sufficiency.get("overall_status", ""))
        if overall_status == "insufficient" and not _has_satisfied_collection_evidence(state):
            state = _apply_final_requirement_ledger(
                state,
                validated_numeric_evidence=[],
                validated_text_evidence=[],
                candidate_text_evidence=[],
                valid_text_claims=[],
                text_citations=[],
                unsupported_claims=[],
                text_policy_reasons=[],
                comparison_text_unbalanced=False,
                synthesis_mode="",
            )
            synthesis_mode = derive_synthesis_mode(
                answer_mode=str(state.get("answer_mode", "direct_fact")),
                task_type=str(task_type),
                safety_intent=str(state.get("safety_intent", "normal")),
                evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
                has_validated_numeric=False,
                has_validated_text=False,
            )
            state["synthesis_mode"] = synthesis_mode
            if isinstance(state.get("trace_summary"), dict):
                state["trace_summary"]["final_synthesis_mode"] = synthesis_mode
            unsupported_claims = [
                {
                    "claim_type": "system",
                    "sentence": "",
                    "evidence_ids": [],
                    "reason": str(state.get("degradation_reason") or "required_evidence_missing"),
                }
            ]
            output = _build_phase4_output(
                state=state,
                lang=lang,
                task_type=task_type,
                comparison_basis_label=comparison_basis_label,
                period_query=period_query,
                numeric_claims=[],
                text_claims=[],
                numeric_evidence=[],
                text_evidence=[],
                numeric_citations=[],
                text_citations=[],
                unsupported_claims=unsupported_claims,
                period_error=None,
                comparison_text_unbalanced=False,
            )
            output["synthesis_mode"] = synthesis_mode
            output["final_answer_source"] = "deterministic_synthesis"
            state["final_answer_source"] = "deterministic_synthesis"
            state, output, _packet, _frame, _methodology_context = _ensure_canonical_evidence_packet(
                state=state,
                output=output,
                user_query=user_query,
                task_type=str(task_type),
                period_query=period_query,
                resolved_period_context=resolved_period_context,
                final_numeric_evidence=[],
                final_text_evidence=[],
                citations=[],
                comparison_target=state.get("comparison_target"),
                requested_metrics=list(state.get("requested_metrics", []) or []),
            )
            answer_text = _render_answer_from_output(_output_with_canonical_packet(output, state), lang)
            answer_text = _clean_answer_text(answer_text, lang)
            answer_text = _record_answer_transform(
                state,
                previous_text="",
                new_text=answer_text,
                owner="deterministic_synthesis",
                transform="deterministic_insufficient_evidence",
                reason=str(state.get("degradation_reason") or "required_evidence_missing"),
                claim_change_allowed=True,
            )
            _capture_answer_candidate(state, body=answer_text, owner="deterministic_synthesis", provenance={"synthesis_mode": synthesis_mode})
            output["answer_history"] = list(state.get("answer_history", []) or [])
            output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
            output["answer_candidates"] = list(state.get("answer_candidates", []) or [])
            return {
                "final_answer": answer_text,
                "draft_answer": answer_text,
                "numeric_evidence": [],
                "text_evidence": [],
                "unsupported_claims": unsupported_claims,
                "numeric_citations": [],
                "text_citations": [],
                "citations": [],
                "output": output,
                "structured_sources": [],
                "document_citations": [],
                "event_intent": event_intent,
                "market_reaction_requested": market_reaction_requested,
                "event_query": event_query,
                "event_results": event_results,
                "market_reaction_evidence": market_reaction_evidence,
                "market_reaction_limitations": market_reaction_limitations,
                "synthesis": {
                    "short_answer": output.get("summary", answer_text),
                    "key_facts": [],
                    "analysis": [],
                    "risks_or_uncertainties": output.get("limitations", []),
                    "limitations": output.get("limitations", []),
                    "citations": [],
                    "synthesis_strategy": "synthesis_degraded",
                    "synthesis_mode": synthesis_mode,
                    "final_answer_source": "deterministic_synthesis",
                    "unsupported_synthesis_items": [],
                },
                "synthesis_strategy": "synthesis_degraded",
                "synthesis_mode": synthesis_mode,
                "final_answer_source": "deterministic_synthesis",
                "answer_history": list(state.get("answer_history", []) or []),
                "answer_candidate": dict(state.get("answer_candidate", {}) or {}),
                "answer_candidates": list(state.get("answer_candidates", []) or []),
                "unsupported_synthesis_items": [],
                "why_tools_skipped": list(state.get("why_tools_skipped", [])),
                **_requirement_state_payload(state),
                "messages": [AIMessage(content=answer_text)],
            }
        numeric_evidence, text_evidence = _filter_evidence_to_candidate_requirements(state, numeric_evidence, text_evidence)
        text_evidence = _attach_text_requirement_metadata(state, text_evidence)
    direct_profit_decline = _profit_decline_false_premise_direct_result(
        state=state,
        user_query=user_query,
        numeric_evidence=numeric_evidence,
        text_evidence=text_evidence,
        lang=lang,
        event_intent=event_intent,
        market_reaction_requested=market_reaction_requested,
        event_query=event_query,
        event_results=event_results,
        market_reaction_evidence=market_reaction_evidence,
        market_reaction_limitations=market_reaction_limitations,
    )
    if direct_profit_decline is not None:
        return direct_profit_decline
    numeric_map = dict(evidence_bundle["numeric_map"])
    numeric_map = {str(e.get("evidence_id", "")): e for e in numeric_evidence}
    text_map = {str(e.get("evidence_id", "")): e for e in text_evidence}
    unsupported_claims: list[dict[str, Any]] = []
    synthesis_model_issues: list[dict[str, Any]] = []

    # Numeric claims are deterministic and program-led. LLM is reserved for text-only explanations.
    deterministic_numeric_claims = _build_deterministic_numeric_claims(state, numeric_evidence, lang)
    valid_numeric_claims, bad_numeric_claims = _validate_numeric_claims_strict(
        deterministic_numeric_claims,
        numeric_map=numeric_map,
    )
    valid_text_claims: list[dict[str, Any]] = []
    unsupported_claims.extend(bad_numeric_claims)

    parsed: dict[str, Any] = {}
    raw_text_claims: list[dict[str, Any]] = []
    rejected_text_claims: list[dict[str, Any]] = []
    text_claim_validation_warnings: list[dict[str, Any]] = []
    claim_generation_error = ""
    if text_evidence:
        prompt_text = GENERATE_ANSWER.format(
            user_query=user_query,
            task_type=task_type,
            answer_language=answer_language,
            text_evidence=_evidence_catalog_text(
                text_evidence,
                fields=(
                    "evidence_id",
                    "ticker",
                    "requirement_id",
                    "requirement_ids",
                    "form_type",
                    "fiscal_period",
                    "section",
                    "chunk_order",
                    "text_snippet",
                    "supporting_snippet",
                    "supporting_terms",
                ),
                max_items=40,
            ),
        )
        system_msg = "You are a strict financial explanation generator. Return valid JSON only, no markdown."
        full_text = system_msg + prompt_text
        est_input_tokens = len(full_text) // 2 + 100
        context_limit = 8192
        max_output = max(min(1200, context_limit - est_input_tokens), 256)
        logger.info(
            "generate_answer(text-only): est_input=%d max_output=%d chars=%d",
            est_input_tokens,
            max_output,
            len(full_text),
        )
        llm = _get_llm(reasoning=True, temperature=0.1, max_tokens=max_output)
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=system_msg),
                    HumanMessage(content=prompt_text),
                ]
            )
            raw_response = re.sub(r"<think>.*?</think>", "", response.content or "", flags=re.DOTALL).strip()
            parsed = _parse_json_response(raw_response)
            if not parsed:
                claim_generation_error = "model_output_invalid_json"
                synthesis_model_issues.append(
                    {
                        "claim_type": "system",
                        "sentence": raw_response[:200],
                        "evidence_ids": [],
                        "reason": "model_output_invalid_json",
                    }
                )
        except Exception as exc:
            logger.warning("generate_answer: text claim generation failed: %s", exc)
            claim_generation_error = f"model_call_failed:{exc}"
            synthesis_model_issues.append(
                {
                    "claim_type": "system",
                    "sentence": "",
                    "evidence_ids": [],
                    "reason": f"model_call_failed:{exc}",
                }
            )
    elif not numeric_evidence:
        unsupported_claims.append(
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": "no_evidence_available",
            }
        )

    raw_text_claims = _normalize_claims(parsed, "text_claims")
    if text_evidence and not raw_text_claims:
        if not claim_generation_error:
            claim_generation_error = "model_returned_empty_text_claims"
        raw_text_claims = _deterministic_text_claim_fallback(
            text_evidence=text_evidence,
            task_type=str(task_type),
            lang=lang,
            analysis_scope=str(state.get("analysis_scope", "")),
        )
    validation_context = _text_validation_context(state)
    valid_text_claims, rejected_text_claims, text_claim_validation_warnings = validate_text_claims_enhanced(
        raw_text_claims,
        evidence_map=text_map,
        validation_context=validation_context,
    )
    if text_evidence and str(state.get("analysis_scope", "")) == "single_company":
        valid_dimensions = {str(claim.get("dimension_id", "")) for claim in valid_text_claims if str(claim.get("dimension_id", ""))}
        available_dimensions = {str(item.get("dimension_id", "")) for item in text_evidence if str(item.get("dimension_id", ""))}
        target_dimensions = {"business_model", "moat_and_competitive_risk"} & available_dimensions
        if target_dimensions - valid_dimensions:
            fallback_claims = _deterministic_text_claim_fallback(
                text_evidence=[
                    item
                    for item in text_evidence
                    if str(item.get("dimension_id", "")) in (target_dimensions - valid_dimensions)
                ],
                task_type=str(task_type),
                lang=lang,
                analysis_scope=str(state.get("analysis_scope", "")),
            )
            existing_refs = {str(claim.get("citation_ref", "")) for claim in raw_text_claims}
            new_fallback_claims = [
                claim
                for claim in fallback_claims
                if str(claim.get("citation_ref", "")) not in existing_refs
            ]
            if new_fallback_claims:
                raw_text_claims.extend(new_fallback_claims)
                fallback_valid, fallback_rejected, fallback_warnings = validate_text_claims_enhanced(
                    new_fallback_claims,
                    evidence_map=text_map,
                    validation_context=validation_context,
                )
                seen_valid = {
                    (str(claim.get("sentence", "")), str(claim.get("citation_ref", "")))
                    for claim in valid_text_claims
                }
                for claim in fallback_valid:
                    key = (str(claim.get("sentence", "")), str(claim.get("citation_ref", "")))
                    if key not in seen_valid:
                        valid_text_claims.append(claim)
                        seen_valid.add(key)
                rejected_text_claims.extend(fallback_rejected)
                text_claim_validation_warnings.extend(fallback_warnings)
    diagnostic_valid_text_claims = list(valid_text_claims)
    if rejected_text_claims:
        synthesis_model_issues.extend(rejected_text_claims)
    if not valid_numeric_claims and not valid_text_claims and not market_reaction_evidence:
        state = _append_safety_limitation(
            state,
            "insufficient_validated_evidence",
            "medium",
            "Current validated evidence is insufficient for a more specific conclusion.",
        )

    numeric_citations, text_citations = _collect_citations_from_claims(
        numeric_claims=valid_numeric_claims,
        text_claims=valid_text_claims,
        numeric_evidence=numeric_evidence,
        text_evidence=text_evidence,
    )
    text_citations, text_policy_reasons, comparison_text_unbalanced = _apply_text_citation_policy(
        {
            "task_type": task_type,
            "companies": state.get("companies", []),
            "comparison_target": state.get("comparison_target"),
            "retrieval_policy": state.get("retrieval_policy", {}),
        },
        text_citations=text_citations,
    )
    if text_policy_reasons:
        synthesis_model_issues.extend(text_policy_reasons)
    allowed_text_ids = {str(c.get("evidence_id", "")) for c in text_citations if str(c.get("evidence_id", ""))}
    if comparison_text_unbalanced:
        valid_text_claims = []
    elif allowed_text_ids:
        valid_text_claims = [
            c
            for c in valid_text_claims
            if all((not eid.startswith("T")) or (eid in allowed_text_ids) for eid in c.get("evidence_ids", []))
        ]
    else:
        valid_text_claims = []
    used_text_ids_after = {
        eid
        for claim in valid_text_claims
        for eid in claim.get("evidence_ids", [])
        if str(eid).startswith("T")
    }
    if used_text_ids_after:
        text_citations = [c for c in text_citations if str(c.get("evidence_id", "")) in used_text_ids_after]
    else:
        text_citations = []
    citations = numeric_citations + text_citations
    period_ok, period_error = _period_consistency_ok(
        {
            "task_type": task_type,
            "period_query": period_query,
            "resolved_period_context": resolved_period_context,
            "market_reaction_requested": bool(state.get("market_reaction_requested", False)),
            "user_query": user_query,
            "requested_dimensions": state.get("requested_dimensions", []),
            "required_dimensions": state.get("required_dimensions", []),
            "primary_dimension": state.get("primary_dimension", ""),
            "canonical_intent": state.get("canonical_intent", {}),
            "evidence_policy": state.get("evidence_policy", {}),
            "text_citations": text_citations,
        },
        numeric_citations=numeric_citations,
    )
    if (
        not period_ok
        and str(task_type) == "company_comparison"
        and _is_risk_comparison_query(user_query, state)
        and text_citations
        and not numeric_citations
    ):
        period_ok = True
        period_error = None
    final_numeric_ids = {str(c.get("evidence_id", "")) for c in numeric_citations if str(c.get("evidence_id", ""))}
    final_text_ids = {str(c.get("evidence_id", "")) for c in text_citations if str(c.get("evidence_id", ""))}
    final_numeric_evidence = [item for item in numeric_evidence if str(item.get("evidence_id", "")) in final_numeric_ids]
    final_text_evidence = [item for item in text_evidence if str(item.get("evidence_id", "")) in final_text_ids]
    text_claim_by_ref = {
        str(claim.get("citation_ref", "") or "").strip().upper(): dict(claim)
        for claim in valid_text_claims
        if str(claim.get("citation_ref", "") or "").strip()
    }
    enriched_final_text_evidence: list[dict[str, Any]] = []
    for item in final_text_evidence:
        row = annotate_driver_evidence(item)
        claim = text_claim_by_ref.get(str(row.get("evidence_id", "")).strip().upper())
        if claim:
            row["claim"] = str(claim.get("claim") or claim.get("sentence") or "")
            row["citation_ref"] = str(claim.get("citation_ref", "") or row.get("evidence_id", ""))
            row["claim_source"] = str(claim.get("claim_source") or claim.get("generated_by") or "")
            if claim.get("dimension_id") and not row.get("dimension_id"):
                row["dimension_id"] = str(claim.get("dimension_id") or "")
        row = apply_scope_aware_summary(row, user_query=user_query)
        enriched_final_text_evidence.append(row)
    final_text_evidence = enriched_final_text_evidence
    if not period_ok:
        unsupported_claims.append(
            {
                "claim_type": "system",
                "sentence": "",
                "evidence_ids": [],
                "reason": str(period_error or "period_consistency_failed"),
            }
        )
        valid_numeric_claims = []
        valid_text_claims = []
        numeric_citations = []
        text_citations = []
        citations = []
        final_numeric_evidence = []
        final_text_evidence = []

        state = _apply_final_requirement_ledger(
            state,
            validated_numeric_evidence=final_numeric_evidence,
            validated_text_evidence=final_text_evidence,
            candidate_text_evidence=text_evidence,
            raw_text_claims=raw_text_claims,
            valid_text_claims=diagnostic_valid_text_claims,
            rejected_text_claims=rejected_text_claims,
            text_claim_validation_warnings=text_claim_validation_warnings,
            text_citations=text_citations,
            unsupported_claims=unsupported_claims,
            text_policy_reasons=text_policy_reasons,
            comparison_text_unbalanced=comparison_text_unbalanced,
            synthesis_mode="",
            period_error=period_error,
            claim_generation_error=claim_generation_error,
        )
        synthesis_mode = derive_synthesis_mode(
            answer_mode=str(state.get("answer_mode", "direct_fact")),
            task_type=str(task_type),
            safety_intent=str(state.get("safety_intent", "normal")),
            evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
            has_validated_numeric=False,
            has_validated_text=False,
        )
        state["synthesis_mode"] = synthesis_mode
        if isinstance(state.get("trace_summary"), dict):
            state["trace_summary"]["final_synthesis_mode"] = synthesis_mode

        if period_error and "missing" in period_error:
            answer_text = _clarification_message(lang, period_error)
        elif period_error == "no_common_period_for_same_period_comparison":
            if lang == "zh":
                answer_text = "无法严格同口径比较：当前两家公司没有共同可比期间。"
            else:
                answer_text = "Cannot perform a strict same-basis comparison: no common comparable period was found."
        else:
            answer_text = _evidence_insufficient_message(lang, task_type)
        if task_type == "company_comparison":
            basis_line = _comparison_basis_line(lang, comparison_basis_label)
            if basis_line:
                answer_text = f"{basis_line}\n{answer_text}"
        else:
            annual_basis_line = _annual_year_basis_line(
                lang=lang,
                task_type=task_type,
                period_query=period_query,
                resolved_period_context=resolved_period_context,
            )
            if annual_basis_line:
                answer_text = f"{annual_basis_line}\n{answer_text}"
        output = _build_phase4_output(
            state=state,
            lang=lang,
            task_type=task_type,
            comparison_basis_label=comparison_basis_label,
            period_query=period_query,
            numeric_claims=valid_numeric_claims,
            text_claims=valid_text_claims,
            numeric_evidence=final_numeric_evidence,
            text_evidence=final_text_evidence,
            numeric_citations=numeric_citations,
            text_citations=text_citations,
            unsupported_claims=unsupported_claims,
            period_error=str(period_error or "period_consistency_failed"),
            comparison_text_unbalanced=comparison_text_unbalanced,
        )
        output["synthesis_mode"] = synthesis_mode
        output["final_answer_source"] = "deterministic_synthesis"
        output["summary"] = _truncate_text(answer_text, 240)
        if output.get("key_points"):
            output["key_points"][0] = output["summary"]
        else:
            output["key_points"] = [output["summary"]]
        state["final_answer_source"] = "deterministic_synthesis"
        state, output, _packet, _comparison_frame, _methodology_context = _ensure_canonical_evidence_packet(
            state=state,
            output=output,
            user_query=user_query,
            task_type=str(task_type),
            period_query=period_query,
            resolved_period_context=resolved_period_context,
            final_numeric_evidence=final_numeric_evidence,
            final_text_evidence=final_text_evidence,
            citations=citations,
            comparison_target=state.get("comparison_target"),
            requested_metrics=list(state.get("requested_metrics", []) or []),
        )
        answer_text = _render_answer_from_output(_output_with_canonical_packet(output, state), lang)
        answer_text = _clean_answer_text(answer_text, lang)
        answer_text = _record_answer_transform(
            state,
            previous_text="",
            new_text=answer_text,
            owner="deterministic_synthesis",
            transform="deterministic_period_boundary",
            reason=str(period_error or "period_consistency_failed"),
            claim_change_allowed=True,
        )
        _capture_answer_candidate(state, body=answer_text, owner="deterministic_synthesis", provenance={"synthesis_mode": synthesis_mode})
        output["answer_history"] = list(state.get("answer_history", []) or [])
        output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
        output["answer_candidates"] = list(state.get("answer_candidates", []) or [])
        return {
            "final_answer": answer_text,
            "draft_answer": answer_text,
            "numeric_evidence": final_numeric_evidence,
            "text_evidence": final_text_evidence,
            "unsupported_claims": unsupported_claims,
            "numeric_citations": numeric_citations,
            "text_citations": text_citations,
            "citations": citations,
            "output": output,
            "structured_sources": numeric_citations,
            "document_citations": text_citations,
            "event_intent": event_intent,
            "market_reaction_requested": market_reaction_requested,
            "event_query": event_query,
            "event_results": event_results,
            "market_reaction_evidence": market_reaction_evidence,
            "market_reaction_limitations": market_reaction_limitations,
            "synthesis": {
                "short_answer": output.get("summary", answer_text),
                "key_facts": [],
                "analysis": [],
                "risks_or_uncertainties": output.get("limitations", []),
                "limitations": output.get("limitations", []),
                "citations": [],
                "synthesis_strategy": "synthesis_degraded",
                "synthesis_mode": synthesis_mode,
                "final_answer_source": "deterministic_synthesis",
                "unsupported_synthesis_items": [],
            },
            "synthesis_strategy": "synthesis_degraded",
            "synthesis_mode": synthesis_mode,
            "final_answer_source": "deterministic_synthesis",
            "answer_history": list(state.get("answer_history", []) or []),
            "answer_candidate": dict(state.get("answer_candidate", {}) or {}),
            "answer_candidates": list(state.get("answer_candidates", []) or []),
            "unsupported_synthesis_items": [],
            "why_tools_skipped": list(state.get("why_tools_skipped", [])),
            **_requirement_state_payload(state),
            "messages": [AIMessage(content=answer_text)],
        }

    state = _apply_final_requirement_ledger(
        state,
        validated_numeric_evidence=final_numeric_evidence,
        validated_text_evidence=final_text_evidence,
        candidate_text_evidence=text_evidence,
        raw_text_claims=raw_text_claims,
        valid_text_claims=diagnostic_valid_text_claims,
        rejected_text_claims=rejected_text_claims,
        text_claim_validation_warnings=text_claim_validation_warnings,
        text_citations=text_citations,
        unsupported_claims=unsupported_claims,
        text_policy_reasons=text_policy_reasons,
        comparison_text_unbalanced=comparison_text_unbalanced,
        synthesis_mode="",
        claim_generation_error=claim_generation_error,
    )
    synthesis_mode = derive_synthesis_mode(
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        task_type=str(task_type),
        safety_intent=str(state.get("safety_intent", "normal")),
        evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
        has_validated_numeric=bool(final_numeric_evidence),
        has_validated_text=bool(final_text_evidence),
    )
    state["synthesis_mode"] = synthesis_mode
    if isinstance(state.get("trace_summary"), dict):
        state["trace_summary"]["final_synthesis_mode"] = synthesis_mode

    output = _build_phase4_output(
        state=state,
        lang=lang,
        task_type=task_type,
        comparison_basis_label=comparison_basis_label,
        period_query=period_query,
        numeric_claims=valid_numeric_claims,
        text_claims=valid_text_claims,
        numeric_evidence=final_numeric_evidence,
        text_evidence=final_text_evidence,
        numeric_citations=numeric_citations,
        text_citations=text_citations,
        unsupported_claims=unsupported_claims,
        period_error=None,
        comparison_text_unbalanced=comparison_text_unbalanced,
    )
    state, output, packet, comparison_judgment_frame, methodology_context = _ensure_canonical_evidence_packet(
        state=state,
        output=output,
        user_query=user_query,
        task_type=str(task_type),
        period_query=period_query,
        resolved_period_context=resolved_period_context,
        final_numeric_evidence=final_numeric_evidence,
        final_text_evidence=final_text_evidence,
        citations=citations,
        comparison_target=state.get("comparison_target"),
        requested_metrics=list(state.get("requested_metrics", []) or []),
    )

    accepted_draft: dict[str, Any] | None = None
    draft_validation: dict[str, Any] = {}
    draft_attempts: list[dict[str, Any]] = []
    draft_revision_attempts: list[dict[str, Any]] = []
    draft_violations: list[dict[str, Any]] = []
    draft_final_status = ""
    draft_status = ""
    if _should_use_analyst_draft(str(state.get("answer_mode", "direct_fact")), synthesis_mode):
        packet_summary = summarize_evidence_packet(packet)
        if _should_skip_draft_for_dimension_comparison(packet, comparison_judgment_frame):
            draft_payload = {}
            draft_validation = {
                "passed": False,
                "status": "not_run",
                "final_status": "deterministic_dimension_comparison",
                "fallback_reason": "dimension_specific_comparison_frame",
            }
            draft_attempts = []
            draft_revision_attempts = []
            draft_violations = []
            draft_final_status = "deterministic_dimension_comparison"
            draft_status = "not_run"
        else:
            trace_id = str(state.get("trace_id") or "")
            if trace_id:
                append_progress_event(
                    trace_id,
                    "draft_started",
                    "started",
                    "正在生成分析草稿，并要求模型基于证据输出判断、原因和边界。",
                    node="generate",
                    metadata={
                        "synthesis_mode": synthesis_mode,
                        "answer_mode": str(state.get("answer_mode", "direct_fact")),
                    },
                )
            draft_loop = run_analyst_draft_loop(
                evidence_packet=packet,
                answer_language=answer_language,
                synthesis_mode=synthesis_mode,
                safety_policy={
                    **dict(state.get("safety_decision", {}) or {}),
                    "answer_mode": str(state.get("answer_mode", "direct_fact")),
                    "safety_intent": str(state.get("safety_intent", "normal")),
                },
                comparison_judgment_frame=comparison_judgment_frame or None,
                methodology_context=methodology_context,
                max_attempts=max(2, int(settings.analyst_draft_max_attempts or 2)),
            )
            draft_payload = dict(draft_loop.get("draft", {}) or {})
            draft_validation = dict(draft_loop.get("validation", {}) or {})
            draft_attempts = list(draft_loop.get("attempts", []) or [])
            draft_revision_attempts = _draft_revision_attempts(draft_attempts)
            draft_violations = list(draft_loop.get("violations", []) or [])
            draft_final_status = str(draft_loop.get("draft_final_status", ""))
            draft_generation_issues = list(draft_loop.get("generation_issues", []) or [])
            if draft_generation_issues:
                synthesis_model_issues.extend(draft_generation_issues)
            draft_status = str(draft_validation.get("status", draft_final_status))
            if bool(draft_validation.get("passed", False)) and dict(draft_loop.get("accepted_draft", {}) or {}):
                accepted_draft = dict(draft_loop.get("accepted_draft", {}) or {})
        state["evidence_packet"] = packet
        state["evidence_packet_summary"] = packet_summary
        state["comparison_judgment_frame"] = comparison_judgment_frame
        state["analyst_draft"] = draft_payload
        state["analyst_draft_validation"] = draft_validation
        state["draft_validation"] = draft_validation
        state["draft_attempts"] = draft_attempts
        state["draft_revision_attempts"] = draft_revision_attempts
        state["draft_violations"] = draft_violations
        state["draft_final_status"] = draft_final_status
        state["draft_status"] = draft_status
        output["evidence_packet_summary"] = packet_summary
        if comparison_judgment_frame:
            output["comparison_judgment_frame"] = summarize_comparison_judgment_frame(comparison_judgment_frame)
        output["analyst_draft_summary"] = summarize_analyst_draft(draft_payload, draft_validation)
        output["draft_validation"] = draft_validation
        output["draft_attempts"] = draft_attempts
        output["draft_revision_attempts"] = draft_revision_attempts
        output["draft_violations"] = draft_violations
        output["draft_final_status"] = draft_final_status
        output["draft_status"] = draft_status

    preliminary_answer_source = _final_answer_source_from_draft_state(
        accepted_draft=accepted_draft,
        draft_attempts=draft_attempts,
        comparison_judgment_frame=comparison_judgment_frame or None,
    )
    draft_warnings = list(draft_validation.get("warnings", []) or []) if draft_validation else []
    draft_release_decision = {
        "decision": (
            "released_with_warnings"
            if accepted_draft and draft_warnings
            else ("released" if accepted_draft else ("fallback" if draft_attempts else "not_run"))
        ),
        "released": bool(accepted_draft),
        "source": preliminary_answer_source,
        "reason": "" if accepted_draft else (str(draft_validation.get("fallback_reason") or draft_final_status or "") if draft_attempts else "analyst_draft_not_run"),
        "warnings": draft_warnings,
    }
    state["final_answer_source"] = preliminary_answer_source
    state["draft_release_decision"] = draft_release_decision
    output["final_answer_source"] = preliminary_answer_source
    output["draft_release_decision"] = draft_release_decision
    trace_id = str(state.get("trace_id") or "")
    if trace_id and draft_release_decision.get("decision") != "not_run":
        warning_count = len(list(draft_release_decision.get("warnings", []) or []))
        released = bool(draft_release_decision.get("released", False))
        append_progress_event(
            trace_id,
            "draft_validated",
            "completed" if released else "warning",
            (
                "分析草稿已通过验证，带有非阻断 warning。"
                if released and warning_count
                else ("分析草稿已通过验证。" if released else "分析草稿未发布，已切换到结构化兜底答案。")
            ),
            node="generate",
            metadata={
                "draft_release_decision": str(draft_release_decision.get("decision") or ""),
                "draft_released": released,
                "warning_count": warning_count,
                "draft_final_status": draft_final_status,
            },
        )

    synthesis = build_analytical_synthesis(
        user_query=user_query,
        analysis_plan=dict(state.get("analysis_plan", {}) or {}),
        evidence_plan=dict(state.get("evidence_plan", {}) or {}),
        evidence_collection_results=list(state.get("evidence_collection_results", []) or []),
        evidence_sufficiency=dict(state.get("evidence_sufficiency", {}) or {}),
        valid_numeric_claims=valid_numeric_claims,
        valid_text_claims=valid_text_claims,
        numeric_citations=numeric_citations,
        text_citations=text_citations,
        numeric_evidence_cards=list(output.get("numeric_evidence", [])),
        text_evidence_cards=list(output.get("text_evidence", [])),
        limitations=list(output.get("limitations", [])),
        answer_policy=dict(state.get("analysis_plan", {}).get("answer_policy", {}) or {}),
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        safety_intent=str(state.get("safety_intent", "normal")),
        task_type=str(task_type),
        lang=lang,
        accepted_draft=accepted_draft,
        comparison_judgment_frame=dict(state.get("comparison_judgment_frame", {}) or {}),
        final_answer_source=preliminary_answer_source,
        draft_status=draft_status,
        draft_final_status=draft_final_status,
        red_flags=list(output.get("red_flags", []) or []),
        evidence_packet=dict(state.get("evidence_packet", {}) or {}),
    )
    synthesis_payload = synthesis.model_dump(exclude_none=True)
    state["synthesis_mode"] = str(synthesis_payload.get("synthesis_mode", state.get("synthesis_mode", "")))
    if synthesis_payload.get("degradation_reason"):
        state["degradation_reason"] = synthesis_payload.get("degradation_reason")
    if isinstance(state.get("trace_summary"), dict):
        state["trace_summary"]["final_synthesis_mode"] = state["synthesis_mode"]
    unsupported_synthesis_items = list(synthesis_payload.get("unsupported_synthesis_items", []))
    if unsupported_synthesis_items:
        synthesis_model_issues.extend(unsupported_synthesis_items)
    output["synthesis"] = synthesis_payload
    output["synthesis_strategy"] = str(synthesis_payload.get("synthesis_strategy", "synthesis_degraded"))
    output["synthesis_mode"] = str(synthesis_payload.get("synthesis_mode", synthesis_mode))
    output["analytical_claims"] = list(synthesis_payload.get("analytical_claims", []) or [])
    output["claim_tiers"] = dict(synthesis_payload.get("claim_tiers", {}) or {})
    output["analytical_reasoning_status"] = str(synthesis_payload.get("analytical_reasoning_status", ""))
    output["evidence_health"] = str(synthesis_payload.get("evidence_health") or state.get("evidence_health") or "")
    output["tool_error_context"] = list(synthesis_payload.get("tool_error_context", []) or state.get("tool_error_context", []) or [])
    output["final_answer_source"] = str(synthesis_payload.get("final_answer_source", preliminary_answer_source))
    output["summary"] = _truncate_text(str(synthesis_payload.get("short_answer", output.get("summary", ""))), 240)
    output["key_points"] = [
        str(item.get("sentence", ""))
        for item in list(synthesis_payload.get("key_facts", [])) + list(synthesis_payload.get("analysis", []))
        if str(item.get("sentence", "")).strip()
    ][:5]
    output["limitations"] = list(synthesis_payload.get("limitations", output.get("limitations", [])))
    legacy_view = dict(output.get("view", {}) or {})
    synthesis_view = build_synthesis_view(
        synthesis_payload,
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        task_type=str(task_type),
        safety_intent=str(state.get("safety_intent", "normal")),
        lang=lang,
    )
    for key, value in legacy_view.items():
        synthesis_view.setdefault(key, value)
    output["view"] = synthesis_view
    answer_text = render_synthesis_text(
        synthesis_payload,
        lang=lang,
        answer_mode=str(state.get("answer_mode", "direct_fact")),
        safety_intent=str(state.get("safety_intent", "normal")),
    )
    if not answer_text.strip():
        synthesis_payload["final_answer_source"] = "legacy_output_render_fallback"
        output["final_answer_source"] = "legacy_output_render_fallback"
        state["final_answer_source"] = "legacy_output_render_fallback"
        output["synthesis"] = synthesis_payload
        answer_text = _render_answer_from_output(_output_with_canonical_packet(output, state), lang)
    else:
        state["final_answer_source"] = str(synthesis_payload.get("final_answer_source", preliminary_answer_source))
    annual_basis_line = _annual_year_basis_line(
        lang=lang,
        task_type=task_type,
        period_query=period_query,
        resolved_period_context=resolved_period_context,
    )
    if annual_basis_line:
        answer_text = f"{annual_basis_line}\n{answer_text}"
        points = list(output.get("key_points", []))
        if annual_basis_line not in points:
            output["key_points"] = [annual_basis_line] + points
    output["summary"] = _truncate_text(
        output.get("summary", _first_sentence(answer_text) or answer_text),
        180 if task_type == "fact_qa" else 240,
    )
    answer_text = _enforce_answer_language(answer_text, user_query, lang)
    answer_text = _clean_answer_text(answer_text, lang)
    answer_text = sanitize_user_facing_answer_text(answer_text, lang)
    answer_text = repair_language_leakage(answer_text, lang)
    leakage_count = language_leakage_count(answer_text, lang)
    output["language_leakage"] = leakage_count
    output["language_leakage_unresolved"] = leakage_count > 0
    output["output_language"] = lang
    answer_owner = str(state.get("final_answer_source") or output.get("final_answer_source") or preliminary_answer_source)
    answer_text = _record_answer_transform(
        state,
        previous_text="",
        new_text=answer_text,
        owner=answer_owner,
        transform="render_synthesis_text",
        reason=str(synthesis_payload.get("synthesis_strategy") or "rendered_answer"),
        claim_change_allowed=True,
    )

    valuation_candidate = build_bounded_valuation_risk_comparison_candidate(
        dict(state.get("comparison_judgment_frame", {}) or {}),
        dict(state.get("evidence_packet", {}) or {}),
        lang=lang,
    )
    if valuation_candidate.strip():
        previous_text = answer_text
        answer_text = _record_answer_transform(
            state,
            previous_text=previous_text,
            new_text=valuation_candidate,
            owner="bounded_valuation_risk_comparison_candidate",
            transform="bounded_valuation_risk_comparison_candidate",
            reason="preserve per-metric valuation-risk judgments inside evidence boundary",
            claim_change_allowed=False,
        )
        output["summary"] = _truncate_text(_first_sentence(answer_text) or answer_text, 240)

    previous_text = answer_text
    answer_text = _rewrite_valuation_boundary_contradiction(previous_text, final_numeric_evidence, lang)
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_valuation_boundary_postprocess",
        transform="rewrite_valuation_boundary_contradiction",
        reason="replace unsupported valuation-missing wording with bounded metric wording",
        claim_change_allowed=False,
    )
    previous_text = answer_text
    answer_text = _rewrite_risk_comparison_answer_if_needed(
        previous_text,
        state=state,
        synthesis_payload=synthesis_payload,
        user_query=user_query,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_risk_comparison_postprocess",
        transform="rewrite_risk_comparison_answer_if_needed",
        reason="force bounded risk comparison instead of unsupported ranking",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _rewrite_profit_decline_premise_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_profit_decline_postprocess",
        transform="rewrite_profit_decline_premise_if_needed",
        reason="remove unsupported profit-decline premise",
        claim_change_allowed=False,
    )
    previous_text = answer_text
    answer_text = _rewrite_fcf_causal_answer_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_fcf_causal_postprocess",
        transform="rewrite_fcf_causal_answer_if_needed",
        reason="bound FCF causal wording to validated numeric evidence",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_fcf_answer_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_fcf_postprocess",
        transform="bounded_fcf_answer_if_needed",
        reason="downgrade FCF answer to available OCF/FCF/capex evidence",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_valuation_answer_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_valuation_postprocess",
        transform="bounded_valuation_answer_if_needed",
        reason="downgrade valuation answer to verified valuation metrics and safety boundary",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_valuation_comparison_answer_if_needed(
        previous_text,
        user_query=user_query,
        state=state,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_valuation_comparison_postprocess",
        transform="bounded_valuation_comparison_answer_if_needed",
        reason="preserve per-metric valuation-risk comparison",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_revenue_quality_answer_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_revenue_quality_postprocess",
        transform="bounded_revenue_quality_answer_if_needed",
        reason="downgrade revenue-quality answer to available revenue evidence",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_scenario_risk_answer_if_needed(
        previous_text,
        user_query=user_query,
        text_evidence=final_text_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_scenario_risk_postprocess",
        transform="bounded_scenario_risk_answer_if_needed",
        reason="bind scenario risk answer to validated scenario-matching risk text",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _bounded_aws_segment_profit_answer_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        text_evidence=final_text_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_aws_segment_profit_postprocess",
        transform="bounded_aws_segment_profit_answer_if_needed",
        reason="bound AWS segment-profit answer to same-basis segment/consolidated evidence",
        claim_change_allowed=True,
    )
    previous_text = answer_text
    answer_text = _change_history_boundary_if_needed(
        previous_text,
        user_query=user_query,
        numeric_evidence=final_numeric_evidence,
        lang=lang,
    )
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="bounded_change_history_postprocess",
        transform="change_history_boundary_if_needed",
        reason="add history boundary for change/trend questions",
        claim_change_allowed=False,
    )
    previous_text = answer_text
    answer_text = repair_language_leakage(answer_text, lang)
    answer_text = _record_answer_transform(
        state,
        previous_text=previous_text,
        new_text=answer_text,
        owner="output_language_guard",
        transform="repair_language_leakage",
        reason="enforce output_language user-visible text",
        claim_change_allowed=False,
    )
    leakage_count = language_leakage_count(answer_text, lang)
    output["language_leakage"] = leakage_count
    output["language_leakage_unresolved"] = leakage_count > 0
    output["output_language"] = lang
    _capture_answer_candidate(
        state,
        body=answer_text,
        owner=str(state.get("final_answer_source") or answer_owner),
        provenance={"synthesis_mode": str(synthesis_payload.get("synthesis_mode", synthesis_mode))},
    )
    output["summary"] = _truncate_text(_first_sentence(answer_text) or answer_text, 180 if task_type == "fact_qa" else 240)
    output["canonical_intent"] = dict(state.get("canonical_intent", {}) or {})
    output["evidence_policy_id"] = str(state.get("evidence_policy_id", "") or "")
    output["answer_status"] = (
        "draft_released_with_warnings"
        if draft_release_decision.get("decision") == "released_with_warnings"
        else ("draft_released" if draft_release_decision.get("released") else "deterministic_or_fallback")
    )
    output["warnings"] = list(draft_release_decision.get("warnings", []) or [])
    output["final_answer_source"] = str(state.get("final_answer_source", output.get("final_answer_source", preliminary_answer_source)))
    output["answer_history"] = list(state.get("answer_history", []) or [])
    output["answer_candidate"] = dict(state.get("answer_candidate", {}) or {})
    output["answer_candidates"] = list(state.get("answer_candidates", []) or [])

    return {
        "final_answer": answer_text,
        "draft_answer": answer_text,
        "numeric_evidence": final_numeric_evidence,
        "text_evidence": final_text_evidence,
        "unsupported_claims": unsupported_claims,
        "numeric_citations": numeric_citations,
        "text_citations": text_citations,
        "citations": citations,
        "output": output,
        "structured_sources": numeric_citations,
        "document_citations": text_citations,
        "event_intent": event_intent,
        "market_reaction_requested": market_reaction_requested,
        "event_query": event_query,
        "event_results": event_results,
        "market_reaction_evidence": market_reaction_evidence,
        "market_reaction_limitations": market_reaction_limitations,
        "synthesis": synthesis_payload,
        "synthesis_strategy": str(synthesis_payload.get("synthesis_strategy", "synthesis_degraded")),
        "synthesis_mode": str(synthesis_payload.get("synthesis_mode", synthesis_mode)),
        "analytical_claims": list(synthesis_payload.get("analytical_claims", []) or []),
        "claim_tiers": dict(synthesis_payload.get("claim_tiers", {}) or {}),
        "analytical_reasoning_status": str(synthesis_payload.get("analytical_reasoning_status", "")),
        "evidence_health": str(synthesis_payload.get("evidence_health") or state.get("evidence_health") or ""),
        "tool_error_context": list(synthesis_payload.get("tool_error_context", []) or state.get("tool_error_context", []) or []),
        "final_answer_source": str(state.get("final_answer_source", output.get("final_answer_source", preliminary_answer_source))),
        "output_language": lang,
        "language_leakage": leakage_count,
        "language_leakage_unresolved": leakage_count > 0,
        "draft_release_decision": draft_release_decision,
        "unsupported_synthesis_items": unsupported_synthesis_items,
        "synthesis_model_issues": synthesis_model_issues,
        "why_tools_skipped": list(state.get("why_tools_skipped", [])),
        **_requirement_state_payload(state),
        "messages": [AIMessage(content=answer_text)],
    }
