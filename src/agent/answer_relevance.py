"""Answer relevance guard for Research Planner V1."""

from __future__ import annotations

import re
from typing import Any, Mapping

from src.agent.analytical_reasoning import analytical_gap_structure_present, bracket_ref, hypothesis_marker_present
from src.agent.output_language import detect_output_language, language_leakage_terms
from src.agent.types import AnswerRelevanceDecision


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


_BOUNDED_REPAIR_INSTRUCTION = (
    "Select or request a bounded candidate for the missing requested part; "
    "do not generate new uncited analysis inside relevance repair."
)


def _required_parts(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = state.get("research_plan_used")
    if not isinstance(plan, Mapping):
        plan = {}
    parts = plan.get("required_answer_parts") or state.get("required_answer_parts") or []
    return [dict(item) for item in parts if isinstance(item, Mapping)]


def _question_type(state: Mapping[str, Any]) -> str:
    plan = state.get("research_plan_used")
    if isinstance(plan, Mapping):
        return str(plan.get("question_type") or "")
    return ""


def _missing_answer_parts(state: Mapping[str, Any]) -> list[str]:
    values = state.get("missing_required_answer_parts")
    if isinstance(values, list):
        return [str(item) for item in values if str(item)]
    suff = state.get("evidence_sufficiency")
    if isinstance(suff, Mapping):
        return [str(item) for item in suff.get("missing_required_answer_parts", []) or [] if str(item)]
    return []


def _partial_answer_parts(state: Mapping[str, Any]) -> list[str]:
    values = state.get("partial_required_answer_parts")
    if isinstance(values, list):
        return [str(item) for item in values if str(item)]
    suff = state.get("evidence_sufficiency")
    if isinstance(suff, Mapping):
        values = suff.get("partial_required_answer_parts")
        if isinstance(values, list):
            return [str(item) for item in values if str(item)]
    statuses = state.get("answer_part_status_by_id")
    if not isinstance(statuses, Mapping):
        statuses = dict(suff.get("answer_part_status_by_id", {}) or {}) if isinstance(suff, Mapping) else {}
    parts = _required_parts(state)
    required_ids = {str(part.get("id") or "") for part in parts if bool(part.get("required", True))}
    return [
        str(part_id)
        for part_id, item in dict(statuses or {}).items()
        if str(part_id) in required_ids and isinstance(item, Mapping) and str(item.get("status") or "") == "partial"
    ]


def _missing_but_analyzable_parts(state: Mapping[str, Any]) -> list[str]:
    values = state.get("missing_but_analyzable_answer_parts")
    if isinstance(values, list):
        return [str(item) for item in values if str(item)]
    suff = state.get("evidence_sufficiency")
    if isinstance(suff, Mapping):
        values = suff.get("missing_but_analyzable_answer_parts")
        if isinstance(values, list):
            return [str(item) for item in values if str(item)]
    statuses = state.get("answer_part_status_by_id")
    if not isinstance(statuses, Mapping):
        statuses = dict(suff.get("answer_part_status_by_id", {}) or {}) if isinstance(suff, Mapping) else {}
    parts = _required_parts(state)
    required_ids = {str(part.get("id") or "") for part in parts if bool(part.get("required", True))}
    return [
        str(part_id)
        for part_id, item in dict(statuses or {}).items()
        if str(part_id) in required_ids and isinstance(item, Mapping) and str(item.get("status") or "") == "missing_but_analyzable"
    ]


def _answer_part_status(state: Mapping[str, Any], part_id: str) -> dict[str, Any]:
    statuses = state.get("answer_part_status_by_id")
    if not isinstance(statuses, Mapping):
        suff = state.get("evidence_sufficiency")
        statuses = dict(suff.get("answer_part_status_by_id", {}) or {}) if isinstance(suff, Mapping) else {}
    item = dict(statuses.get(part_id, {}) or {}) if isinstance(statuses, Mapping) else {}
    return item


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def _bounded_causal_boundary_present(answer: str) -> bool:
    return _contains_any(
        answer,
        (
            "不能解释原因",
            "无法解释原因",
            "原因证据不足",
            "驱动因素证据不足",
            "不能从当前证据解释",
            "证据边界",
            "不能写成确定事实",
            "不能声称",
            "cannot explain",
            "cannot identify the drivers",
            "insufficient evidence to explain",
            "driver evidence is insufficient",
            "available evidence is insufficient",
        ),
    )


def _causal_driver_claim_present(answer: str) -> bool:
    return _contains_any(
        answer,
        (
            "因为",
            "原因是",
            "驱动",
            "受益于",
            "来自",
            "due to",
            "because",
            "driven by",
            "driver",
            "demand",
            "segment",
            "product mix",
        ),
    )


def _causal_structure_present(answer: str) -> bool:
    return _contains_any(answer, ("总体驱动", "分部/产品", "证据边界", "原因", "驱动", "drivers", "driver", "driven", "cause", "evidence boundary")) or analytical_gap_structure_present(answer)


def _segment_boundary_present(answer: str) -> bool:
    return _contains_any(
        answer,
        (
            "不能完整代表公司总收入增长原因",
            "不能完整代表总营收增长原因",
            "分部层面证据",
            "产品层面证据",
            "segment-level evidence",
            "product-level evidence",
            "cannot fully represent total company revenue growth",
        ),
    )


def _text_citation_present(answer: str) -> bool:
    return bool(re.search(r"\[T\d+\]", str(answer or "")))


def _boundary_only_fallback(answer: str) -> bool:
    lowered = str(answer or "").lower()
    has_boundary = _contains_any(
        lowered,
        (
            "证据边界",
            "证据限制",
            "scope limit",
            "evidence boundary",
            "evidence limits",
            "无法可靠",
            "只能列出边界",
        ),
    )
    if not has_boundary:
        return False
    has_analysis_layer = _contains_any(
        lowered,
        (
            "结论",
            "有限判断",
            "已验证事实",
            "已验证风险文本",
            "合理推断",
            "基于业务模型",
            "财务传导路径",
            "待验证",
            "p/e",
            "p/s",
            "fcf yield",
            "收入",
            "营收",
            "风险排序",
            "conclusion",
            "verified facts",
            "verified risk text",
            "reasonable inference",
            "limited judgment",
            "financial transmission",
            "data to verify",
        ),
    )
    return not has_analysis_layer


def _overview_numeric_validation_issues(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    plan = state.get("evidence_plan")
    requirements = list(dict(plan or {}).get("evidence_requirements", []) or []) if isinstance(plan, Mapping) else []
    legacy_numeric_core = [
        req
        for req in requirements
        if isinstance(req, Mapping)
        and str(req.get("requirement_type") or "") == "numeric"
        and str(req.get("requirement_scope") or ("core" if bool(req.get("required", True)) else "optional_context")) == "core"
        and "legacy" in {str(item) for item in req.get("merged_from", []) or []}
    ]
    if not legacy_numeric_core:
        return []
    issue_reasons = {"metric_mapping_failed", "numeric_validation_failed", "evidence_filter_mismatch"}
    return [
        dict(item)
        for item in state.get("evidence_validation_records", []) or []
        if isinstance(item, Mapping)
        and str(item.get("evidence_type") or "") == "numeric"
        and str(item.get("rejected_evidence_reason") or "") in issue_reasons
    ]


def _claims_financial_metrics_unavailable(answer: str) -> bool:
    return _contains_any(
        answer,
        (
            "没有财务指标",
            "财务指标不可得",
            "未提供财务指标",
            "no financial metrics are available",
            "financial metrics are unavailable",
            "no metrics are available",
        ),
    )


def _state_requested_dimensions(state: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    for source in (
        dict(state.get("canonical_intent", {}) or {}).get("requested_dimensions"),
        dict(state.get("analysis_plan", {}) or {}).get("requested_dimensions"),
        dict(state.get("evidence_packet", {}) or {}).get("requested_dimensions"),
        state.get("requested_dimensions"),
    ):
        for item in source or []:
            text = str(item or "").strip()
            if text and text not in out:
                out.append(text)
    return out


def _sentence_count(answer: str) -> int:
    parts = [part.strip() for part in re.split(r"[。！？!?]+|\n+", str(answer or "")) if part.strip()]
    return len(parts)


def _valuation_metric_value_or_boundary_present(answer: str) -> bool:
    lowered = str(answer or "").lower()
    has_metric_name = _contains_any(
        lowered,
        ("p/e", "p / e", "pe ", "p/s", "p / s", "ps ", "fcf yield", "市盈率", "市销率", "市值", "股价"),
    )
    has_metric_value = has_metric_name and bool(re.search(r"\d", lowered))
    has_evidence_bound_missing = _contains_any(
        lowered,
        (
            "指标缺失",
            "估值指标缺失",
            "缺少估值指标",
            "缺少可验证估值",
            "无法排序",
            "不能排序",
            "不能单一排序",
            "无法单一排序",
            "不能明确排序",
            "缺少 p/e",
            "missing p/e",
            "missing p/s",
            "missing fcf yield",
            "valuation inputs are missing",
            "cannot rank",
            "cannot produce a single ranking",
        ),
    )
    method_only = _contains_any(
        lowered,
        (
            "应围绕",
            "应该围绕",
            "应看",
            "应该看",
            "should be compared",
            "should be judged",
            "must discuss",
            "应使用",
        ),
    )
    return has_metric_value or has_evidence_bound_missing or not method_only


def _change_query_without_change_boundary(query: str, answer: str) -> bool:
    lowered_query = str(query or "").lower()
    if not _contains_any(lowered_query, ("变化", "变动", "趋势", "change", "trend")):
        return False
    lowered_answer = str(answer or "").lower()
    metric_terms: tuple[str, ...] = ()
    if _contains_any(lowered_query, ("毛利率", "gross margin")):
        metric_terms = ("毛利率", "gross margin")
    elif _contains_any(lowered_query, ("收入", "营收", "revenue", "sales")):
        metric_terms = ("收入", "营收", "revenue", "sales")
    elif _contains_any(lowered_query, ("自由现金流", "现金流", "fcf", "cash flow")):
        metric_terms = ("现金流", "自由现金流", "fcf", "cash flow")
    elif _contains_any(lowered_query, ("利润", "盈利", "profit", "income")):
        metric_terms = ("利润", "盈利", "profit", "income")
    if metric_terms and not _contains_any(lowered_answer, metric_terms):
        return True
    explicit_boundary = _contains_any(
        lowered_answer,
        (
            "无法判断变化",
            "无法判断毛利率变化",
            "只能观察最新水平",
            "缺少两期可比",
            "current evidence cannot determine the change",
            "current level only",
        ),
    )
    has_from_to = bool(re.search(r"从[^。.!?\n]{1,80}到", str(answer or ""))) or bool(
        re.search(r"\bfrom\b[^.?!\n]{1,100}\bto\b", lowered_answer)
    )
    if metric_terms:
        return not (explicit_boundary or has_from_to)
    has_change_answer = _contains_any(
        lowered_answer,
        (
            "变化",
            "变动",
            "趋势",
            "历史",
            "两期",
            "同比",
            "环比",
            "对比",
            "无法判断变化",
            "只能观察最新水平",
            "change",
            "trend",
            "historical",
            "period-over-period",
            "current level only",
        ),
    )
    return not (has_change_answer or explicit_boundary or has_from_to)


def _deterministic_dimension_relevance_failures(answer: str, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    query = str(state.get("user_query") or "")
    scope = str(state.get("analysis_scope") or dict(state.get("canonical_intent", {}) or {}).get("analysis_scope") or "")
    task_type = str(state.get("task_type") or "")
    answer_mode = str(state.get("answer_mode") or "")
    dims = set(_state_requested_dimensions(state))
    is_comparison = scope == "comparison" or task_type == "company_comparison" or answer_mode == "comparison_brief"

    if _boundary_only_fallback(answer):
        failures.append(
            {
                "code": "boundary_only_fallback_missing_analysis",
                "message": "A boundary-only fallback cannot be the main answer when bounded analysis is possible.",
            }
        )

    if is_comparison and "cash_flow_quality" in dims:
        has_cash_flow = _contains_any(
            answer,
            ("现金流", "经营现金流", "自由现金流", "fcf", "cash flow", "operating cash flow", "free cash flow", "capital expenditure", "capex"),
        )
        profit_only = _contains_any(answer, ("净利率", "盈利能力", "profitability", "net margin")) and not has_cash_flow
        if not has_cash_flow or profit_only:
            failures.append(
                {
                    "code": "comparison_cash_flow_dimension_missing",
                    "message": "Cash-flow-quality comparison must discuss operating cash flow, free cash flow, capex, or FCF margin instead of profitability only.",
                }
            )

    if is_comparison and "valuation_and_risk_boundary" in dims:
        has_valuation = _contains_any(answer, ("估值", "valuation"))
        has_metric = _contains_any(answer, ("p/e", "p / e", "p/s", "p / s", "fcf yield", "市值", "股价", "market cap", "share price"))
        has_judgment = _contains_any(answer, ("更高", "更低", "分歧", "无法排序", "不能排序", "不能单一排序", "无法单一排序", "higher", "lower", "mixed", "cannot rank", "cannot produce a single ranking", "single ranking is not supported"))
        risk_only = _contains_any(answer, ("risk factors", "风险因素", "普通风险", "竞争风险")) and not (has_valuation and has_metric)
        method_empty = not _valuation_metric_value_or_boundary_present(answer)
        if not (has_valuation and has_metric) or risk_only or method_empty or not has_judgment:
            failures.append(
                {
                    "code": "comparison_valuation_risk_dimension_missing",
                    "message": "Valuation-risk comparison must discuss actual valuation inputs and include a per-metric judgment or evidence-bound missing-input boundary.",
                }
            )

    if is_comparison and "revenue_quality" in dims:
        has_growth = _contains_any(answer, ("增长", "收入增速", "revenue growth", "收入质量", "营收质量", "growth quality"))
        margin_only = _contains_any(answer, ("净利率", "盈利能力", "profitability", "net margin")) and not has_growth
        if not has_growth or margin_only:
            failures.append(
                {
                    "code": "comparison_revenue_growth_dimension_missing",
                    "message": "Growth-quality comparison must discuss growth or revenue quality instead of net margin only.",
                }
            )

    if is_comparison and "moat_and_competitive_risk" in dims:
        lowered_answer = str(answer or "").lower()
        numeric_missing_template = _contains_any(
            lowered_answer,
            (
                "缺少可追溯的数值证据",
                "缺少数值证据",
                "无法输出可靠结论",
                "traceable numeric evidence is missing",
                "numeric evidence is missing",
            ),
        )
        has_risk_content = _contains_any(lowered_answer, ("风险", "risk", "risk text", "风险文本", "披露"))
        if numeric_missing_template or not has_risk_content:
            failures.append(
                {
                    "code": "comparison_risk_text_dimension_missing",
                    "message": "Risk comparison must be bounded by risk text, not blocked for missing numeric evidence.",
                }
            )

    constraints = state.get("format_constraints") if isinstance(state, Mapping) else {}
    constraints = dict(constraints or {}) if isinstance(constraints, Mapping) else {}
    one_sentence_required = bool(constraints.get("one_sentence")) or bool(
        re.search(r"(一句话|一段话|用一句|只用一句|one sentence|single sentence|in one sentence)", query.lower())
    )
    if one_sentence_required:
        zh_chars = len(re.findall(r"[\u4e00-\u9fff]", str(answer or "")))
        if _sentence_count(answer) > 1 or zh_chars > 120:
            failures.append(
                {
                    "code": "one_sentence_constraint_violated",
                    "message": "The query requested one sentence, but the answer has multiple sentences or is too long.",
                }
            )
    if _change_query_without_change_boundary(query, answer):
        failures.append(
            {
                "code": "change_query_history_boundary_missing",
                "message": "Change/trend questions must compare history or state that current evidence only supports the latest level.",
            }
        )
    return failures


def judge_answer_relevance(answer: str, state: Mapping[str, Any]) -> AnswerRelevanceDecision:
    """Judge whether the answer covers the validated ResearchPlan."""
    output_language = str(state.get("output_language") or detect_output_language(str(state.get("user_query") or "")))
    leaked_terms = language_leakage_terms(answer, output_language)
    if leaked_terms:
        failure = {
            "code": "language_leakage",
            "message": "English output contains Chinese section labels or risk labels.",
            "leaked_terms": leaked_terms,
        }
        return AnswerRelevanceDecision(
            decision="repairable",
            status="failed",
            route="repair_answer",
            action="downgrade_to_bounded",
            deterministic_relevance_failures=[failure],
            llm_relevance_notes=[],
            warnings=[failure],
            repair_instructions=["Repair user-visible language leakage without adding new facts."],
            recommended_actions=["downgrade_to_bounded"],
            public_summary="Answer language does not match the requested output language.",
        )
    preflight_failures = _deterministic_dimension_relevance_failures(answer, state)
    if preflight_failures:
        return AnswerRelevanceDecision(
            decision="repairable",
            status="failed",
            route="repair_answer",
            action="downgrade_to_bounded",
            deterministic_relevance_failures=preflight_failures,
            llm_relevance_notes=[],
            warnings=preflight_failures,
            repair_instructions=[_BOUNDED_REPAIR_INSTRUCTION],
            recommended_actions=["downgrade_to_bounded"],
            public_summary="Answer did not satisfy the requested dimension or format constraint.",
        )
    parts = _required_parts(state)
    if not parts:
        return AnswerRelevanceDecision(decision="not_run", status="not_run", route="finalize")

    question_type = _question_type(state)
    missing_evidence_parts = set(_missing_answer_parts(state))
    partial_evidence_parts = set(_partial_answer_parts(state))
    analyzable_gap_parts = set(_missing_but_analyzable_parts(state))
    covered: list[str] = []
    missing: list[str] = []
    warnings: list[dict[str, Any]] = []
    repair: list[str] = []
    deterministic_failures: list[dict[str, Any]] = []

    if question_type == "causal_explanation":
        has_boundary = _bounded_causal_boundary_present(answer)
        has_driver_claim = _causal_driver_claim_present(answer)
        has_text_citation = _text_citation_present(answer)
        has_analytical_framework = analytical_gap_structure_present(answer)
        has_hypothesis_marker = hypothesis_marker_present(answer)
        driver_gap = "identify_growth_drivers" in missing_evidence_parts or "identify_growth_drivers" in analyzable_gap_parts
        if has_driver_claim and not has_text_citation and not has_hypothesis_marker:
            deterministic_failures.append(
                {
                    "code": "driver_claim_without_text_evidence",
                    "message": "Definitive driver claims require validated filing text citations; uncited content must be marked as hypothesis.",
                }
            )
        if not _causal_structure_present(answer) and not has_boundary:
            deterministic_failures.append(
                {
                    "code": "causal_structure_missing",
                    "message": "Why/driver question did not receive a cause structure or evidence-boundary statement.",
                }
            )
        if driver_gap and not has_boundary and not has_analytical_framework:
            deterministic_failures.append(
                {
                    "code": "driver_evidence_boundary_missing",
                    "message": "Driver evidence is missing and the answer did not provide an evidence-boundary analytical framework.",
                }
            )
            repair.append(_BOUNDED_REPAIR_INSTRUCTION)
        if driver_gap and has_boundary and not has_analytical_framework:
            deterministic_failures.append(
                {
                    "code": "causal_analysis_framework_missing",
                    "message": "A why question cannot be answered by only saying evidence is insufficient; it needs a bounded analysis framework.",
                }
            )
            repair.append(_BOUNDED_REPAIR_INSTRUCTION)
        if driver_gap and has_analytical_framework:
            warnings.append(
                {
                    "code": "driver_evidence_missing_but_analyzable",
                    "message": "Direct driver evidence is incomplete; answer is released as analytical_with_gaps.",
                }
            )
        if "quantify_growth" in missing_evidence_parts and not has_boundary:
            deterministic_failures.append(
                {
                    "code": "growth_quantification_missing",
                    "message": "Growth quantification is missing and the answer does not downgrade the conclusion.",
                }
            )
            repair.append(_BOUNDED_REPAIR_INSTRUCTION)
        if "quantify_growth" in partial_evidence_parts:
            warnings.append(
                {
                    "code": "growth_quantification_partial",
                    "message": "Growth quantification is incomplete; the answer can only be released with a warning.",
                }
            )
        if "identify_growth_drivers" in partial_evidence_parts:
            driver_status = _answer_part_status(state, "identify_growth_drivers")
            levels = set(str(item) for item in driver_status.get("driver_levels", []) or [] if str(item))
            if levels and "company_level_driver" not in levels and not _segment_boundary_present(answer):
                warnings.append(
                    {
                        "code": "driver_level_partial",
                        "message": "Only segment/product driver evidence is available; answer must keep that boundary visible.",
                    }
                )
            elif "identify_growth_drivers" in partial_evidence_parts:
                warnings.append(
                    {
                        "code": "driver_evidence_partial",
                        "message": "Driver evidence is partial and cannot receive a clean pass.",
                    }
                )
        if deterministic_failures:
            return AnswerRelevanceDecision(
                decision="repairable",
                status="failed",
                route="repair_answer",
                action="downgrade_to_bounded",
                covered_answer_parts=["quantify_growth"] if "[N" in answer else [],
                missing_answer_parts=sorted(missing_evidence_parts),
                missing_required_answer_parts=sorted(missing_evidence_parts),
                partial_answer_parts=sorted(partial_evidence_parts),
                partial_required_answer_parts=sorted(partial_evidence_parts),
                missing_but_analyzable_answer_parts=sorted(analyzable_gap_parts),
                deterministic_relevance_failures=deterministic_failures,
                llm_relevance_notes=[],
                warnings=warnings or [{"code": "causal_relevance_failed", "message": "Answer did not cover the causal question."}],
                repair_instructions=repair or [_BOUNDED_REPAIR_INSTRUCTION],
                recommended_actions=["downgrade_to_bounded"],
                public_summary="Answer did not cover the causal part of the question.",
            )
        if warnings:
            analytical_status = "analytical_with_gaps" if driver_gap and has_analytical_framework else "passed_with_warnings"
            return AnswerRelevanceDecision(
                decision="warning",
                status=analytical_status,
                route="finalize",
                covered_answer_parts=[
                    str(part.get("id")) for part in parts if str(part.get("id")) and str(part.get("id")) not in missing_evidence_parts
                ],
                missing_answer_parts=sorted(missing_evidence_parts),
                missing_required_answer_parts=sorted(missing_evidence_parts),
                partial_answer_parts=sorted(partial_evidence_parts),
                partial_required_answer_parts=sorted(partial_evidence_parts),
                missing_but_analyzable_answer_parts=sorted(analyzable_gap_parts),
                deterministic_relevance_failures=[],
                llm_relevance_notes=[],
                warnings=warnings,
                public_summary="Answer covers the causal question with explicit evidence limitations.",
            )
        if has_text_citation and has_driver_claim:
            return AnswerRelevanceDecision(
                decision="passed",
                status="passed",
                route="finalize",
                covered_answer_parts=[str(part.get("id")) for part in parts if str(part.get("id"))],
                missing_required_answer_parts=[],
                partial_required_answer_parts=[],
                deterministic_relevance_failures=[],
                llm_relevance_notes=[],
                public_summary="Answer covers required causal explanation parts.",
            )
        return AnswerRelevanceDecision(
            decision="repairable",
            status="failed",
            route="repair_answer",
            action="downgrade_to_bounded",
            covered_answer_parts=["quantify_growth"] if "[N" in answer else [],
            missing_answer_parts=["identify_growth_drivers"],
            missing_required_answer_parts=["identify_growth_drivers"],
            partial_required_answer_parts=sorted(partial_evidence_parts),
            missing_but_analyzable_answer_parts=sorted(analyzable_gap_parts),
            deterministic_relevance_failures=[
                {"code": "driver_explanation_missing", "message": "Answer did not explain growth drivers with text evidence."}
            ],
            llm_relevance_notes=[],
            warnings=[{"code": "driver_explanation_missing", "message": "Answer did not explain growth drivers with text evidence."}],
            repair_instructions=[_BOUNDED_REPAIR_INSTRUCTION],
            recommended_actions=["downgrade_to_bounded"],
            public_summary="Answer did not cover the causal part of the question.",
        )

    if question_type == "overview":
        numeric_validation_issues = _overview_numeric_validation_issues(state)
        if numeric_validation_issues and _claims_financial_metrics_unavailable(answer):
            return AnswerRelevanceDecision(
                decision="repairable",
                status="failed",
                route="repair_answer",
                action="downgrade_to_bounded",
                deterministic_relevance_failures=[
                    {
                        "code": "overview_numeric_validation_issue_misstated_as_unavailable",
                        "message": "Overview answer says financial metrics are unavailable, but numeric retrieval/validation failed and should be disclosed as a validation issue.",
                    }
                ],
                warnings=[{"code": "overview_numeric_validation_issue", "message": "Legacy numeric requirements existed but failed validation."}],
                repair_instructions=["Select a bounded_analysis answer that separates validated facts, reasonable inference, hypotheses to verify, and evidence boundary."],
                recommended_actions=["downgrade_to_bounded"],
                public_summary="Overview answer needs repair because numeric validation failure was misstated as unavailable data.",
            )
        if numeric_validation_issues:
            warnings = [
                {
                    "code": "overview_numeric_validation_issue",
                    "message": "Legacy numeric requirements existed but some structured evidence failed validation.",
                }
            ]
            return AnswerRelevanceDecision(
                decision="warning",
                status="passed_with_warnings",
                route="finalize",
                covered_answer_parts=[
                    str(part.get("id")) for part in parts if str(part.get("id")) and str(part.get("id")) not in missing_evidence_parts
                ],
                missing_answer_parts=sorted(missing_evidence_parts),
                missing_required_answer_parts=sorted(missing_evidence_parts),
                partial_answer_parts=sorted(partial_evidence_parts),
                partial_required_answer_parts=sorted(partial_evidence_parts),
                warnings=warnings,
                public_summary="Overview answer released with numeric validation warnings.",
            )

    for part in parts:
        part_id = str(part.get("id") or "")
        if part_id and part_id not in missing_evidence_parts:
            covered.append(part_id)
        elif part_id:
            missing.append(part_id)
    partial = sorted(partial_evidence_parts)
    if missing:
        return AnswerRelevanceDecision(
            decision="warning",
            status="passed_with_warnings",
            route="finalize",
            covered_answer_parts=covered,
            missing_answer_parts=missing,
            missing_required_answer_parts=missing,
            partial_answer_parts=partial,
            partial_required_answer_parts=partial,
            llm_relevance_notes=[],
            warnings=[{"code": "answer_part_evidence_missing", "message": "Some required answer parts have insufficient evidence."}],
            public_summary="Answer released with explicit evidence limitations.",
        )
    if partial:
        return AnswerRelevanceDecision(
            decision="warning",
            status="passed_with_warnings",
            route="finalize",
            covered_answer_parts=covered,
            partial_answer_parts=partial,
            partial_required_answer_parts=partial,
            llm_relevance_notes=[],
            warnings=[{"code": "answer_part_evidence_partial", "message": "Some required answer parts are only partially satisfied."}],
            public_summary="Answer covers the requested parts with evidence limitations.",
        )
    return AnswerRelevanceDecision(
        decision="passed",
        status="passed",
        route="finalize",
        covered_answer_parts=covered,
        llm_relevance_notes=[],
        public_summary="Answer covers the required answer parts.",
    )


def bounded_causal_fallback_answer(state: Mapping[str, Any]) -> str:
    """Build a public tiered analytical answer for causal questions with evidence gaps."""
    user_query = str(state.get("user_query") or "")
    lang = str(state.get("output_language") or detect_output_language(user_query))
    companies = [str(item).upper() for item in state.get("companies", []) or [] if str(item)]
    company = companies[0] if companies else "该公司"
    numeric_refs: list[str] = []
    for item in list(state.get("numeric_evidence", []) or []):
        if not isinstance(item, Mapping):
            continue
        ref = str(item.get("evidence_id") or "")
        metric = str(item.get("metric") or "")
        if ref.startswith("N") and metric in {"revenue", "revenue_growth"} and ref not in numeric_refs:
            numeric_refs.append(ref)
    citation = "".join(bracket_ref(ref) for ref in numeric_refs[:2])
    partial_parts = set(_partial_answer_parts(state))
    missing_parts = set(_missing_answer_parts(state)) | set(_missing_but_analyzable_parts(state))
    growth_partial = "quantify_growth" in partial_parts or "quantify_growth" in missing_parts
    tool_errors = list(state.get("tool_error_context", []) or [])
    if lang == "zh":
        verified = f"- 已验证收入证据可作为增长量化的起点。{citation}" if citation else "- 当前没有可引用的收入或 driver 证据。"
        if growth_partial:
            verified += "\n- 总营收增长量化仍不完整：需要可比期间收入、增长计算，或 filing 中明确总营收增长率。"
        if tool_errors:
            verified += "\n- 本轮部分工具或检索退化，因此不能把缺失披露当作不存在。"
        network_focus = any(term in user_query.lower() for term in ("网络", "network", "infiniband", "ethernet", "nvlink"))
        judgment = (
            f"有限判断：{company} 网络业务增长大概率与 AI 集群建设、GPU 集群互连以及 NVLink/InfiniBand/Ethernet 需求有关；随后必须说明证据边界。"
            if network_focus
            else f"可以对 {company} 的营收增长做分层分析，但不能把未验证因素写成确定原因。"
        )
        lines = [
            "结论",
            judgment,
            "",
            "已验证事实",
            verified,
            "",
            "合理推断",
            "- 收入数字本身只能支持“发生了增长/规模变化”，不能单独证明原因。",
            "- 若后续补到公司层面 driver text，才能把需求、产品周期、分部增长或客户行为写成公司披露支持的原因。",
            "",
            "待验证假设",
            "- 待验证假设：云厂商和企业 AI 资本开支是否继续扩张。",
            "- 待验证假设：Blackwell / Hopper 等产品周期是否推动出货、ASP 或收入确认。",
            "- 待验证假设：网络互连产品是否随 GPU 集群建设放量。",
            "- 待验证假设：供应链产能释放是否让收入确认加速。",
            "",
            "待验证数据",
            "- 待验证假设：增长也可能来自价格/产品组合、收入确认节奏、供给恢复或一次性订单节奏。",
            "- 需要验证：分部增长是否能代表总公司增长，而不是局部产品线表现。",
            "- 分部收入增速、数据中心与网络收入、ASP/出货量、客户 AI capex、递延收入和订单节奏。",
            "",
            "证据边界",
            "- 直接 driver text 不完整时，不能声称营收增长确定由 AI 需求、产品组合或客户行为驱动。",
            "- 当前不能量化每个因素贡献比例，也不能判断增长是否可持续。",
        ]
    else:
        verified = f"- Verified revenue evidence can be used as the starting point for growth quantification. {citation}" if citation else "- No citable revenue or driver evidence is available in this run."
        if growth_partial:
            verified += "\n- Total revenue growth quantification remains incomplete: comparator revenue plus a growth calculation, or filing text with total revenue growth, is needed."
        if tool_errors:
            verified += "\n- Some retrieval or tool execution degraded in this run, so missing disclosures should not be treated as non-existence."
        lines = [
            "Conclusion",
            f"{company}'s revenue growth can be analyzed in tiers, but unverified factors cannot be written as definitive causes.",
            "",
            "Verified Facts",
            verified,
            "",
            "Reasonable Inference",
            "- Revenue numbers by themselves can show that growth or scale changed, but they cannot prove the cause.",
            "- Company-level driver text is needed before demand, product cycle, segment growth, or customer behavior can be stated as a disclosed cause.",
            "- Hypothesis to verify: growth may also reflect pricing/product mix, revenue-recognition timing, supply recovery, or one-off order timing.",
            "- Verify whether segment growth represents total-company growth rather than local product-line performance.",
            "",
            "Data to Verify",
            "- Hypothesis to verify: whether cloud and enterprise AI capex continued to expand.",
            "- Hypothesis to verify: whether Blackwell / Hopper product cycles lifted shipments, ASP, or revenue recognition.",
            "- Hypothesis to verify: whether networking interconnect products scaled with GPU cluster buildouts.",
            "- Hypothesis to verify: whether supply availability accelerated revenue recognition.",
            "- Segment revenue growth, Data Center and networking revenue, ASP/shipments, customer AI capex, deferred revenue, and order timing.",
            "",
            "Evidence Boundary",
            "- Without direct driver text, the answer cannot claim revenue growth was definitively driven by AI demand, product mix, or customer behavior.",
            "- Current evidence cannot quantify each factor's contribution or assess growth durability.",
        ]
    return "\n".join(lines)
