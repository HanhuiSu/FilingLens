"""Run the 12-query financial analyst regression.

The runner executes each query, writes a trace-shaped debug bundle, and applies
deterministic gates for pass / accepted_warning / fail. It is intentionally
separate from the generic answer benchmark because these cases verify specific
planner-legacy coverage and evidence-quality contracts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.agent.answer_contract import check_answer_evidence_contract  # noqa: E402
from src.agent.progress import read_trace_payload, validate_trace_id, write_trace_payload  # noqa: E402


DEFAULT_BENCHMARK = ROOT / "eval" / "financial_agent_regression_12.jsonl"
DEFAULT_OUT_DIR = ROOT / "eval" / "reports" / "financial_agent_regression_12"

INVESTMENT_ADVICE_RE = re.compile(
    r"\b(strong buy|buy|sell|hold|target price|price target)\b|建议买入|可以买|建议卖出|建议持有|目标价",
    re.I,
)
CERTAIN_FORECAST_RE = re.compile(r"一定会|必然会|肯定会|guaranteed|will definitely", re.I)
SECTION_HEADER_RE = re.compile(r"(?i)^\s*(?:part\s+[ivx]+\.?|item\s+\d+[a-z]?\.?)\s*$")
TEMPLATE_REFUSAL_RE = re.compile(
    r"证据不足以支持一个完整且通过契约校验的结论|当前候选答案未通过契约校验|"
    r"candidate answer did not pass the evidence contract|fully evidence-supported answer",
    re.I,
)
VALUATION_METRIC_RE = re.compile(
    r"\b(?:P/E|PE\s*ratio|P/S|PS\s*ratio|FCF\s*yield|free[- ]cash[- ]flow yield)\b|"
    r"PE比率|PS比率|市盈率|市销率|自由现金流收益率|FCF收益率",
    re.I,
)
VALUATION_MISSING_RE = re.compile(
    r"缺少估值证据|估值证据不足|当前估值证据不足|无法判断(?:价格)?是否便宜或昂贵|"
    r"无法判断估值水平|valuation evidence (?:is )?missing|valuation evidence is insufficient",
    re.I,
)
EXPLICIT_RISK_COMPARISON_RE = re.compile(
    r"(?:AMZN|Amazon|亚马逊|NVDA|NVIDIA|英伟达).{0,40}风险.{0,16}(?:更大|更高|较大|higher|greater)|"
    r"风险.{0,40}(?:AMZN|Amazon|亚马逊|NVDA|NVIDIA|英伟达).{0,16}(?:更大|更高|较大|higher|greater)",
    re.I,
)
BOUNDED_RISK_COMPARISON_RE = re.compile(
    r"(?:无法|不能|证据不足|取决于).{0,50}(?:谁的风险更大|风险大小|风险高低|风险排序|风险比较)|"
    r"(?:谁的风险更大|风险大小|风险高低|风险排序|风险比较).{0,50}(?:无法|不能|证据不足|取决于)|"
    r"(?:不能|无法).{0,20}强行.{0,20}(?:判断|比较).{0,20}风险|"
    r"(?:cannot|insufficient|depends).{0,60}(?:which.*risk|risk ranking|risk comparison)",
    re.I,
)
NEGATION_TERMS = (
    "不",
    "不能",
    "无法",
    "不构成",
    "不提供",
    "不会",
    "not",
    "cannot",
    "can't",
    "do not",
    "does not",
    "without",
    "no ",
)
NEGATION_CONTRAST_TERMS = ("但", "但是", "不过", "然而", "可是", "but", "however", ";", "；")
ACCEPTED_CONTRACT_WARNING_CODES = {
    "caveat_not_visible",
    "missing_medium_confidence_source_caveat",
    "missing_growth_quantification_caveat",
    "missing_segment_scope_caveat",
    "missing_sustainability_caveat",
    "missing_reconciliation_caveat",
}
ACCEPTED_WARNING_CODES = {
    *ACCEPTED_CONTRACT_WARNING_CODES,
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _load_cases(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("case_id") or not case.get("query"):
            raise ValueError(f"{path}:{lineno} must include case_id and query")
        cases.append(case)
        if limit and len(cases) >= limit:
            break
    return cases


def _answer_text(trace: Mapping[str, Any]) -> str:
    output = _as_dict(trace.get("output"))
    report = _as_dict(trace.get("report")) or _as_dict(output.get("report"))
    return "\n".join(
        part
        for part in (
            str(trace.get("final_answer") or ""),
            str(trace.get("answer") or ""),
            str(output.get("summary") or ""),
            str(report.get("markdown") or ""),
        )
        if part
    ).strip()


def _run_direct(query: str, trace_id: str) -> dict[str, Any]:
    from src.agent.graph import compile_agent

    result = dict(compile_agent().invoke({"user_query": query, "trace_id": trace_id}) or {})
    result["trace_id"] = trace_id
    write_trace_payload(trace_id, result)
    return result


def _run_api(query: str, trace_id: str, api_base: str) -> dict[str, Any]:
    with httpx.Client(timeout=420) as client:
        response = client.post(f"{api_base.rstrip('/')}/chat", json={"query": query, "client_trace_id": trace_id})
        response.raise_for_status()
        body = response.json()
        used_trace_id = validate_trace_id(str(body.get("trace_id") or trace_id))
        trace_response = client.get(f"{api_base.rstrip('/')}/trace/{used_trace_id}")
        trace_response.raise_for_status()
        trace = trace_response.json()
        trace["chat_response"] = body
        return trace


def _validation_records(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [dict(item) for item in _as_list(trace.get("evidence_validation_records")) if isinstance(item, Mapping)]
    if records:
        return records
    out: list[dict[str, Any]] = []
    for item in _as_list(trace.get("evidence_collection_results")):
        if not isinstance(item, Mapping):
            continue
        out.append(
            {
                "requirement_id": item.get("requirement_id", ""),
                "evidence_type": item.get("evidence_type", ""),
                "tool_returned_count": int(item.get("tool_returned_count") or len(_as_list(item.get("items")))),
                "validated_evidence_count": int(item.get("validated_evidence_count") or len(_as_list(item.get("items")))),
                "rejected_evidence_reason": str(item.get("rejected_evidence_reason") or item.get("failure_reason") or ""),
                "status": str(item.get("status") or ""),
            }
        )
    return out


def _numeric_counts(trace: Mapping[str, Any]) -> tuple[int, int]:
    returned = 0
    validated = 0
    for record in _validation_records(trace):
        if str(record.get("evidence_type") or "") != "numeric":
            continue
        returned += int(record.get("tool_returned_count") or 0)
        validated += int(record.get("validated_evidence_count") or 0)
    if returned or validated:
        return returned, validated
    numeric = _as_list(_as_dict(trace.get("output")).get("numeric_evidence"))
    packet_numeric = _as_list(_as_dict(trace.get("evidence_packet")).get("numeric_table"))
    count = len(numeric) + len(packet_numeric)
    return count, count


def _text_quality_issues(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for result in _as_list(trace.get("evidence_collection_results")):
        if not isinstance(result, Mapping) or str(result.get("evidence_type") or "") != "text":
            continue
        if str(result.get("status") or "") == "satisfied":
            for item in _as_list(result.get("items")):
                if not isinstance(item, Mapping):
                    continue
                snippet = str(item.get("supporting_snippet") or item.get("text_snippet") or item.get("snippet") or "").strip()
                if SECTION_HEADER_RE.match(snippet):
                    issues.append({"requirement_id": result.get("requirement_id", ""), "reason": "low_information_text_evidence", "snippet": snippet})
        if str(result.get("failure_reason") or "") == "low_information_text_evidence":
            issues.append({"requirement_id": result.get("requirement_id", ""), "reason": "low_information_text_evidence"})
    return issues


def _tool_errors(trace: Mapping[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    for key in ("tool_call_results", "requirement_calls", "tool_results"):
        for item in _as_list(trace.get(key)):
            if isinstance(item, Mapping) and (item.get("error") or _as_dict(item.get("error")).get("message")):
                errors.append({"source": key, **dict(item)})
    return errors


def _dimension_ids(trace: Mapping[str, Any]) -> set[str]:
    ids: set[str] = set()
    for source in (
        trace.get("dimension_status_by_id"),
        trace.get("dimension_status_map"),
        _as_dict(trace.get("evidence_packet")).get("dimension_status_map"),
    ):
        if isinstance(source, Mapping):
            ids.update(str(key) for key in source.keys() if str(key))
    selected = _as_dict(trace.get("selected_analysis_framework"))
    ids.update(str(item) for item in _as_list(selected.get("active_dimension_ids")) if str(item))
    return ids


def _contains_group(text: str, group: list[Any]) -> bool:
    lowered = text.lower()
    return any(str(term).lower() in lowered for term in group if str(term))


def _context_is_negated(text: str, start: int) -> bool:
    before = str(text or "")[max(0, start - 56) : start].lower()
    for term in sorted(NEGATION_TERMS, key=len, reverse=True):
        needle = term.lower()
        idx = before.rfind(needle)
        if idx < 0:
            continue
        between = before[idx + len(needle) :]
        if any(contrast in between for contrast in NEGATION_CONTRAST_TERMS):
            return False
        if len(between) <= 34:
            return True
    return False


def _has_unnegated_match(text: str, pattern: re.Pattern[str]) -> bool:
    return any(not _context_is_negated(text, match.start()) for match in pattern.finditer(text or ""))


def _contains_unnegated_term(text: str, term: str) -> bool:
    lowered = str(text or "").lower()
    needle = str(term or "").lower()
    start = 0
    while needle:
        idx = lowered.find(needle, start)
        if idx < 0:
            return False
        if not _context_is_negated(text, idx):
            return True
        start = idx + len(needle)
    return False


def _root_cause(code: str) -> str:
    if code in {"overview_not_merged", "legacy_core_not_retained", "overview_dimension_missing"}:
        return "planner coverage"
    if code in {"research_plan_missing_from_debug", "requirement_merge_summary_missing"}:
        return "trace UI"
    if code in {"numeric_missing", "returned_numeric_rejected_without_reason", "claims_no_metrics_despite_returned_numeric"}:
        return "numeric validation"
    if code in {"low_information_text_evidence"}:
        return "text quality"
    if code in {"unsupported_claims_present", "contract_blocked", "contract_failed", "final_route_blocked", "scope_overclaim_violation"}:
        return "contract"
    if code in {
        "answer_missing_required_content",
        "answer_too_short",
        "causal_structure_missing",
        "question_not_answered",
        "template_refusal",
        "comparison_not_answered",
    }:
        return "relevance"
    if code in {"investment_advice", "target_price_or_certain_forecast"}:
        return "safety"
    if code in {"valuation_contradiction"}:
        return "synthesis"
    return "synthesis"


def _add_issue(issues: list[dict[str, Any]], code: str, severity: str, message: str, details: Any = None) -> None:
    item = {"code": code, "severity": severity, "root_cause": _root_cause(code), "message": message}
    if details is not None:
        item["details"] = details
    issues.append(item)


def _trace_contract_decision(trace: Mapping[str, Any]) -> str:
    decision = _as_dict(trace.get("contract_decision"))
    result = _as_dict(trace.get("contract_result"))
    output = _as_dict(trace.get("output"))
    output_contract = _as_dict(output.get("contract"))
    return str(
        decision.get("decision")
        or result.get("decision")
        or output_contract.get("decision")
        or ""
    )


def _trace_final_route(trace: Mapping[str, Any]) -> str:
    output = _as_dict(trace.get("output"))
    return str(trace.get("final_route") or output.get("final_route") or "")


def _planner_source(trace: Mapping[str, Any]) -> str:
    return str(trace.get("research_plan_source") or _as_dict(trace.get("research_plan_validation")).get("source") or "")


def _has_valuation_contradiction(answer: str) -> bool:
    return bool(VALUATION_METRIC_RE.search(answer or "") and VALUATION_MISSING_RE.search(answer or ""))


def _comparison_answered_with_boundary(answer: str) -> bool:
    text = answer or ""
    has_both = bool(re.search(r"AMZN|Amazon|亚马逊", text, flags=re.I)) and bool(re.search(r"NVDA|NVIDIA|英伟达", text, flags=re.I))
    if not has_both or "风险" not in text and "risk" not in text.lower():
        return False
    return bool(EXPLICIT_RISK_COMPARISON_RE.search(text) or BOUNDED_RISK_COMPARISON_RE.search(text))


def judge_case(case: Mapping[str, Any], trace: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    answer = _answer_text(trace)
    issues: list[dict[str, Any]] = []
    family = str(case.get("family") or "")
    plan_used = _as_dict(trace.get("research_plan_used"))
    coverage = _as_dict(trace.get("plan_coverage_decision"))
    merge_summary = _as_dict(trace.get("requirement_merge_summary"))
    returned_numeric, validated_numeric = _numeric_counts(trace)

    final_contract_status = str(trace.get("final_contract_status") or "")
    contract_status = str(trace.get("final_contract_status") or trace.get("contract_status") or "")
    contract_decision = _trace_contract_decision(trace)
    final_route = _trace_final_route(trace)
    if contract_status in {"blocked", "failed"} or contract_decision in {"blocked", "failed"}:
        _add_issue(issues, "contract_blocked", "P0", "Runtime contract blocked the answer.")
    if final_route == "blocked":
        _add_issue(issues, "final_route_blocked", "P0", "Final route is blocked.")
    contract_violations = [dict(item) for item in _as_list(contract.get("violations")) if isinstance(item, Mapping)]
    hard_contract_violations = [
        item
        for item in contract_violations
        if str(item.get("type") or item.get("code") or "") not in ACCEPTED_CONTRACT_WARNING_CODES
    ]
    warning_contract_violations = [
        item
        for item in contract_violations
        if str(item.get("type") or item.get("code") or "") in ACCEPTED_CONTRACT_WARNING_CODES
    ]
    if hard_contract_violations:
        _add_issue(issues, "contract_failed", "P0", "Post-hoc answer contract has hard violations.", hard_contract_violations[:5])
    if warning_contract_violations:
        for violation in warning_contract_violations[:5]:
            _add_issue(
                issues,
                str(violation.get("type") or violation.get("code") or "contract_warning"),
                "P1",
                "Post-hoc answer contract warning.",
                violation,
            )
    if _as_list(trace.get("unsupported_claims")):
        _add_issue(issues, "unsupported_claims_present", "P0", "Unsupported claims are present.", _as_list(trace.get("unsupported_claims"))[:5])
    if _has_unnegated_match(answer, INVESTMENT_ADVICE_RE):
        _add_issue(issues, "investment_advice", "P0", "Answer contains buy/sell/hold/target-price language.")
    if CERTAIN_FORECAST_RE.search(answer):
        _add_issue(issues, "target_price_or_certain_forecast", "P0", "Answer uses certain forward-looking language.")
    if not answer:
        _add_issue(issues, "question_not_answered", "P0", "Answer is empty.")
    if TEMPLATE_REFUSAL_RE.search(answer) and family not in {"investment_boundary"}:
        _add_issue(issues, "template_refusal", "P0", "Non-safety query returned a template contract/evidence refusal.")

    if plan_used and not _as_dict(trace.get("research_plan_used")):
        _add_issue(issues, "research_plan_missing_from_debug", "P1", "Research plan was used but missing from debug bundle.")
    if plan_used and not merge_summary:
        _add_issue(issues, "requirement_merge_summary_missing", "P1", "Requirement merge summary is missing.")

    for record in _validation_records(trace):
        if (
            str(record.get("evidence_type") or "") == "numeric"
            and int(record.get("tool_returned_count") or 0) > 0
            and int(record.get("validated_evidence_count") or 0) == 0
            and not str(record.get("rejected_evidence_reason") or "")
        ):
            _add_issue(issues, "returned_numeric_rejected_without_reason", "P1", "Numeric evidence returned but rejected without reason.", record)

    text_issues = _text_quality_issues(trace)
    if text_issues:
        _add_issue(issues, "low_information_text_evidence", "P1", "Low-information filing text was not usable evidence.", text_issues[:5])

    required_groups = [group for group in _as_list(case.get("required_answer_terms")) if isinstance(group, list)]
    missing_groups = [group for group in required_groups if not _contains_group(answer, group)]
    if missing_groups:
        _add_issue(issues, "answer_missing_required_content", "P1", "Answer misses required content groups.", missing_groups[:4])
    min_chars = int(case.get("min_answer_chars") or 0)
    if min_chars and len(answer) < min_chars:
        _add_issue(issues, "answer_too_short", "P1", f"Answer is shorter than {min_chars} characters.", {"answer_chars": len(answer)})
    for term in _as_list(case.get("forbidden_answer_terms")):
        if _contains_unnegated_term(answer, str(term)):
            _add_issue(issues, "question_not_answered", "P0", f"Forbidden answer term present: {term}")

    expected_companies = {str(item).upper() for item in _as_list(case.get("companies")) if str(item)}
    actual_companies = {str(item).upper() for item in _as_list(trace.get("companies")) if str(item)}
    if expected_companies and not expected_companies.issubset(actual_companies | {ticker for ticker in expected_companies if ticker in answer.upper()}):
        _add_issue(issues, "question_not_answered", "P0", "Expected company coverage is missing.", {"expected": sorted(expected_companies), "actual": sorted(actual_companies)})

    if family in {"overview", "risk", "composite", "comparison", "fcf_causal"}:
        coverage_strategy = str(coverage.get("strategy") or "")
        coverage_reason = str(coverage.get("reason") or "")
        research_attempted = coverage_strategy in {"merge", "replace", "augment_only"} or coverage_reason not in {
            "planner_invalid",
            "research_planner_disabled",
            "shadow_mode_uses_legacy_evidence_plan",
        }
        if research_attempted and coverage_strategy not in {"merge", "legacy_only"}:
            _add_issue(issues, "overview_not_merged", "P0", "Non-causal analytical plan did not merge legacy evidence.", coverage)
        legacy_core = int(coverage.get("legacy_core_count") or 0)
        retained = int(coverage.get("retained_legacy_core_count") or 0)
        if legacy_core and retained < legacy_core and coverage_strategy != "legacy_only" and family in {"overview", "risk", "composite", "comparison"}:
            _add_issue(issues, "legacy_core_not_retained", "P0", "Legacy core requirements were dropped.", coverage)

    if family == "overview":
        if validated_numeric <= 0:
            _add_issue(issues, "numeric_missing", "P0", "Overview has no validated numeric evidence.", {"returned": returned_numeric, "validated": validated_numeric})
        if returned_numeric > 0 and validated_numeric == 0 and any(str(term).lower() in answer.lower() for term in _as_list(case.get("forbidden_answer_terms"))):
            _add_issue(issues, "claims_no_metrics_despite_returned_numeric", "P0", "Answer says metrics are unavailable despite returned numeric data.")
        missing_dims = sorted(set(str(item) for item in _as_list(case.get("required_dimensions"))) - _dimension_ids(trace))
        if missing_dims:
            _add_issue(issues, "overview_dimension_missing", "P1", "Overview minimum dimensions are not visible in trace.", missing_dims)

    if family == "causal":
        causal_terms = (("事实", "facts", "verified"), ("推断", "inference"), ("假设", "hypothesis"), ("边界", "boundary", "limit"))
        if not all(_contains_group(answer, list(group)) for group in causal_terms):
            _add_issue(issues, "causal_structure_missing", "P0", "Causal answer does not separate facts, inference, hypotheses, and boundary.")
        scope_violations = _as_list(contract.get("scope_overclaim_violations")) or _as_list(trace.get("scope_overclaim_violations"))
        if scope_violations:
            _add_issue(issues, "scope_overclaim_violation", "P0", "Scope overclaim violations are present.", scope_violations[:5])

    if family == "investment_boundary" and _has_valuation_contradiction(answer):
        _add_issue(
            issues,
            "valuation_contradiction",
            "P0",
            "Answer includes valuation metrics while claiming valuation evidence is missing or insufficient.",
        )

    if family == "comparison":
        if TEMPLATE_REFUSAL_RE.search(answer):
            _add_issue(issues, "template_refusal", "P0", "Comparison query returned a template refusal.")
        if not _comparison_answered_with_boundary(answer):
            _add_issue(
                issues,
                "comparison_not_answered",
                "P0",
                "Comparison answer does not explicitly compare risk or provide a bounded comparison.",
            )

    hard = [item for item in issues if item["severity"] == "P0"]
    warnings = [item for item in issues if item["severity"] == "P1"]
    accepted_warning_codes = {str(item) for item in _as_list(case.get("accept_warning_codes"))}
    accepted_codes = ACCEPTED_WARNING_CODES | accepted_warning_codes
    accepted = not hard and warnings and all(item["code"] in accepted_codes for item in warnings)
    status = "pass" if not issues else ("accepted_warning" if accepted else "fail")
    return {
        "case_id": case.get("case_id"),
        "query": case.get("query"),
        "status": status,
        "issues": issues,
        "root_causes": sorted({item["root_cause"] for item in issues}),
        "answer_chars": len(answer),
        "numeric_returned_count": returned_numeric,
        "numeric_validated_count": validated_numeric,
        "contract_passed": bool(contract.get("passed", False)),
        "contract_status": contract_status,
        "final_contract_status": final_contract_status,
        "contract_decision": contract_decision,
        "final_route": final_route,
        "planner_source": _planner_source(trace),
        "plan_strategy": str(coverage.get("strategy") or ""),
        "trace_id": trace.get("trace_id", ""),
    }


def build_debug_bundle(case: Mapping[str, Any], trace: Mapping[str, Any], contract: Mapping[str, Any], judgment: Mapping[str, Any]) -> dict[str, Any]:
    output = _as_dict(trace.get("output"))
    bundle = {
        "case": dict(case),
        "trace_id": trace.get("trace_id", ""),
        "answer": _answer_text(trace),
        "contract": contract,
        "contract_status": trace.get("final_contract_status") or trace.get("contract_status") or "",
        "final_contract_status": trace.get("final_contract_status") or "",
        "contract_decision": _trace_contract_decision(trace),
        "final_route": _trace_final_route(trace),
        "planner_source": _planner_source(trace),
        "relevance": _as_dict(trace.get("answer_relevance_decision")) or _as_dict(trace.get("relevance_decision")),
        "evidence_health": trace.get("evidence_health") or output.get("evidence_health") or "",
        "research_plan": {
            "raw": _as_dict(trace.get("research_plan_raw")),
            "validated": _as_dict(trace.get("research_plan_validated")),
            "used": _as_dict(trace.get("research_plan_used")),
            "validation": _as_dict(trace.get("research_plan_validation")),
        },
        "plan_coverage_decision": _as_dict(trace.get("plan_coverage_decision")),
        "requirement_merge_summary": _as_dict(trace.get("requirement_merge_summary")),
        "evidence_plan_used": _as_dict(trace.get("evidence_plan_used")),
        "evidence_validation_records": _validation_records(trace),
        "evidence_collection_results": _as_list(trace.get("evidence_collection_results")),
        "requirement_status_map": _as_dict(trace.get("requirement_status_map")),
        "dimension_status_by_id": _as_dict(trace.get("dimension_status_by_id")),
        "unsupported_claims": _as_list(trace.get("unsupported_claims")),
        "tool_errors": _tool_errors(trace),
        "scope_overclaim_check": _as_dict(trace.get("scope_overclaim_check")) or _as_dict(contract.get("scope_overclaim_check")),
        "judgment": dict(judgment),
    }
    return bundle


def summarize(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"pass": 0, "accepted_warning": 0, "fail": 0}
    for item in judgments:
        counts[str(item.get("status") or "fail")] = counts.get(str(item.get("status") or "fail"), 0) + 1
    p0 = [issue for item in judgments for issue in _as_list(item.get("issues")) if _as_dict(issue).get("severity") == "P0"]
    unexplained_p1 = [issue for item in judgments for issue in _as_list(item.get("issues")) if _as_dict(issue).get("severity") == "P1" and item.get("status") == "fail"]
    return {
        "case_count": len(judgments),
        "pass_count": counts["pass"],
        "accepted_warning_count": counts["accepted_warning"],
        "fail_count": counts["fail"],
        "p0_count": len(p0),
        "unexplained_p1_count": len(unexplained_p1),
        "root_causes": sorted({cause for item in judgments for cause in _as_list(item.get("root_causes"))}),
        "pass": counts["fail"] == 0 and len(p0) == 0 and len(unexplained_p1) == 0,
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    summary = _as_dict(report.get("summary"))
    lines = [
        "# FilingLens Regression 12",
        "",
        f"- cases: {summary.get('case_count', 0)}",
        f"- pass: {summary.get('pass_count', 0)}",
        f"- accepted warning: {summary.get('accepted_warning_count', 0)}",
        f"- fail: {summary.get('fail_count', 0)}",
        f"- P0: {summary.get('p0_count', 0)}",
        f"- pass gate: {summary.get('pass', False)}",
        "",
        "| Case | Status | Trace | Route | Contract | Final Contract | Decision | Planner | Plan | Numeric | Root Cause |",
        "|---|---|---|---|---|---|---|---|---|---:|---|",
    ]
    for item in _as_list(report.get("results")):
        if not isinstance(item, Mapping):
            continue
        lines.append(
            "| {case} | {status} | `{trace}` | {route} | {contract} | {final_contract} | {decision} | {planner} | {plan} | {validated}/{returned} | {causes} |".format(
                case=item.get("case_id"),
                status=item.get("status"),
                trace=item.get("trace_id", ""),
                route=item.get("final_route", ""),
                contract=item.get("contract_status", ""),
                final_contract=item.get("final_contract_status", ""),
                decision=item.get("contract_decision", ""),
                planner=item.get("planner_source", ""),
                plan=item.get("plan_strategy", ""),
                validated=item.get("numeric_validated_count", 0),
                returned=item.get("numeric_returned_count", 0),
                causes=", ".join(_as_list(item.get("root_causes"))),
            )
        )
    failures = [item for item in _as_list(report.get("results")) if _as_dict(item).get("status") == "fail"]
    if failures:
        lines.extend(["", "## Failures", ""])
        for item in failures:
            lines.extend([f"### {item.get('case_id')}", "", f"- query: {item.get('query')}", f"- trace_id: `{item.get('trace_id', '')}`"])
            for issue in _as_list(item.get("issues")):
                if isinstance(issue, Mapping):
                    lines.append(f"- [{issue.get('severity')}] {issue.get('code')} ({issue.get('root_cause')}): {issue.get('message')}")
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_regression(args: argparse.Namespace) -> dict[str, Any]:
    cases = _load_cases(args.benchmark, limit=args.limit)
    out_dir: Path = args.out_dir
    bundle_dir = out_dir / "bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    started = time.time()
    for index, case in enumerate(cases, start=1):
        trace_id = validate_trace_id(str(args.trace_prefix or f"finreg-{uuid.uuid4()}") + f"-{case['case_id'].lower()}")
        query = str(case["query"])
        t0 = time.time()
        try:
            if args.api_base:
                trace = _run_api(query, trace_id, args.api_base)
            else:
                trace = _run_direct(query, trace_id)
            stored = read_trace_payload(str(trace.get("trace_id") or trace_id))
            if stored:
                trace = {**trace, **stored}
            contract = check_answer_evidence_contract(trace)
            judgment = judge_case(case, trace, contract)
        except Exception as exc:
            trace = {"trace_id": trace_id, "error": str(exc), "query": query}
            contract = {"passed": False, "violations": [{"type": "runner_exception", "message": str(exc)}], "metrics": {}}
            judgment = {
                "case_id": case.get("case_id"),
                "query": query,
                "status": "fail",
                "issues": [{"code": "runner_exception", "severity": "P0", "root_cause": "runner", "message": str(exc)}],
                "root_causes": ["runner"],
                "trace_id": trace_id,
                "numeric_returned_count": 0,
                "numeric_validated_count": 0,
                "contract_passed": False,
                "contract_status": "exception",
            }
        judgment["elapsed_s"] = round(time.time() - t0, 2)
        bundle = build_debug_bundle(case, trace, contract, judgment)
        bundle_path = bundle_dir / f"{case['case_id']}_{trace_id}.json"
        bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        judgment["debug_bundle_path"] = str(bundle_path)
        results.append(judgment)
        print(
            f"[{index:02d}/{len(cases):02d}] {case['case_id']} {judgment['status']} "
            f"trace={judgment.get('trace_id', '')} numeric={judgment.get('numeric_validated_count', 0)}/{judgment.get('numeric_returned_count', 0)} "
            f"causes={','.join(judgment.get('root_causes', [])) or '-'}"
        )
    report = {
        "benchmark_path": str(args.benchmark),
        "started_at_epoch": started,
        "elapsed_s": round(time.time() - started, 2),
        "trace_dir": str(settings.traces_dir),
        "summary": summarize(results),
        "results": results,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(report, out_dir / "report.md")
    failures = [item for item in results if item.get("status") == "fail"]
    (out_dir / "failures.jsonl").write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in failures),
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the 12-query financial analyst regression.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-base", default="", help="Optional running API base URL, e.g. http://127.0.0.1:8080")
    parser.add_argument("--trace-prefix", default="", help="Stable trace id prefix for reproducible reruns")
    args = parser.parse_args()
    report = run_regression(args)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"].get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
