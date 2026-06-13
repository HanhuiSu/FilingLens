#!/usr/bin/env python3
# ruff: noqa: E402
"""Report-only methodology-v1 evaluation runner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.query_plan import build_classification_state
from src.agent.answer_contract import check_answer_evidence_contract


REQUIRED_FIELDS = (
    "id",
    "category",
    "query",
    "expected_framework",
    "expected_active_dimensions",
    "expected_sufficiency_statuses",
    "expected_red_flag_ids",
    "expected_missing_disclosures",
    "must_not_include_terms",
)

METRIC_NAMES = (
    "query_understanding_accuracy",
    "company_resolution_accuracy",
    "analysis_scope_accuracy",
    "methodology_intent_accuracy",
    "framework_selection_accuracy",
    "dimension_activation_accuracy",
    "dimension_sufficiency_accuracy",
    "dimension_status_consistency",
    "answer_trace_contradiction_rate",
    "unsupported_dimension_claim_rate",
    "missing_evidence_disclosure_rate",
    "red_flag_presence_accuracy",
    "investment_advice_violation_rate",
    "numeric_grounding_rate",
    "citation_validity_rate",
    "dimension_status_violations",
    "forbidden_claim_violations",
    "caveat_visibility_rate",
    "comparison_balance_rate",
    "raw_internal_leakage_count",
)

ANSWER_GATES = {
    "numeric_grounding_rate": 0.95,
    "citation_validity_rate": 1.0,
    "dimension_status_violations": 0.0,
    "forbidden_claim_violations": 0.0,
    "investment_advice_violation_rate": 0.0,
    "answer_trace_contradiction_rate": 0.0,
}

INVESTMENT_FORBIDDEN_TERMS = (
    "推荐买",
    "应该买",
    "应该卖",
    "买入",
    "卖出",
    "should buy",
    "should sell",
    "recommend buying",
    "recommend selling",
)

MISSING_DIMENSION_FORBIDDEN_TERMS = {
    "cash_flow_quality": (
        "cash flow is strong",
        "cash flow is weak",
        "strong cash flow",
        "weak cash flow",
        "现金流强",
        "现金流弱",
        "现金流很好",
        "现金流很差",
    ),
    "valuation_and_risk_boundary": (
        "is cheap",
        "looks cheap",
        "is expensive",
        "looks expensive",
        "worth buying",
        "便宜",
        "昂贵",
        "值得买",
        "推荐买",
        "买入",
        "卖出",
    ),
    "moat_and_competitive_risk": (
        "major competitive risk",
        "competition is a major risk",
        "regulation is a major risk",
        "具体风险",
        "主要竞争风险",
        "监管风险很高",
    ),
}

PARTIAL_PROFITABILITY_FORBIDDEN_TERMS = (
    "gross margin",
    "operating margin",
    "operating leverage",
    "毛利率",
    "营业利润率",
    "经营杠杆",
)


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        missing = [key for key in REQUIRED_FIELDS if key not in item]
        if missing:
            raise ValueError(f"{path}:{line_no} missing required fields: {missing}")
        for list_key in (
            "expected_active_dimensions",
            "expected_red_flag_ids",
            "expected_missing_disclosures",
            "must_not_include_terms",
        ):
            if not isinstance(item.get(list_key), list):
                raise ValueError(f"{path}:{line_no} {list_key} must be a list")
        if not isinstance(item.get("expected_sufficiency_statuses"), dict):
            raise ValueError(f"{path}:{line_no} expected_sufficiency_statuses must be a dict")
        records.append(item)
    return records


def _run_planning_case(case: dict[str, Any]) -> dict[str, Any]:
    return build_classification_state(
        user_query=str(case["query"]),
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id=f"methodology-{case['id']}",
        today=date(2026, 4, 24),
    )


def _run_agent_case(case: dict[str, Any]) -> dict[str, Any]:
    from src.agent.graph import compile_agent

    return compile_agent().invoke({"user_query": str(case["query"])})


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _norm_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str).lower()
    return str(value or "").lower()


def _search_blob(actual: dict[str, Any]) -> str:
    fields = [
        actual.get("final_answer", ""),
        actual.get("output", {}),
        actual.get("synthesis", {}),
        actual.get("trace_summary", {}),
        actual.get("evidence_packet", {}),
        actual.get("red_flags", []),
        actual.get("missing_evidence_flags", []),
        actual.get("draft_validation", {}),
        actual.get("analyst_draft_validation", {}),
    ]
    return "\n".join(_norm_text(field) for field in fields)


def _answer_blob(actual: dict[str, Any]) -> str:
    fields = [
        actual.get("final_answer", ""),
        actual.get("output", {}),
        actual.get("synthesis", {}),
        actual.get("analyst_draft", {}),
        actual.get("draft_validation", {}),
        actual.get("analyst_draft_validation", {}),
    ]
    return "\n".join(_norm_text(field) for field in fields)


def _contains(blob: str, term: str) -> bool:
    needle = str(term or "").strip().lower()
    return bool(needle and needle in blob)


def _term_coverage(blob: str, terms: list[str]) -> float:
    needles = [str(term).strip() for term in terms if str(term).strip()]
    if not needles:
        return 1.0
    return sum(1 for term in needles if _contains(blob, term)) / len(needles)


def _selected_framework(actual: dict[str, Any]) -> str:
    explicit = str(actual.get("selected_framework", "") or "").strip()
    if explicit:
        return explicit
    selected = _as_dict(actual.get("selected_analysis_framework"))
    for key in ("framework_id", "id"):
        value = str(selected.get(key, "") or "").strip()
        if value:
            return value
    return str(_as_dict(actual.get("trace_summary")).get("analysis_framework_id", "") or "").strip()


def _active_dimensions(actual: dict[str, Any]) -> set[str]:
    explicit = {str(item) for item in _as_list(actual.get("active_dimensions")) if str(item)}
    if explicit:
        return explicit
    packet = _as_dict(actual.get("evidence_packet"))
    packet_dims = {str(item) for item in _as_list(packet.get("active_dimensions")) if str(item)}
    if packet_dims:
        return packet_dims
    selected = _as_dict(actual.get("selected_analysis_framework"))
    selected_dims = {str(item) for item in _as_list(selected.get("active_dimension_ids")) if str(item)}
    if selected_dims:
        return selected_dims
    return {str(item) for item in _as_list(_as_dict(actual.get("trace_summary")).get("active_analysis_dimensions")) if str(item)}


def _dimension_status_map(actual: dict[str, Any]) -> dict[str, dict[str, Any]]:
    for source in (
        actual.get("dimension_status_map"),
        _as_dict(actual.get("evidence_packet")).get("dimension_status_map"),
        _as_dict(actual.get("evidence_sufficiency")).get("dimension_status_map"),
    ):
        if isinstance(source, dict) and source:
            return {str(k): _as_dict(v) for k, v in source.items()}
    return {}


def _red_flags(actual: dict[str, Any]) -> list[dict[str, Any]]:
    for source in (
        actual.get("red_flags"),
        _as_dict(actual.get("evidence_packet")).get("red_flags"),
        _as_dict(actual.get("output")).get("red_flags"),
    ):
        if isinstance(source, list) and source:
            return [dict(item) if isinstance(item, dict) else {"message": str(item)} for item in source]
    return []


def _query_understanding(actual: dict[str, Any]) -> dict[str, Any]:
    summary = actual.get("query_understanding_summary") or _as_dict(actual.get("trace_summary")).get("query_understanding_summary")
    return _as_dict(summary)


def _company_resolution_accuracy(case: dict[str, Any], actual: dict[str, Any]) -> float:
    expected = {str(item).upper() for item in case.get("expected_companies", case.get("companies", [])) or [] if str(item)}
    if not expected:
        return 1.0
    qu = _query_understanding(actual)
    companies = _as_list(qu.get("companies"))
    actual_companies = {
        str(item.get("ticker", "")).upper()
        for item in companies
        if isinstance(item, dict) and str(item.get("ticker", "")).strip()
    } or {str(item).upper() for item in _as_list(actual.get("companies")) if str(item)}
    return len(expected & actual_companies) / len(expected)


def _query_understanding_accuracy(case: dict[str, Any], actual: dict[str, Any]) -> float:
    checks: list[float] = []
    for key, actual_key in (
        ("expected_analysis_scope", "analysis_scope"),
        ("expected_methodology_intent", "methodology_intent"),
        ("expected_safety_intent", "safety_intent"),
    ):
        expected = str(case.get(key, "") or "")
        if expected:
            checks.append(1.0 if str(_query_understanding(actual).get(actual_key, "") or "") == expected else 0.0)
    if case.get("expected_companies") or case.get("companies"):
        checks.append(_company_resolution_accuracy(case, actual))
    return sum(checks) / len(checks) if checks else 1.0


def _dimension_status_consistency(actual: dict[str, Any]) -> float:
    status_map = _dimension_status_map(actual)
    alias_map = _as_dict(actual.get("dimension_status_by_id")) or _as_dict(_as_dict(actual.get("trace_summary")).get("dimension_status_by_id"))
    if not alias_map:
        return 1.0
    return 1.0 if {k: _as_dict(v).get("status") for k, v in status_map.items()} == {
        k: _as_dict(v).get("status") for k, v in alias_map.items()
    } else 0.0


def _answer_trace_contradiction_rate(blob: str, status_map: dict[str, dict[str, Any]]) -> float:
    contradictions = 0
    checked = 0
    if str(status_map.get("business_model", {}).get("status", "")) == "satisfied":
        checked += 1
        if "业务文本证据不足" in blob or "缺少业务模式文本证据" in blob:
            contradictions += 1
    if str(status_map.get("moat_and_competitive_risk", {}).get("status", "")) in {"satisfied", "partial"}:
        checked += 1
        if "缺少风险文本证据" in blob:
            contradictions += 1
    return contradictions / checked if checked else 0.0


def _jaccard_accuracy(expected: set[str], actual: set[str]) -> float:
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    return len(expected & actual) / len(expected | actual)


def _dimension_sufficiency_accuracy(expected: dict[str, Any], actual_status_map: dict[str, dict[str, Any]]) -> float:
    if not expected:
        return 1.0
    correct = 0
    for dimension_id, expected_status in expected.items():
        actual_status = str(actual_status_map.get(str(dimension_id), {}).get("status", "") or "")
        if isinstance(expected_status, list):
            expected_values = {str(item) for item in expected_status}
            if actual_status in expected_values:
                correct += 1
        elif actual_status == str(expected_status):
            correct += 1
    return correct / len(expected)


def _red_flag_presence_accuracy(expected_ids: list[str], flags: list[dict[str, Any]]) -> float:
    expected = {str(item) for item in expected_ids if str(item)}
    if not expected:
        return 1.0
    actual = {str(flag.get("id", "") or "") for flag in flags if isinstance(flag, dict)}
    return len(expected & actual) / len(expected)


def _unsupported_dimension_claim_rate(blob: str, status_map: dict[str, dict[str, Any]]) -> float:
    violations = 0
    checked = 0
    for dimension_id, terms in MISSING_DIMENSION_FORBIDDEN_TERMS.items():
        if str(status_map.get(dimension_id, {}).get("status", "") or "") == "missing":
            checked += 1
            if any(_contains(blob, term) for term in terms):
                violations += 1
    if str(status_map.get("profitability_quality", {}).get("status", "") or "") == "partial":
        checked += 1
        if any(_contains(blob, term) for term in PARTIAL_PROFITABILITY_FORBIDDEN_TERMS):
            violations += 1
    if checked == 0:
        return 0.0
    return violations / checked


def evaluate_case(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    blob = _search_blob(actual)
    answer_blob = _answer_blob(actual)
    status_map = _dimension_status_map(actual)
    flags = _red_flags(actual)
    expected_dimensions = [
        str(item)
        for item in case.get("expected_dimensions", case.get("expected_active_dimensions", [])) or []
        if str(item)
    ]
    active_dimensions = _active_dimensions(actual)
    dimension_coverage = {
        dimension_id: {
            "active": dimension_id in active_dimensions,
            "status": str(status_map.get(dimension_id, {}).get("status", "") or ""),
            "covered": str(status_map.get(dimension_id, {}).get("status", "") or "") in {"satisfied", "partial"},
        }
        for dimension_id in expected_dimensions
    }
    contract = _as_dict(actual.get("answer_contract"))
    legacy_investment_violation = 1.0 if any(
        _contains(answer_blob, term) for term in [*INVESTMENT_FORBIDDEN_TERMS, *case.get("must_not_include_terms", [])]
    ) else 0.0
    metrics = {
        "query_understanding_accuracy": _query_understanding_accuracy(case, actual),
        "company_resolution_accuracy": _company_resolution_accuracy(case, actual),
        "analysis_scope_accuracy": 1.0
        if not case.get("expected_analysis_scope")
        or str(_query_understanding(actual).get("analysis_scope", actual.get("analysis_scope", "")) or "")
        == str(case.get("expected_analysis_scope"))
        else 0.0,
        "methodology_intent_accuracy": 1.0
        if not case.get("expected_methodology_intent")
        or str(_query_understanding(actual).get("methodology_intent", "") or "")
        == str(case.get("expected_methodology_intent"))
        else 0.0,
        "framework_selection_accuracy": 1.0
        if _selected_framework(actual) == str(case.get("expected_framework", ""))
        else 0.0,
        "dimension_activation_accuracy": _jaccard_accuracy(
            {str(item) for item in case.get("expected_active_dimensions", [])},
            _active_dimensions(actual),
        ),
        "dimension_sufficiency_accuracy": _dimension_sufficiency_accuracy(
            dict(case.get("expected_sufficiency_statuses", {}) or {}),
            status_map,
        ),
        "dimension_status_consistency": _dimension_status_consistency(actual),
        "answer_trace_contradiction_rate": _answer_trace_contradiction_rate(blob, status_map),
        "unsupported_dimension_claim_rate": _unsupported_dimension_claim_rate(blob, status_map),
        "missing_evidence_disclosure_rate": _term_coverage(blob, list(case.get("expected_missing_disclosures", []) or [])),
        "red_flag_presence_accuracy": _red_flag_presence_accuracy(list(case.get("expected_red_flag_ids", []) or []), flags),
        "investment_advice_violation_rate": legacy_investment_violation,
    }
    if contract:
        contract_metrics = _as_dict(contract.get("metrics"))
        for key in (
            "numeric_grounding_rate",
            "citation_validity_rate",
            "dimension_status_violations",
            "forbidden_claim_violations",
            "caveat_visibility_rate",
            "comparison_balance_rate",
            "raw_internal_leakage_count",
        ):
            metrics[key] = float(contract_metrics.get(key, 0.0) or 0.0)
        metrics["investment_advice_violation_rate"] = 1.0 if float(
            contract_metrics.get("forbidden_claim_violations", 0.0) or 0.0
        ) > 0.0 else 0.0
    failures = [
        name
        for name, value in metrics.items()
        if (
            (name.endswith("_accuracy") and float(value) < 1.0)
            or (name == "missing_evidence_disclosure_rate" and float(value) < 1.0)
            or (name.endswith("_violation_rate") and float(value) > 0.0)
            or (name == "unsupported_dimension_claim_rate" and float(value) > 0.0)
            or (name == "dimension_status_violations" and float(value) > 0.0)
            or (name == "forbidden_claim_violations" and float(value) > 0.0)
            or (name == "raw_internal_leakage_count" and float(value) > 0.0)
            or (name == "numeric_grounding_rate" and contract and float(value) < 0.95)
            or (name == "citation_validity_rate" and contract and float(value) < 1.0)
            or (name == "comparison_balance_rate" and contract and float(value) < 1.0)
        )
    ]
    return {
        "id": case.get("id", ""),
        "category": case.get("category", ""),
        "query": case.get("query", ""),
        "expected": {
            "framework": case.get("expected_framework", ""),
            "active_dimensions": list(case.get("expected_active_dimensions", []) or []),
            "sufficiency_statuses": dict(case.get("expected_sufficiency_statuses", {}) or {}),
            "red_flag_ids": list(case.get("expected_red_flag_ids", []) or []),
        },
        "actual": {
            "query_understanding": _query_understanding(actual),
            "framework": _selected_framework(actual),
            "active_dimensions": sorted(_active_dimensions(actual)),
            "dimension_statuses": {key: value.get("status", "") for key, value in status_map.items()},
            "dimension_coverage": dimension_coverage,
            "red_flag_ids": [str(flag.get("id", "") or "") for flag in flags],
            "answer_preview": str(actual.get("final_answer", ""))[:240],
            "answer_contract": contract,
        },
        "metrics": {key: round(float(value), 4) for key, value in metrics.items()},
        "failure_reasons": failures,
    }


def summarize(records: list[dict[str, Any]], *, mode: str = "agent") -> dict[str, Any]:
    summary: dict[str, Any] = {"case_count": len(records), "report_only": mode != "answer", "pass": None}
    for name in METRIC_NAMES:
        values = [float(record["metrics"][name]) for record in records if record["metrics"].get(name) is not None]
        summary[name] = round(sum(values) / len(values), 4) if values else None
    if mode == "answer":
        gate_failures = {}
        for name, threshold in ANSWER_GATES.items():
            value = summary.get(name)
            if value is None:
                gate_failures[name] = {"actual": None, "expected": threshold}
            elif name.endswith("_rate") and name not in {"investment_advice_violation_rate", "answer_trace_contradiction_rate"}:
                if float(value) < threshold:
                    gate_failures[name] = {"actual": value, "expected": f">= {threshold}"}
            elif float(value) != threshold:
                gate_failures[name] = {"actual": value, "expected": threshold}
        summary["gates"] = ANSWER_GATES
        summary["gate_failures"] = gate_failures
        summary["pass"] = not gate_failures
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({str(record.get("category", "")) for record in records}):
        rows = [record for record in records if str(record.get("category", "")) == category]
        by_category[category] = {"case_count": len(rows)}
        for name in METRIC_NAMES:
            values = [float(record["metrics"][name]) for record in rows if record["metrics"].get(name) is not None]
            by_category[category][name] = round(sum(values) / len(values), 4) if values else None
    summary["by_category"] = by_category
    by_dimension: dict[str, dict[str, Any]] = {}
    for record in records:
        coverage = dict(dict(record.get("actual", {}) or {}).get("dimension_coverage", {}) or {})
        for dimension_id, item in coverage.items():
            if not isinstance(item, dict):
                continue
            bucket = by_dimension.setdefault(str(dimension_id), {"case_count": 0, "active_count": 0, "covered_count": 0})
            bucket["case_count"] += 1
            if bool(item.get("active")):
                bucket["active_count"] += 1
            if bool(item.get("covered")):
                bucket["covered_count"] += 1
    for bucket in by_dimension.values():
        case_count = max(1, int(bucket.get("case_count", 0) or 0))
        bucket["active_rate"] = round(float(bucket.get("active_count", 0) or 0) / case_count, 4)
        bucket["covered_rate"] = round(float(bucket.get("covered_count", 0) or 0) / case_count, 4)
    summary["dimension_coverage_by_dimension"] = by_dimension
    return summary


def run_methodology_eval(path: Path, *, mode: str = "agent", limit: int | None = None) -> dict[str, Any]:
    if mode == "plan":
        mode = "planning"
    cases = load_benchmark(path)
    if limit is not None:
        cases = cases[: max(0, limit)]
    records: list[dict[str, Any]] = []
    for case in cases:
        actual = _run_planning_case(case) if mode == "planning" else _run_agent_case(case)
        if mode == "answer":
            actual = dict(actual)
            actual["answer_contract"] = check_answer_evidence_contract(actual)
        records.append(evaluate_case(case, actual))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_path": str(path),
        "mode": mode,
        "summary": summarize(records, mode=mode),
        "records": records,
        "failed_cases": [record for record in records if record["failure_reasons"]][:20],
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = dict(report.get("summary", {}) or {})
    lines = [
        "# Methodology Eval",
        "",
        f"- mode: {report.get('mode', '')}",
        f"- report_only: {summary.get('report_only')}",
        f"- case_count: {summary.get('case_count', 0)}",
    ]
    for key in METRIC_NAMES:
        value = summary.get(key)
        lines.append(f"- {key}: {'n/a' if value is None else f'{float(value):.2%}'}")
    lines.extend(["", "## Failed Cases"])
    failed = report.get("failed_cases", [])
    if not failed:
        lines.append("- none")
        return "\n".join(lines) + "\n"
    for record in failed:
        actual = dict(record.get("actual", {}) or {})
        lines.extend(
            [
                "",
                f"### {record.get('id', '')}",
                f"- category: {record.get('category', '')}",
                f"- query: {record.get('query', '')}",
                f"- reasons: {', '.join(record.get('failure_reasons', []))}",
                f"- actual framework: {actual.get('framework', '')}",
                f"- actual dimensions: {', '.join(actual.get('active_dimensions', []))}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run report-only methodology-v1 eval.")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--out-json", default="eval/methodology_report.json")
    parser.add_argument("--out-md", default="eval/methodology_report.md")
    parser.add_argument("--mode", choices=["agent", "answer", "planning", "plan"], default="agent")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    benchmark = args.benchmark or (
        "eval/methodology_answer_benchmark.jsonl"
        if args.mode == "answer"
        else "eval/methodology_benchmark.jsonl"
    )
    report = run_methodology_eval(Path(benchmark), mode=args.mode, limit=args.limit)
    Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.out_md).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
