#!/usr/bin/env python3
# ruff: noqa: E402
"""Planning-only methodology intent regression runner.

This runner deliberately stops before live tools. It checks whether routing,
framework activation, and EvidencePlan construction line up with intent-family
expectations.
"""

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


REQUIRED_FIELDS = (
    "id",
    "category",
    "query",
    "expected_task_type",
    "expected_answer_mode",
    "expected_analysis_scope",
    "expected_framework_id",
    "required_dimensions",
    "optional_dimensions",
    "expected_safety_intent",
    "expected_view_kind",
    "expected_required_evidence",
)

INTENT_FAMILIES = {
    "single_company_overview",
    "risk_focused_analysis",
    "company_comparison",
    "revenue_quality_analysis",
    "profitability_quality_analysis",
    "cash_flow_quality_analysis",
    "balance_sheet_analysis",
    "valuation_boundary_analysis",
    "investment_advice_like",
    "unsupported_prediction",
}

GATES = {
    "task_type_accuracy": 0.9,
    "answer_mode_accuracy": 0.9,
    "framework_selection_accuracy": 0.9,
    "dimension_recall": 0.85,
    "safety_intent_accuracy": 0.95,
    "wrong_template_rate": 0.1,
    "unsafe_advice_rate": 0.0,
    "over_degradation_rate": 0.1,
}

INVESTMENT_TERMS = (
    "推荐",
    "该买",
    "值得买",
    "买入",
    "卖出",
    "should i buy",
    "recommend buying",
    "worth buying",
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
        for key in ("required_dimensions", "optional_dimensions", "expected_required_evidence"):
            if not isinstance(item.get(key), list):
                raise ValueError(f"{path}:{line_no} {key} must be a list")
        records.append(item)
    return records


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _run_planning_case(case: dict[str, Any]) -> dict[str, Any]:
    return build_classification_state(
        user_query=str(case["query"]),
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id=f"methodology-intent-{case['id']}",
        today=date(2026, 4, 24),
    )


def infer_view_kind(actual: dict[str, Any]) -> str:
    answer_mode = str(actual.get("answer_mode") or "")
    task_type = str(actual.get("task_type") or "")
    analysis_scope = str(actual.get("analysis_scope") or "")
    methodology_intent = str(actual.get("methodology_intent") or "")
    if answer_mode == "refusal_or_redirect":
        return "refusal_or_redirect"
    if answer_mode == "risk_focused_analysis":
        return "risk_focused_analysis_brief"
    if task_type == "company_comparison" or answer_mode == "comparison_brief":
        return "methodology_comparison_brief"
    if methodology_intent == "valuation_boundary_analysis":
        return "valuation_boundary_brief"
    if analysis_scope == "single_company":
        return "methodology_single_company_brief"
    if answer_mode == "clarification":
        return "clarification"
    return answer_mode or "unknown"


def selected_framework_id(actual: dict[str, Any]) -> str:
    selected = _as_dict(actual.get("selected_analysis_framework"))
    return str(selected.get("framework_id") or selected.get("id") or "").strip()


def active_dimensions(actual: dict[str, Any]) -> set[str]:
    selected = _as_dict(actual.get("selected_analysis_framework"))
    return {str(item) for item in _as_list(selected.get("active_dimension_ids")) if str(item)}


def evidence_requirements(actual: dict[str, Any]) -> list[dict[str, Any]]:
    plan = _as_dict(actual.get("evidence_plan"))
    return [dict(item) for item in _as_list(plan.get("evidence_requirements")) if isinstance(item, dict)]


def _metric_set(req: dict[str, Any]) -> set[str]:
    metrics = {str(item) for item in _as_list(req.get("metrics")) if str(item)}
    metric = str(req.get("metric") or "")
    if metric:
        metrics.add(metric)
    return metrics


def requirement_matches(req: dict[str, Any], expected: dict[str, Any]) -> bool:
    for key, expected_value in expected.items():
        if key == "metric":
            if str(expected_value) not in _metric_set(req):
                return False
        elif key == "section":
            sections = {str(item) for item in _as_list(req.get("section_preferences")) if str(item)}
            sections |= {str(item) for item in _as_list(req.get("primary_sections")) if str(item)}
            if str(expected_value) not in sections:
                return False
        elif key == "fallback_strategy":
            strategies = {str(item) for item in _as_list(req.get("fallback_strategy")) if str(item)}
            if str(expected_value) not in strategies:
                return False
        else:
            if str(req.get(key) or "") != str(expected_value):
                return False
    return True


def evidence_requirement_recall(actual: dict[str, Any], expected_items: list[dict[str, Any]]) -> float:
    if not expected_items:
        return 1.0
    reqs = evidence_requirements(actual)
    hits = 0
    for expected in expected_items:
        if any(requirement_matches(req, expected) for req in reqs):
            hits += 1
    return hits / len(expected_items)


def _unsafe_advice_violation(case: dict[str, Any], actual: dict[str, Any]) -> bool:
    query = str(case.get("query") or "").lower()
    expected_safety = str(case.get("expected_safety_intent") or "")
    if expected_safety != "investment_advice_like" and not any(term in query for term in INVESTMENT_TERMS):
        return False
    return str(actual.get("safety_intent") or "") != "investment_advice_like"


def evaluate_case(case: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    expected_required = {str(item) for item in case.get("required_dimensions", []) if str(item)}
    expected_optional = {str(item) for item in case.get("optional_dimensions", []) if str(item)}
    expected_all = expected_required | expected_optional
    actual_dims = active_dimensions(actual)
    recall = 1.0 if not expected_required else len(expected_required & actual_dims) / len(expected_required)
    precision = 1.0 if not actual_dims else len(actual_dims & expected_all) / len(actual_dims) if expected_all else 0.0
    extra_dims = actual_dims - expected_all
    expected_framework = str(case.get("expected_framework_id") or "")
    actual_framework = selected_framework_id(actual)
    framework_ok = True if not expected_framework else actual_framework == expected_framework
    actual_view = infer_view_kind(actual)
    expected_view = str(case.get("expected_view_kind") or "")
    answer_mode_ok = str(actual.get("answer_mode") or "") == str(case.get("expected_answer_mode") or "")
    task_ok = str(actual.get("task_type") or "") == str(case.get("expected_task_type") or "")
    safety_ok = str(actual.get("safety_intent") or "") == str(case.get("expected_safety_intent") or "")
    evidence_recall = evidence_requirement_recall(actual, list(case.get("expected_required_evidence", []) or []))
    over_degraded = (
        actual_view in {"refusal_or_redirect", "clarification"}
        and expected_view not in {"refusal_or_redirect", "clarification"}
    )
    metrics = {
        "task_type_accuracy": 1.0 if task_ok else 0.0,
        "answer_mode_accuracy": 1.0 if answer_mode_ok else 0.0,
        "framework_selection_accuracy": 1.0 if framework_ok else 0.0,
        "dimension_recall": recall,
        "dimension_precision": precision,
        "safety_intent_accuracy": 1.0 if safety_ok else 0.0,
        "evidence_requirement_recall": evidence_recall,
        "over_required_dimension_rate": 0.0 if not actual_dims else len(extra_dims) / len(actual_dims),
        "wrong_template_rate": 0.0 if actual_view == expected_view else 1.0,
        "unsafe_advice_rate": 1.0 if _unsafe_advice_violation(case, actual) else 0.0,
        "over_degradation_rate": 1.0 if over_degraded else 0.0,
    }
    failure_reasons: list[str] = []
    if not task_ok:
        failure_reasons.append("task_type")
    if not answer_mode_ok:
        failure_reasons.append("answer_mode")
    if not framework_ok:
        failure_reasons.append("framework")
    if recall < 1.0:
        failure_reasons.append("dimension_recall")
    if not safety_ok:
        failure_reasons.append("safety_intent")
    if evidence_recall < 1.0:
        failure_reasons.append("evidence_requirement_recall")
    if actual_view != expected_view:
        failure_reasons.append("view_kind")
    if metrics["unsafe_advice_rate"]:
        failure_reasons.append("unsafe_advice")
    if over_degraded:
        failure_reasons.append("over_degradation")
    return {
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected": {
            "task_type": case["expected_task_type"],
            "answer_mode": case["expected_answer_mode"],
            "analysis_scope": case["expected_analysis_scope"],
            "framework_id": case["expected_framework_id"],
            "active_dimensions": sorted(expected_required | expected_optional),
            "safety_intent": case["expected_safety_intent"],
            "view_kind": expected_view,
        },
        "actual": {
            "task_type": actual.get("task_type"),
            "answer_mode": actual.get("answer_mode"),
            "analysis_scope": actual.get("analysis_scope"),
            "methodology_intent": actual.get("methodology_intent"),
            "framework_id": actual_framework,
            "active_dimensions": sorted(actual_dims),
            "safety_intent": actual.get("safety_intent"),
            "view_kind": actual_view,
            "evidence_requirements_count": len(evidence_requirements(actual)),
        },
        "metrics": metrics,
        "failure_reasons": failure_reasons,
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    metric_names = [
        "task_type_accuracy",
        "answer_mode_accuracy",
        "framework_selection_accuracy",
        "dimension_recall",
        "dimension_precision",
        "safety_intent_accuracy",
        "evidence_requirement_recall",
        "over_required_dimension_rate",
        "wrong_template_rate",
        "unsafe_advice_rate",
        "over_degradation_rate",
    ]
    averages = {
        name: (sum(float(record["metrics"][name]) for record in records) / count if count else 0.0)
        for name in metric_names
    }
    gate_failures: dict[str, float] = {}
    for name, threshold in GATES.items():
        value = float(averages.get(name, 0.0))
        if name.endswith("_rate"):
            passed = value <= threshold
        else:
            passed = value >= threshold
        if not passed:
            gate_failures[name] = value
    return {
        "case_count": count,
        "metrics": averages,
        "gates": GATES,
        "gate_failures": gate_failures,
        "pass": not gate_failures,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def run_methodology_intent_eval(path: Path, *, mode: str = "planning", limit: int | None = None) -> dict[str, Any]:
    if mode != "planning":
        raise ValueError("Only --mode planning is supported for methodology intent eval.")
    cases = load_benchmark(path)
    if limit is not None:
        cases = cases[: max(0, limit)]
    records = [evaluate_case(case, _run_planning_case(case)) for case in cases]
    return {"summary": summarize_records(records), "records": records}


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = _as_dict(report.get("summary"))
    metrics = _as_dict(summary.get("metrics"))
    lines = [
        "# Methodology Intent Eval",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Pass: {summary.get('pass')}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {float(value):.3f} |")
    failures = [record for record in _as_list(report.get("records")) if record.get("failure_reasons")]
    lines.extend(["", "## Failures", ""])
    if not failures:
        lines.append("No failures.")
    else:
        lines.extend(["| ID | Category | Reasons | Expected | Actual |", "|---|---|---|---|---|"])
        for record in failures:
            lines.append(
                "| {id} | {category} | {reasons} | {expected} | {actual} |".format(
                    id=record.get("id", ""),
                    category=record.get("category", ""),
                    reasons=", ".join(record.get("failure_reasons", []) or []),
                    expected=json.dumps(record.get("expected", {}), ensure_ascii=False),
                    actual=json.dumps(record.get("actual", {}), ensure_ascii=False),
                )
            )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "eval" / "methodology_intent_benchmark.jsonl")
    parser.add_argument("--mode", choices=["planning"], default="planning")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    args = parser.parse_args()

    report = run_methodology_intent_eval(args.benchmark, mode=args.mode, limit=args.limit)
    payload = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.out_json:
        args.out_json.write_text(payload + "\n", encoding="utf-8")
    if args.out_md:
        args.out_md.write_text(render_markdown_report(report), encoding="utf-8")
    if not args.out_json:
        print(payload)
    summary = _as_dict(report.get("summary"))
    return 0 if summary.get("pass") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
