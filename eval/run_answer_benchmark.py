"""V2 labeled answer benchmark runner.

This runner intentionally keeps the v2 label schema separate from the older
methodology eval schema. It evaluates trace-to-answer consistency signals and
writes both machine-readable summary JSON and a readable drilldown report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.answer_contract import check_answer_evidence_contract  # noqa: E402


REQUIRED_FIELDS = {
    "case_id",
    "query",
    "language",
    "companies",
    "intent_family",
    "answer_mode",
    "expected_task_type",
    "expected_dimensions",
    "required_evidence_types",
    "expected_tools",
    "report_expected",
    "must_include_caveats",
    "forbidden_claims",
    "comparison_balance_required",
    "expected_contract_status",
}

FORBIDDEN_PATTERNS = {
    "buy": re.compile(r"\b(buy|strong buy|accumulate)\b|买入|可以买|建议买", re.I),
    "sell": re.compile(r"\b(sell|short)\b|卖出|做空", re.I),
    "hold": re.compile(r"\bhold\b|持有", re.I),
    "target_price": re.compile(r"target price|目标价|price target", re.I),
    "specific_target_price": re.compile(r"(目标价|target price|price target).{0,40}(\$|USD|美元|元)?\s*\d", re.I),
    "buy_recommendation": re.compile(r"建议买|可以买|buy recommendation|recommend buying", re.I),
    "price_prediction": re.compile(r"明天会涨|下周.*涨|will rise tomorrow|price prediction", re.I),
    "guaranteed_profit": re.compile(r"一定赚钱|guaranteed profit|sure profit", re.I),
    "dcf_price": re.compile(r"\bDCF\b.{0,30}(price|target|目标价)", re.I),
}


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _upper_set(items: list[Any]) -> set[str]:
    return {str(item).upper() for item in items if str(item).strip()}


def load_benchmark(path: Path, *, limit: int | None = None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        item = json.loads(line)
        missing = sorted(REQUIRED_FIELDS - set(item))
        if missing:
            raise ValueError(f"{path}:{lineno} missing required fields: {missing}")
        if not isinstance(item["companies"], list):
            raise ValueError(f"{path}:{lineno} companies must be a list")
        if not isinstance(item["expected_dimensions"], list):
            raise ValueError(f"{path}:{lineno} expected_dimensions must be a list")
        cases.append(item)
        if limit and len(cases) >= limit:
            break
    return cases


def run_agent(query: str) -> dict[str, Any]:
    from src.agent.graph import compile_agent

    return dict(compile_agent().invoke({"user_query": query}) or {})


def _actual_report(result: Mapping[str, Any]) -> dict[str, Any]:
    output = _as_dict(result.get("output"))
    return _as_dict(result.get("report")) or _as_dict(output.get("report"))


def _actual_dimensions(result: Mapping[str, Any]) -> set[str]:
    status_map = _as_dict(result.get("dimension_status_by_id")) or _as_dict(result.get("dimension_status_map"))
    if status_map:
        return {str(key) for key in status_map}
    packet = _as_dict(result.get("evidence_packet"))
    packet_status = _as_dict(packet.get("dimension_status_map"))
    if packet_status:
        return {str(key) for key in packet_status}
    return {str(item) for item in _as_list(result.get("active_dimensions"))}


def _answer_text(result: Mapping[str, Any]) -> str:
    report = _actual_report(result)
    return "\n".join(
        [
            str(result.get("final_answer") or ""),
            str(_as_dict(result.get("output")).get("summary") or ""),
            str(report.get("markdown") or ""),
        ]
    )


def _forbidden_hits(case: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    text = _answer_text(result)
    hits: list[str] = []
    for name in _as_list(case.get("forbidden_claims")):
        pattern = FORBIDDEN_PATTERNS.get(str(name), re.compile(re.escape(str(name)), re.I))
        if pattern.search(text):
            hits.append(str(name))
    return hits


def _contract_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    try:
        contract = check_answer_evidence_contract(dict(result))
    except Exception as exc:  # pragma: no cover - defensive for live eval traces
        return {"passed": False, "error": str(exc), "metrics": {}}
    return contract


def evaluate_case(case: Mapping[str, Any], result: Mapping[str, Any]) -> dict[str, Any]:
    actual_companies = _upper_set(_as_list(result.get("companies")))
    expected_companies = _upper_set(_as_list(case.get("companies")))
    actual_tools = {str(item) for item in _as_list(result.get("selected_tools"))}
    expected_tools = {str(item) for item in _as_list(case.get("expected_tools"))}
    actual_dimensions = _actual_dimensions(result)
    expected_dimensions = {str(item) for item in _as_list(case.get("expected_dimensions"))}
    contract_status = str(result.get("final_contract_status") or result.get("contract_status") or "not_checked")
    expected_contract = {str(item) for item in _as_list(case.get("expected_contract_status"))}
    report_present = bool(_actual_report(result))
    forbidden_hits = _forbidden_hits(case, result)
    contract = _contract_metrics(result)
    contract_metrics = _as_dict(contract.get("metrics"))

    company_ok = expected_companies.issubset(actual_companies) if expected_companies else True
    task_type_ok = (
        str(case.get("expected_task_type")) in {"", "unsupported_or_safety"}
        or str(result.get("task_type") or "") == str(case.get("expected_task_type"))
    )
    tool_ok = expected_tools.issubset(actual_tools)
    dimension_recall_ok = expected_dimensions.issubset(actual_dimensions)
    report_ok = report_present == bool(case.get("report_expected"))
    contract_status_ok = contract_status in expected_contract if expected_contract else True
    forbidden_ok = not forbidden_hits
    comparison_ok = True
    if bool(case.get("comparison_balance_required")):
        comparison_ok = bool(contract_metrics.get("comparison_balance_rate", 1.0) in {1, 1.0})

    gates = {
        "company_ok": company_ok,
        "task_type_ok": task_type_ok,
        "tool_ok": tool_ok,
        "dimension_recall_ok": dimension_recall_ok,
        "report_trigger_ok": report_ok,
        "contract_status_ok": contract_status_ok,
        "forbidden_ok": forbidden_ok,
        "comparison_ok": comparison_ok,
    }
    return {
        "case_id": case.get("case_id"),
        "query": case.get("query"),
        "passed": all(gates.values()),
        "gates": gates,
        "expected": {
            "companies": sorted(expected_companies),
            "task_type": case.get("expected_task_type"),
            "dimensions": sorted(expected_dimensions),
            "tools": sorted(expected_tools),
            "report_expected": bool(case.get("report_expected")),
            "contract_status": sorted(expected_contract),
        },
        "actual": {
            "companies": sorted(actual_companies),
            "task_type": result.get("task_type"),
            "dimensions": sorted(actual_dimensions),
            "tools": sorted(actual_tools),
            "report_present": report_present,
            "contract_status": contract_status,
            "trace_id": result.get("trace_id", ""),
        },
        "forbidden_hits": forbidden_hits,
        "answer_contract": contract,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(results)
    failures = [item for item in results if not item.get("passed")]

    def rate(name: str) -> float:
        if not count:
            return 0.0
        return sum(1 for item in results if _as_dict(item.get("gates")).get(name)) / count

    def contract_average(name: str, default: float) -> float:
        values: list[float] = []
        for item in results:
            metrics = _as_dict(_as_dict(item.get("answer_contract")).get("metrics"))
            if name in metrics:
                values.append(float(metrics.get(name) or 0.0))
        if not values:
            return default
        return sum(values) / len(values)

    def contract_sum(name: str) -> int:
        total = 0
        for item in results:
            metrics = _as_dict(_as_dict(item.get("answer_contract")).get("metrics"))
            total += int(metrics.get(name) or 0)
        return total

    forbidden_violations = sum(len(_as_list(item.get("forbidden_hits"))) for item in results)
    contract_forbidden_violations = contract_sum("forbidden_claim_violations")
    forbidden_violations += contract_forbidden_violations
    summary = {
        "case_count": count,
        "passed_count": count - len(failures),
        "failed_count": len(failures),
        "company_resolution_accuracy": rate("company_ok"),
        "task_type_accuracy": rate("task_type_ok"),
        "tool_recall_rate": rate("tool_ok"),
        "dimension_recall": rate("dimension_recall_ok"),
        "report_trigger_accuracy": rate("report_trigger_ok"),
        "contract_status_accuracy": rate("contract_status_ok"),
        "numeric_grounding_rate": round(contract_average("numeric_grounding_rate", 1.0), 4),
        "citation_validity_rate": round(contract_average("citation_validity_rate", 1.0), 4),
        "dimension_status_violations": contract_sum("dimension_status_violations"),
        "forbidden_claim_violations": forbidden_violations,
        "raw_internal_leakage_count": contract_sum("raw_internal_leakage_count"),
        "comparison_balance_rate": rate("comparison_ok"),
    }
    summary["hard_gates"] = {
        "company_resolution_accuracy": summary["company_resolution_accuracy"] >= 0.95,
        "report_trigger_accuracy": summary["report_trigger_accuracy"] >= 0.95,
        "numeric_grounding_rate": summary["numeric_grounding_rate"] >= 0.95,
        "citation_validity_rate": summary["citation_validity_rate"] >= 1.0,
        "dimension_status_violations": summary["dimension_status_violations"] == 0,
        "forbidden_claim_violations": forbidden_violations == 0,
        "raw_internal_leakage_count": summary["raw_internal_leakage_count"] == 0,
        "comparison_balance_rate": summary["comparison_balance_rate"] >= 0.95,
    }
    summary["pass"] = all(summary["hard_gates"].values()) and not failures
    return summary


def write_markdown(report: dict[str, Any], path: Path) -> None:
    summary = _as_dict(report.get("summary"))
    lines = [
        "# Answer Benchmark V2 Report",
        "",
        f"- benchmark: `{report.get('benchmark_path')}`",
        f"- cases: {summary.get('case_count', 0)}",
        f"- passed: {summary.get('passed_count', 0)}",
        f"- failed: {summary.get('failed_count', 0)}",
        f"- pass: {summary.get('pass', False)}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        if key in {"hard_gates"}:
            continue
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Failures", ""])
    failures = [item for item in _as_list(report.get("results")) if not _as_dict(item).get("passed")]
    if not failures:
        lines.append("No failures.")
    for item in failures:
        actual = _as_dict(item.get("actual"))
        expected = _as_dict(item.get("expected"))
        lines.extend(
            [
                f"### {item.get('case_id')}",
                "",
                f"- query: {item.get('query')}",
                f"- trace_id: `{actual.get('trace_id', '')}`",
                f"- expected companies: {expected.get('companies', [])}",
                f"- actual companies: {actual.get('companies', [])}",
                f"- expected dimensions: {expected.get('dimensions', [])}",
                f"- actual dimensions: {actual.get('dimensions', [])}",
                f"- expected tools: {expected.get('tools', [])}",
                f"- actual tools: {actual.get('tools', [])}",
                f"- gates: `{json.dumps(item.get('gates', {}), ensure_ascii=False)}`",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_answer_benchmark(path: Path, *, limit: int | None = None) -> dict[str, Any]:
    cases = load_benchmark(path, limit=limit)
    results = []
    for case in cases:
        result = run_agent(str(case["query"]))
        results.append(evaluate_case(case, result))
    report = {
        "benchmark_path": str(path),
        "summary": summarize(results),
        "results": results,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, default=ROOT / "eval" / "answer_benchmark_v2.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-json", type=Path, default=ROOT / "eval" / "reports" / "answer_benchmark_v2_summary.json")
    parser.add_argument("--out-md", type=Path, default=ROOT / "eval" / "reports" / "answer_benchmark_v2_report.md")
    parser.add_argument(
        "--failures-jsonl",
        type=Path,
        default=ROOT / "eval" / "reports" / "answer_benchmark_v2_failures.jsonl",
    )
    args = parser.parse_args()

    report = run_answer_benchmark(args.benchmark, limit=args.limit)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    write_markdown(report, args.out_md)
    failures = [item for item in report["results"] if not item.get("passed")]
    args.failures_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.failures_jsonl.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in failures),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    return 0 if report["summary"].get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
