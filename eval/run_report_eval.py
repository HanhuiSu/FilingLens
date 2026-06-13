#!/usr/bin/env python3
# ruff: noqa: E402
"""Offline report-flow evaluation.

This eval exercises the structured report assembler and trigger policy without
requiring live vLLM, Chroma, or DuckDB queries.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agent.reporting import build_company_analysis_report, should_build_company_report


DEFAULT_CASES = [
    {"id": "report_open_001", "query": "分析 NVIDIA", "expected_report": True},
    {"id": "report_open_002", "query": "analyze NVDA fundamentals", "expected_report": True},
    {"id": "report_narrow_001", "query": "NVIDIA 现金流质量怎么样", "expected_report": False},
    {"id": "report_narrow_002", "query": "NVDA valuation boundary", "expected_report": False},
]


def _load_cases(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return list(DEFAULT_CASES)
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = line.strip()
        if not raw:
            continue
        item = json.loads(raw)
        if "query" not in item or "expected_report" not in item:
            raise ValueError(f"{path}:{line_no} requires query and expected_report")
        item.setdefault("id", f"case_{line_no}")
        cases.append(item)
    return cases


def _methodology_intent(query: str) -> str:
    lowered = query.lower()
    if any(term in lowered for term in ("现金流", "cash flow")):
        return "cash_flow_quality_analysis"
    if any(term in lowered for term in ("估值", "valuation")):
        return "valuation_boundary_analysis"
    return "single_company_overview"


def _state(query: str) -> dict[str, Any]:
    dimension_status = {
        "business_model": {"status": "satisfied", "supporting_evidence_ids": ["T1"]},
        "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1"]},
        "profitability_quality": {"status": "partial", "supporting_evidence_ids": ["N2"], "enhanced_missing": ["gross_margin"]},
        "cash_flow_quality": {"status": "missing", "required_missing": ["free_cash_flow"]},
        "balance_sheet_and_capital_intensity": {"status": "satisfied", "supporting_evidence_ids": ["N3"]},
        "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T2"]},
        "valuation_and_risk_boundary": {"status": "partial", "supporting_evidence_ids": ["N4"]},
    }
    return {
        "user_query": query,
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "methodology_intent": _methodology_intent(query),
        "companies": ["NVDA"],
        "final_answer": "NVDA 的基本面分析应基于已验证证据综合判断。",
        "resolved_period_context": {"label": "latest available filings"},
        "dimension_status_by_id": dimension_status,
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": 100.0, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "net_income", "value": 20.0, "unit": "USD"},
                {"evidence_id": "N3", "ticker": "NVDA", "metric": "total_debt", "value": 5.0, "unit": "USD"},
                {"evidence_id": "N4", "ticker": "NVDA", "metric": "market_cap", "value": 200.0, "unit": "USD"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "NVDA", "dimension_id": "business_model"},
                {"evidence_id": "T2", "ticker": "NVDA", "dimension_id": "moat_and_competitive_risk"},
            ],
            "dimension_status_map": dimension_status,
        },
        "synthesis": {
            "short_answer": "NVDA 的基本面需要综合多维度证据判断。",
            "methodology_answer": {
                "analysis_scope": "single_company",
                "dimension_sections": [
                    {"dimension_id": "business_model", "summary": "业务概览由 filing 文本支持。", "evidence_refs": ["T1"]},
                    {"dimension_id": "revenue_quality", "summary": "收入质量有结构化收入证据支持。", "evidence_refs": ["N1"]},
                    {"dimension_id": "profitability_quality", "summary": "盈利质量有部分证据支持。", "evidence_refs": ["N2"]},
                    {"dimension_id": "balance_sheet_and_capital_intensity", "summary": "资产负债表安全性有债务证据支持。", "evidence_refs": ["N3"]},
                    {"dimension_id": "moat_and_competitive_risk", "summary": "风险因素由 filing 文本支持。", "evidence_refs": ["T2"]},
                    {"dimension_id": "valuation_and_risk_boundary", "summary": "估值只能作为边界指标，不提供目标价。", "evidence_refs": ["N4"]},
                ],
                "limitations": ["现金流证据不足。"],
            },
        },
        "output": {"limitations": [], "view": {"kind": "methodology_single_company_brief"}},
    }


def _citation_valid(report: dict[str, Any]) -> bool:
    for section in report.get("sections", []) or []:
        citations = {str(item) for item in section.get("citations", []) or []}
        allowed = {str(item) for item in section.get("key_evidence_ids", []) or []}
        if not citations.issubset(allowed):
            return False
    return True


def _forbidden_count(report: dict[str, Any]) -> int:
    text = str(report.get("markdown") or "").lower()
    patterns = (r"(?<!不)买入", r"(?<!不)卖出", r"target price", r"should buy", r"should sell")
    return sum(1 for pattern in patterns if re.search(pattern, text))


def run_report_eval(cases: list[dict[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for case in cases:
        state = _state(str(case["query"]))
        expected_report = bool(case["expected_report"])
        triggered = should_build_company_report(state)
        report = build_company_analysis_report(state)
        sections = report.get("sections", []) if isinstance(report, dict) else []
        record = {
            "id": case["id"],
            "query": case["query"],
            "expected_report": expected_report,
            "triggered": triggered,
            "report_contract_status": report.get("contract_status", "") if report else "",
            "section_count": len(sections),
            "section_coverage_rate": len(sections) / 10 if expected_report else 1.0,
            "citation_validity": 1.0 if (not report or _citation_valid(report)) else 0.0,
            "limitations_present": 1.0 if (not expected_report or report.get("overall_limitations")) else 0.0,
            "forbidden_advice_violations": _forbidden_count(report) if report else 0,
        }
        record["passed"] = (
            record["triggered"] == expected_report
            and record["section_coverage_rate"] >= 1.0
            and record["citation_validity"] == 1.0
            and record["limitations_present"] == 1.0
            and record["forbidden_advice_violations"] == 0
        )
        records.append(record)
    count = len(records)
    summary = {
        "case_count": count,
        "report_trigger_accuracy": sum(1 for r in records if r["triggered"] == r["expected_report"]) / max(count, 1),
        "section_coverage_rate": sum(float(r["section_coverage_rate"]) for r in records) / max(count, 1),
        "section_citation_validity": sum(float(r["citation_validity"]) for r in records) / max(count, 1),
        "limitations_presence_rate": sum(float(r["limitations_present"]) for r in records) / max(count, 1),
        "forbidden_advice_violations": sum(int(r["forbidden_advice_violations"]) for r in records),
        "pass": all(bool(r["passed"]) for r in records),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    return {"summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline report-flow eval.")
    parser.add_argument("--benchmark", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()
    report = run_report_eval(_load_cases(args.benchmark))
    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    if not report["summary"]["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
