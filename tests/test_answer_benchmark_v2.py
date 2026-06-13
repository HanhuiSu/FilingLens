"""Tests for the v2 labeled answer benchmark runner."""

from __future__ import annotations

from pathlib import Path

from eval.run_answer_benchmark import evaluate_case, load_benchmark, summarize

ROOT = Path(__file__).resolve().parent.parent


def test_answer_benchmark_v2_schema_and_count():
    cases = load_benchmark(ROOT / "eval" / "answer_benchmark_v2.jsonl")

    assert len(cases) == 36
    assert sum(1 for case in cases if case["report_expected"]) >= 6
    assert any(case["comparison_balance_required"] for case in cases)
    assert any(case["intent_family"] == "advice_or_forecast_boundary" for case in cases)


def test_evaluate_case_uses_labels_and_report_trigger():
    case = {
        "case_id": "fake",
        "query": "分析 NVIDIA",
        "companies": ["NVDA"],
        "expected_task_type": "report_summary",
        "expected_dimensions": ["revenue_quality"],
        "expected_tools": ["query_financial_data"],
        "report_expected": True,
        "forbidden_claims": ["buy"],
        "comparison_balance_required": False,
        "expected_contract_status": ["passed"],
    }
    result = {
        "trace_id": "t1",
        "final_answer": "Revenue is supported [N1].",
        "companies": ["NVDA"],
        "task_type": "report_summary",
        "selected_tools": ["query_financial_data"],
        "dimension_status_by_id": {"revenue_quality": {"status": "satisfied"}},
        "contract_status": "passed",
        "output": {"report": {"markdown": "# Report\nRevenue [N1]."}},
        "evidence_packet": {"numeric_table": [{"evidence_id": "N1", "value": 1}]},
    }

    evaluated = evaluate_case(case, result)

    assert evaluated["passed"] is True
    assert evaluated["gates"]["report_trigger_ok"] is True


def test_summarize_flags_failures():
    summary = summarize(
        [
            {"passed": True, "gates": {"company_ok": True, "report_trigger_ok": True, "comparison_ok": True}, "forbidden_hits": []},
            {"passed": False, "gates": {"company_ok": False, "report_trigger_ok": True, "comparison_ok": True}, "forbidden_hits": ["buy"]},
        ]
    )

    assert summary["case_count"] == 2
    assert summary["failed_count"] == 1
    assert summary["forbidden_claim_violations"] == 1
    assert summary["pass"] is False
