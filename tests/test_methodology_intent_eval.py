"""Planning-only methodology intent regression tests."""

from __future__ import annotations

from pathlib import Path

from eval import run_methodology_intent_eval as runner
from eval.run_methodology_intent_eval import (
    INTENT_FAMILIES,
    evaluate_case,
    infer_view_kind,
    load_benchmark,
    run_methodology_intent_eval,
)
from src.agent.query_plan import build_classification_state


ROOT = Path(__file__).resolve().parents[1]


def _classification(query: str) -> dict:
    from datetime import date

    return build_classification_state(
        user_query=query,
        parsed={"task_type": "fact_qa", "companies": [], "data_route": "hybrid"},
        trace_id="test-methodology-intent",
        today=date(2026, 4, 24),
    )


def test_methodology_intent_benchmark_schema_and_coverage():
    cases = load_benchmark(ROOT / "eval" / "methodology_intent_benchmark.jsonl")
    categories = {case["category"] for case in cases}

    assert len(cases) >= 60
    assert INTENT_FAMILIES.issubset(categories)
    assert all(isinstance(case["expected_required_evidence"], list) for case in cases)
    assert all("expected_view_kind" in case for case in cases)


def test_methodology_intent_eval_metrics_detect_failures():
    case = {
        "id": "fake",
        "category": "valuation_boundary_analysis",
        "query": "苹果现在贵不贵",
        "expected_task_type": "report_summary",
        "expected_answer_mode": "analytical",
        "expected_analysis_scope": "single_company",
        "expected_framework_id": "fundamental_quality_analysis",
        "required_dimensions": ["valuation_and_risk_boundary"],
        "optional_dimensions": [],
        "expected_safety_intent": "normal",
        "expected_view_kind": "valuation_boundary_brief",
        "expected_required_evidence": [{"dimension_id": "valuation_and_risk_boundary", "metric": "price"}],
    }
    actual = {
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "analysis_scope": "single_company",
        "safety_intent": "normal",
        "selected_analysis_framework": {
            "framework_id": "fundamental_quality_analysis",
            "active_dimension_ids": ["revenue_quality"],
        },
        "evidence_plan": {"evidence_requirements": []},
    }

    record = evaluate_case(case, actual)

    assert record["metrics"]["dimension_recall"] == 0.0
    assert record["metrics"]["evidence_requirement_recall"] == 0.0
    assert "dimension_recall" in record["failure_reasons"]
    assert "evidence_requirement_recall" in record["failure_reasons"]


def test_planning_mode_does_not_require_live_agent_dependencies():
    report = run_methodology_intent_eval(
        ROOT / "eval" / "methodology_intent_benchmark.jsonl",
        mode="planning",
        limit=6,
    )

    assert report["summary"]["case_count"] == 6
    assert report["summary"]["pass"] is True
    assert all(record["actual"]["evidence_requirements_count"] > 0 for record in report["records"])


def test_methodology_intent_cli_returns_nonzero_on_gate_failure(monkeypatch):
    def _failing_report(*args, **kwargs):
        return {"summary": {"pass": False}, "records": []}

    monkeypatch.setattr(runner, "run_methodology_intent_eval", _failing_report)
    monkeypatch.setattr(runner.sys, "argv", ["run_methodology_intent_eval.py"])

    assert runner.main() == 1


def test_full_methodology_intent_planning_eval_passes_gates():
    report = run_methodology_intent_eval(ROOT / "eval" / "methodology_intent_benchmark.jsonl", mode="planning")

    assert report["summary"]["pass"] is True
    assert report["summary"]["metrics"]["task_type_accuracy"] >= 0.9
    assert report["summary"]["metrics"]["answer_mode_accuracy"] >= 0.9
    assert report["summary"]["metrics"]["framework_selection_accuracy"] >= 0.9
    assert report["summary"]["metrics"]["dimension_recall"] >= 0.85
    assert report["summary"]["metrics"]["safety_intent_accuracy"] >= 0.95
    assert report["summary"]["metrics"]["unsafe_advice_rate"] == 0.0


def test_methodology_intent_family_routing_examples():
    overview = _classification("分析下 nvidia")
    risk = _classification("nvidia 最大的问题是什么")
    comparison = _classification("apple 和 amazon 更推荐哪个")
    valuation = _classification("苹果现在贵不贵")
    cash_flow = _classification("苹果利润能不能变成现金")
    prediction = _classification("预测明天苹果股价")

    assert overview["methodology_intent"] == "single_company_overview"
    assert overview["answer_mode"] == "analytical"
    assert overview["analysis_scope"] == "single_company"
    assert overview["time_policy"] == "latest_available"

    assert risk["methodology_intent"] == "risk_focused_analysis"
    assert risk["answer_mode"] == "risk_focused_analysis"
    assert infer_view_kind(risk) == "risk_focused_analysis_brief"

    assert comparison["methodology_intent"] == "investment_advice_like"
    assert comparison["task_type"] == "company_comparison"
    assert comparison["safety_intent"] == "investment_advice_like"
    assert infer_view_kind(comparison) == "methodology_comparison_brief"

    assert valuation["methodology_intent"] == "valuation_boundary_analysis"
    assert valuation["selected_analysis_framework"]["active_dimension_ids"] == ["valuation_and_risk_boundary"]
    assert infer_view_kind(valuation) == "valuation_boundary_brief"

    assert cash_flow["methodology_intent"] == "cash_flow_quality_analysis"
    assert cash_flow["selected_analysis_framework"]["active_dimension_ids"] == ["cash_flow_quality"]

    assert prediction["methodology_intent"] == "unsupported_prediction"
    assert prediction["answer_mode"] == "refusal_or_redirect"
    assert prediction["safety_intent"] == "unsupported_or_out_of_scope"


def test_comparison_subintent_reasons_are_structured():
    risk = _classification("AAPL 和 AMZN 哪家公司风险更高？")
    difference = _classification("AAPL 和 AMZN 差异是什么？")

    assert risk["query_understanding_summary"]["methodology_intent"] == "comparison"
    assert risk["methodology_intent"] == "company_comparison"
    assert "comparison_risk_family" in risk["query_understanding_summary"]["intent_reasons"]

    assert difference["query_understanding_summary"]["methodology_intent"] == "comparison"
    assert difference["methodology_intent"] == "company_comparison"
    assert "comparison_difference_family" in difference["query_understanding_summary"]["intent_reasons"]
