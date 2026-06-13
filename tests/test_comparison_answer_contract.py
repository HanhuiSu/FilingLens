"""Comparison-specific answer contract checks."""

from __future__ import annotations

from copy import deepcopy

from src.agent.answer_contract import check_answer_evidence_contract


def _comparison_trace() -> dict:
    return {
        "final_answer": (
            "比较判断：如果只看盈利质量，AAPL 更强；如果看收入规模，AMZN 更大。"
            "风险维度需要同时看两家公司披露。这不是投资建议。[N1][N2][N3][N4][T1][T2]"
        ),
        "task_type": "company_comparison",
        "analysis_scope": "comparison",
        "query_understanding_summary": {
            "companies": [{"ticker": "AAPL"}, {"ticker": "AMZN"}],
            "safety_intent": "investment_advice_like",
        },
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "AAPL", "metric": "revenue", "value": 120_000_000_000, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "AMZN", "metric": "revenue", "value": 150_000_000_000, "unit": "USD"},
                {"evidence_id": "N3", "ticker": "AAPL", "metric": "net_margin", "value": 0.26, "unit": "ratio"},
                {"evidence_id": "N4", "ticker": "AMZN", "metric": "net_margin", "value": 0.14, "unit": "ratio"},
            ],
            "text_snippets": [
                {"evidence_id": "T1", "ticker": "AAPL", "dimension_id": "moat_and_competitive_risk"},
                {"evidence_id": "T2", "ticker": "AMZN", "dimension_id": "moat_and_competitive_risk"},
            ],
            "dimension_status_map": {
                "revenue_quality": {"status": "satisfied", "supporting_evidence_ids": ["N1", "N2"]},
                "profitability_quality": {"status": "satisfied", "supporting_evidence_ids": ["N3", "N4"]},
                "moat_and_competitive_risk": {"status": "satisfied", "supporting_evidence_ids": ["T1", "T2"]},
            },
        },
    }


def test_non_advisory_balanced_comparison_passes():
    result = check_answer_evidence_contract(_comparison_trace())

    assert result["passed"] is True
    assert result["metrics"]["comparison_balance_rate"] == 1.0
    assert result["metrics"]["forbidden_claim_violations"] == 0


def test_investment_action_wording_fails():
    trace = _comparison_trace()
    trace["final_answer"] = "AAPL 比 AMZN 更值得买。[N1][N2]"

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "forbidden_claim" for item in result["violations"])


def test_single_sided_profitability_evidence_fails_comparison_balance():
    trace = deepcopy(_comparison_trace())
    trace["evidence_packet"]["numeric_table"] = [
        row for row in trace["evidence_packet"]["numeric_table"] if row["evidence_id"] != "N4"
    ]

    result = check_answer_evidence_contract(trace)

    assert result["passed"] is False
    assert any(item["type"] == "comparison_balance" for item in result["violations"])
