"""Unit tests for labeled retrieval gold metrics."""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_retrieval_eval import evaluate_case, load_gold, summarize_records


def _case() -> dict:
    return {
        "id": "rag_test",
        "category": "risk",
        "query": "Find AAPL risk factors about competition.",
        "ticker": "AAPL",
        "expected_form_type": "10-K",
        "expected_section": "ITEM_1A",
        "expected_fiscal_period": "SEPTEMBER 27, 2025",
        "must_include_terms": ["competition", "risk"],
    }


def _hit(**overrides) -> dict:
    row = {
        "ticker": "AAPL",
        "form_type": "10-K",
        "section": "ITEM_1A",
        "fiscal_period": "SEPTEMBER 27, 2025",
        "quality": "high",
        "supporting_snippet": "Competition and market demand are important risk factors.",
        "score_breakdown": {"final_score": 0.9},
    }
    row.update(overrides)
    return row


def test_evaluate_case_scores_exact_evidence_hit():
    record = evaluate_case(_case(), [_hit()])

    assert record["metrics"]["recall_at_3"] == 1.0
    assert record["metrics"]["recall_at_5"] == 1.0
    assert record["metrics"]["section_accuracy"] == 1.0
    assert record["metrics"]["period_match_rate"] == 1.0
    assert record["metrics"]["citation_precision"] == 1.0
    assert record["metrics"]["mixed_section_primary_rate"] == 0.0
    assert record["metrics"]["snippet_support_rate"] == 1.0
    assert record["failure_reasons"] == []


def test_evaluate_case_attributes_wrong_section_and_unsupported_snippet():
    record = evaluate_case(
        _case(),
        [
            _hit(section="ITEM_7", supporting_snippet="Revenue increased due to services."),
            _hit(section="MIXED", quality="low", supporting_snippet="Generic filing text."),
        ],
    )

    assert record["metrics"]["recall_at_5"] == 0.0
    assert record["metrics"]["section_accuracy"] == 0.0
    assert record["metrics"]["mixed_section_primary_rate"] == 1.0
    assert record["metrics"]["snippet_support_rate"] == 0.0
    assert "wrong_primary_section" in record["failure_reasons"]
    assert "snippet_unsupported" in record["failure_reasons"]


def test_summary_gate_uses_hard_thresholds():
    passing = evaluate_case(_case(), [_hit()])
    failing = evaluate_case(_case(), [_hit(section="ITEM_7")])
    summary = summarize_records([passing, failing])

    assert summary["case_count"] == 2
    assert summary["recall_at_5"] == 0.5
    assert summary["pass"] is False


def test_retrieval_gold_schema_has_minimum_labeled_cases():
    records = load_gold(Path("eval/retrieval_gold.jsonl"))

    assert len(records) >= 50
    categories = {row["category"] for row in records}
    assert {"risk", "legal_risk", "competition", "business_summary", "revenue_trend"} <= categories
    for row in records:
        assert row["id"]
        assert row["query"]
        assert row["ticker"]
        assert row["expected_form_type"] in {"10-K", "10-Q"}
        assert row["expected_section"].startswith("ITEM_")
        assert row["expected_fiscal_period"]
        assert isinstance(row["must_include_terms"], list)
        assert row["must_include_terms"]
        json.dumps(row)
