from __future__ import annotations

from typing import Any

from src.agent.requirement_executor import execute_evidence_requirements


class FakeTool:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        return self.response


def test_numeric_returned_but_metric_mismatch_records_validation_reason(monkeypatch):
    monkeypatch.setattr(
        "src.agent.requirement_executor.query_financial_data",
        FakeTool(
            {
                "financial_facts": [
                    {
                        "ticker": "AMZN",
                        "metric": "revenue",
                        "period_type": "annual",
                        "period_end": "2025-12-31",
                        "value": 100.0,
                        "unit": "USD",
                    }
                ],
                "price_data": [],
            }
        ),
    )
    req = {
        "requirement_id": "REQ-AMZN-NET-INCOME",
        "requirement_type": "numeric",
        "company": "AMZN",
        "metrics": ["net_income"],
        "required": True,
        "requirement_scope": "core",
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "analytical"},
        {"answer_mode": "analytical", "evidence_requirements": [req]},
        {"period_query": {"period_type": "latest"}, "resolved_period_context": {}, "retrieval_policy": {}, "tickers": ["AMZN"]},
    )

    result = out["evidence_collection_results"][0]
    assert result["status"] == "missing"
    assert result["failure_reason"] == "metric_mapping_failed"
    assert result["tool_returned_count"] == 1
    assert result["validated_evidence_count"] == 0
    assert out["evidence_validation_records"][0]["rejected_evidence_reason"] == "metric_mapping_failed"
