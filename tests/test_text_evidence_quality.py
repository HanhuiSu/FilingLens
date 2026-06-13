from __future__ import annotations

from typing import Any

from src.agent.requirement_executor import execute_evidence_requirements, validate_text_evidence_quality


class FakeSearchTool:
    def __init__(self, response: Any):
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def invoke(self, payload: dict[str, Any]) -> Any:
        self.calls.append(payload)
        return self.response


def test_section_header_only_text_is_low_information():
    quality = validate_text_evidence_quality({"supporting_snippet": "Item 1."})

    assert quality.is_valid is False
    assert quality.reason == "low_information_text_evidence"
    assert quality.is_section_header_only is True


def test_item_header_cannot_satisfy_text_requirement(monkeypatch):
    monkeypatch.setattr(
        "src.agent.requirement_executor.search_filings",
        FakeSearchTool(
            {
                "items": [
                    {
                        "ticker": "AMZN",
                        "section": "ITEM_1",
                        "supporting_snippet": "Item 1.",
                        "text_snippet": "Item 1.",
                    }
                ],
                "diagnostics": {"raw_hit_count": 1, "section_filtered_hit_count": 1},
            }
        ),
    )
    monkeypatch.setattr(
        "src.agent.requirement_executor.search_filings_lexical_fallback",
        lambda **_kwargs: {"items": [], "diagnostics": {"raw_hit_count": 0, "section_filtered_hit_count": 0}},
    )
    req = {
        "requirement_id": "REQ-AMZN-BUSINESS-TEXT",
        "requirement_type": "text",
        "company": "AMZN",
        "section_preferences": ["ITEM_1"],
        "primary_sections": ["ITEM_1"],
        "retrieval_query": "business overview segments AWS North America International",
        "required": True,
        "requirement_scope": "core",
        "min_results": 1,
    }

    out = execute_evidence_requirements(
        {"needs_tools": True, "answer_mode": "analytical"},
        {"answer_mode": "analytical", "evidence_requirements": [req]},
        {"period_query": {"period_type": "latest"}, "resolved_period_context": {}, "retrieval_policy": {"text_top_k": 1}, "tickers": ["AMZN"]},
    )

    result = out["evidence_collection_results"][0]
    assert result["status"] == "missing"
    assert result["failure_reason"] == "low_information_text_evidence"
    assert result["drop_stage"] == "low_information_text_evidence"
