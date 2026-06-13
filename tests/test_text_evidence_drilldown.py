"""Tests for the text evidence drilldown script."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from scripts import drilldown_text_evidence as drilldown


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.reason_phrase = "OK" if status_code < 400 else "ERROR"
        self.text = str(payload)

    def json(self) -> dict[str, Any]:
        return self._payload


def test_drilldown_script_generates_report(monkeypatch, tmp_path: Path):
    query = "apple和amazon你更看好哪个"
    trace = {
        "trace_id": "trace-drilldown",
        "task_type": "company_comparison",
        "answer_mode": "comparison_brief",
        "safety_intent": "investment_advice_like",
        "synthesis_mode": "limited_judgment",
        "evidence_plan": {
            "evidence_requirements": [
                {
                    "requirement_id": "REQ-TEXT-AAPL-COMPARISON_CONTEXT",
                    "requirement_type": "text",
                    "company": "AAPL",
                    "retrieval_query": "AAPL business context",
                }
            ]
        },
        "text_requirement_diagnostics": {
            "REQ-TEXT-AAPL-COMPARISON_CONTEXT": {
                "requirement_id": "REQ-TEXT-AAPL-COMPARISON_CONTEXT",
                "company": "AAPL",
                "retrieval_query": "AAPL business context",
                "raw_hit_count": 2,
                "section_filtered_hit_count": 2,
                "usable_hit_count": 1,
                "snippet_support_passed_count": 1,
                "text_claim_validated_count": 1,
                "text_citation_kept_count": 0,
                "final_validated_text_count": 0,
                "drop_stage": "citation_policy_dropped",
                "raw_text_claim_count": 1,
                "candidate_text_claim_count": 1,
                "validated_text_claim_count": 1,
                "rejected_text_claims": [
                    {
                        "claim_raw_text": "AAPL has risks.",
                        "source_requirement_id": "REQ-TEXT-AAPL-COMPARISON_CONTEXT",
                        "source_snippet_id": "T1",
                        "citation_ref": "T1",
                        "validation_error_code": "claim_too_generic",
                    }
                ],
                "text_claim_validation_warnings": [],
                "claim_generation_error": "",
                "top_raw_snippets": [{"ticker": "AAPL", "section": "ITEM_7", "text_snippet": "raw"}],
                "top_rejected_snippets": [{"reason": "quality_filter_dropped", "text_snippet": "reject"}],
                "rejection_reasons": {"quality_filter_dropped": 1},
            }
        },
        "evidence_packet": {"text_snippets": []},
    }
    chat_body = {"trace_id": "trace-drilldown", "output": {"text_evidence": []}}
    low_level_probe = {
        "duckdb_section_probe": {"status": "ok", "rows": [{"ticker": "AAPL", "raw_hit_count": 3}]},
        "chroma_query_probe": {"status": "ok", "rows": [{"ticker": "AAPL", "query": "risk factors", "raw_hit_count": 5}]},
    }

    def fake_request(method: str, path: str, **_kwargs: Any) -> FakeResponse:
        if method == "POST" and path == "/chat":
            return FakeResponse(chat_body)
        if method == "GET" and path == "/trace/trace-drilldown":
            return FakeResponse(trace)
        return FakeResponse({"detail": "not found"}, status_code=404)

    monkeypatch.setattr(drilldown, "_request", fake_request)
    monkeypatch.setattr(drilldown, "run_low_level_rag_probe", lambda _query="": low_level_probe)
    output = tmp_path / "report.md"
    monkeypatch.setattr(
        sys,
        "argv",
        ["drilldown_text_evidence.py", "--query", query, "--output", str(output)],
    )

    assert drilldown.main() == 0
    report = output.read_text(encoding="utf-8")

    assert "# Text Evidence Drilldown Report" in report
    assert query in report
    assert "trace-drilldown" in report
    assert "Evidence Requirements" in report
    assert "Per-Requirement Retrieval Diagnostics" in report
    assert "Drop Stage Table" in report
    assert "requirement_id" in report
    assert "final_validated_text_count" in report
    assert "Top Raw Snippets" in report
    assert "Top Rejected Snippets" in report
    assert "Text Claim Generation / Validation" in report
    assert "claim_too_generic" in report
    assert "Final Validated Text Evidence" in report
    assert "Failure Stage Summary" in report
    assert "Root Cause Guess" in report
    assert "citation policy" in report.lower()
