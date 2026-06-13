"""Final-answer regressions for composite queries with partial evidence."""

from __future__ import annotations

import json
from pathlib import Path

from src.agent import answering


def _partial_nvda_state() -> dict:
    trace = json.loads(Path("data/traces/bbe284c8-ba64-4dab-ae15-b9c795c4f621.json").read_text())
    state = dict(trace)
    state["evidence_collection_results"] = list(trace["collection_evidence_collection_results"])
    state["evidence_sufficiency"] = dict(trace["collection_evidence_sufficiency"])
    state["evidence_sufficiency_summary"] = dict(trace["collection_evidence_sufficiency_summary"])
    state["requirement_status_map"] = dict(trace["collection_requirement_status_map"])
    state["final_answer"] = ""
    state["draft_answer"] = ""
    state["output"] = {}
    return state


def test_generate_answer_preserves_numeric_evidence_when_risk_text_missing(monkeypatch):
    def no_analyst_draft(**_kwargs):
        return {
            "draft": {},
            "accepted_draft": {},
            "validation": {"passed": False, "status": "skipped_for_test"},
            "attempts": [],
            "violations": [],
            "draft_final_status": "skipped_for_test",
            "generation_issues": [],
        }

    monkeypatch.setattr(answering, "run_analyst_draft_loop", no_analyst_draft)

    out = answering.generate_agent_answer(_partial_nvda_state())

    metrics = {str(item.get("metric")) for item in out["numeric_evidence"]}
    dimensions = dict(out["dimension_status_map"])
    final_satisfied_numeric = [
        item
        for item in out["evidence_collection_results"]
        if item.get("status") == "satisfied" and item.get("evidence_type") in {"numeric", "calculation"}
    ]

    assert out["numeric_evidence"]
    assert out["validated_numeric_evidence_count"] > 0
    assert len(final_satisfied_numeric) >= 10
    assert {"operating_cash_flow", "free_cash_flow"} <= metrics
    assert {"market_cap", "pe_ratio", "ps_ratio", "fcf_yield"} & metrics
    assert dimensions["cash_flow_quality"]["status"] in {"satisfied", "partial"}
    assert dimensions["valuation_and_risk_boundary"]["status"] in {"satisfied", "partial"}
    assert dimensions["moat_and_competitive_risk"]["status"] == "missing"
    assert "现金流质量" in out["final_answer"]
    assert "估值边界" in out["final_answer"]
    assert "当前缺少风险文本证据" in out["final_answer"]
    assert "REQ-" not in out["final_answer"]
