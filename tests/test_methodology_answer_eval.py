"""Tests for answer-level methodology eval integration."""

from __future__ import annotations

import json
from pathlib import Path

from eval import run_methodology_eval as runner


def _benchmark(path: Path) -> Path:
    case = {
        "id": "answer-eval-fake",
        "category": "valuation_boundary",
        "query": "NVIDIA 估值贵不贵",
        "expected_framework": "fundamental_quality_analysis",
        "expected_active_dimensions": ["valuation_and_risk_boundary"],
        "expected_sufficiency_statuses": {"valuation_and_risk_boundary": "satisfied"},
        "expected_red_flag_ids": [],
        "expected_missing_disclosures": [],
        "must_not_include_terms": ["买入", "卖出"],
        "expected_companies": ["NVDA"],
        "expected_analysis_scope": "single_company",
        "expected_methodology_intent": "valuation",
        "expected_safety_intent": "normal",
    }
    path.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _actual_state() -> dict:
    return {
        "final_answer": "NVDA 市值为 $4.31T，P/E 为 100.36x。这只能作为估值边界观察，不构成投资建议。[N1][N2]",
        "selected_framework": "fundamental_quality_analysis",
        "active_dimensions": ["valuation_and_risk_boundary"],
        "analysis_scope": "single_company",
        "query_understanding_summary": {
            "companies": [{"ticker": "NVDA"}],
            "analysis_scope": "single_company",
            "methodology_intent": "valuation",
            "safety_intent": "normal",
        },
        "evidence_packet": {
            "numeric_table": [
                {"evidence_id": "N1", "ticker": "NVDA", "metric": "market_cap", "value": 4_311_463_935_165, "unit": "USD"},
                {"evidence_id": "N2", "ticker": "NVDA", "metric": "pe_ratio", "value": 100.36, "unit": "ratio"},
            ],
            "dimension_status_map": {
                "valuation_and_risk_boundary": {
                    "status": "satisfied",
                    "supporting_evidence_ids": ["N1", "N2"],
                }
            },
        },
    }


def test_answer_mode_runs_contract_checker_and_reports_metrics(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner, "_run_agent_case", lambda case: _actual_state())

    report = runner.run_methodology_eval(_benchmark(tmp_path / "answer.jsonl"), mode="answer")
    actual = report["records"][0]["actual"]

    assert report["mode"] == "answer"
    assert report["summary"]["report_only"] is False
    assert report["summary"]["pass"] is True
    assert report["summary"]["numeric_grounding_rate"] == 1.0
    assert report["summary"]["citation_validity_rate"] == 1.0
    assert actual["answer_contract"]["passed"] is True
    assert actual["answer_preview"] == _actual_state()["final_answer"][:240]


def test_answer_mode_gate_fails_on_contract_violation(tmp_path: Path, monkeypatch):
    bad = _actual_state()
    bad["final_answer"] = "NVDA 市值为 $9.99T，应该买。[N9]"
    original_answer = bad["final_answer"]
    monkeypatch.setattr(runner, "_run_agent_case", lambda case: bad)

    report = runner.run_methodology_eval(_benchmark(tmp_path / "answer.jsonl"), mode="answer")
    actual = report["records"][0]["actual"]

    assert report["summary"]["pass"] is False
    assert report["summary"]["gate_failures"]
    assert report["summary"]["numeric_grounding_rate"] < 1.0
    assert actual["answer_contract"]["passed"] is False
    assert actual["answer_preview"] == original_answer[:240]
    assert bad["final_answer"] == original_answer
    assert "answer_contract" not in bad


def test_planning_mode_stays_report_only(tmp_path: Path):
    report = runner.run_methodology_eval(_benchmark(tmp_path / "answer.jsonl"), mode="planning")

    assert report["summary"]["report_only"] is True
    assert report["summary"]["pass"] is None
