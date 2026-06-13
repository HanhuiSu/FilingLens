"""Release gate tests for the Phase 5 layered hard gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_phase5_release_eval import FORMAL_METRIC_NAMES, _evaluate_gate  # noqa: E402


def _metrics(**overrides):
    values = {name: 1.0 for name in FORMAL_METRIC_NAMES}
    values.update(overrides)
    return values


def _formal(**metric_overrides):
    avg = _metrics(**metric_overrides)
    by_task = {
        "fact_qa": {"metrics": _metrics(numeric_fact_accuracy=0.96), "signals": {}},
        "trend_analysis": {"metrics": _metrics(numeric_fact_accuracy=0.86), "signals": {}},
        "company_comparison": {"metrics": _metrics(numeric_fact_accuracy=0.86), "signals": {}},
        "report_summary": {"metrics": _metrics(), "signals": {}},
    }
    if "company_comparison_numeric_fact_accuracy" in metric_overrides:
        by_task["company_comparison"]["metrics"]["numeric_fact_accuracy"] = metric_overrides[
            "company_comparison_numeric_fact_accuracy"
        ]
    if "trend_analysis_numeric_fact_accuracy" in metric_overrides:
        by_task["trend_analysis"]["metrics"]["numeric_fact_accuracy"] = metric_overrides[
            "trend_analysis_numeric_fact_accuracy"
        ]
    return {
        "summary": {
            "total": 25,
            "ok": 25,
            "errors": 0,
            "average_metrics": avg,
            "auxiliary_metrics": {
                "output_protocol_stability_rate": 1.0,
                "empty_answer_rate": 0.0,
                "trace_contract_ok_rate": 1.0,
                "mixed_primary_rate": 0.0,
            },
            "by_expected_task_type": by_task,
        },
        "records": [
            {
                "id": "sm01",
                "requires_citation": True,
                "expected_event_count": 0,
                "metrics": _metrics(citation_required_accuracy=1.0, retrieval_hit_rate=1.0),
            }
        ],
    }


def _specialized():
    return {"output_protocol": {"pass_rate": 1.0}}


def _spot():
    return {
        "pass_rate": 1.0,
        "checklist": [{"id": "fq01", "demo_ready": True}, {"id": "cp01", "demo_ready": True}],
    }


def _smoke():
    return {"all_passed": True}


def test_low_answer_accuracy_fails_correctness_and_overall():
    gate = _evaluate_gate(_formal(answer_accuracy=0.275), _specialized(), _spot(), _smoke())
    assert gate["correctness_pass"] is False
    assert gate["overall_pass"] is False
    assert "answer_accuracy" in gate["failed_checks"]


def test_company_comparison_low_numeric_fact_accuracy_fails_gate():
    gate = _evaluate_gate(
        _formal(company_comparison_numeric_fact_accuracy=0.25),
        _specialized(),
        _spot(),
        _smoke(),
    )
    assert gate["correctness_pass"] is False
    assert gate["overall_pass"] is False
    assert "company_comparison_numeric_fact_accuracy" in gate["failed_checks"]


def test_all_four_gate_layers_pass_for_release():
    gate = _evaluate_gate(_formal(), _specialized(), _spot(), _smoke())
    assert gate["contract_pass"] is True
    assert gate["correctness_pass"] is True
    assert gate["retrieval_pass"] is True
    assert gate["demo_pass"] is True
    assert gate["overall_pass"] is True
