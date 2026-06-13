from __future__ import annotations

from src.agent.evidence_planner import evaluate_plan_coverage


def _legacy_plan(core_count: int = 18) -> dict:
    return {
        "evidence_requirements": [
            {
                "requirement_id": f"REQ-LEGACY-{idx}",
                "requirement_type": "numeric",
                "company": "AMZN",
                "metric": "revenue",
                "metrics": ["revenue"],
                "dimension_id": "revenue_quality",
                "required": True,
                "requirement_scope": "core",
            }
            for idx in range(core_count)
        ],
        "core_requirement_ids": [f"REQ-LEGACY-{idx}" for idx in range(core_count)],
    }


def _research_plan(question_type: str = "overview", core_count: int = 3) -> tuple[dict, dict]:
    plan = {"question_type": question_type, "companies": ["AMZN"], "required_answer_parts": []}
    evidence_plan = {
        "research_plan": plan,
        "evidence_requirements": [
            {
                "requirement_id": f"REQ-RP-{idx}",
                "requirement_type": "numeric",
                "company": "AMZN",
                "metric": "revenue",
                "metrics": ["revenue"],
                "required": True,
                "requirement_scope": "core",
            }
            for idx in range(core_count)
        ],
        "core_requirement_ids": [f"REQ-RP-{idx}" for idx in range(core_count)],
    }
    return plan, evidence_plan


def test_overview_undercovered_research_plan_forces_merge():
    plan, research_evidence = _research_plan("overview", core_count=3)

    decision = evaluate_plan_coverage(
        research_plan=plan,
        research_evidence_plan=research_evidence,
        legacy_evidence_plan=_legacy_plan(18),
        state={"required_dimensions": ["business_model", "revenue_quality"]},
        planner_valid=True,
        mode="expanded",
    )

    assert decision.strategy == "merge"
    assert decision.legacy_core_count == 18
    assert decision.research_core_count == 3
    assert decision.retained_legacy_core_count == 18
    assert decision.coverage_ratio < 0.8
    assert "research_plan_under_covered_legacy_core" in decision.warnings


def test_causal_research_plan_keeps_replace_strategy():
    plan, research_evidence = _research_plan("causal_explanation", core_count=5)

    decision = evaluate_plan_coverage(
        research_plan=plan,
        research_evidence_plan=research_evidence,
        legacy_evidence_plan=_legacy_plan(18),
        state={},
        planner_valid=True,
        mode="expanded",
    )

    assert decision.strategy == "replace"
    assert decision.retained_legacy_core_count == 0


def test_composite_request_merges_legacy_coverage():
    plan, research_evidence = _research_plan("unknown", core_count=4)

    decision = evaluate_plan_coverage(
        research_plan=plan,
        research_evidence_plan=research_evidence,
        legacy_evidence_plan=_legacy_plan(10),
        state={"required_dimensions": ["cash_flow_quality", "valuation_and_risk_boundary", "moat_and_competitive_risk"]},
        planner_valid=True,
        mode="expanded",
    )

    assert decision.strategy == "merge"
    assert decision.retained_legacy_core_count == 10
