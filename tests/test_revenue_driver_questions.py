"""Regression tests for revenue-growth driver questions."""

from __future__ import annotations

from src.agent.answer_relevance import bounded_causal_fallback_answer, judge_answer_relevance
from src.agent.evidence_planner import build_requirements_from_research_plan
from src.agent.evidence_sufficiency import evaluate_evidence_sufficiency
from src.agent.plan_validator import validate_research_plan


def test_nvda_revenue_why_numeric_only_is_not_fully_answered():
    validation = validate_research_plan(
        {},
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
        answer_mode="direct_fact",
        safety_intent="normal",
    )
    plan = validation.plan.model_dump(exclude_none=True)
    evidence_plan = build_requirements_from_research_plan(
        {
            "user_query": "为什么 NVIDIA 的营收增长这么多",
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "companies": ["NVDA"],
        },
        plan,
    ).model_dump(exclude_none=True)

    numeric_req = next(req for req in evidence_plan["evidence_requirements"] if req.get("evidence_role") == "current_revenue")
    sufficiency = evaluate_evidence_sufficiency(
        evidence_plan,
        [{"requirement_id": numeric_req["requirement_id"], "status": "satisfied", "evidence_type": "numeric", "items": [{"evidence_id": "N1"}]}],
    ).model_dump(exclude_none=True)

    assert validation.valid is True
    assert validation.used_fallback is True
    assert sufficiency["overall_status"] != "sufficient"
    assert sufficiency["answer_part_status_by_id"]["quantify_growth"]["status"] == "partial"
    assert "quantify_growth" in sufficiency["partial_required_answer_parts"]
    assert "identify_growth_drivers" in sufficiency["missing_but_analyzable_answer_parts"]
    assert sufficiency["evidence_gap_by_answer_part"]["identify_growth_drivers"]["reason"] == "driver_text_evidence_missing_but_analyzable"

    state = {
        "user_query": "为什么 NVIDIA 的营收增长这么多",
        "companies": ["NVDA"],
        "research_plan_used": plan,
        "required_answer_parts": plan["required_answer_parts"],
        "missing_required_answer_parts": sufficiency["missing_required_answer_parts"],
        "missing_but_analyzable_answer_parts": sufficiency["missing_but_analyzable_answer_parts"],
        "numeric_evidence": [{"evidence_id": "N1", "metric": "revenue"}],
    }
    assert judge_answer_relevance("NVIDIA revenue grew [N1].", state).route == "repair_answer"
    bounded = bounded_causal_fallback_answer(state)
    assert "待验证假设" in bounded
    assert "证据边界" in bounded
