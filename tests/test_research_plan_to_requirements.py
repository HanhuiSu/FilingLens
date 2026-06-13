"""Tests for converting ResearchPlan records into EvidencePlan requirements."""

from __future__ import annotations

from src.agent.evidence_planner import build_requirements_from_research_plan
from src.agent.plan_validator import deterministic_causal_research_plan


def test_causal_research_plan_builds_numeric_and_driver_text_requirements():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)

    evidence_plan = build_requirements_from_research_plan(
        {
            "user_query": "为什么 NVIDIA 的营收增长这么多",
            "task_type": "fact_qa",
            "answer_mode": "direct_fact",
            "safety_intent": "normal",
            "companies": ["NVDA"],
            "period_query": {"period_type": "latest"},
        },
        plan,
    ).model_dump(exclude_none=True)

    requirements = evidence_plan["evidence_requirements"]
    roles = {req.get("evidence_role"): req for req in requirements}
    numeric = roles["current_revenue"]
    driver = next(req for req in requirements if req.get("evidence_request_id") == "growth_driver_text")

    assert evidence_plan["expected_synthesis_style"] == "causal_explanation"
    assert evidence_plan["plan_source"] == "deterministic_causal_fallback"
    assert evidence_plan["research_plan"]["question_type"] == "causal_explanation"
    assert {"current_revenue", "comparator_revenue", "revenue_growth_calculation", "revenue_growth_text", "driver_text"} <= set(roles)
    assert numeric["metrics"] == ["revenue"]
    assert numeric["answer_part_ids"] == ["quantify_growth"]
    assert roles["revenue_growth_calculation"]["requirement_type"] == "calculation"
    assert roles["revenue_growth_calculation"]["required"] is True
    assert roles["revenue_growth_calculation"]["requirement_scope"] == "core"
    assert roles["revenue_growth_text"]["requirement_type"] == "text"
    assert roles["revenue_growth_text"]["answer_part_ids"] == ["quantify_growth"]
    assert driver["requirement_type"] == "text"
    assert driver["evidence_role"] == "driver_text"
    assert driver["required"] is True
    assert driver["requirement_scope"] == "core"
    assert driver["answer_part_ids"] == ["identify_growth_drivers"]
    assert {"ITEM_7", "ITEM_2", "MD&A"} <= set(driver["section_preferences"])


def test_planner_metric_aliases_are_normalized_before_requirement_validation():
    plan = {
        "question_type": "overview",
        "user_goal": "Analyze Amazon overview.",
        "companies": ["AMZN"],
        "required_answer_parts": [{"id": "verified_evidence", "description": "Show metrics.", "required": True}],
        "evidence_requests": [
            {
                "id": "overview_numeric",
                "type": "numeric",
                "scope": "core",
                "company": "AMZN",
                "metrics": ["latest_revenue", "profit", "price", "capex"],
                "answer_part_ids": ["verified_evidence"],
            }
        ],
    }

    evidence_plan = build_requirements_from_research_plan(
        {
            "user_query": "AMZN overview",
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "companies": ["AMZN"],
            "period_query": {"period_type": "latest"},
        },
        plan,
    ).model_dump(exclude_none=True)

    req = evidence_plan["evidence_requirements"][0]
    assert req["metrics"] == ["revenue", "net_income", "adjusted_close", "capital_expenditure"]
    assert evidence_plan.get("rejected_requirements", []) == []


def test_unsupported_planner_metric_is_rejected_explicitly():
    plan = {
        "question_type": "overview",
        "user_goal": "Analyze Amazon overview.",
        "companies": ["AMZN"],
        "required_answer_parts": [{"id": "verified_evidence", "description": "Show metrics.", "required": True}],
        "evidence_requests": [
            {
                "id": "unsupported_numeric",
                "type": "numeric",
                "scope": "core",
                "company": "AMZN",
                "metrics": ["magic_metric"],
                "answer_part_ids": ["verified_evidence"],
            }
        ],
    }

    evidence_plan = build_requirements_from_research_plan(
        {
            "user_query": "AMZN overview",
            "task_type": "report_summary",
            "answer_mode": "analytical",
            "companies": ["AMZN"],
        },
        plan,
    ).model_dump(exclude_none=True)

    assert evidence_plan["evidence_requirements"] == []
    assert evidence_plan["rejected_requirements"][0]["reason"] == "unsupported_planner_metric"
    assert evidence_plan["rejected_requirements"][0]["value"] == "magic_metric"
