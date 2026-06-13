"""Tests for ResearchPlan validation and deterministic fallback."""

from __future__ import annotations

from src.agent.plan_validator import validate_research_plan


def test_causal_intent_only_planner_output_uses_explicit_fallback_plan():
    result = validate_research_plan(
        {"question_type": "revenue"},
        user_query="为什么 NVIDIA 的营收增长这么多",
        companies=["NVDA"],
        answer_mode="direct_fact",
        safety_intent="normal",
    )

    plan = result.plan.model_dump()
    assert result.valid is True
    assert result.used_fallback is True
    assert result.fallback_reason == "planner_missing_or_intent_only_for_causal_query"
    assert plan["question_type"] == "causal_explanation"
    assert {
        "quantify_growth",
        "identify_growth_drivers",
        "verified_evidence",
        "inferred_drivers",
        "hypotheses_to_verify",
        "counterpoints",
        "evidence_boundary",
        "state_evidence_boundary",
    } <= {part["id"] for part in plan["required_answer_parts"]}
    roles = {req["evidence_role"] for req in plan["evidence_requests"]}
    assert {
        "current_revenue",
        "comparator_revenue",
        "revenue_growth_calculation",
        "revenue_growth_text",
        "driver_text",
    } <= roles
    assert any(req["id"] == "growth_driver_text" and req["type"] == "text" for req in plan["evidence_requests"])
    assert "bounded analytical answer" in plan["fallback_answer_policy"]


def test_non_causal_overview_query_cannot_be_validated_as_causal():
    result = validate_research_plan(
        {
            "question_type": "causal_explanation",
            "user_goal": "Explain why revenue changed.",
            "companies": ["AMZN"],
            "required_answer_parts": [
                {"id": "quantify_growth", "description": "Quantify growth", "required": True},
                {"id": "identify_growth_drivers", "description": "Identify drivers", "required": True},
            ],
            "evidence_requests": [
                {
                    "id": "growth_driver_text",
                    "type": "text",
                    "scope": "core",
                    "company": "AMZN",
                    "sections": ["ITEM_7"],
                    "queries": ["revenue growth drivers"],
                    "answer_part_ids": ["identify_growth_drivers"],
                }
            ],
        },
        user_query="amazon overview",
        companies=["AMZN"],
        answer_mode="analytical",
        safety_intent="normal",
    )

    assert result.plan.question_type == "overview"


def test_validator_normalizes_and_rejects_unsupported_fields():
    raw = {
        "question_type": "causal_explanation",
        "user_goal": "Explain revenue growth drivers.",
        "companies": ["UNKNOWN"],
        "required_answer_parts": [{"id": "identify_growth_drivers", "description": "Identify drivers"}],
        "evidence_requests": [
            {
                "id": "driver",
                "type": "text",
                "scope": "unsupported_scope",
                "company": "UNKNOWN",
                "metrics": ["made_up_metric"],
                "sections": ["ITEM_99"],
                "queries": ["revenue growth drivers"],
                "tool": "made_up_tool",
                "answer_part_ids": ["identify_growth_drivers"],
            }
        ],
    }

    result = validate_research_plan(
        raw,
        user_query="why did NVIDIA revenue grow",
        companies=["NVDA"],
        answer_mode="direct_fact",
        safety_intent="normal",
    )

    assert result.valid is True
    assert result.plan.companies == ["NVDA"]
    assert any(item["reason"] == "scope_normalized_to_core" for item in result.warnings)
    rejected_reasons = {item["reason"] for item in result.rejected_items}
    assert {"unsupported_company", "metric_not_allowed", "section_not_allowed", "tool_not_allowed"} <= rejected_reasons
    assert any("identify_growth_drivers" in req.answer_part_ids and req.type == "text" for req in result.plan.evidence_requests)
    assert "bounded analytical answer" in result.plan.fallback_answer_policy


def test_causal_validator_overrides_non_explicit_fallback_policy():
    result = validate_research_plan(
        {
            "question_type": "causal_explanation",
            "user_goal": "Explain revenue growth.",
            "companies": ["NVDA"],
            "required_answer_parts": [{"id": "identify_growth_drivers", "description": "Identify drivers"}],
            "evidence_requests": [
                {
                    "id": "growth_drivers_text",
                    "type": "text",
                    "scope": "core",
                    "company": "NVDA",
                    "sections": ["ITEM_7"],
                    "queries": ["revenue growth drivers"],
                    "answer_part_ids": ["identify_growth_drivers"],
                }
            ],
            "fallback_answer_policy": "Use available evidence.",
        },
        user_query="why did NVIDIA revenue grow",
        companies=["NVDA"],
        answer_mode="direct_fact",
        safety_intent="normal",
    )

    request_ids = [req.id for req in result.plan.evidence_requests]
    assert request_ids.count("growth_drivers_text") == 1
    assert "growth_driver_text" not in request_ids
    assert len([req for req in result.plan.evidence_requests if req.evidence_role == "driver_text"]) == 1
    assert "definitive cause" in result.plan.fallback_answer_policy
    assert "bounded analytical answer" in result.plan.fallback_answer_policy
