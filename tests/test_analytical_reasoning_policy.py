from __future__ import annotations

from src.agent.analytical_reasoning import analytical_reasoning_payload
from src.agent.plan_validator import deterministic_causal_research_plan, validate_research_plan


def test_causal_validator_injects_reasoning_policy_and_minimum_answer_policy():
    validation = validate_research_plan(
        {},
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
        answer_mode="analytical",
        safety_intent="normal",
    )
    plan = validation.plan.model_dump(exclude_none=True)
    part_ids = {part["id"] for part in plan["required_answer_parts"]}

    assert plan["reasoning_policy"]["allow_inference"] is True
    assert plan["reasoning_policy"]["allow_hypotheses"] is True
    assert plan["reasoning_policy"]["must_separate_claim_tiers"] is True
    assert "verified_evidence" in part_ids
    assert "inferred_drivers" in part_ids
    assert "hypotheses_to_verify" in part_ids
    assert "minimum_answer_policy" in plan


def test_analytical_reasoning_payload_counts_claim_tiers_for_causal_gap():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    payload = analytical_reasoning_payload(
        research_plan=plan,
        numeric_cards=[{"evidence_id": "N1", "ticker": "NVDA", "metric": "revenue", "value": "68.1B"}],
        text_cards=[],
        requirement_summary={"missing_but_analyzable_answer_parts": ["identify_growth_drivers"]},
        evidence_sufficiency={"evidence_health": "degraded"},
        lang="zh",
    )

    assert payload["analytical_reasoning_status"] == "used"
    assert payload["evidence_health"] == "degraded"
    assert payload["claim_tiers"]["evidence_backed"] == 1
    assert payload["claim_tiers"]["evidence_inferred"] >= 1
    assert payload["claim_tiers"]["hypothesis_to_verify"] >= 3


def test_segment_driver_claim_is_not_upgraded_to_company_driver():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    payload = analytical_reasoning_payload(
        research_plan=plan,
        numeric_cards=[],
        text_cards=[
            {
                "evidence_id": "T1",
                "citation_ref": "T1",
                "driver_levels": ["segment_level_driver", "product_level_driver"],
                "supporting_snippet": "Data Center networking revenue grew 142% driven by NVLink and InfiniBand platforms.",
            }
        ],
        requirement_summary={"partial_required_answer_parts": ["identify_growth_drivers"]},
        evidence_sufficiency={"evidence_health": "partial"},
        lang="zh",
    )

    backed = [claim["text"] for claim in payload["analytical_claims"] if claim["tier"] == "evidence_backed"]
    inferred = [claim["text"] for claim in payload["analytical_claims"] if claim["tier"] == "evidence_inferred"]
    assert any(("分部层面" in text or "产品层面" in text or "分部/产品层面" in text) for text in backed)
    assert any("不能完整代表 NVDA 总营收增长原因" in text for text in inferred)


def test_tool_degraded_hypothesis_claim_keeps_explicit_marker():
    plan = deterministic_causal_research_plan(
        user_query="为什么 NVIDIA 营收增长这么多",
        companies=["NVDA"],
    ).model_dump(exclude_none=True)
    payload = analytical_reasoning_payload(
        research_plan=plan,
        numeric_cards=[],
        text_cards=[],
        requirement_summary={},
        evidence_sufficiency={
            "evidence_health": "degraded",
            "tool_error_context": [{"requirement_id": "REQ-TEXT", "kind": "tool_execution_error"}],
        },
        lang="zh",
    )

    tool_claims = [claim for claim in payload["analytical_claims"] if claim["id"] == "hv_tool"]
    assert tool_claims
    assert tool_claims[0]["tier"] == "hypothesis_to_verify"
    assert tool_claims[0]["text"].startswith("待验证假设：")
