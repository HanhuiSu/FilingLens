from __future__ import annotations

from src.agent.evidence_planner import merge_evidence_requirements
from src.agent.types import CoverageDecision


def test_requirement_merge_retains_legacy_core_and_unions_answer_parts():
    legacy = {
        "user_query": "AMZN overview",
        "task_type": "report_summary",
        "answer_mode": "analytical",
        "evidence_requirements": [
            {
                "requirement_id": "REQ-LEGACY-REV",
                "requirement_type": "numeric",
                "company": "AMZN",
                "metric": "revenue",
                "metrics": ["revenue"],
                "dimension_id": "revenue_quality",
                "required": True,
                "requirement_scope": "core",
                "fallback_strategy": ["latest_period"],
            },
            {
                "requirement_id": "REQ-LEGACY-RISK",
                "requirement_type": "text",
                "company": "AMZN",
                "dimension_id": "moat_and_competitive_risk",
                "section_preferences": ["ITEM_1A"],
                "retrieval_query": "risk factors competition",
                "required": True,
                "requirement_scope": "core",
            },
        ],
        "core_requirement_ids": ["REQ-LEGACY-REV", "REQ-LEGACY-RISK"],
    }
    research = {
        "research_plan": {"question_type": "overview", "companies": ["AMZN"]},
        "required_answer_parts": [{"id": "verified_evidence", "required": True}],
        "evidence_requirements": [
            {
                "requirement_id": "REQ-RP-REV",
                "requirement_type": "numeric",
                "company": "AMZN",
                "metric": "revenue",
                "metrics": ["revenue"],
                "dimension_id": "revenue_quality",
                "required": True,
                "requirement_scope": "core",
                "answer_part_ids": ["verified_evidence"],
                "fallback_strategy": ["relax_period"],
            },
            {
                "requirement_id": "REQ-RP-BUSINESS",
                "requirement_type": "text",
                "company": "AMZN",
                "dimension_id": "business_model",
                "section_preferences": ["ITEM_1"],
                "retrieval_query": "business segments AWS North America International",
                "required": True,
                "requirement_scope": "core",
            },
        ],
        "core_requirement_ids": ["REQ-RP-REV", "REQ-RP-BUSINESS"],
    }
    decision = CoverageDecision(
        strategy="merge",
        legacy_core_count=2,
        research_core_count=2,
        retained_legacy_core_count=2,
        coverage_ratio=1.0,
    )

    merged = merge_evidence_requirements(
        legacy_evidence_plan=legacy,
        research_evidence_plan=research,
        coverage_decision=decision,
    ).model_dump(exclude_none=True)

    reqs = merged["evidence_requirements"]
    by_id = {req["requirement_id"]: req for req in reqs}
    assert "REQ-LEGACY-REV" in by_id
    assert "REQ-LEGACY-RISK" in by_id
    assert any(req["dimension_id"] == "business_model" for req in reqs)
    assert by_id["REQ-LEGACY-REV"]["merged_from"] == ["legacy", "research_plan"]
    assert by_id["REQ-LEGACY-REV"]["answer_part_ids"] == ["verified_evidence"]
    assert set(by_id["REQ-LEGACY-REV"]["fallback_strategy"]) == {"latest_period", "relax_period"}
    assert merged["plan_source"] == "merged"
    assert merged["requirement_merge_summary"]["deduped_requirements"] == 1
    assert merged["requirement_merge_summary"]["legacy_research_count"] == 1
